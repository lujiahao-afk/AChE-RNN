import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED, AllChem, DataStructs
from rdkit import RDLogger
import warnings

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

# ==================== 1. 加载先验模型和数据 ====================
print("=" * 60)
print("步骤1: 加载先验模型和参考数据")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载检查点
checkpoint = torch.load('best_rnn_model.pth', map_location=device, weights_only=False)
vocab = checkpoint['vocab']
char_to_idx = checkpoint['char_to_idx']
idx_to_char = checkpoint['idx_to_char']
vocab_size = len(vocab)
hp = checkpoint['hyperparams']
MAX_LEN = hp['max_len']

# 模型定义
class SmilesRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super(SmilesRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        embed = self.dropout(self.embedding(x))
        output, hidden = self.lstm(embed, hidden)
        logits = self.fc(output)
        return logits, hidden

# 创建先验模型(冻结, 用于计算KL散度约束)
prior_model = SmilesRNN(vocab_size, hp['embed_dim'], hp['hidden_dim'], hp['num_layers']).to(device)
prior_model.load_state_dict(checkpoint['model_state_dict'])
prior_model.eval()
for param in prior_model.parameters():
    param.requires_grad = False

# 创建Agent模型(可训练, 会被强化学习更新)
agent_model = SmilesRNN(vocab_size, hp['embed_dim'], hp['hidden_dim'], hp['num_layers']).to(device)
agent_model.load_state_dict(checkpoint['model_state_dict'])

print(f"先验模型已冻结")
print(f"Agent模型已加载, 参数量: {sum(p.numel() for p in agent_model.parameters()):,}")

# 加载训练数据计算参考指纹
train_df = pd.read_csv('ache_inhibitors_cleaned.csv')
train_smiles = train_df['canonical_smiles'].dropna().tolist()

# 计算训练集分子指纹
train_fps = []
train_smiles_canonical = set()
for smi in train_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        canonical = Chem.MolToSmiles(mol)
        train_smiles_canonical.add(canonical)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        train_fps.append(fp)

print(f"训练集分子数: {len(train_fps)}")
print(f"训练集指纹已计算")

# ==================== 2. 定义打分函数 ====================
print("\n" + "=" * 60)
print("步骤2: 定义多维度打分函数")
print("=" * 60)

def score_molecules(smiles_list):
    """
    综合打分函数:
    1. 有效性: SMILES是否能被RDKit解析 (0或1)
    2. 类药性: QED分数 (0~1)
    3. 相似性: 与训练集中最相似分子的Tanimoto相似度 (0~1)
    4. 新颖性: 是否在训练集中 (奖励新分子)
    5. Lipinski: 是否满足五规则 (0或1)
    """
    scores = []
    details = []

    for smi in smiles_list:
        score = 0.0
        detail = {'valid': False, 'qed': 0, 'sim': 0, 'novel': False, 'lipinski': False}

        # 解析分子
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scores.append(0.0)
            details.append(detail)
            continue

        detail['valid'] = True
        canonical = Chem.MolToSmiles(mol)

        # QED类药性分数 (权重0.3)
        qed_score = QED.qed(mol)
        detail['qed'] = round(qed_score, 3)
        score += 0.3 * qed_score

        # 与训练集的相似性 (权重0.3) - 保留与已知活性分子的相似度
        gen_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        max_sim = 0
        for train_fp in train_fps:
            sim = DataStructs.TanimotoSimilarity(gen_fp, train_fp)
            if sim > max_sim:
                max_sim = sim
                if max_sim > 0.9:
                    break
        detail['sim'] = round(max_sim, 3)
        score += 0.3 * max_sim

        # 新颖性奖励 (权重0.2) - 不在训练集中得满分
        is_novel = canonical not in train_smiles_canonical
        detail['novel'] = is_novel
        score += 0.2 * (1.0 if is_novel else 0.0)

        # Lipinski合规 (权重0.2)
        lipinski_ok = (
            Descriptors.MolWt(mol) <= 500 and
            Descriptors.MolLogP(mol) <= 5 and
            Lipinski.NumHAcceptors(mol) <= 10 and
            Lipinski.NumHDonors(mol) <= 5
        )
        detail['lipinski'] = lipinski_ok
        score += 0.2 * (1.0 if lipinski_ok else 0.0)

        scores.append(score)
        details.append(detail)

    return scores, details

# 测试打分函数
test_smi = 'O=C(NC1CCN(Cc2ccccc2)CC1)c1ccccc1'
test_score, test_detail = score_molecules([test_smi])
print(f"打分函数测试: {test_smi}")
print(f"  分数: {test_score[0]:.3f}, 详情: {test_detail[0]}")

# ==================== 3. 生成函数(带log概率) ====================
def generate_with_logprobs(model, batch_size=64, temperature=1.0, max_len=100):
    """
    生成分子并记录每个字符选择的log概率
    用于计算策略梯度
    """
    model.eval()

    # 初始化
    input_ids = torch.full((batch_size, 1), char_to_idx['<START>'], dtype=torch.long, device=device)
    hidden = None
    generated_seqs = [[char_to_idx['<START>']] for _ in range(batch_size)]
    log_probs_list = [[] for _ in range(batch_size)]
    done = [False] * batch_size

    with torch.no_grad():
        for step in range(max_len):
            logits, hidden = model(input_ids, hidden)
            last_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(last_logits, dim=-1)

            # 采样
            next_tokens = torch.multinomial(probs, 1).squeeze(-1)

            # 计算log概率
            log_probs = torch.log(probs.gather(1, next_tokens.unsqueeze(-1)).squeeze(-1) + 1e-10)

            for i in range(batch_size):
                if not done[i]:
                    token = next_tokens[i].item()
                    generated_seqs[i].append(token)
                    log_probs_list[i].append(log_probs[i].item())

                    if token == char_to_idx['<END>']:
                        done[i] = True

            input_ids = next_tokens.unsqueeze(-1)

            if all(done):
                break

    # 解码
    smiles_list = []
    for seq in generated_seqs:
        smi = decode_sequence(seq)
        smiles_list.append(smi)

    return smiles_list, log_probs_list

def decode_sequence(indices):
    chars = []
    for idx in indices:
        c = idx_to_char.get(idx, '')
        if c == '<END>':
            break
        if c not in ['<PAD>', '<START>']:
            chars.append(c)
    return ''.join(chars)

# ==================== 4. 强化学习训练循环 ====================
print("\n" + "=" * 60)
print("步骤3: 开始强化学习训练")
print("=" * 60)

# 超参数
RL_EPOCHS = 50           # RL训练轮数
BATCH_SIZE = 64          # 每轮生成的分子数
KL_WEIGHT = 0.5          # KL散度约束权重(防止偏离先验太远)
LEARNING_RATE = 1e-4     # RL学习率
TEMPERATURE = 1.0        # 采样温度

optimizer = torch.optim.Adam(agent_model.parameters(), lr=LEARNING_RATE)

# 记录训练过程
history = {
    'epoch': [], 'mean_score': [], 'valid_rate': [],
    'novel_rate': [], 'lipinski_rate': [], 'mean_qed': []
}

for epoch in range(RL_EPOCHS):
    agent_model.eval()

    # ---- 生成分子 ----
    smiles_list, log_probs_list = generate_with_logprobs(
        agent_model, batch_size=BATCH_SIZE, temperature=TEMPERATURE
    )

    # ---- 打分 ----
    scores, details = score_molecules(smiles_list)
    scores_tensor = torch.tensor(scores, dtype=torch.float32, device=device)

    # ---- 计算损失(策略梯度 + KL约束) ----
    agent_model.train()
    optimizer.zero_grad()

    # 重新前向传播计算log概率(保留梯度)
    total_loss = torch.tensor(0.0, device=device)
    valid_count = 0

    for i in range(BATCH_SIZE):
        if not details[i]['valid']:
            continue
        valid_count += 1

        seq = [char_to_idx['<START>']]
        for c in smiles_list[i]:
            if c in char_to_idx:
                seq.append(char_to_idx[c])
        seq.append(char_to_idx['<END>'])

        if len(seq) < 3:
            continue

        input_seq = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
        target_seq = torch.tensor([seq[1:]], dtype=torch.long, device=device)

        # Agent的log概率
        agent_logits, _ = agent_model(input_seq)
        agent_log_probs = torch.log_softmax(agent_logits, dim=-1)
        agent_token_log_probs = agent_log_probs.gather(2, target_seq.unsqueeze(-1)).squeeze(-1)
        agent_seq_log_prob = agent_token_log_probs.sum()

        # 先验的log概率(KL约束用)
        with torch.no_grad():
            prior_logits, _ = prior_model(input_seq)
            prior_log_probs = torch.log_softmax(prior_logits, dim=-1)
            prior_token_log_probs = prior_log_probs.gather(2, target_seq.unsqueeze(-1)).squeeze(-1)
            prior_seq_log_prob = prior_token_log_probs.sum()

        # 策略梯度损失: -score * log_prob (最大化分数)
        advantage = scores_tensor[i] - scores_tensor.mean()
        pg_loss = -advantage * agent_seq_log_prob

        # KL散度约束: 防止Agent偏离先验太远
        kl_loss = KL_WEIGHT * (agent_seq_log_prob - prior_seq_log_prob)

        total_loss = total_loss + pg_loss + kl_loss

    if valid_count > 0:
        total_loss = total_loss / valid_count
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent_model.parameters(), max_norm=1.0)
        optimizer.step()

    # ---- 统计 ----
    valid_rate = sum(1 for d in details if d['valid']) / BATCH_SIZE * 100
    novel_rate = sum(1 for d in details if d['novel']) / max(valid_count, 1) * 100
    lipinski_rate = sum(1 for d in details if d['lipinski']) / max(valid_count, 1) * 100
    mean_qed = np.mean([d['qed'] for d in details if d['valid']]) if valid_count > 0 else 0
    mean_score = np.mean(scores)

    history['epoch'].append(epoch + 1)
    history['mean_score'].append(round(mean_score, 4))
    history['valid_rate'].append(round(valid_rate, 1))
    history['novel_rate'].append(round(novel_rate, 1))
    history['lipinski_rate'].append(round(lipinski_rate, 1))
    history['mean_qed'].append(round(mean_qed, 4))

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{RL_EPOCHS}] "
              f"Score: {mean_score:.3f} | Valid: {valid_rate:.0f}% | "
              f"Novel: {novel_rate:.0f}% | Lipinski: {lipinski_rate:.0f}% | "
              f"QED: {mean_qed:.3f}")

# 保存RL微调后的模型
torch.save({
    'model_state_dict': agent_model.state_dict(),
    'vocab': vocab,
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
    'hyperparams': hp,
    'rl_history': history
}, 'rl_tuned_model.pth')
print(f"\nRL微调模型已保存到: rl_tuned_model.pth")

# ==================== 5. 最终生成与评估 ====================
print("\n" + "=" * 60)
print("步骤4: 用RL微调模型生成最终分子")
print("=" * 60)

all_molecules = []
for temp in [0.5, 0.7, 0.9]:
    for _ in range(5):
        smiles, _ = generate_with_logprobs(agent_model, batch_size=100, temperature=temp)
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                canonical = Chem.MolToSmiles(mol)
                props = {
                    'SMILES': canonical,
                    'MW': round(Descriptors.MolWt(mol), 1),
                    'LogP': round(Descriptors.MolLogP(mol), 2),
                    'HBA': Lipinski.NumHAcceptors(mol),
                    'HBD': Lipinski.NumHDonors(mol),
                    'TPSA': round(Descriptors.TPSA(mol), 1),
                    'QED': round(QED.qed(mol), 3),
                    'Lipinski': (Descriptors.MolWt(mol) <= 500 and
                                 Descriptors.MolLogP(mol) <= 5 and
                                 Lipinski.NumHAcceptors(mol) <= 10 and
                                 Lipinski.NumHDonors(mol) <= 5),
                    'IsNovel': canonical not in train_smiles_canonical,
                    'Temperature': temp
                }
                all_molecules.append(props)

results_df = pd.DataFrame(all_molecules)
results_df = results_df.drop_duplicates(subset='SMILES')

# 筛选类药且新颖的分子
druglike_novel = results_df[(results_df['Lipinski']) & (results_df['IsNovel'])]
druglike_novel = druglike_novel.sort_values('QED', ascending=False)

print(f"\n总去重分子: {len(results_df)}")
print(f"类药且新颖: {len(druglike_novel)}")
print(f"Lipinski合规率: {results_df['Lipinski'].mean()*100:.1f}%")
print(f"新颖率: {results_df['IsNovel'].mean()*100:.1f}%")
print(f"平均QED: {results_df['QED'].mean():.3f}")

print(f"\nTop 25 RL优化后的类药分子:")
print("-" * 105)
print(f"{'#':>3} {'SMILES':<58} {'MW':>7} {'LogP':>6} {'QED':>6} {'Lip':>4} {'Nov':>4}")
print("-" * 105)
for i, (_, row) in enumerate(druglike_novel.head(25).iterrows()):
    smi = row['SMILES'][:56]
    print(f"{i+1:>3} {smi:<58} {row['MW']:>7} {row['LogP']:>6} {row['QED']:>6} "
          f"{'Y' if row['Lipinski'] else 'N':>4} {'Y' if row['IsNovel'] else 'N':>4}")

# 保存
results_df.to_csv('rl_generated_all.csv', index=False)
druglike_novel.to_csv('rl_generated_druglike.csv', index=False)

# ==================== 6. 训练过程对比 ====================
print("\n" + "=" * 60)
print("步骤5: RL训练过程趋势")
print("=" * 60)
print(f"{'Epoch':>6} {'Score':>8} {'Valid%':>8} {'Novel%':>8} {'Lipin%':>8} {'QED':>8}")
print("-" * 50)
for i in range(0, len(history['epoch']), 5):
    print(f"{history['epoch'][i]:>6} {history['mean_score'][i]:>8} "
          f"{history['valid_rate'][i]:>7}% {history['novel_rate'][i]:>7}% "
          f"{history['lipinski_rate'][i]:>7}% {history['mean_qed'][i]:>8}")

print(f"\n文件已保存:")
print(f"  rl_generated_all.csv      - 全部生成分子")
print(f"  rl_generated_druglike.csv  - 类药且新颖分子")
print(f"  rl_tuned_model.pth         - RL微调模型")
