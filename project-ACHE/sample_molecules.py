import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED, AllChem, DataStructs, Draw
from rdkit import RDLogger
import os
from datetime import datetime

RDLogger.DisableLog('rdApp.*')

# ==================== 1. 加载模型 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 优先加载RL微调模型, 如果没有则加载基础模型
if os.path.exists('rl_tuned_model.pth'):
    checkpoint = torch.load('rl_tuned_model.pth', map_location=device, weights_only=False)
    print("已加载: RL微调模型")
elif os.path.exists('best_rnn_model.pth'):
    checkpoint = torch.load('best_rnn_model.pth', map_location=device, weights_only=False)
    print("已加载: 基础RNN模型")
else:
    raise FileNotFoundError("未找到模型文件!")

vocab = checkpoint['vocab']
char_to_idx = checkpoint['char_to_idx']
idx_to_char = checkpoint['idx_to_char']
vocab_size = len(vocab)
hp = checkpoint['hyperparams']

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

model = SmilesRNN(vocab_size, hp['embed_dim'], hp['hidden_dim'], hp['num_layers']).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"设备: {device}")

# 加载训练集(用于计算新颖性)
train_df = pd.read_csv('ache_inhibitors_cleaned.csv')
train_smiles_set = set()
train_fps = []
for smi in train_df['canonical_smiles'].dropna():
    mol = Chem.MolFromSmiles(smi)
    if mol:
        canonical = Chem.MolToSmiles(mol)
        train_smiles_set.add(canonical)
        train_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

# ==================== 2. 工具函数 ====================
def decode_sequence(indices):
    chars = []
    for idx in indices:
        c = idx_to_char.get(idx, '')
        if c == '<END>':
            break
        if c not in ['<PAD>', '<START>']:
            chars.append(c)
    return ''.join(chars)

def generate_smiles(model, temperature=0.8, max_len=120):
    """单个分子生成"""
    input_idx = torch.tensor([[char_to_idx['<START>']]], dtype=torch.long).to(device)
    generated = [char_to_idx['<START>']]
    hidden = None

    with torch.no_grad():
        for _ in range(max_len):
            logits, hidden = model(input_idx, hidden)
            last_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(last_logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()

            if next_idx == char_to_idx['<END>']:
                break
            generated.append(next_idx)
            input_idx = torch.tensor([[next_idx]], dtype=torch.long).to(device)

    return decode_sequence(generated)

def batch_generate(model, n=100, temperature=0.8, max_len=120):
    """批量生成"""
    smiles_list = []
    for _ in range(n):
        smi = generate_smiles(model, temperature, max_len)
        smiles_list.append(smi)
    return smiles_list

def evaluate_molecule(smi):
    """评估单个分子的完整属性"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    canonical = Chem.MolToSmiles(mol)

    # 基本属性
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hba = Lipinski.NumHAcceptors(mol)
    hbd = Lipinski.NumHDonors(mol)
    tpsa = Descriptors.TPSA(mol)
    rotbonds = Lipinski.NumRotatableBonds(mol)
    qed_score = QED.qed(mol)
    rings = Lipinski.RingCount(mol)

    # Lipinski五规则
    lipinski = (mw <= 500 and logp <= 5 and hba <= 10 and hbd <= 5)

    # Veber规则 (口服生物利用度)
    veber = (rotbonds <= 10 and tpsa <= 140)

    # 新颖性
    is_novel = canonical not in train_smiles_set

    # 与训练集的最大相似度
    gen_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    max_sim = 0
    for train_fp in train_fps:
        sim = DataStructs.TanimotoSimilarity(gen_fp, train_fp)
        if sim > max_sim:
            max_sim = sim
            if max_sim > 0.95:
                break

    # 原子数
    num_atoms = mol.GetNumAtoms()

    return {
        'SMILES': canonical,
        'MW': round(mw, 1),
        'LogP': round(logp, 2),
        'HBA': hba,
        'HBD': hbd,
        'TPSA': round(tpsa, 1),
        'RotBonds': rotbonds,
        'Rings': rings,
        'Atoms': num_atoms,
        'QED': round(qed_score, 3),
        'Lipinski': lipinski,
        'Veber': veber,
        'IsNovel': is_novel,
        'MaxSimilarity': round(max_sim, 3)
    }

# ==================== 3. 大规模采样 ====================
print("\n" + "=" * 70)
print("开始大规模采样")
print("=" * 70)

# 采样参数 - 可自行修改
SAMPLE_CONFIG = {
    'temperatures': [0.3, 0.5, 0.7, 0.9, 1.0, 1.2],  # 多温度
    'samples_per_temp': 500,                            # 每个温度生成数
    'max_len': 120,                                     # 最大序列长度
}

all_results = []
total_attempts = 0
total_valid = 0

for temp in SAMPLE_CONFIG['temperatures']:
    n = SAMPLE_CONFIG['samples_per_temp']
    print(f"\n温度={temp}: 生成{n}个分子...", end=" ", flush=True)

    temp_valid = 0
    temp_results = []

    for i in range(n):
        smi = generate_smiles(model, temperature=temp, max_len=SAMPLE_CONFIG['max_len'])
        total_attempts += 1

        props = evaluate_molecule(smi)
        if props is not None:
            props['Temperature'] = temp
            props['BatchID'] = f"T{temp}_{i+1}"
            temp_results.append(props)
            temp_valid += 1
            total_valid += 1

    all_results.extend(temp_results)
    print(f"有效: {temp_valid}/{n} ({temp_valid/n*100:.1f}%)")

print(f"\n{'=' * 70}")
print(f"采样完成!")
print(f"总尝试: {total_attempts}")
print(f"总有效: {total_valid} ({total_valid/total_attempts*100:.1f}%)")

# ==================== 4. 结果分析 ====================
results_df = pd.DataFrame(all_results)
results_df = results_df.drop_duplicates(subset='SMILES')

print(f"\n去重后分子数: {len(results_df)}")

# 各类筛选
druglike = results_df[results_df['Lipinski'] == True]
novel = results_df[results_df['IsNovel'] == True]
druglike_novel = results_df[(results_df['Lipinski'] == True) & (results_df['IsNovel'] == True)]
high_qed = results_df[results_df['QED'] >= 0.7]
veber_pass = results_df[results_df['Veber'] == True]

print(f"\n--- 分子筛选统计 ---")
print(f"  Lipinski合规:    {len(druglike):>5} ({len(druglike)/len(results_df)*100:.1f}%)")
print(f"  新颖分子:        {len(novel):>5} ({len(novel)/len(results_df)*100:.1f}%)")
print(f"  类药且新颖:      {len(druglike_novel):>5} ({len(druglike_novel)/len(results_df)*100:.1f}%)")
print(f"  高QED (>=0.7):   {len(high_qed):>5} ({len(high_qed)/len(results_df)*100:.1f}%)")
print(f"  Veber合规:       {len(veber_pass):>5} ({len(veber_pass)/len(results_df)*100:.1f}%)")

# 按温度统计
print(f"\n--- 各温度有效率 ---")
for temp in SAMPLE_CONFIG['temperatures']:
    temp_df = results_df[results_df['Temperature'] == temp]
    if len(temp_df) > 0:
        print(f"  T={temp}: {len(temp_df)}个 | 平均QED={temp_df['QED'].mean():.3f} | "
              f"Lipinski={temp_df['Lipinski'].mean()*100:.0f}% | 新颖={temp_df['IsNovel'].mean()*100:.0f}%")

# ==================== 5. 按QED排序输出Top分子 ====================
top_molecules = druglike_novel.sort_values('QED', ascending=False)

print(f"\n{'=' * 90}")
print(f"Top 30 类药且新颖的分子 (按QED排序)")
print(f"{'=' * 90}")
print(f"{'#':>3} {'SMILES':<52} {'MW':>7} {'LogP':>6} {'QED':>6} {'Sim':>5} {'T':>4}")
print("-" * 90)

top_list = []
for i, (_, row) in enumerate(top_molecules.head(30).iterrows()):
    smi_display = row['SMILES'][:50]
    print(f"{i+1:>3} {smi_display:<52} {row['MW']:>7} {row['LogP']:>6} {row['QED']:>6} "
          f"{row['MaxSimilarity']:>5} {row['Temperature']:>4}")
    top_list.append(row['SMILES'])

# ==================== 6. 分子多样性分析 ====================
print(f"\n--- 分子属性统计 ---")
print(f"  MW:    {results_df['MW'].mean():.1f} ± {results_df['MW'].std():.1f}  "
      f"(范围: {results_df['MW'].min():.0f} ~ {results_df['MW'].max():.0f})")
print(f"  LogP:  {results_df['LogP'].mean():.2f} ± {results_df['LogP'].std():.2f}  "
      f"(范围: {results_df['LogP'].min():.1f} ~ {results_df['LogP'].max():.1f})")
print(f"  QED:   {results_df['QED'].mean():.3f} ± {results_df['QED'].std():.3f}  "
      f"(范围: {results_df['QED'].min():.3f} ~ {results_df['QED'].max():.3f})")
print(f"  HBA:   {results_df['HBA'].mean():.1f} ± {results_df['HBA'].std():.1f}")
print(f"  HBD:   {results_df['HBD'].mean():.1f} ± {results_df['HBD'].std():.1f}")
print(f"  Rings: {results_df['Rings'].mean():.1f} ± {results_df['Rings'].std():.1f}")

# ==================== 7. 保存结果 ====================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 保存全部结果
all_filename = f'samples_all_{timestamp}.csv'
results_df.to_csv(all_filename, index=False)

# 保存类药分子
druglike_filename = f'samples_druglike_{timestamp}.csv'
druglike_novel.to_csv(druglike_filename, index=False)

# 保存Top分子
top_filename = f'samples_top30_{timestamp}.csv'
top_molecules.head(30).to_csv(top_filename, index=False)

print(f"\n--- 文件已保存 ---")
print(f"  {all_filename}  ({len(results_df)}个分子)")
print(f"  {druglike_filename}  ({len(druglike_novel)}个类药新颖分子)")
print(f"  {top_filename}  (Top30分子)")

# ==================== 8. 继续采样选项 ====================
print(f"\n{'=' * 70}")
print("如需继续采样, 修改SAMPLE_CONFIG参数后重新运行")
print("或直接调用以下函数:")
print(f"  generate_smiles(model, temperature=0.7)  # 单个生成")
print(f"  batch_generate(model, n=100, temperature=0.5)  # 批量生成")
print(f"{'=' * 70}")
