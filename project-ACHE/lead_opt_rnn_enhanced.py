import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED, AllChem, DataStructs, BRICS
from rdkit import RDLogger
import random

RDLogger.DisableLog('rdApp.*')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 1. 加载模型 ====================
checkpoint = torch.load('best_rnn_model.pth', map_location=device, weights_only=False)
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

# 加载训练集
train_df = pd.read_csv('ache_inhibitors_cleaned.csv')
train_smiles_set = set()
train_fps = []
for smi in train_df['canonical_smiles'].dropna():
    mol = Chem.MolFromSmiles(smi)
    if mol:
        train_smiles_set.add(Chem.MolToSmiles(mol))
        train_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

def decode_seq(indices):
    chars = []
    for idx in indices:
        c = idx_to_char.get(idx, '')
        if c == '<END>':
            break
        if c not in ['<PAD>', '<START>']:
            chars.append(c)
    return ''.join(chars)

# ==================== 2. 选择先导 ====================
LEADS = {
    'Donepezil': 'O=C1C=CC(=O)c2cc(OCCN3CCCCC3)c(OCc3ccccc3)cc21',
    'Galantamine': 'COc1ccc2c3c1O[C@H]1C=C[C@@H](O)C[C@@]13CCN(C)C2',
    'Tacrine': 'Nc1c2c(nc3ccccc13)CCCC2',
    'Gen_HighQED': 'O=C(CCN1CCCC1)Nc1ccc2nc3n(c(=O)c2c1)CCC3',
}

SELECTED = 'Donepezil'  # <-- 修改这里
lead_smi = LEADS[SELECTED]
lead_mol = Chem.MolFromSmiles(lead_smi)
lead_fp = AllChem.GetMorganFingerprintAsBitVect(lead_mol, 2, nBits=2048)
lead_qed = QED.qed(lead_mol)
lead_logp = Descriptors.MolLogP(lead_mol)

print(f"先导: {SELECTED}")
print(f"SMILES: {lead_smi}")
print(f"QED={lead_qed:.3f}  LogP={lead_logp:.2f}  MW={Descriptors.MolWt(lead_mol):.1f}")

# ==================== 3. RNN核心优化策略 ====================
print("\n" + "=" * 70)
print("RNN驱动的先导优化")
print("=" * 70)

all_results = []

# --- 策略A: 多位置截断续写 ---
print("\n策略A: 多位置截断续写")
# 在不同位置截断，让RNN生成不同的后续
lead_len = len(lead_smi)
cut_positions = [int(lead_len * r) for r in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]

for cut_pos in cut_positions:
    prefix = lead_smi[:cut_pos]
    for temp in [0.5, 0.7, 0.9]:
        for _ in range(50):
            # 编码前缀
            tokens = [char_to_idx['<START>']]
            for c in prefix:
                if c in char_to_idx:
                    tokens.append(char_to_idx[c])

            input_idx = torch.tensor([tokens], dtype=torch.long).to(device)
            hidden = None
            generated = list(tokens)

            with torch.no_grad():
                logits, hidden = model(input_idx, hidden)
                for _ in range(100):
                    last_logits = logits[:, -1, :] / temp
                    probs = torch.softmax(last_logits, dim=-1)
                    next_idx = torch.multinomial(probs, 1).item()
                    if next_idx == char_to_idx['<END>']:
                        break
                    generated.append(next_idx)
                    input_idx = torch.tensor([[next_idx]], dtype=torch.long).to(device)
                    logits, hidden = model(input_idx, hidden)

            smi = decode_seq(generated)
            mol = Chem.MolFromSmiles(smi)
            if mol:
                canonical = Chem.MolToSmiles(mol)
                gen_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                sim = DataStructs.TanimotoSimilarity(lead_fp, gen_fp)
                if 0.3 <= sim <= 0.95:
                    all_results.append({
                        'SMILES': canonical,
                        'Strategy': f'Trunc_{cut_pos}',
                        'Sim': round(sim, 3),
                    })

print(f"  截断续写: {len(all_results)}个")

# --- 策略B: 前缀+RNN自由生成(以先导子结构为种子) ---
print("\n策略B: 子结构种子生成")
# 提取先导的子结构作为种子
substructs = []
# 取先导的前10、15、20个字符作为种子
for seed_len in [8, 12, 16, 20]:
    if seed_len < lead_len:
        substructs.append(lead_smi[:seed_len])

# 也从先导中随机取子串
for _ in range(5):
    start = random.randint(0, max(0, lead_len - 10))
    end = min(lead_len, start + random.randint(5, 15))
    substructs.append(lead_smi[start:end])

count_before = len(all_results)
for seed in substructs:
    for _ in range(30):
        tokens = [char_to_idx['<START>']]
        for c in seed:
            if c in char_to_idx:
                tokens.append(char_to_idx[c])

        input_idx = torch.tensor([tokens], dtype=torch.long).to(device)
        hidden = None
        generated = list(tokens)

        with torch.no_grad():
            logits, hidden = model(input_idx, hidden)
            for _ in range(100):
                last_logits = logits[:, -1, :] / 0.7
                probs = torch.softmax(last_logits, dim=-1)
                next_idx = torch.multinomial(probs, 1).item()
                if next_idx == char_to_idx['<END>']:
                    break
                generated.append(next_idx)
                input_idx = torch.tensor([[next_idx]], dtype=torch.long).to(device)
                logits, hidden = model(input_idx, hidden)

        smi = decode_seq(generated)
        mol = Chem.MolFromSmiles(smi)
        if mol and 10 <= mol.GetNumAtoms() <= 50:
            canonical = Chem.MolToSmiles(mol)
            gen_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            sim = DataStructs.TanimotoSimilarity(lead_fp, gen_fp)
            if 0.2 <= sim <= 0.9:
                all_results.append({
                    'SMILES': canonical,
                    'Strategy': f'Seed_{seed[:10]}',
                    'Sim': round(sim, 3),
                })

print(f"  子结构种子: {len(all_results) - count_before}个")

# --- 策略C: 先导+RNN局部修饰(保持核心,替换侧链) ---
print("\n策略C: 核心保持+侧链RNN替换")
# 找到先导中的环系统(核心),替换连接的侧链
ring_info = lead_mol.GetRingInfo()
ring_atoms = set()
for ring in ring_info.AtomRings():
    ring_atoms.update(ring)

# 找非环原子(侧链)
side_chain_atoms = [a.GetIdx() for a in lead_mol.GetAtoms() if a.GetIdx() not in ring_atoms]

if side_chain_atoms:
    # 将先导转为SMILES, 在侧链位置让RNN重新生成
    count_before = len(all_results)
    for _ in range(100):
        # 随机保留先导的一部分(包含核心环), 让RNN从那里续写
        # 简单方法: 截断到包含核心的位置
        core_end = max(ring_atoms) if ring_atoms else lead_len // 2
        # 在SMILES中找到对应位置(近似)
        prefix = lead_smi[:min(len(lead_smi), core_end + 5)]

        tokens = [char_to_idx['<START>']]
        for c in prefix:
            if c in char_to_idx:
                tokens.append(char_to_idx[c])

        input_idx = torch.tensor([tokens], dtype=torch.long).to(device)
        hidden = None
        generated = list(tokens)

        with torch.no_grad():
            logits, hidden = model(input_idx, hidden)
            for _ in range(80):
                last_logits = logits[:, -1, :] / 0.6
                probs = torch.softmax(last_logits, dim=-1)
                next_idx = torch.multinomial(probs, 1).item()
                if next_idx == char_to_idx['<END>']:
                    break
                generated.append(next_idx)
                input_idx = torch.tensor([[next_idx]], dtype=torch.long).to(device)
                logits, hidden = model(input_idx, hidden)

        smi = decode_seq(generated)
        mol = Chem.MolFromSmiles(smi)
        if mol:
            canonical = Chem.MolToSmiles(mol)
            gen_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            sim = DataStructs.TanimotoSimilarity(lead_fp, gen_fp)
            if 0.3 <= sim <= 0.95:
                all_results.append({
                    'SMILES': canonical,
                    'Strategy': 'Core_Keep',
                    'Sim': round(sim, 3),
                })

    print(f"  核心保持: {len(all_results) - count_before}个")
else:
    print("  跳过(先导无明显侧链)")

# --- 策略D: RNN从零生成+相似性过滤(间接优化) ---
print("\n策略D: 高相似性筛选")
count_before = len(all_results)
for temp in [0.5, 0.7]:
    for _ in range(300):
        tokens = [char_to_idx['<START>']]
        input_idx = torch.tensor([tokens], dtype=torch.long).to(device)
        hidden = None
        generated = [char_to_idx['<START>']]

        with torch.no_grad():
            logits, hidden = model(input_idx, hidden)
            for _ in range(100):
                last_logits = logits[:, -1, :] / temp
                probs = torch.softmax(last_logits, dim=-1)
                next_idx = torch.multinomial(probs, 1).item()
                if next_idx == char_to_idx['<END>']:
                    break
                generated.append(next_idx)
                input_idx = torch.tensor([[next_idx]], dtype=torch.long).to(device)
                logits, hidden = model(input_idx, hidden)

        smi = decode_seq(generated)
        mol = Chem.MolFromSmiles(smi)
        if mol:
            canonical = Chem.MolToSmiles(mol)
            gen_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            sim = DataStructs.TanimotoSimilarity(lead_fp, gen_fp)
            if 0.4 <= sim <= 0.85:  # 与先导中等相似
                all_results.append({
                    'SMILES': canonical,
                    'Strategy': 'HighSim_Filter',
                    'Sim': round(sim, 3),
                })

print(f"  高相似筛选: {len(all_results) - count_before}个")

# ==================== 4. 去重评估 ====================
print("\n" + "=" * 70)
print("去重与评估")
print("=" * 70)

seen = set()
unique = []
for a in all_results:
    if a['SMILES'] not in seen and a['SMILES'] != lead_smi:
        seen.add(a['SMILES'])
        unique.append(a)

print(f"总生成: {len(all_results)}, 去重后: {len(unique)}")

evaluated = []
for a in unique:
    mol = Chem.MolFromSmiles(a['SMILES'])
    if mol is None:
        continue

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    qed = QED.qed(mol)
    hba = Lipinski.NumHAcceptors(mol)
    hbd = Lipinski.NumHDonors(mol)
    tpsa = Descriptors.TPSA(mol)
    rings = Lipinski.RingCount(mol)

    lipinski = (mw <= 500 and logp <= 5 and hba <= 10 and hbd <= 5)
    is_novel = Chem.MolToSmiles(mol) not in train_smiles_set

    # 优化评分: QED提升 + 相似性保持
    qed_gain = qed - lead_qed
    opt_score = 0.4 * max(qed_gain, 0) + 0.3 * a['Sim'] + 0.2 * (1 if lipinski else 0) + 0.1 * (1 if is_novel else 0)

    evaluated.append({
        **a,
        'MW': round(mw, 1),
        'LogP': round(logp, 2),
        'QED': round(qed, 3),
        'HBA': hba,
        'HBD': hbd,
        'TPSA': round(tpsa, 1),
        'Rings': rings,
        'Lipinski': lipinski,
        'Novel': is_novel,
        'QED_gain': round(qed_gain, 3),
        'LogP_gain': round(lead_logp - logp, 2),
        'OptScore': round(opt_score, 4),
    })

eval_df = pd.DataFrame(evaluated)
eval_df = eval_df.sort_values('OptScore', ascending=False)

# ==================== 5. 输出结果 ====================
print(f"\n先导: {SELECTED}  QED={lead_qed:.3f}  LogP={lead_logp:.2f}")

# 各策略统计
print(f"\n各策略产出:")
for strat in eval_df['Strategy'].unique():
    subset = eval_df[eval_df['Strategy'] == strat]
    lip_rate = subset['Lipinski'].mean() * 100
    print(f"  {strat:<20}: {len(subset):>4}个 | 类药率={lip_rate:.0f}% | "
          f"平均QED={subset['QED'].mean():.3f} | 平均Sim={subset['Sim'].mean():.3f}")

# 有QED提升的
qed_improved = eval_df[eval_df['QED_gain'] > 0]
print(f"\nQED有提升: {len(qed_improved)}个")

# 类药且有提升
both = eval_df[(eval_df['Lipinski']) & (eval_df['QED_gain'] >= 0)]
print(f"类药且QED不降: {len(both)}个")

print(f"\n{'=' * 110}")
print(f"Top 30 优化类似物")
print(f"{'=' * 110}")
print(f"{'#':>3} {'SMILES':<48} {'MW':>6} {'LogP':>5} {'QED':>5} {'Sim':>5} "
      f"{'dQED':>6} {'dLogP':>6} {'Strat':<16} {'Lip':>3} {'Nov':>3}")
print(f"{'-' * 110}")

for i, (_, row) in enumerate(eval_df.head(30).iterrows()):
    smi = row['SMILES'][:46]
    lip = 'Y' if row['Lipinski'] else 'N'
    nov = 'Y' if row['Novel'] else 'N'
    print(f"{i+1:>3} {smi:<48} {row['MW']:>6} {row['LogP']:>5.1f} "
          f"{row['QED']:>5.3f} {row['Sim']:>5.2f} "
          f"{row['QED_gain']:>6.3f} {row['LogP_gain']:>6.2f} "
          f"{row['Strategy']:<16} {lip:>3} {nov:>3}")

# ==================== 6. 保存 ====================
eval_df.to_csv(f'rnn_lead_opt_{SELECTED}.csv', index=False)
both.to_csv(f'rnn_lead_opt_best_{SELECTED}.csv', index=False)

print(f"\n文件已保存:")
print(f"  rnn_lead_opt_{SELECTED}.csv      (全部)")
print(f"  rnn_lead_opt_best_{SELECTED}.csv  (类药+QED不降)")
