import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED, AllChem, DataStructs
from rdkit.Chem import BRICS, Draw, RWMol, rdMMPA
from rdkit import RDLogger
import random
import itertools

RDLogger.DisableLog('rdApp.*')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 1. 定义先导化合物库 ====================
print("=" * 70)
print("步骤1: 定义先导化合物库")
print("=" * 70)

# 已知AChE抑制剂 + 您生成的高QED分子作为候选先导，也可以替换为自己发现的先导化合物
LEAD_CANDIDATES = {
    # --- 已知药物 ---
    'Donepezil (多奈哌齐)': {
        'smiles': 'O=C1C=CC(=O)c2cc(OCCN3CCCCC3)c(OCc3ccccc3)cc21',
        'source': '已上市药物',
        'note': '最常用AChE抑制剂'
    },
    'Galantamine (加兰他敏)': {
        'smiles': 'COc1ccc2c3c1O[C@H]1C=C[C@@H](O)C[C@@]13CCN(C)C2',
        'source': '已上市药物',
        'note': '天然生物碱'
    },
    'Rivastigmine (利凡斯的明)': {
        'smiles': 'CCN(C)C(=O)Oc1cccc(C(C)C)c1',
        'source': '已上市药物',
        'note': '氨基甲酸酯类'
    },
    'Tacrine (他克林)': {
        'smiles': 'Nc1c2c(nc3ccccc13)CCCC2',
        'source': '已上市药物',
        'note': '第一代AChE抑制剂(已撤市)'
    },
    # --- 您生成的高QED分子 ---
    'Generated_#1 (QED=0.930)': {
        'smiles': 'O=C(CCN1CCCC1)Nc1ccc2nc3n(c(=O)c2c1)CCC3',
        'source': 'RNN生成',
        'note': '最高QED生成分子'
    },
    'Generated_#2 (QED=0.911)': {
        'smiles': 'COc1cccc(CN(C)CC2=Nc3nc4c(N)c32)CCCC4)c1',
        'source': 'RNN生成',
        'note': '氨基喹啉骨架'
    },
    'Generated_#13 (QED=0.859)': {
        'smiles': 'COc1ccc2c3c1OC1C[C@@H](O)CC[C@@]31CCN(C)C2',
        'source': 'RNN生成',
        'note': '类似加兰他敏骨架'
    },
}

# 打印先导化合物信息
print(f"\n{'Name':<35} {'MW':>7} {'LogP':>6} {'QED':>6} {'HBA':>4} {'HBD':>4} {'TPSA':>7} {'Source':<10}")
print("-" * 100)

for name, info in LEAD_CANDIDATES.items():
    mol = Chem.MolFromSmiles(info['smiles'])
    if mol:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        qed = QED.qed(mol)
        hba = Lipinski.NumHAcceptors(mol)
        hbd = Lipinski.NumHDonors(mol)
        tpsa = Descriptors.TPSA(mol)
        print(f"{name:<35} {mw:>7.1f} {logp:>6.2f} {qed:>6.3f} {hba:>4} {hbd:>4} {tpsa:>7.1f} {info['source']:<10}")

# ==================== 2. 用户选择先导化合物 ====================
# 修改这里选择要优化的先导化合物
SELECTED_LEAD = 'Donepezil (多奈哌齐)'  # <-- 修改这里选择不同先导

# 定义优化目标
OPTIMIZATION_GOALS = {
    'reduce_LogP': True,      # 降低LogP (改善水溶性)
    'target_LogP': 3.0,       # 目标LogP
    'improve_QED': True,      # 提高QED
    'maintain_MW': True,      # 保持分子量在合理范围
    'target_MW': (300, 500),  # 目标分子量范围
    'reduce_HBD': False,      # 是否减少氢键供体
}

print(f"\n选择的先导化合物: {SELECTED_LEAD}")
print(f"SMILES: {LEAD_CANDIDATES[SELECTED_LEAD]['smiles']}")
print(f"优化目标: {OPTIMIZATION_GOALS}")

# ==================== 3. BRICS片段分解 ====================
print("\n" + "=" * 70)
print("步骤2: BRICS分子片段分解")
print("=" * 70)

lead_smi = LEAD_CANDIDATES[SELECTED_LEAD]['smiles']
lead_mol = Chem.MolFromSmiles(lead_smi)

# BRICS分解
fragments = list(BRICS.BRICSDecompose(lead_mol, returnMols=True))
frag_smiles = [Chem.MolToSmiles(f) for f in fragments]

print(f"\n先导化合物: {lead_smi}")
print(f"BRICS分解得到 {len(frag_smiles)} 个片段:")
for i, fs in enumerate(frag_smiles):
    print(f"  片段{i+1}: {fs}")

# ==================== 4. 定义替换片段库 ====================
print("\n" + "=" * 70)
print("步骤3: 构建替换片段库")
print("=" * 70)

# 常用药物化学替换片段
REPLACEMENT_FRAGMENTS = {
    # --- 芳香环替换 ---
    'aromatic': [
        'c1ccccc1',          # 苯
        'c1cccnc1',          # 吡啶
        'c1ccncc1',          # 吡啶(另一种编号)
        'c1cc[nH]c1',        # 吡咯
        'c1ccoc1',           # 呋喃
        'c1ccsc1',           # 噻吩
        'c1cncnc1',          # 嘧啶
        'c1ccc2[nH]ccc2c1',  # 吲哚
        'c1ccc2ncccc2c1',    # 喹啉
        'c1ccc2c(c1)ncn2',   # 苯并咪唑
    ],
    # --- 杂环替换 ---
    'heterocycle': [
        'C1CCNCC1',          # 哌啶
        'C1CCOCC1',          # 吗啉
        'C1CCN(C1)C',        # N-甲基吡咯烷
        'C1CNCCN1',          # 哌嗪
        'C1COCCN1',          # 吗啉
        'C1=CCNCC1',         # 四氢吡啶
        'c1cc[nH]n1',        # 吡唑
        'c1ccno1',           # 异噁唑
        'c1cn[nH]c1',        # 咪唑
        'C1CCCCN1',          # 己内酰胺
    ],
    # --- 极性基团替换 ---
    'polar': [
        'O',                 # 羟基
        'C(=O)O',            # 羧基
        'C(=O)N',            # 酰胺
        'NC(=O)N',           # 脲
        'C(=O)OC',           # 甲酯
        'S(=O)(=O)N',        # 磺酰胺
        'C#N',               # 氰基
        '[N+](=O)[O-]',      # 硝基
        'N(C)C',             # 二甲胺
        'N1CCOCC1',          # 吗啉基
    ],
    # --- 链长变化 ---
    'linker': [
        '',                  # 直接连接
        'C',                 # 亚甲基
        'CC',                # 亚乙基
        'CCC',               # 三碳链
        'C=C',               # 双键
        'C#C',               # 三键
        'CO',                # 醚键
        'CNC',               # 甲胺基
        'C(=O)NH',           # 酰胺键
        'CC(=O)',            # 乙酰基
    ],
    # --- 卤素替换 ---
    'halogen': [
        'F',                 # 氟
        'Cl',                # 氯
        'Br',                # 溴
        'CF3',               # 三氟甲基
        'OC(F)(F)F',         # 三氟甲氧基
    ],
}

for cat, frags in REPLACEMENT_FRAGMENTS.items():
    print(f"  {cat}: {len(frags)}个片段")

# ==================== 5. RNN引导的先导优化 ====================
print("\n" + "=" * 70)
print("步骤4: RNN引导的先导化合物优化")
print("=" * 70)

# 加载RNN模型
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

def decode_seq(indices):
    chars = []
    for idx in indices:
        c = idx_to_char.get(idx, '')
        if c == '<END>':
            break
        if c not in ['<PAD>', '<START>']:
            chars.append(c)
    return ''.join(chars)

def generate_from_prefix(model, prefix_smi, temperature=0.7, max_len=120):
    """以先导化合物的部分SMILES为前缀, 让RNN续写生成类似物"""
    # 编码前缀
    prefix_tokens = [char_to_idx['<START>']]
    for c in prefix_smi:
        if c in char_to_idx:
            prefix_tokens.append(char_to_idx[c])

    input_idx = torch.tensor([prefix_tokens], dtype=torch.long).to(device)
    hidden = None
    generated = list(prefix_tokens)

    with torch.no_grad():
        # 先把前缀过一遍RNN获取hidden state
        logits, hidden = model(input_idx, hidden)

        # 从最后一个字符开始续写
        for _ in range(max_len):
            last_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(last_logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()

            if next_idx == char_to_idx['<END>']:
                break

            generated.append(next_idx)
            input_idx = torch.tensor([[next_idx]], dtype=torch.long).to(device)
            logits, hidden = model(input_idx, hidden)

    return decode_seq(generated)

# ==================== 6. 多策略生成类似物 ====================
print("\n生成类似物中...")

lead_fp = AllChem.GetMorganFingerprintAsBitVect(lead_mol, 2, nBits=2048)

all_analogs = []

# --- 策略1: 截断先导化合物, 让RNN续写 ---
print("\n策略1: 截断续写 (Truncate & Continue)")
truncation_ratios = [0.3, 0.5, 0.7]  # 截断先导的不同比例
for ratio in truncation_ratios:
    trunc_len = max(5, int(len(lead_smi) * ratio))
    prefix = lead_smi[:trunc_len]

    for _ in range(200):
        smi = generate_from_prefix(model, prefix, temperature=0.7)
        mol = Chem.MolFromSmiles(smi)
        if mol:
            canonical = Chem.MolToSmiles(mol)
            gen_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            sim = DataStructs.TanimotoSimilarity(lead_fp, gen_fp)
            if 0.3 <= sim <= 0.95:  # 与先导有一定相似但不完全相同
                all_analogs.append({
                    'SMILES': canonical,
                    'Strategy': f'Truncate_{ratio}',
                    'Similarity_to_lead': round(sim, 3),
                    'Prefix': prefix[:30] + '...'
                })

print(f"  截断续写: {len([a for a in all_analogs if 'Truncate' in a['Strategy']])}个类似物")

# --- 策略2: BRICS重组 (片段嫁接) ---
print("\n策略2: BRICS片段重组 (Fragment Recombination)")
try:
    # 用训练集分子的BRICS片段作为替换池
    train_df = pd.read_csv('ache_inhibitors_cleaned.csv')
    pool_fragments = set()

    for smi in train_df['canonical_smiles'].dropna().sample(min(500, len(train_df))):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            frags = BRICS.BRICSDecompose(mol)
            pool_fragments.update(frags)

    pool_fragments = list(pool_fragments)
    print(f"  片段池大小: {len(pool_fragments)}")

    # 尝试BRICS重组
    for _ in range(300):
        # 随机选2-4个片段组合
        n_frags = random.randint(2, 4)
        selected = random.sample(pool_fragments, min(n_frags, len(pool_fragments)))

        try:
            combined = BRICS.BRICSBuild([Chem.MolFromSmiles(f) for f in selected if Chem.MolFromSmiles(f)])
            for product in itertools.islice(combined, 5):  # 每组合最多取5个产物
                smi = Chem.MolToSmiles(product)
                mol = Chem.MolFromSmiles(smi)
                if mol and 10 <= mol.GetNumAtoms() <= 50:
                    canonical = Chem.MolToSmiles(mol)
                    gen_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    sim = DataStructs.TanimotoSimilarity(lead_fp, gen_fp)
                    if 0.2 <= sim <= 0.9:
                        all_analogs.append({
                            'SMILES': canonical,
                            'Strategy': 'BRICS_Recomb',
                            'Similarity_to_lead': round(sim, 3),
                            'Prefix': '+'.join(selected[:3])
                        })
        except:
            continue

    brics_count = len([a for a in all_analogs if 'BRICS' in a['Strategy']])
    print(f"  BRICS重组: {brics_count}个类似物")
except Exception as e:
    print(f"  BRICS重组出错: {e}")

# --- 策略3: 侧链替换 (Side Chain Replacement) ---
print("\n策略3: 侧链替换 (Side Chain Replacement)")
try:
    # 用RDKit的原子编辑方式替换侧链
    for _ in range(300):
        mol = Chem.RWMol(lead_mol)

        # 随机选择一个可编辑的原子
        atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() in [6, 7, 8]
                 and a.GetDegree() <= 3 and not a.IsInRing()]

        if not atoms:
            continue

        atom = random.choice(atoms)
        idx = atom.GetIdx()

        # 随机替换为其他基团
        replacements = {
            6: [7, 8],       # C -> N, O
            7: [6, 8],       # N -> C, O
            8: [6, 7],       # O -> C, N
        }

        new_atomic = random.choice(replacements.get(atom.GetAtomicNum(), [6]))
        mol.ReplaceAtom(idx, new_atomic)

        try:
            new_mol = mol.GetMol()
            Chem.SanitizeMol(new_mol)
            smi = Chem.MolToSmiles(new_mol)

            if smi != lead_smi:
                gen_fp = AllChem.GetMorganFingerprintAsBitVect(new_mol, 2, nBits=2048)
                sim = DataStructs.TanimotoSimilarity(lead_fp, gen_fp)
                if sim > 0.5:
                    all_analogs.append({
                        'SMILES': smi,
                        'Strategy': 'Atom_Swap',
                        'Similarity_to_lead': round(sim, 3),
                        'Prefix': f'Atom{idx}:{atom.GetSymbol()}->{Chem.Atom(new_atomic).GetSymbol()}'
                    })
        except:
            continue

    swap_count = len([a for a in all_analogs if 'Atom_Swap' in a['Strategy']])
    print(f"  原子替换: {swap_count}个类似物")
except Exception as e:
    print(f"  原子替换出错: {e}")

# ==================== 7. 去重与评估 ====================
print("\n" + "=" * 70)
print("步骤5: 去重与综合评估")
print("=" * 70)

# 去重
seen = set()
unique_analogs = []
for a in all_analogs:
    if a['SMILES'] not in seen:
        seen.add(a['SMILES'])
        unique_analogs.append(a)

print(f"总生成: {len(all_analogs)}, 去重后: {len(unique_analogs)}")

# 计算完整属性
evaluated = []
for a in unique_analogs:
    mol = Chem.MolFromSmiles(a['SMILES'])
    if mol is None:
        continue

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    qed = QED.qed(mol)
    hba = Lipinski.NumHAcceptors(mol)
    hbd = Lipinski.NumHDonors(mol)
    tpsa = Descriptors.TPSA(mol)
    rotbonds = Lipinski.NumRotatableBonds(mol)

    lipinski = (mw <= 500 and logp <= 5 and hba <= 10 and hbd <= 5)

    # 与先导的差异度 (1-sim)
    lead_mw = Descriptors.MolWt(lead_mol)
    lead_logp = Descriptors.MolLogP(lead_mol)
    lead_qed = QED.qed(lead_mol)

    # 综合优化评分
    opt_score = 0
    if OPTIMIZATION_GOALS['reduce_LogP']:
        # LogP改善奖励
        logp_improve = max(0, lead_logp - logp) / max(abs(lead_logp), 1)
        opt_score += 0.3 * min(logp_improve, 1)
    if OPTIMIZATION_GOALS['improve_QED']:
        qed_improve = max(0, qed - lead_qed)
        opt_score += 0.4 * qed_improve
    # 保持相似性 (不能偏离太远)
    sim_bonus = 0.3 * a['Similarity_to_lead']
    opt_score += sim_bonus

    evaluated.append({
        **a,
        'MW': round(mw, 1),
        'LogP': round(logp, 2),
        'QED': round(qed, 3),
        'HBA': hba,
        'HBD': hbd,
        'TPSA': round(tpsa, 1),
        'RotBonds': rotbonds,
        'Lipinski': lipinski,
        'OptScore': round(opt_score, 4),
        'MW_change': round(mw - lead_mw, 1),
        'LogP_change': round(logp - lead_logp, 2),
        'QED_change': round(qed - lead_qed, 3),
    })

eval_df = pd.DataFrame(evaluated)

# ==================== 8. 输出结果 ====================
print(f"\n先导化合物属性:")
print(f"  SMILES: {lead_smi}")
print(f"  MW={Descriptors.MolWt(lead_mol):.1f}  LogP={Descriptors.MolLogP(lead_mol):.2f}  "
      f"QED={QED.qed(lead_mol):.3f}")

# 按优化评分排序
eval_df = eval_df.sort_values('OptScore', ascending=False)

# 筛选类药且有改善的
improved = eval_df[
    (eval_df['Lipinski'] == True) &
    (eval_df['QED_change'] >= -0.05)  # QED不能下降太多
]

print(f"\n类药且性质保持/改善: {len(improved)}个")

# 按策略统计
print(f"\n各策略产出:")
for strat in eval_df['Strategy'].unique():
    n = len(eval_df[eval_df['Strategy'] == strat])
    n_lip = len(eval_df[(eval_df['Strategy'] == strat) & (eval_df['Lipinski'])])
    print(f"  {strat}: {n}个 (类药{n_lip}个)")

print(f"\n{'=' * 110}")
print(f"Top 30 优化类似物 (按OptScore排序)")
print(f"{'=' * 110}")
print(f"{'#':>3} {'SMILES':<48} {'MW':>6} {'LogP':>5} {'QED':>5} {'Sim':>5} "
      f"{'dMW':>6} {'dLogP':>6} {'dQED':>6} {'Strategy':<14}")
print(f"{'-' * 110}")

for i, (_, row) in enumerate(eval_df.head(30).iterrows()):
    smi = row['SMILES'][:46]
    print(f"{i+1:>3} {smi:<48} {row['MW']:>6} {row['LogP']:>5.1f} "
          f"{row['QED']:>5.3f} {row['Similarity_to_lead']:>5.2f} "
          f"{row['MW_change']:>6.1f} {row['LogP_change']:>6.2f} "
          f"{row['QED_change']:>6.3f} {row['Strategy']:<14}")

# ==================== 9. 与先导对比分析 ====================
print(f"\n{'=' * 70}")
print("与先导化合物对比分析")
print(f"{'=' * 70}")

top5 = eval_df.head(5)
print(f"\n先导: MW={Descriptors.MolWt(lead_mol):.1f}  LogP={Descriptors.MolLogP(lead_mol):.2f}  QED={QED.qed(lead_mol):.3f}")
print(f"\nTop5类似物变化:")
for i, (_, row) in enumerate(top5.iterrows()):
    print(f"  #{i+1}: MW {row['MW_change']:+.1f} | LogP {row['LogP_change']:+.2f} | QED {row['QED_change']:+.3f} | Sim={row['Similarity_to_lead']:.3f}")

# 改善最多的分子
if len(eval_df[eval_df['QED_change'] > 0]) > 0:
    best_qed = eval_df.loc[eval_df['QED_change'].idxmax()]
    print(f"\nQED改善最多: +{best_qed['QED_change']:.3f}")
    print(f"  SMILES: {best_qed['SMILES']}")
    print(f"  QED: {best_qed['QED']} (先导: {QED.qed(lead_mol):.3f})")

if len(eval_df[eval_df['LogP_change'] < 0]) > 0:
    best_logp = eval_df.loc[eval_df['LogP_change'].idxmin()]
    print(f"\nLogP降低最多: {best_logp['LogP_change']:.2f}")
    print(f"  SMILES: {best_logp['SMILES']}")
    print(f"  LogP: {best_logp['LogP']} (先导: {Descriptors.MolLogP(lead_mol):.2f})")

# ==================== 10. 保存结果 ====================
eval_df.to_csv(f'lead_opt_{SELECTED_LEAD.split("(")[0].strip()}.csv', index=False)
improved.to_csv(f'lead_opt_improved_{SELECTED_LEAD.split("(")[0].strip()}.csv', index=False)

print(f"\n文件已保存:")
print(f"  lead_opt_*.csv          (全部类似物)")
print(f"  lead_opt_improved_*.csv (改善类似物)")
