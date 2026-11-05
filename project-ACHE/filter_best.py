import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED, AllChem, DataStructs
import numpy as np

# ==================== 1. 加载全部生成结果 ====================
# 找到最新的采样文件
import glob
all_files = sorted(glob.glob('samples_all_*.csv'))
druglike_files = sorted(glob.glob('samples_druglike_*.csv'))

df = pd.read_csv(all_files[-1])
print(f"加载分子数: {len(df)}")

# ==================== 2. 严格类药性过滤 ====================
print("\n" + "=" * 60)
print("严格类药性筛选")
print("=" * 60)

# 药物化学常用过滤规则
def strict_druglike_filter(row):
    """严格的类药性过滤"""
    rules = {
        'MW_200_500': 200 <= row['MW'] <= 500,
        'LogP_0_5': 0 <= row['LogP'] <= 5,
        'HBA_2_10': 2 <= row['HBA'] <= 10,
        'HBD_0_5': 0 <= row['HBD'] <= 5,
        'TPSA_20_130': 20 <= row['TPSA'] <= 130,
        'RotBonds_0_10': row['RotBonds'] <= 10,
        'Rings_1_5': 1 <= row['Rings'] <= 5,
        'Atoms_10_50': 10 <= row['Atoms'] <= 50,
        'QED_ge_0.5': row['QED'] >= 0.5,
        'Novel': row['IsNovel'] == True,
    }
    return all(rules.values()), rules

# 应用过滤
passed = []
failed_reasons = []

for _, row in df.iterrows():
    passed_filter, rules = strict_druglike_filter(row)
    if passed_filter:
        passed.append(row)
    else:
        failed_reasons.append({k: v for k, v in rules.items() if not v})

strict_df = pd.DataFrame(passed)
print(f"严格过滤后: {len(strict_df)} / {len(df)} ({len(strict_df)/len(df)*100:.1f}%)")

# 统计失败原因
if failed_reasons:
    reason_counts = {}
    for reasons in failed_reasons:
        for r in reasons:
            reason_counts[r] = reason_counts.get(r, 0) + 1
    print("\n未通过原因统计:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}个 ({count/len(df)*100:.1f}%)")

# ==================== 3. 综合评分排序 ====================
print("\n" + "=" * 60)
print("综合评分排序")
print("=" * 60)

def composite_score(row):
    """
    综合评分 = QED权重 + 相似性权重 + Lipinski权重 + 新颖性权重
    """
    score = 0.0

    # QED (0~1, 越高越好)
    score += 0.4 * row['QED']

    # 相似性 (0.3~0.7最佳, 太低可能无效, 太高缺乏新颖性)
    sim = row['MaxSimilarity']
    if 0.3 <= sim <= 0.7:
        sim_score = 1.0
    elif sim < 0.3:
        sim_score = sim / 0.3
    else:
        sim_score = max(0, 1 - (sim - 0.7) / 0.3)
    score += 0.2 * sim_score

    # 分子量在理想范围 (300~450)
    mw = row['MW']
    if 300 <= mw <= 450:
        mw_score = 1.0
    elif mw < 300:
        mw_score = mw / 300
    else:
        mw_score = max(0, 1 - (mw - 450) / 200)
    score += 0.15 * mw_score

    # LogP在理想范围 (1~4)
    logp = row['LogP']
    if 1 <= logp <= 4:
        lp_score = 1.0
    elif logp < 1:
        lp_score = max(0, logp)
    else:
        lp_score = max(0, 1 - (logp - 4) / 3)
    score += 0.15 * lp_score

    # TPSA在理想范围 (40~90)
    tpsa = row['TPSA']
    if 40 <= tpsa <= 90:
        tpsa_score = 1.0
    else:
        tpsa_score = max(0, 1 - abs(tpsa - 65) / 65)
    score += 0.1 * tpsa_score

    return round(score, 4)

strict_df['CompositeScore'] = strict_df.apply(composite_score, axis=1)
strict_df = strict_df.sort_values('CompositeScore', ascending=False)

# ==================== 4. 输出Top分子 ====================
print(f"\n筛选后Top 30分子 (综合评分排序):")
print(f"{'=' * 110}")
print(f"{'#':>3} {'SMILES':<50} {'MW':>6} {'LogP':>5} {'QED':>5} {'Sim':>5} "
      f"{'TPSA':>6} {'Score':>6}")
print(f"{'-' * 110}")

for i, (_, row) in enumerate(strict_df.head(30).iterrows()):
    smi = row['SMILES'][:48]
    print(f"{i+1:>3} {smi:<50} {row['MW']:>6} {row['LogP']:>5.1f} "
          f"{row['QED']:>5.3f} {row['MaxSimilarity']:>5.2f} "
          f"{row['TPSA']:>6.1f} {row['CompositeScore']:>6.4f}")

# ==================== 5. 结构多样性聚类 ====================
print(f"\n" + "=" * 60)
print("结构多样性分析 (Top50)")
print("=" * 60)

top50 = strict_df.head(50).copy()
fps = []
for smi in top50['SMILES']:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    else:
        fps.append(None)

# 计算成对相似度矩阵
n = len(fps)
sim_matrix = np.zeros((n, n))
for i in range(n):
    if fps[i] is None:
        continue
    for j in range(i+1, n):
        if fps[j] is None:
            continue
        sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
        sim_matrix[i][j] = sim
        sim_matrix[j][i] = sim

# 平均相似度 (越低越多样)
upper_tri = sim_matrix[np.triu_indices(n, k=1)]
avg_sim = upper_tri[upper_tri > 0].mean() if len(upper_tri[upper_tri > 0]) > 0 else 0

print(f"Top50分子平均两两相似度: {avg_sim:.3f}")
if avg_sim < 0.4:
    print("  -> 多样性: 优秀")
elif avg_sim < 0.6:
    print("  -> 多样性: 良好")
else:
    print("  -> 多样性: 一般 (结构较集中)")

# 找出最不相似的分子对
min_sim_idx = np.unravel_index(np.argmin(sim_matrix + np.eye(n) * 2), sim_matrix.shape)
print(f"\n最不相似的一对分子 (相似度={sim_matrix[min_sim_idx]:.3f}):")
print(f"  A: {top50.iloc[min_sim_idx[0]]['SMILES']}")
print(f"  B: {top50.iloc[min_sim_idx[1]]['SMILES']}")

# ==================== 6. 与已知药物对比 ====================
print(f"\n" + "=" * 60)
print("与已知乙酰胆碱酯酶抑制剂对比")
print("=" * 60)

# 已知AChE抑制剂
known_drugs = {
    'Donepezil': 'O=C1C=CC(=O)c2cc(OCCN3CCCCC3)c(OCc3ccccc3)cc21',
    'Rivastigmine': 'CCN(C)C(=O)OC1=CC=CC(=C1)C(C)C',
    'Galantamine': 'COc1ccc2c3c1O[C@H]1C=C[C@@H](O)C[C@@]31CCN(C)C2',
}

print(f"\n已知药物属性:")
print(f"{'Drug':<15} {'MW':>7} {'LogP':>6} {'QED':>6} {'HBA':>4} {'HBD':>4} {'TPSA':>6}")
print(f"{'-' * 60}")
for name, smi in known_drugs.items():
    mol = Chem.MolFromSmiles(smi)
    if mol:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        qed = QED.qed(mol)
        hba = Lipinski.NumHAcceptors(mol)
        hbd = Lipinski.NumHDonors(mol)
        tpsa = Descriptors.TPSA(mol)
        print(f"{name:<15} {mw:>7.1f} {logp:>6.2f} {qed:>6.3f} {hba:>4} {hbd:>4} {tpsa:>6.1f}")

# 找与已知药物最相似的生成分子
print(f"\n与已知药物最相似的生成分子:")
for name, smi in known_drugs.items():
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    ref_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

    best_sim = 0
    best_smi = ""
    for _, row in strict_df.iterrows():
        gen_mol = Chem.MolFromSmiles(row['SMILES'])
        if gen_mol:
            gen_fp = AllChem.GetMorganFingerprintAsBitVect(gen_mol, 2, nBits=2048)
            sim = DataStructs.TanimotoSimilarity(ref_fp, gen_fp)
            if sim > best_sim:
                best_sim = sim
                best_smi = row['SMILES']

    print(f"\n  {name} (相似度={best_sim:.3f}):")
    print(f"    {best_smi}")

# ==================== 7. 保存最终结果 ====================
strict_df.to_csv('final_druglike_molecules.csv', index=False)
strict_df.head(50).to_csv('final_top50_molecules.csv', index=False)

print(f"\n" + "=" * 60)
print("文件已保存:")
print(f"  final_druglike_molecules.csv  ({len(strict_df)}个严格筛选分子)")
print(f"  final_top50_molecules.csv     (Top50分子)")
print("=" * 60)
