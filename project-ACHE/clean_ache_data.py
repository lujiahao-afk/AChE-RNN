import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
import warnings
warnings.filterwarnings('ignore')

RDLogger.DisableLog('rdApp.error')

# ========================================
# 2.1 读取数据
# ========================================
print("=" * 60)
print("步骤1: 读取数据")
print("=" * 60)

df = pd.read_csv('structure.tsv', sep='\t')
print(f"原始数据: {len(df)} 条记录")
print(f"列数: {len(df.columns)}")

# ========================================
# 2.2 查看数据分布
# ========================================
print("\n" + "=" * 60)
print("步骤2: 查看数据分布")
print("=" * 60)

# 查看Standard Type分布
print("\n活性类型分布:")
print(df['Standard Type'].value_counts())

# 查看Standard Units分布
print("\n单位分布:")
print(df['Standard Units'].value_counts())

# 查看IC₅₀值分布
print(f"\nIC₅₀值统计 (nM):")
print(f"  最小值: {df['Standard Value'].min()}")
print(f"  最大值: {df['Standard Value'].max()}")
print(f"  中位数: {df['Standard Value'].median()}")
print(f"  平均值: {df['Standard Value'].mean():.1f}")

# 查看pChEMBL值分布
print(f"\npChEMBL值统计:")
print(f"  最小值: {df['pChEMBL Value'].min()}")
print(f"  最大值: {df['pChEMBL Value'].max()}")
print(f"  中位数: {df['pChEMBL Value'].median()}")

# ========================================
# 2.3 筛选IC50类型 + 有效数据
# ========================================
print("\n" + "=" * 60)
print("步骤3: 筛选有效数据")
print("=" * 60)

# 只保留IC50类型
df_ic50 = df[df['Standard Type'] == 'IC50'].copy()
print(f"IC50类型: {len(df_ic50)} 条")

# 只保留nM单位
df_ic50 = df_ic50[df_ic50['Standard Units'] == 'nM'].copy()
print(f"单位为nM: {len(df_ic50)} 条")

# 排除Data Validity Comment不为空的（无效数据）
# 空字符串或NaN视为有效
df_valid = df_ic50[
    (df_ic50['Data Validity Comment'].isna()) |
    (df_ic50['Data Validity Comment'] == '')
].copy()
print(f"排除无效数据后: {len(df_valid)} 条")

# 排除SMILES为空的
df_valid = df_valid[df_valid['Smiles'].notna()].copy()
print(f"排除空SMILES后: {len(df_valid)} 条")

# ========================================
# 2.4 按活性强度筛选
# ========================================
print("\n" + "=" * 60)
print("步骤4: 按活性筛选")
print("=" * 60)

# 转换Standard Value为数值
df_valid['Standard Value'] = pd.to_numeric(
    df_valid['Standard Value'], errors='coerce'
)
df_valid = df_valid.dropna(subset=['Standard Value'])

# 不同阈值的分子数量
thresholds = [100, 500, 1000, 5000, 10000]
print("\n不同IC₅₀阈值下的分子数量:")
for t in thresholds:
    count = len(df_valid[df_valid['Standard Value'] <= t])
    print(f"  IC₅₀ ≤ {t:>6} nM: {count:>5} 个分子")

# 选择 IC₅₀ ≤ 10000 nM (10 μM) 的分子
IC50_THRESHOLD = 10000
df_active = df_valid[df_valid['Standard Value'] <= IC50_THRESHOLD].copy()
print(f"\n选择 IC₅₀ ≤ {IC50_THRESHOLD} nM: {len(df_active)} 条记录")

# ========================================
# 2.5 验证SMILES有效性
# ========================================
print("\n" + "=" * 60)
print("步骤5: 验证SMILES有效性")
print("=" * 60)

valid_indices = []
canonical_smiles_list = []
invalid_count = 0

for idx, row in df_active.iterrows():
    smi = row['Smiles']
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        try:
            can_smi = Chem.MolToSmiles(mol)
            valid_indices.append(idx)
            canonical_smiles_list.append(can_smi)
        except:
            invalid_count += 1
    else:
        invalid_count += 1

df_active = df_active.loc[valid_indices].copy()
df_active['canonical_smiles'] = canonical_smiles_list

print(f"无效SMILES: {invalid_count} 个")
print(f"有效SMILES: {len(df_active)} 个")

# ========================================
# 2.6 去重（同一分子可能有多个测定值）
# ========================================
print("\n" + "=" * 60)
print("步骤6: 去重")
print("=" * 60)

print(f"去重前: {len(df_active)} 条记录")

# 方法：同一canonical_smiles取IC₅₀中位数
df_dedup = df_active.groupby('canonical_smiles').agg({
    'Molecule ChEMBL ID': 'first',
    'Molecule Name': 'first',
    'Standard Value': 'median',          # IC₅₀取中位数
    'pChEMBL Value': 'median',           # pChEMBL取中位数
    'Molecular Weight': 'first',
    'AlogP': 'first'
}).reset_index()

# 重命名
df_dedup = df_dedup.rename(columns={'Standard Value': 'IC50_nM'})

print(f"去重后: {len(df_dedup)} 个唯一分子")

# ========================================
# 2.7 添加分子描述符（可选，用于后续分析）
# ========================================
print("\n" + "=" * 60)
print("步骤7: 添加分子描述符")
print("=" * 60)

mols = [Chem.MolFromSmiles(smi) for smi in df_dedup['canonical_smiles']]

df_dedup['MW_rdkit'] = [Descriptors.MolWt(m) for m in mols]
df_dedup['LogP_rdkit'] = [Descriptors.MolLogP(m) for m in mols]
df_dedup['HBA'] = [Descriptors.NumHAcceptors(m) for m in mols]
df_dedup['HBD'] = [Descriptors.NumHDonors(m) for m in mols]
df_dedup['TPSA'] = [Descriptors.TPSA(m) for m in mols]
df_dedup['RotBonds'] = [Descriptors.NumRotatableBonds(m) for m in mols]

# 检查Lipinski五规则
df_dedup['Lipinski'] = (
    (df_dedup['MW_rdkit'] <= 500) &
    (df_dedup['LogP_rdkit'] <= 5) &
    (df_dedup['HBA'] <= 10) &
    (df_dedup['HBD'] <= 5)
)

print(f"Lipinski类药分子: {df_dedup['Lipinski'].sum()} / {len(df_dedup)}")

# 描述符统计
print("\n分子描述符统计:")
for col in ['MW_rdkit', 'LogP_rdkit', 'HBA', 'HBD', 'TPSA', 'IC50_nM']:
    print(f"  {col:>12}: {df_dedup[col].min():.1f} - {df_dedup[col].max():.1f} "
          f"(均值: {df_dedup[col].mean():.1f})")

# ========================================
# 2.8 保存结果
# ========================================
print("\n" + "=" * 60)
print("步骤8: 保存结果")
print("=" * 60)

# 保存完整数据（含描述符）
df_dedup.to_csv('ache_inhibitors_cleaned.csv', index=False)
print(f"完整数据已保存: ache_inhibitors_cleaned.csv ({len(df_dedup)} 个分子)")

# 保存纯SMILES文件（用于REINVENT训练）
with open('ache_smiles.txt', 'w') as f:
    for smi in df_dedup['canonical_smiles']:
        f.write(smi + '\n')
print(f"SMILES文件已保存: ache_smiles.txt")

# 保存强抑制剂（IC₅₀ ≤ 1000 nM）
df_strong = df_dedup[df_dedup['IC50_nM'] <= 1000]
with open('ache_smiles_strong.txt', 'w') as f:
    for smi in df_strong['canonical_smiles']:
        f.write(smi + '\n')
print(f"强抑制剂SMILES已保存: ache_smiles_strong.txt ({len(df_strong)} 个)")

# 保存类药分子
df_druglike = df_dedup[df_dedup['Lipinski'] == True]
with open('ache_smiles_druglike.txt', 'w') as f:
    for smi in df_druglike['canonical_smiles']:
        f.write(smi + '\n')
print(f"类药分子SMILES已保存: ache_smiles_druglike.txt ({len(df_druglike)} 个)")

print("\n" + "=" * 60)
print("数据清洗完成！")
print("=" * 60)
print(f"  原始记录: {len(df)}")
print(f"  最终分子: {len(df_dedup)}")
print(f"  强抑制剂: {len(df_strong)} (IC₅₀ ≤ 1000 nM)")
print(f"  类药分子: {len(df_druglike)}")
