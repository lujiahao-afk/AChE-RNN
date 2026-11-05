import pandas as pd

# 读取TSV文件（注意是Tab分隔）（python chakan.py）
df = pd.read_csv('structure.tsv', sep='\t')

# 查看基本形状
print(f"总行数: {len(df)}")
print(f"总列数: {len(df.columns)}")

# 查看所有列名
print("\n列名列表:")
for i, col in enumerate(df.columns):
    print(f"  {i+1}. {col}")

# 查看前5行关键列
print("\n前5条数据（关键列）:")
print(df[['Molecule ChEMBL ID', 'Smiles', 'Standard Type',
          'Standard Value', 'Standard Units', 'pChEMBL Value']].head())

# 查看数据类型
print("\n数据类型:")
print(df[['Standard Value', 'pChEMBL Value', 'Molecular Weight']].dtypes)

# 查看缺失值
print("\n缺失值统计:")
print(df[['Smiles', 'Standard Value', 'pChEMBL Value']].isnull().sum())
