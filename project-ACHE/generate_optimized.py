import torch
import torch.nn as nn
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED
import numpy as np

# ==================== 1. 加载模型 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

model = SmilesRNN(hp['vocab_size'], hp['embed_dim'], hp['hidden_dim'], hp['num_layers']).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("模型加载成功!")

# ==================== 2. 改进的生成函数 ====================
def decode_sequence(indices):
    chars = []
    for idx in indices:
        c = idx_to_char.get(idx, '')
        if c == '<END>':
            break
        if c not in ['<PAD>', '<START>']:
            chars.append(c)
    return ''.join(chars)

def generate_smiles(model, temperature=0.7, max_len=120):
    """改进的生成函数: 更长序列 + 温度采样"""
    input_idx = torch.tensor([[char_to_idx['<START>']]], dtype=torch.long).to(device)
    generated = [char_to_idx['<START>']]
    hidden = None

    with torch.no_grad():
        for step in range(max_len):
            logits, hidden = model(input_idx, hidden)
            last_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(last_logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()

            if next_idx == char_to_idx['<END>']:
                break

            generated.append(next_idx)
            input_idx = torch.tensor([[next_idx]], dtype=torch.long).to(device)

    return decode_sequence(generated)

def is_valid_smiles(smi):
    """检查SMILES是否有效"""
    if len(smi) < 5:
        return False
    mol = Chem.MolFromSmiles(smi)
    return mol is not None

def calc_properties(smi):
    """计算分子属性"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return {
        'SMILES': smi,
        'MW': round(Descriptors.MolWt(mol), 1),
        'LogP': round(Descriptors.MolLogP(mol), 2),
        'HBA': Lipinski.NumHAcceptors(mol),
        'HBD': Lipinski.NumHDonors(mol),
        'TPSA': round(Descriptors.TPSA(mol), 1),
        'RotBonds': Lipinski.NumRotatableBonds(mol),
        'QED': round(QED.qed(mol), 3),
        'Lipinski': (Descriptors.MolWt(mol) <= 500 and
                     Descriptors.MolLogP(mol) <= 5 and
                     Lipinski.NumHAcceptors(mol) <= 10 and
                     Lipinski.NumHDonors(mol) <= 5)
    }

# ==================== 3. 批量生成 ====================
print("\n" + "=" * 50)
print("使用不同温度批量生成分子")
print("=" * 50)

# 加载训练集用于计算新颖性
train_df = pd.read_csv('ache_inhibitors_cleaned.csv')
train_smiles_set = set()
for smi in train_df['canonical_smiles'].dropna():
    mol = Chem.MolFromSmiles(smi)
    if mol:
        train_smiles_set.add(Chem.MolToSmiles(mol))

all_results = []
temperatures = [0.5, 0.7, 0.9, 1.0]

for temp in temperatures:
    valid_list = []
    attempts = 300

    for _ in range(attempts):
        smi = generate_smiles(model, temperature=temp, max_len=120)
        if is_valid_smiles(smi):
            mol = Chem.MolFromSmiles(smi)
            canonical = Chem.MolToSmiles(mol)
            if canonical not in [v['SMILES'] for v in valid_list]:
                props = calc_properties(canonical)
                if props:
                    props['Temperature'] = temp
                    props['IsNovel'] = canonical not in train_smiles_set
                    valid_list.append(props)

    print(f"\n温度={temp}: 尝试{attempts}次, 有效={len(valid_list)}个 "
          f"(有效率={len(valid_list)/attempts*100:.1f}%)")
    all_results.extend(valid_list)

# ==================== 4. 汇总结果 ====================
results_df = pd.DataFrame(all_results)

print("\n" + "=" * 50)
print("生成结果汇总")
print("=" * 50)
print(f"总有效分子数: {len(results_df)}")
print(f"新颖分子数: {results_df['IsNovel'].sum()}")
print(f"Lipinski合规数: {results_df['Lipinski'].sum()}")

# 按QED排序, 筛选类药分子
druglike = results_df[(results_df['Lipinski'] == True) & (results_df['IsNovel'] == True)]
druglike = druglike.sort_values('QED', ascending=False)

print(f"\n类药且新颖的分子数: {len(druglike)}")
print("\nTop 20 类药分子 (按QED排序):")
print("-" * 100)
print(f"{'SMILES':<60} {'MW':>7} {'LogP':>6} {'HBA':>4} {'HBD':>4} {'QED':>6} {'Lip':>4}")
print("-" * 100)
for _, row in druglike.head(20).iterrows():
    smi_display = row['SMILES'][:58]
    print(f"{smi_display:<60} {row['MW']:>7} {row['LogP']:>6} {row['HBA']:>4} {row['HBD']:>4} {row['QED']:>6} {'Yes' if row['Lipinski'] else 'No':>4}")

# 保存结果
results_df.to_csv('generated_molecules_full.csv', index=False)
druglike.to_csv('generated_druglike_molecules.csv', index=False)
print(f"\n全部结果已保存到: generated_molecules_full.csv")
print(f"类药分子已保存到: generated_druglike_molecules.csv")
