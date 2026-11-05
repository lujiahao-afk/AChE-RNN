import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from rdkit import Chem
import random
import os

# ==================== 1. 读取数据 ====================
print("=" * 50)
print("步骤1: 读取数据")
print("=" * 50)

# 从CSV读取SMILES
df = pd.read_csv('ache_inhibitors_cleaned.csv')
smiles_list = df['canonical_smiles'].dropna().tolist()

# 用RDKit验证并标准化
valid_smiles = []
for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        canonical = Chem.MolToSmiles(mol)
        valid_smiles.append(canonical)

print(f"原始SMILES数: {len(smiles_list)}")
print(f"有效SMILES数: {len(valid_smiles)}")
print(f"示例SMILES: {valid_smiles[0]}")

# ==================== 2. 构建词汇表 ====================
print("\n" + "=" * 50)
print("步骤2: 构建词汇表")
print("=" * 50)

# 收集所有独特字符
all_chars = set()
for smi in valid_smiles:
    all_chars.update(smi)

# 特殊标记: <PAD>=0, <START>=1, <END>=2
vocab = ['<PAD>', '<START>', '<END>'] + sorted(list(all_chars))
char_to_idx = {c: i for i, c in enumerate(vocab)}
idx_to_char = {i: c for i, c in enumerate(vocab)}
vocab_size = len(vocab)

print(f"词汇表大小: {vocab_size}")
print(f"字符集: {''.join(sorted(all_chars))}")

# ==================== 3. 编码SMILES为序列 ====================
print("\n" + "=" * 50)
print("步骤3: 编码SMILES序列")
print("=" * 50)

MAX_LEN = 100  # 最大序列长度

def encode_smiles(smi):
    """将SMILES编码为整数序列, 前面加<START>, 后面加<END>"""
    encoded = [char_to_idx['<START>']]
    for c in smi:
        if c in char_to_idx:
            encoded.append(char_to_idx[c])
    encoded.append(char_to_idx['<END>'])
    return encoded[:MAX_LEN]  # 截断

def decode_sequence(indices):
    """将整数序列解码回SMILES"""
    chars = []
    for idx in indices:
        c = idx_to_char.get(idx, '')
        if c == '<END>':
            break
        if c not in ['<PAD>', '<START>']:
            chars.append(c)
    return ''.join(chars)

# 编码所有有效SMILES
encoded_data = [encode_smiles(smi) for smi in valid_smiles]
print(f"编码完成, 示例编码: {encoded_data[0]}")
print(f"解码验证: {decode_sequence(encoded_data[0])}")

# ==================== 4. 创建Dataset ====================
print("\n" + "=" * 50)
print("步骤4: 创建数据集")
print("=" * 50)

class SmilesDataset(Dataset):
    def __init__(self, encoded_list):
        self.data = encoded_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        # 输入: 前n-1个字符, 目标: 后n-1个字符(预测下一个)
        x = seq[:-1]
        y = seq[1:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def collate_fn(batch):
    """动态填充到批次内最大长度"""
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)

    padded_xs = torch.zeros(len(xs), max_len, dtype=torch.long)
    padded_ys = torch.zeros(len(ys), max_len, dtype=torch.long)

    for i, (x, y) in enumerate(zip(xs, ys)):
        padded_xs[i, :len(x)] = x
        padded_ys[i, :len(y)] = y

    return padded_xs, padded_ys

# 划分训练集和验证集 (90%训练, 10%验证)
random.seed(42)
random.shuffle(encoded_data)
split = int(len(encoded_data) * 0.9)
train_data = encoded_data[:split]
val_data = encoded_data[split:]

train_dataset = SmilesDataset(train_data)
val_dataset = SmilesDataset(val_data)

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f"训练集: {len(train_dataset)} 条")
print(f"验证集: {len(val_dataset)} 条")

# ==================== 5. 定义RNN模型 ====================
print("\n" + "=" * 50)
print("步骤5: 定义RNN模型")
print("=" * 50)

class SmilesRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super(SmilesRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        embed = self.dropout(self.embedding(x))
        output, hidden = self.lstm(embed, hidden)
        logits = self.fc(output)
        return logits, hidden

# 超参数
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
LEARNING_RATE = 0.003
EPOCHS = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = SmilesRNN(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"模型总参数量: {total_params:,}")
print(model)

# ==================== 6. 训练模型 ====================
print("\n" + "=" * 50)
print("步骤6: 开始训练")
print("=" * 50)

criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

best_val_loss = float('inf')

for epoch in range(EPOCHS):
    # ---- 训练阶段 ----
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits, _ = model(batch_x)

        # reshape: (batch, seq_len, vocab) -> (batch*seq_len, vocab)
        loss = criterion(logits.reshape(-1, vocab_size), batch_y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()

        # 计算准确率(忽略PAD)
        preds = logits.argmax(dim=-1)
        mask = batch_y != 0
        train_correct += ((preds == batch_y) & mask).sum().item()
        train_total += mask.sum().item()

    avg_train_loss = train_loss / len(train_loader)
    train_acc = train_correct / train_total * 100

    # ---- 验证阶段 ----
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits, _ = model(batch_x)
            loss = criterion(logits.reshape(-1, vocab_size), batch_y.reshape(-1))
            val_loss += loss.item()

            preds = logits.argmax(dim=-1)
            mask = batch_y != 0
            val_correct += ((preds == batch_y) & mask).sum().item()
            val_total += mask.sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total * 100

    scheduler.step(avg_val_loss)

    # 保存最优模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab': vocab,
            'char_to_idx': char_to_idx,
            'idx_to_char': idx_to_char,
            'hyperparams': {
                'embed_dim': EMBED_DIM,
                'hidden_dim': HIDDEN_DIM,
                'num_layers': NUM_LAYERS,
                'vocab_size': vocab_size,
                'max_len': MAX_LEN
            }
        }, 'best_rnn_model.pth')

    # 每5轮打印一次
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.1f}%")

print(f"\n训练完成! 最优验证损失: {best_val_loss:.4f}")
print("模型已保存到: best_rnn_model.pth")

# ==================== 7. 生成新分子 ====================
print("\n" + "=" * 50)
print("步骤7: 生成新分子")
print("=" * 50)

def generate_smiles(model, start_char='C', temperature=0.8, max_len=80):
    """用训练好的模型生成SMILES"""
    model.eval()

    # 起始字符
    input_idx = torch.tensor([[char_to_idx['<START>'], char_to_idx.get(start_char, char_to_idx['C'])]], dtype=torch.long).to(device)
    generated = [char_to_idx['<START>'], char_to_idx.get(start_char, char_to_idx['C'])]
    hidden = None

    with torch.no_grad():
        for _ in range(max_len):
            logits, hidden = model(input_idx, hidden)

            # 取最后一个时间步的输出
            last_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(last_logits, dim=-1)

            # 采样下一个字符
            next_idx = torch.multinomial(probs, 1).item()

            if next_idx == char_to_idx['<END>']:
                break

            generated.append(next_idx)
            input_idx = torch.tensor([[next_idx]], dtype=torch.long).to(device)

    return decode_sequence(generated)

# 生成多个分子
print("正在生成新分子...\n")
generated_smiles = []
valid_count = 0

for i in range(100):  # 尝试生成100个
    smi = generate_smiles(model, temperature=0.8)
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        canonical = Chem.MolToSmiles(mol)
        if canonical not in valid_smiles:  # 新分子
            generated_smiles.append(canonical)
            valid_count += 1

print(f"生成100次, 有效分子: {valid_count} 个")
print(f"有效性: {valid_count}%\n")

print("生成的新分子示例(前20个):")
for i, smi in enumerate(generated_smiles[:20]):
    print(f"  {i+1}. {smi}")

# 保存生成的分子
if generated_smiles:
    gen_df = pd.DataFrame({'generated_smiles': generated_smiles})
    gen_df.to_csv('generated_molecules.csv', index=False)
    print(f"\n生成的分子已保存到: generated_molecules.csv")
