import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams['figure.figsize'] = (16, 10)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# ---- 1. 基础RNN训练Loss ----
ax = axes[0, 0]
# 从训练输出中提取的完整数据
train_loss = [1.9391, 0.7079, 0.4893, 0.3926, 0.3368, 0.3050, 0.2797, 0.2633, 0.2522, 0.2423, 0.2348]
val_loss   = [1.3521, 0.6595, 0.4688, 0.3977, 0.3580, 0.3403, 0.3288, 0.3175, 0.3201, 0.3148, 0.3127]
# 对应的epoch (每5轮记录一次: 1,5,10,15,20,25,30,35,40,45,50)
rnn_epochs = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

ax.plot(rnn_epochs, train_loss, 'b-o', label='Train Loss', linewidth=2, markersize=6)
ax.plot(rnn_epochs, val_loss, 'r-s', label='Val Loss', linewidth=2, markersize=6)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Basic RNN Training Loss', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# ---- 2. RNN准确率 ----
ax = axes[0, 1]
train_acc = [45.1, 76.7, 83.5, 86.3, 88.2, 89.1, 89.9, 90.3, 90.6, 90.9, 91.2]
val_acc   = [58.8, 78.9, 84.6, 86.9, 87.9, 88.7, 89.2, 89.4, 89.3, 89.7, 89.7]

ax.plot(rnn_epochs, train_acc, 'g-o', label='Train Acc', linewidth=2, markersize=6)
ax.plot(rnn_epochs, val_acc, 'm-s', label='Val Acc', linewidth=2, markersize=6)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Basic RNN Training Accuracy', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# ---- 3. RL训练趋势(分数和QED) ----
ax = axes[0, 2]
rl_epochs = [1, 6, 11, 16, 21, 26, 31, 36, 41, 46]
rl_scores = [0.4242, 0.4636, 0.4314, 0.3644, 0.3483, 0.4408, 0.3934, 0.3121, 0.3259, 0.3661]
rl_qed    = [0.4077, 0.4911, 0.4633, 0.3963, 0.3859, 0.5259, 0.4838, 0.5151, 0.4705, 0.5379]

ax.plot(rl_epochs, rl_scores, 'b-o', label='Mean Score', linewidth=2, markersize=6)
ax.plot(rl_epochs, rl_qed, 'r-s', label='Mean QED', linewidth=2, markersize=6)
ax.set_xlabel('RL Epoch', fontsize=12)
ax.set_ylabel('Score / QED', fontsize=12)
ax.set_title('RL Training Progress', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# ---- 4. RL合规率趋势 ----
ax = axes[1, 0]
rl_valid  = [71.9, 71.9, 70.3, 62.5, 62.5, 67.2, 64.1, 46.9, 53.1, 56.2]
rl_novel  = [52.2, 56.5, 48.9, 52.5, 42.5, 48.8, 39.0, 60.0, 44.1, 36.1]
rl_lip    = [56.5, 65.2, 57.8, 50.0, 40.0, 72.1, 58.5, 66.7, 58.8, 72.2]

x_pos = np.arange(len(rl_epochs))
width = 0.25
bars1 = ax.bar(x_pos - width, rl_valid, width, label='Valid%', color='#4CAF50', alpha=0.85)
bars2 = ax.bar(x_pos, rl_novel, width, label='Novel%', color='#2196F3', alpha=0.85)
bars3 = ax.bar(x_pos + width, rl_lip, width, label='Lipinski%', color='#FF9800', alpha=0.85)

ax.set_xticks(x_pos)
ax.set_xticklabels(rl_epochs)
ax.set_xlabel('RL Epoch', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('RL Compliance Rates', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# ---- 5. 生成分子QED分布 ----
ax = axes[1, 1]
all_df = pd.read_csv('rl_generated_all.csv')
druglike_df = pd.read_csv('rl_generated_druglike.csv')

ax.hist(all_df['QED'], bins=30, alpha=0.5, color='gray', label='All Generated', density=True, edgecolor='white')
ax.hist(druglike_df['QED'], bins=20, alpha=0.7, color='#2196F3', label='Druglike+Novel', density=True, edgecolor='white')
ax.set_xlabel('QED Score', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('QED Distribution', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# ---- 6. MW vs LogP散点图 ----
ax = axes[1, 2]
scatter = ax.scatter(druglike_df['MW'], druglike_df['LogP'],
                     c=druglike_df['QED'], cmap='RdYlGn',
                     s=40, alpha=0.8, edgecolors='gray', linewidth=0.5)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('QED', fontsize=11)

# Lipinski边界线
ax.axhline(y=5, color='red', linestyle='--', alpha=0.6, linewidth=1.5, label='LogP=5')
ax.axvline(x=500, color='red', linestyle='--', alpha=0.6, linewidth=1.5, label='MW=500')
ax.set_xlabel('Molecular Weight', fontsize=12)
ax.set_ylabel('LogP', fontsize=12)
ax.set_title('MW vs LogP (colored by QED)', fontsize=14)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout(pad=2.0)
plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("图表已保存到: training_results.png")
