import matplotlib.pyplot as plt

# 提取自你提供的训练日志
epochs = [0.03, 0.05, 0.08, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
losses = [12.74, 8.01, 4.92, 4.46, 4.06, 3.76, 3.74, 3.74, 3.74, 3.74, 3.74]

plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, marker='o', color='#2c3e50', linewidth=2, markersize=6)

# 图表美化
plt.title('Fine-tuning LLaVA on VQA-RAD: Training Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss (Cross Entropy)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(y=3.74, color='r', linestyle=':', label='Convergence Baseline')
plt.legend()

# 保存图片用于报告
plt.savefig('training_loss_curve.png', dpi=300)
plt.show()
print("✅ Loss 曲线图已保存为 training_loss_curve.png")