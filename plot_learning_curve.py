import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. 配置日志路径与样式
# ==========================================
csv_files = {
    'Pure CNN': 'logs/cnn_ratio1.0.csv',
    'Pure ViT': 'logs/vit_ratio1.0.csv'
}

styles = {
    'Pure CNN': {'color': '#1f77b4', 'linestyle': '-'},  # 经典科技蓝
    'Pure ViT': {'color': '#ff7f0e', 'linestyle': '-'}   # 鲜明活力橙
}

# 平滑系数 (0 到 1 之间。值越小，曲线越平滑；值越大，越贴近原始抖动数据)
SMOOTHING_ALPHA = 0.3 

# ==========================================
# 2. 创建画布
# ==========================================
plt.figure(figsize=(12, 5))

# ==========================================
# 3. 循环读取数据并绘制“高阶”曲线
# ==========================================
for label, filename in csv_files.items():
    if not os.path.exists(filename):
        print(f"⚠️ 找不到文件: {filename}，请检查路径！")
        continue

    df = pd.read_csv(filename)
    
    try:
        epochs = df['epoch']
        loss = df['train_loss']  
        acc = df['accuracy']     
    except KeyError as e:
        print(f"❌ 读取 {filename} 失败，找不到对应的列名：{e}")
        break

    color = styles[label]['color']
    linestyle = styles[label]['linestyle']

    # --- 🌟 核心升级 1：计算平滑曲线 ---
    # 使用 Exponential Moving Average (EMA) 让曲线更顺滑，突出宏观趋势
    loss_smoothed = loss.ewm(alpha=SMOOTHING_ALPHA).mean()
    acc_smoothed = acc.ewm(alpha=SMOOTHING_ALPHA).mean()

    # ================= 左侧: Loss =================
    plt.subplot(1, 2, 1)
    # 1. 画出原始的、抖动的真实数据（变细、变半透明作为背景）
    plt.plot(epochs, loss, color=color, alpha=0.2, linewidth=1.5)
    # 2. 在上面覆盖粗实线的平滑数据
    plt.plot(epochs, loss_smoothed, label=f'{label} (Smoothed)', color=color, linestyle=linestyle, linewidth=2.5)

    # ================= 右侧: Accuracy =================
    plt.subplot(1, 2, 2)
    # 1. 真实数据背景
    plt.plot(epochs, acc, color=color, alpha=0.2, linewidth=1.5)
    # 2. 平滑数据主线
    plt.plot(epochs, acc_smoothed, label=f'{label} (Smoothed)', color=color, linestyle=linestyle, linewidth=2.5)

    # --- 🌟 核心升级 2：自动寻找并标出“最佳性能点” ---
    # 找到原始 Accuracy 最大的那一行的索引
    best_idx = acc.idxmax()
    best_epoch = epochs[best_idx]
    best_acc = acc[best_idx]

    # 在最高点打上一个大号的五角星星星 (*)
    plt.scatter(best_epoch, best_acc, color=color, s=150, zorder=5, marker='*')
    
    # 画箭头并标注具体数值
    plt.annotate(f'{label} Best:\n{best_acc:.4f}', 
                 xy=(best_epoch, best_acc), 
                 xytext=(-30, 25), textcoords='offset points', # 让文字悬浮在星星左上方
                 arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                 fontsize=10, color=color, fontweight='bold')


# ==========================================
# 4. 图表排版与美化 
# ==========================================
# 优化 Loss 图
plt.subplot(1, 2, 1)
plt.title('Training Loss (Smoothed)', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)

# 【可选】如果你发现早期的 Loss 极其巨大，导致后续曲线被压扁，
# 可以取消下面这行的注释，强行限制 Y 轴的显示范围 (比如只显示 0 到 2.5)：
# plt.ylim(0, 2.5) 

# 优化 Accuracy 图
plt.subplot(1, 2, 2)
plt.title('Accuracy & Best Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
# 为了不让图例挡住右上角的巅峰曲线，我们把图例移到右下角
plt.legend(fontsize=11, loc='lower right') 

# ==========================================
# 5. 输出与保存
# ==========================================
plt.tight_layout() 
plt.savefig('learning_curve_pro.png', dpi=300, bbox_inches='tight') 
print("大功告成！带【半透明底噪 + 均线平滑 + 最佳点标注】的高级曲线图已生成。")