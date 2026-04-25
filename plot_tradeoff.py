import matplotlib.pyplot as plt

# ==========================================
# 1. 实验数据配置 (填入你的结果)
# ==========================================
# 格式: '实验名称': (平均每轮耗时秒数, 最佳准确率%)
# ⚠️ 注意：以下带 'Mock' 注释的是我为你捏造的占位数据，仅用于展示图表格式。
# 等你的实验跑完后，请替换为真实的耗时和准确率。

results = {
    'CNN (10%)':  {'time': 18.42,  'acc': 70.95, 'type': 'CNN'},  # Mock
    'CNN (20%)':  {'time': 24.53,  'acc': 79.89, 'type': 'CNN'},  # Mock
    'CNN (100%)': {'time': 61.30, 'acc': 90.41, 'type': 'CNN'},  # 真实数据 (请核对时间)
    
    'ViT (10%)':  {'time': 28.05,  'acc': 56.37, 'type': 'ViT'},  # Mock
    'ViT (20%)':  {'time': 41.95,  'acc': 63.31, 'type': 'ViT'},  # Mock
    'ViT (100%)': {'time': 152.82, 'acc': 80.71, 'type': 'ViT'}   # 真实数据 (请核对时间)
}

# 样式配置 (严格按照要求：CNN蓝色，ViT橙色)
colors = {'CNN': '#1f77b4', 'ViT': '#ff7f0e'}
markers = {'CNN': 'o', 'ViT': 's'}  # CNN用圆点，ViT用方块，更容易区分

# ==========================================
# 2. 创建画布
# ==========================================
plt.figure(figsize=(10, 6))

# ==========================================
# 3. 绘制散点并添加文字标注
# ==========================================
# 用于记录是否已经在图例中添加过标签，避免重复
added_to_legend = {'CNN': False, 'ViT': False}

for name, data in results.items():
    t = data['time']
    acc = data['acc']
    m_type = data['type']
    
    # 绘制散点
    if not added_to_legend[m_type]:
        plt.scatter(t, acc, color=colors[m_type], marker=markers[m_type], 
                    s=120, label=f'Pure {m_type}', zorder=5, alpha=0.9)
        added_to_legend[m_type] = True
    else:
        plt.scatter(t, acc, color=colors[m_type], marker=markers[m_type], 
                    s=120, zorder=5, alpha=0.9)
        
    # 在每个点旁边标注实验名称
    # xytext=(x偏移量, y偏移量) 控制文字的位置，防止挡住圆点
    plt.annotate(name, 
                 xy=(t, acc), 
                 xytext=(8, -5), textcoords='offset points',
                 fontsize=10, color='#333333', fontweight='bold')

# ==========================================
# 4. 图表排版与美化 (论文级质感)
# ==========================================
plt.title('Speed-Accuracy Tradeoff Comparison', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Average Training Time per Epoch (Seconds)', fontsize=12)
plt.ylabel('Best Validation Accuracy (%)', fontsize=12)

# 添加网格线，方便人眼对齐坐标
plt.grid(True, linestyle='--', alpha=0.6, zorder=0)

# 图例放在右下角
plt.legend(fontsize=12, loc='lower right', frameon=True, shadow=True)

# 动态调整坐标轴范围，让数据点不要紧贴边缘
plt.xlim(0, max(d['time'] for d in results.values()) * 1.15)
plt.ylim(min(d['acc'] for d in results.values()) * 0.9, 
         max(d['acc'] for d in results.values()) * 1.05)

# ==========================================
# 5. 保存输出
# ==========================================
plt.tight_layout()
save_path = 'speed_accuracy_tradeoff.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"大功告成！权衡图已保存至: {save_path}")