import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.vit import ViT
from utils.dataset import get_cifar10_loaders

# ==========================================
# 1. 实例化模型与加载权重
# ==========================================
model = ViT(patch_size=4, embed_dim=192)

ckpt_path = 'checkpoints/vit_ratio1.0_best.pth'
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    print("模型权重完美契合，加载成功！正在生成 CIFAR-10 全类别热力图...")
else:
    print(f"找不到权重文件 {ckpt_path}，请检查路径。")
    exit()

model.eval()

# ==========================================
# 2. 获取测试集与环境准备
# ==========================================
_, test_loader = get_cifar10_loaders(data_ratio=1.0, batch_size=1, augment=False)

os.makedirs('attention_maps', exist_ok=True)
target_classes = set()  # 用来记录已经画过哪些类别的图

# ==========================================
# 3. 循环遍历提取并可视化
# ==========================================
for images, labels in test_loader:
    label = labels[0].item()
    
    # 如果这个类别已经画过了，就跳过找下一张
    if label in target_classes:
        continue
        
    target_classes.add(label)
    image = images[0:1] 
    
    # --- 提取注意力 ---
    with torch.no_grad():
        all_attn = model.get_attention_maps(image)

    last_attn = all_attn[-1][0]          
    attn_mean = last_attn.mean(0)        
    cls_attn = attn_mean[0, 1:]          
    
    # patch_size=4, 64/4=16，所以 reshape 为 16x16
    attn_map = cls_attn.reshape(16, 16).numpy()
    attn_resized = cv2.resize(attn_map, (64, 64))

    # --- 处理原图 (反归一化) ---
    img_show = image[0].permute(1, 2, 0).numpy()
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    img_show = std * img_show + mean
    img_show = np.clip(img_show, 0, 1)

    # --- 🌟 OpenCV 高级融合 ---
    attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min())
    attn_uint8 = (attn_norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(attn_uint8, cv2.COLORMAP_JET)

    img_uint8 = (img_show * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # --- 绘制与保存 ---
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_show)
    plt.title(f'Original (Class: {label})')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(overlay_rgb)
    plt.title(f'Attention Map (Class: {label})')
    plt.axis('off')

    save_path = f'attention_maps/class_{label}_sample.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close() 
    
    print(f"[{len(target_classes)}/10] 已保存: {save_path}")
    
    # 集齐 10 张，退出循环
    if len(target_classes) >= 10:
        break

print("大功告成！CIFAR-10 全部 10 个类别的注意力图已生成完毕！")