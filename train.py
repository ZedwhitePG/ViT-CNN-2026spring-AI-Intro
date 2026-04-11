"""
train.py

1. 支持 CNN 和 ViT 两种模型
2. 支持不同数据比例（用于小数据退化实验）
3. 记录训练时间、准确率、Macro-F1
4. 保存最佳模型

使用方法：
    python train.py --model cnn --data_ratio 1.0 --epochs 20
    python train.py --model vit --data_ratio 0.2 --epochs 10
"""

import argparse      # 用于解析命令行参数
import time          # 用于计时
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR  # 余弦退火学习率调度器
from sklearn.metrics import accuracy_score, f1_score   # 计算准确率和 F1 分数

# 这是Capoo写的dataset.py
from utils.dataset import get_cifar10_loaders
# 导入Zedwhite写的CNN和ViT模型
from models.cnn import CNN          
from models.vit import ViT          


def train_epoch(model, loader, optimizer, criterion, device):
    """
    训练一个 epoch（完整遍历一次训练集）
    
    参数：
        model: 神经网络模型
        loader: 训练数据的 DataLoader
        optimizer: 优化器（如 AdamW）
        criterion: 损失函数（如交叉熵）
        device: 运行设备（cuda 或 cpu）
    
    返回：
        epoch_loss: 当前 epoch 的平均损失值
    """
    model.train()  # 设置为训练模式（启用 dropout、batch norm 等）
    running_loss = 0.0  # 累计损失
    
    # 遍历 DataLoader，每次取出一个 batch
    for images, labels in loader:
        # 将数据移动到指定设备（GPU 或 CPU）
        images, labels = images.to(device), labels.to(device)
        
        # 梯度清零（因为 PyTorch 默认会累积梯度）
        optimizer.zero_grad()
        
        # 前向传播：输入图像，得到预测结果
        outputs = model(images)
        
        # 计算损失：比较预测结果和真实标签
        loss = criterion(outputs, labels)
        
        # 反向传播：计算梯度
        loss.backward()
        
        # 更新模型参数
        optimizer.step()
        
        # 累计损失（乘以 batch_size 是因为后面要除以总样本数）
        running_loss += loss.item() * images.size(0)
    
    # 计算平均损失（总损失 / 总样本数）
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, device):
    """
    评估模型在测试集上的表现
    
    参数：
        model: 神经网络模型
        loader: 测试数据的 DataLoader
        device: 运行设备
    
    返回：
        acc: 准确率（Accuracy），即预测正确的比例
        f1: Macro-F1 分数，每个类别 F1 的平均值
    """
    model.eval()  # 设置为评估模式（禁用 dropout、batch norm 使用全局统计）
    all_preds = []   # 存储所有预测结果
    all_labels = []  # 存储所有真实标签
    
    # torch.no_grad() 上下文管理器：禁用梯度计算，节省内存和计算
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 获取预测类别：torch.max 返回 (最大值, 索引)
            # dim=1 表示在类别维度上取最大值
            _, preds = torch.max(outputs, 1)
            
            # 将预测结果和标签从 GPU 移到 CPU，并转换为 numpy 列表
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算准确率：预测正确的数量 / 总数量
    acc = accuracy_score(all_labels, all_preds)
    
    # 计算 Macro-F1：每个类别 F1 分数的算术平均
    # average='macro' 表示对每个类别独立计算 F1，然后取平均
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return acc, f1


def train_model(model, train_loader, test_loader, epochs, lr, device, save_path=None):
    """
    完整的训练流程
    
    参数：
        model: 神经网络模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        epochs: 训练轮数
        lr: 学习率
        device: 设备
        save_path: 最佳模型保存路径（如果为 None 则不保存）
    
    返回：
        avg_epoch_time: 每个 epoch 的平均训练时间（秒）
        best_acc: 最佳测试准确率
        best_f1: 最佳测试 Macro-F1
    """
    # 定义损失函数：交叉熵损失，适用于多分类问题
    criterion = nn.CrossEntropyLoss()
    
    # 定义优化器：AdamW 是 Adam 的改进版，能更好地处理权重衰减
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # 学习率调度器：余弦退火，每个 epoch 后调整学习率
    # T_max=epochs 表示在 epochs 轮内完成一个余弦周期
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    epoch_times = []  # 记录每个 epoch 的训练时间
    best_acc = 0.0    # 记录最佳准确率
    best_f1 = 0.0     # 记录最佳 F1 分数
    
    # 主训练循环
    for epoch in range(1, epochs + 1):
        # ========== 训练阶段 ==========
        start_time = time.time()  # 记录开始时间
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        epoch_time = time.time() - start_time  # 计算耗时
        epoch_times.append(epoch_time)
        
        # ========== 评估阶段 ==========
        acc, f1 = evaluate(model, test_loader, device)
        
        # 打印当前 epoch 的结果
        print(f"Epoch {epoch:2d}/{epochs} | Loss: {train_loss:.4f} | Time: {epoch_time:.2f}s | Acc: {acc:.4f} | F1: {f1:.4f}")
        
        # ========== 保存最佳模型 ==========
        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'acc': acc,
                    'f1': f1,
                }, save_path)
                print(f"  -> Best model saved (acc={acc:.4f})")
        
        # 更新学习率
        scheduler.step()
    
    # 计算平均每个 epoch 的训练时间
    avg_time = sum(epoch_times) / len(epoch_times)
    return avg_time, best_acc, best_f1


def main():
    """
    主函数：解析参数、加载数据、创建模型、训练、输出结果
    """
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description='Train CNN or ViT on CIFAR-10')
    
    # 数据相关参数
    parser.add_argument('--data_ratio', type=float, default=1.0,
                        help='训练集使用比例 (0,1]，用于小数据退化实验 (default: 1.0)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小，即每次训练使用的图片数量 (default: 64)')
    
    # 模型相关参数
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'vit'],
                        help='模型类型：cnn 或 vit (default: cnn)')
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数，完整遍历训练集的次数 (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率，控制参数更新的步长 (default: 0.001)')
    
    # 其他参数
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='运行设备：cuda（GPU）或 cpu (default: 自动选择)')
    parser.add_argument('--save_path', type=str, default='best_model.pth',
                        help='最佳模型保存路径 (default: best_model.pth)')
    
    # 解析参数
    args = parser.parse_args()
    
    # ========== 打印配置信息 ==========
    print("=" * 60)
    print("训练配置")
    print("=" * 60)
    print(f"设备: {args.device}")
    print(f"模型: {args.model}")
    print(f"数据比例: {args.data_ratio * 100:.0f}%")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"模型保存路径: {args.save_path}")
    print("=" * 60)
    
    # ========== 加载数据 ==========
    print("\n[1/4] 加载数据...")
    train_loader, test_loader = get_cifar10_loaders(
        data_ratio=args.data_ratio,
        batch_size=args.batch_size,
        augment=True,      # 训练集使用随机水平翻转等数据增强
        num_workers=2      # 使用 2 个子进程加载数据，加快速度
    )
    print(f"训练集批次数: {len(train_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    
    # ========== 创建模型 ==========
    print(f"\n[2/4] 创建 {args.model.upper()} 模型...")
    
    if args.model == 'cnn':
        # 使用同伴编写的 CNN 模型
        model = CNN(
            num_classes=10,           # CIFAR-10 有 10 个类别
            channels=[64, 128, 256, 512],  # 四层卷积的通道数
            dropout=0.1               # dropout 概率
        )
    elif args.model == 'vit':
        # 使用同伴编写的 ViT 模型
        model = ViT(
            img_size=64,              # 输入图像尺寸（CIFAR-10 已缩放到 64x64）
            patch_size=8,             # patch 大小，64/8=8，共 8x8=64 个 patch
            in_channels=3,            # RGB 三通道
            num_classes=10,           # CIFAR-10 有 10 个类别
            embed_dim=256,            # 嵌入维度
            depth=6,                  # Transformer 层数
            num_heads=8,              # 多头注意力的头数
            mlp_ratio=4.0,            # MLP 隐层扩展比例
            dropout=0.1,              # dropout 概率
            drop_path_ratio=0.1       # 随机深度丢弃比例
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # 将模型移动到指定设备（GPU 或 CPU）
    model = model.to(args.device)
    
    # 打印模型参数量（用于了解模型大小）
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 如果模型有 count_parameters 方法，也调用它（同伴的 CNN 和 ViT 都有这个方法）
    if hasattr(model, 'count_parameters'):
        print(f"模型参数量（count_parameters）: {model.count_parameters():,}")
    
    # 打印模型结构摘要（可选，用于调试）
    print(f"\n模型结构:\n{model}")
    
    # ========== 训练 ==========
    print(f"\n[3/4] 开始训练...")
    avg_time, best_acc, best_f1 = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        save_path=args.save_path
    )
    
    # ========== 输出最终结果 ==========
    print("\n[4/4] 训练完成！")
    print("=" * 60)
    print("最终结果")
    print("=" * 60)
    print(f"平均每轮训练时间: {avg_time:.2f} 秒")
    print(f"最佳测试准确率: {best_acc:.4f} ({best_acc * 100:.2f}%)")
    print(f"最佳测试 Macro-F1: {best_f1:.4f}")
    print(f"模型已保存至: {args.save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()