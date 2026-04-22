"""
train.py - 训练脚本（支持学习率预热）
"""

import argparse
import time
import os
import numpy as np          # 新增
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

from utils.dataset import get_cifar10_loaders
from models.cnn import CNN
from models.vit import ViT
from config import Config


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, f1


def train_model(model, train_loader, test_loader, cfg, warmup_epochs=0):
    """
    完整的训练流程，支持学习率预热，并自动记录日志到 CSV
    """
    import csv
    import os
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # 创建 logs 目录
    os.makedirs("logs", exist_ok=True)
    
    # 生成日志文件名
    log_filename = f"logs/{cfg.model_name}_ratio{cfg.data_ratio}.csv"
    
    # 如果是第一次运行（文件不存在），写入表头
    file_exists = os.path.isfile(log_filename)
    log_file = open(log_filename, 'a', newline='')
    csv_writer = csv.writer(log_file)
    if not file_exists:
        csv_writer.writerow(['epoch', 'train_loss', 'accuracy', 'macro_f1', 'lr'])
    
    epoch_times = []
    best_acc = 0.0
    best_f1 = 0.0

    # 学习率调度函数（预热 + 余弦退火）
    def adjust_lr(epoch):
        if epoch < warmup_epochs:
            lr = cfg.lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (cfg.epochs - warmup_epochs)
            lr = cfg.lr * 0.5 * (1 + np.cos(np.pi * progress))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    for epoch in range(1, cfg.epochs + 1):
        start_time = time.time()

        if warmup_epochs > 0:
            adjust_lr(epoch - 1)

        train_loss = train_epoch(model, train_loader, optimizer, criterion, cfg.device)
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        
        acc, f1 = evaluate(model, test_loader, cfg.device)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 写入 CSV
        csv_writer.writerow([epoch, f"{train_loss:.6f}", f"{acc:.6f}", f"{f1:.6f}", f"{current_lr:.8f}"])
        log_file.flush()  # 立即写入磁盘，防止中断丢失数据
        
        print(f"Epoch {epoch:3d}/{cfg.epochs} | LR: {current_lr:.6f} | Loss: {train_loss:.4f} | Time: {epoch_time:.2f}s | Acc: {acc:.4f} | F1: {f1:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'acc': acc,
                'f1': f1,
            }, cfg.save_path)
            print(f"  -> Best model saved (acc={acc:.4f})")
    
    log_file.close()
    avg_time = sum(epoch_times) / len(epoch_times)
    return avg_time, best_acc, best_f1


def main():
    cfg = Config()
    
    # ========== 命令行参数 ==========
    parser = argparse.ArgumentParser(description='Train CNN or ViT on CIFAR-10')
    parser.add_argument('--model', type=str, default=cfg.model_name, choices=['cnn', 'vit'])
    parser.add_argument('--data_ratio', type=float, default=cfg.data_ratio)
    parser.add_argument('--batch_size', type=int, default=cfg.batch_size)
    parser.add_argument('--epochs', type=int, default=cfg.epochs)
    parser.add_argument('--lr', type=float, default=cfg.lr)
    args = parser.parse_args()
    
    # 应用配置
    cfg.model_name = args.model
    cfg.data_ratio = args.data_ratio

    # ViT 专用参数覆盖
    if cfg.model_name == 'vit':
        cfg.epochs = cfg.vit_epochs
        cfg.lr = cfg.vit_lr
        cfg.batch_size = cfg.vit_batch_size
        warmup_epochs = cfg.vit_warmup_epochs
    else:
        cfg.epochs = args.epochs
        cfg.lr = args.lr
        cfg.batch_size = args.batch_size
        warmup_epochs = 0  # CNN 不使用预热
    
    # ========== 打印配置 ==========
    print("=" * 60)
    print("实验配置")
    print("=" * 60)
    for key, value in vars(cfg).items():
        if not key.startswith("__") and not callable(getattr(cfg, key)):
            print(f"{key:20} = {value}")
    print(f"{'warmup_epochs':20} = {warmup_epochs}")
    print(f"{'save_path':20} = {cfg.save_path}")
    print("=" * 60)
    
    # ========== 加载数据 ==========
    print("\n[1/4] 加载数据...")
    train_loader, test_loader = get_cifar10_loaders(
        data_ratio=cfg.data_ratio,
        batch_size=cfg.batch_size,
        augment=True,
        num_workers=cfg.num_workers
    )
    print(f"训练集批次数: {len(train_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    
    # ========== 创建模型 ==========
    print(f"\n[2/4] 创建 {cfg.model_name.upper()} 模型...")
    
    if cfg.model_name == 'cnn':
        model = CNN(
            num_classes=10,
            channels=cfg.cnn_channels,
            dropout=cfg.cnn_dropout
        )
    elif cfg.model_name == 'vit':
        model = ViT(
            img_size=cfg.vit_img_size,
            patch_size=cfg.vit_patch_size,
            in_channels=cfg.vit_in_channels,
            num_classes=10,
            embed_dim=cfg.vit_embed_dim,
            depth=cfg.vit_depth,
            num_heads=cfg.vit_num_heads,
            mlp_ratio=cfg.vit_mlp_ratio,
            dropout=cfg.vit_dropout,
            drop_path_ratio=cfg.vit_drop_path_ratio
        )
    
    model = model.to(cfg.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    if hasattr(model, 'count_parameters'):
        print(f"模型参数量: {model.count_parameters():,}")
    
    # ========== 训练 ==========
    print(f"\n[3/4] 开始训练...")
    avg_time, best_acc, best_f1 = train_model(model, train_loader, test_loader, cfg, warmup_epochs)
    
    # ========== 输出结果 ==========
    print("\n[4/4] 训练完成！")
    print("=" * 60)
    print("最终结果")
    print("=" * 60)
    print(f"平均每轮训练时间: {avg_time:.2f} 秒")
    print(f"最佳测试准确率: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"最佳测试 Macro-F1: {best_f1:.4f}")
    print(f"模型已保存至: {cfg.save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()