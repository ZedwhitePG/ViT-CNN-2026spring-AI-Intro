import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np

def get_cifar10_loaders(data_ratio=1.0, batch_size=64, augment=True, num_workers=0):
    """
    函数功能：返回 CIFAR-10 (64x64) 的 DataLoader，支持训练集按比例采样。
    Args:
        data_ratio (float): 训练集使用的比例 (0,1]，以满足小数据退化实验的需求
                           （至少要测试1.0, 0.2, 0.1三种比例）
        batch_size (int): 批次大小
        augment (bool): 是否对训练集做数据增强（随机水平翻转）
        num_workers (int): 数据加载使用的进程数，默认为0表示在主进程中加载数据
    Returns:
        train_loader, test_loader
    """

    # 定义均值和标准差（CIFAR-10 标准值）
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    # 训练集变换
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),    # 随机水平翻转
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    # 测试集变换（无需增强）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 数据集路径
    train_path = './data/cifar10-64/train'
    test_path = './data/cifar10-64/test'

    # 加载完整训练集
    full_train_dataset = ImageFolder(root=train_path, transform=train_transform)
    test_dataset = ImageFolder(root=test_path, transform=test_transform)

    # 根据 data_ratio 采样训练集
    if data_ratio < 1.0:
        num_samples = int(len(full_train_dataset) * data_ratio)
        # 随机打乱并选择前 num_samples 个索引 
        indices = np.random.permutation(len(full_train_dataset))[:num_samples]  
        # 创建子集数据集  
        train_dataset = Subset(full_train_dataset, indices)
    else:
        train_dataset = full_train_dataset

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader

# 测试代码
if __name__ == "__main__":
    #在此处设置 data_ratio（采样比例）, batch_size（批次大小）, augment（是否增强）来测试函数
    train_loader, test_loader = get_cifar10_loaders(data_ratio=1.0, batch_size=4, augment=True)
    print("训练集批次数量：", len(train_loader))
    print("测试集批次数量：", len(test_loader))
    for images, labels in train_loader:
        print("images shape:", images.shape)
        print("labels:", labels)
        break