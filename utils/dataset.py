import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from config import TrainConfig


def build_transforms(augment=True, strong_aug=False, config=TrainConfig):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    train_ops = []
    if augment:
        if config.use_random_crop:
            train_ops.append(transforms.RandomCrop(config.img_size, padding=config.random_crop_padding))
        train_ops.append(transforms.RandomHorizontalFlip())
        if config.use_color_jitter:
            brightness, contrast, saturation, hue = config.color_jitter
            train_ops.append(transforms.ColorJitter(brightness, contrast, saturation, hue))

    train_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if augment and config.use_random_erasing:
        train_ops.append(transforms.RandomErasing(p=config.random_erasing_p))

    eval_ops = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def get_cifar10_loaders(
    data_ratio=TrainConfig.data_ratio,
    batch_size=TrainConfig.batch_size,
    augment=TrainConfig.use_augmentation,
    strong_aug=TrainConfig.use_strong_aug,
    num_workers=TrainConfig.num_workers,
    seed=TrainConfig.seed,
    train_dir=TrainConfig.train_dir,
    test_dir=TrainConfig.test_dir,
    max_train_samples=None,
    max_test_samples=None,
):
    """
    Return CIFAR-10 64x64 train/test loaders with optional ratio sampling.

    Mixup and CutMix are intentionally not implemented in this project stage.
    Random augmentation is applied only to the training set.
    """
    if not (0 < data_ratio <= 1.0):
        raise ValueError(f"data_ratio must be in (0, 1], got {data_ratio}")

    train_transform, test_transform = build_transforms(augment=augment, strong_aug=strong_aug)
    full_train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = ImageFolder(root=test_dir, transform=test_transform)

    rng = np.random.default_rng(seed)

    if data_ratio < 1.0 or max_train_samples is not None:
        num_samples = int(len(full_train_dataset) * data_ratio)
        if max_train_samples is not None:
            num_samples = min(num_samples, max_train_samples)
        num_samples = max(1, num_samples)
        indices = rng.permutation(len(full_train_dataset))[:num_samples]
        train_dataset = Subset(full_train_dataset, indices.tolist())
    else:
        train_dataset = full_train_dataset

    if max_test_samples is not None:
        num_samples = max(1, min(len(test_dataset), max_test_samples))
        indices = rng.permutation(len(test_dataset))[:num_samples]
        test_dataset = Subset(test_dataset, indices.tolist())

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_cifar10_loaders(data_ratio=0.1, batch_size=4, strong_aug=True, num_workers=0)
    print("train batches:", len(train_loader))
    print("test batches:", len(test_loader))
    for images, labels in train_loader:
        print("images shape:", images.shape)
        print("labels:", labels)
        break
