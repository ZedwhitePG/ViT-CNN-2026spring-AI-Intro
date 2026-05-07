"""
Unified experiment entry point.

Examples:
    python main.py --model vit --data_ratio 1.0 --epochs 100
    python main.py --model cnn --data_ratio 0.2 --epochs 100
    python main.py --model vit --data_ratio 0.1 --epochs 2 --debug
"""

import argparse
import os
import random

import numpy as np
import torch

from config import CnnConfig, TrainConfig, VitConfig
from models.cnn import CNN
from models.vit import ViT
from train import build_optimizer, build_scheduler, train_model
from utils.dataset import get_cifar10_loaders


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def apply_cli_overrides(args):
    TrainConfig.model_type = args.model
    TrainConfig.data_ratio = args.data_ratio
    TrainConfig.num_epochs = args.epochs
    TrainConfig.batch_size = args.batch_size
    TrainConfig.num_workers = args.num_workers
    TrainConfig.seed = args.seed
    TrainConfig.learning_rate = args.lr
    TrainConfig.lr = args.lr
    TrainConfig.weight_decay = args.weight_decay
    TrainConfig.warmup_epochs = args.warmup_epochs
    TrainConfig.min_lr = args.min_lr
    TrainConfig.use_amp = not args.no_amp
    TrainConfig.use_strong_aug = args.strong_aug

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU.")
        TrainConfig.device = "cpu"
    else:
        TrainConfig.device = args.device


def build_model(model_type):
    if model_type == "cnn":
        return CNN(**CnnConfig.to_cnn_kwargs())
    if model_type == "vit":
        return ViT(**VitConfig.to_vit_kwargs())
    raise ValueError(f"Unknown model type: {model_type}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run one CNN/ViT CIFAR-10 experiment.")
    parser.add_argument("--model", choices=["cnn", "vit"], default=TrainConfig.model_type)
    parser.add_argument("--data_ratio", type=float, choices=TrainConfig.data_fractions, default=TrainConfig.data_ratio)
    parser.add_argument("--epochs", type=int, default=TrainConfig.num_epochs)
    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--warmup_epochs", type=int, default=TrainConfig.warmup_epochs)
    parser.add_argument("--min_lr", type=float, default=TrainConfig.min_lr)
    parser.add_argument("--num_workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=TrainConfig.device)
    parser.add_argument("--strong_aug", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        args.epochs = min(args.epochs, 2)
        args.batch_size = min(args.batch_size, 32)
        args.num_workers = 0

    apply_cli_overrides(args)
    set_seed(TrainConfig.seed)

    os.makedirs(TrainConfig.log_dir, exist_ok=True)
    os.makedirs(TrainConfig.checkpoint_dir, exist_ok=True)
    os.makedirs(TrainConfig.figure_dir, exist_ok=True)

    if args.debug:
        experiment_name = f"debug_{TrainConfig.experiment_name(args.model, args.data_ratio)}"
        log_path = os.path.join(TrainConfig.log_dir, f"{experiment_name}.csv")
        best_path = os.path.join(TrainConfig.checkpoint_dir, f"{experiment_name}_best.pth")
        last_path = os.path.join(TrainConfig.checkpoint_dir, f"{experiment_name}_last.pth")
    else:
        log_path = TrainConfig.log_path(args.model, args.data_ratio)
        best_path = TrainConfig.best_checkpoint_path(args.model, args.data_ratio)
        last_path = TrainConfig.last_checkpoint_path(args.model, args.data_ratio)

    print("=" * 72)
    print("Experiment configuration")
    print("=" * 72)
    print(f"model={args.model}")
    print(f"data_ratio={args.data_ratio}")
    print(f"epochs={args.epochs}")
    print(f"batch_size={args.batch_size}")
    print(f"lr={TrainConfig.learning_rate}")
    print(f"weight_decay={TrainConfig.weight_decay}")
    print(f"device={TrainConfig.device}")
    print(f"use_amp={TrainConfig.use_amp}")
    print(f"use_strong_aug={TrainConfig.use_strong_aug}")
    print(f"log_path={log_path}")
    print(f"best_checkpoint={best_path}")
    print("=" * 72)

    train_loader, val_loader = get_cifar10_loaders(
        data_ratio=args.data_ratio,
        batch_size=args.batch_size,
        augment=TrainConfig.use_augmentation,
        strong_aug=args.strong_aug,
        num_workers=args.num_workers,
        seed=args.seed,
        max_train_samples=256 if args.debug else None,
        max_test_samples=512 if args.debug else None,
    )

    model = build_model(args.model)
    optimizer = build_optimizer(model, TrainConfig)
    scheduler = build_scheduler(optimizer, args.epochs, TrainConfig)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total_params={total_params:,}")
    print(f"trainable_params={trainable_params:,}")

    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        device=TrainConfig.device,
        model_type=args.model,
        data_ratio=args.data_ratio,
        log_path=log_path,
        best_checkpoint_path=best_path,
        last_checkpoint_path=last_path,
        use_strong_aug=args.strong_aug,
        config=TrainConfig,
    )

    print("=" * 72)
    print("Experiment finished")
    print("=" * 72)
    print(f"best_val_acc={result['best_val_acc']:.4f}")
    print(f"best_val_f1={result['best_val_f1']:.4f}")
    print(f"best_epoch={result['best_epoch']}")
    print(f"log_path={result['log_path']}")
    print(f"best_checkpoint_path={result['best_checkpoint_path']}")


if __name__ == "__main__":
    main()
