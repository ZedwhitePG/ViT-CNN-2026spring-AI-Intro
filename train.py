"""
Training utilities for CNN and ViT experiments.

This module owns the shared training loop used by main.py:
- AdamW with config-driven lr and weight decay
- warmup + cosine learning-rate schedule
- AMP on CUDA, automatic FP32 fallback otherwise
- gradient clipping
- label smoothing
- per-epoch CSV logs
- structured best/last checkpoints
"""

import argparse
import csv
import math
import os
import time
from contextlib import nullcontext
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

from config import CnnConfig, TrainConfig, VitConfig, get_full_config
from models.cnn import CNN
from models.vit import ViT
from utils.dataset import get_cifar10_loaders


class WarmupCosineScheduler:
    """Epoch-level linear warmup followed by cosine decay."""

    def __init__(self, optimizer, base_lr, min_lr, warmup_epochs, total_epochs):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_epochs = max(0, warmup_epochs)
        self.total_epochs = max(1, total_epochs)
        self.last_epoch = -1

    def get_lr(self, epoch_index):
        if self.warmup_epochs > 0 and epoch_index < self.warmup_epochs:
            return self.base_lr * float(epoch_index + 1) / float(self.warmup_epochs)

        cosine_epochs = max(1, self.total_epochs - self.warmup_epochs)
        progress = (epoch_index - self.warmup_epochs + 1) / cosine_epochs
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def step(self, epoch_index):
        self.last_epoch = epoch_index
        lr = self.get_lr(epoch_index)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def state_dict(self):
        return {
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "warmup_epochs": self.warmup_epochs,
            "total_epochs": self.total_epochs,
            "last_epoch": self.last_epoch,
        }


class ConstantScheduler:
    """Small scheduler shim for scheduler='none'."""

    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.lr = lr
        self.last_epoch = -1

    def step(self, epoch_index):
        self.last_epoch = epoch_index
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr
        return self.lr

    def state_dict(self):
        return {"lr": self.lr, "last_epoch": self.last_epoch}


def build_optimizer(model, config=TrainConfig):
    return optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )


def build_scheduler(optimizer, epochs, config=TrainConfig):
    if config.scheduler == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer=optimizer,
            base_lr=config.learning_rate,
            min_lr=config.min_lr,
            warmup_epochs=config.warmup_epochs,
            total_epochs=epochs,
        )
    if config.scheduler == "none":
        return ConstantScheduler(optimizer, config.learning_rate)
    raise ValueError(f"Unsupported scheduler: {config.scheduler}")


def _as_device(device):
    return torch.device(device if isinstance(device, str) else device)


def _amp_enabled(device, use_amp):
    device = _as_device(device)
    return bool(use_amp and device.type == "cuda" and torch.cuda.is_available())


def _autocast_context(device, enabled):
    if enabled:
        try:
            return torch.amp.autocast("cuda")
        except AttributeError:
            return torch.cuda.amp.autocast()
    return nullcontext()


def _grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_epoch(model, loader, optimizer, criterion, device, scaler=None, use_amp=False, grad_clip_norm=1.0):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    amp_enabled = _amp_enabled(device, use_amp)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if amp_enabled:
            with _autocast_context(device, enabled=True):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            running_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average="macro")
    return val_loss, val_acc, val_f1


def _write_header(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "val_f1",
            "lr",
            "epoch_time",
            "best_val_acc",
            "model_type",
            "data_ratio",
            "use_amp",
            "use_strong_aug",
            "seed",
        ])


def _append_log(log_path, row: Dict):
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "val_f1",
            "lr",
            "epoch_time",
            "best_val_acc",
            "model_type",
            "data_ratio",
            "use_amp",
            "use_strong_aug",
            "seed",
        ])
        writer.writerow(row)


def _save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_acc, best_val_f1, model_type, data_ratio):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_val_acc": best_val_acc,
        "best_val_f1": best_val_f1,
        "config": get_full_config(),
        "model_type": model_type,
        "data_ratio": data_ratio,
    }, path)


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer=None,
    scheduler=None,
    epochs=TrainConfig.num_epochs,
    device=TrainConfig.device,
    model_type=TrainConfig.model_type,
    data_ratio=TrainConfig.data_ratio,
    log_path: Optional[str] = None,
    best_checkpoint_path: Optional[str] = None,
    last_checkpoint_path: Optional[str] = None,
    use_strong_aug=TrainConfig.use_strong_aug,
    config=TrainConfig,
):
    device = _as_device(device)
    model = model.to(device)
    log_path = log_path or config.log_path(model_type, data_ratio)
    best_checkpoint_path = best_checkpoint_path or config.best_checkpoint_path(model_type, data_ratio)
    last_checkpoint_path = last_checkpoint_path or config.last_checkpoint_path(model_type, data_ratio)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = optimizer or build_optimizer(model, config)
    scheduler = scheduler or build_scheduler(optimizer, epochs, config)
    amp_enabled = _amp_enabled(device, config.use_amp)
    scaler = _grad_scaler(amp_enabled)

    _write_header(log_path)

    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    epoch_times = []

    print(
        f"optimizer=AdamW lr={config.learning_rate} weight_decay={config.weight_decay} "
        f"scheduler={config.scheduler} warmup_epochs={config.warmup_epochs} "
        f"use_amp={amp_enabled} label_smoothing={config.label_smoothing} "
        f"grad_clip_norm={config.grad_clip_norm}"
    )

    for epoch_index in range(epochs):
        epoch = epoch_index + 1
        lr = scheduler.step(epoch_index)
        start_time = time.time()

        train_loss, train_acc = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_amp=config.use_amp,
            grad_clip_norm=config.grad_clip_norm,
        )
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_epoch = epoch
            _save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_acc,
                best_val_f1,
                model_type,
                data_ratio,
            )
            print(f"  -> saved best checkpoint: {best_checkpoint_path}")

        if config.save_last:
            _save_checkpoint(
                last_checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_acc,
                best_val_f1,
                model_type,
                data_ratio,
            )

        _append_log(log_path, {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "train_acc": f"{train_acc:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "val_acc": f"{val_acc:.6f}",
            "val_f1": f"{val_f1:.6f}",
            "lr": f"{lr:.10f}",
            "epoch_time": f"{epoch_time:.2f}",
            "best_val_acc": f"{best_val_acc:.6f}",
            "model_type": model_type,
            "data_ratio": data_ratio,
            "use_amp": amp_enabled,
            "use_strong_aug": use_strong_aug,
            "seed": config.seed,
        })

        print(
            f"Epoch {epoch:03d}/{epochs} | lr={lr:.8f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
            f"best_acc={best_val_acc:.4f} | time={epoch_time:.2f}s"
        )

    return {
        "avg_epoch_time": sum(epoch_times) / max(1, len(epoch_times)),
        "best_val_acc": best_val_acc,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "log_path": log_path,
        "best_checkpoint_path": best_checkpoint_path,
        "last_checkpoint_path": last_checkpoint_path,
    }


def build_model(model_type):
    if model_type == "cnn":
        return CNN(**CnnConfig.to_cnn_kwargs())
    if model_type == "vit":
        return ViT(**VitConfig.to_vit_kwargs())
    raise ValueError(f"Unknown model_type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description="Train CNN or ViT on CIFAR-10.")
    parser.add_argument("--model", choices=["cnn", "vit"], default=TrainConfig.model_type)
    parser.add_argument("--data_ratio", type=float, default=TrainConfig.data_ratio)
    parser.add_argument("--epochs", type=int, default=TrainConfig.num_epochs)
    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--strong_aug", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        args.epochs = min(args.epochs, 2)
        args.batch_size = min(args.batch_size, 32)

    train_loader, val_loader = get_cifar10_loaders(
        data_ratio=args.data_ratio,
        batch_size=args.batch_size,
        augment=TrainConfig.use_augmentation,
        strong_aug=args.strong_aug,
        num_workers=TrainConfig.num_workers,
        max_train_samples=256 if args.debug else None,
        max_test_samples=512 if args.debug else None,
    )
    model = build_model(args.model)
    if args.debug:
        experiment_name = f"debug_{TrainConfig.experiment_name(args.model, args.data_ratio)}"
        log_path = os.path.join(TrainConfig.log_dir, f"{experiment_name}.csv")
        best_checkpoint_path = os.path.join(TrainConfig.checkpoint_dir, f"{experiment_name}_best.pth")
        last_checkpoint_path = os.path.join(TrainConfig.checkpoint_dir, f"{experiment_name}_last.pth")
    else:
        log_path = TrainConfig.log_path(args.model, args.data_ratio)
        best_checkpoint_path = TrainConfig.best_checkpoint_path(args.model, args.data_ratio)
        last_checkpoint_path = TrainConfig.last_checkpoint_path(args.model, args.data_ratio)

    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        device=args.device,
        model_type=args.model,
        data_ratio=args.data_ratio,
        log_path=log_path,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        use_strong_aug=args.strong_aug,
    )
    print(result)


if __name__ == "__main__":
    main()
