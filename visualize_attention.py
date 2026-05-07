"""
Generate ViT attention maps from a trained checkpoint.

Examples:
    python visualize_attention.py --checkpoint checkpoints/vit_ratio1.0_best.pth
    python visualize_attention.py --checkpoint checkpoints/vit_ratio1.0_best.pth --image path/to/image.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import TrainConfig, VitConfig
from models.vit import ViT
from utils.dataset import get_cifar10_loaders


CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD = np.array([0.2023, 0.1994, 0.2010])


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint
    return {"model_state_dict": checkpoint}


def vit_kwargs_from_checkpoint(checkpoint):
    config = checkpoint.get("config", {})
    vit_config = config.get("vit") if isinstance(config, dict) else None
    if isinstance(vit_config, dict):
        allowed = {
            "img_size", "patch_size", "in_channels", "num_classes", "embed_dim",
            "depth", "num_heads", "mlp_ratio", "dropout", "drop_path_ratio",
        }
        return {key: value for key, value in vit_config.items() if key in allowed}
    return VitConfig.to_vit_kwargs()


def preprocess_image(path, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN.tolist(), CIFAR10_STD.tolist()),
    ])
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0), os.path.basename(path)


def denormalize(image_tensor):
    image = image_tensor.detach().cpu()[0].permute(1, 2, 0).numpy()
    image = CIFAR10_STD * image + CIFAR10_MEAN
    return np.clip(image, 0, 1)


def make_attention_overlay(model, image, device):
    model.eval()
    image = image.to(device)
    with torch.no_grad():
        all_attn = model.get_attention_maps(image)

    last_attn = all_attn[-1][0]
    cls_attn = last_attn.mean(dim=0)[0, 1:]
    grid_size = int(cls_attn.numel() ** 0.5)
    attn_map = cls_attn.reshape(1, 1, grid_size, grid_size)
    output_size = model.img_size if isinstance(model.img_size, tuple) else (model.img_size, model.img_size)
    attn_map = F.interpolate(attn_map, size=output_size, mode="bilinear", align_corners=False)[0, 0]
    attn_map = attn_map.detach().cpu().numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    return denormalize(image), attn_map


def save_attention_figure(image_np, attn_map, save_path, title):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image_np)
    plt.imshow(attn_map, cmap="jet", alpha=0.45)
    plt.title(title)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def checkpoint_experiment_name(checkpoint_path):
    name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    for suffix in ("_best", "_last"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize ViT attention maps.")
    parser.add_argument("--checkpoint", default=TrainConfig.best_checkpoint_path("vit", 1.0))
    parser.add_argument("--image", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max_images", type=int, default=3)
    parser.add_argument("--all_classes", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=TrainConfig.device)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    vit_kwargs = vit_kwargs_from_checkpoint(checkpoint)

    model = ViT(**vit_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"ViT config: {vit_kwargs}")

    if args.image:
        image, image_name = preprocess_image(args.image, vit_kwargs["img_size"])
        image_np, attn_map = make_attention_overlay(model, image, device)
        experiment_name = checkpoint_experiment_name(args.checkpoint)
        output = args.output or os.path.join(TrainConfig.figure_dir, f"attention_map_{experiment_name}.png")
        save_attention_figure(image_np, attn_map, output, f"Attention: {image_name}")
        print(f"Saved: {output}")
        return

    _, test_loader = get_cifar10_loaders(data_ratio=1.0, batch_size=1, augment=False, num_workers=0)
    seen_classes = set()
    saved_count = 0
    experiment_name = checkpoint_experiment_name(args.checkpoint)

    for images, labels in test_loader:
        label = labels[0].item()
        if args.all_classes:
            if label in seen_classes:
                continue
            seen_classes.add(label)
        elif saved_count >= args.max_images:
            break

        image_np, attn_map = make_attention_overlay(model, images, device)
        save_path = os.path.join(
            TrainConfig.figure_dir,
            f"attention_map_{experiment_name}_sample{saved_count + 1}.png",
        )
        save_attention_figure(image_np, attn_map, save_path, f"Attention: class {label}")
        saved_count += 1
        print(f"[{saved_count}] Saved: {save_path}")

        if args.all_classes and len(seen_classes) >= 10:
            break
        if not args.all_classes and saved_count >= args.max_images:
            break


if __name__ == "__main__":
    main()
