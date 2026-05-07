"""
Project-wide configuration.

Keep model, training, data, logging, and checkpoint settings here so that
train.py, main.py, dataset.py, and visualize_attention.py do not each carry
their own hard-coded experiment parameters.

Usage:
    from config import VitConfig, CnnConfig, TrainConfig
    model = ViT(**VitConfig.to_vit_kwargs())
    model = CNN(**CnnConfig.to_cnn_kwargs())
"""

import torch

# ViT 模型超参
class VitConfig:
    # Image and patch settings.
    img_size        = 64
    image_size      = img_size
    patch_size      = 8
    in_channels     = 3

    # Transformer structure.
    embed_dim       = 256
    depth           = 6
    num_heads       = 8
    mlp_ratio       = 4.0

    # Regularization.
    dropout         = 0.1
    drop_path_ratio = 0.1

    # Classification.
    num_classes     = 10

    @classmethod
    def to_vit_kwargs(cls) -> dict:
        """Return keyword arguments accepted by ViT."""
        return dict(
            img_size        = cls.img_size,
            patch_size      = cls.patch_size,
            in_channels     = cls.in_channels,
            num_classes     = cls.num_classes,
            embed_dim       = cls.embed_dim,
            depth           = cls.depth,
            num_heads       = cls.num_heads,
            mlp_ratio       = cls.mlp_ratio,
            dropout         = cls.dropout,
            drop_path_ratio = cls.drop_path_ratio,
        )

    @classmethod
    def to_dict(cls) -> dict:
        return cls.to_vit_kwargs()


# CNN 模型超参
class CnnConfig:
    channels    = [64, 128, 256, 512]   # 四层卷积通道数
    dropout     = 0.1
    num_classes = 10

    @classmethod
    def to_cnn_kwargs(cls) -> dict:
        return dict(
            num_classes = cls.num_classes,
            channels    = cls.channels,
            dropout     = cls.dropout,
        )

    @classmethod
    def to_dict(cls) -> dict:
        return cls.to_cnn_kwargs()


# 训练超参
class TrainConfig:
    # Basic training settings.
    seed         = 42
    num_classes  = 10
    num_epochs   = 100
    batch_size   = 128
    num_workers  = 4
    device       = "cuda" if torch.cuda.is_available() else "cpu"

    # Experiment settings.
    model_type     = "cnn"          # "cnn" or "vit"
    data_ratio     = 1.0            # 1.0, 0.2, or 0.1
    data_fractions = [0.1, 0.2, 1.0]

    # Data settings.
    img_size   = 64
    data_root  = "./data"
    train_dir  = "./data/cifar10-64/train"
    test_dir   = "./data/cifar10-64/test"

    # Optimizer.
    learning_rate = 3e-4
    lr            = learning_rate   # Backward-compatible alias.
    weight_decay  = 0.05
    betas         = (0.9, 0.999)

    # Learning rate scheduler.
    scheduler     = "warmup_cosine"
    lr_scheduler  = scheduler       # Backward-compatible alias.
    warmup_epochs = 5
    min_lr        = 1e-6

    # AMP and regularization.
    use_amp          = True
    label_smoothing  = 0.1
    grad_clip_norm   = 1.0

    # Data augmentation. Mixup/CutMix are intentionally disabled for this stage.
    use_augmentation     = True
    use_strong_aug       = False
    use_random_crop      = True
    use_color_jitter     = True
    use_random_erasing   = True
    random_erasing_p     = 0.25
    random_crop_padding  = 8
    color_jitter         = (0.4, 0.4, 0.4, 0.1)
    use_mixup            = False
    use_cutmix           = False

    # Output paths.
    save_dir       = "./checkpoints"
    log_dir        = "./logs"
    figure_dir     = "./figures"
    output_dir     = "./outputs"
    checkpoint_dir = "./checkpoints"

    # Logging and checkpoint policy.
    log_interval   = 10
    save_best_only = True
    save_last      = True

    @classmethod
    def experiment_name(cls, model_type=None, data_ratio=None) -> str:
        model_type = model_type or cls.model_type
        data_ratio = cls.data_ratio if data_ratio is None else data_ratio
        return f"{model_type}_ratio{data_ratio}"

    @classmethod
    def log_path(cls, model_type=None, data_ratio=None) -> str:
        return f"{cls.log_dir}/{cls.experiment_name(model_type, data_ratio)}.csv"

    @classmethod
    def best_checkpoint_path(cls, model_type=None, data_ratio=None) -> str:
        return f"{cls.checkpoint_dir}/{cls.experiment_name(model_type, data_ratio)}_best.pth"

    @classmethod
    def last_checkpoint_path(cls, model_type=None, data_ratio=None) -> str:
        return f"{cls.checkpoint_dir}/{cls.experiment_name(model_type, data_ratio)}_last.pth"

    @classmethod
    def to_dict(cls) -> dict:
        fields = {
            "seed", "num_classes", "num_epochs", "batch_size", "num_workers",
            "device", "model_type", "data_ratio", "data_fractions", "img_size",
            "data_root", "train_dir", "test_dir", "learning_rate", "lr",
            "weight_decay", "betas", "scheduler", "lr_scheduler",
            "warmup_epochs", "min_lr", "use_amp", "label_smoothing",
            "grad_clip_norm", "use_augmentation", "use_strong_aug",
            "use_random_crop", "use_color_jitter", "use_random_erasing",
            "random_erasing_p", "random_crop_padding", "color_jitter",
            "use_mixup", "use_cutmix", "save_dir", "log_dir", "figure_dir",
            "output_dir", "checkpoint_dir", "log_interval", "save_best_only",
            "save_last",
        }
        return {name: getattr(cls, name) for name in fields}


def get_full_config() -> dict:
    """Return a serializable snapshot for logs and checkpoints."""
    return {
        "train": TrainConfig.to_dict(),
        "vit": VitConfig.to_dict(),
        "cnn": CnnConfig.to_dict(),
    }
