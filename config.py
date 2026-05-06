"""
超参数集中管理
调参时只需要改这个文件的class参数

用法：
    from config import VitConfig, CnnConfig, TrainConfig
    model = ViT(**VitConfig.to_vit_kwargs())
    model = CNN(**CnnConfig.to_cnn_kwargs())
"""

# ViT 模型超参
class VitConfig:
    # 图像 & patch
    img_size        = 64        # 输入图像边长（数据集固定 64×64，不要改）
    patch_size      = 8         # patch 边长 → num_patches = (64/8)^2 = 64
    in_channels     = 3         # RGB

    # Transformer 结构
    embed_dim       = 256       # token 嵌入维度
    depth           = 6         # Transformer Block 层数
    num_heads       = 8         # 多头注意力头数（必须整除 embed_dim）
    mlp_ratio       = 4.0       # MLP 隐层扩展倍数

    # 正则化
    dropout         = 0.1       # attention / projection / pos_drop
    drop_path_ratio = 0.1       # stochastic depth 最大概率（线性递增到此值）

    # 分类
    num_classes     = 10        # CIFAR-10

    @classmethod
    def to_vit_kwargs(cls) -> dict:
        """返回可直接解包给 ViT(**kwargs) 的字典"""
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


# 训练超参
class TrainConfig:
    # 基础
    seed        = 42
    num_classes = 10
    num_epochs  = 50

    # 数据
    img_size    = 64
    batch_size  = 128
    num_workers = 4

    # 数据规模（加分项：小数据退化实验）
    # 支持多档：1.0 = 全量，0.2 = 20%，0.1 = 10%
    data_fractions = [1.0, 0.2, 0.1]

    # 优化器（AdamW）
    lr              = 3e-4
    weight_decay    = 0.05
    betas           = (0.9, 0.999)

    # 学习率调度（Cosine Annealing）
    lr_scheduler    = "cosine"      # "cosine" | "step" | "none"
    warmup_epochs   = 5             # 线性 warmup 轮数
    min_lr          = 1e-6          # cosine 最低学习率

    # 数据增强
    use_augmentation = True
    # 具体增强在 dataloader.py 里实现，这里只做开关

    # 硬件
    device      = "cuda"            # "cuda" | "cpu"
    use_amp     = True              # 混合精度训练（RTX 4070 支持，建议开启）

    # 路径
    data_root   = "./data"
    output_dir  = "./outputs"
    checkpoint_dir = "./checkpoints"

    # 日志 & 保存
    log_interval    = 10            # 每隔多少 batch 打印一次 loss
    save_best_only  = True          # 只保存 val acc 最高的 checkpoint
