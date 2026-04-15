# config.py
# 集中管理所有超参数，方便调参和实验记录
import os

class Config:
    # ========== 数据参数 ==========
    data_ratio = 1.0          # 训练集使用比例 (1.0, 0.2, 0.1)
    batch_size = 64           # 批次大小（CNN 默认用 64，ViT 会单独覆盖）
    num_workers = 2           # 数据加载进程数
    
    # ========== 训练参数 ==========
    epochs = 20               # 训练轮数（CNN 默认，ViT 会单独覆盖）
    lr = 1e-3                 # 学习率（CNN 默认，ViT 会单独覆盖）
    device = 'cuda'           # 'cuda' 或 'cpu'
    
    # ========== 模型选择 ==========
    model_name = 'cnn'        # 'cnn' 或 'vit'
    
    # ========== CNN 参数 ==========
    cnn_channels = [64, 128, 256, 512]
    cnn_dropout = 0.1
    
    # ========== ViT 参数 ==========
    vit_img_size = 64
    vit_patch_size = 4                # 修改：8 → 4
    vit_in_channels = 3
    vit_embed_dim = 192               # 修改：256 → 192（必须能被 num_heads 整除，192/8=24 ✅）
    vit_depth = 6
    vit_num_heads = 8
    vit_mlp_ratio = 4.0
    vit_dropout = 0.1
    vit_drop_path_ratio = 0.1
    
    # ========== ViT 专用训练参数（覆盖默认值） ==========
    vit_epochs = 40                  # ViT 训练 100 轮
    vit_lr = 5e-4                     # ViT 学习率 0.0005
    vit_batch_size = 32              # ViT 批次大小（显存限制）
    vit_warmup_epochs = 3             # ViT 预热轮数
    
    # ========== 优化器参数 ==========
    weight_decay = 0.01               # AdamW 的权重衰减
    
    # ========== 模型保存路径（动态生成） ==========
    @property
    def save_path(self):
        """根据模型名称和数据比例自动生成保存路径"""
        os.makedirs("checkpoints", exist_ok=True)
        return f"checkpoints/{self.model_name}_ratio{self.data_ratio}_best.pth"


# 方便直接打印配置
if __name__ == "__main__":
    cfg = Config()
    print("=" * 50)
    print("当前实验配置")
    print("=" * 50)
    for key, value in vars(cfg).items():
        if not key.startswith("__") and not callable(getattr(cfg, key)):
            print(f"{key:20} = {value}")
    print(f"{'save_path':20} = {cfg.save_path}")