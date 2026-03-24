"""
这是Zedwhite手搓的第一个AI代码 古法编程保证原创性 vibecoding含量尽可能低
Vision Transformer ———— 图像识别分类
"""

import torch
import torch.nn as nn


# 1. Patch Embedding
#    本质上是将一张二维图像切成多个小块，通过卷积层将每个小块映射到一个固定维度的向量空间中，形成一个序列。
#    因为Transformer只能处理序列数据，所以不得不这样干。
class PatchEmbed(nn.Module):
    """
    把图像切分为不重叠的 patch，然后用一个 Conv2d将每个 patch 映射到 embed_dim 维向量。

    Args:
        img_size   (int): 输入图像边长，默认 64
        patch_size (int): patch 边长，默认 8
        in_c(int): 输入通道数，默认 3 (RGB)
        embed_dim  (int): 输出嵌入维度，默认 256
    
    """
    def __init__(self, img_size=64, patch_size=8, in_c=3, embed_dim=256, norm_layer=nn.LayerNorm):
        super().__init__()

        # 这里是对于图像矩形的可分性验证，保证图像尺寸能够被 patch 大小整除。
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.embed_dim = embed_dim

        # 本质上调用了一个卷积层，用步长等于 patch 大小的卷积，一次完成切块和线性投影，以及标准化。
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # x: (B, C, H, W) 我决定以后在矩阵运算的后面写出他各个维度的含义
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x)          # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)          # (B, embed_dim, N)
        x = x.transpose(1, 2)     # (B, N, embed_dim)
        x = self.norm(x)          # (B, N, embed_dim)
        return x

# 2. Multi-Head Self-Attention
#    经典的多头自注意力机制，分成多个头来捕捉不同特征，相应地操作QKV，最后合并输出注意力结果。
class MultiHeadSelfAttention(nn.Module):
    """
    标准多头自注意力，支持输出 attention weights（用于可视化）。

    Args:
        embed_dim (int): token 维度
        num_heads (int): attention head 数量
        dropout   (float): 统一设置 attention/projection dropout
        qkv_bias  (bool) : QKV 线性层是否使用 bias
        qk_scale  (float, optional): 自定义缩放因子，默认使用 1/sqrt(head_dim)
        attn_drop_ratio (float, optional): attention dropout，优先级高于 dropout
        proj_drop_ratio (float, optional): projection dropout，优先级高于 dropout
    """
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        dropout=0.1,
        qkv_bias=True,
        qk_scale=None,
        attn_drop_ratio=None,
        proj_drop_ratio=None,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = qk_scale or self.head_dim ** -0.5   # 1/√d_k

        # dropout：若未单独指定则沿用 dropout。
        # dropout是为了在训练的时候避免“死记硬背”，关闭部分神经元，强迫模型学会更鲁棒的特征表示。
        attn_drop_ratio = dropout if attn_drop_ratio is None else attn_drop_ratio
        proj_drop_ratio = dropout if proj_drop_ratio is None else proj_drop_ratio

        self.qkv        = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj       = nn.Linear(embed_dim, embed_dim)
        self.attn_drop  = nn.Dropout(attn_drop_ratio)    
        self.proj_drop  = nn.Dropout(proj_drop_ratio)

    def forward(self, x, return_attn=False):
        B, N, C = x.shape

        # QKV 投影并拆分 就是一个矩阵运算 没啥好说的 具体看论文Attention Is All You Need的公式
        # 这里B是batchsize，N是patchnum，C是embeddim 我们一块处理三个矩阵 然后再拆开
        qkv = self.qkv(x)                              # (B, N, 3C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)              # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)                        # 各 (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v)                               # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)     # (B, N, embed_dim)
        out = self.proj(out)
        out = self.proj_drop(out)

        # 这里为后续可视化留了一个后门 可以把return_attn设为True来返回attn矩阵 有助于查看计算机把注意力放在哪个patch上
        if return_attn:
            return out, attn
        return out

# 3. MLP Block（Feed-Forward Network）
# MLP特征分析 进行升维 GELU非线性分析 降维还原 Dropout防止过拟合
# 吐槽一下 Claude编程总是追求优雅封装简洁 但是这样会使得可拓展性极差 所以我不得不古法编程
class MLP(nn.Module):
    """
    两层全连接 + GELU + Dropout。

    Args:
        in_features    (int)  : 输入维度
        hidden_features (int) : 隐层维度（通常 = in_features × mlp_ratio）
        out_features   (int) : 输出维度
        act_layer      (nn.Module): 激活函数
        drop           (float): dropout 概率
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
# 准备一个 DropPath 模块 用于 Transformer Block 中的残差连接丢弃
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    随机深度实现，在残差连接中有概率丢弃整个路径，以增强模型的泛化能力。
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)    

# 4. Transformer Encoder Block
#    Pre-LN 结构（LayerNorm 在 attention/MLP 之前，训练更稳定）
#    核心在 通过attention进行不同patch之间的特征交互 再通过MLP自己思考分析特征 
class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Block：
        out = x + MHSA(LN(x))
        out = out + MLP(LN(out))

    Args:
        embed_dim (int)  : token 维度
        num_heads (int)  : attention heads
        mlp_ratio (float): MLP 隐层扩展比例
        dropout   (float): dropout 概率
    """
    def __init__(
        self, embed_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.1,
        qkv_bias=True, qk_scale=None,
        attn_drop_ratio=None, proj_drop_ratio=None, drop_path_ratio=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        #先用多头注意力线性分析 再用MLP扩展特征线性分析 最后用残差连接和drop path增强模型鲁棒性
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim, num_heads=num_heads, dropout=dropout, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=proj_drop_ratio,
        )
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0.0 else nn.Identity()
        self.norm2 = norm_layer(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            act_layer=act_layer,
            drop=dropout,
        )

    #这里依旧保留了输出可视化的后门 方便查看注意力权重
    def forward(self, x, return_attn=False):
        attn_out = self.attn(self.norm1(x), return_attn=return_attn)
        if return_attn:
            attn_out, attn_weights = attn_out

        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if return_attn:
            return x, attn_weights
        return x
    
# 5. ViT 主体结构
class ViT(nn.Module):
    """
    超参数总览（与 config.py 对应）：
        img_size        = 64
        patch_size      = 8    → num_patches = 64
        embed_dim       = 256
        depth           = 6
        num_heads       = 8
        mlp_ratio       = 4.0
        dropout         = 0.1
        drop_path_ratio = 0.1
        num_classes     = 10
    """
    def __init__(
        self,
        img_size=64,
        patch_size=8,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        drop_path_ratio=0.1,
    ):
        super().__init__()
 
        # 超参数记录，方便外部读取打印
        self.img_size        = img_size
        self.patch_size      = patch_size
        self.embed_dim       = embed_dim
        self.depth           = depth
        self.num_heads       = num_heads
        self.mlp_ratio       = mlp_ratio
        self.num_classes     = num_classes
        self.drop_path_ratio = drop_path_ratio
 
        # Patch Embedding 就是用一遍我写好的模块
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
 
        # CLS token 通过 attention 主动从所有patch读取信息 变成序列的形式 Transformer才能处理
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
 
        # Position embedding 这里是很关键的 在序列化的同时 我们丢失了patch的相对位置信息
        # 所以我们需要给每个patch加上一个位置编码，让模型知道它们在图像中的位置关系，这样才能捕捉空间结构。
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
 
        # Stochastic depth decay（越深的层 drop_path 概率越大）
        dpr = torch.linspace(0, drop_path_ratio, depth).tolist()
 
        # Transformer blocks 也是用一遍自己写好的模块
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path_ratio=dpr[i],
            )
            for i in range(depth)
        ])
 
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
 
        self._init_weights()
 
    # 初始化权重 定起点
    def _init_weights(self):
        # 截断正态分布 见论文 防止过大权重导致训练不稳定
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
 
    # 前传函数，分成两步：先提取特征，再通过分类头输出类别概率。也就是应用__init__的各个模块 原理不再赘述
    def forward_features(self, x, return_attn=False):
        B = x.shape[0]
 
        x = self.patch_embed(x)                         # (B, N, D)
 
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)                  # (B, N+1, D)
        x = x + self.pos_embed
        x = self.pos_drop(x)
 
        attn_weights = None
        for i, blk in enumerate(self.blocks):
            if return_attn and i == len(self.blocks) - 1:
                x, attn_weights = blk(x, return_attn=True)
            else:
                x = blk(x)
 
        x = self.norm(x)
        cls_out = x[:, 0]                               # (B, D)
        # 这个cls就是代表这张图片的序列特征了 后续分类头会用它来输出类别概率
        
        # 可视化分析可用
        if return_attn:
            return cls_out, attn_weights
        return cls_out
 
    # pytorch调用接口
    def forward(self, x, return_attn=False):
        if return_attn:
            feat, attn = self.forward_features(x, True)
            return self.head(feat), attn
        return self.head(self.forward_features(x))
 
    # 这里开始就是Claude写的为了方便后续实验的代码 我不太懂 大家可以看看
    def get_attention_maps(self, x):
        """
        提取所有层的 attention weights，用于 attention_map.py 多层可视化。
 
        Returns:
            all_attn (list of Tensor): 每层 (B, num_heads, N+1, N+1)，共 depth 层
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
 
        all_attn = []
        for blk in self.blocks:
            x, attn = blk(x, return_attn=True)
            all_attn.append(attn.detach())              # (B, heads, N+1, N+1)
 
        return all_attn
 
    def count_parameters(self):
        """返回可训练参数量，对比实验用。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
 
    def __repr__(self):
        n = self.count_parameters()
        return (
            f"ViT(img={self.img_size}, patch={self.patch_size}, "
            f"dim={self.embed_dim}, depth={self.depth}, heads={self.num_heads}, "
            f"mlp_ratio={self.mlp_ratio}, drop_path={self.drop_path_ratio}) "
            f"— {n:,} params"
        )