"""
准备一个手搓简易版的CNN 来对比Transformer
在设计上，争取保持与 ViT 的参数量相近，以实现公平对比。
原理参考：https://blog.csdn.net/ironmanjay/article/details/128689946?ops_request_misc=elastic_search_misc&request_id=84b96504e62b62e7d4a633ab1ba94d2c&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-2-128689946-null-null.142^v102^pc_search_result_base4&utm_term=CNN&spm=1018.2226.3001.4187
Copyright @Zedwhite_PG

"""

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    单个卷积块：卷积 标准化 池化 尺寸减半

    Args:
        in_channels  (int): 输入通道数 RGB为3
        out_channels (int): 输出通道数 卷积核的数量
        dropout      (float): dropout 概率
    """
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 两次卷积后再池化 比较符合常用特征提取习惯
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)
    
class CNN(nn.Module):
    """
    4层卷积 CNN，使得参数量与 ViT 相近，
    用于与 ViT 公平对比。

    Args:
        num_classes (int)  : 分类类别数，默认 10
        channels    (list) : 四层卷积的通道数
        dropout     (float): dropout 概率
    """
    def __init__(
        self,
        num_classes=10,
        channels=None,
        dropout=0.1,
    ):
        super().__init__()

        if channels is None:
            channels = [64, 128, 256, 512]

        assert len(channels) == 4, "channels 必须是长度为 4 的列表"

        # 超参数记录
        self.num_classes = num_classes
        self.channels    = channels

        # 4 个卷积块，每块将特征图减半
        # 64×64 → 32×32 → 16×16 → 8×8 → 4×4
        self.layer1 = ConvBlock(3,           channels[0], dropout)
        self.layer2 = ConvBlock(channels[0], channels[1], dropout)
        self.layer3 = ConvBlock(channels[1], channels[2], dropout)
        self.layer4 = ConvBlock(channels[2], channels[3], dropout)

        # 全局平均池化：(B, 512, 4, 4) → (B, 512, 1, 1) → (B, 512)
        # 比 Flatten 参数量更少，且对尺寸不敏感
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 分类头 加深分类头 补偿参数量差距
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[3], channels[3]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channels[3], num_classes),
        )

        self._init_weights()

    # 初始化权重
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 3, 64, 64)
        x = self.layer1(x)   # (B, 64,  32, 32)
        x = self.layer2(x)   # (B, 128, 16, 16)
        x = self.layer3(x)   # (B, 256, 8,  8)
        x = self.layer4(x)   # (B, 512, 4,  4)

        x = self.gap(x)      # (B, 512, 1, 1)
        x = x.flatten(1)     # (B, 512)
        # 最终得到一个一维特征向量，送入分类头进行分类

        x = self.classifier(x)   # (B, 10)
        return x

    def count_parameters(self):
        """返回可训练参数量，对比实验用。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # 方便打印模型信息，展示参数量
    def __repr__(self):
        n = self.count_parameters()
        return (
            f"CNN(channels={self.channels}, num_classes={self.num_classes}) "
            f"— {n:,} params"
        )
