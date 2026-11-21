"""采用 ResNet 编码器和紧凑 U-Net 解码器的二值分割网络。

网络输入为 ``B×3×H×W`` 的归一化 RGB 图像，输出字典包含：

* ``logits``：``B×1×H×W`` 的前景 logits，供监督损失和 CRCC 使用；
* ``features``：解码末端的 ``B×64×H'×W'`` 特征图。

当前 CRCC 训练路径只读取 ``logits``。``features`` 为对比方法或后续扩展保留，
目前不进入 CRM、SAC、PRA，也不受 CRCC 损失直接约束。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBlock(nn.Module):
    """由两组卷积、组归一化和 ReLU 组成的解码器基本块。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """将低分辨率特征上采样，与同尺度编码器跳连特征拼接后融合。"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResNetUNet(nn.Module):
    """返回前景 logits 与 64 通道解码特征的二值分割模型。

    编码器可选 ResNet-18、ResNet-34 或 ResNet-50；解码器逐级融合四个编码阶段
    的跳连特征。分割头将最后的 64 通道特征映射为单通道 logits，并插值恢复到
    输入空间尺寸。Student 与 EMA Teacher 使用完全相同的该网络结构。
    """

    feature_dim = 64

    def __init__(self, backbone: str = "resnet50", pretrained: bool = False) -> None:
        super().__init__()
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            encoder = models.resnet18(weights=weights)
            channels = (64, 64, 128, 256, 512)
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            encoder = models.resnet34(weights=weights)
            channels = (64, 64, 128, 256, 512)
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            encoder = models.resnet50(weights=weights)
            channels = (64, 256, 512, 1024, 2048)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        c0, c1, c2, c3, c4 = channels
        self.up4 = UpBlock(c4, c3, 256)
        self.up3 = UpBlock(256, c2, 128)
        self.up2 = UpBlock(128, c1, 64)
        self.up1 = UpBlock(64, c0, self.feature_dim)
        self.segmentation_head = nn.Conv2d(self.feature_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """完成一次分割前向，并同时返回全尺寸 logits 和末端解码特征。

        参数：
            x: ``B×3×H×W`` 的归一化图像批次。

        返回：
            ``logits`` 为 ``B×1×H×W``；``features`` 为 64 通道末端解码特征。
            当前 CRCC 主流程仅使用 ``logits``。
        """

        input_size = x.shape[-2:]
        x0 = self.stem(x)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.up1(self.up2(self.up3(self.up4(x4, x3), x2), x1), x0)
        logits = self.segmentation_head(features)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return {"logits": logits, "features": features}
