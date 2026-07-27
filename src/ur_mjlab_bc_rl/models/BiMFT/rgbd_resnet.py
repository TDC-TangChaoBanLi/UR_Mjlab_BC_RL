"""BiMFT 四通道 RGB-D ResNet-18 视觉编码器。

使用 torchvision.models.resnet18，将第一层 3→4 通道，
去掉 avgpool 和 fc，输出固定 grid 的空间 token 序列 [B, grid_h*grid_w, d_model]。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class RGBDResNet18(nn.Module):
    """四通道 ResNet-18，输出 token 序列。

    Args:
        d_model: 输出通道维度（1×1 conv projection 的目标通道数）
        grid_h: 输出空间网格高度
        grid_w: 输出空间网格宽度
        pretrained: 是否使用 ImageNet 预训练权重初始化 RGB 三通道
    """

    def __init__(
        self,
        d_model: int = 512,
        grid_h: int = 15,
        grid_w: int = 20,
        pretrained: bool = True,
    ):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        net = resnet18(weights=weights)

        # 第一层卷积: 3 通道 → 4 通道
        old_conv = net.conv1
        conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,       # type: ignore[arg-type]
            stride=old_conv.stride,                 # type: ignore[arg-type]
            padding=old_conv.padding,               # type: ignore[arg-type]
            bias=False,
        )

        with torch.no_grad():
            if weights is not None:
                conv1.weight[:, :3].copy_(old_conv.weight)
                conv1.weight[:, 3:4].copy_(
                    old_conv.weight.mean(dim=1, keepdim=True)
                )
            else:
                nn.init.kaiming_normal_(
                    conv1.weight, mode="fan_out", nonlinearity="relu"
                )

        self.conv1 = conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        self.grid_pool = nn.AdaptiveAvgPool2d((grid_h, grid_w))

        backbone_dim = 512  # ResNet-18 layer4 输出通道数
        self.proj = (
            nn.Identity()
            if backbone_dim == d_model
            else nn.Conv2d(backbone_dim, d_model, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 4, H, W] → [B, grid_h*grid_w, d_model]"""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)                       # [B, 512, H', W']

        x = self.grid_pool(x)                    # [B, 512, grid_h, grid_w]
        x = self.proj(x)                         # [B, d_model, grid_h, grid_w]

        # [B, D, H, W] → [B, D, H*W] → [B, H*W, D]
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x
