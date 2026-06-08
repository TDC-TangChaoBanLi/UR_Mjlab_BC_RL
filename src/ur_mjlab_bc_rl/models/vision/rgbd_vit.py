"""RGBD ViT 编码器.

输入: (B, 4, H, W) RGBD 张量
输出: EncoderOutput（包含向量表示和可选的 token 序列）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..specs import EncoderOutput
from .base import VisualEncoderBase


class RGBDPatchEmbed(nn.Module):
    """RGBD Patch Embedding: Conv2d(4, embed_dim, patch_size, stride=patch_size).
    
    支持非正方形图像输入。
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (128, 128),
        patch_size: int = 16,
        in_channels: int = 4,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        
        # 计算 patch 数量（支持非正方形图像）
        h, w = image_size
        self.num_patches = (h // patch_size) * (w // patch_size)
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, H, W) -> (B, embed_dim, H/P, W/P) -> (B, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class RGBDViT(VisualEncoderBase):
    """轻量 RGBD Vision Transformer.

    Config:
        image_size: (H, W) - 支持非正方形图像
        patch_size: 16
        in_channels: 4
        embed_dim: 256
        depth: 4
        num_heads: 4
        mlp_ratio: 4
        output_dim: 256
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (128, 128),
        patch_size: int = 16,
        in_channels: int = 4,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        
        # 计算 patch 数量（支持非正方形图像）
        h, w = image_size
        self.num_patches = (h // patch_size) * (w // patch_size)

        self.patch_embed = RGBDPatchEmbed(image_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, output_dim)

        # Init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        """
        Args:
            x: (B, 4, H, W) RGBD 图像，RGB 在 [0,1], Depth 在 [0,1].

        Returns:
            EncoderOutput：包含向量表示的输出
        """
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, E)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, E)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.norm(x[:, 0])  # class token
        vector = self.head(x)
        return EncoderOutput(vector=vector, tokens=None)

    def get_output_dim(self) -> int:
        """返回输出维度。"""
        return self.head.out_features  # 返回 Linear 层的输出维度