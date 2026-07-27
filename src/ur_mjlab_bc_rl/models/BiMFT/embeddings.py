"""BiMFT 位置编码与时间编码模块."""

from __future__ import annotations

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════
# 2D 可学习位置编码（行列分解）
# ═══════════════════════════════════════════════════════════

class Factorized2DPositionEmbedding(nn.Module):
    """行列分解可学习 2D 位置编码。

    存储 row[H,1,D] + col[1,W,D]，相加后展平为 [1, H*W, D]。
    比直接保存 H*W 个向量更容易修改网格大小。
    """

    def __init__(self, grid_h: int, grid_w: int, d_model: int):
        super().__init__()
        self.row = nn.Parameter(torch.zeros(1, grid_h, 1, d_model))
        self.col = nn.Parameter(torch.zeros(1, 1, grid_w, d_model))
        nn.init.trunc_normal_(self.row, std=0.02)
        nn.init.trunc_normal_(self.col, std=0.02)

    def forward(self) -> torch.Tensor:
        pos = self.row + self.col               # [1, H, W, D]
        return pos.flatten(1, 2)                # [1, H*W, D]


# ═══════════════════════════════════════════════════════════
# 连续时间编码（Fourier Features + MLP）
# ═══════════════════════════════════════════════════════════

class ContinuousTimeEmbedding(nn.Module):
    """Fourier Features + MLP 连续时间编码。

    将相对时间差（秒）编码为 d_model 维特征向量。
    """

    def __init__(self, d_model: int, num_frequencies: int = 32):
        super().__init__()
        frequencies = 2.0 ** torch.arange(num_frequencies)
        self.register_buffer(
            "frequencies",
            frequencies * torch.pi,
            persistent=False,
        )
        self.mlp = nn.Sequential(
            nn.Linear(num_frequencies * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        """dt: 任意 shape，最后一维自动扩展为 d_model."""
        phase = dt.unsqueeze(-1) * self.frequencies
        features = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        return self.mlp(features)
