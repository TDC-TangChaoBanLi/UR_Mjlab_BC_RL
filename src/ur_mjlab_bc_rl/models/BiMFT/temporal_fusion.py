"""BiMFT 时序融合 — AnchoredTemporalFusion。

当前帧 (t=T-1) 作为 Query，历史帧 (t=0..T-2) 展平为 memory，
通过多层 CrossAttentionBlock 将历史信息动态注入当前帧。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import CrossAttentionBlock


class AnchoredTemporalFusion(nn.Module):
    """当前帧锚定的时序 Cross-Attention 融合。

    Args:
        d_model: token 维度
        n_heads: 注意力头数
        d_ff: FFN 中间维度
        dropout: dropout 概率
        num_layers: Cross-Attention 层数
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        num_layers: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, N, D] → [B, N, D] (当前帧输出)"""
        assert x.ndim == 4, f"Expected [B,T,N,D], got shape {x.shape}"

        query = x[:, -1]                             # [B, N, D]
        memory = x[:, :-1].flatten(1, 2)             # [B, (T-1)*N, D]

        for layer in self.layers:
            query = layer(query, memory)

        return query
