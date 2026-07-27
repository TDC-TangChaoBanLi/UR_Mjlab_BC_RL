"""BiMFT 高频状态编码器 — 关节和力/力矩 Token 编码。

VectorTokenEncoder: 单个向量 → d_model token 的 MLP 投影
HighRateModalityEncoder: 每个相机槽 3 个高频采样 → TransformerEncoder 短序列编码
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════
# 向量 Token 编码器
# ═══════════════════════════════════════════════════════════

class VectorTokenEncoder(nn.Module):
    """单向量 → token 的 MLP 投影: Linear → LayerNorm → GELU → Linear."""

    def __init__(self, input_dim: int, d_model: int, hidden_dim: int | None = None):
        super().__init__()
        hidden = hidden_dim or d_model
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., input_dim] → [..., d_model]"""
        return self.net(x)


# ═══════════════════════════════════════════════════════════
# 高频短序列编码器
# ═══════════════════════════════════════════════════════════

class HighRateModalityEncoder(nn.Module):
    """高频模态编码器。

    对每个相机时刻的 R 个采样（R=3）做:
      1. VectorTokenEncoder 投影 → [B, T, R, d_model]
      2. 注入 time_emb, arm_emb, modality_emb
      3. 合并 B 和 T → [B*T, R, d_model]
      4. nn.TransformerEncoder 短序列编码
      5. 恢复 → [B, T, R, d_model]

    Args:
        input_dim: 原始输入向量维度（关节: 7, 力觉: 6）
        d_model: token 嵌入维度
        n_heads: 注意力头数
        d_ff: FFN 中间维度
        dropout: dropout 概率
        num_layers: TransformerEncoder 层数
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        num_layers: int,
    ):
        super().__init__()
        self.input_proj = VectorTokenEncoder(
            input_dim=input_dim,
            d_model=d_model,
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer=enc_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        arm_emb: torch.Tensor,
        modality_emb: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: [B, T, R, input_dim] → [B, T, R, d_model]"""
        b, t, r, _ = x.shape

        # 投影 + embedding 注入
        y = self.input_proj(x)                       # [B, T, R, d_model]
        y = y + time_emb + arm_emb + modality_emb    # 广播: [B,T,R,D] + [B,T,R,D]

        # 合并 batch 和时间
        y = y.reshape(b * t, r, -1)                  # [B*T, R, d_model]

        mask = None
        if padding_mask is not None:
            mask = padding_mask.reshape(b * t, r)

        y = self.temporal_encoder(y, src_key_padding_mask=mask)

        return y.reshape(b, t, r, -1)                # [B, T, R, d_model]
