"""状态编码器 — MLP 或单层 Linear（由 YAML 配置隐式决定）。"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..specs import EncoderOutput
from ..modules.mlp import MLP
from .base import StateEncoderBase


class MLPStateEncoder(StateEncoderBase):
    """状态编码器。

    根据 hidden_dims 自动选择模式：
      - hidden_dims 非空 → MLP（含隐藏层 + 激活 + Dropout）
      - hidden_dims 为 None 或空 → 单层 Linear（ACT 原始方案）

    Args:
        input_dim: 输入状态维度
        hidden_dims: 隐藏层维度列表；None / 空 → 单层线性
        output_dim: 输出特征维度
        activation: 激活函数名称（仅 MLP 模式生效）
        dropout: Dropout 比例（仅 MLP 模式生效）
        output_norm: 是否在输出加 LayerNorm
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        output_dim: int = 128,
        activation: str = "silu",
        dropout: float = 0.0,
        output_norm: bool = True,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim

        if hidden_dims:
            # MLP 模式
            self.is_linear = False
            self.encoder = MLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                activation=activation,
                dropout=dropout,
            )
        else:
            # 线性模式（ACT 原始方案：单层 Linear）
            self.is_linear = True
            self.encoder = nn.Linear(input_dim, output_dim)

        self.output_norm = nn.LayerNorm(output_dim) if output_norm else nn.Identity()

    def forward(self, state: torch.Tensor) -> EncoderOutput:
        """处理状态向量。

        Args:
            state: [B, input_dim]

        Returns:
            EncoderOutput：包含向量 [B, output_dim] 和 token [B, 1, output_dim]
        """
        vector = self.encoder(state)  # [B, output_dim]
        vector = self.output_norm(vector)
        tokens = vector.unsqueeze(1)
        return EncoderOutput(vector=vector, tokens=tokens)

    def get_output_dim(self) -> int:
        """返回输出维度。"""
        return self.output_dim
