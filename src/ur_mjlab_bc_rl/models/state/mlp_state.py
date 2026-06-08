"""MLP 状态编码器。"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..specs import EncoderOutput
from ..modules.mlp import MLP
from .base import StateEncoderBase


class MLPStateEncoder(StateEncoderBase):
    """MLP 状态编码器。
    
    使用多层感知机处理机器人状态向量。
    
    Args:
        input_dim: 输入状态维度
        hidden_dims: 隐藏层维度列表
        output_dim: 输出特征维度
        activation: 激活函数名称
        dropout: Dropout 比例
        output_norm: 是否在输出加 LayerNorm（对齐视觉编码器 scale）
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
        
        if hidden_dims is None:
            hidden_dims = [128, 128]
        
        self.output_dim = output_dim
        
        self.mlp = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            dropout=dropout,
        )
        self.output_norm = nn.LayerNorm(output_dim) if output_norm else nn.Identity()

    def forward(self, state: torch.Tensor) -> EncoderOutput:
        """处理状态向量。
        
        Args:
            state: [B, input_dim]
        
        Returns:
            EncoderOutput：包含向量 [B, output_dim] 和 token [B, 1, output_dim]
        """
        vector = self.mlp(state)  # [B, output_dim]
        vector = self.output_norm(vector)
        tokens = vector.unsqueeze(1)
        return EncoderOutput(vector=vector, tokens=tokens)

    def get_output_dim(self) -> int:
        """返回输出维度。"""
        return self.output_dim
