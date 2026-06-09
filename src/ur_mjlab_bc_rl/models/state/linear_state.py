"""线性状态编码器（ACT 原始方案）。

ACT 使用单层 Linear 将关节状态映射到 hidden_dim，不做 MLP。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..specs import EncoderOutput
from .base import StateEncoderBase


class LinearStateEncoder(StateEncoderBase):
    """线性状态编码器：单层 Linear + 可选 LayerNorm。

    ACT 原始方案：
        self.input_proj_robot_state = nn.Linear(14, hidden_dim)

    本项目适配：
        nn.Linear(state_dim, output_dim) + optional LayerNorm

    Args:
        input_dim: 输入状态维度（arm_joint_pos + gripper_pos = 7）
        output_dim: 输出特征维度（通常 = hidden_dim = 256）
        output_norm: 是否在输出加 LayerNorm（默认 True）
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        output_norm: bool = True,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim) if output_norm else nn.Identity()

    def forward(self, state: torch.Tensor) -> EncoderOutput:
        """处理状态向量。

        Args:
            state: [B, input_dim]

        Returns:
            EncoderOutput：包含向量 [B, output_dim] 和 token [B, 1, output_dim]
        """
        vector = self.linear(state)           # [B, output_dim]
        vector = self.output_norm(vector)
        tokens = vector.unsqueeze(1)          # [B, 1, output_dim]
        return EncoderOutput(vector=vector, tokens=tokens)

    def get_output_dim(self) -> int:
        """返回输出维度。"""
        return self.output_dim
