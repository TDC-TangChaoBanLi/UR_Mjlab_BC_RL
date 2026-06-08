"""状态编码器基类。"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..specs import EncoderOutput


class StateEncoderBase(nn.Module):
    """状态编码器的抽象基类。
    
    所有状态编码器子类必须实现 forward 方法，返回 EncoderOutput。
    """

    def forward(self, state: torch.Tensor) -> EncoderOutput:
        """处理机器人状态向量。
        
        Args:
            state: 机器人状态向量 [B, state_dim]
        
        Returns:
            EncoderOutput：包含向量和可选 token 的编码输出
        
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def get_output_dim(self) -> int:
        """获取输出维度。"""
        raise NotImplementedError("Subclasses must implement get_output_dim method")
