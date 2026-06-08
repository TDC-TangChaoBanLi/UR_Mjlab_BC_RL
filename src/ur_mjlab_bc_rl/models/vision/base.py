"""视觉编码器基类。

所有视觉编码器都应继承此类。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..specs import EncoderOutput


class VisualEncoderBase(nn.Module):
    """视觉编码器的抽象基类。
    
    所有视觉编码器子类必须实现 forward 方法，返回 EncoderOutput。
    """

    def forward(self, rgbd: torch.Tensor) -> EncoderOutput:
        """处理 RGBD 图像并返回编码输出。
        
        Args:
            rgbd: RGBD 图像张量，形状为 [B, 4, H, W]
                  RGB 通常在 [0, 1]，深度在 [0, 1]
        
        Returns:
            EncoderOutput：包含 vector 和/或 tokens 的编码输出
        
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def get_output_dim(self) -> int:
        """获取输出维度。
        
        对于返回向量的编码器，返回向量的维度。
        对于返回 token 的编码器，返回 token 的嵌入维度。
        
        Returns:
            输出维度
        """
        raise NotImplementedError("Subclasses must implement get_output_dim method")
