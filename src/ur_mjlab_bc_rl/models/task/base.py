"""任务编码器基类。"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..specs import EncoderOutput


class TaskEncoderBase(nn.Module):
    """任务编码器的抽象基类。
    
    所有任务编码器子类必须实现 forward 方法，返回 EncoderOutput。
    """

    def forward(self, task_id: torch.Tensor) -> EncoderOutput:
        """处理任务 ID。
        
        Args:
            task_id: 任务 ID 张量 [B] 或 [B, 1]
        
        Returns:
            EncoderOutput：包含向量和可选 token 的编码输出
        
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def get_output_dim(self) -> int:
        """获取输出维度。"""
        raise NotImplementedError("Subclasses must implement get_output_dim method")
