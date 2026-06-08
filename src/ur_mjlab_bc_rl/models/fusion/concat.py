"""Concat 融合模块。

简单的串联融合：直接拼接所有编码器的向量输出。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..specs import EncoderOutput
from .base import FusionBase


class ConcatFusion(FusionBase):
    """Concat 融合模块。
    
    直接拼接视觉、状态和任务特征。
    适合调试，稳定性好。
    
    args:
        output_dim: 可选，若指定则添加映射层到该维度
    """

    def __init__(self, output_dim: int | None = None) -> None:
        super().__init__()
        self.output_dim_value = output_dim
        self.proj = None

    def forward(
        self,
        visual: EncoderOutput,
        state: EncoderOutput,
        task: EncoderOutput,
    ) -> torch.Tensor:
        """融合三个编码器的向量输出。
        
        Args:
            visual: 视觉编码器输出
            state: 状态编码器输出
            task: 任务编码器输出
        
        Returns:
            [B, D_v + D_s + D_t] 拼接后的特征
        """
        visual_vec = visual.get_vector()  # [B, D_v]
        state_vec = state.get_vector()    # [B, D_s]
        task_vec = task.get_vector()      # [B, D_t]
        
        fused = torch.cat([visual_vec, state_vec, task_vec], dim=-1)
        
        if self.proj is not None:
            fused = self.proj(fused)
        
        return fused

    def get_output_dim(self) -> int:
        """获取输出维度。"""
        if self.output_dim_value is not None:
            return self.output_dim_value
        raise ValueError("Output dimension not set for ConcatFusion")
