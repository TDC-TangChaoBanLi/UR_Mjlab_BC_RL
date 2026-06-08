"""融合模块基类。"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..specs import EncoderOutput


class FusionBase(nn.Module):
    """融合模块的抽象基类。
    
    所有融合模块子类必须实现 forward 方法。
    """

    def forward(
        self,
        visual: EncoderOutput,
        state: EncoderOutput,
        task: EncoderOutput,
    ) -> torch.Tensor:
        """融合多个编码器的输出。
        
        Args:
            visual: 视觉编码器输出
            state: 状态编码器输出
            task: 任务编码器输出
        
        Returns:
            融合后的特征张量 [B, D_fused]
        
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def get_output_dim(self) -> int:
        """获取融合后的输出维度。"""
        raise NotImplementedError("Subclasses must implement get_output_dim method")
