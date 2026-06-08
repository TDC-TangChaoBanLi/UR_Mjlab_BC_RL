"""FiLM 融合模块。

使用任务条件调制视觉和状态特征。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..specs import EncoderOutput
from ..modules.mlp import MLP
from .base import FusionBase


class FiLMFusion(FusionBase):
    """FiLM (Feature-wise Linear Modulation) 融合模块。
    
    基础特征：concat(visual, state)
    任务特征生成调制参数（gamma, beta）
    输出：(1 + gamma) * base + beta
    
    Args:
        visual_dim: 视觉特征维度
        state_dim: 状态特征维度
        task_dim: 任务特征维度
        hidden_dims: 用于生成 gamma 和 beta 的 MLP 隐藏层维度
        use_residual_gamma: 是否使用 (1 + gamma) 而不是 gamma
    """

    def __init__(
        self,
        visual_dim: int,
        state_dim: int,
        task_dim: int,
        hidden_dims: list[int] | None = None,
        use_residual_gamma: bool = True,
    ) -> None:
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128]
        
        self.visual_dim = visual_dim
        self.state_dim = state_dim
        self.task_dim = task_dim
        self.use_residual_gamma = use_residual_gamma
        
        self.base_dim = visual_dim + state_dim
        self.output_dim_value = self.base_dim
        
        # 从任务特征生成 FiLM 参数
        self.film_generator = MLP(
            input_dim=task_dim,
            hidden_dims=hidden_dims,
            output_dim=2 * self.base_dim,  # gamma 和 beta
        )

    def forward(
        self,
        visual: EncoderOutput,
        state: EncoderOutput,
        task: EncoderOutput,
    ) -> torch.Tensor:
        """使用 FiLM 融合三个编码器的输出。
        
        Args:
            visual: 视觉编码器输出
            state: 状态编码器输出
            task: 任务编码器输出
        
        Returns:
            [B, base_dim] 调制后的特征
        """
        visual_vec = visual.get_vector()  # [B, visual_dim]
        state_vec = state.get_vector()    # [B, state_dim]
        task_vec = task.get_vector()      # [B, task_dim]
        
        # 基础特征
        base = torch.cat([visual_vec, state_vec], dim=-1)  # [B, base_dim]
        
        # 生成 FiLM 参数
        film_params = self.film_generator(task_vec)  # [B, 2*base_dim]
        gamma, beta = torch.chunk(film_params, 2, dim=-1)  # 各 [B, base_dim]
        
        # 应用调制
        if self.use_residual_gamma:
            fused = (1.0 + gamma) * base + beta
        else:
            fused = gamma * base + beta
        
        return fused

    def get_output_dim(self) -> int:
        """获取输出维度。"""
        return self.output_dim_value
