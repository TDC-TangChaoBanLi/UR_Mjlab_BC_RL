"""Embedding 任务编码器。"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..specs import EncoderOutput
from .base import TaskEncoderBase


class EmbeddingTaskEncoder(TaskEncoderBase):
    """Embedding 任务编码器。
    
    使用查找表 (Embedding) 编码离散的任务 ID。
    
    Args:
        num_tasks: 任务总数
        embedding_dim: 嵌入维度
        output_dim: 输出特征维度（可选，若不同于 embedding_dim 则需要映射层）
    """

    def __init__(
        self,
        num_tasks: int,
        embedding_dim: int,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        
        self.num_tasks = num_tasks
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim or embedding_dim
        
        self.task_embed = nn.Embedding(num_tasks, embedding_dim)
        
        # 如果 output_dim 不等于 embedding_dim，添加映射层
        if self.output_dim != embedding_dim:
            self.proj = nn.Linear(embedding_dim, self.output_dim)
        else:
            self.proj = None

    def forward(self, task_id: torch.Tensor) -> EncoderOutput:
        """处理任务 ID。
        
        Args:
            task_id: [B] 或 [B, 1] 任务 ID
        
        Returns:
            EncoderOutput：包含向量 [B, output_dim] 和 token [B, 1, output_dim]
        """
        # 确保 task_id 是 1D 的
        if task_id.dim() == 2:
            task_id = task_id.squeeze(-1)
        
        vector = self.task_embed(task_id)  # [B, embedding_dim]
        
        if self.proj is not None:
            vector = self.proj(vector)  # [B, output_dim]
        
        # 生成 token：[B, 1, output_dim]
        tokens = vector.unsqueeze(1)
        
        return EncoderOutput(vector=vector, tokens=tokens)

    def get_output_dim(self) -> int:
        """返回输出维度。"""
        return self.output_dim
