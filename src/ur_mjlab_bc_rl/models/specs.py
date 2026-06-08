"""统一的观测/动作/任务规格和编码器输出格式。

包含：
- EncoderOutput: 编码器的统一输出格式
- ObsSpec: 观测规格
- ActionSpec: 动作规格
- TaskSpec: 任务规格
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class EncoderOutput:
    """统一的编码器输出格式。
    
    支持向量输出、token 输出或两者兼有。
    
    Attributes:
        vector: 编码后的向量特征 [B, D]
        tokens: 编码后的 token 序列 [B, N, D]
        mask: Token 掩码 [B, N]（可选）
    """
    vector: Optional[torch.Tensor] = None  # [B, D]
    tokens: Optional[torch.Tensor] = None  # [B, N, D]
    mask: Optional[torch.Tensor] = None     # [B, N]

    def get_vector(self) -> torch.Tensor:
        """获取向量表示。
        
        如果有 vector，直接返回；
        否则从 tokens 池化得到。
        """
        if self.vector is not None:
            return self.vector
        if self.tokens is not None:
            # 简单的平均池化
            if self.mask is not None:
                masked_tokens = self.tokens * self.mask.unsqueeze(-1)
                sum_tokens = masked_tokens.sum(dim=1)
                token_counts = self.mask.sum(dim=1, keepdim=True)
                return sum_tokens / (token_counts + 1e-8)
            else:
                return self.tokens.mean(dim=1)
        raise ValueError("EncoderOutput must have either vector or tokens")

    def get_tokens(self) -> torch.Tensor:
        """获取 token 序列。"""
        if self.tokens is not None:
            return self.tokens
        raise ValueError("EncoderOutput does not have tokens")


@dataclass
class ObsSpec:
    """观测规格（Actor 的输入规格）。
    
    Attributes:
        camera_shape: RGBD 图像形状 (C, H, W)，通常为 (4, H, W)
        actor_state_dim: 机器人状态维度
        task_dim: 任务 ID 维度（通常为 1 或 num_tasks）
        num_tasks: 任务总数
    """
    camera_shape: tuple[int, int, int] = (4, 128, 128)
    actor_state_dim: int = 27
    task_dim: int = 1
    num_tasks: int = 3

    def to_dict(self) -> dict:
        """转换为字典格式。"""
        return {
            "camera_shape": self.camera_shape,
            "actor_state_dim": self.actor_state_dim,
            "task_dim": self.task_dim,
            "num_tasks": self.num_tasks,
        }


@dataclass
class ActionSpec:
    """动作规格。
    
    Attributes:
        action_dim: 动作维度（通常为 7：6 个 ee_delta + 1 个 gripper）
        action_min: 动作最小值
        action_max: 动作最大值
    """
    action_dim: int = 7
    action_min: Optional[torch.Tensor] = None
    action_max: Optional[torch.Tensor] = None

    def to_dict(self) -> dict:
        """转换为字典格式。"""
        return {
            "action_dim": self.action_dim,
            "action_min": self.action_min.tolist() if self.action_min is not None else None,
            "action_max": self.action_max.tolist() if self.action_max is not None else None,
        }


@dataclass
class TaskSpec:
    """任务规格。
    
    Attributes:
        num_tasks: 任务总数
        task_names: 任务名称列表
    """
    num_tasks: int = 3
    task_names: list[str] = field(default_factory=lambda: ["pick_place", "push_t", "peg_in_slot"])

    def to_dict(self) -> dict:
        """转换为字典格式。"""
        return {
            "num_tasks": self.num_tasks,
            "task_names": self.task_names,
        }


@dataclass
class CriticObsSpec:
    """评论家观测规格（Critic 的输入规格）。
    
    Attributes:
        privileged_state_dim: 特权状态维度（物体位姿、接触信息等）
        task_dim: 任务 ID 维度
        num_tasks: 任务总数
    """
    privileged_state_dim: int = 50
    task_dim: int = 1
    num_tasks: int = 3

    def to_dict(self) -> dict:
        """转换为字典格式。"""
        return {
            "privileged_state_dim": self.privileged_state_dim,
            "task_dim": self.task_dim,
            "num_tasks": self.num_tasks,
        }
