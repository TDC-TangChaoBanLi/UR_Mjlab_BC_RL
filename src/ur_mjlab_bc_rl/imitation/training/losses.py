"""模仿学习损失函数。"""

from __future__ import annotations

import torch
import torch.nn as nn


class SquaredErrorLoss(nn.Module):
    """平方误差损失（MSE）。"""

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, pred_action: torch.Tensor, expert_action: torch.Tensor) -> torch.Tensor:
        """计算损失。
        
        Args:
            pred_action: [B, action_dim] 预测动作
            expert_action: [B, action_dim] 专家动作
        
        Returns:
            标量损失值
        """
        return self.loss_fn(pred_action, expert_action)


class L1Loss(nn.Module):
    """L1 损失。"""

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, pred_action: torch.Tensor, expert_action: torch.Tensor) -> torch.Tensor:
        """计算损失。"""
        return self.loss_fn(pred_action, expert_action)


class HuberLoss(nn.Module):
    """Huber 损失（L1 和 L2 的混合）。"""

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.loss_fn = nn.HuberLoss(delta=delta)

    def forward(self, pred_action: torch.Tensor, expert_action: torch.Tensor) -> torch.Tensor:
        """计算损失。"""
        return self.loss_fn(pred_action, expert_action)


def build_loss(loss_type: str = "mse", **kwargs) -> nn.Module:
    """构建损失函数。
    
    Args:
        loss_type: 损失类型（"mse", "l1", "huber"）
        **kwargs: 损失函数参数
    
    Returns:
        损失函数模块
    """
    if loss_type == "mse":
        return SquaredErrorLoss()
    elif loss_type == "l1":
        return L1Loss()
    elif loss_type == "huber":
        return HuberLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
