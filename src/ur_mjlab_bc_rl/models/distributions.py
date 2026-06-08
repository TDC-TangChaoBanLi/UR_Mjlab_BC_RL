"""高斯分布模块。

支持从均值和对数标准差采样、计算对数概率等。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributions as dist


class GaussianDistribution:
    """高斯分布。
    
    用于 PPO 算法中的连续动作采样和概率计算。
    """

    def __init__(self, action_dim: int, log_std_init: float = -0.5):
        """初始化。
        
        Args:
            action_dim: 动作维度
            log_std_init: 初始对数标准差
        """
        self.action_dim = action_dim
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

    def sample(self, mean: torch.Tensor) -> torch.Tensor:
        """从分布采样。
        
        Args:
            mean: [B, action_dim] 均值
        
        Returns:
            [B, action_dim] 采样的动作
        """
        std = torch.exp(self.log_std)
        normal = dist.Normal(torch.zeros_like(mean), torch.ones_like(mean))
        action = mean + std * normal.sample()
        return action

    def log_prob(self, action: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        """计算动作的对数概率。
        
        Args:
            action: [B, action_dim] 动作
            mean: [B, action_dim] 均值
        
        Returns:
            [B] 对数概率
        """
        std = torch.exp(self.log_std)
        var = std ** 2
        log_prob = -0.5 * (
            ((action - mean) ** 2 / var).sum(dim=-1)
            + torch.log(2 * torch.tensor(3.14159265) * var).sum(dim=-1)
        )
        return log_prob

    def entropy(self) -> torch.Tensor:
        """计算熵。
        
        Returns:
            标量张量，表示熵
        """
        std = torch.exp(self.log_std)
        return (self.log_std + 0.5 * torch.log(2 * torch.tensor(3.14159265) * torch.ones_like(std))).sum()
