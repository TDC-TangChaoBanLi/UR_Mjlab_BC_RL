"""MLP 模块."""

from __future__ import annotations

import torch.nn as nn


class MLP(nn.Module):
    """多层感知机。

    Config:
        input_dim: 输入维度
        hidden_dims: 隐藏层维度列表
        output_dim: 输出维度
        activation: 激活函数名称 ("relu", "silu", "gelu")
        dropout: dropout 概率
        layer_norm: 是否使用 LayerNorm
    """

    ACTIVATIONS = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "silu",
        dropout: float = 0.0,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        act_cls = self.ACTIVATIONS.get(activation, nn.SiLU)

        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-1], output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
