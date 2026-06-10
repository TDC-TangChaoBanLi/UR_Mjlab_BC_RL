"""状态编码器工厂。"""

from __future__ import annotations

from typing import Any

from .base import StateEncoderBase
from .mlp_state import MLPStateEncoder


def build_state_encoder(cfg: dict[str, Any]) -> StateEncoderBase:
    """根据配置构建状态编码器。

    通过 hidden 是否存在自动选择模式：
      - 有 hidden → MLP 模式（含隐藏层）
      - 无 hidden → 线性模式（单层 Linear）

    cfg 格式:
      state_encoder:
        input_dim: 7
        hidden: {dim:128, activation:silu, ...}   # 可选
        output: {dim:256, layer_norm:true}
    """
    input_dim = cfg.get("input_dim", 7)
    out_cfg = cfg.get("output", {})
    output_dim = out_cfg.get("dim", 256)
    output_norm = out_cfg.get("layer_norm", True)

    hidden_cfg = cfg.get("hidden")
    if hidden_cfg is not None:
        # MLP 模式
        return MLPStateEncoder(
            input_dim=input_dim,
            hidden_dims=[hidden_cfg.get("dim", 128)],
            output_dim=output_dim,
            activation=hidden_cfg.get("activation", "silu"),
            dropout=hidden_cfg.get("dropout", 0.0),
            output_norm=output_norm,
        )
    else:
        # 线性模式
        return MLPStateEncoder(
            input_dim=input_dim,
            hidden_dims=None,
            output_dim=output_dim,
            output_norm=output_norm,
        )
