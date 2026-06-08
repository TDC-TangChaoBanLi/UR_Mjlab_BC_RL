"""状态编码器工厂。"""

from __future__ import annotations

from typing import Any

from .base import StateEncoderBase
from .mlp_state import MLPStateEncoder


def build_state_encoder(cfg: dict[str, Any]) -> StateEncoderBase:
    """根据配置构建状态编码器。
    
    Args:
        cfg: 配置字典，必须包含 'type' 字段
        
    Returns:
        状态编码器实例
        
    Raises:
        ValueError: 如果类型不支持
    """
    encoder_type = cfg.get("type", "mlp")
    
    if encoder_type == "mlp":
        return MLPStateEncoder(
            input_dim=cfg.get("input_dim"),
            hidden_dims=cfg.get("hidden_dims", [128, 128]),
            output_dim=cfg.get("output_dim", 128),
            activation=cfg.get("activation", "silu"),
            dropout=cfg.get("dropout", 0.0),
            output_norm=cfg.get("output_norm", True),
        )
    
    else:
        raise ValueError(
            f"Unknown state encoder type: {encoder_type}. "
            f"Supported types: 'mlp'"
        )
