"""任务编码器工厂。"""

from __future__ import annotations

from typing import Any

from .base import TaskEncoderBase
from .embedding_task import EmbeddingTaskEncoder


def build_task_encoder(cfg: dict[str, Any]) -> TaskEncoderBase:
    """根据配置构建任务编码器。
    
    Args:
        cfg: 配置字典，必须包含 'type' 字段
        
    Returns:
        任务编码器实例
        
    Raises:
        ValueError: 如果类型不支持
    """
    encoder_type = cfg.get("type", "embedding")
    
    if encoder_type == "embedding":
        return EmbeddingTaskEncoder(
            num_tasks=cfg.get("num_tasks"),
            embedding_dim=cfg.get("embedding_dim"),
            output_dim=cfg.get("output_dim", None),
        )
    
    else:
        raise ValueError(
            f"Unknown task encoder type: {encoder_type}. "
            f"Supported types: 'embedding'"
        )
