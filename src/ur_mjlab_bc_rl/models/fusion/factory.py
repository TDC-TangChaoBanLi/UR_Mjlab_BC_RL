"""融合模块工厂。

根据配置构建不同类型的融合模块。
"""

from __future__ import annotations

from typing import Any

from .base import FusionBase
from .concat import ConcatFusion
from .film import FiLMFusion


def build_fusion(cfg: dict[str, Any], visual_dim: int, state_dim: int, task_dim: int) -> FusionBase:
    """根据配置构建融合模块。
    
    Args:
        cfg: 融合配置字典，必须包含 'type' 字段
        visual_dim: 视觉特征维度
        state_dim: 状态特征维度
        task_dim: 任务特征维度
        
    Returns:
        融合模块实例
        
    Raises:
        ValueError: 如果类型不支持
    """
    fusion_type = cfg.get("type", "concat")
    
    if fusion_type == "concat":
        output_dim = visual_dim + state_dim + task_dim
        fusion = ConcatFusion(output_dim=output_dim)
        return fusion
    
    elif fusion_type == "film":
        return FiLMFusion(
            visual_dim=visual_dim,
            state_dim=state_dim,
            task_dim=task_dim,
            hidden_dims=cfg.get("hidden_dims", [128]),
            use_residual_gamma=cfg.get("use_residual_gamma", True),
        )
    
    elif fusion_type == "self_attention":
        raise NotImplementedError("SelfAttentionFusion is not yet implemented")
    
    elif fusion_type == "cross_attention":
        raise NotImplementedError("CrossAttentionFusion is not yet implemented")
    
    elif fusion_type == "policy_token":
        raise NotImplementedError("PolicyTokenFusion is not yet implemented")
    
    else:
        raise ValueError(
            f"Unknown fusion type: {fusion_type}. "
            f"Supported types: 'concat', 'film', 'self_attention', 'cross_attention', 'policy_token'"
        )
