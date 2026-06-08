"""视觉编码器工厂。

根据配置构建不同类型的视觉编码器。
"""

from __future__ import annotations

from typing import Any

from .base import VisualEncoderBase
from .rgbd_cnn import RGBDCNNEncoder
from .rgbd_vit import RGBDViT


def build_visual_encoder(cfg: dict[str, Any]) -> VisualEncoderBase:
    """根据配置构建视觉编码器。
    
    Args:
        cfg: 配置字典，必须包含 'type' 字段
        
    Returns:
        视觉编码器实例
        
    Raises:
        ValueError: 如果类型不支持或配置无效
    """
    encoder_type = cfg.get("type", "rgbd_cnn")
    
    if encoder_type == "rgbd_cnn":
        return RGBDCNNEncoder(
            in_channels=cfg.get("in_channels", 4),
            image_size=tuple(cfg.get("image_size", [128, 128])),
            hidden_dims=cfg.get("hidden_dims"),
            output_dim=cfg.get("output_dim", 256),
            kernel_size=cfg.get("kernel_size", 3),
            pool_size=cfg.get("pool_size", 2),
            # 残差模式参数
            stem_cfg=cfg.get("stem"),
            stages=cfg.get("stages"),
            block_cfg={
                k: cfg[k]
                for k in ("kernel_size", "activation", "norm", "dropout")
                if k in cfg
            } or None,
            head_cfg=cfg.get("head"),
        )
    
    elif encoder_type == "rgbd_vit":
        return RGBDViT(
            image_size=tuple(cfg.get("image_size", [128, 128])),
            patch_size=cfg.get("patch_size", 16),
            in_channels=cfg.get("in_channels", 4),
            embed_dim=cfg.get("embed_dim", 256),
            depth=cfg.get("depth", 4),
            num_heads=cfg.get("num_heads", 4),
            mlp_ratio=cfg.get("mlp_ratio", 4.0),
            dropout=cfg.get("dropout", 0.1),
            output_dim=cfg.get("output_dim", 256),
        )
    
    else:
        raise ValueError(
            f"Unknown visual encoder type: {encoder_type}. "
            f"Supported types: 'rgbd_cnn', 'rgbd_vit'"
        )