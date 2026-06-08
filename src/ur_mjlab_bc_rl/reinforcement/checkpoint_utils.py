"""BC Checkpoint 工具。

提供 BC checkpoint 的加载、检查和转换功能。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import numpy as np


def inspect_bc_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    """检查 BC checkpoint 内容。

    Args:
        checkpoint_path: checkpoint 文件路径

    Returns:
        包含 checkpoint 元信息的字典
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    info: dict[str, Any] = {
        "path": str(path),
        "file_size_mb": path.stat().st_size / (1024 * 1024),
        "keys": list(checkpoint.keys()) if isinstance(checkpoint, dict) else ["tensor"],
    }

    if isinstance(checkpoint, dict):
        if "actor_state_dict" in checkpoint:
            state = checkpoint["actor_state_dict"]
            info["num_params"] = sum(v.numel() for v in state.values())
            info["layer_names"] = list(state.keys())[:20]  # 前 20 层
            # 显示关键层形状
            info["key_shapes"] = {
                k: list(v.shape) for k, v in list(state.items())[:10]
            }

        if "model_cfg" in checkpoint:
            cfg = checkpoint["model_cfg"]
            if isinstance(cfg, dict):
                info["model_type"] = cfg.get("type", "unknown")
                info["model_cfg_keys"] = list(cfg.keys())[:20]

        if "train_losses" in checkpoint:
            losses = checkpoint["train_losses"]
            info["num_epochs_trained"] = len(losses)
            info["final_loss"] = float(losses[-1]) if losses else None

    return info


def load_bc_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    """加载 BC checkpoint。

    Args:
        checkpoint_path: checkpoint 文件路径

    Returns:
        checkpoint 字典，包含：
        - "actor_state_dict": OrderedDict of weights
        - "model_cfg": model configuration dict (if available)
        - "optimizer_state_dict": optimizer state (if available)
        - "train_losses": training loss history (if available)
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected dict checkpoint, got {type(checkpoint)}")

    if "actor_state_dict" not in checkpoint:
        # 可能是完整模型保存，尝试提取
        if hasattr(checkpoint, "state_dict"):
            checkpoint = {"actor_state_dict": checkpoint.state_dict()}
        else:
            raise ValueError("Checkpoint does not contain 'actor_state_dict'")

    return checkpoint


def print_bc_checkpoint_info(checkpoint_path: str | Path) -> None:
    """打印 BC checkpoint 信息。"""
    info = inspect_bc_checkpoint(checkpoint_path)

    print(f"\n{'=' * 60}")
    print(f"BC Checkpoint: {Path(info['path']).name}")
    print(f"{'=' * 60}")
    print(f"  文件大小: {info['file_size_mb']:.2f} MB")
    print(f"  键: {info['keys']}")

    if "num_params" in info:
        print(f"  参数数量: {info['num_params']:,}")
    if "model_type" in info:
        print(f"  模型类型: {info['model_type']}")
    if "num_epochs_trained" in info:
        print(f"  训练 epoch 数: {info['num_epochs_trained']}")
    if "final_loss" in info and info["final_loss"] is not None:
        print(f"  最终损失: {info['final_loss']:.6f}")

    if "key_shapes" in info:
        print(f"\n  关键层形状:")
        for name, shape in info["key_shapes"].items():
            print(f"    {name}: {shape}")

    print(f"{'=' * 60}\n")


# CLI entry point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python -m ur_mjlab_bc_rl.reinforcement.checkpoint_utils <checkpoint_path>")
        sys.exit(1)

    print_bc_checkpoint_info(sys.argv[1])
