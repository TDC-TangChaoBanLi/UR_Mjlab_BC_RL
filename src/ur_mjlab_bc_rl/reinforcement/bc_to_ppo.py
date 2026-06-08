"""BC → PPO 桥接。

将 BC 预训练的 UR5MultimodalActor 权重
转换为 MjLab/RSL-RL PPO 兼容格式。

架构差异说明：
- BC Actor: 多模态编码器（视觉+状态+任务） → 融合 → MLP → 动作头
- RSL-RL PPO: 简单 MLP (obs → hidden → action_mean + value)

转换策略：
1. 特征提取模式：使用 BC Actor 的编码器+融合部分作为固定特征提取器
2. 权重迁移模式：提取 BC policy_mlp 权重映射到 PPO actor MLP
3. 从零训练模式：仅使用 BC checkpoint 的 normalizer 统计信息

当前实现：保存 BC 模型权重供 PPO 训练脚本使用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


def extract_policy_mlp_weights(
    bc_checkpoint: dict, output_path: Optional[str | Path] = None
) -> dict[str, torch.Tensor]:
    """从 BC checkpoint 中提取 policy MLP 权重。

    提取的权重可用于初始化 RSL-RL 的 actor MLP。

    Args:
        bc_checkpoint: BC checkpoint 字典
        output_path: 可选的保存路径

    Returns:
        提取的 policy MLP 权重字典
    """
    state_dict = bc_checkpoint["actor_state_dict"]
    policy_weights = {}

    for key, value in state_dict.items():
        # 提取 policy_mlp 相关的权重
        if "policy_mlp" in key:
            # 重命名：policy_mlp.net.0.weight → mlp.0.weight
            new_key = key.replace("policy_mlp.", "")
            policy_weights[new_key] = value

    if output_path is not None:
        torch.save({"policy_mlp_weights": policy_weights}, output_path)
        print(f"✓ Policy MLP 权重保存到: {output_path}")

    return policy_weights


def extract_encoder_weights(
    bc_checkpoint: dict, output_path: Optional[str | Path] = None
) -> dict[str, torch.Tensor]:
    """从 BC checkpoint 中提取所有编码器权重。

    Args:
        bc_checkpoint: BC checkpoint 字典
        output_path: 可选的保存路径

    Returns:
        编码器权重字典
    """
    state_dict = bc_checkpoint["actor_state_dict"]
    encoder_weights = {}

    encoder_prefixes = [
        "visual_encoder", "state_encoder", "task_encoder", "fusion",
    ]

    for key, value in state_dict.items():
        for prefix in encoder_prefixes:
            if key.startswith(prefix):
                encoder_weights[key] = value
                break

    if output_path is not None:
        torch.save({"encoder_weights": encoder_weights}, output_path)
        print(f"✓ 编码器权重保存到: {output_path}")

    return encoder_weights


def create_ppo_init_checkpoint(
    bc_checkpoint_path: str | Path,
    output_path: str | Path,
    extract_mode: str = "all",
) -> None:
    """从 BC checkpoint 创建 PPO 初始化 checkpoint。

    Args:
        bc_checkpoint_path: BC checkpoint 路径
        output_path: 输出路径
        extract_mode: 提取模式
            - "policy_only": 仅提取 policy MLP 权重
            - "encoder_only": 仅提取编码器权重
            - "all": 提取所有权重
    """
    from .checkpoint_utils import load_bc_checkpoint

    bc_ckpt = load_bc_checkpoint(bc_checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ppo_init = {
        "source": str(bc_checkpoint_path),
        "model_cfg": bc_ckpt.get("model_cfg", {}),
    }

    if extract_mode in ("policy_only", "all"):
        policy_weights = extract_policy_mlp_weights(bc_ckpt)
        ppo_init["policy_mlp_weights"] = policy_weights

    if extract_mode in ("encoder_only", "all"):
        encoder_weights = extract_encoder_weights(bc_ckpt)
        ppo_init["encoder_weights"] = encoder_weights

    torch.save(ppo_init, output_path)
    print(f"\n✓ PPO 初始化 checkpoint 保存到: {output_path}")
    print(f"  提取模式: {extract_mode}")
    print(f"  Policy MLP 层数: {len(ppo_init.get('policy_mlp_weights', {}))}")
    print(f"  编码器层数: {len(ppo_init.get('encoder_weights', {}))}")
