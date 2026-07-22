"""配置加载工具 — 模型配置参数加载。

仿真相关配置已迁移至 ur_mjlab_bc_rl.simulate.config_loader。
"""

from __future__ import annotations

from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _load(path: str) -> dict:
    if not (_CONFIG_DIR / Path(path)).exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(_CONFIG_DIR / Path(path)) as f:
        return yaml.safe_load(f)


# ── 模型 ────────────────────────────────────────────────

def load_multimodal_model() -> dict:
    """加载共享的多模态模型架构（BC 和 PPO 共用）。"""
    return _load("model/multimodal.yaml")


def load_imitation_config() -> dict:
    """加载模仿学习训练配置（引用 multimodal.yaml 的模型架构）。"""
    return _load("model/imitation.yaml")


def load_reinforcement_config() -> dict:
    """加载强化学习训练配置（引用 multimodal.yaml 的模型架构）。"""
    return _load("model/reinforcement.yaml")


def load_aloha_act_model() -> dict:
    """加载 ALOHA ACT 模型配置。"""
    return _load("model/aloha_act.yaml")


# ── 配置打印工具 ──────────────────────────────────────────


def print_model_config(cfg: dict) -> None:
    """打印多模态模型配置的详细信息（通用遍历方式）。

    Args:
        cfg: 模型配置字典（来自 load_multimodal_model）
    """
    print(f"\n{'='*60}")
    print("模型配置详情")
    print(f"{'='*60}")

    # 模块名称映射（美化输出）
    module_names = {
        "visual_encoder": "视觉编码器",
        "state_encoder": "状态编码器",
        "task_encoder": "任务编码器",
        "fusion": "融合模块",
        "policy_mlp": "策略 MLP",
        "action_dim": "动作配置",
    }

    for key, value in cfg.items():
        # 获取模块名称
        name = module_names.get(key, key.replace("_", " ").title())
        
        # 打印模块标题
        print(f"\n[{name}]")
        
        # 递归打印配置项
        _print_config_item(value, indent=2)

    print(f"\n{'='*60}")


def _print_config_item(value, indent: int = 0) -> None:
    """递归打印配置项。"""
    prefix = " " * indent
    
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                print(f"{prefix}{k}:")
                _print_config_item(v, indent + 2)
            else:
                print(f"{prefix}{k}: {v}")
    elif isinstance(value, list):
        # 对于列表，打印为单行
        print(f"{prefix}{value}")
    else:
        print(f"{prefix}{value}")