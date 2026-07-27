"""BiMFT Bimanual Multimodal Fusion Transformer Policy — LeRobot 配置。

注册为 "bimft" 类型，通过 --policy.type=bimft 使用。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from lerobot.configs import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.optim import AdamWConfig
from lerobot.optim import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("bimft")
@dataclass
class BiMFTConfig(PreTrainedConfig):
    """BiMFT 双臂多模态融合 Transformer 策略配置。

    多相机时序输入:
      - 左腕 / 右腕 / 全局 RGB-D, 4 帧 @ 30 Hz
      - 左右关节+夹爪, 12 采样 @ 90 Hz
      - 左右末端力/力矩, 12 采样 @ 90 Hz

    输出: 左右各 K=30 × 7 维动作块 (action chunking)
    """

    # ── 训练超参数 ────────────────────────────────────
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4
    scheduler_warmup_steps: int = 500

    # ── 混合精度（节省显存 ~40%） ─────────────────────
    use_amp: bool = True

    # ── 模型配置路径 ─────────────────────────────────
    model_cfg_path: str = "configs/model/bimft.yaml"

    # ── 损失类型 ──────────────────────────────────────
    loss_type: str = "huber"

    # ── 策略类型参数 ─────────────────────────────────
    # n_obs_steps: 视觉帧数 (4 = 4 帧 RGB-D)
    # horizon:     动作块长度 (30)
    # n_action_steps: 每次执行的动作步数 (3 = 前 3 个 @ 90Hz)
    n_obs_steps: int = 4
    horizon: int = 30
    n_action_steps: int = 3

    # ── 输入/输出特征（运行时由 LeRobot 填充）─────────
    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)
    normalization_mapping: dict[str, str] = field(default_factory=dict)
    device: str = "cuda"

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.horizon:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) 大于 horizon ({self.horizon})"
            )

    def validate_features(self) -> None:
        """验证输入/输出特征兼容性。

        仅在 features 已填充时验证；LeRobot 在构造时可能尚未填充。
        """
        if not self.input_features and not self.output_features:
            return  # 尚未填充，跳过验证

        available = set(self.input_features.keys())

        state_ok = "observation.state" in available
        joint_ok = "observation.state.joint.position" in available
        if not (state_ok or joint_ok):
            logging.warning(
                f"缺少 observation.state 或 observation.state.joint.position。"
                f"可用特征: {sorted(available)}"
            )

        if "action" not in self.output_features and "action.joint.position" not in self.output_features:
            logging.warning("缺少 output feature 'action'")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            betas=(0.9, 0.95),
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig | None:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=0,
            peak_lr=self.optimizer_lr,
            decay_lr=self.optimizer_lr * 0.01,
        )

    @property
    def observation_delta_indices(self) -> list[int] | None:
        """返回观测时序偏移: [-3, -2, -1, 0] 表示取过去 3 帧 + 当前帧。

        对应 n_obs_steps=4: LeRobot dataloader 会堆叠 4 个连续帧。
        """
        return list(range(-(self.n_obs_steps - 1), 1))  # [-3, -2, -1, 0]

    @property
    def action_delta_indices(self) -> list[int]:
        """10 行 × 每行 3 采样 = 30 步 action @ 90Hz."""
        return list(range(self.horizon // 3))  # [0..9] for horizon=30

    @property
    def reward_delta_indices(self) -> None:
        return None
