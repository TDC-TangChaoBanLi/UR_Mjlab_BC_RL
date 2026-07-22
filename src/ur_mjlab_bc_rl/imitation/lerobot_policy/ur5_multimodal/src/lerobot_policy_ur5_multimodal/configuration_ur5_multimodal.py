"""UR5 Multimodal BC Policy 配置。

注册为 "ur5_multimodal" 类型，可通过 --policy.type=ur5_multimodal 使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.optim import AdamWConfig
from lerobot.optim import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("ur5_multimodal")
@dataclass
class UR5MultimodalConfig(PreTrainedConfig):
    """UR5 多模态 BC 策略配置。

    特点：
    - 单帧 RGBD 观测 + robot state + task embedding
    - 单步动作预测（非 action chunking）
    - 可用于单臂/双臂，单目/多目（通过 input_features 自适应）
    """

    # ── 训练超参数 ────────────────────────────────────
    optimizer_lr: float = 1e-3
    optimizer_weight_decay: float = 1e-6
    scheduler_warmup_steps: int = 100

    # ── 模型架构超参数 ───────────────────────────────
    model_cfg_path: str = "configs/model/multimodal.yaml"
    visual_encoder_type: str = "rescnn"
    fusion_type: str = "concat"
    policy_hidden_dims: list[int] = field(default_factory=lambda: [1024, 512, 128])
    policy_activation: str = "elu"
    loss_type: str = "l1"
    state_dropout: float = 0.0

    # ── 策略类型参数 ─────────────────────────────────
    n_obs_steps: int = 1
    horizon: int = 1
    n_action_steps: int = 1

    # ── 输入/输出特征（运行时填充）───────────────────
    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)
    normalization_mapping: dict[str, str] = field(default_factory=dict)
    device: str = "cuda"

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.horizon:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) 不能大于 horizon ({self.horizon})"
            )

    def validate_features(self) -> None:
        """验证输入/输出特征兼容性。"""
        required = {"observation.images.rgb", "observation.images.depth", "observation.state"}
        available = set(self.input_features.keys())
        missing = required - available
        if missing:
            raise ValueError(f"缺少必需的输入特征: {missing}. 可用: {available}.")
        if "action" not in self.output_features:
            raise ValueError("缺少必需的输出特征: 'action'")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig | None:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=0,
            peak_lr=self.optimizer_lr,
            decay_lr=self.optimizer_lr * 0.01,
        )

    @property
    def observation_delta_indices(self) -> list[int] | None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return [0]

    @property
    def reward_delta_indices(self) -> None:
        return None
