"""LeRobot EnvConfig 注册 — UR5e 双臂 MuJoCo 环境。

使用 --env.type=ur5e_dual 在 lerobot-eval CLI 中使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.processor import (
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from .ur5e_dual_env import UR5eDualEnv  # 同包内仍可用相对导入


@EnvConfig.register_subclass("ur5e_dual")
@dataclass
class UR5eDualEnvConfig(EnvConfig):
    """UR5e 双臂 MuJoCo 环境配置。

    用法:
        lerobot-eval --env.type=ur5e_dual --env.task=dual_pick_place
    """

    task: str = "dual_pick_place"
    fps: int = 30

    # ── features: 使用 gym obs key，与 UR5eDualEnv._step_physics_and_collect() 输出一致 ──
    # 图像 shape 为 channel-first: (C, H, W)
    # 状态 shape 为 (R, D): R=3 高频采样, D=状态维度
    # LeRobot delta 堆叠 T=4 帧后 → [B, T, R, D] → BiMFT _build_policy_batch 直接使用
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
            "images.A_realsense_link_CAMERA.rgb": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 480, 640),
            ),
            "images.A_realsense_link_CAMERA.depth": PolicyFeature(
                type=FeatureType.VISUAL, shape=(1, 480, 640),
            ),
            "images.B_realsense_link_CAMERA.rgb": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 480, 640),
            ),
            "images.B_realsense_link_CAMERA.depth": PolicyFeature(
                type=FeatureType.VISUAL, shape=(1, 480, 640),
            ),
            "images.global_realsense_link_CAMERA.rgb": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 480, 640),
            ),
            "images.global_realsense_link_CAMERA.depth": PolicyFeature(
                type=FeatureType.VISUAL, shape=(1, 480, 640),
            ),
            # 高频状态: (R=3, D) — 每帧图像对应 3 个 90Hz 采样
            "state.joint.position": PolicyFeature(
                type=FeatureType.STATE, shape=(3, 14),
            ),
            "state.sensor.force": PolicyFeature(
                type=FeatureType.STATE, shape=(3, 6),
            ),
            "state.sensor.torque": PolicyFeature(
                type=FeatureType.STATE, shape=(3, 6),
            ),
        }
    )

    # ── features_map: gym obs key → LeRobot 标准 key ──
    # gym env 使用简短键，RenameObservationsProcessorStep 添加 observation. 前缀
    # env_to_policy_features 反向查找（LeRobot→policy），保持与 features 键一致
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "state.joint.position": "observation.state.joint.position",
            "state.sensor.force": "observation.state.sensor.force",
            "state.sensor.torque": "observation.state.sensor.torque",
            "images.A_realsense_link_CAMERA.rgb": "observation.images.A_realsense_link_CAMERA.rgb",
            "images.A_realsense_link_CAMERA.depth": "observation.images.A_realsense_link_CAMERA.depth",
            "images.B_realsense_link_CAMERA.rgb": "observation.images.B_realsense_link_CAMERA.rgb",
            "images.B_realsense_link_CAMERA.depth": "observation.images.B_realsense_link_CAMERA.depth",
            "images.global_realsense_link_CAMERA.rgb": "observation.images.global_realsense_link_CAMERA.rgb",
            "images.global_realsense_link_CAMERA.depth": "observation.images.global_realsense_link_CAMERA.depth",
        }
    )

    @property
    def gym_kwargs(self) -> dict:
        return {
            "task_name": self.task,
        }

    def create_envs(
        self,
        n_envs: int,
        use_async_envs: bool = False,
    ) -> dict[str, dict[int, gym.vector.VectorEnv]]:
        """创建 VectorEnv。

        单任务 env，返回 {suite: {0: vec_env}}。
        """
        env_cls = (
            gym.vector.AsyncVectorEnv
            if (use_async_envs and n_envs > 1)
            else gym.vector.SyncVectorEnv
        )

        def _make_one():
            return UR5eDualEnv(
                task_name=self.task,
                max_steps=getattr(self, "max_episode_steps", None),
            )

        vec = env_cls([_make_one for _ in range(n_envs)])
        return {self.type: {0: vec}}

    def get_env_processors(
        self,
    ) -> tuple[
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    ]:
        """返回 env preprocessor / postprocessor。

        preprocessor: 将 gym obs 的简洁 key 重命名为 LeRobot 标准 key
        postprocessor: 空（无需变换）
        """
        preprocessor = PolicyProcessorPipeline(
            steps=[
                RenameObservationsProcessorStep(rename_map=self.features_map),
            ]
        )
        postprocessor = PolicyProcessorPipeline(steps=[])
        return preprocessor, postprocessor
