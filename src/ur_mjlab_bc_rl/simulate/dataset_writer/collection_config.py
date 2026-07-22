"""数据采集配置。

从 YAML 文件加载全局仿真参数和任务配置。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ── 子配置 ─────────────────────────────────────────────

@dataclass
class SimConfig:
    physics_dt: float = 0.001       # 物理仿真步长 (1000 Hz)
    policy_dt: float = 0.01         # 策略执行间隔 (100 Hz)
    camera_dt: float = 0.033333     # 相机帧间隔 (~30 Hz)


@dataclass
class RobotConfig:
    arm_joints: list[str] = field(default_factory=lambda: [
        "ur_shoulder_pan_joint", "ur_shoulder_lift_joint",
        "ur_elbow_joint", "ur_wrist_1_joint",
        "ur_wrist_2_joint", "ur_wrist_3_joint",
    ])
    gripper_joints: list[str] = field(default_factory=lambda: [
        "robotiq_85_left_knuckle_joint",
    ])
    ee_site: str = "_tcp"
    default_qpos: list[float] = field(default_factory=lambda: [
        0.0, -1.32, 1.32, -1.57, -1.57, 0.0,
    ])
    action_dim: int = 7


@dataclass
class CameraConfig:
    name: str = "realsense_link_CAMERA"
    image_height: int = 240
    image_width: int = 320

    @property
    def image_size(self) -> tuple[int, int]:
        return (self.image_height, self.image_width)


@dataclass
class CollectionParams:
    max_steps: int = 30000
    max_attempts: int = 3


@dataclass
class TaskConfig:
    scene: str = ""
    teacher: str = ""
    task_id: int = 0
    depth_range: tuple[float, float] = (0.1, 0.8)
    objects: dict[str, Any] = field(default_factory=dict)


# ── 顶层配置 ───────────────────────────────────────────

@dataclass
class CollectionConfig:
    sim: SimConfig = field(default_factory=SimConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    collection: CollectionParams = field(default_factory=CollectionParams)
    tasks: dict[str, TaskConfig] = field(default_factory=dict)

    # configs 目录（运行时设置）
    scene_dir: str = "assets/mujoco/scenes"

    @property
    def state_dim(self) -> int:
        """默认 state 维度: arm(6) + gripper(1) + last_action(7) = 14."""
        return (
            len(self.robot.arm_joints)
            + len(self.robot.gripper_joints)
            + self.robot.action_dim
        )

    # ── 工厂方法 ────────────────────────────────────

    @classmethod
    def from_yaml(
        cls, config_dir: str | Path | None = None
    ) -> "CollectionConfig":
        """从 YAML 配置文件加载。

        默认加载 configs/simulation/default.yaml + configs/simulation/tasks.yaml。
        回退到 configs/imitation/。
        """
        if config_dir is None:
            candidates = []
            project_root = Path(__file__).resolve().parents[4]
            for sub in ("simulation", "imitation"):
                candidates.append(project_root / "configs" / sub)
            for c in candidates:
                if (c / "default.yaml").exists():
                    config_dir = c
                    break

        if config_dir is None:
            return cls()

        config_dir = Path(config_dir)
        cfg = cls()

        # 加载 default.yaml
        default_path = config_dir / "default.yaml"
        if default_path.exists():
            with open(default_path) as f:
                raw = yaml.safe_load(f) or {}

            if "sim" in raw:
                cfg.sim = SimConfig(**raw["sim"])
            if "robot" in raw:
                r = raw["robot"]
                cfg.robot = RobotConfig(
                    arm_joints=r.get("arm_joints", cfg.robot.arm_joints),
                    gripper_joints=r.get("gripper_joints", cfg.robot.gripper_joints),
                    ee_site=r.get("ee_site", cfg.robot.ee_site),
                    default_qpos=r.get("default_qpos", cfg.robot.default_qpos),
                )
            if "camera" in raw:
                c = raw["camera"]
                sz = c.get("image_size", [240, 320])
                cfg.camera = CameraConfig(
                    name=c.get("name", cfg.camera.name),
                    image_height=int(sz[0]),
                    image_width=int(sz[1]),
                )
            if "collection" in raw:
                col = raw["collection"]
                cfg.collection = CollectionParams(
                    max_steps=col.get("max_steps", cfg.collection.max_steps),
                    max_attempts=col.get("max_attempts", cfg.collection.max_attempts),
                )
            if "scene_dir" in raw:
                cfg.scene_dir = raw["scene_dir"]

        # 加载 tasks.yaml
        tasks_path = config_dir / "tasks.yaml"
        if tasks_path.exists():
            with open(tasks_path) as f:
                raw_tasks = yaml.safe_load(f) or {}
            for name, t in raw_tasks.items():
                cfg.tasks[name] = TaskConfig(
                    scene=str(t.get("scene", "")),
                    teacher=str(t.get("teacher", "")),
                    task_id=int(t.get("task_id", 0)),
                    depth_range=tuple(t.get("depth_range", [0.1, 0.8])),
                    objects=t.get("objects", {}),
                )

        return cfg

    def get_task(self, name: str) -> TaskConfig:
        if name not in self.tasks:
            raise KeyError(f"未知任务: {name}. 已知: {list(self.tasks)}")
        return self.tasks[name]

    def get_scene_path(self, task_name: str) -> Path:
        task = self.get_task(task_name)
        return Path(self.scene_dir) / task.scene
