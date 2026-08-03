"""仿真配置加载 — 迁自 simulate/config_loader.py。

从三层 YAML 加载：
  simulate_default.yaml → scenes/scene_*.yaml → tasks/tasks_*.yaml
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


_CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"


def _load(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════
# 数据类
# ═══════════════════════════════════════════════════════

@dataclass
class SimParams:
    physics_dt: float = 0.001
    policy_dt: float = 0.01


@dataclass
class CollectionParams:
    max_time: float = 30.0
    max_attempts: int = 3


@dataclass
class RobotConfig:
    """场景中一个机械臂实例的配置。"""
    name: str
    prefix: str
    arm_joints: list[str]
    gripper_joints: list[str]
    ee_site: str
    default_qpos: list[float]

    @property
    def prefixed_arm_joints(self) -> list[str]:
        return [f"{self.prefix}{j}" for j in self.arm_joints]

    @property
    def prefixed_gripper_joints(self) -> list[str]:
        return [f"{self.prefix}{j}" for j in self.gripper_joints]

    @property
    def prefixed_ee_site(self) -> str:
        return f"{self.prefix}{self.ee_site}"

    @property
    def n_arm_joints(self) -> int:
        return len(self.arm_joints)

    @property
    def n_gripper_joints(self) -> int:
        return len(self.gripper_joints)


@dataclass
class CameraConfig:
    """场景中一个相机的配置。"""
    name: str
    fps: int = 30
    image_size: tuple[int, int] = (320, 240)
    type: str = "rgb_depth"
    depth_range: tuple[float, float] = (0.1, 0.8)

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    @property
    def width(self) -> int:
        return self.image_size[0]

    @property
    def height(self) -> int:
        return self.image_size[1]


@dataclass
class ObjectRandomization:
    """物体随机化参数。"""
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]
    roll_range: tuple[float, float] = (0.0, 0.0)
    pitch_range: tuple[float, float] = (0.0, 0.0)
    yaw_range: tuple[float, float] = (0.0, 0.0)

    @staticmethod
    def from_dict(d: dict) -> "ObjectRandomization":
        pos = d.get("pos", d)
        euler = d.get("euler", {})
        return ObjectRandomization(
            x_range=tuple(pos.get("x_range", (0.0, 0.0))),
            y_range=tuple(pos.get("y_range", (0.0, 0.0))),
            z_range=tuple(pos.get("z_range", (0.0, 0.0))),
            roll_range=tuple(euler.get("roll_range", (0.0, 0.0))),
            pitch_range=tuple(euler.get("pitch_range", (0.0, 0.0))),
            yaw_range=tuple(euler.get("yaw_range", (0.0, 0.0))),
        )


@dataclass
class TaskConfig:
    name: str
    scene_file: str
    teacher: str
    task_id: int
    objects: dict[str, ObjectRandomization] = field(default_factory=dict)


@dataclass
class SceneConfig:
    """完整仿真配置。"""
    sim: SimParams
    collection: CollectionParams
    robots: list[RobotConfig]
    cameras: list[CameraConfig]
    task: TaskConfig

    @property
    def n_arms(self) -> int:
        return len(self.robots)

    @property
    def state_dim(self) -> int:
        d = sum(r.n_arm_joints + r.n_gripper_joints for r in self.robots)
        d += sum(r.n_arm_joints + r.n_gripper_joints for r in self.robots)
        return d

    @property
    def action_dim(self) -> int:
        return sum(r.n_arm_joints + r.n_gripper_joints for r in self.robots)

    def robot_by_prefix(self, prefix: str) -> RobotConfig:
        for r in self.robots:
            if r.prefix == prefix:
                return r
        raise KeyError(f"无 prefix={prefix!r} 的机器人")


# ═══════════════════════════════════════════════════════
# 加载入口
# ═══════════════════════════════════════════════════════

_SCENE_FILE_TO_YAML: dict[str, str] = {
    "pick_place": "scene_single",
    "push_t": "scene_single",
    "peg_in_slot": "scene_single",
    "dual_pick_place": "scene_dual",
}


def _find_task(task_name: str) -> dict:
    tasks_dir = _CONFIG_ROOT / "tasks"
    for f in sorted(tasks_dir.glob("tasks_*.yaml")):
        raw = _load(f)
        if task_name in raw:
            return raw
    raise KeyError(f"任务 {task_name!r} 未找到")


def get_task_list() -> list[str]:
    """获取所有可用任务名称。"""
    tasks: list[str] = []
    tasks_dir = _CONFIG_ROOT / "tasks"
    for f in sorted(tasks_dir.glob("tasks_*.yaml")):
        raw = _load(f)
        tasks.extend(raw.keys())
    return tasks


def load_scene_config(task_name: str) -> SceneConfig:
    """加载完整仿真配置。"""
    # 1. sim + collection
    sim_raw = _load(_CONFIG_ROOT / "simulate_default.yaml")
    sim = SimParams(**sim_raw.get("sim", {}))
    collection = CollectionParams(**sim_raw.get("collection", {}))

    # 2. task
    tasks_raw = _find_task(task_name)
    task_raw = tasks_raw[task_name]
    scene_file = task_raw["scene_file"]
    scene_basename = Path(scene_file).stem

    scene_key = _SCENE_FILE_TO_YAML.get(scene_basename, f"scene_{scene_basename}")
    scene_raw = _load(_CONFIG_ROOT / "scenes" / f"{scene_key}.yaml")

    # 3. robots
    robots = [
        RobotConfig(
            name=r["name"],
            prefix=r.get("prefix", ""),
            arm_joints=r["arm_joints"],
            gripper_joints=r.get("gripper_joints", []),
            ee_site=r.get("ee_site", "_tcp"),
            default_qpos=r.get("default_qpos", []),
        )
        for r in scene_raw.get("robot", [])
    ]

    # 4. cameras
    cameras = [
        CameraConfig(
            name=c["name"],
            fps=c.get("fps", 30),
            image_size=(c.get("image_size", [320, 240])[0],
                        c.get("image_size", [320, 240])[1]),
            type=c.get("type", "rgb_depth"),
            depth_range=tuple(c.get("depth_range", [0.1, 0.8])),
        )
        for c in scene_raw.get("camera", [])
    ]

    # 5. task objects
    obj_raw = task_raw.get("domain_randomization") or task_raw.get("objects", {})
    objects = {
        name: ObjectRandomization.from_dict(cfg)
        for name, cfg in obj_raw.items()
    }
    task = TaskConfig(
        name=task_name,
        scene_file=scene_file,
        teacher=task_raw["teacher"],
        task_id=task_raw.get("task_id", 0),
        objects=objects,
    )

    return SceneConfig(sim=sim, collection=collection,
                       robots=robots, cameras=cameras, task=task)
