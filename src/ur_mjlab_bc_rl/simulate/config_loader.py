"""仿真参数加载工具。

从 configs/simulation/ 下的 YAML 文件加载配置。
回退到 configs/imitation/ 以兼容旧路径。
"""

from __future__ import annotations

from pathlib import Path

import yaml


def _find_config_dir() -> Path:
    """自动查找配置目录，优先 simulation/，回退 imitation/。"""
    project_root = Path(__file__).resolve().parents[3]
    for sub in ("simulation", "imitation"):
        d = project_root / "configs" / sub
        if (d / "default.yaml").exists():
            return d
    return project_root / "configs" / "simulation"


_CONFIG_DIR = _find_config_dir()


def _load(filename: str) -> dict:
    path = _CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


# ── 仿真参数 ───────────────────────────────────────────

def load_default() -> dict:
    """加载默认仿真参数（default.yaml）。"""
    return _load("default.yaml")


def load_tasks() -> dict:
    """加载任务配置（tasks.yaml）。"""
    return _load("tasks.yaml")


def load_task(task_name: str) -> dict | None:
    """获取指定任务配置。"""
    return load_tasks().get(task_name)


def get_sim_params() -> dict:
    """获取仿真频率参数。"""
    return load_default().get("sim", {})


def get_arm_joints() -> list[str]:
    """获取机械臂关节名称列表。"""
    return load_default().get("robot", {}).get(
        "arm_joints",
        ["ur_shoulder_pan_joint", "ur_shoulder_lift_joint",
         "ur_elbow_joint", "ur_wrist_1_joint",
         "ur_wrist_2_joint", "ur_wrist_3_joint"],
    )


def get_gripper_joints() -> list[str]:
    """获取夹爪关节名称列表。"""
    return load_default().get("robot", {}).get(
        "gripper_joints",
        ["robotiq_85_left_knuckle_joint"],
    )


def get_default_qpos() -> list[float]:
    """获取默认机械臂关节角。"""
    return load_default().get("robot", {}).get(
        "default_qpos",
        [0.0, -1.32, 1.32, -1.57, -1.57, 0.0],
    )


def get_camera_name() -> str:
    """获取相机名称。"""
    return load_default().get("camera", {}).get(
        "name", "realsense_link_CAMERA"
    )


def get_image_size() -> tuple[int, int]:
    """获取图像尺寸 (H, W)。"""
    s = load_default().get("camera", {}).get("image_size", [240, 320])
    return tuple(s)


def get_collection_params() -> dict:
    """获取采集参数。"""
    return load_default().get("collection", {})


def get_scene_dir() -> str:
    """获取场景文件目录。"""
    return load_default().get("scene_dir", "assets/mujoco/scenes")
