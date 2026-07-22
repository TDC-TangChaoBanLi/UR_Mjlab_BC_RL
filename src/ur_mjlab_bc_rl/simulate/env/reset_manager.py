"""MuJoCo 环境重置管理器。

负责各任务初始化和物体位姿随机化。
通过 MujocoInterface 操作仿真状态。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .mujoco_interface import MujocoInterface


# ── 默认参数（configs 不存在时使用）─────────────────────

_DEFAULT_ARM_QPOS = [0.0, -1.32, 1.32, -1.57, -1.57, 0.0]
_DEFAULT_ARM_JOINTS = [
    "ur_shoulder_pan_joint", "ur_shoulder_lift_joint",
    "ur_elbow_joint", "ur_wrist_1_joint",
    "ur_wrist_2_joint", "ur_wrist_3_joint",
]
_DEFAULT_GRIPPER_JOINTS = ["robotiq_85_left_knuckle_joint"]


def _load_tasks(config_dir: str | Path | None = None) -> dict[str, Any]:
    """从 YAML 配置文件加载任务物体随机化参数。

    优先从 configs/simulation/tasks.yaml 加载，
    回退到 configs/imitation/tasks.yaml。
    """
    if config_dir is None:
        # 搜索可能的配置目录
        candidates = []
        for root in (Path(__file__).resolve().parents[4],):
            candidates.append(root / "configs" / "simulation" / "tasks.yaml")
            candidates.append(root / "configs" / "imitation" / "tasks.yaml")
        for c in candidates:
            if c.exists():
                config_dir = c.parent
                break

    if config_dir is None:
        return {}

    config_dir = Path(config_dir)
    tasks_path = config_dir / "tasks.yaml"
    if not tasks_path.exists():
        return {}

    with open(tasks_path) as f:
        return yaml.safe_load(f) or {}


def _load_default(config_dir: str | Path | None = None) -> dict[str, Any]:
    """加载仿真默认参数。"""
    if config_dir is None:
        candidates = []
        for root in (Path(__file__).resolve().parents[4],):
            candidates.append(root / "configs" / "simulation" / "default.yaml")
            candidates.append(root / "configs" / "imitation" / "default.yaml")
        for c in candidates:
            if c.exists():
                config_dir = c.parent
                break

    if config_dir is None:
        return {}

    config_dir = Path(config_dir)
    default_path = config_dir / "default.yaml"
    if not default_path.exists():
        return {}

    with open(default_path) as f:
        return yaml.safe_load(f) or {}


class ResetManager:
    """环境重置管理器。

    通过 MujocoInterface 操作仿真状态：
    - 重置机械臂到指定初始关节角
    - 随机化操作物体的位姿
    - 同步 actuator 控制信号
    """

    def __init__(
        self,
        mj_interface: MujocoInterface,
        arm_joint_names: list[str] | None = None,
        gripper_joint_names: list[str] | None = None,
        *,
        config_dir: str | Path | None = None,
    ) -> None:
        self.mj = mj_interface

        if arm_joint_names is None:
            arm_joint_names = list(_DEFAULT_ARM_JOINTS)
        if gripper_joint_names is None:
            gripper_joint_names = list(_DEFAULT_GRIPPER_JOINTS)

        self.arm_joint_names = arm_joint_names
        self.gripper_joint_names = gripper_joint_names

        # 预计算 actuator ID
        self._arm_actuator_ids = [
            self.mj.get_actuator_id(n + "_ACTUATOR")
            for n in arm_joint_names
        ]
        self._gripper_actuator_ids = [
            self.mj.get_actuator_id(n + "_ACTUATOR")
            for n in gripper_joint_names
        ]

        # 加载默认参数
        default_cfg = _load_default(config_dir)
        self._default_qpos = np.array(
            default_cfg.get("robot", {}).get(
                "default_qpos", _DEFAULT_ARM_QPOS
            ),
            dtype=np.float64,
        )

        # 加载任务物体参数
        self._task_objects = {
            name: cfg.get("objects", {})
            for name, cfg in _load_tasks(config_dir).items()
        }

    # ── 重置入口 ───────────────────────────────────────

    def reset(
        self,
        task: str = "pick_place",
        arm_qpos: np.ndarray | None = None,
        randomize_objects: bool = True,
    ) -> None:
        """完整的重置流程。

        1. mj_resetData 重置物理状态
        2. 设置机械臂初始关节角
        3. 随机化物体位姿
        4. 同步 actuator ctrl
        5. mj_forward 更新运动学

        Args:
            task: 任务名 ("pick_place", "push_t", "peg_slot")
            arm_qpos: 自定义臂关节角 [6]，None 用默认值
            randomize_objects: 是否随机化物体位姿
        """
        self.mj.reset()

        if arm_qpos is None:
            arm_qpos = self._default_qpos.copy()
        self._set_arm_qpos(arm_qpos)
        self._set_gripper_qpos(0.0)

        if randomize_objects and task in self._task_objects:
            self._randomize_objects(self._task_objects[task])

        self.mj.forward()
        self._sync_ctrl_from_qpos()

    # ── 关节设置 ───────────────────────────────────────

    def _set_arm_qpos(self, qpos: np.ndarray) -> None:
        for i, name in enumerate(self.arm_joint_names):
            self.mj.set_joint_qpos(name, qpos[i])

    def _set_gripper_qpos(self, val: float) -> None:
        for name in self.gripper_joint_names:
            self.mj.set_joint_qpos(name, val)

    def _sync_ctrl_from_qpos(self) -> None:
        for act_id in self._arm_actuator_ids + self._gripper_actuator_ids:
            jname = (
                self.mj.model.actuator(act_id).name.replace("_ACTUATOR", "")
            )
            adr = self.mj.get_joint_qposadr(jname)
            if adr >= 0:
                self.mj.data.ctrl[act_id] = self.mj.data.qpos[adr]

    # ── 物体随机化 ────────────────────────────────────

    def _randomize_objects(
        self, objects: dict[str, dict[str, Any]]
    ) -> None:
        for obj_name, params in objects.items():
            x_range = params.get(
                "x_range", [0.35, 0.55]
            )
            y_range = params.get(
                "y_range", [-0.15, 0.15]
            )
            z = float(params.get("z", 0.65))
            self._randomize_single_object(
                obj_name,
                tuple(x_range),
                tuple(y_range),
                z,
            )

    def _randomize_single_object(
        self,
        object_name: str,
        x_range: tuple[float, float] = (0.35, 0.55),
        y_range: tuple[float, float] = (-0.20, 0.20),
        z: float = 0.65,
        yaw_range: tuple[float, float] = (-np.pi, np.pi),
    ) -> None:
        jnt_id = self.mj.get_body_joint_id(object_name)
        if jnt_id is None:
            return

        qpos_addr = self.mj.model.jnt_qposadr[jnt_id]
        if qpos_addr < 0:
            return

        pos_x = float(np.random.uniform(*x_range))
        pos_y = float(np.random.uniform(*y_range))
        yaw = float(np.random.uniform(*yaw_range))
        qw = np.cos(yaw / 2.0)
        qz = np.sin(yaw / 2.0)

        self.mj.data.qpos[qpos_addr + 0] = pos_x
        self.mj.data.qpos[qpos_addr + 1] = pos_y
        self.mj.data.qpos[qpos_addr + 2] = z
        self.mj.data.qpos[qpos_addr + 3] = qw
        self.mj.data.qpos[qpos_addr + 4] = 0.0
        self.mj.data.qpos[qpos_addr + 5] = 0.0
        self.mj.data.qpos[qpos_addr + 6] = qz
