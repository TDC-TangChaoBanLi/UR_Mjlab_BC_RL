"""MuJoCo 环境重置管理器。

负责各任务的初始化和物体位姿随机化。
通过 MujocoInterface 访问 MuJoCo。
参数从 configs/imitation/ 加载。
"""

from __future__ import annotations

import numpy as np

from .mujoco_interface import MujocoInterface
from ...config_loader import get_default_qpos, load_tasks


def _load_task_objects() -> dict[str, dict]:
    """从配置文件加载任务物体随机化参数（缓存）。"""
    tasks = load_tasks()
    return {
        name: cfg["objects"]
        for name, cfg in tasks.items()
    }


TASK_OBJECTS = _load_task_objects()


class ResetManager:
    """环境重置管理器。

    通过 MujocoInterface 操作仿真状态，负责：
    - 重置机械臂到指定初始关节角
    - 随机化操作物体的位姿
    - 同步 actuator 控制信号
    """

    def __init__(
        self,
        mj_interface: MujocoInterface,
        arm_joint_names: list[str] | None = None,
        gripper_joint_names: list[str] | None = None,
    ) -> None:
        """初始化重置管理器。

        Args:
            mj_interface: MuJoCo 仿真接口。
            arm_joint_names: 机械臂关节名称列表（6个）。
            gripper_joint_names: 夹爪关节名称列表。
        """
        self.mj = mj_interface

        if arm_joint_names is None:
            arm_joint_names = [
                "ur_shoulder_pan_joint", "ur_shoulder_lift_joint",
                "ur_elbow_joint", "ur_wrist_1_joint",
                "ur_wrist_2_joint", "ur_wrist_3_joint",
            ]
        if gripper_joint_names is None:
            gripper_joint_names = ["robotiq_85_left_knuckle_joint"]

        self.arm_joint_names = arm_joint_names
        self.gripper_joint_names = gripper_joint_names

        # 预计算 actuator ID
        self._arm_actuator_ids = [
            self.mj.get_actuator_id(n + "_ACTUATOR") for n in arm_joint_names
        ]
        self._gripper_actuator_ids = [
            self.mj.get_actuator_id(n + "_ACTUATOR") for n in gripper_joint_names
        ]

    # ── 重置入口 ───────────────────────────────────────────

    def reset(
        self,
        task: str = "pick_place",
        arm_qpos: np.ndarray | None = None,
        randomize_objects: bool = True,
    ) -> None:
        """完整的重置流程。

        1. 调用 mj_resetData 重置物理状态
        2. 设置机械臂初始关节角
        3. 随机化物体位姿
        4. 同步 actuator ctrl 到初始 qpos
        5. 执行 mj_forward 更新运动学

        Args:
            task: 任务名 ("pick_place", "push_t", "peg_slot")。
            arm_qpos: 自定义臂关节角 [6]，None 则使用 DEFAULT_ARM_QPOS。
            randomize_objects: 是否随机化物体位姿。
        """
        self.mj.reset()

        if arm_qpos is None:
            arm_qpos = np.array(get_default_qpos())
        self._set_arm_qpos(arm_qpos)
        self._set_gripper_qpos(0.0)

        if randomize_objects and task in TASK_OBJECTS:
            self._randomize_objects(TASK_OBJECTS[task])

        self.mj.forward()
        self._sync_ctrl_from_qpos()

    # ── 关节设置 ──────────────────────────────────────────

    def _set_arm_qpos(self, qpos: np.ndarray) -> None:
        """设置臂关节位置。"""
        for i, name in enumerate(self.arm_joint_names):
            self.mj.set_joint_qpos(name, qpos[i])

    def _set_gripper_qpos(self, val: float) -> None:
        """设置夹爪关节位置。"""
        for name in self.gripper_joint_names:
            self.mj.set_joint_qpos(name, val)

    def _sync_ctrl_from_qpos(self) -> None:
        """将 actuator ctrl 同步到当前 qpos。"""
        for act_id in self._arm_actuator_ids + self._gripper_actuator_ids:
            jname = self.mj.model.actuator(act_id).name.replace("_ACTUATOR", "")
            adr = self.mj.get_joint_qposadr(jname)
            if adr >= 0:
                self.mj.data.ctrl[act_id] = self.mj.data.qpos[adr]

    # ── 物体随机化 ────────────────────────────────────────

    def _randomize_objects(self, objects: dict[str, dict]) -> None:
        """随机化一组物体的位姿。

        Args:
            objects: {物体名: {x_range, y_range, z}} 字典。
        """
        for obj_name, params in objects.items():
            self._randomize_single_object(obj_name, **params)

    def _randomize_single_object(
        self,
        object_name: str,
        x_range: tuple[float, float] = (0.35, 0.55),
        y_range: tuple[float, float] = (-0.20, 0.20),
        z: float = 0.65,
        yaw_range: tuple[float, float] = (-np.pi, np.pi),
    ) -> None:
        """随机化单个物体的位姿。

        通过修改物体对应 freejoint 的 qpos 实现。

        Args:
            object_name: 物体 body 名称。
            x_range: x 坐标范围。
            y_range: y 坐标范围。
            z: z 坐标（固定高度）。
            yaw_range: 偏航角范围（绕 z 轴旋转）。
        """
        jnt_id = self.mj.get_body_joint_id(object_name)
        if jnt_id is None:
            return  # 物体不存在则静默跳过

        qpos_addr = self.mj.model.jnt_qposadr[jnt_id]
        if qpos_addr < 0:
            return

        # 生成随机位姿
        pos_x = float(np.random.uniform(*x_range))
        pos_y = float(np.random.uniform(*y_range))
        yaw = float(np.random.uniform(*yaw_range))
        qw = np.cos(yaw / 2.0)
        qz = np.sin(yaw / 2.0)

        # 写入 qpos: [x, y, z, qw, qx, qy, qz]
        self.mj.data.qpos[qpos_addr + 0] = pos_x
        self.mj.data.qpos[qpos_addr + 1] = pos_y
        self.mj.data.qpos[qpos_addr + 2] = z
        self.mj.data.qpos[qpos_addr + 3] = qw
        self.mj.data.qpos[qpos_addr + 4] = 0.0
        self.mj.data.qpos[qpos_addr + 5] = 0.0
        self.mj.data.qpos[qpos_addr + 6] = qz

    # ── 兼容旧接口 ────────────────────────────────────────

    def reset_to_default(self) -> None:
        """重置到默认姿态（兼容旧接口）。"""
        self.reset(task="pick_place", randomize_objects=False)

    def randomize_object_pose(
        self,
        object_name: str,
        x_range: tuple[float, float] = (0.35, 0.55),
        y_range: tuple[float, float] = (-0.20, 0.20),
        z: float = 0.65,
        yaw_range: tuple[float, float] = (-np.pi, np.pi),
    ) -> None:
        """随机化单个物体位姿（兼容旧接口）。"""
        self._randomize_single_object(object_name, x_range, y_range, z, yaw_range)
        self.mj.forward()
