"""多臂多物体环境重置管理器。

负责：
- 将各臂重置到 default_qpos
- 随机化物体位姿（支持 x/y/z 范围）
- 同步 actuator 控制信号
"""

from __future__ import annotations

import mujoco
import numpy as np

from ..utils.config_loader import RobotConfig, ObjectRandomization
from ..simulate.mujoco_wrapper import MujocoWrapper as MujocoInterface


class ResetManager:
    """多臂多物体重置管理器。"""

    def __init__(
        self,
        mj: MujocoInterface,
        robots: list[RobotConfig],
        objects: dict[str, ObjectRandomization],
    ) -> None:
        self.mj = mj
        self.robots = robots
        self.objects = objects

        # 预计算各臂 actuator ID
        self._actuator_ids: dict[str, tuple[list[int], list[int]]] = {}
        for r in robots:
            arm_ids = [mj.get_actuator_id(f"{j}_ACTUATOR")
                       for j in r.prefixed_arm_joints]
            grip_ids = [mj.get_actuator_id(f"{j}_ACTUATOR")
                        for j in r.prefixed_gripper_joints]
            self._actuator_ids[r.prefix] = (arm_ids, grip_ids)

    def reset(self, *, randomize_objects: bool = True) -> None:
        """完整重置流程。

        1. mj_resetData
        2. 各臂设 default_qpos
        3. 随机化物体位姿
        4. 同步 actuator ctrl
        5. mj_forward
        """
        mujoco.mj_resetData(self.mj.model, self.mj.data)

        ctrl = self.mj.get_ctrl()
        for r in self.robots:
            arm_ids, grip_ids = self._actuator_ids[r.prefix]
            ctrl[arm_ids] = np.asarray(r.default_qpos, dtype=np.float64)
            ctrl[grip_ids] = 0.0
            self.mj.set_ctrl(ctrl)
            # 设 qpos 使 arm 初始到位
            for i, jname in enumerate(r.prefixed_arm_joints):
                self.mj.set_joint_qpos(jname, r.default_qpos[i])

        if randomize_objects:
            self._randomize_objects()

        mujoco.mj_forward(self.mj.model, self.mj.data)

    def _randomize_objects(self) -> None:
        for obj_name, rand in self.objects.items():
            jnt_id = self.mj.get_body_joint_id(obj_name)
            if jnt_id is None:
                continue

            adr = self.mj.model.jnt_qposadr[jnt_id]
            # 位置随机化
            x = np.random.uniform(*rand.x_range)
            y = np.random.uniform(*rand.y_range)
            z = np.random.uniform(*rand.z_range)
            self.mj.data.qpos[adr:adr + 3] = [x, y, z]

            # 姿态随机化（euler → quaternion）
            has_rot = (
                rand.roll_range != (0.0, 0.0)
                or rand.pitch_range != (0.0, 0.0)
                or rand.yaw_range != (0.0, 0.0)
            )
            if has_rot:
                roll = np.random.uniform(*rand.roll_range)
                pitch = np.random.uniform(*rand.pitch_range)
                yaw = np.random.uniform(*rand.yaw_range)
                quat = np.zeros(4)
                mujoco.mju_euler2Quat(quat, [roll, pitch, yaw], "XYZ")
                self.mj.data.qpos[adr + 3:adr + 7] = quat  # [qw, qx, qy, qz]
