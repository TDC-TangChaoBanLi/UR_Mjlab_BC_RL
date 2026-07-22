"""Scripted Teacher 基类。

所有 scripted teacher 都继承此类，提供状态机框架和工具方法。
输出绝对目标位姿 [x,y,z, qw,qx,qy,qz, gripper_cmd]，
由 IK 求解器转换为关节级控制命令。
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import mujoco


class TeacherState(Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


class Teacher:
    """Scripted teacher 基类。

    提供:
    - 状态机框架 (reset / step / is_done / is_success)
    - 末端/物体位姿查询
    - 位姿增量计算
    - 动作构建 (绝对位姿 + 夹爪)

    子类需要实现:
    - step() → np.ndarray  [8]: [x,y,z, qw,qx,qy,qz, gripper]
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self.state = TeacherState.RUNNING
        self.current_step = 0

    def reset(self) -> None:
        self.state = TeacherState.RUNNING
        self.current_step = 0

    def step(self) -> np.ndarray:
        raise NotImplementedError

    def is_success(self) -> bool:
        return self.state == TeacherState.SUCCESS

    def is_failure(self) -> bool:
        return self.state == TeacherState.FAILURE

    def is_done(self) -> bool:
        return self.state != TeacherState.RUNNING

    # ── 位姿查询 ───────────────────────────────────────

    def get_ee_pose(self) -> np.ndarray:
        """获取末端执行器位姿 [x,y,z, qw,qx,qy,qz]."""
        try:
            site_id = self.model.site("_tcp").id
            pos = self.data.site_xpos[site_id].copy()
            xmat = self.data.site_xmat[site_id].copy().reshape(3, 3)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, xmat.ravel())
            return np.concatenate([pos, quat])
        except Exception:
            return np.zeros(7)

    def get_object_pose(self, name: str) -> np.ndarray:
        """获取物体位姿 [x,y,z, qw,qx,qy,qz]."""
        try:
            body_id = self.model.body(name).id
            pos = self.data.xpos[body_id].copy()
            xmat = self.data.xmat[body_id].copy().reshape(3, 3)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, xmat.ravel())
            return np.concatenate([pos, quat])
        except Exception:
            return np.zeros(7)

    # ── 增量计算 ───────────────────────────────────────

    def compute_delta_pos(
        self, target: np.ndarray, current: np.ndarray, speed: float = 0.01,
    ) -> np.ndarray:
        """计算位置增量（限幅到 speed）。"""
        delta = target - current
        dist = np.linalg.norm(delta)
        if dist < 1e-6:
            return np.zeros(3)
        return np.clip(delta / dist * speed, -speed, speed)

    def compute_delta_rot(
        self,
        target_quat: np.ndarray,
        current_quat: np.ndarray,
        speed: float = 0.05,
    ) -> np.ndarray:
        """计算姿态增量（限幅到 speed）。"""
        ori_err = np.zeros(3)
        mujoco.mju_subQuat(ori_err, target_quat, current_quat)
        angle = np.linalg.norm(ori_err)
        if angle < 1e-6:
            return np.zeros(3)
        return np.clip(ori_err / angle * min(angle, speed), -speed, speed)

    # ── 动作构建 ───────────────────────────────────────

    def make_action(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None = None,
        gripper_cmd: float = 0.0,
    ) -> np.ndarray:
        """构建绝对位姿动作 [x,y,z, qw,qx,qy,qz, gripper]。

        Args:
            target_pos: [3] 目标位置（世界坐标系）
            target_quat: [4] 目标四元数，None 用恒等姿态
            gripper_cmd: [-1, 1]，正=打开

        Returns:
            [8] 动作向量
        """
        action = np.zeros(8, dtype=np.float64)
        action[:3] = target_pos
        if target_quat is not None:
            action[3:7] = target_quat
        else:
            action[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        action[7] = gripper_cmd
        return action
