"""专家 Teacher 基类。

所有 scripted teacher 都应继承此类。
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
import mujoco


class TeacherState(Enum):
    """Teacher 状态枚举。"""
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


class Teacher:
    """Scripted teacher 基类。
    
    提供状态机框架，用于自动生成专家轨迹。
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """初始化 teacher。
        
        Args:
            model: MuJoCo 模型
            data: MuJoCo 数据
        """
        self.model = model
        self.data = data
        self.state = TeacherState.RUNNING

    def reset(self) -> None:
        """重置 teacher 状态。"""
        self.state = TeacherState.RUNNING

    def step(self) -> np.ndarray:
        """执行一步，返回动作。
        
        返回：
            [7] 动作向量
        
        异常：
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("Subclasses must implement step method")

    def is_success(self) -> bool:
        """检查是否成功。"""
        return self.state == TeacherState.SUCCESS

    def is_failure(self) -> bool:
        """检查是否失败。"""
        return self.state == TeacherState.FAILURE

    def is_done(self) -> bool:
        """检查是否完成（成功或失败）。"""
        return self.is_success() or self.is_failure()

    def get_ee_pose(self) -> np.ndarray:
        """获取末端位姿。"""
        try:
            site_id = self.model.site("_tcp").id
            pos = self.data.site_xpos[site_id].copy()
            xmat = self.data.site_xmat[site_id].copy().reshape(3, 3)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, xmat.ravel())
            return np.concatenate([pos, quat])
        except:
            return np.zeros(7)

    def get_object_pose(self, object_name: str) -> np.ndarray:
        """获取物体位姿。"""
        try:
            body_id = self.model.body(object_name).id
            pos = self.data.xpos[body_id].copy()
            xmat = self.data.xmat[body_id].copy().reshape(3, 3)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, xmat.ravel())
            return np.concatenate([pos, quat])
        except:
            return np.zeros(7)

    def compute_delta_pos(
        self,
        target: np.ndarray,
        current: np.ndarray,
        speed: float = 0.01,
    ) -> np.ndarray:
        """计算位置增量（保留兼容）。"""
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
        """计算姿态增量（保留兼容）。"""
        ori_err = np.zeros(3)
        mujoco.mju_subQuat(ori_err, target_quat, current_quat)
        angle = np.linalg.norm(ori_err)
        if angle < 1e-6:
            return np.zeros(3)
        return np.clip(ori_err / angle * min(angle, speed), -speed, speed)

    def make_action(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None = None,
        gripper_cmd: float = 0.0,
    ) -> np.ndarray:
        """构建绝对位姿动作 [x,y,z, qw,qx,qy,qz, gripper]。

        mink 的 VelocityLimit 会自然限制关节运动速度，
        因此 teacher 只需输出期望的绝对位姿即可。

        Args:
            target_pos: [3] 目标位置（世界坐标系）
            target_quat: [4] 目标四元数，None 则保持当前姿态
            gripper_cmd: [-1, 1]，正=打开

        Returns:
            [8] 动作向量
        """
        action = np.zeros(8)
        action[:3] = target_pos
        if target_quat is not None:
            action[3:7] = target_quat
        else:
            action[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # identity
        action[7] = gripper_cmd
        return action
