"""Push-T 任务的 Scripted Teacher。"""

from __future__ import annotations

from enum import Enum

import numpy as np
import mujoco

from .base import Teacher, TeacherState


class PushTState(Enum):
    """Push-T 状态枚举。"""
    APPROACH = 0
    CONTACT = 1
    PUSH = 2
    CORRECT_YAW = 3
    CORRECT_POS = 4
    RETREAT = 5
    SUCCESS = 6


class PushTTeacher(Teacher):
    """Push-T Scripted Teacher。

    将 T 形物体推动到目标位姿（goal_marker 位置 + 朝向）。

    状态机：
    0. APPROACH:    移动到 T 形上方
    1. CONTACT:     下降到接触 T 形
    2. PUSH:        推动 T 形向目标移动
    3. CORRECT_YAW: 调整 T 形朝向
    4. CORRECT_POS: 微调位置
    5. RETREAT:     撤退
    6. SUCCESS:     完成
    """

    # 目标朝向：T 形需要对齐到的 yaw 角（绕 z 轴）
    # 默认 0 rad（横梁沿 y 轴）
    TARGET_YAW = 0.0

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        super().__init__(model, data)
        self.phase = PushTState.APPROACH
        self.phase_step = 0
        self.phase_max_steps = 80
        self.push_steps = 0

    def reset(self) -> None:
        super().reset()
        self.phase = PushTState.APPROACH
        self.phase_step = 0
        self.push_steps = 0

    def step(self) -> np.ndarray:
        self.current_step += 1
        self.phase_step += 1

        action = np.zeros(7)

        try:
            if self.phase == PushTState.APPROACH:
                action = self._approach()
            elif self.phase == PushTState.CONTACT:
                action = self._contact()
            elif self.phase == PushTState.PUSH:
                action = self._push()
            elif self.phase == PushTState.CORRECT_YAW:
                action = self._correct_yaw()
            elif self.phase == PushTState.CORRECT_POS:
                action = self._correct_pos()
            elif self.phase == PushTState.RETREAT:
                action = self._retreat()
            elif self.phase == PushTState.SUCCESS:
                self.state = TeacherState.SUCCESS
        except Exception:
            self.state = TeacherState.FAILURE

        return action

    # ---- 各阶段实现 ----

    def _approach(self) -> np.ndarray:
        """移动到 T 形上方。"""
        t_pose = self.get_object_pose("t_shape")
        target = t_pose[:3] + np.array([0.0, 0.0, 0.12])

        ee = self.get_ee_pose()
        action = np.zeros(7)
        action[:3] = self.compute_delta_pos(target, ee[:3], speed=0.02)

        if np.linalg.norm(target - ee[:3]) < 0.02:
            self.phase = PushTState.CONTACT
            self.phase_step = 0
        return action

    def _contact(self) -> np.ndarray:
        """下降接触 T 形。"""
        t_pose = self.get_object_pose("t_shape")
        # 接触点：T 形中心偏上方一点点
        target = t_pose[:3] + np.array([0.0, 0.0, 0.025])

        ee = self.get_ee_pose()
        action = np.zeros(7)
        action[:3] = self.compute_delta_pos(target, ee[:3], speed=0.008)
        action[6] = -1.0  # 合拢夹爪，用外侧推

        if np.linalg.norm(target - ee[:3]) < 0.012 or self.phase_step > 60:
            self.phase = PushTState.PUSH
            self.phase_step = 0
            self.push_steps = 0
        return action

    def _push(self) -> np.ndarray:
        """推动 T 形向目标位置移动。"""
        t_pose = self.get_object_pose("t_shape")
        goal_pose = self.get_object_pose("goal_marker")

        # 方向：从 T 当前位置指向目标
        push_dir = goal_pose[:3] - t_pose[:3]
        push_dir[2] = 0.0  # 忽略 z 方向
        dist = np.linalg.norm(push_dir)

        action = np.zeros(7)
        if dist > 0.005:
            push_dir = push_dir / dist
            # 在 T 形的另一侧施加推力
            action[:3] = push_dir * 0.015
        else:
            action[:3] = np.zeros(3)

        action[6] = -1.0
        self.push_steps += 1

        # 位置足够近则进入朝向校正
        if dist < 0.015 or self.push_steps > 200:
            self.phase = PushTState.CORRECT_YAW
            self.phase_step = 0
        return action

    def _correct_yaw(self) -> np.ndarray:
        """校正 T 形朝向（绕 z 轴旋转）。"""
        t_pose = self.get_object_pose("t_shape")
        goal_pose = self.get_object_pose("goal_marker")

        # 计算当前 yaw 与目标 yaw 的差异
        t_quat = t_pose[3:]
        goal_quat = goal_pose[3:]

        # 将四元数转为欧拉角，取 yaw
        t_euler = np.zeros(3)
        goal_euler = np.zeros(3)
        mujoco.mju_quat2Vel(t_euler, t_quat, 1.0)
        mujoco.mju_quat2Vel(goal_euler, goal_quat, 1.0)

        # 使用简化的 2D 方法：
        # 四元数 [w, x, y, z], yaw from quat
        def quat_to_yaw(q: np.ndarray) -> float:
            w, x, y, z = q
            return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        current_yaw = quat_to_yaw(t_quat)
        target_yaw = quat_to_yaw(goal_quat)
        yaw_diff = target_yaw - current_yaw
        # 归一化到 [-pi, pi]
        yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))

        action = np.zeros(7)
        action[5] = np.clip(yaw_diff * 0.3, -0.05, 0.05)  # z 轴旋转
        action[6] = -1.0

        if abs(yaw_diff) < 0.05 or self.phase_step > 80:
            self.phase = PushTState.CORRECT_POS
            self.phase_step = 0
        return action

    def _correct_pos(self) -> np.ndarray:
        """微调位置。"""
        t_pose = self.get_object_pose("t_shape")
        goal_pose = self.get_object_pose("goal_marker")

        delta = goal_pose[:3] - t_pose[:3]
        delta[2] = 0.0
        dist = np.linalg.norm(delta)

        action = np.zeros(7)
        if dist > 0.003:
            action[:3] = np.clip(delta * 0.5, -0.01, 0.01)
        action[6] = -1.0

        if dist < 0.008 or self.phase_step > 100:
            self.phase = PushTState.RETREAT
            self.phase_step = 0
        return action

    def _retreat(self) -> np.ndarray:
        """撤退。"""
        ee = self.get_ee_pose()
        target = ee[:3] + np.array([0.0, 0.0, 0.08])

        action = np.zeros(7)
        action[:3] = self.compute_delta_pos(target, ee[:3], speed=0.02)
        action[6] = 1.0  # 打开夹爪

        if np.linalg.norm(target - ee[:3]) < 0.02:
            self.phase = PushTState.SUCCESS
            self.phase_step = 0
        return action
