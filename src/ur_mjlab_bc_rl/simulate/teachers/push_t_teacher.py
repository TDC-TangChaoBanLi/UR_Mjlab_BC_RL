"""Push-T 任务的 Scripted Teacher。"""

from __future__ import annotations

from enum import Enum

import numpy as np
import mujoco

from .base import Teacher, TeacherState


class PushTState(Enum):
    APPROACH = 0
    CONTACT = 1
    PUSH = 2
    CORRECT_YAW = 3
    CORRECT_POS = 4
    RETREAT = 5
    SUCCESS = 6


class PushTTeacher(Teacher):
    """Push-T Scripted Teacher。

    将 T 形物体推到目标 marker 位置并对齐朝向。

    状态机:
    0. APPROACH:    移动到 T 形上方
    1. CONTACT:     下降接触 T 形
    2. PUSH:        推动 T 形向目标
    3. CORRECT_YAW: 调整 T 形朝向
    4. CORRECT_POS: 微调位置
    5. RETREAT:     撤退
    6. SUCCESS:     完成
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        super().__init__(model, data)
        self.phase = PushTState.APPROACH
        self.phase_step = 0
        self.push_steps = 0

    def reset(self) -> None:
        super().reset()
        self.phase = PushTState.APPROACH
        self.phase_step = 0
        self.push_steps = 0

    def step(self) -> np.ndarray:
        self.current_step += 1
        self.phase_step += 1

        try:
            stages = {
                PushTState.APPROACH: self._approach,
                PushTState.CONTACT: self._contact,
                PushTState.PUSH: self._push,
                PushTState.CORRECT_YAW: self._correct_yaw,
                PushTState.CORRECT_POS: self._correct_pos,
                PushTState.RETREAT: self._retreat,
                PushTState.SUCCESS: lambda: None,
            }
            fn = stages.get(self.phase)
            if fn is not None:
                fn()
            if self.phase == PushTState.SUCCESS:
                self.state = TeacherState.SUCCESS
        except Exception:
            self.state = TeacherState.FAILURE

        return self.make_action(
            getattr(self, "_action_pos", np.zeros(3)),
            gripper_cmd=getattr(self, "_action_gripper", 0.0),
        )

    # ── 各阶段 ─────────────────────────────────────────

    def _approach(self) -> None:
        t_pose = self.get_object_pose("t_shape")
        target = t_pose[:3] + np.array([0, 0, 0.12])

        ee = self.get_ee_pose()
        self._action_pos = self.compute_delta_pos(target, ee[:3], speed=0.02)
        self._action_gripper = 0.0

        if np.linalg.norm(target - ee[:3]) < 0.02:
            self.phase = PushTState.CONTACT
            self.phase_step = 0

    def _contact(self) -> None:
        t_pose = self.get_object_pose("t_shape")
        target = t_pose[:3] + np.array([0, 0, 0.025])

        ee = self.get_ee_pose()
        self._action_pos = self.compute_delta_pos(target, ee[:3], speed=0.008)
        self._action_gripper = -1.0

        if np.linalg.norm(target - ee[:3]) < 0.012 or self.phase_step > 60:
            self.phase = PushTState.PUSH
            self.phase_step = 0
            self.push_steps = 0

    def _push(self) -> None:
        t_pose = self.get_object_pose("t_shape")
        goal = self.get_object_pose("goal_marker")

        push_dir = goal[:3] - t_pose[:3]
        push_dir[2] = 0.0
        dist = np.linalg.norm(push_dir)

        if dist > 0.005:
            push_dir /= dist
            self._action_pos = push_dir * 0.015
        else:
            self._action_pos = np.zeros(3)

        self._action_gripper = -1.0
        self.push_steps += 1

        if dist < 0.015 or self.push_steps > 200:
            self.phase = PushTState.CORRECT_YAW
            self.phase_step = 0

    def _correct_yaw(self) -> None:
        t_pose = self.get_object_pose("t_shape")
        goal = self.get_object_pose("goal_marker")

        def quat_to_yaw(q):
            w, x, y, z = q
            return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        yaw_diff = quat_to_yaw(goal[3:]) - quat_to_yaw(t_pose[3:])
        yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))

        self._action_pos = np.zeros(3)
        self._action_pos[5] = np.clip(yaw_diff * 0.3, -0.05, 0.05)  # z 旋转
        self._action_gripper = -1.0

        if abs(yaw_diff) < 0.05 or self.phase_step > 80:
            self.phase = PushTState.CORRECT_POS
            self.phase_step = 0

    def _correct_pos(self) -> None:
        t_pose = self.get_object_pose("t_shape")
        goal = self.get_object_pose("goal_marker")

        delta = goal[:3] - t_pose[:3]
        delta[2] = 0.0
        dist = np.linalg.norm(delta)

        if dist > 0.003:
            self._action_pos = np.clip(delta * 0.5, -0.01, 0.01)
        else:
            self._action_pos = np.zeros(3)
        self._action_gripper = -1.0

        if dist < 0.008 or self.phase_step > 100:
            self.phase = PushTState.RETREAT
            self.phase_step = 0

    def _retreat(self) -> None:
        ee = self.get_ee_pose()
        target = ee[:3] + np.array([0, 0, 0.08])

        self._action_pos = self.compute_delta_pos(target, ee[:3], speed=0.02)
        self._action_gripper = 1.0

        if np.linalg.norm(target - ee[:3]) < 0.02:
            self.phase = PushTState.SUCCESS
            self.phase_step = 0
