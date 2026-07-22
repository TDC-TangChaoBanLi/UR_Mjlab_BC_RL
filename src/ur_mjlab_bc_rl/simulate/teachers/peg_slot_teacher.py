"""Peg-in-Slot 任务的 Scripted Teacher。"""

from __future__ import annotations

from enum import Enum

import numpy as np
import mujoco

from .base import Teacher, TeacherState


class PegSlotState(Enum):
    APPROACH = 0
    DESCEND_GRASP = 1
    CLOSE = 2
    LIFT = 3
    MOVE_ABOVE = 4
    ALIGN = 5
    INSERT = 6
    OPEN = 7
    RETREAT = 8
    SUCCESS = 9


class PegSlotTeacher(Teacher):
    """Peg-in-Slot Scripted Teacher。

    抓取 peg 并插入 slot_block 凹槽中。

    状态机:
    0. APPROACH:       移动到 peg 上方
    1. DESCEND_GRASP:  下降抓取 peg
    2. CLOSE:          闭合夹爪
    3. LIFT:           抬起 peg
    4. MOVE_ABOVE:     移动到 slot 上方
    5. ALIGN:          对齐 xy + 方向
    6. INSERT:         低速下降插入
    7. OPEN:           释放 peg
    8. RETREAT:        撤退
    9. SUCCESS:        完成
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        super().__init__(model, data)
        self.phase = PegSlotState.APPROACH
        self.phase_step = 0
        self.insert_steps = 0

    def reset(self) -> None:
        super().reset()
        self.phase = PegSlotState.APPROACH
        self.phase_step = 0
        self.insert_steps = 0

    def step(self) -> np.ndarray:
        self.current_step += 1
        self.phase_step += 1

        try:
            stages = {
                PegSlotState.APPROACH: self._approach,
                PegSlotState.DESCEND_GRASP: self._descend_grasp,
                PegSlotState.CLOSE: self._close,
                PegSlotState.LIFT: self._lift,
                PegSlotState.MOVE_ABOVE: self._move_above,
                PegSlotState.ALIGN: self._align,
                PegSlotState.INSERT: self._insert,
                PegSlotState.OPEN: self._open,
                PegSlotState.RETREAT: self._retreat,
                PegSlotState.SUCCESS: lambda: None,
            }
            fn = stages.get(self.phase)
            if fn is not None:
                fn()
            if self.phase == PegSlotState.SUCCESS:
                self.state = TeacherState.SUCCESS
        except Exception:
            self.state = TeacherState.FAILURE

        return self.make_action(
            getattr(self, "_action_pos", np.zeros(3)),
            gripper_cmd=getattr(self, "_action_gripper", 0.0),
        )

    # ── 各阶段 ─────────────────────────────────────────

    def _approach(self) -> None:
        peg = self.get_object_pose("peg")
        target = peg[:3] + np.array([0, 0, 0.12])

        ee = self.get_ee_pose()
        self._action_pos = self.compute_delta_pos(target, ee[:3], speed=0.02)
        self._action_gripper = 0.0

        if np.linalg.norm(target - ee[:3]) < 0.02:
            self.phase = PegSlotState.DESCEND_GRASP
            self.phase_step = 0

    def _descend_grasp(self) -> None:
        peg = self.get_object_pose("peg")
        target = peg[:3] + np.array([0, 0, 0.04])

        ee = self.get_ee_pose()
        self._action_pos = self.compute_delta_pos(target, ee[:3], speed=0.008)
        self._action_gripper = 1.0

        if np.linalg.norm(target - ee[:3]) < 0.012:
            self.phase = PegSlotState.CLOSE
            self.phase_step = 0

    def _close(self) -> None:
        self._action_pos = np.zeros(3)
        self._action_gripper = -1.0

        if self.phase_step > 40:
            self.phase = PegSlotState.LIFT
            self.phase_step = 0

    def _lift(self) -> None:
        ee = self.get_ee_pose()
        target = ee[:3] + np.array([0, 0, 0.08])

        self._action_pos = self.compute_delta_pos(target, ee[:3], speed=0.015)
        self._action_gripper = -1.0

        if np.linalg.norm(target - ee[:3]) < 0.015:
            self.phase = PegSlotState.MOVE_ABOVE
            self.phase_step = 0

    def _move_above(self) -> None:
        slot = self.get_object_pose("slot_block")
        target = slot[:3] + np.array([0, 0, 0.12])

        ee = self.get_ee_pose()
        self._action_pos = self.compute_delta_pos(target, ee[:3], speed=0.02)
        self._action_gripper = -1.0

        if np.linalg.norm(target - ee[:3]) < 0.02:
            self.phase = PegSlotState.ALIGN
            self.phase_step = 0

    def _align(self) -> None:
        slot = self.get_object_pose("slot_block")
        peg = self.get_object_pose("peg")

        delta_xy = slot[:2] - peg[:2]
        self._action_pos = np.zeros(3)
        self._action_pos[:2] = np.clip(delta_xy * 0.3, -0.01, 0.01)
        self._action_gripper = -1.0

        xy_err = np.linalg.norm(delta_xy)
        if xy_err < 0.008 or self.phase_step > 120:
            self.phase = PegSlotState.INSERT
            self.phase_step = 0
            self.insert_steps = 0

    def _insert(self) -> None:
        slot = self.get_object_pose("slot_block")
        peg = self.get_object_pose("peg")
        target_z = slot[2] + 0.025

        ee = self.get_ee_pose()
        self._action_pos = np.zeros(3)
        if ee[2] > target_z + 0.005:
            self._action_pos[2] = -0.005
        self._action_gripper = -1.0

        self.insert_steps += 1

        if peg[2] < slot[2] + 0.015 or self.insert_steps > 250:
            self.phase = PegSlotState.OPEN
            self.phase_step = 0

    def _open(self) -> None:
        self._action_pos = np.zeros(3)
        self._action_gripper = 1.0

        if self.phase_step > 30:
            self.phase = PegSlotState.RETREAT
            self.phase_step = 0

    def _retreat(self) -> None:
        ee = self.get_ee_pose()
        target = ee[:3] + np.array([0, 0, 0.08])

        self._action_pos = self.compute_delta_pos(target, ee[:3], speed=0.02)
        self._action_gripper = 1.0

        if np.linalg.norm(target - ee[:3]) < 0.02:
            self.phase = PegSlotState.SUCCESS
            self.phase_step = 0
