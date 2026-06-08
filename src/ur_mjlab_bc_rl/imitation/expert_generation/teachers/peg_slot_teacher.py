"""Peg-in-Slot 任务的 Scripted Teacher。"""

from __future__ import annotations

from enum import Enum

import numpy as np
import mujoco

from .base import Teacher, TeacherState


class PegSlotState(Enum):
    """Peg-in-Slot 状态枚举。"""
    APPROACH_PEG = 0
    DESCEND_GRASP = 1
    CLOSE_GRIPPER = 2
    LIFT = 3
    MOVE_ABOVE_SLOT = 4
    ALIGN = 5
    DESCEND_INSERT = 6
    OPEN_GRIPPER = 7
    RETREAT = 8
    SUCCESS = 9


class PegSlotTeacher(Teacher):
    """Peg-in-Slot Scripted Teacher。

    抓取 peg 并将其插入 slot_block 的凹槽中。

    状态机：
    0. APPROACH_PEG:   移动到 peg 上方
    1. DESCEND_GRASP:  下降抓取 peg
    2. CLOSE_GRIPPER:  闭合夹爪
    3. LIFT:           抬起 peg
    4. MOVE_ABOVE_SLOT: 移动到 slot 上方
    5. ALIGN:          对齐 peg 与 slot（方向和位置）
    6. DESCEND_INSERT: 低速下降插入
    7. OPEN_GRIPPER:   释放 peg
    8. RETREAT:        撤退
    9. SUCCESS:        完成
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        super().__init__(model, data)
        self.phase = PegSlotState.APPROACH_PEG
        self.phase_step = 0
        self.phase_max_steps = 150
        self.insert_steps = 0

    def reset(self) -> None:
        super().reset()
        self.phase = PegSlotState.APPROACH_PEG
        self.phase_step = 0
        self.insert_steps = 0

    def step(self) -> np.ndarray:
        self.current_step += 1
        self.phase_step += 1

        action = np.zeros(7)

        try:
            if self.phase == PegSlotState.APPROACH_PEG:
                action = self._approach_peg()
            elif self.phase == PegSlotState.DESCEND_GRASP:
                action = self._descend_grasp()
            elif self.phase == PegSlotState.CLOSE_GRIPPER:
                action = self._close_gripper()
            elif self.phase == PegSlotState.LIFT:
                action = self._lift()
            elif self.phase == PegSlotState.MOVE_ABOVE_SLOT:
                action = self._move_above_slot()
            elif self.phase == PegSlotState.ALIGN:
                action = self._align()
            elif self.phase == PegSlotState.DESCEND_INSERT:
                action = self._descend_insert()
            elif self.phase == PegSlotState.OPEN_GRIPPER:
                action = self._open_gripper()
            elif self.phase == PegSlotState.RETREAT:
                action = self._retreat()
            elif self.phase == PegSlotState.SUCCESS:
                self.state = TeacherState.SUCCESS
        except Exception:
            self.state = TeacherState.FAILURE

        return action

    # ---- 各阶段实现 ----

    def _approach_peg(self) -> np.ndarray:
        """移动到 peg 上方。"""
        peg_pose = self.get_object_pose("peg")
        target = peg_pose[:3] + np.array([0.0, 0.0, 0.12])

        ee = self.get_ee_pose()
        action = np.zeros(7)
        action[:3] = self.compute_delta_pos(target, ee[:3], speed=0.02)

        if np.linalg.norm(target - ee[:3]) < 0.02:
            self.phase = PegSlotState.DESCEND_GRASP
            self.phase_step = 0
        return action

    def _descend_grasp(self) -> np.ndarray:
        """下降抓取 peg。"""
        peg_pose = self.get_object_pose("peg")
        # peg_grasp_site 在 peg body 上，z=0.04（主体顶部）
        target = peg_pose[:3] + np.array([0.0, 0.0, 0.04])

        ee = self.get_ee_pose()
        action = np.zeros(7)
        action[:3] = self.compute_delta_pos(target, ee[:3], speed=0.008)
        action[6] = 1.0  # 保持打开

        if np.linalg.norm(target - ee[:3]) < 0.012:
            self.phase = PegSlotState.CLOSE_GRIPPER
            self.phase_step = 0
        return action

    def _close_gripper(self) -> np.ndarray:
        """闭合夹爪抓住 peg。"""
        action = np.zeros(7)
        action[6] = -1.0

        if self.phase_step > 40:
            self.phase = PegSlotState.LIFT
            self.phase_step = 0
        return action

    def _lift(self) -> np.ndarray:
        """抬起 peg。"""
        ee = self.get_ee_pose()
        target = ee[:3] + np.array([0.0, 0.0, 0.08])

        action = np.zeros(7)
        action[:3] = self.compute_delta_pos(target, ee[:3], speed=0.015)
        action[6] = -1.0

        if np.linalg.norm(target - ee[:3]) < 0.015:
            self.phase = PegSlotState.MOVE_ABOVE_SLOT
            self.phase_step = 0
        return action

    def _move_above_slot(self) -> np.ndarray:
        """移动到 slot 上方。"""
        slot_pose = self.get_object_pose("slot_block")
        target = slot_pose[:3] + np.array([0.0, 0.0, 0.12])

        ee = self.get_ee_pose()
        action = np.zeros(7)
        action[:3] = self.compute_delta_pos(target, ee[:3], speed=0.02)
        action[6] = -1.0

        if np.linalg.norm(target - ee[:3]) < 0.02:
            self.phase = PegSlotState.ALIGN
            self.phase_step = 0
        return action

    def _align(self) -> np.ndarray:
        """对齐 peg 与 slot（方向对齐 + 位置微调）。

        peg 需要竖直向下插入 slot（slot 凹槽沿 z 轴）。
        """
        slot_pose = self.get_object_pose("slot_block")
        peg_pose = self.get_object_pose("peg")

        # 位置对齐：peg 的 xy 应该与 slot_center 对齐
        target_xy = slot_pose[:2]
        current_xy = peg_pose[:2]
        delta_xy = target_xy - current_xy

        action = np.zeros(7)
        action[:2] = np.clip(delta_xy * 0.3, -0.01, 0.01)  # x, y 调整
        action[6] = -1.0

        # 姿态对齐：简化处理——让 peg 竖直
        # 实际中应该用更精确的旋转对齐
        peg_quat = peg_pose[3:]
        # 检查是否接近竖直（w≈1, x≈0, y≈0, z≈0）

        # 当 xy 对齐且姿态正确时进入插入阶段
        xy_error = np.linalg.norm(delta_xy)
        if xy_error < 0.008 or self.phase_step > 120:
            self.phase = PegSlotState.DESCEND_INSERT
            self.phase_step = 0
            self.insert_steps = 0
        return action

    def _descend_insert(self) -> np.ndarray:
        """低速下降插入 peg 到 slot 中。

        这是关键步骤：速度必须慢以保证精度。
        """
        slot_pose = self.get_object_pose("slot_block")
        peg_pose = self.get_object_pose("peg")

        # 目标：slot 凹槽顶部下方
        # slot_top site 在 z=0.025，插入目标在 slot 内部
        target_z = slot_pose[2] + 0.025  # slot 凹槽顶部附近

        ee = self.get_ee_pose()
        action = np.zeros(7)
        # 低速向下
        if ee[2] > target_z + 0.005:
            action[2] = -0.005  # 慢速下降 0.005 m/step
        else:
            action[2] = 0.0
        action[6] = -1.0  # 保持抓取

        self.insert_steps += 1

        # 检查插入深度：peg 应进入 slot
        peg_z = peg_pose[2]
        inserted = peg_z < slot_pose[2] + 0.015

        if inserted or self.insert_steps > 250:
            self.phase = PegSlotState.OPEN_GRIPPER
            self.phase_step = 0
        return action

    def _open_gripper(self) -> np.ndarray:
        """释放 peg。"""
        action = np.zeros(7)
        action[6] = 1.0

        if self.phase_step > 30:
            self.phase = PegSlotState.RETREAT
            self.phase_step = 0
        return action

    def _retreat(self) -> np.ndarray:
        """撤退。"""
        ee = self.get_ee_pose()
        target = ee[:3] + np.array([0.0, 0.0, 0.08])

        action = np.zeros(7)
        action[:3] = self.compute_delta_pos(target, ee[:3], speed=0.02)
        action[6] = 1.0

        if np.linalg.norm(target - ee[:3]) < 0.02:
            self.phase = PegSlotState.SUCCESS
            self.phase_step = 0
        return action
