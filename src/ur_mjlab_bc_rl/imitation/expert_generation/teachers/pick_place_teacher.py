"""Pick-and-Place 任务的 Scripted Teacher。

输出绝对目标位姿 [x,y,z, qw,qx,qy,qz, gripper]，
mink 的 VelocityLimit 自然地限制运动速度。
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import mujoco

from .base import Teacher, TeacherState


CUBE_NAME = "cube"
PLATE_NAME = "plate"

GRIPPER_ACTION_TIMEOUT = 50  # 夹爪动作超时步数
MAX_RETRIES = 3  # 最大重试次数
GRASP_DIST_THRESHOLD = 0.08  # 物块距夹爪超过此值视为掉落
PLACE_DIST_THRESHOLD = 0.08  # 物块距盘子中心超过此值视为未放置成功

class PickPlaceState(Enum):
    MOVE_ABOVE_CUBE = 0
    DESCEND_TO_GRASP = 1
    CLOSE_GRIPPER = 2
    LIFT = 3
    MOVE_ABOVE_PLATE = 4
    DESCEND_TO_PLACE = 5
    OPEN_GRIPPER = 6
    RETREAT = 7
    SUCCESS = 8


class PickPlaceTeacher(Teacher):
    """Pick-and-Place Scripted Teacher — 输出绝对目标位姿。"""

    # 夹爪朝下的四元数：TCP z=world -Z, TCP x=world Y, TCP y=world X
    _GRASP_QUAT = np.array([0.0, 0.7071, 0.7071, 0.0])  # w,x,y,z

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        super().__init__(model, data)
        self.phase = PickPlaceState.MOVE_ABOVE_CUBE
        self.phase_step = 0
        self._target_pos: np.ndarray = np.zeros(3)
        self._retry_count = 0

    def reset(self) -> None:
        super().reset()
        self.phase = PickPlaceState.MOVE_ABOVE_CUBE
        self.phase_step = 0
        self._retry_count = 0

    def step(self) -> np.ndarray:
        try:
            if self.phase == PickPlaceState.MOVE_ABOVE_CUBE:
                return self._move_above_cube()
            elif self.phase == PickPlaceState.DESCEND_TO_GRASP:
                return self._descend_to_grasp()
            elif self.phase == PickPlaceState.CLOSE_GRIPPER:
                return self._close_gripper()
            elif self.phase == PickPlaceState.LIFT:
                return self._lift()
            elif self.phase == PickPlaceState.MOVE_ABOVE_PLATE:
                return self._move_above_plate()
            elif self.phase == PickPlaceState.DESCEND_TO_PLACE:
                return self._descend_to_place()
            elif self.phase == PickPlaceState.OPEN_GRIPPER:
                return self._open_gripper()
            elif self.phase == PickPlaceState.RETREAT:
                return self._retreat()
            elif self.phase == PickPlaceState.SUCCESS:
                self.state = TeacherState.SUCCESS
                return self.make_action(np.zeros(3), gripper_cmd=0.0)
        except Exception as e:
            print(e)
            self.state = TeacherState.FAILURE
        return self.make_action(np.zeros(3))

    # ── 各阶段：计算绝对目标位姿 ──

    def _move_above_cube(self) -> np.ndarray:
        self.phase_step += 1
        if self.phase_step == 1:
            cube_pose = self.get_object_pose(CUBE_NAME)
            self._target_pos = cube_pose[:3] + np.array([0, 0, 0.1])
        ee = self.get_ee_pose()
        if np.linalg.norm(self._target_pos - ee[:3]) < 0.02:
            self.phase = PickPlaceState.DESCEND_TO_GRASP
            self.phase_step = 0
        return self.make_action(self._target_pos, self._GRASP_QUAT, gripper_cmd=0.0)

    def _descend_to_grasp(self) -> np.ndarray:
        self.phase_step += 1
        cube_pose = self.get_object_pose(CUBE_NAME)
        target_pos = cube_pose[:3] + np.array([0, 0, 0.02])  # 下降到更接近立方体的位置
        ee = self.get_ee_pose()
        if np.linalg.norm(target_pos - ee[:3]) < 0.01:
            self.phase = PickPlaceState.CLOSE_GRIPPER
            self.phase_step = 0
        return self.make_action(target_pos, self._GRASP_QUAT, gripper_cmd=0.0)

    def _close_gripper(self) -> np.ndarray:
        self.phase_step += 1
        ee = self.get_ee_pose()
        if self.phase_step > GRIPPER_ACTION_TIMEOUT:
            self.phase = PickPlaceState.LIFT
            self.phase_step = 0
        return self.make_action(ee[:3], ee[3:], gripper_cmd=0.8)

    def _lift(self) -> np.ndarray:
        self.phase_step += 1
        ee = self.get_ee_pose()

        # 检测物块是否在夹爪附近（防止抓空/掉落）
        if self.phase_step > 1:
            cube_pose = self.get_object_pose(CUBE_NAME)
            if np.linalg.norm(cube_pose[:3] - ee[:3]) > GRASP_DIST_THRESHOLD:
                return self._retry("lift: cube dropped")

        if self.phase_step == 1:
            self._target_pos = ee[:3] + np.array([0, 0, 0.1])
        if np.linalg.norm(self._target_pos - ee[:3]) < 0.02:
            self.phase = PickPlaceState.MOVE_ABOVE_PLATE
            self.phase_step = 0
        return self.make_action(self._target_pos, ee[3:], gripper_cmd=0.8)

    def _move_above_plate(self) -> np.ndarray:
        self.phase_step += 1
        ee = self.get_ee_pose()

        # 持续检测物块是否还在夹爪附近
        if self.phase_step > 1:
            cube_pose = self.get_object_pose(CUBE_NAME)
            if np.linalg.norm(cube_pose[:3] - ee[:3]) > GRASP_DIST_THRESHOLD:
                return self._retry("move_above_plate: cube lost")

        plate_pose = self.get_object_pose(PLATE_NAME)
        target_pos = plate_pose[:3] + np.array([0, 0, 0.1])
        if np.linalg.norm(target_pos - ee[:3]) < 0.02:
            self.phase = PickPlaceState.DESCEND_TO_PLACE
            self.phase_step = 0
        return self.make_action(target_pos, self._GRASP_QUAT, gripper_cmd=0.8)

    def _descend_to_place(self) -> np.ndarray:
        self.phase_step += 1
        ee = self.get_ee_pose()

        if self.phase_step > 1:
            cube_pose = self.get_object_pose(CUBE_NAME)
            if np.linalg.norm(cube_pose[:3] - ee[:3]) > GRASP_DIST_THRESHOLD:
                return self._retry("descend_to_place: cube lost")

        plate_pose = self.get_object_pose(PLATE_NAME)
        target_pos = plate_pose[:3] + np.array([0, 0, 0.1])
        if np.linalg.norm(target_pos - ee[:3]) < 0.01:
            self.phase = PickPlaceState.OPEN_GRIPPER
            self.phase_step = 0
        return self.make_action(target_pos, self._GRASP_QUAT, gripper_cmd=0.8)

    def _open_gripper(self) -> np.ndarray:
        self.phase_step += 1
        ee = self.get_ee_pose()
        if self.phase_step > GRIPPER_ACTION_TIMEOUT:
            self.phase = PickPlaceState.RETREAT
            self.phase_step = 0
        return self.make_action(ee[:3], ee[3:], gripper_cmd=0.0)

    def _retreat(self) -> np.ndarray:
        self.phase_step += 1
        ee = self.get_ee_pose()
        if self.phase_step == 1:
            self._target_pos = ee[:3] + np.array([0, 0, 0.15])
        if np.linalg.norm(self._target_pos - ee[:3]) < 0.02:
            # 检查物块是否在盘子上
            cube_pose = self.get_object_pose(CUBE_NAME)
            plate_pose = self.get_object_pose(PLATE_NAME)
            dist = np.linalg.norm(cube_pose[:2] - plate_pose[:2])
            if dist < PLACE_DIST_THRESHOLD and cube_pose[2] > plate_pose[2]:
                self.phase = PickPlaceState.SUCCESS
            else:
                return self._retry("retreat: cube not on plate")
            self.phase_step = 0
        return self.make_action(self._target_pos, ee[3:], gripper_cmd=0.0)

    # ── 重试逻辑 ──

    def _retry(self, reason: str = "") -> np.ndarray:
        """返回初始状态重试；超过最大次数则标记失败。"""
        self._retry_count += 1
        if self._retry_count > MAX_RETRIES:
            self.state = TeacherState.FAILURE
            return self.make_action(np.zeros(3), gripper_cmd=0.0)
        self.phase = PickPlaceState.MOVE_ABOVE_CUBE
        self.phase_step = 0
        return self.make_action(np.zeros(3), gripper_cmd=0.0)