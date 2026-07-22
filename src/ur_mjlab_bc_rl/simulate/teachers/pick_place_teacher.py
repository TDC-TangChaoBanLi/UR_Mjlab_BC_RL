"""Pick-and-Place Scripted Teacher — 含姿态检测和方向调整。

状态机:
  0. MOVE_ABOVE    — 移动到物块上方，计算最佳夹取角
  1. ROTATE        — 旋转夹爪到最佳方向（保持在上方）
  2. DESCEND       — 下降抓取
  3. CLOSE         — 闭合夹爪 + 抓取检测
  4. LIFT          — 抬起 + 掉落检测
  5. MOVE_TO_PLATE — 移动到盘子 + 掉落检测
  6. PLACE         — 下降放置
  7. OPEN          — 释放
  8. RETREAT       — 撤退 + 放置检测
  9. SUCCESS

输出绝对目标位姿 [x,y,z, qw,qx,qy,qz, gripper_cmd]。
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import mujoco

from .base import Teacher, TeacherState

CUBE_NAME = "cube"
PLATE_NAME = "plate"

# 夹爪朝下基础四元数: TCP Z → world -Z, TCP X → world Y
GRASP_QUAT = np.array([0.0, 0.7071, 0.7071, 0.0])

# 阈值 (actuator 控制下适当放宽)
APPROACH_POS = 0.03     # 位置到达阈值
APPROACH_ROT = 0.1      # 姿态到达阈值 (rad)
GRASP_DIST = 0.08       # 抓取检测
DROP_DIST = 0.10        # 掉落检测
PLACE_DIST = 0.10       # 放置检测
GRIPPER_WAIT = 50       # 夹爪动作等待步数
SETTLE_WAIT = 30        # 沉降等待
MAX_RETRIES = 3


class PickPlaceState(Enum):
    MOVE_ABOVE = 0 # 移动到物块上方
    ROTATE = 1 # 旋转夹爪到最佳方向
    DESCEND = 2 # 下降到抓取处
    CLOSE = 3 # 闭合夹爪
    LIFT = 4 # 抬起
    MOVE_TO_PLATE = 5 # 移动到盘子上方
    PLACE = 6 # 下降放置
    OPEN = 7 # 释放夹爪
    RETREAT = 8 # 撤退
    GO_BACK = 9 # 回到初始位置
    SUCCESS = 10 # 成功放置


class PickPlaceTeacher(Teacher):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        super().__init__(model, data)
        self.phase = PickPlaceState.MOVE_ABOVE
        self.phase_step = 0
        self._init_pose = self.get_ee_pose()
        self._target_pos = np.zeros(3)
        self._target_quat = GRASP_QUAT.copy()
        self._grasp_yaw = 0.0
        self._retry_count = 0

    def reset(self) -> None:
        super().reset()
        self._init_pose = self.get_ee_pose()
        self.phase = PickPlaceState.MOVE_ABOVE
        self.phase_step = 0
        self._retry_count = 0

    def step(self) -> np.ndarray:
        self.current_step += 1
        self.phase_step += 1
        try:
            return {
                PickPlaceState.MOVE_ABOVE: self._move_above,
                PickPlaceState.ROTATE: self._rotate,
                PickPlaceState.DESCEND: self._descend,
                PickPlaceState.CLOSE: self._close,
                PickPlaceState.LIFT: self._lift,
                PickPlaceState.MOVE_TO_PLATE: self._move_to_plate,
                PickPlaceState.PLACE: self._place,
                PickPlaceState.OPEN: self._open,
                PickPlaceState.RETREAT: self._retreat,
                PickPlaceState.GO_BACK: self._go_back,
                PickPlaceState.SUCCESS: self._success,
            }[self.phase]()
        except Exception:
            self.state = TeacherState.FAILURE
            return self.make_action(np.zeros(3))

    # ── MOVE_ABOVE ────────────────────────────────────

    def _move_above(self) -> np.ndarray:
        """移动到物块上方，计算最优夹取角。"""
        if self.phase_step == 1:
            cube = self.get_object_pose(CUBE_NAME)
            self._target_pos = cube[:3] + np.array([0, 0, 0.12])

        ee = self.get_ee_pose()
        if np.linalg.norm(self._target_pos - ee[:3]) < APPROACH_POS:
            # self.phase = PickPlaceState.ROTATE
            self.phase = PickPlaceState.DESCEND
            self.phase_step = 0
        return self.make_action(self._target_pos, self._target_quat)

    # ── ROTATE ───────────────────────────────────────

    def _rotate(self) -> np.ndarray:
        """在物块上方旋转夹爪到最佳方向。"""
        if self.phase_step == 1:
            cube = self.get_object_pose(CUBE_NAME)
            self._grasp_yaw = self._compute_grasp_yaw(cube[3:])
            self._target_quat = self._make_grasp_quat(self._grasp_yaw)
        
        ee = self.get_ee_pose()
        # 用 mju_subQuat 计算姿态误差
        err = np.zeros(3)
        mujoco.mju_subQuat(err, self._target_quat, ee[3:7])
        ang = np.linalg.norm(err)

        if ang < APPROACH_ROT:
            self.phase = PickPlaceState.DESCEND
            self.phase_step = 0

        # 保持位置在上方
        return self.make_action(self._target_pos, self._target_quat)

    # ── DESCEND ──────────────────────────────────────

    def _descend(self) -> np.ndarray:
        """下降到抓取位置。"""
        if self.phase_step == 1:
            cube = self.get_object_pose(CUBE_NAME)
            self._target_pos = cube[:3].copy()
            self._target_pos[2] += 0.01
        ee = self.get_ee_pose()
        if np.linalg.norm(self._target_pos - ee[:3]) < 0.003:
            self.phase = PickPlaceState.CLOSE
            self.phase_step = 0
        return self.make_action(self._target_pos, self._target_quat)

    # ── CLOSE ────────────────────────────────────────

    def _close(self) -> np.ndarray:
        """闭合夹爪，检测抓取。"""
        ee = self.get_ee_pose()
        if self.phase_step > GRIPPER_WAIT:
            cube = self.get_object_pose(CUBE_NAME)
            if np.linalg.norm(cube[:3] - ee[:3]) < GRASP_DIST:
                self.phase = PickPlaceState.LIFT
                self.phase_step = 0
            else:
                return self._retry()
        return self.make_action(ee[:3], ee[3:7], gripper_cmd=0.8)

    # ── LIFT ─────────────────────────────────────────

    def _lift(self) -> np.ndarray:
        """抬起，检测掉落。"""
        ee = self.get_ee_pose()
        if self.phase_step > 1:
            cube = self.get_object_pose(CUBE_NAME)
            if np.linalg.norm(cube[:3] - ee[:3]) > DROP_DIST:
                return self._retry()

        if self.phase_step == 1:
            self._target_pos = ee[:3] + np.array([0, 0, 0.10])
            self._target_quat = ee[3:7]

        if np.linalg.norm(self._target_pos - ee[:3]) < APPROACH_POS:
            self.phase = PickPlaceState.MOVE_TO_PLATE
            self.phase_step = 0
        return self.make_action(self._target_pos, self._target_quat, gripper_cmd=0.8)

    # ── MOVE_TO_PLATE ────────────────────────────────

    def _move_to_plate(self) -> np.ndarray:
        """移动到盘子上方，检测掉落。"""
        ee = self.get_ee_pose()
        if self.phase_step > 1:
            cube = self.get_object_pose(CUBE_NAME)
            if np.linalg.norm(cube[:3] - ee[:3]) > DROP_DIST:
                return self._retry()
        if self.phase_step == 1:
            self._target_quat = GRASP_QUAT.copy()

        plate = self.get_object_pose(PLATE_NAME)
        target = plate[:3] + np.array([0, 0, 0.10])
        if np.linalg.norm(target - ee[:3]) < APPROACH_POS:
            self.phase = PickPlaceState.PLACE
            self.phase_step = 0
        return self.make_action(target, self._target_quat, gripper_cmd=0.8)

    # ── PLACE ────────────────────────────────────────

    def _place(self) -> np.ndarray:
        """下降放置。"""
        ee = self.get_ee_pose()
        if self.phase_step > 1:
            cube = self.get_object_pose(CUBE_NAME)
            if np.linalg.norm(cube[:3] - ee[:3]) > DROP_DIST:
                return self._retry()
        if self.phase_step == 1:
            self._target_quat = GRASP_QUAT.copy()

        plate = self.get_object_pose(PLATE_NAME)
        target = plate[:3] + np.array([0, 0, 0.08])
        if np.linalg.norm(target - ee[:3]) < 0.01:
            self.phase = PickPlaceState.OPEN
            self.phase_step = 0
        return self.make_action(target, self._target_quat, gripper_cmd=0.8)

    # ── OPEN ─────────────────────────────────────────

    def _open(self) -> np.ndarray:
        """释放。"""
        ee = self.get_ee_pose()
        if self.phase_step > GRIPPER_WAIT:
            self.phase = PickPlaceState.RETREAT
            self.phase_step = 0
        return self.make_action(ee[:3], ee[3:7], gripper_cmd=0.0)

    # ── RETREAT ──────────────────────────────────────

    def _retreat(self) -> np.ndarray:
        """撤退，检测放置成功。"""
        ee = self.get_ee_pose()
        if self.phase_step == 1:
            self._target_pos = ee[:3] + np.array([0, 0, 0.15])
            self._target_quat = GRASP_QUAT.copy()

        if np.linalg.norm(self._target_pos - ee[:3]) < APPROACH_POS:
            if self.phase_step < SETTLE_WAIT:
                return self.make_action(self._target_pos, ee[3:7], gripper_cmd=0.0)

            cube = self.get_object_pose(CUBE_NAME)
            plate = self.get_object_pose(PLATE_NAME)
            if np.linalg.norm(cube[:2] - plate[:2]) < PLACE_DIST:
                self.phase = PickPlaceState.GO_BACK
                self.phase_step = 0
            else:
                return self._retry()
        return self.make_action(self._target_pos, self._target_quat, gripper_cmd=0.0)

    # ── GO_BACK ──────────────────────────────────────


    def _go_back(self) -> np.ndarray:
        """回到初始位置。"""
        ee = self.get_ee_pose()
        if self.phase_step == 1:
            self._target_pos = self._init_pose[:3].copy()
            self._target_quat = self._init_pose[3:7].copy()

        if np.linalg.norm(self._target_pos - ee[:3]) < APPROACH_POS:
            if self.phase_step < SETTLE_WAIT:
                return self.make_action(self._target_pos, ee[3:7], gripper_cmd=0.0)

            cube = self.get_object_pose(CUBE_NAME)
            plate = self.get_object_pose(PLATE_NAME)
            if np.linalg.norm(cube[:2] - plate[:2]) < PLACE_DIST:
                self.phase = PickPlaceState.SUCCESS
                self.phase_step = 0
            else:
                return self._retry()
        return self.make_action(self._target_pos, self._target_quat, gripper_cmd=0.0)


    # ── SUCCESS ──────────────────────────────────────

    def _success(self) -> np.ndarray:
        self.state = TeacherState.SUCCESS
        return self.make_action(np.zeros(3))

    # ── 夹取姿态 ─────────────────────────────────────

    def _compute_grasp_yaw(self, cube_quat: np.ndarray) -> float:
        axes = self._cube_axes(cube_quat)
        z_dots = [abs(np.dot(a, [0, 0, 1])) for a in axes]
        vertical_idx = np.argmax(z_dots)

        candidates = []
        for i in range(3):
            if i == vertical_idx:
                continue
            n = axes[i]
            if abs(n[2]) > 0.7:
                continue
            for sign in [1.0, -1.0]:
                nx, ny = sign * n[0], sign * n[1]
                if abs(nx) + abs(ny) < 0.01:
                    continue
                candidates.append(float(np.arctan2(ny, nx)))

        if not candidates:
            return 0.0

        ee = self.get_ee_pose()
        cy = self._quat_to_yaw(ee[3:])
        best = min(candidates, key=lambda t: min(
            abs(t - cy), abs(t - cy + 2*np.pi), abs(t - cy - 2*np.pi)))
        return best

    @staticmethod
    def _cube_axes(quat: np.ndarray) -> list[np.ndarray]:
        w, x, y, z = quat
        R = np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y],
        ])
        return [R[:, 0], R[:, 1], R[:, 2]]

    @staticmethod
    def _quat_to_yaw(quat: np.ndarray) -> float:
        w, x, y, z = quat
        return float(np.arctan2(2*(w*z + x*y), 1-2*(y*y+z*z)))

    @staticmethod
    def _make_grasp_quat(yaw: float) -> np.ndarray:
        cy, sy = np.cos(yaw/2), np.sin(yaw/2)
        a = np.array([cy, 0, 0, sy])
        b = GRASP_QUAT
        return np.array([
            a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
            a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
            a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
            a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
        ])

    def _retry(self) -> np.ndarray:
        self._retry_count += 1
        if self._retry_count > MAX_RETRIES:
            self.state = TeacherState.FAILURE
            return self.make_action(np.zeros(3))
        self.phase = PickPlaceState.MOVE_ABOVE
        self.phase_step = 0
        return self.make_action(np.zeros(3))
