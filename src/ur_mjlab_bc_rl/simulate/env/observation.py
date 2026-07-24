"""多臂多相机观测采集器。

从 MuJoCo 状态和相机缓存构造统一观测。
每臂独立采集 joint pos/gripper pos/last_action，
每相机独立渲染（各有自己的 fps）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config_loader import RobotConfig, CameraConfig
from .camera import CameraSensor
from .mujoco_interface import MujocoInterface


class ObservationCollector:
    """多臂多相机观测采集器。

    collect() 返回:
      {
        "state": { "arm_joint_pos": np.ndarray, "gripper_pos": np.ndarray,
                   "last_action": np.ndarray },
        "images": { camera_name: {"rgb": ..., "depth": ...}, ... },
        "task_id": int,
      }
    """

    def __init__(
        self,
        mj: MujocoInterface,
        cameras: dict[str, CameraSensor],
        robots: list[RobotConfig],
    ) -> None:
        self._mj = mj
        self.cameras = cameras
        self.robots = robots

        total_action_dim = sum(r.n_arm_joints + r.n_gripper_joints
                               for r in robots)
        self._last_action = np.zeros(total_action_dim, dtype=np.float32)

    # ── 公开接口 ───────────────────────────────────────

    def reset(self) -> None:
        self._last_action.fill(0.0)

    def update_last_action(self, action: np.ndarray) -> None:
        arr = np.asarray(action, dtype=np.float32).ravel()
        if len(arr) == len(self._last_action):
            self._last_action = arr.copy()
        else:
            self._last_action[:min(len(arr), len(self._last_action))] = arr[:len(self._last_action)]

    def collect(self, task_id: int) -> dict[str, Any]:
        """采集一帧观测。

        state 中每个 key 的值是所有臂拼接后的一维 float32 数组，
        排列顺序按 robots 列表的顺序。
        """
        arm_pos_parts = []
        grip_pos_parts = []
        for r in self.robots:
            arm_pos_parts.append(
                self._mj.get_joint_qpos(r.prefixed_arm_joints).astype(np.float32))
            grip_pos_parts.append(
                self._mj.get_joint_qpos(r.prefixed_gripper_joints).astype(np.float32))

        state = {
            "arm_joint_pos": np.concatenate(arm_pos_parts),
            "gripper_pos": np.concatenate(grip_pos_parts),
            "last_action": self._last_action.copy(),
        }

        images = {}
        for name, cam in self.cameras.items():
            frame = cam.read(copy=True)
            images[name] = frame

        return {"state": state, "images": images, "task_id": int(task_id)}
