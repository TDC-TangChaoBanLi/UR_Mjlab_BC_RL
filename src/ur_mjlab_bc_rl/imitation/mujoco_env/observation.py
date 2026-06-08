"""观测采集器：从 MuJoCo 状态和相机缓存构造统一观测。"""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import numpy as np
import torch

from .camera import CameraSensor
from .mujoco_interface import MujocoInterface


DEFAULT_ARM_JOINTS = [
    "ur_shoulder_pan_joint",
    "ur_shoulder_lift_joint",
    "ur_elbow_joint",
    "ur_wrist_1_joint",
    "ur_wrist_2_joint",
    "ur_wrist_3_joint",
]
DEFAULT_GRIPPER_JOINTS = ["robotiq_85_left_knuckle_joint"]
DEFAULT_STATE_KEYS = ("arm_joint_pos", "gripper_pos", "last_action")


class ObservationCollector:
    """从 MuJoCo 构造策略/数据集观测。

    默认状态字段：
        arm_joint_pos: 6
        gripper_pos: 1
        last_action: 7

    因此默认 state_dim = 14。
    """

    def __init__(
        self,
        mj_interface: MujocoInterface,
        camera: CameraSensor,
        arm_joint_names: Sequence[str] | None = None,
        gripper_joint_names: Sequence[str] | None = None,
        depth_min: float = 0.0,
        depth_max: float = 2.0,
        *,
        include_arm_vel: bool = False,
        include_ee_pose: bool = False,
        include_last_action: bool = True,
        tcp_site_name: str = "_tcp",
        action_dim: int = 7,
    ) -> None:
        self._mj = mj_interface
        self.camera = camera
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        if self.depth_max <= self.depth_min:
            raise ValueError("depth_max 必须大于 depth_min。")

        self._arm_joints = list(arm_joint_names) if arm_joint_names is not None else list(DEFAULT_ARM_JOINTS)
        self._gripper_joints = (
            list(gripper_joint_names) if gripper_joint_names is not None else list(DEFAULT_GRIPPER_JOINTS)
        )
        self.include_arm_vel = bool(include_arm_vel)
        self.include_ee_pose = bool(include_ee_pose)
        self.include_last_action = bool(include_last_action)
        self.tcp_site_name = tcp_site_name
        self.action_dim = int(action_dim)
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)

    @property
    def state_keys(self) -> tuple[str, ...]:
        keys = ["arm_joint_pos"]
        if self.include_arm_vel:
            keys.append("arm_joint_vel")
        keys.append("gripper_pos")
        if self.include_ee_pose:
            keys.extend(["ee_pos", "ee_quat"])
        if self.include_last_action:
            keys.append("last_action")
        return tuple(keys)

    @property
    def state_dim(self) -> int:
        dim = len(self._arm_joints)
        if self.include_arm_vel:
            dim += len(self._arm_joints)
        dim += len(self._gripper_joints)
        if self.include_ee_pose:
            dim += 3 + 4
        if self.include_last_action:
            dim += self.action_dim
        return dim

    def update_last_action(self, action: np.ndarray) -> None:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size < self.action_dim:
            raise ValueError(f"action 维度不足，期望至少 {self.action_dim}，实际 {arr.size}")
        self._last_action = arr[: self.action_dim].copy()

    def reset(self) -> None:
        self._last_action.fill(0.0)

    def close(self) -> None:
        self.camera.close()

    def collect(self, task_id: int = 0) -> dict:
        """构造一次观测。"""
        rgbd = self.camera.read(copy=False)
        depth_raw = rgbd["depth"]
        depth_norm = np.clip(
            (depth_raw - self.depth_min) / (self.depth_max - self.depth_min),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)

        return {
            "state": self._build_state(),
            "rgb": rgbd["rgb"],
            "depth": depth_norm,
            "task_id": int(task_id),
        }

    def _build_state(self) -> OrderedDict[str, np.ndarray]:
        state: OrderedDict[str, np.ndarray] = OrderedDict()
        state["arm_joint_pos"] = self._mj.get_joint_qpos(self._arm_joints).astype(np.float32, copy=False)

        if self.include_arm_vel:
            state["arm_joint_vel"] = self._mj.get_joint_qvel(self._arm_joints).astype(np.float32, copy=False)

        state["gripper_pos"] = self._mj.get_joint_qpos(self._gripper_joints).astype(np.float32, copy=False)

        if self.include_ee_pose:
            try:
                ee = np.asarray(self._mj.get_site_pose(self.tcp_site_name), dtype=np.float32)
            except Exception:
                ee = np.zeros(7, dtype=np.float32)
            state["ee_pos"] = ee[:3].astype(np.float32, copy=False)
            state["ee_quat"] = ee[3:7].astype(np.float32, copy=False)

        if self.include_last_action:
            state["last_action"] = self._last_action.copy()

        return state


def flatten_state_from_obs(obs: dict, state_keys: Sequence[str] | None = None) -> np.ndarray:
    state = obs["state"]
    if isinstance(state, dict):
        keys = list(state_keys) if state_keys is not None else list(state.keys())
        return np.concatenate([np.asarray(state[k], dtype=np.float32).reshape(-1) for k in keys])
    return np.asarray(state, dtype=np.float32).reshape(-1)


def convert_obs_to_model_input(
    obs: dict,
    device: str = "cpu",
    state_keys: Sequence[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """将 ObservationCollector 输出转为模型输入。

    输出：
        camera: [1,4,H,W]，RGB + depth
        state:  [1,D]
        task:   [1,1]
    """
    rgb = obs["rgb"].astype(np.float32).transpose(2, 0, 1) / 255.0
    depth = obs["depth"]
    if depth.dtype == np.uint8:
        depth = depth.astype(np.float32) / 255.0
    if depth.ndim == 2:
        depth = depth[None, :, :]

    camera_np = np.concatenate([rgb, depth.astype(np.float32)], axis=0)
    state_np = flatten_state_from_obs(obs, state_keys=state_keys).astype(np.float32)

    camera = torch.from_numpy(camera_np).unsqueeze(0).to(device)
    state = torch.from_numpy(state_np).unsqueeze(0).to(device)
    task = torch.tensor([[int(obs.get("task_id", 0))]], dtype=torch.long, device=device)
    return camera, state, task
