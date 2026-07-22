"""Episode 内存数据结构。

单条轨迹的内存表示，用于仿真采集过程中缓存观测和动作。
采集完成后通过 LeRobotDataset 写入器持久化。

设计：
- 观测深度以 float32 米值存储（MuJoCo 原生单位）
- RGB 以 uint8 HWC 存储
- 动作以 float32 存储
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterator, Sequence

import numpy as np


DEFAULT_STATE_KEYS: tuple[str, ...] = (
    "arm_joint_pos",
    "gripper_pos",
    "last_action",
)


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    """标准化 RGB 为 uint8 HWC。

    兼容 float32 [0,1]、uint8、CHW 转置等格式。
    """
    arr = np.asarray(rgb)

    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"rgb 形状应为 HWC/CHW 三通道，实际 {arr.shape}")

    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.max(initial=0) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(arr)


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    """标准化深度为 float32 HW（米值）。

    输入：
    - float32 HW（直接返回）
    - float32 1xHxW 或 HxWx1 → HW
    """
    arr = np.asarray(depth)

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f"depth 只支持单通道，实际 {arr.shape}")

    if arr.ndim != 2:
        raise ValueError(f"depth 形状应为 HW，实际 {arr.shape}")

    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


def flatten_state(
    state: OrderedDict[str, np.ndarray],
    state_keys: Sequence[str] | None = DEFAULT_STATE_KEYS,
) -> np.ndarray:
    """将 dict 状态展平为一维 float32 向量。"""
    keys = list(state_keys) if state_keys is not None else sorted(state.keys())
    parts = [
        np.asarray(state[k], dtype=np.float32).reshape(-1) for k in keys
    ]
    return np.concatenate(parts) if parts else np.empty((0,), dtype=np.float32)


def copy_observation(obs: dict, *, copy_arrays: bool = True) -> dict:
    """安全复制一帧观测。"""
    state = obs["state"]
    if isinstance(state, dict):
        state_copy = OrderedDict(
            (k, np.asarray(v, dtype=np.float32).copy()
             if copy_arrays else np.asarray(v, dtype=np.float32))
            for k, v in state.items()
        )
    else:
        state_copy = (
            np.asarray(state, dtype=np.float32).copy()
            if copy_arrays
            else np.asarray(state, dtype=np.float32)
        )

    rgb = _normalize_rgb(obs["rgb"])
    if copy_arrays:
        rgb = rgb.copy()

    depth = _normalize_depth(obs["depth"])
    if copy_arrays:
        depth = depth.copy()

    return {
        "state": state_copy,
        "rgb": rgb,
        "depth": depth,
        "task_id": int(obs.get("task_id", 0)),
    }


class Episode:
    """单条轨迹。

    内存中的观测-动作序列缓存。
    observations[i] 包含采集步 i 的 state/rgb/depth/task_id。
    actions[i] 是在 obs[i] 后执行的动作。
    """

    def __init__(self) -> None:
        self.observations: list[dict] = []
        self.actions: list[np.ndarray] = []

    def add(
        self, obs: dict, action: np.ndarray, *, copy_arrays: bool = True
    ) -> None:
        """追加一帧。"""
        self.observations.append(
            copy_observation(obs, copy_arrays=copy_arrays)
        )
        self.actions.append(
            np.asarray(action, dtype=np.float32).reshape(-1).copy()
        )

    def __len__(self) -> int:
        return len(self.observations)

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()

    def iter_steps(self) -> Iterator[tuple[dict, np.ndarray]]:
        """逐帧迭代 (obs, action)。"""
        if len(self.observations) != len(self.actions):
            raise ValueError(
                f"obs/action 数量不一致: "
                f"{len(self.observations)} vs {len(self.actions)}"
            )
        return zip(self.observations, self.actions, strict=True)

    def iter_batches(
        self, batch_size: int = 16
    ) -> Iterator[tuple[list[dict], list[np.ndarray]]]:
        """按 batch 分块迭代（用于 IPC 传输）。"""
        n = len(self)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            yield self.observations[start:end], self.actions[start:end]
