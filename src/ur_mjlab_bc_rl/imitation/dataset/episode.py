"""Episode 数据结构与轻量工具。

本文件只保留“内存中的单条轨迹”表示，不再写临时文件。

设计目标：
1. worker 进程内部只缓存当前 episode；
2. episode 成功后通过 multiprocessing.Queue 分批发送给主进程；
3. 主进程收到 batch 后立即写入 LeRobotDataset，不再累计 collected_episodes；
4. 深度图在进入 Episode 时默认转为 uint8，减少 4 倍 IPC/内存压力。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np


DEFAULT_STATE_KEYS: tuple[str, ...] = (
    "arm_joint_pos",
    "gripper_pos",
    "last_action",
)


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    """返回 uint8 HWC RGB。"""
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


def normalize_depth_to_uint8(depth: np.ndarray) -> np.ndarray:
    """返回 uint8 HW depth。

    输入可以是：
    - float32/float64，默认已归一化到 [0,1]；
    - uint8，直接返回；
    - 1xHxW 或 HxWx1，会压成 HW。
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

    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)

    arr = np.clip(arr, 0.0, 1.0)
    return np.ascontiguousarray((arr * 255.0).astype(np.uint8))


def flatten_state(
    state: dict | np.ndarray | Sequence[float],
    state_keys: Sequence[str] | None = DEFAULT_STATE_KEYS,
) -> np.ndarray:
    """将状态转成 float32 一维向量。

    对 dict 类型，强烈建议显式传入 state_keys，避免依赖 dict.values() 的隐式顺序。
    """
    if isinstance(state, dict):
        keys = list(state_keys) if state_keys is not None else sorted(state.keys())
        parts: list[np.ndarray] = []
        for key in keys:
            if key not in state:
                raise KeyError(f"state 中缺少 key: {key}")
            parts.append(np.asarray(state[key], dtype=np.float32).reshape(-1))
        flat = np.concatenate(parts, axis=0) if parts else np.empty((0,), dtype=np.float32)
    else:
        flat = np.asarray(state, dtype=np.float32).reshape(-1)

    return np.ascontiguousarray(flat.astype(np.float32, copy=False))


def copy_observation(
    obs: dict,
    *,
    copy_arrays: bool = True,
    depth_to_uint8: bool = True,
) -> dict:
    """复制一帧观测，避免后续渲染覆盖底层数组。"""
    state = obs["state"]
    if isinstance(state, dict):
        state_copy = OrderedDict(
            (k, np.asarray(v, dtype=np.float32).copy() if copy_arrays else np.asarray(v, dtype=np.float32))
            for k, v in state.items()
        )
    else:
        state_copy = np.asarray(state, dtype=np.float32).copy() if copy_arrays else np.asarray(state, dtype=np.float32)

    rgb = normalize_rgb(obs["rgb"])
    if copy_arrays:
        rgb = rgb.copy()

    if depth_to_uint8:
        depth = normalize_depth_to_uint8(obs["depth"])
    else:
        depth = np.asarray(obs["depth"], dtype=np.float32)
    if copy_arrays:
        depth = depth.copy()

    return {
        "state": state_copy,
        "rgb": rgb,
        "depth": depth,
        "task_id": int(obs.get("task_id", 0)),
    }


@dataclass
class Episode:
    """单条轨迹。

    observations: 每步包含 state/rgb/depth/task_id。
    actions: 每步动作，通常为 [7]，即 6 个机械臂关节 + 1 个夹爪。

    注意：本类只用于 worker 当前正在采集的 episode，不建议在主进程累计很多个 Episode。
    """

    observations: list[dict] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)

    def add(self, obs: dict, action: np.ndarray, *, copy_arrays: bool = True) -> None:
        self.observations.append(copy_observation(obs, copy_arrays=copy_arrays, depth_to_uint8=True))
        self.actions.append(np.asarray(action, dtype=np.float32).reshape(-1).copy())

    def __len__(self) -> int:
        return len(self.observations)

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()

    def iter_steps(self) -> Iterator[tuple[dict, np.ndarray]]:
        if len(self.observations) != len(self.actions):
            raise ValueError(
                f"observations/actions 数量不一致: {len(self.observations)} vs {len(self.actions)}"
            )
        yield from zip(self.observations, self.actions, strict=True)

    def iter_batches(self, batch_size: int = 16) -> Iterator[tuple[list[dict], list[np.ndarray]]]:
        """按 batch 分块迭代，便于通过 multiprocessing.Queue 发送。"""
        if batch_size <= 0:
            raise ValueError("batch_size 必须 > 0")
        n = len(self)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            yield self.observations[start:end], self.actions[start:end]
