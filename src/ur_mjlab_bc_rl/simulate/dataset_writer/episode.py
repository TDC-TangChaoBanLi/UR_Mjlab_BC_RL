"""Episode 内存数据结构。

支持多相机图像存储。
observations[i]["images"] = {camera_name: {"rgb": ..., "depth": ...}, ...}
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.max(initial=0) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


def _copy_images(images: dict, copy_arrays: bool) -> dict:
    """深拷贝多相机图像。"""
    out = {}
    for cam, frame in images.items():
        rgb = _normalize_rgb(frame["rgb"])
        if copy_arrays:
            rgb = rgb.copy()
        depth = _normalize_depth(frame["depth"])
        if copy_arrays:
            depth = depth.copy()
        out[cam] = {"rgb": rgb, "depth": depth}
    return out


def _copy_state(state: dict, copy_arrays: bool) -> dict:
    """深拷贝状态 dict。"""
    return {
        k: np.asarray(v, dtype=np.float32).copy()
        if copy_arrays else np.asarray(v, dtype=np.float32)
        for k, v in state.items()
    }


class Episode:
    """单条轨迹。"""

    def __init__(self) -> None:
        self.observations: list[dict] = []
        self.actions: list[np.ndarray] = []

    def add(
        self, obs: dict[str, Any], action: np.ndarray,
        *, copy_arrays: bool = True
    ) -> None:
        self.observations.append({
            "state": _copy_state(obs["state"], copy_arrays),
            "images": _copy_images(obs["images"], copy_arrays),
            "task_id": int(obs.get("task_id", 0)),
        })
        self.actions.append(
            np.asarray(action, dtype=np.float32).ravel().copy())

    def __len__(self) -> int:
        return len(self.observations)

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
