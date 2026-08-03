"""RGB-D 相机传感器 — 精简自 simulate/env/camera.py。

每个相机实例绑定 MuJoCo 场景中的一个 camera 实体，
支持 capture → read 缓存模式。
"""

from __future__ import annotations

import numpy as np
import mujoco

from .mujoco_wrapper import MujocoWrapper


class CameraSensor:
    """单目 RGB-D 相机。

    封装 MuJoCo Renderer，每次 capture() 渲染新帧，
    read() 返回内部缓存（零拷贝）。

    Args:
        mj: MujocoWrapper 实例。
        camera_name: MuJoCo 相机名称。
        image_size: (height, width)。
    """

    def __init__(
        self,
        mj: MujocoWrapper,
        camera_name: str,
        image_size: tuple[int, int] = (480, 640),
    ) -> None:
        mj._ensure_env()
        self._mj = mj
        self._camera_name = camera_name
        H, W = image_size

        try:
            self._camera_id = mj.model.camera(camera_name).id
        except KeyError as exc:
            raise ValueError(f"相机不存在: {camera_name!r}") from exc

        self._renderer: mujoco.Renderer | None = mujoco.Renderer(
            mj.model, height=int(H), width=int(W),
        )
        self._rgb = np.zeros((H, W, 3), dtype=np.uint8)
        self._depth = np.zeros((H, W), dtype=np.float32)

    @property
    def height(self) -> int:
        return int(self._rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self._rgb.shape[1])

    # ── 采集 ───────────────────────────────────────────

    def capture(self) -> None:
        """渲染一帧 RGB + Depth。"""
        self._ensure_open()
        assert self._renderer is not None
        self._renderer.update_scene(self._mj.data, camera=self._camera_id)
        self._renderer.disable_depth_rendering()
        # render() 返回的数组已经是 contiguous uint8 (H,W,3)，无需 ascontiguousarray
        self._rgb = self._renderer.render()
        self._renderer.enable_depth_rendering()
        # render() 返回的深度数组已经是 contiguous float32 (H,W)，无需转换
        self._depth = self._renderer.render()

    def read(self, *, copy: bool = False) -> dict[str, np.ndarray]:
        """读取最新帧。

        Returns:
            {"rgb": uint8 (H,W,3), "depth": float32 (H,W)}
        """
        if copy:
            return {"rgb": self._rgb.copy(), "depth": self._depth.copy()}
        return {"rgb": self._rgb, "depth": self._depth}

    def read_rgb(self, *, copy: bool = False) -> np.ndarray:
        """仅返回 RGB 图像。"""
        return self._rgb.copy() if copy else self._rgb

    def read_depth(self, *, copy: bool = False) -> np.ndarray:
        """仅返回深度图像（米制 float32）。"""
        return self._depth.copy() if copy else self._depth

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ── 内部 ───────────────────────────────────────────

    def _ensure_open(self) -> None:
        if self._renderer is None:
            raise RuntimeError("CameraSensor 已关闭")
