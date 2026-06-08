"""MuJoCo RGB-D 相机传感器。"""

from __future__ import annotations

import numpy as np
import mujoco

from .mujoco_interface import MujocoInterface


class CameraSensor:
    """单目 RGB-D 相机。

    Args:
        mj_interface: MuJoCo 仿真接口。
        camera_name: MuJoCo 相机名称。
        image_size: 输出图像尺寸 (H, W)。
    """

    def __init__(
        self,
        mj_interface: MujocoInterface,
        camera_name: str = "realsense_link_CAMERA",
        image_size: tuple[int, int] = (128, 128),
    ) -> None:
        self._mj = mj_interface
        self._camera_name = camera_name
        H, W = image_size

        try:
            self._camera_id = self._mj.model.camera(camera_name).id
        except KeyError as exc:
            raise ValueError(f"相机不存在: {camera_name!r}") from exc

        self._renderer: mujoco.Renderer | None = mujoco.Renderer(
            self._mj.model,
            height=int(H),
            width=int(W),
        )
        self._rgb = np.zeros((H, W, 3), dtype=np.uint8)
        self._depth = np.zeros((H, W), dtype=np.float32)

    @property
    def image_size(self) -> tuple[int, int]:
        return int(self._rgb.shape[0]), int(self._rgb.shape[1])

    def capture(self) -> None:
        """渲染一帧 RGB 和 depth。"""
        self._ensure_open()
        self._rgb = self._render_rgb()
        self._depth = self._render_depth()

    def read(self, *, copy: bool = False) -> dict[str, np.ndarray]:
        """读取最近一次采集的 RGB-D。

        copy=False 时返回内部缓存引用；通常由 Episode.add 再统一复制。
        """
        if copy:
            return {"rgb": self._rgb.copy(), "depth": self._depth.copy()}
        return {"rgb": self._rgb, "depth": self._depth}

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _render_rgb(self) -> np.ndarray:
        self._ensure_open()
        assert self._renderer is not None
        self._renderer.disable_depth_rendering()
        self._renderer.update_scene(self._mj.data, camera=self._camera_id)
        return np.ascontiguousarray(self._renderer.render())

    def _render_depth(self) -> np.ndarray:
        self._ensure_open()
        assert self._renderer is not None
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self._mj.data, camera=self._camera_id)
        return np.ascontiguousarray(self._renderer.render().astype(np.float32, copy=False))

    def _ensure_open(self) -> None:
        if self._renderer is None:
            raise RuntimeError("CameraSensor 已关闭，不能继续 capture/read。")
