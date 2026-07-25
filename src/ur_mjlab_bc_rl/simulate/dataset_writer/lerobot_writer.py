"""LeRobot 数据集写入器。

将 Episode 数据写入 LeRobotDataset，支持多相机原生深度。
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from lerobot.configs import DepthEncoderConfig
from lerobot.configs import RGBEncoderConfig

from ..config_loader import CameraConfig
from .episode import Episode

os.environ.setdefault("FFMPEG_LOGLEVEL", "error")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
for _name in ("lerobot", "datasets", "PIL", "torchvision", "ffmpeg", "av"):
    logging.getLogger(_name).setLevel(logging.WARNING)


@contextlib.contextmanager
def _quiet_stderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old, 2)
        os.close(old)
        os.close(devnull)


# ═══════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════

@dataclass(slots=True)
class LeRobotDatasetConfig:
    repo_id: str
    root: str | Path
    fps: int
    state_dim: int
    action_dim: int
    cameras: list[CameraConfig] = field(default_factory=list)
    robot_type: str = "mujoco_ur5"
    use_rgb: bool = True
    use_depth: bool = True
    streaming_encoding: bool = True
    vcodec: str = "auto"               # "auto" 优先硬件编码 h264_nvenc ，回退 libsvtav1
    preset: str | None = None           # h264 预设 (ultrafast/fast/medium)，None=默认
    batch_encoding_size: int = 1
    encoder_threads: int | None = 4
    encoder_queue_maxsize: int = 30
    image_writer_threads: int = 0
    image_writer_processes: int = 0

    def resolved_root(self) -> Path:
        return Path(self.root).expanduser().resolve()

    def build_features(self) -> dict[str, dict[str, Any]]:
        features: dict[str, dict[str, Any]] = {
            "observation.state": {
                "dtype": "float32",
                "shape": (int(self.state_dim),),
                "names": [f"s{i}" for i in range(int(self.state_dim))],
            },
            "action": {
                "dtype": "float32",
                "shape": (int(self.action_dim),),
                "names": [f"a{i}" for i in range(int(self.action_dim))],
            },
        }
        for c in self.cameras:
            H, W = int(c.height), int(c.width)
            prefix = f"observation.images.{c.name}"
            if self.use_rgb:
                features[f"{prefix}.rgb"] = {
                    "dtype": "video", "shape": (3, H, W),
                    "names": ["channel", "height", "width"],
                }
            if self.use_depth:
                features[f"{prefix}.depth"] = {
                    "dtype": "video", "shape": (1, H, W),
                    "names": ["channel", "height", "width"],
                    "info": {"is_depth_map": True},
                }
        return features


# ═══════════════════════════════════════════════════════
# 写入器
# ═══════════════════════════════════════════════════════

class LeRobotDatasetWriter:
    def __init__(self, dataset: Any, config: LeRobotDatasetConfig) -> None:
        self.dataset = dataset
        self.config = config
        self.root = config.resolved_root()
        self._closed = False
        self.episodes_written = 0

    @classmethod
    def create_new(
        cls, config: LeRobotDatasetConfig, *, overwrite: bool = False,
    ) -> "LeRobotDatasetWriter":
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = config.resolved_root()
        if overwrite and root.exists():
            shutil.rmtree(root)

        depth_encoder = DepthEncoderConfig(
            depth_min=float(config.cameras[0].depth_range[0]) if config.cameras else 0.1,
            depth_max=float(config.cameras[0].depth_range[1]) if config.cameras else 2.0,
        )

        rgb_encoder = RGBEncoderConfig(
            vcodec=config.vcodec,
            preset=config.preset,
            g=None,  # auto GOP，NVENC 需要 g > b_frames+1
        )

        kwargs = dict(
            repo_id=config.repo_id,
            root=root,
            fps=int(config.fps),
            robot_type=config.robot_type,
            features=config.build_features(),
            use_videos=True,
            streaming_encoding=bool(config.streaming_encoding),
            rgb_encoder=rgb_encoder,
            batch_encoding_size=int(config.batch_encoding_size),
            encoder_threads=config.encoder_threads,
            encoder_queue_maxsize=int(config.encoder_queue_maxsize),
            image_writer_threads=int(config.image_writer_threads),
            image_writer_processes=int(config.image_writer_processes),
            depth_encoder=depth_encoder,
        )

        with _quiet_stderr():
            dataset = LeRobotDataset.create(**kwargs)
        return cls(dataset, config)

    def append_episode(self, episode: Episode, *, task_label: str = "") -> None:
        if len(episode) == 0:
            return

        cameras = self.config.cameras
        for obs, action in zip(episode.observations, episode.actions):
            frame = self._format_frame(obs, action, cameras)
            frame["task"] = task_label
            with _quiet_stderr():
                self.dataset.add_frame(frame)

        with _quiet_stderr():
            self.dataset.save_episode()

        self.episodes_written += 1

    def make_stream_callback(self):
        """创建流式写入回调 + flush/discard。

        回调将帧缓存在本地列表，不直接写入 LeRobot。
        返回 (callback, flush, discard) 三元组：
          - callback(obs, action): 缓存一帧
          - flush(task_label):     将缓存帧写入 LeRobot 并 save_episode
          - discard():              丢弃缓存帧（失败 episode 用）

        避免失败帧污染数据集，且内存占用恒定（~几 MB）。
        """
        cameras = self.config.cameras
        buffer: list[dict[str, Any]] = []

        def _callback(obs, action):
            frame = self._format_frame(obs, action, cameras)
            buffer.append(frame)

        def _flush(task_label: str = ""):
            if not buffer:
                return
            for frame in buffer:
                frame["task"] = task_label
                with _quiet_stderr():
                    self.dataset.add_frame(frame)
            with _quiet_stderr():
                self.dataset.save_episode()
            self.episodes_written += 1
            buffer.clear()

        def _discard():
            buffer.clear()

        return _callback, _flush, _discard

    @staticmethod
    def _format_frame(obs, action, cameras) -> dict[str, Any]:
        """将 (obs, action) 格式化为 LeRobot add_frame 所需的 dict。"""
        state_parts = [
            obs["state"]["arm_joint_pos"],
            obs["state"]["gripper_pos"],
            obs["state"]["last_action"],
        ]
        frame: dict[str, Any] = {
            "observation.state": np.concatenate(
                [np.asarray(p, dtype=np.float32).ravel() for p in state_parts]),
            "action": np.asarray(action, dtype=np.float32).ravel(),
        }
        images = obs.get("images", {})
        for c in cameras:
            if c.name in images:
                frame[f"observation.images.{c.name}.rgb"] = images[c.name]["rgb"]
                frame[f"observation.images.{c.name}.depth"] = \
                    images[c.name]["depth"][np.newaxis, ...]
        return frame

    def finalize(self) -> str:
        if not self._closed:
            self.dataset.finalize()
            self._closed = True
        return str(self.root)

    def close(self) -> None:
        self.finalize()
