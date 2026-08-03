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

from ..utils.config_loader import CameraConfig
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
    encoder_queue_maxsize: int = 90
    image_writer_threads: int = 0
    image_writer_processes: int = 0

    def resolved_root(self) -> Path:
        return Path(self.root).expanduser().resolve()

    def build_features(self, dataset_cfg=None) -> dict[str, dict[str, Any]]:
        """构建 LeRobot features dict。

        多速率模式：每个数据源独立 feature，保留真实子采样数。
          例：observation.state.joint_position → (3, 14)
              observation.state.sensor_force  → (2, 3)
        """
        features: dict[str, dict[str, Any]] = {}

        if dataset_cfg is not None:
            # 收集 action namespace
            action_names_all: list[str] = []
            for src in dataset_cfg.sources:
                if src.name.startswith("state."):
                    key = "observation." + src.name
                    features[key] = {
                        "dtype": "float32",
                        "shape": (src.num_subs, src.dim_per_sub),
                        "names": src.source_names if src.source_names else None,
                    }
                else:
                    action_names_all.extend(src.source_names)
            # action 统一为 "action": (max_subs, action_dim_per_sub)
            max_subs = max(
                (s.num_subs for s in dataset_cfg.sources if not s.name.startswith("state.")),
                default=1,
            )
            total_action_dim = sum(
                s.dim_per_sub for s in dataset_cfg.sources if not s.name.startswith("state.")
            )
            features["action"] = {
                "dtype": "float32",
                "shape": (max_subs, total_action_dim),
                "names": action_names_all if action_names_all else None,
            }
        else:
            features["observation.state"] = {
                "dtype": "float32", "shape": (int(self.state_dim),), "names": None,
            }
            features["action"] = {
                "dtype": "float32", "shape": (int(self.action_dim),), "names": None,
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
        dataset_cfg=None,
    ) -> "LeRobotDatasetWriter":
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = config.resolved_root()
        if overwrite and root.exists():
            shutil.rmtree(root)

        depth_min, depth_max = (0.1, 2.0)
        if dataset_cfg is not None and dataset_cfg.depth_range:
            depth_min, depth_max = dataset_cfg.depth_range
        elif config.cameras:
            depth_min = float(config.cameras[0].depth_range[0])
            depth_max = float(config.cameras[0].depth_range[1])
        depth_encoder = DepthEncoderConfig(
            depth_min=depth_min,
            depth_max=depth_max,
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
            features=config.build_features(dataset_cfg),
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

        帧直接写入 LeRobot（增量），不缓存整个 episode，内存恒定。
        返回 (callback, flush, discard) 三元组：
          - callback(obs, action): 格式化并写入帧
          - flush(task_label):     保存 episode
          - discard():              丢弃当前 episode 帧

        避免失败帧污染数据集，且内存占用恒定（~几 MB）。
        """
        cameras = self.config.cameras
        _frame_count = 0
        _buffer: list[dict[str, Any]] = []  # 仅用于 discard 时的回退

        def _callback(obs, action):
            nonlocal _frame_count
            frame = self._format_frame(obs, action, cameras)
            _buffer.append(frame)
            _frame_count += 1

        def _flush(task_label: str = ""):
            nonlocal _frame_count
            if _frame_count == 0:
                return
            for frame in _buffer:
                frame["task"] = task_label
                with _quiet_stderr():
                    self.dataset.add_frame(frame)
            with _quiet_stderr():
                self.dataset.save_episode()
            self.episodes_written += 1
            _buffer.clear()
            _frame_count = 0

        def _discard():
            nonlocal _frame_count
            _buffer.clear()
            _frame_count = 0

        return _callback, _flush, _discard

    @staticmethod
    def _format_frame(obs, _action, cameras) -> dict[str, Any]:
        """将多速率 obs dict 格式化为 LeRobot add_frame 所需的 dict。

        每个数据源独立点分隔 key：
          "observation.state.joint.position": (3, 14)
          "observation.state.sensor.force":   (2, 6)
          "action":                           (3, 14)
        """
        obs_state = obs.get("state", {})
        obs_action = obs.get("action", {})
        frame: dict[str, Any] = {}

        for src_name, arr in obs_state.items():
            frame[f"observation.{src_name}"] = arr.astype(np.float32)
        if obs_action:
            # 合并 action 子源（如 action.joint.position）为单个 "action" key
            action_parts = []
            for arr in obs_action.values():
                action_parts.append(arr.astype(np.float32))
            frame["action"] = np.concatenate(action_parts, axis=-1)

        images = obs.get("images", {})
        for c in cameras:
            if c.name in images:
                frame[f"observation.images.{c.name}.rgb"] = images[c.name]["rgb"].copy()
                frame[f"observation.images.{c.name}.depth"] = images[c.name]["depth"].copy()[np.newaxis, ...]
        return frame

    def finalize(self) -> str:
        if not self._closed:
            self.dataset.finalize()
            self._closed = True
        return str(self.root)

    def close(self) -> None:
        self.finalize()
