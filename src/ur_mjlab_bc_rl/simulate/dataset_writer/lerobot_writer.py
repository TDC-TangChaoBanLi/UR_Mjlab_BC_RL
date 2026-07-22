"""LeRobot 数据集写入器。

将 Episode 数据写入 LeRobotDataset，支持原生深度采集。

使用 LeRobot v3.0 格式：
- RGB: 3 通道 uint8 video (RGBEncoderConfig)
- Depth: 单通道 float32 米值 → 12-bit 量化 video (DepthEncoderConfig)
- State: float32 parquet
- Action: float32 parquet
"""

from __future__ import annotations

import contextlib
import gc
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from lerobot.configs import DepthEncoderConfig, RGBEncoderConfig
from .episode import DEFAULT_STATE_KEYS, Episode, flatten_state

# ── 抑制编码器日志 ─────────────────────────────────────

os.environ.setdefault("FFMPEG_LOGLEVEL", "error")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
for _name in ("lerobot", "datasets", "PIL", "torchvision", "ffmpeg", "av"):
    logging.getLogger(_name).setLevel(logging.WARNING)


@contextlib.contextmanager
def _quiet_stderr():
    """静默 ffmpeg stderr 输出。"""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old, 2)
        os.close(old)
        os.close(devnull)


# ── 配置 ───────────────────────────────────────────────

@dataclass(slots=True)
class LeRobotDatasetConfig:
    """LeRobot 数据集写入配置。"""

    repo_id: str
    root: str | Path
    fps: int
    state_dim: int
    action_dim: int
    image_height: int
    image_width: int

    state_keys: Sequence[str] = field(
        default_factory=lambda: DEFAULT_STATE_KEYS
    )
    robot_type: str = "mujoco_ur5"
    use_rgb: bool = True
    use_depth: bool = True

    # 深度编码器（LeRobot 原生）
    depth_min: float = 0.1
    depth_max: float = 2.0

    # 编码参数
    streaming_encoding: bool = True
    batch_encoding_size: int = 1
    encoder_threads: int | None = 4
    encoder_queue_maxsize: int = 30
    image_writer_threads: int = 0
    image_writer_processes: int = 0

    def resolved_root(self) -> Path:
        return Path(self.root).expanduser().resolve()

    def build_features(self) -> dict[str, dict[str, Any]]:
        """构建 LeRobot v3.0 特征声明（含原生深度）。"""
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
        H, W = int(self.image_height), int(self.image_width)

        if self.use_rgb:
            features["observation.images.rgb"] = {
                "dtype": "video",
                "shape": (3, H, W),
                "names": ["channel", "height", "width"],
            }

        if self.use_depth:
            # 原生深度：单通道 + is_depth_map 标记
            features["observation.images.depth"] = {
                "dtype": "video",
                "shape": (1, H, W),
                "names": ["channel", "height", "width"],
                "info": {"is_depth_map": True},
            }

        return features


# ── 写入器 ─────────────────────────────────────────────

class LeRobotDatasetWriter:
    """MuJoCo RGB-D 数据 → LeRobotDataset 写入器。

    支持 LeRobot 原生深度：
    - 深度以 float32 米值传入
    - 使用 DepthEncoderConfig 进行 12-bit 量化编码
    - 读取时自动反量化
    """

    def __init__(self, dataset: Any, config: LeRobotDatasetConfig) -> None:
        self.dataset = dataset
        self.config = config
        self.root = config.resolved_root()
        self._closed = False
        self._pending_steps = 0
        self.episodes_written = 0

    @classmethod
    def create_new(
        cls,
        config: LeRobotDatasetConfig,
        *,
        overwrite: bool = False,
    ) -> "LeRobotDatasetWriter":
        """创建新数据集。"""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = config.resolved_root()
        if overwrite and root.exists():
            shutil.rmtree(root)

        depth_encoder = DepthEncoderConfig(
            depth_min=float(config.depth_min),
            depth_max=float(config.depth_max),
        )

        kwargs = dict(
            repo_id=config.repo_id,
            root=root,
            fps=int(config.fps),
            robot_type=config.robot_type,
            features=config.build_features(),
            use_videos=True,
            streaming_encoding=bool(config.streaming_encoding),
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

    @classmethod
    def resume_existing(
        cls, config: LeRobotDatasetConfig
    ) -> "LeRobotDatasetWriter":
        """追加到已有数据集。"""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = config.resolved_root()
        if not (root / "meta" / "info.json").exists():
            raise FileNotFoundError(f"不是有效的 LeRobotDataset: {root}")

        kwargs = dict(
            repo_id=config.repo_id,
            root=root,
            streaming_encoding=bool(config.streaming_encoding),
            batch_encoding_size=int(config.batch_encoding_size),
            encoder_threads=config.encoder_threads,
            encoder_queue_maxsize=int(config.encoder_queue_maxsize),
            image_writer_threads=int(config.image_writer_threads),
            image_writer_processes=int(config.image_writer_processes),
        )

        with _quiet_stderr():
            dataset = LeRobotDataset.resume(**kwargs)
        return cls(dataset, config)

    # ── 写入接口 ───────────────────────────────────────

    def add_step(
        self, obs: dict, action: np.ndarray, *, task_label: str
    ) -> None:
        """写入一帧。"""
        self._ensure_open()
        frame = self._make_frame(obs, action, task_label=task_label)
        with _quiet_stderr():
            self.dataset.add_frame(frame)
        self._pending_steps += 1
        del frame

    def append_episode(
        self,
        episode: Episode,
        *,
        task_label: str,
        clear_source: bool = True,
    ) -> None:
        """写入整条 Episode。"""
        try:
            for obs, action in episode.iter_steps():
                self.add_step(obs, action, task_label=task_label)
            self.save_current_episode()
        except Exception:
            self.discard_current_episode()
            raise
        finally:
            if clear_source:
                episode.clear()
                gc.collect()

    def append_step_batch(
        self,
        observations: Sequence[dict],
        actions: Sequence[np.ndarray],
        *,
        task_label: str,
    ) -> None:
        """批量写入帧。"""
        if len(observations) != len(actions):
            raise ValueError(
                f"batch 长度不一致: "
                f"{len(observations)} vs {len(actions)}"
            )
        for obs, action in zip(observations, actions, strict=True):
            self.add_step(obs, action, task_label=task_label)

    def save_current_episode(self) -> None:
        """保存当前 episode buffer。"""
        self._ensure_open()
        if self._pending_steps <= 0:
            raise RuntimeError("当前没有待保存的 frame。")
        with _quiet_stderr():
            self.dataset.save_episode()
        self._pending_steps = 0
        self.episodes_written += 1
        gc.collect()

    def discard_current_episode(self) -> None:
        """丢弃当前 episode buffer。"""
        if self._closed:
            return
        if hasattr(self.dataset, "clear_episode_buffer"):
            with _quiet_stderr():
                self.dataset.clear_episode_buffer(delete_images=True)
        self._pending_steps = 0
        gc.collect()

    def checkpoint(self) -> None:
        """finalize + resume 释放编码器状态。"""
        self._ensure_open()
        if self._pending_steps:
            raise RuntimeError("还有未保存 frame，不能 checkpoint。")

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        with _quiet_stderr():
            self.dataset.finalize()
        with _quiet_stderr():
            self.dataset = LeRobotDataset.resume(
                repo_id=self.config.repo_id,
                root=self.config.resolved_root(),
                streaming_encoding=self.config.streaming_encoding,
                batch_encoding_size=self.config.batch_encoding_size,
                encoder_threads=self.config.encoder_threads,
                encoder_queue_maxsize=self.config.encoder_queue_maxsize,
                image_writer_threads=self.config.image_writer_threads,
                image_writer_processes=self.config.image_writer_processes,
            )
        gc.collect()

    def finalize(self) -> Path:
        """完成写入，关闭数据集。"""
        if self._closed:
            return self.root
        if self._pending_steps:
            raise RuntimeError(
                "还有未保存 frame。请先 save_current_episode() "
                "或 discard_current_episode()。"
            )
        with _quiet_stderr():
            self.dataset.finalize()
        self._closed = True
        gc.collect()
        return self.root

    # ── 帧构建 ─────────────────────────────────────────

    def _make_frame(
        self, obs: dict, action: np.ndarray, *, task_label: str
    ) -> dict[str, Any]:
        """将观测+动作转为 LeRobotDataset.add_frame 格式。"""
        cfg = self.config
        H, W = int(cfg.image_height), int(cfg.image_width)

        # state
        state = flatten_state(obs["state"], state_keys=cfg.state_keys)
        if state.shape != (int(cfg.state_dim),):
            raise ValueError(
                f"state 维度错误，期望 {(cfg.state_dim,)}, 实际 {state.shape}"
            )

        # action
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.shape != (int(cfg.action_dim),):
            raise ValueError(
                f"action 维度错误，期望 {(cfg.action_dim,)}, "
                f"实际 {action_arr.shape}"
            )

        frame: dict[str, Any] = {
            "observation.state": state,
            "action": np.ascontiguousarray(action_arr),
            "task": str(task_label),
        }

        if cfg.use_rgb:
            rgb = obs["rgb"]
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            if rgb.shape[:2] != (H, W):
                raise ValueError(
                    f"rgb 分辨率错误，期望 {(H, W)}, 实际 {rgb.shape[:2]}"
                )
            frame["observation.images.rgb"] = rgb.transpose(2, 0, 1)

        if cfg.use_depth:
            depth = np.asarray(obs["depth"], dtype=np.float32)
            if depth.ndim == 3:
                if depth.shape[0] == 1:
                    depth = depth[0]
                elif depth.shape[-1] == 1:
                    depth = depth[..., 0]
            if depth.shape[:2] != (H, W):
                raise ValueError(
                    f"depth 分辨率错误，期望 {(H, W)}, "
                    f"实际 {depth.shape[:2]}"
                )
            # LeRobot 原生深度：单通道 float32 米值
            frame["observation.images.depth"] = depth[np.newaxis, :, :]

        return frame

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                "LeRobotDatasetWriter 已 finalize，不能继续写入。"
            )

    def __enter__(self) -> "LeRobotDatasetWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.discard_current_episode()
        if not self._closed:
            try:
                self.finalize()
            except RuntimeError:
                self.discard_current_episode()
                self.finalize()
