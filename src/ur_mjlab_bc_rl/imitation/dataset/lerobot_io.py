"""LeRobot 数据集写入与读取。

全新接口：不再保留旧的 LeRobotSaver / append_episodes 兼容层。

核心原则：
1. LeRobotDataset 只在主进程中创建和写入；
2. worker 只负责采集 episode，并通过 Queue 分批发送；
3. 主进程收到一批 frame 后立即 add_frame，不保存临时文件，也不累计 collected_episodes；
4. 可通过 checkpoint_every_n_episodes 定期 finalize + resume 释放编码器/Writer 状态。
"""

from __future__ import annotations

import contextlib
import gc
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .episode import DEFAULT_STATE_KEYS, Episode, flatten_state, normalize_depth_to_uint8, normalize_rgb

os.environ.setdefault("FFMPEG_LOGLEVEL", "error")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("SVT_LOG", "0")
os.environ.setdefault("AV_LOG_FORCE_NOCOLOR", "1")
for _name in ("lerobot", "datasets", "PIL", "torchvision", "ffmpeg", "av"):
    logging.getLogger(_name).setLevel(logging.WARNING)


@contextlib.contextmanager
def quiet_native_stderr(enabled: bool = True):
    """临时静默 native stderr。

    ffmpeg / av / SVT 有时绕过 Python logging，直接写 fd=2。
    """
    if not enabled:
        yield
        return

    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        os.close(devnull)


@dataclass(slots=True)
class LeRobotDatasetConfig:
    repo_id: str
    root: str | Path
    fps: int
    state_dim: int
    action_dim: int
    image_height: int
    image_width: int

    state_keys: Sequence[str] = field(default_factory=lambda: DEFAULT_STATE_KEYS)
    robot_type: str = "mujoco_ur5"

    use_rgb: bool = True
    use_depth: bool = True

    streaming_encoding: bool = True
    batch_encoding_size: int = 1
    encoder_threads: int | None = 4
    encoder_queue_maxsize: int = 30
    image_writer_threads: int = 0
    image_writer_processes: int = 0

    quiet_stderr: bool = True

    def resolved_root(self) -> Path:
        return Path(self.root).expanduser().resolve()

    def build_features(self) -> dict[str, dict[str, Any]]:
        image_dtype = "video"
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
        if self.use_rgb:
            features["observation.images.rgb"] = {
                "dtype": image_dtype,
                "shape": (3, int(self.image_height), int(self.image_width)),
                "names": ["channel", "height", "width"],
            }
        if self.use_depth:
            features["observation.images.depth"] = {
                "dtype": image_dtype,
                "shape": (3, int(self.image_height), int(self.image_width)),
                "names": ["channel", "height", "width"],
            }
        return features


class LeRobotMujocoDatasetWriter:
    """MuJoCo RGB-D 数据到 LeRobotDataset 的写入器。"""

    def __init__(self, dataset: Any, config: LeRobotDatasetConfig) -> None:
        self.dataset = dataset
        self.config = config
        self.root = config.resolved_root()
        self.repo_id = config.repo_id
        self._closed = False
        self._pending_steps = 0
        self.episodes_written = 0

    @classmethod
    def create_new(
        cls,
        config: LeRobotDatasetConfig,
        *,
        overwrite: bool = False,
    ) -> "LeRobotMujocoDatasetWriter":
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = config.resolved_root()
        if overwrite and root.exists():
            shutil.rmtree(root)

        kwargs = dict(
            repo_id=config.repo_id,
            root=root,
            fps=int(config.fps),
            robot_type=config.robot_type,
            features=config.build_features(),
            streaming_encoding=bool(config.streaming_encoding),
            batch_encoding_size=int(config.batch_encoding_size),
            encoder_threads=config.encoder_threads,
            encoder_queue_maxsize=int(config.encoder_queue_maxsize),
            image_writer_threads=int(config.image_writer_threads),
            image_writer_processes=int(config.image_writer_processes),
        )

        with quiet_native_stderr(config.quiet_stderr):
            dataset = LeRobotDataset.create(**kwargs)
        return cls(dataset, config)

    @classmethod
    def resume_existing(cls, config: LeRobotDatasetConfig) -> "LeRobotMujocoDatasetWriter":
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = config.resolved_root()
        if not (root / "meta" / "info.json").exists():
            raise FileNotFoundError(f"不是有效的 LeRobotDataset 目录: {root}")

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

        with quiet_native_stderr(config.quiet_stderr):
            dataset = LeRobotDataset.resume(**kwargs)
        return cls(dataset, config)

    def add_step(self, obs: dict, action: np.ndarray, *, task_label: str) -> None:
        """写入当前 episode 的一帧。"""
        self._ensure_open()
        frame = self.make_frame(obs, action, task_label=task_label)
        with quiet_native_stderr(self.config.quiet_stderr):
            self.dataset.add_frame(frame)
        self._pending_steps += 1
        del frame

    def append_episode(self, episode: Episode, *, task_label: str, clear_source: bool = True) -> None:
        """写入一条内存中的 Episode。

        主进程不应累计很多 Episode；该接口主要用于收到 worker 的一条 episode 后立即写入。
        """
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
        """写入当前 episode 的一个 frame batch。"""
        if len(observations) != len(actions):
            raise ValueError(f"observations/actions batch 长度不一致: {len(observations)} vs {len(actions)}")
        for obs, action in zip(observations, actions, strict=True):
            self.add_step(obs, action, task_label=task_label)

    def save_current_episode(self, *, parallel_encoding: bool = True) -> None:
        self._ensure_open()
        if self._pending_steps <= 0:
            raise RuntimeError("当前没有待保存的 frame。")
        with quiet_native_stderr(self.config.quiet_stderr):
            self.dataset.save_episode(parallel_encoding=parallel_encoding)
        self._pending_steps = 0
        self.episodes_written += 1
        gc.collect()

    def discard_current_episode(self) -> None:
        if self._closed:
            return
        if hasattr(self.dataset, "clear_episode_buffer"):
            with quiet_native_stderr(self.config.quiet_stderr):
                self.dataset.clear_episode_buffer(delete_images=True)
        self._pending_steps = 0
        gc.collect()

    def checkpoint(self) -> None:
        """finalize + resume，释放 writer/视频编码器状态。"""
        self._ensure_open()
        if self._pending_steps:
            raise RuntimeError("还有未保存 frame，不能 checkpoint。")

        with quiet_native_stderr(self.config.quiet_stderr):
            self.dataset.finalize()

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        with quiet_native_stderr(self.config.quiet_stderr):
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
        if self._closed:
            return self.root
        if self._pending_steps:
            raise RuntimeError("还有未保存 frame。请先 save_current_episode() 或 discard_current_episode()。")
        with quiet_native_stderr(self.config.quiet_stderr):
            self.dataset.finalize()
        self._closed = True
        gc.collect()
        return self.root

    def make_frame(self, obs: dict, action: np.ndarray, *, task_label: str) -> dict[str, Any]:
        """转换为 LeRobotDataset.add_frame 的输入。"""

        state = flatten_state(obs["state"], state_keys=self.config.state_keys)
        if state.shape != (int(self.config.state_dim),):
            raise ValueError(f"state 维度错误，期望 {(self.config.state_dim,)}, 实际 {state.shape}")

        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.shape != (int(self.config.action_dim),):
            raise ValueError(f"action 维度错误，期望 {(self.config.action_dim,)}, 实际 {action_arr.shape}")

        frame: dict[str, Any] = {
            "observation.state": state,
            "action": np.ascontiguousarray(action_arr),
            "task": str(task_label),
        }

        if self.config.use_rgb:
            rgb = normalize_rgb(obs["rgb"])
            self._check_hw(rgb, "rgb")
            # LeRobot 编码器期望 numpy 数组而非 PIL Image
            frame["observation.images.rgb"] = rgb.transpose(2, 0, 1)  # HWC -> CHW

        if self.config.use_depth:
            depth_u8 = normalize_depth_to_uint8(obs["depth"])
            if depth_u8.shape != (int(self.config.image_height), int(self.config.image_width)):
                raise ValueError(
                    f"depth 分辨率错误，期望 {(self.config.image_height, self.config.image_width)}, 实际 {depth_u8.shape}"
                )
            # LeRobot 编码器期望 numpy 数组而非 PIL Image
            # 扩展为 3 通道以匹配配置的 shape (3, H, W)
            frame["observation.images.depth"] = np.stack([depth_u8] * 3, axis=0)

        return frame

    def _check_hw(self, image: np.ndarray, name: str) -> None:
        if image.shape[:2] != (int(self.config.image_height), int(self.config.image_width)):
            raise ValueError(
                f"{name} 分辨率错误，期望 {(self.config.image_height, self.config.image_width)}, 实际 {image.shape[:2]}"
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("LeRobotMujocoDatasetWriter 已 finalize，不能继续写入。")

    def __enter__(self) -> "LeRobotMujocoDatasetWriter":
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


class LeRobotRgbdTorchDataset(Dataset):
    """训练侧 PyTorch 包装。

    输出：
        camera: [4,H,W] float32，RGB + depth
        actor_state: [D] float32
        task: [1] long
        action: [A] float32
    """

    def __init__(self, root: str | Path, repo_id: str, *, task_id: int = 0) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self._lds = LeRobotDataset(repo_id=repo_id, root=str(Path(root).expanduser().resolve()))
        self._len = int(self._lds.num_frames)
        self.task_id = int(task_id)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        f = self._lds[idx]
        # LeRobot 视频解码后已是 float32 [0,1]，无需再除以 255
        rgb = f["observation.images.rgb"].float()
        depth = f["observation.images.depth"].float()
        if depth.ndim == 3:
            depth = depth[:1]
        return {
            "camera": torch.cat([rgb, depth], dim=0),
            "actor_state": f["observation.state"].float(),
            "task": torch.tensor([self.task_id], dtype=torch.long),
            "action": f["action"].float(),
        }