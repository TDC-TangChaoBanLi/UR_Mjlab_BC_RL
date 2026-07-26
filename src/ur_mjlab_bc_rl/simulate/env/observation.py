"""多速率观测采集器 — 支持嵌套 recode_scale 缓冲采样。"""

from __future__ import annotations
from typing import Any
import numpy as np
from ..config_loader import RobotConfig
from ..dataset_config import DatasetConfig, DataSource
from .camera import CameraSensor
from .mujoco_interface import MujocoInterface


class ObservationCollector:
    """多速率缓冲采集器。

    sample()    — 每个物理步调用，按各数据源的子采样网格记录
    is_ready()  — 缓冲区已满（max_scale 个子采样都已记录）
    flush()     — 帧边界调用：捕获相机 → 打包缓冲区 → 清空 → 返回帧 dict
    """

    def __init__(
        self,
        mj: MujocoInterface,
        cameras: dict[str, CameraSensor],
        robots: list[RobotConfig],
        dataset_cfg: DatasetConfig,
    ) -> None:
        self._mj = mj
        self._cameras = cameras
        self._robots = robots
        self._cfg = dataset_cfg

        # 验证名称
        import logging
        _log = logging.getLogger(__name__)
        for w in dataset_cfg.validate(mj.model):
            _log.warning(w)

        total_action_dim = sum(r.n_arm_joints + r.n_gripper_joints for r in robots)
        self._last_action = np.zeros(total_action_dim, dtype=np.float32)
        self._sub_step = 0
        self._buffers: dict[str, list] = {}
        self._camera_frame: dict[str, dict] = {}

    # ── 生命周期 ──

    def reset(self) -> None:
        self._last_action.fill(0.0)
        self._sub_step = 0
        self._buffers.clear()
        self._camera_frame.clear()

    def update_last_action(self, action: np.ndarray) -> None:
        arr = np.asarray(action, dtype=np.float32).ravel()
        n = min(len(arr), len(self._last_action))
        self._last_action[:n] = arr[:n]

    # ── 控制器观测（IK 用） ──

    def get_joint_positions(self) -> np.ndarray:
        """获取所有臂的关节位置（仅 arm joints，不含 gripper），供控制器 IK 使用。"""
        parts = []
        for r in self._robots:
            parts.append(self._mj.get_joint_qpos(r.prefixed_arm_joints).astype(np.float32))
        return np.concatenate(parts)

    # ── 采样与刷新 ──

    def sample(self) -> None:
        """在当前子步记录所有应采样的数据源。

        不同 num_subs 的数据源在 max_scale 网格中均匀分布：
          num_subs=3 (max=3): sub_step 0,1,2 各采样一次
          num_subs=2 (max=3): sub_step 0,2 各采样一次 (period=ceil(3/2)=2)
        """
        for src in self._cfg.sources:
            period = max(1, (self._cfg.max_scale + src.num_subs - 1) // src.num_subs)
            if self._sub_step % period == 0 and self._sub_step // period < src.num_subs:
                local_idx = self._sub_step // period
                data = self._read_source(src).astype(np.float32)
                self._buffers.setdefault(src.name, []).append((local_idx, data))
        self._sub_step += 1

    def is_ready(self) -> bool:
        return self._sub_step >= self._cfg.max_scale

    def flush(self, task_id: int) -> dict[str, Any]:
        """捕获相机 → 打包缓冲区 → 清空 → 返回帧 dict。

        Returns:
            { "state": {src_name: (subs, dim)}, "action": {...},
              "images": {cam: {"rgb":..., "depth":...}}, "task_id": int }
        """
        # 捕获相机（帧边界最新一帧）
        for name, cam in self._cameras.items():
            cam.capture()
            self._camera_frame[name] = cam.read(copy=False)

        # 打包（每个源保留真实子采样数，不 padding）
        state: dict[str, np.ndarray] = {}
        action: dict[str, np.ndarray] = {}
        for src in self._cfg.sources:
            buf = self._buffers.get(src.name, [])
            if not buf:
                continue
            buf.sort(key=lambda x: x[0])
            stacked = np.stack([b[1] for b in buf])
            if src.name.startswith("state."):
                state[src.name] = stacked
            else:
                action[src.name] = stacked

        images = dict(self._camera_frame)

        # 清空
        self._sub_step = 0
        self._buffers.clear()
        self._camera_frame.clear()

        return {"state": state, "action": action, "images": images, "task_id": int(task_id)}

    # ── 内部：按类型读取 ──

    def _read_source(self, src: DataSource) -> np.ndarray:
        if src.source_type == "joint_pos":
            return self._read_joint_pos(src.source_names)
        if src.source_type.startswith("sensor."):
            return self._read_sensor(src.read_names)
        if src.source_type == "action":
            return self._last_action.copy()
        return np.array([])

    def _read_joint_pos(self, names: list[str]) -> np.ndarray:
        out = np.zeros(len(names), dtype=np.float32)
        for i, n in enumerate(names):
            try:
                adr = self._mj.model.jnt_qposadr[self._mj.model.joint(n).id]
                out[i] = float(self._mj.data.qpos[adr])
            except Exception:
                pass
        return out

    def _read_sensor(self, names: list[str]) -> np.ndarray:
        out = np.zeros(len(names) * 3, dtype=np.float32)
        for i, n in enumerate(names):
            try:
                d = self._mj.get_sensor_data(n)
                m = min(len(d), 3)
                out[i * 3:i * 3 + m] = d[:m]
            except Exception:
                pass
        return out
