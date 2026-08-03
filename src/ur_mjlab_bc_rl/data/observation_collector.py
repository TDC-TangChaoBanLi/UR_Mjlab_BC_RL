"""多速率观测采集器 — 支持嵌套 recode_scale 缓冲采样。"""

from __future__ import annotations
from typing import Any
import numpy as np
from ..utils.config_loader import RobotConfig
from .dataset_config import DatasetConfig, DataSource
from ..simulate.cameras import CameraSensor
from ..simulate.mujoco_wrapper import MujocoWrapper as MujocoInterface


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

        # ── 预缓存 qpos/sensor address，避免每帧重复 name→id→adr 查找 ──
        self._src_readers: list[tuple[str, callable]] = []
        for src in dataset_cfg.sources:
            if src.source_type == "joint_pos":
                adrs = []
                for n in src.source_names:
                    try:
                        adrs.append(mj.model.jnt_qposadr[mj.model.joint(n).id])
                    except Exception:
                        adrs.append(-1)
                self._src_readers.append((src.name, self._make_joint_reader(adrs)))
            elif src.source_type.startswith("sensor."):
                sadrs: list[tuple[int, int]] = []
                for n in src.read_names:
                    try:
                        sid = mj.model.sensor(n).id
                        sadr = mj.model.sensor_adr[sid]
                        sdim = mj.model.sensor_dim[sid]
                        sadrs.append((sadr, sdim))
                    except Exception:
                        sadrs.append((-1, 0))
                self._src_readers.append((src.name, self._make_sensor_reader(sadrs)))
            elif src.source_type == "action":
                self._src_readers.append((src.name, lambda mj: self._last_action.copy()))
            else:
                self._src_readers.append((src.name, self._read_empty))

    @staticmethod
    def _make_joint_reader(adrs: list[int]):
        """返回闭包：从 mj.data.qpos 读取关节位置。"""
        valid = [(i, a) for i, a in enumerate(adrs) if a >= 0]
        n = len(adrs)
        def _read(mj) -> np.ndarray:
            out = np.zeros(n, dtype=np.float32)
            qpos = mj.data.qpos
            for i, a in valid:
                out[i] = float(qpos[a])
            return out
        return _read

    @staticmethod
    def _make_sensor_reader(sadrs: list[tuple[int, int]]):
        """返回闭包：从 mj.data.sensordata 读取传感器。"""
        n = len(sadrs) * 3
        valid = [(i, sadr, min(sdim, 3)) for i, (sadr, sdim) in enumerate(sadrs) if sadr >= 0]
        def _read(mj) -> np.ndarray:
            out = np.zeros(n, dtype=np.float32)
            sdata = mj.data.sensordata
            for i, sadr, sdim in valid:
                out[i * 3:i * 3 + sdim] = sdata[sadr:sadr + sdim]
            return out
        return _read

    @staticmethod
    def _read_empty(mj) -> np.ndarray:
        return np.array([], dtype=np.float32)

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
        for idx, src in enumerate(self._cfg.sources):
            period = max(1, (self._cfg.max_scale + src.num_subs - 1) // src.num_subs)
            if self._sub_step % period == 0 and self._sub_step // period < src.num_subs:
                local_idx = self._sub_step // period
                _name, reader = self._src_readers[idx]
                data = reader(self._mj).astype(np.float32)
                self._buffers.setdefault(src.name, []).append((local_idx, data))
        self._sub_step += 1

    def is_ready(self) -> bool:
        return self._sub_step >= self._cfg.max_scale

    def capture_camera(self, name: str) -> None:
        """按相机自身 FPS 调度，由 SimulationManager 在仿环中调用。

        与 flush() 解耦：相机渲染频率 = 场景配置的 fps，
        记录频率 = dataset 配置的 recode_hz。
        """
        if name in self._cameras:
            self._cameras[name].capture()
            self._camera_frame[name] = self._cameras[name].read(copy=False)

    def flush(self, task_id: int) -> dict[str, Any]:
        """读取已渲染的相机帧 → 打包缓冲区 → 清空 → 返回帧 dict。

        相机捕获由 SimulationManager 按各自 FPS 调度，
        flush() 只读取最新的缓存帧。

        Returns:
            { "state": {src_name: (subs, dim)}, "action": {...},
              "images": {cam: {"rgb":..., "depth":...}}, "task_id": int }
        """
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

    # ── 内部读取函数在 __init__ 中通过闭包/工厂创建，无需额外方法 ──
