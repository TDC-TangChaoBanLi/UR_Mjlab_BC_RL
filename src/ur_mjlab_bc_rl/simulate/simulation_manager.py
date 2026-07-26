"""仿真管理器 — 时序编排层。

负责仿真循环中所有时序相关逻辑：
  - 物理步进（physics_dt → policy_dt 倍频）
  - 多相机独立帧率调度
  - viewer wall-clock 同步
  - controller.action → ctrl → obs 流水线

不包含重试/采集计数逻辑（由上层脚本负责）。
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import mujoco

from .config_loader import SceneConfig
from .controllers import Controller
from .env import (
    MujocoInterface, CameraSensor, ObservationCollector,
    ResetManager,
)
from .dataset_config import DatasetConfig
from .dataset_writer import Episode

VIEWER_FPS = 30


class SimulationManager:
    """仿真时序编排器。

    持有所有仿真子组件，对外提供 run_episode(controller) 接口。
    每次调用运行一条完整 episode，返回 Episode 容器。
    """

    def __init__(
        self,
        config: SceneConfig,
        *,
        render: bool = False,
        dataset_cfg: DatasetConfig,
    ) -> None:
        self._config = config
        self._render = render

        # ── 子组件 ─────────────────────────────────────
        self._mj = MujocoInterface(config.task.scene_file, render=render)
        if render:
            self._mj.set_viewer_camera((0.45, 0.0, 0.65), 1.8, -25.0, 130.0)

        self._cameras: dict[str, CameraSensor] = {}
        for c in config.cameras:
            self._cameras[c.name] = CameraSensor(
                self._mj, c.name, (c.height, c.width),
            )

        self._collector = ObservationCollector(
            self._mj, self._cameras, config.robots,
            dataset_cfg=dataset_cfg,
        )
        self._dataset_cfg = dataset_cfg
        self._reset_mgr = ResetManager(
            self._mj, config.robots, config.task.objects,
        )

        # ── actuator ID 映射（按 robots 顺序）─────────
        self._arm_act_ids: list[list[int]] = []
        self._grip_act_ids: list[list[int]] = []
        total_dim = 0
        for r in config.robots:
            arm_ids = [self._mj.get_actuator_id(f"{j}_ACTUATOR")
                       for j in r.prefixed_arm_joints]
            grip_ids = [self._mj.get_actuator_id(f"{j}_ACTUATOR")
                        for j in r.prefixed_gripper_joints]
            self._arm_act_ids.append(arm_ids)
            self._grip_act_ids.append(grip_ids)
            total_dim += len(arm_ids) + len(grip_ids)
        self._total_action_dim = total_dim

        # ── 缓存 ───────────────────────────────────────
        self._pdt = config.sim.physics_dt
        self._adt = config.sim.policy_dt
        self._max_time = config.collection.max_time
        self._viewer_interval = 1.0 / VIEWER_FPS if VIEWER_FPS > 0 else float("inf")

    # ── 属性 ───────────────────────────────────────────

    @property
    def model(self) -> "mujoco.MjModel":
        """MuJoCo 模型（供 controller 构造使用）。"""
        return self._mj.model

    @property
    def data(self) -> "mujoco.MjData":
        """MuJoCo 数据（供 controller 构造使用）。"""
        return self._mj.data

    # ── 公开接口 ───────────────────────────────────────

    def run_episode(
        self,
        controller: Controller,
        *,
        max_time: float | None = None,
        frame_callback: Callable[[dict, np.ndarray], None] | None = None,
    ) -> int:
        """运行一条完整 episode，返回采集的帧数。"""
        t_max = max_time if max_time is not None else self._max_time
        dcfg = self._dataset_cfg
        frame_dt = dcfg.frame_interval_s
        frame_count = 0

        # ── 重置 ──
        self._reset_mgr.reset(randomize_objects=True)
        controller.reset()
        self._collector.reset()

        t_sim = t_policy = t_frame = t_sample = t_viewer = 0.0
        wall_start = time.perf_counter()
        sample_dt = dcfg.sample_interval_s  # 1 / (recode_hz * max_scale)

        while t_sim < t_max:
            self._mj.step()
            t_sim += self._pdt
            t_policy += self._pdt
            t_frame += self._pdt
            t_sample += self._pdt
            t_viewer += self._pdt

            # 子采样步进（recode_hz × max_scale Hz）
            if t_sample >= sample_dt:
                t_sample -= sample_dt
                self._collector.sample()

            # viewer
            if self._render:
                target = wall_start + t_sim
                now = time.perf_counter()
                if target > now:
                    time.sleep(target - now)
                if t_viewer >= self._viewer_interval:
                    self._mj.sync_viewer()
                    t_viewer -= self._viewer_interval
                if not self._mj.is_viewer_running():
                    break

            # 帧边界：捕获相机 + 打包 + 回调
            if t_frame >= frame_dt and self._collector.is_ready():
                t_frame -= frame_dt
                obs = self._collector.flush(task_id=self._config.task.task_id)
                frame_count += 1
                if frame_callback:
                    frame_callback(obs, None)

            # policy 步进
            if t_policy < self._adt:
                continue
            t_policy -= self._adt

            action = controller.step({
                "state": {"arm_joint_pos": self._collector.get_joint_positions()},
                "task_id": self._config.task.task_id,
            })
            self._apply_action(action)
            self._collector.update_last_action(action)

            if controller.is_done():
                break

        return frame_count

    def close(self) -> None:
        """释放所有子组件资源。"""
        for cam in self._cameras.values():
            cam.close()
        self._cameras.clear()
        self._mj.close()

    # ── 内部 ───────────────────────────────────────────

    def _apply_action(self, action: np.ndarray) -> None:
        """将 flat action 写入 MuJoCo ctrl。

        action layout: [arm0_joints..., grip0..., arm1_joints..., grip1..., ...]
        """
        ctrl = self._mj.get_ctrl()
        arr = np.asarray(action, dtype=np.float64).ravel()
        offset = 0
        for i in range(len(self._config.robots)):
            n_arm = len(self._arm_act_ids[i])
            n_grip = len(self._grip_act_ids[i])
            for j, aid in enumerate(self._arm_act_ids[i]):
                ctrl[aid] = arr[offset + j] if offset + j < len(arr) else 0.0
            offset += n_arm
            for j, gid in enumerate(self._grip_act_ids[i]):
                ctrl[gid] = arr[offset + j] if offset + j < len(arr) else 0.0
            offset += n_grip
        self._mj.set_ctrl(ctrl)
