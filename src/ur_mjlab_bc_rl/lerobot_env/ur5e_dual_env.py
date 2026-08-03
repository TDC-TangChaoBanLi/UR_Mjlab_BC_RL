"""UR5e 双臂 MuJoCo Gymnasium 环境。

遵循 LeRobot "Adding a Benchmark" 规范：
  - 使用标准观测 key（pixels dict + ndarray 透传）
  - _ensure_env() 延迟 GPU 分配
  - info["is_success"] 在每步返回
  - metadata 含 render_fps

时序结构（匹配 BiMFT 策略）:
  - 图像帧率 = 相机 FPS（30Hz），每步渲染 1 帧
  - 高频状态：每帧图像对应 R=3 个关节/力觉采样（等效 90Hz）
  - 状态返回 shape (R, D)，LeRobot delta 堆叠后为 [T, R, D]
    即 4 帧图像 + 4×3=12 帧状态数据

仅供 lerobot-eval CLI 推理使用，不用于数据采集。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from ur_mjlab_bc_rl.simulate.mujoco_wrapper import MujocoWrapper
from ur_mjlab_bc_rl.simulate.cameras import CameraSensor
from ur_mjlab_bc_rl.utils.config_loader import (
    load_scene_config, CameraConfig as CC, RobotConfig as RC, SceneConfig,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 每帧图像对应的高频状态采样数（匹配 BiMFT joint_high_rate / force_high_rate）
_DEFAULT_HIGH_RATE = 3


class UR5eDualEnv(gym.Env):
    """UR5e 双臂 MuJoCo 环境 — LeRobot 兼容。

    调用方式:
        env = UR5eDualEnv(task_name="dual_pick_place")
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)

    参数:
        task_name: 任务名（对应 configs/tasks/tasks_*.yaml 中的 key）
        render_mode: "human" 开启 MuJoCo viewer, None/"rgb_array" 为无头
        max_steps: 最大步数（覆盖配置文件中的值）
        high_rate: 每帧图像对应的高频状态采样数（默认 3，匹配 90Hz 状态 @ 30Hz 图像）
        depth_range_m: 深度裁剪范围 (min, max) 米, 需与训练时 DepthEncoder 的 depth_min/depth_max 一致
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(
        self,
        task_name: str = "dual_pick_place",
        render_mode: str | None = None,
        max_steps: int | None = None,
        high_rate: int = _DEFAULT_HIGH_RATE,
        depth_range_m: tuple[float, float] = (0.1, 2.0),
    ):
        super().__init__()

        self._task_name = task_name
        self._render_mode = render_mode
        self._high_rate = high_rate
        self._depth_range_m = depth_range_m  # 与 dataset_dual.yaml 的 depth_range 一致

        # 加载场景配置
        self._config: SceneConfig = load_scene_config(task_name)
        self._task_config = self._config.task
        self._sim_config = self._config.sim
        self._robot_configs: list[RC] = list(self._config.robots)
        self._camera_configs: list[CC] = list(self._config.cameras)

        # ── 时序参数 ──
        # policy_dt 对齐相机帧率（取最慢相机），保证 1 帧图像 = 1 个 policy step
        self._physics_dt = self._sim_config.physics_dt
        if self._camera_configs:
            self._policy_dt = max(cc.dt for cc in self._camera_configs)
        else:
            self._policy_dt = self._sim_config.policy_dt
        # 每 policy step 的物理步数
        self._steps_per_policy = max(1, int(round(self._policy_dt / self._physics_dt)))
        # 每个高频子采样的物理步数
        self._steps_per_high_rate = max(
            1, self._steps_per_policy // self._high_rate
        )
        self._max_episode_steps = max_steps or int(
            self._config.collection.max_time / self._policy_dt
        )

        # 延迟初始化
        self._mj: MujocoWrapper | None = None
        self._cameras: dict[str, CameraSensor] = {}
        self._step_count: int = 0

        # ── 观测空间 ──
        # 图像: 每步 1 帧, channel-first (C, H, W)
        # depth 使用 float32 毫米（与 LeRobot depth_output_unit='mm' 一致）
        pixels_dict: dict[str, spaces.Box] = {}
        for cc in self._camera_configs:
            H, W = cc.height, cc.width
            pixels_dict[f"{cc.name}.rgb"] = spaces.Box(
                0, 255, shape=(3, H, W), dtype=np.uint8,
            )
            # 深度: float32 毫米, 裁剪到 depth_range_m 范围（匹配 DepthEncoder）
            depth_low = depth_range_m[0] * 1000.0
            depth_high = depth_range_m[1] * 1000.0
            pixels_dict[f"{cc.name}.depth"] = spaces.Box(
                depth_low, depth_high, shape=(1, H, W), dtype=np.float32,
            )

        total_joints = sum(
            r.n_arm_joints + r.n_gripper_joints for r in self._robot_configs
        )

        # 状态: 高频采样 → shape (R, D)
        # LeRobot delta 堆叠 T=4 帧后 → [B, T, R, D] = [B, 4, 3, D]
        obs_dict: dict[str, spaces.Space] = {
            **{f"images.{k}": v for k, v in pixels_dict.items()},
            "state.joint.position": spaces.Box(
                -np.inf, np.inf,
                shape=(self._high_rate, total_joints), dtype=np.float32,
            ),
            "state.sensor.force": spaces.Box(
                -np.inf, np.inf,
                shape=(self._high_rate, 2 * 3), dtype=np.float32,
            ),
            "state.sensor.torque": spaces.Box(
                -np.inf, np.inf,
                shape=(self._high_rate, 2 * 3), dtype=np.float32,
            ),
        }
        self.observation_space = spaces.Dict(obs_dict)

        # ── 动作空间 ──
        # action 为绝对关节位置（弧度），由策略直接输出、无需 [-1,1] 归一化，
        # 因此用无界 Box 声明（避免与归一化语义冲突）。
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_joints,), dtype=np.float32,
        )

        # actuator 映射（_ensure_env 后填充）
        self._arm_act_ids: list[list[int]] = []
        self._grip_act_ids: list[list[int]] = []
        self._actuator_limits: list[tuple[float, float]] = []

    def _ensure_env(self) -> None:
        """延迟初始化 MuJoCo + 相机 — GPU ctx 在 worker 进程分配。"""
        if self._mj is not None:
            return

        use_viewer = self._render_mode == "human"
        self._mj = MujocoWrapper(
            str(_PROJECT_ROOT / self._task_config.scene_file),
            render=use_viewer,
        )
        self._mj.open()

        # MuJoCo viewer: 设置视角
        if use_viewer:
            self._mj.set_viewer_camera((0.45, 0.0, 0.65), 1.8, -25.0, 130.0)

        for cc in self._camera_configs:
            self._cameras[cc.name] = CameraSensor(
                self._mj, cc.name, (cc.height, cc.width),
            )

        # 构建 actuator 映射和关节限位
        for r in self._robot_configs:
            arm_ids = [
                self._mj.get_actuator_id(f"{j}_ACTUATOR")
                for j in r.prefixed_arm_joints
            ]
            grip_ids = [
                self._mj.get_actuator_id(f"{j}_ACTUATOR")
                for j in r.prefixed_gripper_joints
            ]
            self._arm_act_ids.append(arm_ids)
            self._grip_act_ids.append(grip_ids)

            for jname in r.prefixed_arm_joints:
                jid = self._mj.get_joint_id(jname)
                lo = self._mj.model.jnt_range[jid][0]
                hi = self._mj.model.jnt_range[jid][1]
                self._actuator_limits.append((float(lo), float(hi)))
            for jname in r.prefixed_gripper_joints:
                jid = self._mj.get_joint_id(jname)
                lo = self._mj.model.jnt_range[jid][0]
                hi = self._mj.model.jnt_range[jid][1]
                self._actuator_limits.append((float(lo), float(hi)))

    # ── Gym 接口 ───────────────────────────────────────

    @property
    def task(self) -> str:
        return self._task_name

    @property
    def task_description(self) -> str:
        return self._task_name.replace("_", " ").title()

    def reset(
        self, *, seed: int | None = None, options: dict | None = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        self._ensure_env()

        # ── 1. 重置仿真数据 ──
        mujoco.mj_resetData(self._mj.model, self._mj.data)

        # ── 2. 设置机械臂到 default_qpos + 夹爪归零 ──
        ctrl = self._mj.get_ctrl()
        for i, r in enumerate(self._robot_configs):
            arm_ids = self._arm_act_ids[i]
            grip_ids = self._grip_act_ids[i]
            for j, jname in enumerate(r.prefixed_arm_joints):
                q = r.default_qpos[j]
                self._mj.set_joint_qpos(jname, q)
                ctrl[arm_ids[j]] = q
            for j, jname in enumerate(r.prefixed_gripper_joints):
                self._mj.set_joint_qpos(jname, 0.0)
                ctrl[grip_ids[j]] = 0.0
        self._mj.set_ctrl(ctrl)

        # ── 3. 随机化物体位姿（位置 + 朝向）──
        self._randomize_objects()

        # ── 4. 前向动力学 ──
        mujoco.mj_forward(self._mj.model, self._mj.data)

        self._step_count = 0

        obs = self._collect_obs()
        info: dict = {"is_success": False}
        return obs, info

    def step(
        self, action: np.ndarray,
    ) -> tuple[dict, float, bool, bool, dict]:
        self._step_count += 1

        action = np.asarray(action, dtype=np.float64)
        if action.ndim == 2 and action.shape[0] == self._high_rate:
            # [R, D] 高频动作序列：每个 90Hz 子采样段切换不同 ctrl，
            # 与数据集 action [3, 14]（90Hz）一致
            obs = self._step_physics_and_collect(action_seq=action)
        else:
            # [D] 单动作：整个 policy_dt 内保持恒定（兼容旧调用）
            ctrl = self._denormalize_action(action)
            self._apply_action(ctrl)
            obs = self._step_physics_and_collect()

        terminated = bool(self._check_success())
        truncated = self._step_count >= self._max_episode_steps

        reward = 1.0 if terminated else 0.0
        info = {"is_success": terminated}

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """渲染当前帧。

        render_mode="human": 打开 MuJoCo viewer 窗口，返回 None
        render_mode="rgb_array": 返回第一相机 RGB 图像 (H, W, 3) uint8
        """
        self._ensure_env()
        if self._render_mode == "human":
            self._mj.render()
            return None
        if not self._camera_configs:
            return None
        cc = self._camera_configs[0]
        return self._mj.get_camera_rgb(cc.name, cc.height, cc.width)

    def close(self) -> None:
        for cam in self._cameras.values():
            cam.close()
        self._cameras.clear()
        if self._mj is not None:
            self._mj.close()
            self._mj = None

    # ── 内部 ───────────────────────────────────────────

    def _randomize_objects(self) -> None:
        """根据 task 配置随机放置物体（位置 + 朝向）。

        参照 ResetManager._randomize_objects():
          - 直接操作 qpos（freejoint 的 7 维: x,y,z,qw,qx,qy,qz）
          - euler 随机化用 mju_euler2Quat 转四元数
        """
        rng = random.Random(int(self.np_random.integers(0, 2**31)))
        for obj_name, rand_cfg in self._task_config.objects.items():
            jnt_id = self._mj.get_body_joint_id(obj_name)
            if jnt_id is None:
                continue

            adr = self._mj.model.jnt_qposadr[jnt_id]

            # 位置随机化
            x = rng.uniform(*rand_cfg.x_range)
            y = rng.uniform(*rand_cfg.y_range)
            z = rng.uniform(*rand_cfg.z_range)
            self._mj.data.qpos[adr:adr + 3] = [x, y, z]

            # 朝向随机化（euler → quaternion）
            has_rot = (
                rand_cfg.roll_range != (0.0, 0.0)
                or rand_cfg.pitch_range != (0.0, 0.0)
                or rand_cfg.yaw_range != (0.0, 0.0)
            )
            if has_rot:
                roll = rng.uniform(*rand_cfg.roll_range)
                pitch = rng.uniform(*rand_cfg.pitch_range)
                yaw = rng.uniform(*rand_cfg.yaw_range)
                quat = np.zeros(4, dtype=np.float64)
                mujoco.mju_euler2Quat(quat, [roll, pitch, yaw], "XYZ")
                self._mj.data.qpos[adr + 3:adr + 7] = quat  # qw,qx,qy,qz

    # ── 内部：高频时序采集 ───────────────────────────

    def _step_physics_and_collect(
        self, action_seq: np.ndarray | None = None,
    ) -> dict:
        """执行 policy_dt 时长仿真并收集观测。

        时序结构（以 30Hz 图像, R=3 为例）:
          物理步 0..10  → 高频采样 #0 (t ≈ -2/90s)
          物理步 11..21 → 高频采样 #1 (t ≈ -1/90s)
          物理步 22..32 → 高频采样 #2 + 相机渲染 (t = 0)

        Args:
            action_seq: 可选，shape [R, D] 的高频动作序列（90Hz）。
                提供时，每个子采样段开始前切换到对应动作（与数据集 90Hz 动作一致）；
                为 None 时使用当前 ctrl 恒定（兼容单动作调用）。

        Returns:
            obs dict with:
              images.*.rgb/depth: shape (C, H, W) — 1 帧图像
              state.joint.position:  shape (R, D_joint)  — R 个高频采样
              state.sensor.force:    shape (R, D_force)
              state.sensor.torque:   shape (R, D_torque)
        """
        self._ensure_env()
        R = self._high_rate
        D_joint = sum(
            r.n_arm_joints + r.n_gripper_joints for r in self._robot_configs
        )
        D_wrench = 2 * 3  # A_ + B_ 各 3 维

        # 预分配高频状态缓冲区
        joint_samples = np.zeros((R, D_joint), dtype=np.float32)
        force_samples = np.zeros((R, D_wrench), dtype=np.float32)
        torque_samples = np.zeros((R, D_wrench), dtype=np.float32)

        total_steps = self._steps_per_policy
        steps_per_sample = self._steps_per_high_rate

        for r_idx in range(R):
            # 若提供高频动作序列，每个子采样段开始前切换到对应动作（90Hz 执行）
            if action_seq is not None:
                ctrl = self._denormalize_action(action_seq[r_idx])
                self._apply_action(ctrl)

            # 执行 steps_per_sample 次物理步进
            start_step = r_idx * steps_per_sample
            end_step = min(start_step + steps_per_sample, total_steps)
            for _ in range(end_step - start_step):
                self._mj.step()

            # 采集该子采样的状态
            joint_samples[r_idx] = self._collect_joint_state()
            f, t = self._collect_wrench_state()
            force_samples[r_idx] = f
            torque_samples[r_idx] = t

        # 执行剩余物理步（如果 total_steps 不能被 R 整除）
        remaining = total_steps - R * steps_per_sample
        for _ in range(max(0, remaining)):
            self._mj.step()

        # ── 渲染相机（在最后一步之后）──
        pixels: dict[str, np.ndarray] = {}
        for cc in self._camera_configs:
            cam = self._cameras[cc.name]
            cam.capture()
            frame = cam.read()
            rgb = frame["rgb"].transpose(2, 0, 1)  # HWC → CHW
            # 深度: MuJoCo 米制 → 裁剪到 depth_range_m → 转毫米
            depth_m = frame["depth"]
            d_min_m, d_max_m = self._depth_range_m
            depth_mm = np.clip(depth_m, d_min_m, d_max_m).astype(np.float32) * 1000.0
            pixels[f"{cc.name}.rgb"] = rgb
            pixels[f"{cc.name}.depth"] = depth_mm[np.newaxis, ...]  # HW → 1HW

        return {
            **{f"images.{k}": v for k, v in pixels.items()},
            "state.joint.position": joint_samples,
            "state.sensor.force": force_samples,
            "state.sensor.torque": torque_samples,
        }

    def _collect_joint_state(self) -> np.ndarray:
        """采集当前关节位置（双臂拼接）→ shape (D_joint,)。"""
        parts = []
        for r in self._robot_configs:
            arm = self._mj.get_joint_qpos(r.prefixed_arm_joints)
            grip = self._mj.get_joint_qpos(r.prefixed_gripper_joints)
            parts.append(np.concatenate([arm, grip]).astype(np.float32))
        return np.concatenate(parts)

    def _collect_wrench_state(self) -> tuple[np.ndarray, np.ndarray]:
        """采集当前力/力矩（双臂拼接）→ (force[D_wrench], torque[D_wrench])。"""
        force_parts = []
        torque_parts = []
        for prefix in ["A_", "B_"]:
            try:
                force_parts.append(
                    self._mj.get_sensor(f"{prefix}ur_ft_frame_SENSOR_FORCE")
                )
            except Exception:
                force_parts.append(np.zeros(3, dtype=np.float32))
            try:
                torque_parts.append(
                    self._mj.get_sensor(f"{prefix}ur_ft_frame_SENSOR_TORQUE")
                )
            except Exception:
                torque_parts.append(np.zeros(3, dtype=np.float32))
        return (
            np.concatenate(force_parts).astype(np.float32),
            np.concatenate(torque_parts).astype(np.float32),
        )

    def _collect_obs(self) -> dict:
        """采集当前观测 — 返回与 observation_space 一致的 shape。

        状态返回 shape (R, D)，其中 R=self._high_rate。
        在 reset 时所有 R 行相同（无历史），在 step 时由 _step_physics_and_collect 填充。
        """
        self._ensure_env()

        pixels: dict[str, np.ndarray] = {}
        for cc in self._camera_configs:
            cam = self._cameras[cc.name]
            cam.capture()
            frame = cam.read()
            rgb = frame["rgb"].transpose(2, 0, 1)  # HWC → CHW
            # 深度: MuJoCo 米制 → 裁剪到 depth_range_m → 转毫米
            depth_m = frame["depth"]
            d_min_m, d_max_m = self._depth_range_m
            depth_mm = np.clip(depth_m, d_min_m, d_max_m).astype(np.float32) * 1000.0
            pixels[f"{cc.name}.rgb"] = rgb
            pixels[f"{cc.name}.depth"] = depth_mm[np.newaxis, ...]  # HW → 1HW

        joint_pos = self._collect_joint_state()
        force, torque = self._collect_wrench_state()

        # 扩展为 (R, D) — 所有行相同（reset 场景无历史高频数据）
        R = self._high_rate
        return {
            **{f"images.{k}": v for k, v in pixels.items()},
            "state.joint.position": np.tile(joint_pos, (R, 1)).astype(np.float32),
            "state.sensor.force": np.tile(force, (R, 1)).astype(np.float32),
            "state.sensor.torque": np.tile(torque, (R, 1)).astype(np.float32),
        }

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """动作直接作为执行器目标（绝对关节位置，弧度）。

        注意: 数据集 action 是绝对关节位置（训练时 normalization_mapping 为空，
        动作未归一化），模型输出即真实目标关节角。因此这里不再做 [-1,1] 反
        归一化，直接透传给 position actuator。
        """
        return np.asarray(action, dtype=np.float64)

    def _apply_action(self, ctrl: np.ndarray) -> None:
        """写入 actuator。"""
        full_ctrl = self._mj.get_ctrl()
        offset = 0
        for i in range(len(self._robot_configs)):
            n_arm = len(self._arm_act_ids[i])
            n_grip = len(self._grip_act_ids[i])
            for j, aid in enumerate(self._arm_act_ids[i]):
                if offset + j < len(ctrl):
                    full_ctrl[aid] = ctrl[offset + j]
            offset += n_arm
            for j, gid in enumerate(self._grip_act_ids[i]):
                if offset + j < len(ctrl):
                    full_ctrl[gid] = ctrl[offset + j]
            offset += n_grip
        self._mj.set_ctrl(full_ctrl)

    def _check_success(self) -> bool:
        """成功检测 — 暂时总返回 False。

        TODO: 根据 task 类型实现具体成功条件。
        """
        return False
