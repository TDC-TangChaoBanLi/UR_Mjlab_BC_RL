"""仿真控制器协议与内置实现。

Controller 协议定义统一的控制接口，所有控制器返回 joint-level 动作。
内置实现：
  - ScriptedTeacherController — 封装 scripted teacher + IK 流水线
  - PolicyController          — 封装 BC / ACT 策略模型推理
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import mujoco


# ═══════════════════════════════════════════════════════
# Controller 协议
# ═══════════════════════════════════════════════════════

@runtime_checkable
class Controller(Protocol):
    """统一控制接口。

    所有控制器返回 joint-level 动作：
      [arm0_joints(6), gripper0(1), arm1_joints(6), gripper1(1), ...]
    按 SceneConfig.robots 的顺序拼接。
    """

    def reset(self) -> None:
        """重置控制器内部状态（新 episode 开始前调用）。"""
        ...

    def step(self, observation: dict[str, Any]) -> np.ndarray:
        """根据观测产生 joint-level 动作。"""
        ...

    def is_done(self) -> bool:
        """当前 episode 是否已终止（成功/失败/超时）。"""
        ...

    def is_success(self) -> bool:
        """当前 episode 是否成功完成。"""
        ...


# ═══════════════════════════════════════════════════════
# Scripted Teacher 控制器
# ═══════════════════════════════════════════════════════

_TEACHER_REGISTRY: dict[str, type] = {}


def _init_teacher_registry() -> None:
    from .teachers import (
        PickPlaceTeacher, PushTTeacher, PegSlotTeacher,
        DualPickPlaceTeacher,
    )
    _TEACHER_REGISTRY.update({
        "PickPlaceTeacher": PickPlaceTeacher,
        "PushTTeacher": PushTTeacher,
        "PegSlotTeacher": PegSlotTeacher,
        "DualPickPlaceTeacher": DualPickPlaceTeacher,
    })


class ScriptedTeacherController:
    """封装 scripted teacher + MinkIK 流水线。

    每个 arm 独立维护一个 teacher + IK 求解器。
    teacher 输出目标位姿 → IK 求解 → 返回 joint-level 动作。
    """

    def __init__(
        self,
        config,             # SceneConfig
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        if not _TEACHER_REGISTRY:
            _init_teacher_registry()

        tcls = _TEACHER_REGISTRY[config.task.teacher]

        from .env.ik_solver import MinkIK

        self._robots = list(config.robots)
        self._policy_dt = config.sim.policy_dt

        # 多臂 teacher（_is_multi_arm=True）：单实例同时控制多个臂
        if getattr(tcls, '_is_multi_arm', False):
            self._multi_arm = True
            self._teacher: Any = tcls(model, data, prefix="")
            self._iks: dict[str, MinkIK] = {}
            for r in self._robots:
                self._iks[r.prefix] = MinkIK(
                    model,
                    init_qpos=data.qpos.copy(),
                    dt=self._policy_dt,
                    ee_site_name=r.prefixed_ee_site,
                    vel_limit=[10.0] * 6,
                    arm_joint_names=r.prefixed_arm_joints,
                )
        else:
            self._multi_arm = False
            self._teachers: dict[str, Any] = {}
            self._iks = {}
            for r in self._robots:
                self._teachers[r.prefix] = tcls(model, data, prefix=r.prefix)
                self._iks[r.prefix] = MinkIK(
                    model,
                    init_qpos=data.qpos.copy(),
                    dt=self._policy_dt,
                    ee_site_name=r.prefixed_ee_site,
                    vel_limit=[10.0] * 6,
                    arm_joint_names=r.prefixed_arm_joints,
                )

    # ── Controller 接口 ────────────────────────────────

    def reset(self) -> None:
        if self._multi_arm:
            self._teacher.reset()
        else:
            for t in self._teachers.values():
                t.reset()

    def step(self, observation: dict[str, Any]) -> np.ndarray:
        state = observation["state"]
        arm_joint_pos = np.asarray(state["arm_joint_pos"], dtype=np.float64)

        if self._multi_arm:
            tgt_dict = self._teacher.step()
        else:
            tgt_dict = {}
            for prefix, t in self._teachers.items():
                tgt_dict.update(t.step())

        actions: list[np.ndarray] = []
        offset = 0
        for r in self._robots:
            tgt = tgt_dict[r.prefix]
            target_pose = np.asarray(tgt[:7], dtype=np.float64)
            grip_cmd = float(tgt[7])

            n_arm = r.n_arm_joints
            cur_arm_qpos = arm_joint_pos[offset:offset + n_arm]
            jt = self._iks[r.prefix].solve(cur_arm_qpos, target_pose, dt=self._policy_dt)

            actions.append(jt.copy().astype(np.float32))
            actions.append(np.array([grip_cmd], dtype=np.float32))
            offset += n_arm

        return np.concatenate(actions).astype(np.float32)

    def is_done(self) -> bool:
        if self._multi_arm:
            return self._teacher.is_done()
        return all(t.is_done() for t in self._teachers.values())

    def is_success(self) -> bool:
        if self._multi_arm:
            return self._teacher.is_success()
        return all(t.is_success() for t in self._teachers.values())


# ═══════════════════════════════════════════════════════
# 策略模型控制器
# ═══════════════════════════════════════════════════════

class PolicyController:
    """封装 BC / ACT 策略模型推理。

    根据 model_type 自动选择推理路径：
      - "bc"  — 标准 BC 模型
      - "act" — ACT (DETRVAE) 模型，含 EnsembleBuffer
    """

    def __init__(
        self,
        model,                              # torch.nn.Module
        model_type: str,                    # "bc" | "act"
        device: str = "cpu",
        deterministic: bool = True,
        chunk_size: int = 10,
        depth_range: tuple[float, float] = (0.1, 0.8),
    ) -> None:
        self._model = model
        self._model_type = model_type
        self._device = device
        self._deterministic = deterministic
        self._chunk_size = chunk_size
        self._depth_range = depth_range
        self._ensemble_buffer: Any = None

    # ── Controller 接口 ────────────────────────────────

    def reset(self) -> None:
        if self._model_type == "act" and self._chunk_size > 0:
            from ur_mjlab_bc_rl.models.ALHAH_ACT.backbone import EnsembleBuffer
            self._ensemble_buffer = EnsembleBuffer(
                chunk_size=self._chunk_size,
                action_dim=7,
            ).to(self._device)

    def step(self, observation: dict[str, Any]) -> np.ndarray:
        import torch
        import numpy as np

        with torch.no_grad():
            if self._model_type == "bc":
                action = self._step_bc(observation)
            elif self._model_type == "act":
                action = self._step_act(observation)
            else:
                raise ValueError(f"未知 model_type: {self._model_type!r}")

        anp = action.cpu().numpy().squeeze(0)
        return np.clip(anp, -6.28, 6.28).astype(np.float32)

    def is_done(self) -> bool:
        return False

    def is_success(self) -> bool:
        return False

    # ── 内部推理 ───────────────────────────────────────

    def _preprocess_camera(self, frame: dict) -> "Any":
        """单相机帧 → (1, 4, H, W) tensor [RGB(3)+depth(1)]。"""
        import torch
        import numpy as np

        rgb = np.asarray(frame["rgb"], dtype=np.float32) / 255.0
        if rgb.ndim == 3 and rgb.shape[-1] == 3:
            rgb = rgb.transpose(2, 0, 1)  # (H,W,3) → (3,H,W)

        depth = np.asarray(frame["depth"], dtype=np.float32)
        d_min, d_max = self._depth_range
        depth = np.clip((depth - d_min) / (d_max - d_min + 1e-8), 0.0, 1.0)
        if depth.ndim == 2:
            depth = depth[None, :, :]       # (H,W) → (1,H,W)

        img = np.concatenate([rgb, depth], axis=0)  # (4, H, W)
        return torch.from_numpy(img).unsqueeze(0).to(self._device)

    def _preprocess_state(self, obs: dict) -> "Any":
        """提取 arm_joint_pos + gripper_pos → (1, D) tensor。"""
        import torch
        import numpy as np

        state = obs["state"]
        parts = [
            np.asarray(state["arm_joint_pos"], dtype=np.float32).ravel(),
            np.asarray(state["gripper_pos"], dtype=np.float32).ravel(),
        ]
        flat = np.concatenate(parts)
        return torch.from_numpy(flat).unsqueeze(0).to(self._device)

    def _step_bc(self, obs: dict) -> "Any":
        import torch

        ct = self._preprocess_camera(obs["images"][self._camera_name(obs)])
        st = self._preprocess_state(obs)
        tt = torch.tensor([[obs.get("task_id", 0)]], dtype=torch.long,
                          device=self._device)

        return self._model(
            {"camera": ct, "actor_state": st, "task": tt},
            deterministic=self._deterministic,
        )

    def _step_act(self, obs: dict) -> "Any":
        ct = self._preprocess_camera(obs["images"][self._camera_name(obs)])
        st = self._preprocess_state(obs)

        chunk = self._model.get_action(st, ct)
        if self._ensemble_buffer is not None:
            self._ensemble_buffer.add(chunk[0])
            return self._ensemble_buffer.get_action()
        return chunk[0]

    @staticmethod
    def _camera_name(obs: dict) -> str:
        return next(iter(obs["images"]))
