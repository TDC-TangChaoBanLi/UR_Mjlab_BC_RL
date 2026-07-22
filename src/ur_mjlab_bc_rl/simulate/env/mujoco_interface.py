"""MuJoCo 仿真接口层。

封装所有原始 MuJoCo API 操作，作为环境其他模块的统一底层接口。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import mujoco


class MujocoInterface:
    """MuJoCo 仿真接口。

    封装 model/data/viewer 的生命周期和所有底层 API 调用。

    Usage:
        mj = MujocoInterface("assets/mujoco/scenes/pick_place.xml", render=True)
        mj.reset()
        for _ in range(1000):
            mj.step()
        mj.close()
    """

    def __init__(
        self,
        scene_path: str | Path,
        render: bool = False,
    ) -> None:
        self._scene_path = Path(scene_path)

        self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(
            str(self._scene_path)
        )
        self.data: mujoco.MjData = mujoco.MjData(self.model)

        self.viewer: Optional[mujoco.viewer.Handle] = None
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    # ── 仿真控制 ───────────────────────────────────────

    def reset(self) -> None:
        """重置仿真（mj_resetData + mj_forward）。"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, n: int = 1) -> None:
        """推进仿真 n 步。"""
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def forward(self) -> None:
        """执行前向动力学/运动学。"""
        mujoco.mj_forward(self.model, self.data)

    def close(self) -> None:
        """关闭 viewer 释放资源。"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def is_viewer_running(self) -> bool:
        """检查 viewer 是否仍在运行。"""
        return self.viewer is not None and self.viewer.is_running()

    def set_viewer_camera(
        self,
        lookat: tuple[float, float, float] = (0.45, 0.0, 0.65),
        distance: float = 1.8,
        elevation: float = -25.0,
        azimuth: float = 130.0,
    ) -> None:
        """设置 viewer 相机参数。"""
        if self.viewer is not None:
            with self.viewer.lock():
                self.viewer.cam.lookat[:] = lookat
                self.viewer.cam.distance = distance
                self.viewer.cam.elevation = elevation
                self.viewer.cam.azimuth = azimuth

    def sync_viewer(self) -> None:
        """同步 viewer 显示。"""
        if self.viewer is not None:
            self.viewer.sync()

    # ── 状态读写 ───────────────────────────────────────

    def get_qpos(self) -> np.ndarray:
        """获取所有关节位置（副本）。"""
        return self.data.qpos.copy()

    def set_qpos(self, qpos: np.ndarray) -> None:
        """覆盖所有关节位置。"""
        self.data.qpos[:] = qpos

    def get_qvel(self) -> np.ndarray:
        """获取所有关节速度（副本）。"""
        return self.data.qvel.copy()

    def set_qvel(self, qvel: np.ndarray) -> None:
        """覆盖所有关节速度。"""
        self.data.qvel[:] = qvel

    def get_ctrl(self) -> np.ndarray:
        """获取所有执行器控制量（副本）。"""
        return self.data.ctrl.copy()

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        """写入所有执行器控制量。"""
        self.data.ctrl[:] = ctrl

    def get_time(self) -> float:
        """获取当前仿真时间。"""
        return self.data.time

    # ── 关节查询 ───────────────────────────────────────

    def get_joint_id(self, name: str) -> int:
        """获取关节 ID。"""
        return self.model.joint(name).id

    def get_joint_qposadr(self, name: str) -> int:
        """获取关节在 qpos 中的起始地址。"""
        return self.model.jnt_qposadr[self.get_joint_id(name)]

    def get_joint_dofadr(self, name: str) -> int:
        """获取关节在 qvel 中的起始地址。"""
        return self.model.jnt_dofadr[self.get_joint_id(name)]

    def get_actuator_id(self, name: str) -> int:
        """获取执行器 ID。"""
        return self.model.actuator(name).id

    def get_body_joint_id(self, body_name: str) -> Optional[int]:
        """获取物体 freejoint 的 ID，不存在则返回 None。"""
        body_id = self.model.body(body_name).id
        jntadr = self.model.body_jntadr[body_id]
        if jntadr < 0:
            return None
        jntnum = self.model.body_jntnum[body_id]
        if jntnum == 0:
            return None
        return jntadr

    def get_joint_qpos(self, joint_names: list[str]) -> np.ndarray:
        """批量获取指定关节位置。"""
        result = np.zeros(len(joint_names), dtype=np.float64)
        for i, name in enumerate(joint_names):
            adr = self.get_joint_qposadr(name)
            if adr >= 0:
                result[i] = self.data.qpos[adr]
        return result

    def get_joint_qvel(self, joint_names: list[str]) -> np.ndarray:
        """批量获取指定关节速度。"""
        result = np.zeros(len(joint_names), dtype=np.float64)
        for i, name in enumerate(joint_names):
            adr = self.get_joint_dofadr(name)
            if adr >= 0:
                result[i] = self.data.qvel[adr]
        return result

    def set_joint_qpos(self, name: str, value: float) -> None:
        """设置单个关节位置。"""
        adr = self.get_joint_qposadr(name)
        if adr >= 0:
            self.data.qpos[adr] = value

    # ── 位姿查询 ───────────────────────────────────────

    def get_body_pose(self, body_name: str) -> np.ndarray:
        """获取刚体位姿 [x, y, z, qw, qx, qy, qz]。"""
        body_id = self.model.body(body_name).id
        pos = self.data.xpos[body_id].copy()
        xmat = self.data.xmat[body_id].copy().reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, xmat.ravel())
        return np.concatenate([pos, quat])

    def get_site_pose(self, site_name: str) -> np.ndarray:
        """获取 site 位姿 [x, y, z, qw, qx, qy, qz]。"""
        site_id = self.model.site(site_name).id
        pos = self.data.site_xpos[site_id].copy()
        xmat = self.data.site_xmat[site_id].copy().reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, xmat.ravel())
        return np.concatenate([pos, quat])
