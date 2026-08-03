"""MuJoCo 底层封装 — 精简自 simulate/env/mujoco_interface.py。

提供纯 MuJoCo API 操作，支持 _ensure_env() 延迟初始化模式
（兼容 gym.vector.AsyncVectorEnv 的 fork 机制）。

Usage:
    mj = MujocoWrapper("assets/mujoco/scenes/dual_pick_place.xml")
    mj.open()          # 或延迟到首次 step/reset 时自动调用
    mj.step(ctrl)
    mj.close()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import mujoco

from ..utils.config_loader import CameraConfig


class MujocoWrapper:
    """MuJoCo 仿真封装 — 纯底层 API。

    不包含时序编排、controller、观测采集等高层逻辑。
    使用 _ensure_env() 模式延迟 GPU/EGL 分配。
    """

    def __init__(
        self,
        scene_path: str | Path,
        render: bool = False,
    ) -> None:
        self._scene_path = Path(scene_path)
        self._render_flag = render

        # 延迟初始化（_ensure_env 时创建）
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.viewer: Optional[mujoco.viewer.Handle] = None

    # ── 生命周期 ───────────────────────────────────────

    def open(self) -> None:
        """显式初始化（可选，不调用则首次 step/reset 自动初始化）。"""
        self._ensure_env()

    def _ensure_env(self) -> None:
        """延迟初始化 MuJoCo — 兼容 AsyncVectorEnv fork。

        GPU/EGL 上下文必须在 worker 进程内分配，
        不能在父进程 __init__ 时分配。
        """
        if self.model is not None:
            return  # 已初始化

        self.model = mujoco.MjModel.from_xml_path(str(self._scene_path))
        self.data = mujoco.MjData(self.model)

        if self._render_flag:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def close(self) -> None:
        """释放所有资源。"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.model = None
        self.data = None

    # ── 仿真控制 ───────────────────────────────────────

    def reset(self) -> None:
        """重置仿真数据 + 前向。"""
        self._ensure_env()
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, n: int = 1) -> None:
        """推进仿真 n 步（纯物理步进）。"""
        self._ensure_env()
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def forward(self) -> None:
        """执行前向动力学。"""
        self._ensure_env()
        mujoco.mj_forward(self.model, self.data)

    @property
    def physics_dt(self) -> float:
        """物理仿真步长。"""
        self._ensure_env()
        return self.model.opt.timestep

    @property
    def sim_time(self) -> float:
        """当前仿真时间。"""
        self._ensure_env()
        return self.data.time

    # ── Viewer ─────────────────────────────────────────

    def is_viewer_running(self) -> bool:
        return self.viewer is not None and self.viewer.is_running()

    def set_viewer_camera(
        self,
        lookat: tuple[float, float, float] = (0.45, 0.0, 0.65),
        distance: float = 1.8,
        elevation: float = -25.0,
        azimuth: float = 130.0,
    ) -> None:
        if self.viewer is not None:
            with self.viewer.lock():
                self.viewer.cam.lookat[:] = lookat
                self.viewer.cam.distance = distance
                self.viewer.cam.elevation = elevation
                self.viewer.cam.azimuth = azimuth

    def sync_viewer(self) -> None:
        if self.viewer is not None:
            self.viewer.sync()

    def render(self) -> None:
        """渲染并同步 MuJoCo viewer（仅当 viewer 开启时生效）。"""
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    # ── 状态读写 ───────────────────────────────────────

    def get_qpos(self) -> np.ndarray:
        self._ensure_env()
        return self.data.qpos.copy()

    def set_qpos(self, qpos: np.ndarray) -> None:
        self._ensure_env()
        self.data.qpos[:] = qpos

    def get_qvel(self) -> np.ndarray:
        self._ensure_env()
        return self.data.qvel.copy()

    def set_qvel(self, qvel: np.ndarray) -> None:
        self._ensure_env()
        self.data.qvel[:] = qvel

    def get_ctrl(self) -> np.ndarray:
        self._ensure_env()
        return self.data.ctrl.copy()

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        self._ensure_env()
        self.data.ctrl[:] = ctrl

    # ── 关节/执行器查询 ────────────────────────────────

    def get_joint_id(self, name: str) -> int:
        self._ensure_env()
        return self.model.joint(name).id

    def get_joint_qposadr(self, name: str) -> int:
        self._ensure_env()
        return self.model.jnt_qposadr[self.get_joint_id(name)]

    def get_actuator_id(self, name: str) -> int:
        self._ensure_env()
        return self.model.actuator(name).id

    def get_joint_qpos(self, joint_names: list[str]) -> np.ndarray:
        """获取指定关节位置。"""
        self._ensure_env()
        return np.array([
            self.data.qpos[self.get_joint_qposadr(n)] for n in joint_names
        ], dtype=np.float64)

    def get_joint_qvel(self, joint_names: list[str]) -> np.ndarray:
        """获取指定关节速度。"""
        self._ensure_env()
        dofadrs = [self.model.jnt_dofadr[self.get_joint_id(n)] for n in joint_names]
        return np.array([self.data.qvel[a] for a in dofadrs], dtype=np.float64)

    def set_joint_qpos(self, name: str, value: float) -> None:
        """设置单个关节位置。"""
        self._ensure_env()
        adr = self.get_joint_qposadr(name)
        if adr >= 0:
            self.data.qpos[adr] = value

    def get_body_joint_id(self, body_name: str) -> None | int:
        """获取物体 freejoint 的 ID，不存在则返回 None。"""
        self._ensure_env()
        body_id = self.model.body(body_name).id
        jntadr = self.model.body_jntadr[body_id]
        if jntadr < 0:
            return None
        jntnum = self.model.body_jntnum[body_id]
        if jntnum == 0:
            return None
        return jntadr

    def get_sensor(self, name: str) -> np.ndarray:
        """读取传感器数据（别名 get_sensor_data）。"""
        return self.get_sensor_data(name)

    def get_sensor_data(self, name: str) -> np.ndarray:
        """读取传感器数据。"""
        self._ensure_env()
        sid = self.model.sensor(name).id
        sadr = self.model.sensor_adr[sid]
        sdim = self.model.sensor_dim[sid]
        return self.data.sensordata[sadr:sadr + sdim].copy()

    def get_body_pos(self, name: str) -> np.ndarray:
        """获取 body 世界坐标位置。"""
        self._ensure_env()
        return self.data.body(name).xpos.copy()

    def get_body_pR(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """获取 body 位置和旋转矩阵。"""
        self._ensure_env()
        body = self.data.body(name)
        return body.xpos.copy(), body.xmat.reshape(3, 3).copy()

    # ── 相机 ───────────────────────────────────────────

    def get_camera_rgb(
        self, cam_name: str, height: int, width: int,
    ) -> np.ndarray:
        """渲染指定相机的 RGB 图像。"""
        self._ensure_env()
        cam_id = self.model.camera(cam_name).id
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        renderer.disable_depth_rendering()
        renderer.update_scene(self.data, camera=cam_id)
        rgb = renderer.render()
        renderer.close()
        return np.ascontiguousarray(rgb)

    def get_camera_depth(
        self, cam_name: str, height: int, width: int,
    ) -> np.ndarray:
        """渲染指定相机的深度图像（米制）。"""
        self._ensure_env()
        cam_id = self.model.camera(cam_name).id
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        renderer.enable_depth_rendering()
        renderer.update_scene(self.data, camera=cam_id)
        depth = renderer.render()
        renderer.close()
        return np.ascontiguousarray(depth.astype(np.float32, copy=False))

    def get_body_pose(self, body_name: str) -> "np.ndarray":
        """获取刚体位姿 [x, y, z, qw, qx, qy, qz]。"""
        self._ensure_env()
        body_id = self.model.body(body_name).id
        pos = self.data.xpos[body_id].copy()
        xmat = self.data.xmat[body_id].copy().reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, xmat.ravel())
        return np.concatenate([pos, quat])

    def get_site_pose(self, site_name: str) -> "np.ndarray":
        """获取 site 位姿 [x, y, z, qw, qx, qy, qz]。"""
        self._ensure_env()
        site_id = self.model.site(site_name).id
        pos = self.data.site_xpos[site_id].copy()
        xmat = self.data.site_xmat[site_id].copy().reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, xmat.ravel())
        return np.concatenate([pos, quat])

    def get_body_names(self, prefix: str = "") -> list[str]:
        """获取 body 名称列表（按 prefix 过滤）。"""
        self._ensure_env()
        names = []
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name.startswith(prefix):
                names.append(name)
        return names

    def set_body_pose(
        self, body_name: str, pos: np.ndarray, rot: np.ndarray | None = None,
    ) -> None:
        """设置 free body 的位置和朝向。"""
        self._ensure_env()
        jntadr = self.model.body_jntadr[self.model.body(body_name).id]
        if jntadr < 0:
            return
        self.data.qpos[jntadr:jntadr + 3] = pos[:3]
        if rot is not None:
            # 旋转变为四元数（简化：使用单位四元数）
            self.data.qpos[jntadr + 3:jntadr + 7] = [1.0, 0.0, 0.0, 0.0]
