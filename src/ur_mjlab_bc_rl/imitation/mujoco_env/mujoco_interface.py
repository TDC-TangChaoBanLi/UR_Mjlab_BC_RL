"""MuJoCo 仿真接口层。

封装所有原始 MuJoCo API 操作，作为环境其他模块的统一底层接口。
上层模块（ResetManager、ObservationCollector 等）通过本接口访问 MuJoCo，
不再直接持有 mjModel/mjData。

职责：
  - 启动仿真（加载 MJCF → model + data，可选 viewer）
  - 重置仿真（mj_resetData）
  - 推进仿真（mj_step）
  - 获取/写入 qpos、qvel、ctrl
  - 获取刚体/site 位姿
  - 获取传感器数据
  - 写入执行器控制量
  - 关节/执行器 ID 及地址查询
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import mujoco


class MujocoInterface:
    """MuJoCo 仿真接口。

    封装 model/data/viewer 的生命周期和所有底层 API 调用，
    对外暴露语义清晰的方法。

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
        """初始化 MuJoCo 仿真接口。

        Args:
            scene_path: MJCF 场景文件路径。
            render: 是否启动交互式 viewer（launch_passive）。
        """
        self._scene_path = Path(scene_path)

        # ── 加载模型与数据 ──
        self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(
            str(self._scene_path)
        )
        self.data: mujoco.MjData = mujoco.MjData(self.model)

        # ── 可选可视化 ──
        self.viewer: Optional[mujoco.viewer.Handle] = None
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    # ═══════════════════════════════════════════════════════════
    # 仿真控制
    # ═══════════════════════════════════════════════════════════

    def reset(self) -> None:
        """重置仿真到初始状态（mj_resetData + mj_forward）。"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, n: int = 1) -> None:
        """推进仿真 n 步。

        Args:
            n: 步数，默认 1。
        """
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def forward(self) -> None:
        """执行前向运动学/动力学（mj_forward）。"""
        mujoco.mj_forward(self.model, self.data)

    def close(self) -> None:
        """关闭 viewer 并释放资源。"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # ═══════════════════════════════════════════════════════════
    # 状态读写：qpos / qvel / ctrl
    # ═══════════════════════════════════════════════════════════

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

    # ═══════════════════════════════════════════════════════════
    # 位姿查询
    # ═══════════════════════════════════════════════════════════

    def get_body_pose(self, body_name: str) -> np.ndarray:
        """获取刚体位姿 [x, y, z, qw, qx, qy, qz]。

        Args:
            body_name: 刚体名称。

        Returns:
            shape (7,) 位姿数组。
        """
        body_id = self.model.body(body_name).id
        pos = self.data.xpos[body_id].copy()
        xmat = self.data.xmat[body_id].copy().reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, xmat.ravel())
        return np.concatenate([pos, quat])

    def get_site_pose(self, site_name: str) -> np.ndarray:
        """获取 site 位姿 [x, y, z, qw, qx, qy, qz]。

        Args:
            site_name: site 名称。

        Returns:
            shape (7,) 位姿数组。
        """
        site_id = self.model.site(site_name).id
        pos = self.data.site_xpos[site_id].copy()
        xmat = self.data.site_xmat[site_id].copy().reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, xmat.ravel())
        return np.concatenate([pos, quat])

    # ═══════════════════════════════════════════════════════════
    # 关节查询
    # ═══════════════════════════════════════════════════════════

    def get_joint_id(self, name: str) -> int:
        """获取关节 ID。

        Args:
            name: 关节名称。

        Returns:
            关节 ID（model.joint(name).id）。
        """
        return self.model.joint(name).id

    def get_joint_qposadr(self, name: str) -> int:
        """获取关节在 qpos 中的起始地址。

        Args:
            name: 关节名称。

        Returns:
            qpos 地址索引。
        """
        jid = self.get_joint_id(name)
        return self.model.jnt_qposadr[jid]

    def get_joint_dofadr(self, name: str) -> int:
        """获取关节在 qvel 中的起始地址。

        Args:
            name: 关节名称。

        Returns:
            dof 地址索引。
        """
        jid = self.get_joint_id(name)
        return self.model.jnt_dofadr[jid]

    def get_joint_qpos(self, names: list[str]) -> np.ndarray:
        """按关节名列表获取当前关节位置。

        Args:
            names: 关节名称列表。

        Returns:
            shape (len(names),) 关节位置数组。
        """
        return np.array([
            self.data.qpos[self.get_joint_qposadr(n)] for n in names
        ])

    def set_joint_qpos(self, name: str, value: float) -> None:
        """设置单个关节的位置。

        Args:
            name: 关节名称。
            value: 目标位置。
        """
        adr = self.get_joint_qposadr(name)
        if adr >= 0:
            self.data.qpos[adr] = value

    def set_joint_qpos_batch(self, mapping: dict[str, float]) -> None:
        """批量设置多个关节位置。

        Args:
            mapping: {关节名: 目标值} 字典。
        """
        for name, value in mapping.items():
            self.set_joint_qpos(name, value)

    def get_joint_qvel(self, names: list[str]) -> np.ndarray:
        """按关节名列表获取当前关节速度。

        Args:
            names: 关节名称列表。

        Returns:
            shape (len(names),) 关节速度数组。
        """
        return np.array([
            self.data.qvel[self.get_joint_dofadr(n)] for n in names
        ])

    # ═══════════════════════════════════════════════════════════
    # 执行器查询
    # ═══════════════════════════════════════════════════════════

    def get_actuator_id(self, name: str) -> int:
        """获取执行器 ID。

        Args:
            name: 执行器名称。

        Returns:
            执行器 ID。
        """
        return self.model.actuator(name).id

    # ═══════════════════════════════════════════════════════════
    # 传感器
    # ═══════════════════════════════════════════════════════════

    def get_sensor_data(self, name: str | None = None) -> np.ndarray:
        """获取传感器数据。

        Args:
            name: 传感器名称。若为 None，返回全部 sensordata 副本。

        Returns:
            传感器数据数组。
        """
        if name is None:
            return self.data.sensordata.copy()
        sensor_id = self.model.sensor(name).id
        adr = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        return self.data.sensordata[adr : adr + dim].copy()

    # ═══════════════════════════════════════════════════════════
    # Viewer 工具
    # ═══════════════════════════════════════════════════════════

    def sync_viewer(self) -> None:
        """同步 viewer 显示（若已启用）。"""
        if self.viewer is not None:
            self.viewer.sync()

    def is_viewer_running(self) -> bool:
        """检查 viewer 是否仍在运行。"""
        if self.viewer is None:
            return False
        return self.viewer.is_running()

    def set_viewer_camera(
        self,
        lookat: tuple[float, float, float] = (0.45, 0.0, 0.65),
        distance: float = 1.8,
        elevation: float = -25.0,
        azimuth: float = 130.0,
    ) -> None:
        """设置 viewer 相机视角（仅当 viewer 启用时生效）。

        Args:
            lookat: 相机注视点 (x, y, z)。
            distance: 相机距离。
            elevation: 仰角 (度)。
            azimuth: 方位角 (度)。
        """
        if self.viewer is None:
            return
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.viewer.cam.lookat[:] = lookat
        self.viewer.cam.distance = distance
        self.viewer.cam.elevation = elevation
        self.viewer.cam.azimuth = azimuth

    # ═══════════════════════════════════════════════════════════
    # 几何 / Body 辅助
    # ═══════════════════════════════════════════════════════════

    def get_body_joint_id(self, body_name: str) -> int | None:
        """查找指定 body 对应的 freejoint ID。

        Args:
            body_name: 刚体名称。

        Returns:
            关节 ID，若找不到则返回 None。
        """
        try:
            body_id = self.model.body(body_name).id
        except KeyError:
            return None

        for jid in range(self.model.njnt):
            if self.model.jnt_bodyid[jid] == body_id:
                return jid
        return None

    def body_exists(self, body_name: str) -> bool:
        """检查 body 是否存在于模型中。"""
        try:
            self.model.body(body_name)
            return True
        except KeyError:
            return False
