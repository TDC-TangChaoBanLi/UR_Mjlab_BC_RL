"""纯 MuJoCo 环境接口。

不依赖 MjLab，直接使用 MuJoCo 进行环境交互。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import mujoco
import mujoco.viewer


class MuJoCoEnv:
    """纯 MuJoCo 环境基类。
    
    提供基础的 MJCF 加载、reset、step 等接口。
    """

    def __init__(
        self,
        mjcf_path: str | Path,
        render: bool = False,
    ) -> None:
        """初始化环境。
        
        Args:
            mjcf_path: MJCF 文件路径
            render: 是否启用渲染
        """
        self.mjcf_path = Path(mjcf_path)
        self.render = render
        
        # 加载模型
        self.model = mujoco.MjModel.from_file(str(self.mjcf_path))
        self.data = mujoco.MjData(self.model)
        
        # 渲染器（可选）
        self.viewer = None
        if render:
            self.viewer = mujoco.viewer.launch(self.model, self.data)

    def reset(self) -> None:
        """重置环境到初始状态。"""
        mujoco.mj_resetData(self.model, self.data)

    def step(self, action: np.ndarray | None = None, nstep: int = 1) -> None:
        """执行一步或多步模拟。
        
        Args:
            action: 可选的控制动作（如果为 None，则不修改控制量）
            nstep: 执行的步数
        """
        if action is not None:
            self.data.ctrl[:] = action
        
        for _ in range(nstep):
            mujoco.mj_step(self.model, self.data)
        
        if self.viewer is not None:
            self.viewer.sync()

    def get_time(self) -> float:
        """获取当前模拟时间。"""
        return self.data.time

    def get_qpos(self) -> np.ndarray:
        """获取所有关节位置。"""
        return self.data.qpos.copy()

    def get_qvel(self) -> np.ndarray:
        """获取所有关节速度。"""
        return self.data.qvel.copy()

    def set_qpos(self, qpos: np.ndarray) -> None:
        """设置关节位置。"""
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

    def set_qvel(self, qvel: np.ndarray) -> None:
        """设置关节速度。"""
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def get_body_pose(self, body_name: str) -> np.ndarray:
        """获取刚体位姿 [x, y, z, qw, qx, qy, qz]。
        
        Args:
            body_name: 刚体名称
        
        Returns:
            [7] 位姿数组
        """
        body_id = self.model.body(body_name).id
        pos = self.data.xpos[body_id].copy()
        xmat = self.data.xmat[body_id].copy().reshape(3, 3)
        
        # 旋转矩阵转四元数
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, xmat.ravel())
        
        return np.concatenate([pos, quat])

    def get_site_pose(self, site_name: str) -> np.ndarray:
        """获取 site 位姿 [x, y, z, qw, qx, qy, qz]。
        
        Args:
            site_name: site 名称
        
        Returns:
            [7] 位姿数组
        """
        site_id = self.model.site(site_name).id
        pos = self.data.site_xpos[site_id].copy()
        xmat = self.data.site_xmat[site_id].copy().reshape(3, 3)
        
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, xmat.ravel())
        
        return np.concatenate([pos, quat])

    def get_ee_pose(self, ee_site_name: str = "_tcp") -> np.ndarray:
        """获取末端执行器位姿。
        
        Args:
            ee_site_name: 末端 site 名称
        
        Returns:
            [7] 位姿数组
        """
        return self.get_site_pose(ee_site_name)

    def close(self) -> None:
        """关闭环境和渲染器。"""
        if self.viewer is not None:
            self.viewer.close()


# 导出子模块
from .mujoco_env import *  # noqa
from .expert_generation import *  # noqa
from .dataset import *  # noqa
