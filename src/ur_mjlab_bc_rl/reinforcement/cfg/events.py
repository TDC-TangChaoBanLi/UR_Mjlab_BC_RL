"""操作任务 MDP — 自定义事件（reset）。"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv


def reset_object_pose_uniform(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor,
    object_name: str,
    x_range: tuple[float, float] = (0.35, 0.55),
    y_range: tuple[float, float] = (-0.20, 0.20),
    z: float = 0.165,
    yaw_range: tuple[float, float] = (-3.14, 3.14),
) -> None:
    """均匀分布重置物体位姿，自动添加环境原点偏移。"""
    n = len(env_ids)
    obj = env.scene[object_name]

    # 生成随机位姿（相对于环境原点）
    pos_x = torch.rand(n, device=env.device) * (x_range[1] - x_range[0]) + x_range[0]
    pos_y = torch.rand(n, device=env.device) * (y_range[1] - y_range[0]) + y_range[0]
    pos_z = torch.full((n,), z, device=env.device)

    # 添加环境原点偏移（多环境 env_spacing）
    env_origins = env.scene.env_origins[env_ids]  # (n, 3)
    pos_x = pos_x + env_origins[:, 0]
    pos_y = pos_y + env_origins[:, 1]

    yaw = torch.rand(n, device=env.device) * (yaw_range[1] - yaw_range[0]) + yaw_range[0]
    qw = torch.cos(yaw / 2)
    qz = torch.sin(yaw / 2)

    pose = torch.stack([pos_x, pos_y, pos_z, qw, torch.zeros_like(qw), torch.zeros_like(qw), qz], dim=-1)
    obj.data.write_root_pose(pose, env_ids)


def reset_arm_to_default(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
    """重置机械臂到默认位姿。"""
    arm = env.scene["UR5e"]
    default_qpos = torch.tensor(
        [0.0, -1.57, 1.57, -1.57, -1.57, 0.0], device=env.device
    )
    arm.data.joint_pos[env_ids, :6] = default_qpos.unsqueeze(0).expand(len(env_ids), -1)
    arm.data.joint_vel[env_ids, :] = 0.0
    # Gripper open
    arm.data.joint_pos[env_ids, 6] = 0.0
