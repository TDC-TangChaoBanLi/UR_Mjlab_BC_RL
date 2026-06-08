"""操作任务 MDP — 自定义终止条件."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv


def object_dropped(env: ManagerBasedRlEnv, object_name: str, z_threshold: float = 0.05) -> torch.Tensor:
    """物体掉落检测（z < threshold）。"""
    obj_z = env.scene[object_name].data.body_com_pos_w[:, 0, 2]
    return obj_z < z_threshold


def pick_place_success(
    env: ManagerBasedRlEnv, cube_entity: str, plate_entity: str,
    pos_tol: float = 0.05, z_offset: float = 0.01, gripper_open_thresh: float = 0.05,
) -> torch.Tensor:
    """Pick-and-Place 成功：方块在盘子中心，夹爪释放。"""
    cube_pos = env.scene[cube_entity].data.body_com_pos_w[:, 0]
    plate_pos = env.scene[plate_entity].data.body_com_pos_w[:, 0]

    dist_xy = torch.norm(cube_pos[:, :2] - plate_pos[:, :2], dim=-1)
    above = cube_pos[:, 2] > (plate_pos[:, 2] + z_offset)

    # Gripper: joint 6 (left_knuckle)
    gripper_joint = env.scene["UR5e"].data.joint_pos[:, 6]
    gripper_open = gripper_joint < gripper_open_thresh

    return (dist_xy < pos_tol) & above & gripper_open


def push_t_success(
    env: ManagerBasedRlEnv, t_entity: str, goal_entity: str,
    pos_tol: float = 0.03, yaw_tol: float = 0.1745,
) -> torch.Tensor:
    """Push-T 成功：位置 + yaw 误差在阈值内。"""
    t_pos = env.scene[t_entity].data.body_com_pos_w[:, 0]
    t_quat = env.scene[t_entity].data.body_com_quat_w[:, 0]
    goal_pos = env.scene[goal_entity].data.body_com_pos_w[:, 0]
    goal_quat = env.scene[goal_entity].data.body_com_quat_w[:, 0]

    pos_err = torch.norm(t_pos[:, :2] - goal_pos[:, :2], dim=-1)
    t_yaw = torch.atan2(2 * t_quat[:, 0] * t_quat[:, 3], 1 - 2 * t_quat[:, 3]**2)
    g_yaw = torch.atan2(2 * goal_quat[:, 0] * goal_quat[:, 3], 1 - 2 * goal_quat[:, 3]**2)
    yaw_err = torch.abs(t_yaw - g_yaw)
    yaw_err = torch.min(yaw_err, 2 * torch.pi - yaw_err)

    return (pos_err < pos_tol) & (yaw_err < yaw_tol)
