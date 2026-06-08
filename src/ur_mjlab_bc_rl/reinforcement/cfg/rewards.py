"""操作任务 MDP — 自定义奖励函数."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv


def reaching_reward(
    env: ManagerBasedRlEnv, object_name: str, ee_entity: str,
    std: float = 0.1,
) -> torch.Tensor:
    """末端接近物体的奖励（高斯核）。"""
    # 使用最后一个 body (= TCP/_tcp) 的位置
    ee_pos = env.scene[ee_entity].data.body_com_pos_w[:, -1]
    obj_pos = env.scene[object_name].data.body_com_pos_w[:, 0]
    dist = torch.norm(ee_pos - obj_pos, dim=-1)
    return torch.exp(-dist**2 / (2 * std**2))


def placing_reward(
    env: ManagerBasedRlEnv, object_name: str, target_entity: str,
    std: float = 0.1,
) -> torch.Tensor:
    """物体接近目标位置的奖励。"""
    obj_pos = env.scene[object_name].data.body_com_pos_w[:, 0]
    target_pos = env.scene[target_entity].data.body_com_pos_w[:, 0]
    dist = torch.norm(obj_pos[:, :2] - target_pos[:, :2], dim=-1)
    return torch.exp(-dist**2 / (2 * std**2))


def push_to_goal_reward(
    env: ManagerBasedRlEnv, object_name: str, goal_entity: str,
    pos_std: float = 0.1, yaw_std: float = 0.3,
) -> torch.Tensor:
    """Push-T 奖励：位置 + yaw 误差。"""
    obj_pos = env.scene[object_name].data.body_com_pos_w[:, 0]
    obj_quat = env.scene[object_name].data.body_com_quat_w[:, 0]
    goal_pos = env.scene[goal_entity].data.body_com_pos_w[:, 0]
    goal_quat = env.scene[goal_entity].data.body_com_quat_w[:, 0]

    pos_err = torch.norm(obj_pos[:, :2] - goal_pos[:, :2], dim=-1)
    r_pos = torch.exp(-pos_err**2 / (2 * pos_std**2))

    obj_yaw = torch.atan2(2 * (obj_quat[:, 0] * obj_quat[:, 3]), 1 - 2 * obj_quat[:, 3]**2)
    goal_yaw = torch.atan2(2 * (goal_quat[:, 0] * goal_quat[:, 3]), 1 - 2 * goal_quat[:, 3]**2)
    yaw_err = torch.abs(obj_yaw - goal_yaw)
    yaw_err = torch.min(yaw_err, 2 * torch.pi - yaw_err)
    r_yaw = torch.exp(-yaw_err**2 / (2 * yaw_std**2))

    return 0.5 * r_pos + 0.5 * r_yaw


def insertion_reward(
    env: ManagerBasedRlEnv, peg_entity: str, slot_entity: str,
    pos_std: float = 0.02, depth_weight: float = 3.0,
) -> torch.Tensor:
    """Peg-in-Slot 奖励：对齐 + 插入深度。"""
    peg_pos = env.scene[peg_entity].data.body_com_pos_w[:, 0]
    slot_pos = env.scene[slot_entity].data.body_com_pos_w[:, 0]
    # Tip: peg body + 0.065 in Z
    peg_tip_z = peg_pos[:, 2] + 0.065
    slot_top_z = slot_pos[:, 2] + 0.025

    lateral_err = torch.norm(peg_pos[:, :2] - slot_pos[:, :2], dim=-1)
    depth = torch.clamp(slot_top_z - peg_tip_z, min=0.0)

    r_align = torch.exp(-lateral_err**2 / (2 * pos_std**2))
    r_depth = depth_weight * depth

    return r_align + r_depth
