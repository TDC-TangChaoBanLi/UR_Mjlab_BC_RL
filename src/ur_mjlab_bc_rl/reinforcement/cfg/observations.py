"""操作任务 MDP — 自定义观测函数."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg


def ee_to_object_pos(env: ManagerBasedRlEnv, object_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """末端到物体的相对位置（世界系）。"""
    # 使用 body_com_pos_w 获取 body 位置
    ee_pos = env.scene[asset_cfg.name].data.body_com_pos_w[:, asset_cfg.body_ids[0]]
    obj_pos = env.scene[object_name].data.body_com_pos_w[:, 0]
    return obj_pos - ee_pos


def object_to_goal_pos(env: ManagerBasedRlEnv, object_name: str, goal_entity: str) -> torch.Tensor:
    """物体到目标的相对位置。"""
    obj_pos = env.scene[object_name].data.body_com_pos_w[:, 0]
    goal_pos = env.scene[goal_entity].data.body_com_pos_w[:, 0]
    return goal_pos - obj_pos
