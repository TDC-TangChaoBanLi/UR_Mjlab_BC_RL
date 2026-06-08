"""多任务 UR5 操作环境配置.

基于 mjlab ManagerBasedRlEnvCfg，支持:
- Pick-and-Place (task 0)
- Push-T (task 1)
- Peg-in-Slot (task 2)

特性:
- 手眼 RGBD 相机 (128x128)
- 关节增量 + 夹爪动作 (7-dim)
- mujoco-warp GPU 加速
- RSL-RL PPO 训练
"""

from __future__ import annotations

import torch

from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.envs.mdp.actions import RelativeJointPositionActionCfg
from mjlab.envs.mdp.events import reset_root_state_uniform
from mjlab.envs.mdp.observations import joint_pos_rel, joint_vel_rel, last_action
from mjlab.envs.mdp.rewards import action_rate_l2
from mjlab.envs.mdp.terminations import time_out
from mjlab.sensor import CameraSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.tasks.manipulation import mdp as manipulation_mdp

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

from ..models.policy.rsl_adapter import UR5MultimodalModelCfg

from .cfg.constants import (
    UR5E_ENTITY_NAME, CUBE_ENTITY_NAME, PLATE_ENTITY_NAME,
    T_SHAPE_ENTITY_NAME, GOAL_MARKER_ENTITY_NAME,
    PEG_ENTITY_NAME, SLOT_ENTITY_NAME,
    EE_SITE_NAME,
)
from .cfg.scene import (
    get_ur5e_entity_cfg, get_cube_entity_cfg, get_plate_entity_cfg,
    get_t_shape_entity_cfg, get_goal_marker_entity_cfg,
    get_peg_entity_cfg, get_slot_entity_cfg,
)
from .cfg.observations import ee_to_object_pos, object_to_goal_pos
from .cfg.rewards import reaching_reward, placing_reward, push_to_goal_reward, insertion_reward
from .cfg.terminations import object_dropped, pick_place_success, push_t_success
from .cfg.events import reset_object_pose_uniform, reset_arm_to_default


# =========================================================================
# 通用: 相机传感器 + 观测
# =========================================================================

def _wrist_rgbd_camera() -> CameraSensorCfg:
    """手腕 RGBD 相机传感器 (128x128)。"""
    return CameraSensorCfg(
        name="wrist_rgbd",
        camera_name="UR5e/realsense_link_CAMERA",  # entity 前缀!
        width=128,
        height=128,
        data_types=("rgb", "depth"),
    )


def _make_vision_obs_terms(entity_name: str) -> dict:
    """创建视觉观测 term（camera_rgb + camera_depth）。"""
    return {
        "camera_rgb": ObservationTermCfg(
            func=manipulation_mdp.camera_rgb,
            params={"sensor_name": "wrist_rgbd"},
        ),
        "camera_depth": ObservationTermCfg(
            func=manipulation_mdp.camera_depth,
            params={"sensor_name": "wrist_rgbd", "cutoff_distance": 2.0},
        ),
    }


# =========================================================================
# 通用: 基础 RL 配置
# =========================================================================

def _make_base_ppo_runner(experiment_name: str) -> RslRlOnPolicyRunnerCfg:
    """创建 PPO runner 配置，从 configs/model/ppo.yaml 加载模型参数。"""
    from ..config_loader import load_multimodal_model, load_reinforcement_config
    model = load_multimodal_model()
    rl = load_reinforcement_config()

    return RslRlOnPolicyRunnerCfg(
        actor=UR5MultimodalModelCfg(
            architecture_cfg=model,
            distribution_cfg=rl["distribution_cfg"],
        ),
        critic=RslRlModelCfg(**rl["critic"]),
        algorithm=RslRlPpoAlgorithmCfg(**rl["algorithm"]),
        experiment_name=experiment_name,
    )


def _make_sim() -> SimulationCfg:
    return SimulationCfg(
        mujoco=MujocoCfg(timestep=0.002, iterations=10, ls_iterations=20, impratio=10),
    )


def _make_viewer() -> ViewerConfig:
    return ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name=UR5E_ENTITY_NAME,
        body_name="",
        distance=1.5,
        elevation=-5.0,
        azimuth=120.0,
    )


# =========================================================================
# 任务 0: Pick-and-Place
# =========================================================================

def make_pick_place_env_cfg() -> ManagerBasedRlEnvCfg:
    scene = SceneCfg(
        num_envs=1,
        env_spacing=1.0,
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={
            UR5E_ENTITY_NAME: get_ur5e_entity_cfg(),
            CUBE_ENTITY_NAME: get_cube_entity_cfg(),
            PLATE_ENTITY_NAME: get_plate_entity_cfg(),
        },
        sensors=(_wrist_rgbd_camera(),),
    )

    # ---- State observations ----
    state_terms = {
        "joint_pos": ObservationTermCfg(
            func=joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg(UR5E_ENTITY_NAME, joint_names=(".*",))},
        ),
        "joint_vel": ObservationTermCfg(
            func=joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg(UR5E_ENTITY_NAME, joint_names=(".*",))},
        ),
        "ee_to_cube": ObservationTermCfg(
            func=ee_to_object_pos,
            params={
                "object_name": CUBE_ENTITY_NAME,
                "asset_cfg": SceneEntityCfg(UR5E_ENTITY_NAME, body_names=("_tcp",)),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "cube_to_plate": ObservationTermCfg(
            func=object_to_goal_pos,
            params={"object_name": CUBE_ENTITY_NAME, "goal_entity": PLATE_ENTITY_NAME},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "actions": ObservationTermCfg(func=last_action),
    }

    # ---- Vision observations (only for actor) ----
    vision_terms = _make_vision_obs_terms(UR5E_ENTITY_NAME)
    actor_terms = {**state_terms, **vision_terms}
    critic_terms = {**state_terms}

    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
    }

    # Actions: 6 arm + 1 gripper (gripper handled by separate ctrl write)
    actions: dict[str, ActionTermCfg] = {
        "joint_delta_pos": RelativeJointPositionActionCfg(
            entity_name=UR5E_ENTITY_NAME,
            actuator_names=(".*",),
            scale=0.05,
        ),
    }

    events: dict[str, EventTermCfg] = {
        "reset_base": EventTermCfg(
            func=reset_root_state_uniform, mode="reset",
            params={"pose_range": {}, "velocity_range": {},
                    "asset_cfg": SceneEntityCfg(UR5E_ENTITY_NAME, joint_names=(".*",))},
        ),
        "reset_arm": EventTermCfg(func=reset_arm_to_default, mode="reset"),
        "reset_cube": EventTermCfg(
            func=reset_object_pose_uniform, mode="reset",
            params={"object_name": CUBE_ENTITY_NAME, "x_range": (0.35, 0.55),
                    "y_range": (-0.20, 0.20), "z": 0.165, "yaw_range": (-3.14, 3.14)},
        ),
        "reset_plate": EventTermCfg(
            func=reset_object_pose_uniform, mode="reset",
            params={"object_name": PLATE_ENTITY_NAME, "x_range": (0.45, 0.65),
                    "y_range": (-0.25, 0.25), "z": 0.135, "yaw_range": (-3.14, 3.14)},
        ),
    }

    rewards: dict[str, RewardTermCfg] = {
        "reaching": RewardTermCfg(func=reaching_reward, weight=1.0,
                                  params={"object_name": CUBE_ENTITY_NAME, "ee_entity": UR5E_ENTITY_NAME, "std": 0.15}),
        "placing": RewardTermCfg(func=placing_reward, weight=2.0,
                                 params={"object_name": CUBE_ENTITY_NAME, "target_entity": PLATE_ENTITY_NAME, "std": 0.15}),
        "action_penalty": RewardTermCfg(func=action_rate_l2, weight=-0.01),
    }

    terminations: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
        "cube_dropped": TerminationTermCfg(func=object_dropped,
                                           params={"object_name": CUBE_ENTITY_NAME, "z_threshold": 0.05}),
        "success": TerminationTermCfg(func=pick_place_success,
                                      params={"cube_entity": CUBE_ENTITY_NAME, "plate_entity": PLATE_ENTITY_NAME}),
    }

    return ManagerBasedRlEnvCfg(
        scene=scene, observations=observations, actions=actions,
        commands={}, events=events, rewards=rewards, terminations=terminations,
        viewer=_make_viewer(), sim=_make_sim(),
        decimation=25, episode_length_s=10.0,
    )


def pick_place_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_pick_place_env_cfg()
    if play:
        cfg.observations["actor"].enable_corruption = False
    return cfg


def pick_place_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return _make_base_ppo_runner("pick_place")


# =========================================================================
# 任务 1: Push-T
# =========================================================================

def make_push_t_env_cfg() -> ManagerBasedRlEnvCfg:
    scene = SceneCfg(
        num_envs=1, env_spacing=1.0,
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={
            UR5E_ENTITY_NAME: get_ur5e_entity_cfg(),
            T_SHAPE_ENTITY_NAME: get_t_shape_entity_cfg(),
            GOAL_MARKER_ENTITY_NAME: get_goal_marker_entity_cfg(),
        },
        sensors=(_wrist_rgbd_camera(),),
    )

    state_terms = {
        "joint_pos": ObservationTermCfg(
            func=joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg(UR5E_ENTITY_NAME, joint_names=(".*",))},
        ),
        "joint_vel": ObservationTermCfg(
            func=joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg(UR5E_ENTITY_NAME, joint_names=(".*",))},
        ),
        "ee_to_t": ObservationTermCfg(
            func=ee_to_object_pos,
            params={"object_name": T_SHAPE_ENTITY_NAME,
                    "asset_cfg": SceneEntityCfg(UR5E_ENTITY_NAME, body_names=("_tcp",))},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "t_to_goal": ObservationTermCfg(
            func=object_to_goal_pos,
            params={"object_name": T_SHAPE_ENTITY_NAME, "goal_entity": GOAL_MARKER_ENTITY_NAME},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "actions": ObservationTermCfg(func=last_action),
    }

    vision_terms = _make_vision_obs_terms(UR5E_ENTITY_NAME)
    actor_terms = {**state_terms, **vision_terms}
    critic_terms = {**state_terms}

    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
    }

    actions: dict[str, ActionTermCfg] = {
        "joint_delta_pos": RelativeJointPositionActionCfg(
            entity_name=UR5E_ENTITY_NAME,
            actuator_names=(".*",),
            scale=0.05,
        ),
    }

    events: dict[str, EventTermCfg] = {
        "reset_base": EventTermCfg(
            func=reset_root_state_uniform, mode="reset",
            params={"pose_range": {}, "velocity_range": {},
                    "asset_cfg": SceneEntityCfg(UR5E_ENTITY_NAME, joint_names=(".*",))},
        ),
        "reset_arm": EventTermCfg(func=reset_arm_to_default, mode="reset"),
        "reset_t": EventTermCfg(
            func=reset_object_pose_uniform, mode="reset",
            params={"object_name": T_SHAPE_ENTITY_NAME, "x_range": (0.35, 0.55),
                    "y_range": (-0.20, 0.20), "z": 0.152, "yaw_range": (-3.14, 3.14)},
        ),
        "reset_goal": EventTermCfg(
            func=reset_object_pose_uniform, mode="reset",
            params={"object_name": GOAL_MARKER_ENTITY_NAME, "x_range": (0.35, 0.55),
                    "y_range": (-0.20, 0.20), "z": 0.147, "yaw_range": (-3.14, 3.14)},
        ),
    }

    rewards: dict[str, RewardTermCfg] = {
        "push_reward": RewardTermCfg(
            func=push_to_goal_reward, weight=1.0,
            params={"object_name": T_SHAPE_ENTITY_NAME, "goal_entity": GOAL_MARKER_ENTITY_NAME,
                    "pos_std": 0.1, "yaw_std": 0.3},
        ),
        "action_penalty": RewardTermCfg(func=action_rate_l2, weight=-0.01),
    }

    terminations: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
        "t_dropped": TerminationTermCfg(func=object_dropped,
                                        params={"object_name": T_SHAPE_ENTITY_NAME, "z_threshold": 0.05}),
        "success": TerminationTermCfg(func=push_t_success,
                                      params={"t_entity": T_SHAPE_ENTITY_NAME, "goal_entity": GOAL_MARKER_ENTITY_NAME}),
    }

    return ManagerBasedRlEnvCfg(
        scene=scene, observations=observations, actions=actions,
        commands={}, events=events, rewards=rewards, terminations=terminations,
        viewer=_make_viewer(), sim=_make_sim(),
        decimation=25, episode_length_s=15.0,
    )


def push_t_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_push_t_env_cfg()
    if play:
        cfg.observations["actor"].enable_corruption = False
    return cfg


def push_t_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return _make_base_ppo_runner("push_t")


# =========================================================================
# 任务 2: Peg-in-Slot
# =========================================================================

def make_peg_in_slot_env_cfg() -> ManagerBasedRlEnvCfg:
    scene = SceneCfg(
        num_envs=1, env_spacing=1.0,
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={
            UR5E_ENTITY_NAME: get_ur5e_entity_cfg(),
            PEG_ENTITY_NAME: get_peg_entity_cfg(),
            SLOT_ENTITY_NAME: get_slot_entity_cfg(),
        },
        sensors=(_wrist_rgbd_camera(),),
    )

    state_terms = {
        "joint_pos": ObservationTermCfg(
            func=joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg(UR5E_ENTITY_NAME, joint_names=(".*",))},
        ),
        "joint_vel": ObservationTermCfg(
            func=joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg(UR5E_ENTITY_NAME, joint_names=(".*",))},
        ),
        "ee_to_peg": ObservationTermCfg(
            func=ee_to_object_pos,
            params={"object_name": PEG_ENTITY_NAME,
                    "asset_cfg": SceneEntityCfg(UR5E_ENTITY_NAME, body_names=("_tcp",))},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "peg_to_slot": ObservationTermCfg(
            func=object_to_goal_pos,
            params={"object_name": PEG_ENTITY_NAME, "goal_entity": SLOT_ENTITY_NAME},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "actions": ObservationTermCfg(func=last_action),
    }

    vision_terms = _make_vision_obs_terms(UR5E_ENTITY_NAME)
    actor_terms = {**state_terms, **vision_terms}
    critic_terms = {**state_terms}

    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
    }

    actions: dict[str, ActionTermCfg] = {
        "joint_delta_pos": RelativeJointPositionActionCfg(
            entity_name=UR5E_ENTITY_NAME,
            actuator_names=(".*",),
            scale=0.05,
        ),
    }

    events: dict[str, EventTermCfg] = {
        "reset_base": EventTermCfg(
            func=reset_root_state_uniform, mode="reset",
            params={"pose_range": {}, "velocity_range": {},
                    "asset_cfg": SceneEntityCfg(UR5E_ENTITY_NAME, joint_names=(".*",))},
        ),
        "reset_arm": EventTermCfg(func=reset_arm_to_default, mode="reset"),
        "reset_peg": EventTermCfg(
            func=reset_object_pose_uniform, mode="reset",
            params={"object_name": PEG_ENTITY_NAME, "x_range": (0.35, 0.50),
                    "y_range": (-0.15, 0.15), "z": 0.162, "yaw_range": (-3.14, 3.14)},
        ),
        "reset_slot": EventTermCfg(
            func=reset_object_pose_uniform, mode="reset",
            params={"object_name": SLOT_ENTITY_NAME, "x_range": (0.50, 0.65),
                    "y_range": (-0.15, 0.15), "z": 0.152, "yaw_range": (-3.14, 3.14)},
        ),
    }

    rewards: dict[str, RewardTermCfg] = {
        "reaching": RewardTermCfg(func=reaching_reward, weight=1.0,
                                  params={"object_name": PEG_ENTITY_NAME, "ee_entity": UR5E_ENTITY_NAME, "std": 0.15}),
        "insertion": RewardTermCfg(func=insertion_reward, weight=3.0,
                                   params={"peg_entity": PEG_ENTITY_NAME, "slot_entity": SLOT_ENTITY_NAME,
                                           "pos_std": 0.02, "depth_weight": 3.0}),
        "action_penalty": RewardTermCfg(func=action_rate_l2, weight=-0.01),
    }

    terminations: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
        "peg_dropped": TerminationTermCfg(func=object_dropped,
                                          params={"object_name": PEG_ENTITY_NAME, "z_threshold": 0.05}),
    }

    return ManagerBasedRlEnvCfg(
        scene=scene, observations=observations, actions=actions,
        commands={}, events=events, rewards=rewards, terminations=terminations,
        viewer=_make_viewer(), sim=_make_sim(),
        decimation=25, episode_length_s=20.0,
    )


def peg_in_slot_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_peg_in_slot_env_cfg()
    if play:
        cfg.observations["actor"].enable_corruption = False
    return cfg


def peg_in_slot_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return _make_base_ppo_runner("peg_in_slot")
