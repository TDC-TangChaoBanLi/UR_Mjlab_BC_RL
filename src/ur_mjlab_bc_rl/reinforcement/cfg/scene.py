"""实体配置 — UR5e 机器人 + 操作物体.

使用 mjlab EntityCfg + MjSpec 模式创建实体。
"""

from pathlib import Path

import mujoco
import numpy as np

from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator.xml_actuator import XmlActuatorCfg

from .constants import (
    UR5E_ENTITY_NAME, UR5E_ARM_JOINTS, UR5E_GRIPPER_JOINTS,
    CUBE_ENTITY_NAME, PLATE_ENTITY_NAME,
    T_SHAPE_ENTITY_NAME, GOAL_MARKER_ENTITY_NAME,
    PEG_ENTITY_NAME, SLOT_ENTITY_NAME,
)

# MJCF 路径（相对于项目根目录）
_UR5E_MJCF: Path = Path("assets/mujoco/ur5/ur5e_full.xml")


# =========================================================================
# UR5e 机器人实体
# =========================================================================

_UR5E_ALL_ACTUATOR_JOINTS = UR5E_ARM_JOINTS  # 仅 arm 6 关节，gripper 单独处理

_UR5E_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    rot=(0.0, 0.0, 0.0, 1.0),
    joint_pos={
        UR5E_ARM_JOINTS[0]: 0.0,
        UR5E_ARM_JOINTS[1]: -1.57,
        UR5E_ARM_JOINTS[2]: 1.57,
        UR5E_ARM_JOINTS[3]: -1.57,
        UR5E_ARM_JOINTS[4]: -1.57,
        UR5E_ARM_JOINTS[5]: 0.0,
    },
    joint_vel={".*": 0.0},
)


def _get_ur5e_spec() -> mujoco.MjSpec:
    """从 MJCF 文件加载 UR5e 完整模型。"""
    return mujoco.MjSpec.from_file(str(_UR5E_MJCF))


def get_ur5e_entity_cfg() -> EntityCfg:
    """创建 UR5e 机械臂实体配置。"""
    return EntityCfg(
        spec_fn=_get_ur5e_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=_UR5E_ALL_ACTUATOR_JOINTS),)
        ),
        init_state=_UR5E_INIT_STATE,
    )


# =========================================================================
# 物体实体（使用 MjSpec 程序化创建）
# =========================================================================

def _make_box_spec(
    name: str, size: tuple[float, float, float], mass: float,
    rgba: tuple[float, float, float, float],
) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=name)
    body.add_freejoint()
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=size,
        mass=mass,
        rgba=rgba,
    )
    return spec


def _make_cylinder_spec(
    name: str, radius: float, height: float, mass: float,
    rgba: tuple[float, float, float, float],
    add_site: str | None = None,
) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=name)
    body.add_freejoint()
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(radius, height),
        mass=mass,
        rgba=rgba,
    )
    if add_site:
        body.add_site(name=add_site, pos=(0, 0, height))
    return spec


def _make_t_shape_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=T_SHAPE_ENTITY_NAME)
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.01, 0.04, 0.01),
                  mass=0.04, rgba=(0.2, 0.75, 0.3, 1), pos=(0, 0, 0))
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.035, 0.01, 0.01),
                  mass=0.04, rgba=(0.2, 0.75, 0.3, 1), pos=(0, 0.04, 0))
    body.add_site(name="t_center", pos=(0, 0, 0))
    return spec


def _make_peg_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=PEG_ENTITY_NAME)
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=(0.015, 0.03),
                  mass=0.04, rgba=(0.9, 0.5, 0.1, 1), pos=(0, 0, 0.015))
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=(0.008, 0.01),
                  mass=0.02, rgba=(0.95, 0.6, 0.2, 1), pos=(0, 0, 0.055))
    body.add_site(name="peg_grasp_site", pos=(0, 0, 0.04))
    body.add_site(name="peg_tip_site", pos=(0, 0, 0.065))
    return spec


def _make_slot_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=SLOT_ENTITY_NAME)
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.03, 0.03, 0.02),
                  mass=0.06, rgba=(0.5, 0.5, 0.55, 1), pos=(0, 0, 0))
    # 凹槽壁
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.012, 0.006, 0.015),
                  mass=0.01, rgba=(0.5, 0.5, 0.55, 1), pos=(0, 0.024, 0.005))
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.012, 0.006, 0.015),
                  mass=0.01, rgba=(0.5, 0.5, 0.55, 1), pos=(0, -0.024, 0.005))
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.006, 0.018, 0.015),
                  mass=0.01, rgba=(0.5, 0.5, 0.55, 1), pos=(-0.024, 0, 0.005))
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.006, 0.018, 0.015),
                  mass=0.01, rgba=(0.5, 0.5, 0.55, 1), pos=(0.024, 0, 0.005))
    body.add_site(name="slot_bottom", pos=(0, 0, -0.005))
    body.add_site(name="slot_top", pos=(0, 0, 0.025))
    return spec


# ---- Entity config getters ----

def get_cube_entity_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=lambda: _make_box_spec(
            CUBE_ENTITY_NAME, (0.025, 0.025, 0.025), 0.05, (0.85, 0.2, 0.2, 1)
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.45, 0.0, 0.165), rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={}, joint_vel={".*": 0.0},
        ),
    )


def get_plate_entity_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=lambda: _make_cylinder_spec(
            PLATE_ENTITY_NAME, 0.08, 0.005, 0.1, (0.2, 0.4, 0.8, 1),
            add_site="plate_center",
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.135), rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={}, joint_vel={".*": 0.0},
        ),
    )


def get_t_shape_entity_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_make_t_shape_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.40, -0.05, 0.152), rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={}, joint_vel={".*": 0.0},
        ),
    )


def get_goal_marker_entity_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=lambda: _make_cylinder_spec(
            GOAL_MARKER_ENTITY_NAME, 0.04, 0.005, 0.01, (0.1, 0.8, 0.2, 0.3),
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.50, 0.05, 0.147), rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={}, joint_vel={".*": 0.0},
        ),
    )


def get_peg_entity_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_make_peg_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.38, 0.0, 0.162), rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={}, joint_vel={".*": 0.0},
        ),
    )


def get_slot_entity_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_make_slot_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.52, 0.0, 0.152), rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={}, joint_vel={".*": 0.0},
        ),
    )
