"""多任务 UR5 操作 — 共享常量."""

# 实体名称
UR5E_ENTITY_NAME = "UR5e"
CUBE_ENTITY_NAME = "cube"
PLATE_ENTITY_NAME = "plate"
T_SHAPE_ENTITY_NAME = "t_shape"
GOAL_MARKER_ENTITY_NAME = "goal_marker"
PEG_ENTITY_NAME = "peg"
SLOT_ENTITY_NAME = "slot_block"

# Site 名称
EE_SITE_NAME = "_tcp"
PLATE_CENTER_SITE = "plate_center"
T_CENTER_SITE = "t_center"
PEG_GRASP_SITE = "peg_grasp_site"
PEG_TIP_SITE = "peg_tip_site"
SLOT_TOP_SITE = "slot_top"
SLOT_BOTTOM_SITE = "slot_bottom"

# 关节名称
UR5E_ARM_JOINTS = (
    "ur_shoulder_pan_joint",
    "ur_shoulder_lift_joint",
    "ur_elbow_joint",
    "ur_wrist_1_joint",
    "ur_wrist_2_joint",
    "ur_wrist_3_joint",
)
UR5E_GRIPPER_JOINTS = ("robotiq_85_left_knuckle_joint",)

# 任务 ID
TASK_IDS = {"pick_place": 0, "push_t": 1, "peg_in_slot": 2}
