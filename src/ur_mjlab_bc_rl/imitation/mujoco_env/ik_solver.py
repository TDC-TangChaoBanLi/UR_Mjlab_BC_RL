"""动作执行器。

将高级动作（绝对目标位姿）转换为 MuJoCo 控制命令。
使用 mink 库求解逆运动学 (IK)，参考 UR_Ctrl_Demo 的 UR5MinkIK 设计。

动作格式：
  [x, y, z, qw, qx, qy, qz, gripper_cmd]
  - x,y,z:     目标位置（世界坐标系）
  - qw,qx,qy,qz: 目标四元数
  - gripper_cmd: [-1, 1]，正=打开

分层架构：
  规划层 (teacher) → IK 层 (mink, 带 VelocityLimit) → 控制层 (设 ctrl) → 仿真层 (mj_step)
"""

from __future__ import annotations

import numpy as np
import mujoco
import mink


class MinkIK:
    """末端执行器增量动作执行器（基于 mink IK）。

    参考 UR_Ctrl_Demo 的 UR5MinkIK：
    - Configuration 创建一次，通过 sync_from_mujoco_state 同步状态
    - PostureTask 做关节正则化（而非 DofFreezingTask 硬约束）
    - ConfigurationLimit + VelocityLimit 约束关节
    - integrate_inplace 就地积分

    两种模式：
    - "ik":   mink IK + 混合 qpos/ctrl 同步（推荐）
    - "position_delta": 雅可比伪逆（回退）
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        init_qpos: np.ndarray,
        ee_site_name: str = "_tcp",
        solver: str = "daqp",
        pos_cost: float = 1.0,
        ori_cost: float = 1.0,
        posture_cost: float = 1e-3,
        lm_damping: float = 1e-6,
        arm_joint_names: list[str] | None = None,
        # gripper_joint_names: list[str] | None = None,
    ) -> None:
        
        self.configuration = mink.Configuration(model)
        self.model = self.configuration.model
        self.data = self.configuration.data

        self.solver = solver
        self.ee_site_name = ee_site_name

        if arm_joint_names is None:
            arm_joint_names = [
                "ur_shoulder_pan_joint", "ur_shoulder_lift_joint",
                "ur_elbow_joint", "ur_wrist_1_joint",
                "ur_wrist_2_joint", "ur_wrist_3_joint",
            ]
        self.arm_joint_names = arm_joint_names
        self.arm_joint_ids = [model.joint(name).id for name in arm_joint_names]
        self.arm_joint_qpos_adr = [model.jnt_qposadr[jid] for jid in self.arm_joint_ids]

        # if gripper_joint_names is None:
        #     gripper_joint_names = ["robotiq_85_left_knuckle_joint"]
        # self.gripper_joint_names = gripper_joint_names
        # self.gripper_joint_ids = [model.joint(name).id for name in gripper_joint_names]

        # ── mink 任务──
        self.ee_task = mink.FrameTask(
            frame_name=ee_site_name, 
            frame_type="site",
            position_cost=pos_cost, 
            orientation_cost=ori_cost,
            lm_damping=lm_damping,
        )
        self.posture_task = mink.PostureTask(model, cost=posture_cost)
        # self.posture_task.set_target(self._config.q)
        self.tasks = [self.ee_task, self.posture_task]

        # ── mink 限制 ──
        self.limits = []
        self.limits.append(mink.ConfigurationLimit(model=model))
        self.limits.append(
            mink.VelocityLimit(
                model=model,
                velocities={arm_joint_name: 0.5 for arm_joint_name in arm_joint_names},
            ),
        )
        # ── 避障约束 ──
        self._build_collision_avoidance(model)

        self.configuration.update(init_qpos)
        self.posture_task.set_target(self.configuration.q)

        

    def _build_collision_avoidance(self, model: mujoco.MjModel) -> None:
        """构建机械臂自身碰撞 + 臂-桌面避障约束。

        使用 geom name（COLLISION_*, VISUAL_*）构建碰撞对。
        """
        arm_collision_names: list[str] = []
        env_names: list[str] = []

        for i in range(model.ngeom):
            body_id = model.geom_bodyid[i]
            if body_id < 0:
                continue
            body_name = model.body(body_id).name
            gname = model.geom(i).name or ""

            # 机械臂碰撞 geom
            if body_name.startswith(("ur_", "realsense_")):
                if gname.startswith("COLLISION_"):
                    arm_collision_names.append(gname)
            # 桌面
            elif body_name == "table_top":
                if gname.startswith("COLLISION_"):
                    env_names.append(gname)

        if arm_collision_names:
            # 自身碰撞
            self.limits.append(
                mink.CollisionAvoidanceLimit(
                    model=model,
                    geom_pairs=[(arm_collision_names, arm_collision_names)],
                    gain=0.95,
                    minimum_distance_from_collisions=0.002,
                    collision_detection_distance=0.03,
                )
            )
        if arm_collision_names and env_names:
            # 臂-桌面碰撞
            self.limits.append(
                mink.CollisionAvoidanceLimit(
                    model=model,
                    geom_pairs=[(arm_collision_names, env_names)],
                    gain=0.95,
                    minimum_distance_from_collisions=0.05,
                    collision_detection_distance=0.3,
                )
            )

    def _sync_config_from_data(self, qpos: np.ndarray) -> None:
        # 只更新机械臂关节的状态，而不是整个配置空间
        for i, adr in enumerate(self.arm_joint_qpos_adr):
            self.configuration.q[adr] = qpos[i]
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)


    # ── 公共接口 ──────────────────────────────────────────

    def reset(self, init_qpos: np.ndarray) -> None:
        """重置 IK 求解器。"""
        self.configuration.update(init_qpos)
        self.posture_task.set_target(self.configuration.q)

    def solve_ik(self, current_qpos: np.ndarray, target_pose: np.ndarray, dt: float = 0.001) -> np.ndarray:
        """执行动作（绝对目标位姿）。

        Args:
            current_qpos: [6] (arm joint angles)
                - 当前臂关节角
            target_pose: [7] (x, y, z, qw, qx, qy, qz)
                - 目标位置 [3] + 目标四元数 [4]
        """
        self._sync_config_from_data(current_qpos)

        target_pos = target_pose[:3]
        target_quat = target_pose[3:7]

        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, target_quat)

        T_target = np.eye(4)
        T_target[:3, :3] = rot_mat.reshape(3, 3)
        T_target[:3, 3] = target_pos


        self.ee_task.set_target(mink.SE3.from_matrix(T_target))

        vel = mink.solve_ik(
            self.configuration, 
            self.tasks, 
            dt,
            solver=self.solver, 
            limits=self.limits,
        )
        self.configuration.integrate_inplace(vel, dt)

        q_ref = np.array([self.configuration.q[adr] for adr in self.arm_joint_qpos_adr])

        return q_ref



