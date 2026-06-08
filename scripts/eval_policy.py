#!/usr/bin/env python3
"""策略评估脚本。

在 MjLab 环境中评估训练好的策略（BC 或 PPO）。

使用方式:
  # 评估 BC 策略
  python scripts/eval_policy.py --task pick_place \\
      --checkpoint outputs/checkpoints/bc_best.pt --episodes 20

  # 评估 PPO 策略  
  python scripts/eval_policy.py --task UR5-PickPlace \\
      --checkpoint logs/rsl_rl/pick_place/model_0.pt --episodes 20 \\
      --video

  # 比较 BC vs PPO
  python scripts/eval_policy.py --task pick_place \\
      --bc-checkpoint outputs/checkpoints/bc_best.pt \\
      --ppo-checkpoint logs/rsl_rl/pick_place/model_0.pt \\
      --episodes 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def evaluate_bc_policy(
    checkpoint_path: Path,
    task: str,
    num_episodes: int = 10,
    deterministic: bool = True,
    device: str = "cpu",
    render: bool = True,
    max_steps: int = 3000,
) -> dict:
    """使用 BC 训练的 UR5MultimodalActor 进行评估。

    在纯 MuJoCo 环境中运行，多频率仿真循环：
      physics: 1000 Hz  mj_step
      policy:  100 Hz  推理+执行
      camera:  30 Hz   渲染 RGBD（观测用）

    Args:
        checkpoint_path: BC checkpoint 路径
        task: 任务名称（pick_place, push_t, peg_slot）
        num_episodes: 评估 episode 数
        deterministic: 是否使用确定性策略
        device: 计算设备
        render: 是否显示 viewer
        max_steps: 每 episode 最大 physics 步数

    Returns:
        评估统计信息字典
    """
    from ur_mjlab_bc_rl.models.policy.multimodal_backbone import UR5MultimodalBackbone
    from ur_mjlab_bc_rl.imitation.mujoco_env import (
        MujocoInterface, CameraSensor, ObservationCollector, ResetManager,
    )
    from ur_mjlab_bc_rl.imitation.mujoco_env.observation import convert_obs_to_model_input
    from ur_mjlab_bc_rl.config_loader import (
        get_sim_params, get_arm_joints, get_gripper_joints, get_camera_name, get_image_size, load_tasks,
    )

    SIM = get_sim_params()
    ARM_JOINTS = get_arm_joints()
    GRIPPER_JOINTS = get_gripper_joints()
    CAMERA_NAME = get_camera_name()
    IMAGE_SIZE = get_image_size()


    TASK_CONFIG = {
        name: {"scene": cfg["scene"], "task_id": cfg["task_id"]}
        for name, cfg in load_tasks().items()
    }

    config = TASK_CONFIG.get(task)
    if config is None:
        raise ValueError(f"Unknown task: {task}")

    scene_path = PROJECT_ROOT / "assets" / "mujoco" / "scenes" / config["scene"]

    # ── 加载 checkpoint ──
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 优先使用 checkpoint 中保存的配置，否则从配置文件加载
    # 这确保评估时使用与训练相同的模型架构
    model_cfg = ckpt.get("model_cfg")
    if model_cfg is None:
        print(f"  ⚠ Checkpoint 中未找到 model_cfg，从配置文件加载...")
        from ur_mjlab_bc_rl.config_loader import load_multimodal_model
        model_cfg = load_multimodal_model()
    else:
        print(f"  ✓ 从 checkpoint 加载模型配置")
    
    actor = UR5MultimodalBackbone(model_cfg=model_cfg)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.to(device)
    actor.eval()

    # ── 仿真接口 ──
    mj = MujocoInterface(str(scene_path), render=render)
    if render:
        mj.set_viewer_camera(lookat=(0.45, 0.0, 0.65), distance=1.8, elevation=-25., azimuth=130.)

    depth_range = load_tasks()[task]["depth_range"]
    
    camera = CameraSensor(
        mj, 
        CAMERA_NAME, 
        IMAGE_SIZE
        )
    collector = ObservationCollector(
        mj, 
        camera, 
        ARM_JOINTS,  
        GRIPPER_JOINTS,
        depth_range[0], 
        depth_range[1]
        )
    reset_mgr = ResetManager(mj)

    arm_act_ids = [mj.get_actuator_id(n + "_ACTUATOR") for n in ARM_JOINTS]
    gripper_act_ids = [mj.get_actuator_id(n + "_ACTUATOR") for n in GRIPPER_JOINTS]

    # ── 评估循环 ──
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    success_count = 0

    for ep_idx in range(num_episodes):
        reset_mgr.reset(task=task, randomize_objects=True)
        collector.reset()

        # 频率计时器
        time_since_policy = 0.0
        time_since_camera = 0.0

        # 预捕获一帧
        camera.capture()

        total_reward = 0.0
        policy_step_count = 0

        for _ in range(max_steps):
            # ═══════════════════════════════
            # Physics (1000 Hz)
            # ═══════════════════════════════
            mj.step()
            time_since_policy += SIM["physics_dt"]
            time_since_camera += SIM["physics_dt"]

            if render and not mj.is_viewer_running():
                print("\n  可视化窗口已关闭，停止评估")
                break

            # ═══════════════════════════════
            # Camera (30 Hz)
            # ═══════════════════════════════
            if time_since_camera >= SIM["camera_dt"]:
                camera.capture()
                time_since_camera -= SIM["camera_dt"]

            # ═══════════════════════════════
            # Policy (100 Hz) — 推理+执行
            # ═══════════════════════════════
            if time_since_policy >= SIM["policy_dt"]:
                time_since_policy -= SIM["policy_dt"]

                obs = collector.collect(task_id=config["task_id"])

                # 转换观测为模型输入
                camera_t, state, task_t = convert_obs_to_model_input(obs, device)

                with torch.no_grad():
                    action = actor(
                        {"camera": camera_t, "actor_state": state, "task": task_t},
                        deterministic=deterministic,
                    )
                     
                action_np = action.cpu().numpy().squeeze(0)  # [7] (6 臂 + 1 夹爪)
                action_np.clip(-6.28, 6.28, out=action_np)
                arm_actions = action_np[:6]
                gripper_actions = action_np[6:]

                # 执行
                ctrl = mj.get_ctrl()
                for i, act_id in enumerate(arm_act_ids):
                    ctrl[act_id] = arm_actions[i]
                # 夹爪控制
                for i, act_id in enumerate(gripper_act_ids):
                    ctrl[act_id] = gripper_actions[i]
                mj.set_ctrl(ctrl)


                collector.update_last_action(action_np)

                policy_step_count += 1

        episode_lengths.append(policy_step_count)
        episode_rewards.append(total_reward)

        if (ep_idx + 1) % max(1, num_episodes // 10) == 0:
            print(f"  Episode {ep_idx + 1}/{num_episodes}: "
                  f"reward={total_reward:.2f}, steps={policy_step_count}")

    collector.close()
    mj.close()

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "success_rate": success_count / num_episodes,
        "num_episodes": num_episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="策略评估")
    parser.add_argument("--task", type=str, default="pick_place",
                        help="任务名称")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="策略 checkpoint 路径")
    parser.add_argument("--episodes", type=int, default=10,
                        help="评估 episode 数")
    parser.add_argument("--no-deterministic", action="store_true",
                        help="使用随机策略（非确定性）")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="计算设备")
    parser.add_argument("--no-render", action="store_true",
                        help="禁用可视化渲染")
    parser.add_argument("--max-steps", type=int, default=3000,
                        help="每 episode 最大 physics 步数")
    parser.add_argument("--output", type=str, default=None,
                        help="评估结果输出路径 (JSON)")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint 不存在: {checkpoint_path}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"策略评估")
    print(f"{'=' * 60}")
    print(f"  任务: {args.task}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Episodes: {args.episodes}")
    print(f"  确定性: {not args.no_deterministic}")
    print(f"  设备: {args.device}")
    print(f"  可视化: {not args.no_render}")

    # 判断 checkpoint 类型
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "actor_state_dict" in ckpt:
        # BC checkpoint
        print(f"\n  类型: BC (UR5MultimodalBackbone)")
        results = evaluate_bc_policy(
            checkpoint_path=checkpoint_path,
            task=args.task,
            num_episodes=args.episodes,
            deterministic=not args.no_deterministic,
            device=args.device,
            render=not args.no_render,
            max_steps=args.max_steps,
        )
    else:
        print(f"\n  类型: PPO (RSL-RL)")
        print(f"  PPO checkpoint 评估请使用 mjlab play 命令：")
        print(f"    mjlab play UR5-{args.task.replace('_', '-').title()} "
              f"--checkpoint-file {checkpoint_path} --num-envs 1")
        sys.exit(0)

    # 输出结果
    print(f"\n{'=' * 60}")
    print(f"评估结果")
    print(f"{'=' * 60}")
    print(f"  平均奖励: {results['mean_reward']:.4f}")
    print(f"  平均步数: {results['mean_length']:.1f}")
    print(f"  成功率: {results['success_rate']:.1%}")
    print(f"{'=' * 60}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ 结果保存到: {output_path}")


if __name__ == "__main__":
    main()