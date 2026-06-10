#!/usr/bin/env python3
"""策略评估脚本。

在 MjLab 环境中评估训练好的策略（BC / ACT / PPO）。

使用方式:
  # 评估 BC 策略
  python scripts/eval_policy.py --task pick_place \\
      --checkpoint outputs/checkpoints/bc_best.pt --episodes 20

  # 评估 ACT 策略
  python scripts/eval_policy.py --task pick_place \\
      --checkpoint outputs/checkpoints/pick_place/xxx/best_actor.pt --episodes 20

  # 评估 PPO 策略  
  python scripts/eval_policy.py --task UR5-PickPlace \\
      --checkpoint logs/rsl_rl/pick_place/model_0.pt --episodes 20 \\
      --video
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def evaluate_with_model(
    model: torch.nn.Module,
    model_type: str,
    task: str,
    num_episodes: int = 10,
    device: str = "cpu",
    render: bool = True,
    max_steps: int = 30000,
    *,
    deterministic: bool = True,
    chunk_size: int = 10,
) -> dict:
    """通用策略评估函数。

    在 MuJoCo 多频率仿真循环中评估一个已加载的策略模型：
      physics: 1000 Hz  mj_step
      policy:  100 Hz  推理+执行
      camera:  30 Hz   渲染 RGBD（观测用）

    Args:
        model:        已加载并置于 device 的策略网络
        model_type:   模型类型 ("bc" | "act")
        task:         任务名称（pick_place, push_t, peg_slot）
        num_episodes: 评估 episode 数
        device:       计算设备
        render:       是否显示 viewer
        max_steps:    每 episode 最大 physics 步数
        deterministic: BC 是否确定性推理
        chunk_size:   ACT 动作分块大小 K

    Returns:
        评估统计信息字典
    """
    from ur_mjlab_bc_rl.imitation.mujoco_env import (
        MujocoInterface, CameraSensor, ObservationCollector, ResetManager,
    )
    from ur_mjlab_bc_rl.imitation.mujoco_env.observation import (
        convert_obs_to_model_input, flatten_state_from_obs,
    )
    from ur_mjlab_bc_rl.config_loader import (
        get_sim_params, get_arm_joints, get_gripper_joints,
        get_camera_name, get_image_size, load_tasks,
    )

    if model_type == "act":
        from ur_mjlab_bc_rl.models.policy.aloha_act_backbone import EnsembleBuffer

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
    depth_range = load_tasks()[task]["depth_range"]

    # ── 模型推理准备 ──
    model.to(device)
    model.eval()

    if model_type == "bc":
        state_keys = None  # 默认全部：arm_joint_pos + gripper_pos + last_action = 14
    elif model_type == "act":
        state_keys = ["arm_joint_pos", "gripper_pos"]  # 7 dims
        action_dim = 7

    # ── 仿真接口 ──
    mj = MujocoInterface(str(scene_path), render=render)
    if render:
        mj.set_viewer_camera(lookat=(0.45, 0.0, 0.65), distance=1.8, elevation=-25., azimuth=130.)

    camera = CameraSensor(mj, CAMERA_NAME, IMAGE_SIZE)
    collector = ObservationCollector(
        mj, camera, ARM_JOINTS, GRIPPER_JOINTS,
        depth_range[0], depth_range[1]
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

        time_since_policy = 0.0
        time_since_camera = 0.0
        camera.capture()

        ensemble_buf = None
        if model_type == "act":
            ensemble_buf = EnsembleBuffer(chunk_size=chunk_size, action_dim=action_dim).to(device)

        total_reward = 0.0
        policy_step_count = 0

        for _ in range(max_steps):
            # ── Physics (1000 Hz) ──
            mj.step()
            time_since_policy += SIM["physics_dt"]
            time_since_camera += SIM["physics_dt"]

            if render and not mj.is_viewer_running():
                print("\n  可视化窗口已关闭，停止评估")
                break

            # ── Camera (30 Hz) ──
            if time_since_camera >= SIM["camera_dt"]:
                camera.capture()
                time_since_camera -= SIM["camera_dt"]

            # ── Policy (100 Hz) ──
            if time_since_policy >= SIM["policy_dt"]:
                time_since_policy -= SIM["policy_dt"]
                obs = collector.collect(task_id=config["task_id"])

                with torch.no_grad():
                    if model_type == "bc":
                        camera_t, state_t, task_t = convert_obs_to_model_input(
                            obs, device, state_keys=state_keys,
                        )
                        action = model(
                            {"camera": camera_t, "actor_state": state_t, "task": task_t},
                            deterministic=deterministic,
                        )
                        action_np = action.cpu().numpy().squeeze(0)

                    elif model_type == "act":
                        # 图像 [1, 4, H, W]
                        rgb = obs["rgb"].astype(np.float32).transpose(2, 0, 1) / 255.0
                        depth = obs["depth"].astype(np.float32)
                        if depth.ndim == 2:
                            depth = depth[None, :, :]
                        camera_np = np.concatenate([rgb, depth], axis=0)
                        camera_t = torch.from_numpy(camera_np).unsqueeze(0).to(device)

                        # 状态 [1, 7]
                        state_np = flatten_state_from_obs(obs, state_keys=state_keys)
                        qpos_t = torch.from_numpy(state_np).unsqueeze(0).to(device)

                        chunk = model.get_action(qpos_t, camera_t)  # [1, K, 7]
                        ensemble_buf.add(chunk[0])
                        action_np = ensemble_buf.get_action().cpu().numpy()

                action_np = np.clip(action_np, -6.28, 6.28)
                arm_actions = action_np[:6]
                gripper_actions = action_np[6:]

                # 执行
                ctrl = mj.get_ctrl()
                for i, act_id in enumerate(arm_act_ids):
                    ctrl[act_id] = arm_actions[i]
                for i, act_id in enumerate(gripper_act_ids):
                    ctrl[act_id] = gripper_actions[i]
                mj.set_ctrl(ctrl)

                if model_type == "bc":
                    collector.update_last_action(action_np)

                policy_step_count += 1

        episode_lengths.append(policy_step_count)
        episode_rewards.append(total_reward)

        if (ep_idx + 1) % max(1, num_episodes // 10) == 0:
            suffix = f"reward={total_reward:.2f}, " if model_type == "bc" else ""
            print(f"  Episode {ep_idx + 1}/{num_episodes}: {suffix}steps={policy_step_count}")

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
    parser.add_argument("--episodes", type=int, default=5,
                        help="评估 episode 数")
    parser.add_argument("--no-deterministic", action="store_true",
                        help="使用随机策略（非确定性，仅 BC）")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="计算设备")
    parser.add_argument("--no-render", action="store_true",
                        help="禁用可视化渲染")
    parser.add_argument("--max-steps", type=int, default=30000,
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

    # ── 加载 checkpoint & 构建模型 ──
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # 检测 checkpoint 类型
    is_act = ckpt.get("model_type") == "act" or "model_state_dict" in ckpt
    is_bc = "actor_state_dict" in ckpt

    if is_act:
        model_type = "act"
        print(f"\n  类型: ACT (DETRVAE)")

        model_cfg = ckpt.get("model_cfg")
        if model_cfg is None:
            from ur_mjlab_bc_rl.config_loader import load_aloha_act_model
            model_cfg = load_aloha_act_model()

        from ur_mjlab_bc_rl.models.policy.aloha_act_backbone import build_detr_vae
        model = build_detr_vae(model_cfg)
        state_key = "model_state_dict" if "model_state_dict" in ckpt else "actor_state_dict"
        model.load_state_dict(ckpt[state_key])

        print(f"  Chunk size: {model_cfg.get('chunk_size', 10)}")
        print(f"  可训练参数: {sum(p.numel() for p in model.parameters()):,}")

        results = evaluate_with_model(
            model=model, model_type=model_type,
            task=args.task, num_episodes=args.episodes,
            device=args.device, render=not args.no_render,
            max_steps=args.max_steps,
            chunk_size=model_cfg.get("chunk_size", 10),
        )

    elif is_bc:
        model_type = "bc"
        print(f"\n  类型: BC (UR5MultimodalBackbone)")

        model_cfg = ckpt.get("model_cfg")
        if model_cfg is None:
            from ur_mjlab_bc_rl.config_loader import load_multimodal_model
            model_cfg = load_multimodal_model()

        from ur_mjlab_bc_rl.models.policy.multimodal_backbone import UR5MultimodalBackbone
        model = UR5MultimodalBackbone(model_cfg=model_cfg)
        model.load_state_dict(ckpt["actor_state_dict"])

        print(f"  可训练参数: {sum(p.numel() for p in model.parameters()):,}")

        results = evaluate_with_model(
            model=model, model_type=model_type,
            task=args.task, num_episodes=args.episodes,
            device=args.device, render=not args.no_render,
            max_steps=args.max_steps,
            deterministic=not args.no_deterministic,
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