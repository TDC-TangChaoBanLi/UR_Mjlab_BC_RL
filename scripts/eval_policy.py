#!/usr/bin/env python3
"""策略评估脚本 — 在 MuJoCo 环境中评估 BC / ACT / PPO 策略。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def evaluate(
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
    """通用策略评估。

    仿真循环：physics(1000Hz) → camera(≈30Hz) → policy(100Hz)。
    """
    from ur_mjlab_bc_rl.simulate.env import (
        MujocoInterface, CameraSensor, ObservationCollector,
        ResetManager, convert_obs_to_model_input, flatten_state,
    )
    from ur_mjlab_bc_rl.simulate.config_loader import (
        get_sim_params, get_arm_joints, get_gripper_joints,
        get_camera_name, get_image_size, load_tasks,
    )

    if model_type == "act":
        from ur_mjlab_bc_rl.models.policy.aloha_act_backbone import EnsembleBuffer

    SIM = get_sim_params()
    ARM = get_arm_joints()
    GRP = get_gripper_joints()
    CAM = get_camera_name()
    ISZ = get_image_size()

    tasks = load_tasks()
    cfg = tasks.get(task)
    if cfg is None:
        raise ValueError(f"Unknown task: {task}")

    sp = PROJECT_ROOT / "assets" / "mujoco" / "scenes" / cfg["scene"]
    dr = cfg["depth_range"]
    tid = cfg["task_id"]

    model.to(device)
    model.eval()

    state_keys = None if model_type == "bc" else ["arm_joint_pos", "gripper_pos"]
    action_dim = 7

    mj = MujocoInterface(str(sp), render=render)
    if render:
        mj.set_viewer_camera((0.45, 0.0, 0.65), 1.8, -25.0, 130.0)

    cam = CameraSensor(mj, CAM, ISZ)
    col = ObservationCollector(mj, cam, ARM, GRP)
    rm = ResetManager(mj, ARM, GRP)

    a_ids = [mj.get_actuator_id(n + "_ACTUATOR") for n in ARM]
    g_ids = [mj.get_actuator_id(n + "_ACTUATOR") for n in GRP]

    rewards: list[float] = []
    lengths: list[int] = []
    pdt = SIM["physics_dt"]
    cdt = SIM["camera_dt"]
    adt = SIM["policy_dt"]

    for ep_idx in range(num_episodes):
        rm.reset(task=task, randomize_objects=True)
        col.reset()
        tp = 0.0
        tc = 0.0
        cam.capture()

        ebuf = None
        if model_type == "act":
            ebuf = EnsembleBuffer(chunk_size=chunk_size, action_dim=action_dim).to(device)

        total_reward = 0.0
        steps = 0

        for _ in range(max_steps):
            mj.step()
            tp += pdt
            tc += pdt

            if render and not mj.is_viewer_running():
                print("\n  窗口关闭")
                break

            if tc >= cdt:
                cam.capture()
                tc -= cdt

            if tp >= adt:
                tp -= adt
                obs = col.collect(task_id=tid)

                with torch.no_grad():
                    if model_type == "bc":
                        ct, st, tt = convert_obs_to_model_input(
                            obs, device, state_keys=state_keys,
                            depth_min=dr[0], depth_max=dr[1],
                        )
                        action = model(
                            {"camera": ct, "actor_state": st, "task": tt},
                            deterministic=deterministic,
                        )
                        anp = action.cpu().numpy().squeeze(0)

                    elif model_type == "act":
                        rgb = obs["rgb"].astype(np.float32).transpose(2, 0, 1) / 255.0
                        depth = obs["depth"].astype(np.float32)
                        if depth.ndim == 2:
                            depth = depth[None, :, :]
                        cn = np.concatenate([rgb, depth], axis=0)
                        ct = torch.from_numpy(cn).unsqueeze(0).to(device)
                        sn = flatten_state(obs["state"], state_keys=state_keys)
                        st = torch.from_numpy(sn).unsqueeze(0).to(device)
                        chunk = model.get_action(st, ct)
                        ebuf.add(chunk[0])
                        anp = ebuf.get_action().cpu().numpy()

                anp = np.clip(anp, -6.28, 6.28)
                ctrl = mj.get_ctrl()
                for i, aid in enumerate(a_ids):
                    ctrl[aid] = anp[i] if i < 6 else 0.0
                for i, aid in enumerate(g_ids):
                    ctrl[aid] = anp[6 + i] if 6 + i < len(anp) else 0.0
                mj.set_ctrl(ctrl)

                if model_type == "bc":
                    col.update_last_action(anp)
                steps += 1

        lengths.append(steps)
        rewards.append(total_reward)

        if (ep_idx + 1) % max(1, num_episodes // 10) == 0:
            print(f"  Ep {ep_idx+1}/{num_episodes}: steps={steps}")

    col.close()
    mj.close()

    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_length": float(np.mean(lengths)),
        "num_episodes": num_episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="pick_place")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-deterministic", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--max-steps", type=int, default=30000)
    args = parser.parse_args()

    cp = Path(args.checkpoint)
    if not cp.exists():
        print(f"✗ Checkpoint 不存在: {cp}"); sys.exit(1)

    ckpt = torch.load(cp, map_location="cpu", weights_only=False)
    is_act = ckpt.get("model_type") == "act" or "model_state_dict" in ckpt
    is_bc = "actor_state_dict" in ckpt

    print(f"\n{'='*60}\n策略评估\n{'='*60}")
    print(f"  任务: {args.task} | Episodes: {args.episodes} | 设备: {args.device}")

    if is_act:
        model_type = "act"
        print("  类型: ACT (DETRVAE)")
        from ur_mjlab_bc_rl.config_loader import load_aloha_act_model
        mcfg = ckpt.get("model_cfg") or load_aloha_act_model()
        from ur_mjlab_bc_rl.models.policy.aloha_act_backbone import build_detr_vae
        model = build_detr_vae(mcfg)
        sk = "model_state_dict" if "model_state_dict" in ckpt else "actor_state_dict"
        model.load_state_dict(ckpt[sk])
        cs = mcfg.get("chunk_size", 10)
        print(f"  Chunk: {cs} | Params: {sum(p.numel() for p in model.parameters()):,}")
        results = evaluate(model, model_type, args.task, args.episodes, args.device,
                           not args.no_render, args.max_steps, chunk_size=cs)

    elif is_bc:
        model_type = "bc"
        print("  类型: BC (UR5MultimodalBackbone)")
        from ur_mjlab_bc_rl.config_loader import load_multimodal_model
        mcfg = ckpt.get("model_cfg") or load_multimodal_model()
        from ur_mjlab_bc_rl.models.policy.multimodal_backbone import UR5MultimodalBackbone
        model = UR5MultimodalBackbone(model_cfg=mcfg)
        model.load_state_dict(ckpt["actor_state_dict"])
        deterministic = not args.no_deterministic
        print(f"  确定性: {deterministic}")
        results = evaluate(model, model_type, args.task, args.episodes, args.device,
                           not args.no_render, args.max_steps, deterministic=deterministic)

    else:
        print("✗ 无法识别 checkpoint 类型"); sys.exit(1)

    print(f"\n结果: mean_length={results['mean_length']:.1f}")


if __name__ == "__main__":
    main()
