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
    """通用策略评估 — 使用 SimulationManager + PolicyController。"""
    from ur_mjlab_bc_rl.simulate.config_loader import load_scene_config
    from ur_mjlab_bc_rl.simulate import SimulationManager, PolicyController

    config = load_scene_config(task)
    pdt = config.sim.physics_dt
    max_time = max_steps * pdt

    # 取第一个相机的 depth_range 作为默认值
    depth_range = config.cameras[0].depth_range if config.cameras else (0.1, 0.8)

    model.to(device)
    model.eval()

    mgr = SimulationManager(config, render=render)
    controller = PolicyController(
        model, model_type,
        device=device,
        deterministic=deterministic,
        chunk_size=chunk_size,
        depth_range=depth_range,
    )

    try:
        lengths: list[int] = []
        for ep_idx in range(num_episodes):
            ep = mgr.run_episode(controller, max_time=max_time)
            steps = len(ep)
            lengths.append(steps)

            if (ep_idx + 1) % max(1, num_episodes // 10) == 0:
                print(f"  Ep {ep_idx+1}/{num_episodes}: steps={steps}")
    finally:
        mgr.close()

    return {
        "mean_reward": 0.0,
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
        from ur_mjlab_bc_rl.models.ALHAH_ACT.backbone import build_detr_vae
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
        from ur_mjlab_bc_rl.models.Test_Multimodal.backbone import UR5MultimodalBackbone
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
