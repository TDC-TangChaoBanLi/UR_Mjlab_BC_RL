#!/usr/bin/env python3
"""合并多个 LeRobot 数据集 — 流式读取+写入，无需 Episode 中转。"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("FFMPEG_LOGLEVEL", "error")
os.environ["HF_HUB_OFFLINE"] = "0"
for _name in ("lerobot", "datasets", "PIL", "torchvision", "ffmpeg", "av"):
    logging.getLogger(_name).setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _stream_merge(sources: list[Path], output_root: Path) -> Path:
    """流式合并 LeRobot 数据集。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from ur_mjlab_bc_rl.simulate.dataset_writer import (
        LeRobotDatasetWriter, LeRobotDatasetConfig,
    )

    # 读取第一个数据集获取元信息
    first = sources[0]
    ds0 = LeRobotDataset(repo_id=first.name, root=str(first.parent))
    meta = ds0.meta

    wcfg = LeRobotDatasetConfig(
        repo_id="ur5_merged",
        root=str(output_root),
        fps=meta.fps,
        state_dim=meta.features["observation.state"]["shape"][0],
        action_dim=meta.features["action"]["shape"][0],
        image_height=meta.features["observation.images.rgb"]["shape"][1],
        image_width=meta.features["observation.images.rgb"]["shape"][2],
    )
    writer = LeRobotDatasetWriter.create_new(wcfg, overwrite=True)
    total = 0

    for i, sd in enumerate(sources, 1):
        repo = sd.name
        print(f"\n[{i}/{len(sources)}] {repo}")
        ds = LeRobotDataset(repo_id=repo, root=str(sd.parent))
        n = ds.num_frames
        prev_ep = None
        pct = max(1, n // 10)

        for fi in range(n):
            if fi % pct == 0:
                print(f"  {fi}/{n}...", end="\r", flush=True)
            f = ds[fi]
            ep_idx = f.get("episode_index", 0)

            if prev_ep is not None and ep_idx != prev_ep:
                writer.save_current_episode()
                total += 1

            rgb = f["observation.images.rgb"].numpy()
            if rgb.shape[0] == 3:
                rgb = rgb.transpose(1, 2, 0)

            depth = f["observation.images.depth"].numpy()
            if depth.ndim == 3 and depth.shape[0] in (1, 3):
                depth = depth.squeeze(0) if depth.shape[0] == 1 else depth[0]

            state_flat = f["observation.state"].numpy()
            obs = {
                "state": {
                    "arm_joint_pos": state_flat[:6].astype(np.float32),
                    "gripper_pos": state_flat[6:7].astype(np.float32),
                    "last_action": np.zeros(7, dtype=np.float32),
                },
                "rgb": rgb.astype(np.uint8),
                "depth": depth.astype(np.float32),
                "task_id": 0,
            }
            writer.add_step(obs, f["action"].numpy().astype(np.float32), task_label="pick_place")
            prev_ep = ep_idx

        writer.save_current_episode()
        total += 1
        gc.collect()
        print(f"    → 累计 {total} episodes")

    return writer.finalize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sources", nargs="+")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    sources = [Path(s).resolve() for s in args.sources]
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    valid = []
    print(f"\n{'='*60}\n合并 {len(sources)} 个数据集")
    for sd in sources:
        if not (sd / "meta" / "info.json").exists():
            print(f"  ⚠ 跳过: {sd}"); continue
        with open(sd / "meta" / "info.json") as f:
            meta = json.load(f)
        valid.append(sd)
        print(f"  {sd.name}: {meta['total_episodes']} eps, {meta['total_frames']} fr")

    result = _stream_merge(valid, out)
    print(f"\n✓ 完成: {result}")


if __name__ == "__main__":
    main()
