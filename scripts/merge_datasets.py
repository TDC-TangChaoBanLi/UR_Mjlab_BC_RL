#!/usr/bin/env python3
"""合并多个 LeRobot 数据集。

用法:
  python scripts/merge_datasets.py \\
      outputs/datasets/expert/pick_place/20260605_170954 \\
      outputs/datasets/expert/pick_place/20260605_172609 \\
      ... \\
      --output outputs/datasets/expert/pick_place/merged
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("FFMPEG_LOGLEVEL", "error")
os.environ.setdefault("SVT_LOG", "0")
os.environ["HF_HUB_OFFLINE"] = "0"  # 强制覆盖 lerobot_io 的设置，确保本地数据集可加载
for _name in ("lerobot", "datasets", "PIL", "torchvision", "ffmpeg", "av"):
    logging.getLogger(_name).setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _read_episodes(dataset_dir: Path, repo_id: str) -> list:
    """逐帧读取 LeRobot 数据集，重构 Episode 列表。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from ur_mjlab_bc_rl.imitation.dataset import Episode

    ds = LeRobotDataset(repo_id, root=str(dataset_dir.parent))
    episodes = []
    cur = Episode()
    prev_ep = None

    n = ds.num_frames
    pct_step = max(1, n // 10)
    for fi in range(n):
        if fi % pct_step == 0:
            print(f"    读取 {fi}/{n}...", end="\r", flush=True)
        f = ds[fi]
        ep = f.get("episode_index", 0)
        if prev_ep is not None and ep != prev_ep:
            episodes.append(cur)
            cur = Episode()
            # 每攒 10 个 episode 就返回一批，调用方负责累积
            if len(episodes) >= 10:
                yield episodes
                episodes = []

        rgb = f["observation.images.rgb"].numpy()
        if rgb.shape[0] == 3:
            rgb = rgb.transpose(1, 2, 0)
        depth = f["observation.images.depth"].numpy()
        if depth.ndim == 3 and depth.shape[0] == 3:
            depth = depth[0]

        obs = {
            "state": OrderedDict([
                ("arm_joint_pos", f["observation.state"].numpy()[:6].astype(np.float32)),
                ("gripper_pos", f["observation.state"].numpy()[6:7].astype(np.float32)),
                ("last_action", np.zeros(7, dtype=np.float32)),
            ]),
            "rgb": rgb.astype(np.uint8),
            "depth": depth.astype(np.float32),
            "task_id": 0,
        }
        cur.add(obs, f["action"].numpy().astype(np.float32))
        prev_ep = ep

    if len(cur) > 0:
        episodes.append(cur)
    if episodes:
        yield episodes


def main():
    parser = argparse.ArgumentParser(description="合并 LeRobot 数据集")
    parser.add_argument("sources", nargs="+", type=str, help="源数据集目录")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    args = parser.parse_args()

    sources = [Path(s).resolve() for s in args.sources]
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # 扫描源
    valid = []
    fps = 100
    print(f"\n{'='*60}")
    print(f"合并 {len(sources)} 个数据集")
    for sd in sources:
        info = sd / "meta" / "info.json"
        if not info.exists():
            print(f"  ⚠ 跳过: {sd}")
            continue
        with open(info) as f:
            meta = json.load(f)
        valid.append(sd)
        fps = meta.get("fps", fps)
        print(f"  {sd.name}: {meta['total_episodes']} eps, {meta['total_frames']} fr")

    from ur_mjlab_bc_rl.imitation.dataset import LeRobotSaver
    saver = LeRobotSaver()
    target_repo = "ur5_pick_place_merged"
    dataset = None
    total_eps = 0

    for i, sd in enumerate(valid, start=1):
        repo = sd.name
        print(f"\n[{i}/{len(valid)}] {repo}")
        for batch in _read_episodes(sd, repo):
            if dataset is None:
                dataset = saver.create(str(output_root), target_repo, batch, fps)
            else:
                saver.append_episodes(dataset, batch, start_idx=total_eps)
            total_eps += len(batch)
            batch.clear()
            gc.collect()
            if i < len(valid) or True:
                dataset = saver.reopen(dataset)
        print(f"    → 累计 {total_eps} episodes")

    result = saver.finalize(dataset)
    print(f"\n{'='*60}")
    print(f"✓ 合并完成: {result}")
    print(f"  共 {total_eps} episodes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
