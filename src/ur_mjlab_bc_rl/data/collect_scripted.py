#!/usr/bin/env python3
"""Scripted Teacher 数据采集 — 多臂多相机 MuJoCo 物理仿真。

用法:
    uv run python -m ur_mjlab_bc_rl.data.collect_scripted \\
        --task dual_pick_place \\
        --dataset-config configs/dataset/dataset_dual.yaml \\
        --episodes 50
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# 减少第三方库日志噪音
os.environ.setdefault("FFMPEG_LOGLEVEL", "error")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
for _name in ("lerobot", "datasets", "PIL", "torchvision", "ffmpeg", "av", "x265"):
    logging.getLogger(_name).setLevel(logging.WARNING)

from ur_mjlab_bc_rl.utils.config_loader import load_scene_config, get_task_list
from ur_mjlab_bc_rl.data.simulation_manager import SimulationManager
from ur_mjlab_bc_rl.data.scripted_controller import ScriptedTeacherController
from ur_mjlab_bc_rl.data.dataset_writer import LeRobotDatasetWriter, LeRobotDatasetConfig
from ur_mjlab_bc_rl.data.dataset_config import DatasetConfig


def main() -> None:
    p = argparse.ArgumentParser(description="脚本化专家数据采集")
    p.add_argument("--task", required=True, help="任务名（--task list 列出所有）")
    p.add_argument("--dataset-config", required=True, help="数据集配置文件路径")
    p.add_argument("--episodes", type=int, default=50, help="目标 episode 数")
    p.add_argument("--output", default="outputs/datasets/expert", help="输出目录")
    p.add_argument("--no-render", action="store_true", help="无头模式")
    p.add_argument("--overwrite", action="store_true", help="覆盖已有数据集")
    args = p.parse_args()

    if args.task == "list":
        print("可用任务:", get_task_list())
        return

    config = load_scene_config(args.task)
    task = config.task
    render = not args.no_render
    dcfg = DatasetConfig.from_yaml(args.dataset_config)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.output) / task.name / ts
    root = root.resolve()

    wcfg = LeRobotDatasetConfig(
        repo_id=f"ur5_{task.name}",
        root=root,
        fps=int(round(dcfg.recode_hz)),
        state_dim=dcfg.state_dim,
        action_dim=dcfg.action_dim,
        cameras=config.cameras,
    )
    writer = LeRobotDatasetWriter.create_new(wcfg, overwrite=args.overwrite,
                                              dataset_cfg=dcfg)

    print(f"\n{'='*72}")
    print(f"任务: {task.name} | Teacher: {task.teacher}")
    print(f"  场景: {task.scene_file}  |  记录频率: {dcfg.recode_hz:.0f} Hz  "
          f"(max_scale={dcfg.max_scale})")
    print(f"  臂: {[r.name for r in config.robots]}")
    print(f"  相机: {[c.name for c in config.cameras]}  |  Episodes: {args.episodes}")
    print(f"  state_dim={dcfg.state_dim}  action_dim={dcfg.action_dim}")
    print(f"{'='*72}")

    mgr = SimulationManager(config, render=render, dataset_cfg=dcfg)
    controller = ScriptedTeacherController(config, mgr.model, mgr.data)

    try:
        t0 = time.perf_counter()
        collected = 0
        max_attempts = config.collection.max_attempts
        total_attempts = 0

        flush_thread: threading.Thread | None = None

        while collected < args.episodes:
            cb, flush_ep, discard_ep = writer.make_stream_callback()

            success = False
            for attempt in range(1, max_attempts + 1):
                t_ep = time.perf_counter()
                n_frames = mgr.run_episode(controller, frame_callback=cb)
                t_ep = time.perf_counter() - t_ep
                total_attempts += 1

                if controller.is_success():
                    if flush_thread is not None:
                        flush_thread.join()
                        flush_thread = None
                    flush_thread = threading.Thread(
                        target=flush_ep, args=(task.name,), daemon=True,
                    )
                    flush_thread.start()
                    collected += 1
                    success = True
                    print(f"  ✓ {collected}/{args.episodes}  "
                          f"frames={n_frames}  sim_t={t_ep:.1f}s  "
                          f"total={time.perf_counter() - t0:.0f}s")
                    break
                else:
                    discard_ep()
                    print(f"  ✗ 尝试 {total_attempts} 失败  "
                          f"frames={n_frames}  sim_t={t_ep:.1f}s  "
                          f"total={time.perf_counter() - t0:.0f}s")
            if not success:
                print(f"  ✗ 连续 {max_attempts} 次失败，跳过")

        if flush_thread is not None:
            flush_thread.join()

    except KeyboardInterrupt:
        print("\n  中断：丢弃未完成的 episode…")
        discard_ep()
    finally:
        mgr.close()

    writer.finalize()
    elapsed = time.perf_counter() - t0
    epm = collected / elapsed * 60 if elapsed > 0 else 0
    print(f"\n{'='*72}")
    print(f"✓ {task.name}: {collected}/{args.episodes} episodes "
          f"in {elapsed:.1f}s ({epm:.1f} ep/min, 总尝试 {total_attempts})")
    print(f"  → {root}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
