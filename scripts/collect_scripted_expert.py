#!/usr/bin/env python3
"""Scripted Teacher 数据采集 — 多臂多相机 MuJoCo 物理仿真。"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("FFMPEG_LOGLEVEL", "error")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
for _name in ("lerobot", "datasets", "PIL", "torchvision", "ffmpeg", "av"):
    logging.getLogger(_name).setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ur_mjlab_bc_rl.simulate.config_loader import (
    load_scene_config, get_task_list,
)
from ur_mjlab_bc_rl.simulate import (
    SimulationManager, ScriptedTeacherController,
    LeRobotDatasetWriter, LeRobotDatasetConfig,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--output", default="outputs/datasets/expert")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.task == "list":
        print("可用任务:", get_task_list())
        return

    config = load_scene_config(args.task)
    task = config.task
    render = not args.no_render

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.output) / task.name / ts
    root = root.resolve()

    wcfg = LeRobotDatasetConfig(
        repo_id=f"ur5_{task.name}",
        root=root,
        fps=int(round(1.0 / config.sim.policy_dt)),
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        cameras=config.cameras,
    )
    writer = LeRobotDatasetWriter.create_new(wcfg, overwrite=args.overwrite)

    print(f"\n{'='*72}")
    print(f"任务: {task.name} | Teacher: {task.teacher}")
    print(f"  场景: {task.scene_file}")
    print(f"  臂: {[r.name for r in config.robots]}")
    print(f"  相机: {[c.name for c in config.cameras]}  |  Episodes: {args.episodes}")
    print(f"{'='*72}")

    mgr = SimulationManager(config, render=render)
    controller = ScriptedTeacherController(config, mgr.model, mgr.data)

    try:
        t0 = time.time()
        collected = 0
        max_attempts = config.collection.max_attempts
        total_attempts = 0

        # 流式回调：本地缓冲，成功时后台编码（与下个 episode 仿真并行）
        stream_cb, flush_ep, discard_ep = writer.make_stream_callback()

        flush_thread: threading.Thread | None = None

        while collected < args.episodes:
            # 等待上一个 episode 的编码完成
            if flush_thread is not None:
                flush_thread.join()
                flush_thread = None

            success = False
            for _ in range(max_attempts):
                mgr.run_episode(controller, frame_callback=stream_cb)
                total_attempts += 1
                if controller.is_success():
                    # 后台线程编码，不阻塞下一个 episode 仿真
                    flush_thread = threading.Thread(
                        target=flush_ep, args=(task.name,), daemon=True,
                    )
                    flush_thread.start()
                    collected += 1
                    success = True
                    print(f"  ✓ {collected}/{args.episodes}")
                    break
                else:
                    discard_ep()
                    print(f"  ✗ 尝试 {total_attempts} 失败，重试…")
            if not success:
                print(f"  ✗ 连续 {max_attempts} 次失败，跳过")

        # 等待最后一个 episode 编码完成
        if flush_thread is not None:
            flush_thread.join()
    except KeyboardInterrupt:
        print("\n  中断：丢弃未完成的 episode…")
        discard_ep()  # 丢弃中断时的部分帧
    finally:
        mgr.close()

    writer.finalize()
    elapsed = time.time() - t0
    print(f"\n✓ {task.name}: {collected}/{args.episodes} in {elapsed:.1f}s "
          f"(总尝试 {total_attempts}) → {root}")


if __name__ == "__main__":
    main()
