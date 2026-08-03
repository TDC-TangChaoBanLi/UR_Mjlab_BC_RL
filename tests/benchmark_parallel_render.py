#!/usr/bin/env python3
"""多进程并行渲染验证脚本。

用法:
    cd /home/tdc/CodeProjects/UR_Mjlab_BC_RL
    uv run python tests/benchmark_parallel_render.py
"""

from __future__ import annotations

import logging
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
for _n in ("lerobot", "datasets", "PIL", "torchvision", "ffmpeg", "av", "x265"):
    logging.getLogger(_n).setLevel(logging.WARNING)

import numpy as np

from ur_mjlab_bc_rl.simulate.mujoco_wrapper import MujocoWrapper
from ur_mjlab_bc_rl.simulate.cameras import CameraSensor
from ur_mjlab_bc_rl.simulate.render_pool import RenderProcessPool, CameraSpec
from ur_mjlab_bc_rl.utils.config_loader import load_scene_config


def main():
    config = load_scene_config("dual_pick_place")

    # ── 初始化主进程 MuJoCo ──
    mj = MujocoWrapper(config.task.scene_file, render=False)
    mj.open()
    for _ in range(100):
        mj.step(1)

    # ── 创建 CameraSensor（串行基准）──
    cams = [CameraSensor(mj, c.name, (c.height, c.width)) for c in config.cameras]

    # ── 创建 RenderProcessPool（并行）──
    specs = [CameraSpec(name=c.name, height=c.height, width=c.width)
             for c in config.cameras]
    pool = RenderProcessPool(config.task.scene_file, specs)

    print("=" * 60)
    print("  多进程并行渲染验证")
    print("=" * 60)
    print(f"  相机: {[s.name for s in specs]}")
    print(f"  分辨率: {specs[0].width}x{specs[0].height}")

    try:
        pool.start()
        print(f"  Pool 启动: OK ({len(specs)} workers)")

        # 推进仿真到稳定状态
        for _ in range(20):
            mj.step(1)

        qpos = mj.get_qpos()
        qvel = mj.get_qvel()

        # ── 基准：串行渲染 ──
        N = 20
        for c in cams:
            c.capture()  # warmup
        t0 = time.perf_counter()
        for _ in range(N):
            for c in cams:
                c.capture()
        seq_ms = (time.perf_counter() - t0) / N * 1000

        # ── 验证：并行渲染 ──
        # warmup
        _ = pool.render_all(qpos, qvel)
        t0 = time.perf_counter()
        for _ in range(N):
            _ = pool.render_all(qpos, qvel)
        par_ms = (time.perf_counter() - t0) / N * 1000

        print(f"\n  串行 (CameraSensor):   {seq_ms:.1f} ms/frame")
        print(f"  并行 (RenderPool):      {par_ms:.1f} ms/frame")
        if seq_ms > 0 and par_ms > 0:
            speedup = seq_ms / par_ms
            print(f"  加速比:                 {speedup:.1f}x")
            print(f"  有效帧率提升:            {1000/seq_ms:.1f} → {1000/par_ms:.1f} FPS")

        # ── 像素级正确性验证 ──
        print(f"\n  ── 正确性验证 ──")
        for c in cams:
            c.capture()
        serial_results = {c._camera_name: c.read(copy=True) for c in cams}
        par_results_list = pool.render_all(qpos, qvel)
        par_results = {s.name: r for s, r in zip(specs, par_results_list)}

        all_match = True
        for spec in specs:
            s_rgb = serial_results[spec.name]["rgb"]
            p_rgb = par_results[spec.name]["rgb"]
            s_depth = serial_results[spec.name]["depth"]
            p_depth = par_results[spec.name]["depth"]

            rgb_diff = np.abs(s_rgb.astype(float) - p_rgb.astype(float)).max()
            depth_diff = np.abs(s_depth - p_depth).max()

            status = "✓" if rgb_diff < 2 and depth_diff < 1e-3 else "✗"
            if status == "✗":
                all_match = False
            print(f"  {status} {spec.name}: rgb max_diff={rgb_diff:.1f}  depth max_diff={depth_diff:.4f}")

        if all_match:
            print(f"\n  ✅ 渲染结果一致，多进程并行方案可用")
        else:
            print(f"\n  ⚠️  渲染结果有差异（浮点精度或状态同步问题）")

        # ── 估算实际采集加速 ──
        print(f"\n  ── 实际采集时间估算 (10s sim, 30fps record) ──")
        sim_time = 10.0
        frames = int(sim_time * 30)
        # 串行: frames × seq_ms + physics + IK
        phys_per_frame = 1000 / 30  # 33 physics steps per frame
        phys_time = phys_per_frame * 0.073  # ms per physics step ≈ 2.4ms per frame
        ik_time = (100 / 30) * 2 * 0.5  # ~3.3ms per frame (2 arms, 100Hz IK within 30fps)
        serial_total = frames * (seq_ms + phys_time + ik_time) / 1000
        parallel_total = frames * (par_ms + phys_time + ik_time) / 1000
        print(f"  串行估算: {serial_total:.1f}s wall-clock for {sim_time}s sim ({sim_time/serial_total:.1f}x real-time)")
        print(f"  并行估算: {parallel_total:.1f}s wall-clock for {sim_time}s sim ({sim_time/parallel_total:.1f}x real-time)")

    finally:
        pool.stop()
        for c in cams:
            c.close()
        mj.close()

    print(f"\n{'='*60}")
    print("  验证完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
