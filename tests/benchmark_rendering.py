#!/usr/bin/env python3
"""性能分析脚本 — 定位 CameraSensor 渲染中 CPU-GPU 同步开销。

用法:
    cd /home/tdc/CodeProjects/UR_Mjlab_BC_RL
    uv run python tests/benchmark_rendering.py
"""

from __future__ import annotations

import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

os.environ.setdefault("MUJOCO_GL", "egl")
# 抑制 mujoco/lerobot 日志
import logging
for _n in ("lerobot", "datasets", "PIL", "torchvision", "ffmpeg", "av", "x265"):
    logging.getLogger(_n).setLevel(logging.WARNING)

import numpy as np

from ur_mjlab_bc_rl.simulate.cameras import CameraSensor
from ur_mjlab_bc_rl.simulate.mujoco_wrapper import MujocoWrapper
from ur_mjlab_bc_rl.utils.config_loader import load_scene_config


def sep(title: str) -> None:
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")


def benchmark_capture_single(cam: CameraSensor) -> dict:
    """单次 capture() 分阶段计时。"""
    # 先 warmup
    for _ in range(3):
        cam.capture()

    # --- 用猴子补丁方式拆分计时 ---
    import mujoco

    renderer = cam._renderer
    mj_data = cam._mj.data
    cam_id = cam._camera_id

    times = {"update_scene": [], "render_rgb": [], "render_depth": [], "total": []}

    N = 30
    for _ in range(N):
        t0 = time.perf_counter()

        t1 = time.perf_counter()
        renderer.update_scene(mj_data, camera=cam_id)
        t2 = time.perf_counter()

        renderer.disable_depth_rendering()
        rgb = renderer.render()
        t3 = time.perf_counter()

        renderer.enable_depth_rendering()
        depth_raw = renderer.render()
        t4 = time.perf_counter()

        # 模拟当前 ascontiguousarray 开销
        rgb_c = np.ascontiguousarray(rgb)
        depth_c = np.ascontiguousarray(depth_raw.astype(np.float32, copy=False))
        t5 = time.perf_counter()

        times["update_scene"].append((t2 - t1) * 1000)
        times["render_rgb"].append((t3 - t2) * 1000)
        times["render_depth"].append((t4 - t3) * 1000)
        times["total"].append((t5 - t0) * 1000)

    # 用 render_rgb 减去 update_scene 近似得到纯 GPU render+readback 时间
    # （render_rgb 已包含 update_scene？不，我们已拆分）

    return {k: (np.mean(v), np.std(v), np.min(v), np.max(v))
            for k, v in times.items()}


def benchmark_parallel_vs_sequential(cams: list[CameraSensor]) -> None:
    """对比 3 相机串行 vs 并行渲染。"""
    N = 50

    # ── 串行 ──
    for c in cams:
        c.capture()  # warmup
    t0 = time.perf_counter()
    for _ in range(N):
        for c in cams:
            c.capture()
    seq_ms = (time.perf_counter() - t0) / N * 1000

    # ── 线程并行 ──
    for c in cams:
        c.capture()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(cams)) as pool:
        for _ in range(N):
            futs = [pool.submit(CameraSensor.capture, c) for c in cams]
            [f.result() for f in futs]
    par_ms = (time.perf_counter() - t0) / N * 1000

    print(f"\n  串行 3 相机: {seq_ms:.1f} ms/frame")
    print(f"  并行 3 相机: {par_ms:.1f} ms/frame")
    if seq_ms > 0:
        print(f"  加速比:       {seq_ms / par_ms:.1f}x")
        print(f"  串行时纯阻塞占比: {(seq_ms - par_ms) / seq_ms * 100:.0f}%")


def benchmark_no_copy(cams: list[CameraSensor]) -> None:
    """对比有/无 ascontiguousarray 的开销。"""
    import mujoco
    cam = cams[0]
    renderer = cam._renderer
    cam_id = cam._camera_id
    mj_data = cam._mj.data

    for _ in range(3):
        cam.capture()
    N = 50

    # 当前方案（有 ascontiguousarray）
    t0 = time.perf_counter()
    for _ in range(N):
        renderer.update_scene(mj_data, camera=cam_id)
        renderer.disable_depth_rendering()
        rgb = np.ascontiguousarray(renderer.render())
        renderer.enable_depth_rendering()
        depth = np.ascontiguousarray(renderer.render().astype(np.float32, copy=False))
    cur_ms = (time.perf_counter() - t0) / N * 1000

    # 优化方案（无 ascontiguousarray，直接使用 render 返回值）
    t0 = time.perf_counter()
    for _ in range(N):
        renderer.update_scene(mj_data, camera=cam_id)
        renderer.disable_depth_rendering()
        rgb = renderer.render()
        renderer.enable_depth_rendering()
        depth = renderer.render().astype(np.float32, copy=False)
    opt_ms = (time.perf_counter() - t0) / N * 1000

    print(f"\n  当前 (ascontiguousarray): {cur_ms:.2f} ms")
    print(f"  优化 (直接使用):          {opt_ms:.2f} ms")
    if cur_ms > 0:
        print(f"  节省:                     {(cur_ms - opt_ms):.2f} ms ({(1 - opt_ms/cur_ms)*100:.0f}%)")


def benchmark_mj_step(cams: list[CameraSensor]) -> None:
    """单独测量 mj_step 开销。"""
    mj = cams[0]._mj
    N = 1000

    # 先 forward 一下
    mj.forward()
    t0 = time.perf_counter()
    for _ in range(N):
        mj.step(1)
    step_us = (time.perf_counter() - t0) / N * 1000
    print(f"\n  mj_step 单次: {step_us:.3f} ms ({1/step_us*1000:.0f} Hz)" if step_us > 0.001 else
          f"\n  mj_step 单次: {step_us*1000:.1f} µs ({1/step_us*1000:.0f} Hz)")


def benchmark_simulation_loop_fidelity(cams: list[CameraSensor], config) -> None:
    """模拟真实 run_episode 循环，测量 wall-clock vs sim-time 比率。"""
    mj = cams[0]._mj
    pdt = config.sim.physics_dt       # 0.001
    cam_dt = 1.0 / cams[0]._cam_dt if hasattr(cams[0], '_cam_dt') else 1.0 / 30.0

    # 简化的仿真循环（无 IK/controller）
    SIM_TIME = 1.0  # 仿真 1 秒
    cam_timers = {c._camera_name: 0.0 for c in cams}

    t0 = time.perf_counter()
    t_sim = 0.0
    steps = 0
    renders = 0

    while t_sim < SIM_TIME:
        mj.step(1)
        t_sim += pdt
        steps += 1

        # 相机渲染
        for c in cams:
            name = c._camera_name
            cam_timers[name] += pdt
            if cam_timers[name] >= cam_dt:
                cam_timers[name] -= cam_dt
                c.capture()
                renders += 1

    wall = time.perf_counter() - t0
    ratio = wall / SIM_TIME

    print(f"\n  仿真 {SIM_TIME}s → wall-clock {wall:.2f}s")
    print(f"  慢放倍数: {ratio:.1f}x")
    print(f"  physics steps: {steps}  renders: {renders}")
    print(f"  等效 physics rate: {steps/wall:.0f} Hz (目标 {1/pdt:.0f} Hz)")
    print(f"  渲染耗时占比: {(renders / 3 * 5) / wall * 100 if renders > 0 else 0:.0f}% (粗略估计, 假设 5ms/render)")


def benchmark_depth_dtype(cams: list[CameraSensor]) -> None:
    """检查 render() 返回的 depth 实际 dtype，确认是否需要 astype。"""
    cam = cams[0]
    import mujoco
    renderer = cam._renderer
    renderer.update_scene(cam._mj.data, camera=cam._camera_id)
    renderer.enable_depth_rendering()
    depth = renderer.render()
    print(f"\n  depth dtype: {depth.dtype}  shape: {depth.shape}")
    print(f"  depth contiguous: {depth.flags['C_CONTIGUOUS']}")
    print(f"  depth itemsize: {depth.itemsize} bytes")

    # 测试 astype 是否真的拷贝
    d2 = depth.astype(np.float32, copy=False)
    print(f"  astype(copy=False) is same: {d2 is depth}")
    print(f"  astype(copy=False) dtype: {d2.dtype}")

    renderer.disable_depth_rendering()
    rgb = renderer.render()
    print(f"  rgb dtype: {rgb.dtype}  shape: {rgb.shape}")
    print(f"  rgb contiguous: {rgb.flags['C_CONTIGUOUS']}")


def main():
    print("=" * 60)
    print("  CameraSensor 渲染性能分析")
    print("=" * 60)

    config = load_scene_config("dual_pick_place")
    mj = MujocoWrapper(config.task.scene_file, render=False)
    mj.open()

    # 推进几步让场景稳定
    for _ in range(100):
        mj.step(1)

    cams = [CameraSensor(mj, c.name, (c.height, c.width)) for c in config.cameras]
    print(f"\n  相机: {[c._camera_name for c in cams]}")
    print(f"  分辨率: {cams[0].width}x{cams[0].height}")
    print(f"  physics_dt: {config.sim.physics_dt}")

    # 1. mj_step 开销
    sep("1. mj_step 单步开销")
    benchmark_mj_step(cams)

    # 2. 深度 dtype 检查
    sep("2. 渲染返回值 dtype 检查")
    benchmark_depth_dtype(cams)

    # 3. 单相机 capture 分阶段计时
    sep("3. 单相机 capture() 分阶段耗时")
    stats = benchmark_capture_single(cams[0])
    print(f"\n  {'阶段':<18} {'均值':>8} {'std':>8} {'min':>8} {'max':>8}")
    print(f"  {'─'*18} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for phase in ["update_scene", "render_rgb", "render_depth", "total"]:
        m, s, lo, hi = stats[phase]
        print(f"  {phase:<18} {m:>7.2f}ms {s:>7.2f}ms {lo:>7.2f}ms {hi:>7.2f}ms")
    print(f"\n  ⚠ render_rgb 包含 GPU 光栅化 + GPU→CPU readback")
    print(f"  ⚠ render_depth 包含 GPU 光栅化 + GPU→CPU readback")
    print(f"  ⚠ GPU→CPU readback = render 耗时 - update_scene 耗时 ≈ "
          f"{stats['render_rgb'][0] - stats['update_scene'][0]:.1f}ms (RGB) + "
          f"{stats['render_depth'][0] - stats['update_scene'][0]:.1f}ms (Depth)")

    # 4. 串行 vs 并行
    sep("4. 串行 vs 线程并行 渲染")
    benchmark_parallel_vs_sequential(cams)

    # 5. ascontiguousarray 开销
    sep("5. ascontiguousarray 拷贝开销")
    benchmark_no_copy(cams)

    # 6. 仿真循环回放比
    sep("6. 简化仿真循环 wall-clock vs sim-time")
    benchmark_simulation_loop_fidelity(cams, config)

    # 清理
    for c in cams:
        c.close()
    mj.close()

    print(f"\n{'='*60}")
    print("  分析完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
