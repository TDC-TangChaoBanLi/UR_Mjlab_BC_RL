#!/usr/bin/env python3
"""Scripted Teacher 数据采集 — 纯 actuator 控制 + MuJoCo 物理仿真。"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("FFMPEG_LOGLEVEL", "error")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
for _name in ("lerobot", "datasets", "PIL", "torchvision", "ffmpeg", "av"):
    logging.getLogger(_name).setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ur_mjlab_bc_rl.simulate.config_loader import (  # noqa: E402
    get_arm_joints, get_camera_name, get_collection_params,
    get_gripper_joints, get_image_size, get_sim_params, load_tasks,
)
from ur_mjlab_bc_rl.simulate.env import (  # noqa: E402
    CameraSensor, MinkIK, MujocoInterface,
    ObservationCollector, ResetManager,
)
from ur_mjlab_bc_rl.simulate.dataset_writer import (  # noqa: E402
    Episode, LeRobotDatasetWriter, LeRobotDatasetConfig,
)

SIM = get_sim_params()
COLL = get_collection_params()
ARM = get_arm_joints()
GRP = get_gripper_joints()
CAM_NAME = get_camera_name()
IMG_SIZE = get_image_size()

_TEACHERS: dict[str, Any] = {}


def _init() -> None:
    from ur_mjlab_bc_rl.simulate.teachers import (
        PickPlaceTeacher, PushTTeacher, PegSlotTeacher,
    )
    _TEACHERS.update({
        "PickPlaceTeacher": PickPlaceTeacher,
        "PushTTeacher": PushTTeacher,
        "PegSlotTeacher": PegSlotTeacher,
    })


_init()


def _collect_one(
    task_name: str, teacher_name: str, task_id: int,
    max_steps: int, mj: MujocoInterface, cam: CameraSensor,
    ik: MinkIK, rm: ResetManager,
) -> Episode | None:
    """采集一条 episode（复用已创建的 mj/cam/ik/rm）。"""
    tcls = _TEACHERS[teacher_name]

    a_ids = [mj.get_actuator_id(n + "_ACTUATOR") for n in ARM]
    g_ids = [mj.get_actuator_id(n + "_ACTUATOR") for n in GRP]
    pdt = SIM["physics_dt"]
    cdt = SIM["camera_dt"]
    adt = SIM["policy_dt"]

    col = ObservationCollector(
        mj, cam, ARM, GRP, include_last_action=True, action_dim=7,
    )

    for _ in range(int(COLL["max_attempts"])):
        rm.reset(task=task_name, randomize_objects=True)
        teacher = tcls(mj.model, mj.data)
        teacher.reset()
        ik.reset(mj.get_qpos())
        col.reset()
        ep = Episode()
        cam.capture()
        mj.sync_viewer()

        t_policy = 0.0
        t_camera = 0.0

        for _ in range(max_steps):
            mj.step()
            t_policy += pdt
            t_camera += pdt

            if t_camera >= cdt:
                cam.capture()
                t_camera -= cdt

            if t_policy < adt:
                continue
            t_policy -= adt

            ee = teacher.step()
            tgt = np.asarray(ee[:7], dtype=np.float64)
            gcmd = [float(ee[7])]

            jt = ik.solve(mj.get_qpos(), tgt, dt=adt)

            ctrl = mj.get_ctrl()
            for i, aid in enumerate(a_ids):
                ctrl[aid] = jt[i]
            for i, gid in enumerate(g_ids):
                ctrl[gid] = gcmd[i]
            mj.set_ctrl(ctrl)

            act = np.append(jt.copy(), gcmd).astype(np.float32)
            obs = col.collect(task_id=task_id)
            ep.add(obs, act, copy_arrays=True)
            col.update_last_action(act)

            if teacher.is_done():
                break

        if teacher.is_success() and len(ep) > 0:
            return ep
        ep.clear()

    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True,
                   choices=["pick_place", "push_t", "peg_slot", "all"])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--output", default="outputs/datasets/expert")
    p.add_argument("--max-steps", type=int, default=COLL["max_steps"])
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out = Path(args.output)
    tcs = load_tasks()
    torun = list(tcs.keys()) if args.task == "all" else [args.task]
    sd = PROJECT_ROOT / "assets" / "mujoco" / "scenes"

    for tn in torun:
        tc = tcs.get(tn)
        if tc is None:
            print(f"⚠ 不存在: {tn}"); continue
        sp = sd / tc["scene"]
        if not sp.exists():
            print(f"⚠ 场景不存在: {sp}"); continue

        H, W = IMG_SIZE
        rid = f"ur5_{tn}"
        fps = int(round(1.0 / SIM["policy_dt"]))
        state_dim = len(ARM) + len(GRP) + 7
        dr = tc["depth_range"]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = (out / tn / ts).resolve()

        wcfg = LeRobotDatasetConfig(
            repo_id=rid, root=root, fps=fps,
            state_dim=state_dim, action_dim=7,
            image_height=int(H), image_width=int(W),
            state_keys=("arm_joint_pos", "gripper_pos", "last_action"),
            depth_min=float(dr[0]), depth_max=float(dr[1]),
        )
        w = LeRobotDatasetWriter.create_new(wcfg, overwrite=args.overwrite)

        print(f"\n{'='*72}")
        print(f"任务: {tn} | Teacher: {tc['teacher']} | Episodes: {args.episodes}")
        print(f"{'='*72}")

        render = not args.no_render
        mj = MujocoInterface(str(sp), render=render)
        try:
            if render:
                mj.set_viewer_camera((0.45, 0.0, 0.65), 1.8, -25.0, 130.0)
            cam = CameraSensor(mj, CAM_NAME, IMG_SIZE)
            ik = MinkIK(mj.model, mj.get_qpos())
            rm = ResetManager(mj, ARM, GRP)

            t0 = time.time()
            collected = 0
            for ei in range(args.episodes):
                ep = _collect_one(
                    tn, tc["teacher"], tc["task_id"],
                    args.max_steps, mj, cam, ik, rm,
                )
                if ep is not None:
                    nfr = len(ep)
                    w.append_episode(ep, task_label=tn)
                    collected += 1
                    print(f"  ✓ {collected}/{args.episodes} ({nfr} fr)")
                else:
                    print(f"  ✗ ep {ei} failed")
        finally:
            mj.close()

        result = w.finalize()
        elapsed = time.time() - t0
        print(f"\n✓ {tn}: {collected}/{args.episodes} in {elapsed:.1f}s → {result}")


if __name__ == "__main__":
    main()
