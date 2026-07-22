#!/usr/bin/env python3
"""键盘手动采集专家数据 — 使用 simulate 模块。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ur_mjlab_bc_rl.simulate.env import (
    MujocoInterface, CameraSensor, ObservationCollector,
)
from ur_mjlab_bc_rl.simulate.dataset_writer import (
    Episode, LeRobotDatasetWriter, LeRobotDatasetConfig,
)

SCENE_DIR = PROJECT_ROOT / "assets" / "mujoco" / "scenes"

TASKS = {
    "pick_place": {"scene": "pick_place.xml", "task_id": 0},
    "push_t": {"scene": "push_t.xml", "task_id": 1},
    "peg_slot": {"scene": "peg_in_slot.xml", "task_id": 2},
}


class KeyboardController:
    def __init__(self, pos_step=0.01, rot_step=0.1):
        self.pos_step = pos_step
        self.rot_step = rot_step
        self.gripper_open = True

    def get_action(self, key: int) -> np.ndarray | None:
        a = np.zeros(7)
        if key == ord('W'): a[0] = self.pos_step
        elif key == ord('S'): a[0] = -self.pos_step
        elif key == ord('A'): a[1] = self.pos_step
        elif key == ord('D'): a[1] = -self.pos_step
        elif key == ord('Q'): a[2] = self.pos_step
        elif key == ord('E'): a[2] = -self.pos_step
        elif key == ord('I'): a[3] = self.rot_step
        elif key == ord('K'): a[3] = -self.rot_step
        elif key == ord('J'): a[4] = self.rot_step
        elif key == ord('L'): a[4] = -self.rot_step
        elif key == ord('U'): a[5] = self.rot_step
        elif key == ord('O'): a[5] = -self.rot_step
        elif key == 32:
            self.gripper_open = not self.gripper_open
            a[6] = 0.8 if self.gripper_open else 0.0
        else:
            return None
        return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="pick_place")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--output", default="outputs/datasets/expert")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--pos-step", type=float, default=0.01)
    p.add_argument("--rot-step", type=float, default=0.1)
    args = p.parse_args()

    cfg = TASKS[args.task]
    sp = SCENE_DIR / cfg["scene"]
    if not sp.exists():
        print(f"✗ 场景不存在: {sp}"); return

    mj = MujocoInterface(str(sp), render=True)
    cam = CameraSensor(mj, image_size=(args.image_size, args.image_size))
    col = ObservationCollector(mj, cam)
    kbd = KeyboardController(args.pos_step, args.rot_step)

    episodes: list[Episode] = []
    cur = Episode()
    cam.capture()

    print(f"\n{'='*60}\n键盘控制采集 任务: {args.task}\n{'='*60}")
    print("W/S/A/D/Q/E 移动  I/J/K/L/U/O 旋转  Space 夹爪")
    print("Enter=保存  Backspace=丢弃  Esc=退出\n{'='*60}")

    def key_cb(key: int) -> None:
        nonlocal cur, episodes
        if key == 256: return
        elif key == 13:
            if len(cur) > 10:
                episodes.append(cur)
                print(f"\n[✓ Ep {len(episodes)} ({len(cur)} steps)]")
            cur = Episode()
            col.reset()
        elif key in (8, 127):
            cur = Episode()
            col.reset()
            print("\n[✗ 丢弃]")
        else:
            act = kbd.get_action(key)
            if act is not None:
                obs = col.collect(task_id=cfg["task_id"])
                cur.add(obs, act)

    # 简单循环：用 mujoco.viewer 的回调
    with mujoco.viewer.launch_passive(mj.model, mj.data, key_callback=key_cb) as viewer:
        while viewer.is_running():
            mj.step()
            cam.capture()
            viewer.sync()

    cam.close()
    mj.close()

    # 保存
    if episodes:
        from datetime import datetime
        H = W = args.image_size
        writer = LeRobotDatasetWriter.create_new(
            LeRobotDatasetConfig(
                repo_id=f"ur5_{args.task}",
                root=Path(args.output) / args.task / datetime.now().strftime("%Y%m%d_%H%M%S"),
                fps=100, state_dim=14, action_dim=7,
                image_height=H, image_width=W,
            ),
            overwrite=True,
        )
        for ep in episodes:
            writer.append_episode(ep, task_label=args.task)
        writer.finalize()
        print(f"\n✓ 已保存 {len(episodes)} episodes")


if __name__ == "__main__":
    main()
