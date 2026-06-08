#!/usr/bin/env python3
"""回放模仿学习数据集。

加载保存的 episode 数据并在 MuJoCo viewer 中回放。

示例:
  python scripts/replay_imitation_dataset.py --data outputs/datasets/expert/pick_place/pick_place_episodes.pt
  python scripts/replay_imitation_dataset.py --data outputs/datasets/expert/pick_place --task pick_place
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import mujoco
import mujoco.viewer

# 将项目 src 加入路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ur_mjlab_bc_rl.imitation.mujoco_env import EEDeltaActionExecutor

SCENE_DIR = PROJECT_ROOT / "assets" / "mujoco" / "scenes"

TASK_SCENES = {
    "pick_place": "pick_place.xml",
    "push_t": "push_t.xml",
    "peg_slot": "peg_in_slot.xml",
}


def load_episodes(data_path: Path) -> list[dict]:
    """加载 episode 数据。"""
    episodes = []

    if data_path.is_dir():
        all_files = sorted(data_path.glob("*_episodes.pt")) + sorted(data_path.glob("*.pt"))
        all_files += sorted(data_path.glob("*.npy"))
        for f in all_files:
            if f.suffix == ".pt":
                data = torch.load(f, map_location="cpu", weights_only=False)
            elif f.suffix == ".npy":
                data = np.load(f, allow_pickle=True).item()
            else:
                continue
            if isinstance(data, list):
                episodes.extend(data)
            elif isinstance(data, dict):
                episodes.append(data)
    else:
        if data_path.suffix == ".pt":
            data = torch.load(data_path, map_location="cpu", weights_only=False)
        elif data_path.suffix == ".npy":
            data = np.load(data_path, allow_pickle=True).item()
        else:
            raise ValueError(f"不支持的文件格式: {data_path.suffix}")
        if isinstance(data, list):
            episodes = data
        elif isinstance(data, dict):
            episodes = [data]

    return episodes


def main():
    parser = argparse.ArgumentParser(description="回放模仿学习数据集")
    parser.add_argument("--data", type=str, required=True, help="数据集路径")
    parser.add_argument("--task", type=str, default="pick_place", help="任务名称")
    parser.add_argument("--episode", type=int, default=0, help="回放的 episode 索引")
    parser.add_argument("--speed", type=float, default=1.0, help="回放速度倍率")
    parser.add_argument("--loop", action="store_true", help="循环回放")
    args = parser.parse_args()

    data_path = Path(args.data)
    scene_file = TASK_SCENES.get(args.task, "pick_place.xml")
    scene_path = SCENE_DIR / scene_file

    if not scene_path.exists():
        print(f"场景文件不存在: {scene_path}")
        sys.exit(1)

    # 加载数据
    print(f"加载数据: {data_path}")
    episodes = load_episodes(data_path)
    print(f"加载了 {len(episodes)} 个 episode")

    if not episodes:
        print("没有数据可回放")
        sys.exit(1)

    if args.episode >= len(episodes):
        print(f"Episode 索引 {args.episode} 超出范围 (0-{len(episodes) - 1})")
        sys.exit(1)

    # 提取动作序列
    ep = episodes[args.episode]
    actions_raw = ep.get("actions", [])
    if not actions_raw:
        print("Episode 中没有动作数据")
        sys.exit(1)

    # 转为 numpy
    actions = []
    for a in actions_raw:
        if isinstance(a, list):
            actions.append(np.array(a, dtype=np.float64))
        elif isinstance(a, np.ndarray):
            actions.append(a)
        else:
            actions.append(np.array([a], dtype=np.float64))

    actions = np.array(actions)
    print(f"Episode {args.episode}: {len(actions)} 步")

    # 初始化 MuJoCo
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    action_executor = EEDeltaActionExecutor(model, data)

    step_idx = [0]  # 用列表实现闭包

    def key_callback(key: int) -> None:
        if key == 256:  # ESC
            return
        elif key == ord('R') or key == ord('r'):
            step_idx[0] = 0
            mujoco.mj_resetData(model, data)
            print("\n[重置回放]")

    print(f"\n回放 Episode {args.episode} ({len(actions)} steps)")
    print("按 R 重置, ESC 退出")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            i = step_idx[0]

            if i < len(actions):
                action = actions[i]
                ctrl = action_executor.execute(action)
                data.ctrl[:] = ctrl
                step_idx[0] += 1
            elif args.loop:
                step_idx[0] = 0
                mujoco.mj_resetData(model, data)
                continue

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01 / args.speed)


if __name__ == "__main__":
    main()
