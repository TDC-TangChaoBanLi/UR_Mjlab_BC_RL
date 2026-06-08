#!/usr/bin/env python3
"""键盘手动采集专家数据。"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ur_mjlab_bc_rl.imitation.mujoco_env import CameraSensor, ObservationCollector
from ur_mjlab_bc_rl.imitation.dataset import Episode, LeRobotSaver

SCENE_DIR = PROJECT_ROOT / "assets" / "mujoco" / "scenes"

TASK_CONFIGS = {
    "pick_place": {"scene": "pick_place.xml", "task_id": 0},
    "push_t": {"scene": "push_t.xml", "task_id": 1},
    "peg_slot": {"scene": "peg_in_slot.xml", "task_id": 2},
}


class KeyboardController:
    """键盘控制器。"""
    def __init__(self, pos_step=0.01, rot_step=0.1):
        self.pos_step = pos_step
        self.rot_step = rot_step
        self.gripper_open = True

    def get_action(self, key: int) -> np.ndarray | None:
        action = np.zeros(7)
        # 位置: W/S X轴, A/D Y轴, Q/E Z轴
        if key == ord('W'): action[0] = self.pos_step
        elif key == ord('S'): action[0] = -self.pos_step
        elif key == ord('A'): action[1] = self.pos_step
        elif key == ord('D'): action[1] = -self.pos_step
        elif key == ord('Q'): action[2] = self.pos_step
        elif key == ord('E'): action[2] = -self.pos_step
        # 姿态: I/K Roll, J/L Pitch, U/O Yaw
        elif key == ord('I'): action[3] = self.rot_step
        elif key == ord('K'): action[3] = -self.rot_step
        elif key == ord('J'): action[4] = self.rot_step
        elif key == ord('L'): action[4] = -self.rot_step
        elif key == ord('U'): action[5] = self.rot_step
        elif key == ord('O'): action[5] = -self.rot_step
        elif key == 32:  # Space
            self.gripper_open = not self.gripper_open
            action[6] = 0.8 if self.gripper_open else 0.0
        else:
            return None
        return action


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="pick_place")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--output", default="outputs/datasets/expert")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--pos-step", type=float, default=0.01)
    p.add_argument("--rot-step", type=float, default=0.1)
    args = p.parse_args()

    config = TASK_CONFIGS[args.task]
    scene_path = SCENE_DIR / config["scene"]
    if not scene_path.exists():
        print(f"✗ 场景不存在: {scene_path}"); return

    from ur_mjlab_bc_rl.imitation.mujoco_env import MujocoInterface
    mj = MujocoInterface(str(scene_path), render=True)
    camera = CameraSensor(mj, image_size=(args.image_size, args.image_size))
    collector = ObservationCollector(mj, camera)
    controller = KeyboardController(args.pos_step, args.rot_step)

    episodes: list[Episode] = []
    current_ep = Episode()
    camera.capture()

    print(f"\n{'='*60}\n键盘控制采集 任务: {args.task}\n{'='*60}")
    print("W/S/A/D/Q/E 移动  I/J/K/L/U/O 旋转  Space 夹爪")
    print("Enter=保存  Backspace=丢弃  Esc=退出\n{'='*60}")

    def key_callback(key: int) -> None:
        nonlocal current_ep, episodes
        if key == 256: return  # ESC - viewer handles exit
        elif key == 13:  # Enter
            if len(current_ep) > 10:
                episodes.append(current_ep)
                print(f"\n[✓ Episode {len(episodes)} 已保存 ({len(current_ep)} steps)]")
            current_ep = Episode()
            collector.reset()
        elif key == 8 or key == 127:  # Backspace
            current_ep = Episode()
            collector.reset()
            print("\n[✗ 已丢弃]")
        else:
            action = controller.get_action(key)
            if action is not None:
                ctrl = mj.get_ctrl()
                for i, n in enumerate(["ur_shoulder_pan_joint","ur_shoulder_lift_joint","ur_elbow_joint","ur_wrist_1_joint","ur_wrist_2_joint","ur_wrist_3_joint"]):
                    ctrl[mj.get_actuator_id(n+"_ACTUATOR")] = action[i]
                ctrl[mj.get_actuator_id("robotiq_85_left_knuckle_joint_ACTUATOR")] = float(action[6])
                mj.set_ctrl(ctrl)

    with mujoco.viewer.launch_passive(mj.model, mj.data, key_callback=key_callback) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = mj.model.camera("realsense_link_CAMERA").id

        while viewer.is_running():
            mj.step()
            camera.capture()
            obs = collector.collect(task_id=config["task_id"])
            action = np.array([controller.get_action(0)[6] if controller.get_action(0) is not None else 0.8])
            full_action = np.zeros(7); full_action[6] = action[0]
            current_ep.add(obs, full_action)
            viewer.sync()
            time.sleep(0.001)

    # 保存
    if episodes:
        ts = time.strftime("%Y%m%d_%H%M%S")
        saver = LeRobotSaver()
        saver.save(episodes, str(Path(args.output)/args.task/ts), f"ur5_{args.task}_keyboard")
    elif len(current_ep) > 10:
        ts = time.strftime("%Y%m%d_%H%M%S")
        saver = LeRobotSaver()
        saver.save([current_ep], str(Path(args.output)/args.task/ts), f"ur5_{args.task}_keyboard")

    collector.close()
    mj.close()
    print("\n✓ 完成")


if __name__ == "__main__":
    main()
