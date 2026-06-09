#!/usr/bin/env python3
"""Scripted Teacher 专家数据采集脚本。

无临时文件版本：
1. worker 进程只在内存中缓存当前成功 episode；
2. 成功后通过 multiprocessing.Queue 按 batch 发送给主进程；
3. 主进程唯一持有 LeRobotDataset writer，收到 frame batch 后立即写入；
4. 不写 memmap，不写临时 episode 文件，不在主进程累计 collected_episodes。
"""

from __future__ import annotations

import argparse
import gc
import logging
import multiprocessing as mp
import os
import queue
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ── 抑制 LeRobot / ffmpeg 日志 ─────────────────────────────
os.environ.setdefault("FFMPEG_LOGLEVEL", "error")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("SVT_LOG", "0")
for _name in ("lerobot", "datasets", "PIL", "torchvision", "ffmpeg", "av"):
    logging.getLogger(_name).setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ur_mjlab_bc_rl.config_loader import (  # noqa: E402
    get_arm_joints,
    get_camera_name,
    get_collection_params,
    get_gripper_joints,
    get_image_size,
    get_sim_params,
    load_tasks,
)
from ur_mjlab_bc_rl.imitation.dataset import (  # noqa: E402
    LeRobotDatasetConfig,
    LeRobotMujocoDatasetWriter,
)
from ur_mjlab_bc_rl.imitation.expert_generation import Episode  # noqa: E402

SIM_PARAMS = get_sim_params()
COLL_PARAMS = get_collection_params()
ARM_JOINT_NAMES = get_arm_joints()
GRIPPER_JOINT_NAMES = get_gripper_joints()
CAMERA_NAME = get_camera_name()
IMAGE_SIZE = get_image_size()

TEACHER_REGISTRY: dict[str, Any] = {}


def _init_teacher_registry() -> None:
    from ur_mjlab_bc_rl.imitation.expert_generation import (  # noqa: WPS433
        PegSlotTeacher,
        PickPlaceTeacher,
        PushTTeacher,
    )

    TEACHER_REGISTRY.update(
        {
            "PickPlaceTeacher": PickPlaceTeacher,
            "PushTTeacher": PushTTeacher,
            "PegSlotTeacher": PegSlotTeacher,
        }
    )


_init_teacher_registry()


def _collect_one_episode(task: dict[str, Any]) -> Episode | None:
    """在 worker 内采集一条成功 episode。"""
    worker_id = int(task["worker_id"])
    scene_path = str(task["scene_path"])
    teacher_name = str(task["teacher_name"])
    task_id = int(task["task_id"])
    max_steps = int(task["max_steps"])
    depth_range = task["depth_range"]
    enable_render = bool(task["enable_render"])

    np.random.seed((worker_id + 1) * 42 + int(time.time() * 1000) % 100000)

    from ur_mjlab_bc_rl.imitation.mujoco_env import (  # noqa: WPS433
        CameraSensor,
        MinkIK,
        MujocoInterface,
        ObservationCollector,
        ResetManager,
    )

    mj_interface = None
    obs_collector = None

    task_name_map = {0: "pick_place", 1: "push_t", 2: "peg_slot"}
    task_name = task_name_map[task_id]
    teacher_class = TEACHER_REGISTRY[teacher_name]

    try:
        mj_interface = MujocoInterface(scene_path, render=enable_render)
        if enable_render:
            mj_interface.set_viewer_camera(
                lookat=(0.45, 0.0, 0.65),
                distance=1.8,
                elevation=-25.0,
                azimuth=130.0,
            )

        camera = CameraSensor(mj_interface, CAMERA_NAME, IMAGE_SIZE)
        obs_collector = ObservationCollector(
            mj_interface,
            camera,
            ARM_JOINT_NAMES,
            GRIPPER_JOINT_NAMES,
            depth_range[0],
            depth_range[1],
            include_arm_vel=False,
            include_ee_pose=False,
            include_last_action=False,
            action_dim=7,
        )
        ik_solver = MinkIK(mj_interface.model, mj_interface.get_qpos())
        reset_manager = ResetManager(mj_interface)

        arm_actuator_ids = [
            mj_interface.get_actuator_id(name + "_ACTUATOR")
            for name in ARM_JOINT_NAMES
        ]
        gripper_actuator_ids = [
            mj_interface.get_actuator_id(name + "_ACTUATOR")
            for name in GRIPPER_JOINT_NAMES
        ]

        for _attempt_idx in range(int(COLL_PARAMS["max_attempts"])):
            reset_manager.reset(task=task_name, randomize_objects=True)
            teacher = teacher_class(mj_interface.model, mj_interface.data)
            teacher.reset()
            ik_solver.reset(mj_interface.get_qpos())
            obs_collector.reset()
            episode = Episode()
            mj_interface.sync_viewer()

            time_since_policy = 0.0
            time_since_camera = 0.0
            camera.capture()

            for _step_idx in range(max_steps):
                mj_interface.step()
                time_since_policy += SIM_PARAMS["physics_dt"]
                time_since_camera += SIM_PARAMS["physics_dt"]

                if enable_render and not mj_interface.is_viewer_running():
                    break

                if time_since_camera >= SIM_PARAMS["camera_dt"]:
                    camera.capture()
                    time_since_camera -= SIM_PARAMS["camera_dt"]

                if time_since_policy < SIM_PARAMS["policy_dt"]:
                    continue

                time_since_policy -= SIM_PARAMS["policy_dt"]

                ee_action = teacher.step()
                if len(ee_action) != 8:
                    raise ValueError(f"无效 Teacher 动作维度: {len(ee_action)}")

                target_pose = np.asarray(ee_action[:7], dtype=np.float32).copy()
                gripper_cmd = [float(ee_action[7])]

                current_joint_pos = mj_interface.get_joint_qpos(ARM_JOINT_NAMES)
                joint_target = ik_solver.solve_ik(
                    current_joint_pos,
                    target_pose,
                    dt=SIM_PARAMS["policy_dt"],
                )

                ctrl = mj_interface.get_ctrl()
                for idx, act_id in enumerate(arm_actuator_ids):
                    ctrl[act_id] = joint_target[idx]
                for idx, act_id in enumerate(gripper_actuator_ids):
                    ctrl[act_id] = gripper_cmd[idx]
                mj_interface.set_ctrl(ctrl)

                action = np.append(joint_target.copy(), gripper_cmd).astype(np.float32)
                obs = obs_collector.collect(task_id=task_id)
                episode.add(obs, action, copy_arrays=True)
                obs_collector.update_last_action(action)

                if teacher.is_done():
                    break

            if teacher.is_success() and len(episode) > 0:
                return episode

            episode.clear()
            gc.collect()

        return None

    finally:
        if obs_collector is not None:
            obs_collector.close()
        elif mj_interface is not None:
            mj_interface.close()
        gc.collect()


def _worker_loop(
    worker_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    send_lock,
    frame_chunk_size: int,
) -> None:
    """worker 主循环。

    send_lock 保证一个 episode 的 start/chunk/end 消息不会与其他 worker 交错，
    主进程可以顺序写入 LeRobot 当前 episode buffer。
    """
    try:
        while True:
            task = task_queue.get()
            if task is None:
                break

            task["worker_id"] = worker_id
            episode_idx = int(task["episode_idx"])
            task_label = str(task["task_label"])

            try:
                episode = _collect_one_episode(task)
            except Exception as exc:
                result_queue.put(
                    {
                        "kind": "failed",
                        "worker_id": worker_id,
                        "episode_idx": episode_idx,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            if episode is None:
                result_queue.put(
                    {
                        "kind": "failed",
                        "worker_id": worker_id,
                        "episode_idx": episode_idx,
                        "error": "max_attempts reached",
                    }
                )
                continue

            # 一个 episode 的消息连续发送，避免主进程 LeRobot 当前 episode buffer 混入其他 worker 的帧。
            with send_lock:
                result_queue.put(
                    {
                        "kind": "episode_start",
                        "worker_id": worker_id,
                        "episode_idx": episode_idx,
                        "task_label": task_label,
                        "num_frames": len(episode),
                    }
                )
                for observations, actions in episode.iter_batches(frame_chunk_size):
                    result_queue.put(
                        {
                            "kind": "episode_chunk",
                            "worker_id": worker_id,
                            "episode_idx": episode_idx,
                            "task_label": task_label,
                            "observations": observations,
                            "actions": actions,
                        }
                    )
                result_queue.put(
                    {
                        "kind": "episode_end",
                        "worker_id": worker_id,
                        "episode_idx": episode_idx,
                    }
                )

            episode.clear()
            gc.collect()

    finally:
        result_queue.put({"kind": "worker_done", "worker_id": worker_id})


def _build_writer(
    *,
    task_name: str,
    output_root: Path,
    merge_dir: str | None,
    overwrite: bool,
    encoder_threads: int,
    encoder_queue_maxsize: int,
) -> tuple[LeRobotMujocoDatasetWriter, Path, str]:
    H, W = IMAGE_SIZE
    repo_id = f"ur5_{task_name}"
    fps = int(round(1.0 / SIM_PARAMS["policy_dt"]))

    # 当前 ObservationCollector 默认字段：arm_joint_pos + gripper_pos + last_action。
    state_dim = len(ARM_JOINT_NAMES) + len(GRIPPER_JOINT_NAMES) + 7
    action_dim = 7

    if merge_dir:
        root = Path(merge_dir).expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = (output_root / task_name / timestamp).resolve()

    config = LeRobotDatasetConfig(
        repo_id=repo_id,
        root=root,
        fps=fps,
        state_dim=state_dim,
        action_dim=action_dim,
        image_height=int(H),
        image_width=int(W),
        state_keys=("arm_joint_pos", "gripper_pos", "last_action"),
        robot_type="mujoco_ur5",
        use_rgb=True,
        use_depth=True,
        streaming_encoding=True,
        batch_encoding_size=1,
        encoder_threads=encoder_threads,
        encoder_queue_maxsize=encoder_queue_maxsize,
        image_writer_threads=0,
        image_writer_processes=0,
    )

    if merge_dir:
        writer = LeRobotMujocoDatasetWriter.resume_existing(config)
    else:
        writer = LeRobotMujocoDatasetWriter.create_new(config, overwrite=overwrite)
    return writer, root, repo_id


def _drain_results(
    *,
    result_queue: mp.Queue,
    writer: LeRobotMujocoDatasetWriter,
    num_workers: int,
    target_episodes: int,
    checkpoint_every: int,
    result_timeout: float,
) -> tuple[int, int]:
    workers_done = 0
    collected = 0
    failed = 0
    active_task_label: str | None = None
    active_episode_idx: int | None = None

    while workers_done < num_workers:
        try:
            msg = result_queue.get(timeout=result_timeout)
        except queue.Empty:
            print("⚠ 等待 worker 结果超时，继续检查...")
            continue

        kind = msg.get("kind")

        if kind == "worker_done":
            workers_done += 1
            continue

        if kind == "failed":
            failed += 1
            print(
                f"⚠ Worker {msg['worker_id']} episode {msg['episode_idx']} 失败: {msg.get('error', '')}"
            )
            continue

        if kind == "episode_start":
            active_task_label = str(msg["task_label"])
            active_episode_idx = int(msg["episode_idx"])
            print(
                f"  ▶ Worker {msg['worker_id']} episode {active_episode_idx} 开始写入 "
                f"({msg['num_frames']} frames)"
            )
            continue

        if kind == "episode_chunk":
            writer.append_step_batch(
                msg["observations"],
                msg["actions"],
                task_label=str(msg["task_label"]),
            )
            # 显式删除消息中的大数组引用。
            msg.clear()
            gc.collect()
            continue

        if kind == "episode_end":
            writer.save_current_episode()
            collected += 1
            print(f"  ✓ 已写入 {collected}/{target_episodes} episodes")

            active_task_label = None
            active_episode_idx = None

            if checkpoint_every > 0 and collected % checkpoint_every == 0:
                writer.checkpoint()
                print(f"  💾 checkpoint: 已 finalize+resume，episodes={collected}")
            continue

        raise RuntimeError(f"未知 result_queue 消息: {kind!r}")

    return collected, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Scripted Teacher 数据采集，无临时文件流式写入 LeRobotDataset。")
    parser.add_argument("--task", type=str, required=True, choices=["pick_place", "push_t", "peg_slot", "all"])
    parser.add_argument("--episodes", type=int, default=50, help="每个任务计划采集的 episode 数量。")
    parser.add_argument("--output", type=str, default="outputs/datasets/expert")
    parser.add_argument("--max-steps", type=int, default=COLL_PARAMS["max_steps"])
    parser.add_argument("--no-render", action="store_true", default=False)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--merge", type=str, default=None, help="追加到已有 LeRobotDataset root。")
    parser.add_argument("--overwrite", action="store_true", help="创建新数据集时覆盖已有目录。")

    parser.add_argument("--frame-chunk-size", type=int, default=16, help="worker 每次通过 Queue 发送多少帧。越大，写入效率越高，但内存占用也越高。")
    parser.add_argument("--queue-size", type=int, default=4, help="result_queue 最大消息数，用于反压限制内存。越大，写入效率越高，但内存占用也越高。")
    parser.add_argument("--checkpoint-every", type=int, default=20, help="每 N 个 episode finalize+resume 一次；0 表示不自动 checkpoint。")
    parser.add_argument("--encoder-threads", type=int, default=4)
    parser.add_argument("--encoder-queue-maxsize", type=int, default=30)
    parser.add_argument("--result-timeout", type=float, default=120.0)
    args = parser.parse_args()

    output_root = Path(args.output)
    all_task_configs = load_tasks()
    tasks_to_run = list(all_task_configs.keys()) if args.task == "all" else [args.task]
    scene_dir = PROJECT_ROOT / "assets" / "mujoco" / "scenes"

    for task_name in tasks_to_run:
        task_config = all_task_configs.get(task_name)
        if task_config is None:
            print(f"⚠ 任务配置不存在: {task_name}")
            continue

        scene_path = scene_dir / task_config["scene"]
        if not scene_path.exists():
            print(f"⚠ 场景文件不存在: {scene_path}")
            continue

        print("\n" + "=" * 72)
        print(f"任务: {task_name}")
        print(f"Teacher: {task_config['teacher']}")
        print(f"目标 episodes: {args.episodes}")
        print(f"并行 workers: {args.num_workers}")
        print(f"渲染: {not args.no_render}")
        print("=" * 72)

        writer, save_root, repo_id = _build_writer(
            task_name=task_name,
            output_root=output_root,
            merge_dir=args.merge,
            overwrite=args.overwrite,
            encoder_threads=args.encoder_threads,
            encoder_queue_maxsize=args.encoder_queue_maxsize,
        )

        ctx = mp.get_context("spawn")
        task_queue: mp.Queue = ctx.Queue()
        result_queue: mp.Queue = ctx.Queue(maxsize=max(1, args.queue_size))
        send_lock = ctx.Lock()

        depth_range = task_config["depth_range"]
        for episode_idx in range(args.episodes):
            worker_id_hint = episode_idx % max(1, args.num_workers)
            task_queue.put(
                {
                    "episode_idx": episode_idx,
                    "scene_path": str(scene_path),
                    "teacher_name": task_config["teacher"],
                    "task_id": task_config["task_id"],
                    "task_label": task_name,
                    "max_steps": args.max_steps,
                    "depth_range": depth_range,
                    "enable_render": (not args.no_render) and (worker_id_hint == 0),
                }
            )

        num_workers = min(max(1, args.num_workers), args.episodes)
        for _ in range(num_workers):
            task_queue.put(None)

        processes: list[mp.Process] = []
        try:
            for worker_id in range(num_workers):
                p = ctx.Process(
                    target=_worker_loop,
                    args=(
                        worker_id,
                        task_queue,
                        result_queue,
                        send_lock,
                        max(1, args.frame_chunk_size),
                    ),
                    daemon=False,
                )
                p.start()
                processes.append(p)

            collected, failed = _drain_results(
                result_queue=result_queue,
                writer=writer,
                num_workers=num_workers,
                target_episodes=args.episodes,
                checkpoint_every=args.checkpoint_every,
                result_timeout=args.result_timeout,
            )

            for p in processes:
                p.join(timeout=5.0)

            result = writer.finalize()
            print(f"\n✓ 任务 {task_name} 完成")
            print(f"  成功写入: {collected}/{args.episodes}")
            print(f"  失败跳过: {failed}")
            print(f"  保存目录: {result}")
            print(f"  repo_id: {repo_id}")

        except KeyboardInterrupt:
            print("\n⚠ 用户中断，正在关闭 worker 和 writer...")
            for p in processes:
                if p.is_alive():
                    p.terminate()
            writer.discard_current_episode()
            writer.finalize()
            raise

        except Exception:
            for p in processes:
                if p.is_alive():
                    p.terminate()
            writer.discard_current_episode()
            writer.finalize()
            raise

    print("\n" + "=" * 72)
    print(f"全部完成，数据集根目录: {output_root}")
    print("=" * 72)


if __name__ == "__main__":
    main()