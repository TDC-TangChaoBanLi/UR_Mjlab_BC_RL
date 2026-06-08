#!/usr/bin/env python3
"""PPO 强化学习微调训练脚本。

在 BC 预训练基础上使用 MjLab PPO 进行微调。

使用方式:
  # 从零开始训练
  python scripts/train_ppo_finetune.py --task UR5-PickPlace --headless

  # 从 BC checkpoint 初始化后训练
  python scripts/train_ppo_finetune.py --task UR5-PickPlace \\
      --bc-checkpoint outputs/checkpoints/pick_place/best_actor.pt \\
      --headless

  # 多任务训练
  python scripts/train_ppo_finetune.py --task UR5-PickPlace --num-envs 64 \\
      --total-steps 5000000 --headless

MjLab 内部使用 RSL-RL 进行 PPO 训练。
本脚本提供便捷的命令行接口和 BC checkpoint 集成。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


TASK_MAP = {
    "pick_place": "UR5-PickPlace",
    "push_t": "UR5-PushT",
    "peg_slot": "UR5-PegInSlot",
    "UR5-PickPlace": "UR5-PickPlace",
    "UR5-PushT": "UR5-PushT",
    "UR5-PegInSlot": "UR5-PegInSlot",
}


def check_bc_checkpoint(path: str | Path) -> dict | None:
    """检查 BC checkpoint 是否存在并有效。"""
    from ur_mjlab_bc_rl.reinforcement import inspect_bc_checkpoint

    path = Path(path)
    if not path.exists():
        print(f"⚠ BC checkpoint 不存在: {path}")
        return None

    try:
        info = inspect_bc_checkpoint(path)
        print(f"✓ BC checkpoint: {path.name}")
        print(f"  - 参数数量: {info.get('num_params', 'N/A'):,}")
        print(f"  - 文件大小: {info.get('file_size_mb', 0):.1f} MB")
        return info
    except Exception as e:
        print(f"⚠ 无法读取 BC checkpoint: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="PPO 强化学习微调训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --task UR5-PickPlace --headless
  %(prog)s --task pick_place --bc-checkpoint outputs/checkpoints/bc_best.pt --headless
  %(prog)s --task UR5-PushT --num-envs 32 --total-steps 2000000 --headless
        """,
    )
    parser.add_argument("--task", type=str, required=True,
                        help="任务 ID（UR5-PickPlace, UR5-PushT, UR5-PegInSlot 或简写）")
    parser.add_argument("--bc-checkpoint", type=str, default=None,
                        help="BC 预训练 checkpoint 路径")
    parser.add_argument("--num-envs", type=int, default=16,
                        help="并行环境数")
    parser.add_argument("--total-steps", type=int, default=5000000,
                        help="总训练步数")
    parser.add_argument("--headless", action="store_true",
                        help="无头模式（不显示 viewer）")
    parser.add_argument("--log-dir", type=str, default="logs/rsl_rl",
                        help="日志目录")
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的 checkpoint 路径")
    args = parser.parse_args()

    # 解析任务 ID
    task_id = TASK_MAP.get(args.task, args.task)

    # 检查 BC checkpoint
    if args.bc_checkpoint:
        info = check_bc_checkpoint(args.bc_checkpoint)
        if info is None:
            print("继续从零开始训练...")

    # 构建 mjlab train 命令
    print(f"\n{'=' * 60}")
    print(f"PPO 训练: {task_id}")
    print(f"{'=' * 60}")
    print(f"  并行环境数: {args.num_envs}")
    print(f"  总步数: {args.total_steps:,}")
    print(f"  无头模式: {args.headless}")
    print(f"  日志目录: {args.log_dir}")
    if args.resume:
        print(f"  恢复: {args.resume}")
    print(f"{'=' * 60}\n")

    # MjLab train 命令
    cmd = [
        sys.executable, "-m", "mjlab", "train", task_id,
        "--num-envs", str(args.num_envs),
        "--max-iterations", str(args.total_steps // (args.num_envs * 24)),  # ~24 steps per env per iter
    ]

    if args.headless:
        cmd.append("--headless")

    if args.resume:
        cmd.extend(["--resume", args.resume])

    if args.bc_checkpoint:
        cmd.extend(["--checkpoint-file", args.bc_checkpoint])

    print(f"执行: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 训练失败 (exit code: {e.returncode})")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n训练已中断")


if __name__ == "__main__":
    main()
