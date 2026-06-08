#!/usr/bin/env python3
"""修复 checkpoint 中损坏的 BatchNorm running stats。

问题：训练时 BN running_var 累积了极小值（~1e-5），导致 eval 模式下信号放大 250x，
     视觉编码器输出恒为定值。

修复：重置 BN running stats → 用训练数据在 train 模式下预热 → 保存修复后的 checkpoint。

使用:
  python scripts/fix_checkpoint_bn.py \\
      --checkpoint outputs/checkpoints/pick_place/20260606_235640/checkpoint_epoch_200.pt \\
      --data outputs/datasets/expert/pick_place/20260606_230941 \\
      --output outputs/checkpoints/pick_place/20260606_235640/checkpoint_epoch_200_fixed.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def find_lerobot_root(data_path: Path) -> str:
    """查找 LeRobotDataset 根目录。"""
    if (data_path / "meta" / "info.json").exists():
        return str(data_path)
    for sub in sorted(data_path.iterdir()):
        if sub.is_dir() and (sub / "meta" / "info.json").exists():
            return str(sub)
    for sub in sorted(data_path.rglob("meta/info.json")):
        return str(sub.parent.parent)
    raise FileNotFoundError(f"未找到 LeRobotDataset: {data_path}")


def reset_bn_stats(module: nn.Module) -> int:
    """重置模块中所有 BatchNorm 层的 running stats。"""
    count = 0
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.reset_running_stats()
            # 使用更大的 momentum 加速预热
            m.momentum = 0.5
            count += 1
    return count


def warmup_bn(
    visual_encoder: nn.Module,
    dataloader: DataLoader,
    num_batches: int = 100,
    device: str = "cpu",
) -> None:
    """用训练数据预热 BN running stats。

    Args:
        visual_encoder: 视觉编码器模块
        dataloader: 训练数据加载器
        num_batches: 预热批次数
        device: 计算设备
    """
    visual_encoder.to(device)
    visual_encoder.train()  # BN 使用 batch stats 并更新 running stats

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            camera = batch["camera"].to(device)
            _ = visual_encoder(camera)

            if (i + 1) % 20 == 0:
                # 打印进度和 BN 统计值
                for m in visual_encoder.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        rv = m.running_var.mean().item()
                        print(f"  batch {i + 1}: first BN running_var mean = {rv:.4f}")
                        break

    visual_encoder.eval()
    print(f"  预热完成：{min(num_batches, i + 1)} batches")


def main():
    parser = argparse.ArgumentParser(description="修复 checkpoint BN running stats")
    parser.add_argument("--checkpoint", type=str, required=True, help="输入 checkpoint 路径")
    parser.add_argument("--data", type=str, required=True, help="训练数据目录（LeRobotDataset）")
    parser.add_argument("--output", type=str, default=None, help="输出 checkpoint 路径（默认在原名后加 _fixed）")
    parser.add_argument("--num-batches", type=int, default=100, help="预热批次数")
    parser.add_argument("--batch-size", type=int, default=8, help="预热批量大小")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"✗ Checkpoint 不存在: {ckpt_path}")
        sys.exit(1)

    data_path = Path(args.data)
    root = find_lerobot_root(data_path)
    print(f"\n{'=' * 60}")
    print(f"修复 BN Running Stats")
    print(f"{'=' * 60}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  数据:       {root}")
    print(f"  设备:       {args.device}")
    print(f"  批量/批数:  {args.batch_size}/{args.num_batches}")

    # ── 加载 checkpoint ──
    print(f"\n[1/4] 加载 checkpoint...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt.get("model_cfg")

    from ur_mjlab_bc_rl.models.policy.multimodal_backbone import UR5MultimodalBackbone
    actor = UR5MultimodalBackbone(model_cfg=model_cfg)
    actor.load_state_dict(ckpt["actor_state_dict"])

    # ── 加载数据 ──
    print(f"\n[2/4] 加载数据...")
    from ur_mjlab_bc_rl.imitation.dataset import LeRobotRgbdTorchDataset

    # 推断 repo_id
    repo_id = root.rstrip("/").split("/")[-1]
    if not repo_id or repo_id == "pick_place":
        repo_id = "ur5_pick_place"

    dataset = LeRobotRgbdTorchDataset(root, repo_id)
    print(f"  总帧数: {len(dataset)}")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=False,
    )

    # ── 重置 BN ──
    print(f"\n[3/4] 重置 BN running stats...")
    bn_count = reset_bn_stats(actor.visual_encoder)
    print(f"  重置了 {bn_count} 个 BatchNorm 层")

    # ── 预热 ──
    print(f"\n[4/4] 预热 BN...")
    warmup_bn(actor.visual_encoder, loader, num_batches=args.num_batches, device=args.device)
    actor.to("cpu")

    # ── 验证 ──
    print(f"\n{'=' * 60}")
    print(f"验证修复效果")
    print(f"{'=' * 60}")
    actor.eval()
    torch.manual_seed(42)
    with torch.no_grad():
        x1 = torch.rand(2, 4, 240, 320)  # [0,1] 模拟实数据
        x2 = torch.rand(2, 4, 240, 320)  # 不同的 [0,1] 输入
        v1 = actor.visual_encoder(x1).vector
        v2 = actor.visual_encoder(x2).vector
        varying = not torch.allclose(v1, v2, atol=1e-6)
        print(f"  视觉编码器输出变化: {'✓ 是' if varying else '✗ 否'}")
        print(f"  最大差异: {(v1 - v2).abs().max():.6f}")

    # ── 保存 ──
    output_path = Path(args.output) if args.output else ckpt_path.with_name(ckpt_path.stem + "_fixed" + ckpt_path.suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    new_ckpt = {
        "actor_state_dict": actor.state_dict(),
        "model_cfg": model_cfg,
    }
    torch.save(new_ckpt, output_path)
    print(f"\n✓ 修复后的 checkpoint 已保存到: {output_path}")
    print(f"\n使用方式:")
    print(f"  python scripts/eval_policy.py --task pick_place --checkpoint {output_path}")


if __name__ == "__main__":
    main()
