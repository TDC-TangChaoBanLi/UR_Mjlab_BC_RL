#!/usr/bin/env python3
"""行为克隆 (BC) 训练脚本。

从 LeRobotDataset 加载专家数据，训练 UR5MultimodalActor。

示例:
  python scripts/train_imitation.py \\
      --data outputs/datasets/expert/pick_place/20260603_174353/ur5_pick_place \\
      --epochs 100 --batch 64 --lr 1e-3
"""

from __future__ import annotations

import argparse
import sys
import time
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ur_mjlab_bc_rl.models.policy.multimodal_backbone import UR5MultimodalBackbone
from ur_mjlab_bc_rl.imitation.training import ImitationTrainer
from ur_mjlab_bc_rl.imitation.dataset import LeRobotRgbdTorchDataset
from ur_mjlab_bc_rl.config_loader import load_multimodal_model, print_model_config


def find_lerobot_root(data_path: Path) -> str:
    """在给定路径下查找 LeRobotDataset 根目录（包含 meta/info.json 的目录）。"""
    if (data_path / "meta" / "info.json").exists():
        return str(data_path)
    # 查找子目录
    for sub in sorted(data_path.iterdir()):
        if sub.is_dir() and (sub / "meta" / "info.json").exists():
            return str(sub)
    # 在更深层查找
    for sub in sorted(data_path.rglob("meta/info.json")):
        return str(sub.parent.parent)
    raise FileNotFoundError(f"未找到 LeRobotDataset（需要 meta/info.json）: {data_path}")


def main():
    parser = argparse.ArgumentParser(description="BC 模仿学习训练")
    parser.add_argument("--data", type=str, required=True, help="LeRobotDataset 路径")
    parser.add_argument("--task", type=str, default="pick_place", help="任务名称")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=64, help="批量大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--output", type=str, default="outputs/checkpoints", help="输出目录")
    parser.add_argument("--val-split", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "l1", "huber"], help="损失函数")
    parser.add_argument("--use-vit", action="store_true", help="是否使用 ViT")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复训练")
    parser.add_argument("--save-every", type=int, default=10, help="每 N 个 epoch 保存一次权重")
    parser.add_argument("--state-dropout", type=float, default=0.2, help="随机丢弃 state 的概率（0~1），强制网络依赖视觉")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_path = Path(args.data)
    output_dir = Path(args.output) / args.task / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 保存训练配置 ──
    model_cfg = load_multimodal_model()
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存模型配置
    with open(config_dir / "multimodal.yaml", "w") as f:
        yaml.dump(model_cfg, f, default_flow_style=False, sort_keys=False)
    
    # 复制相关配置文件
    config_files = [
        "model/multimodal.yaml",
        "model/imitation.yaml",
    ]
    src_config_dir = PROJECT_ROOT / "configs"
    for cfg_file in config_files:
        src_path = src_config_dir / cfg_file
        if src_path.exists():
            dst_path = config_dir / cfg_file.replace("/", "_")
            shutil.copy(src_path, dst_path)
    print(f"  ✓ 配置文件已保存到: {config_dir}")

    print(f"\n{'=' * 60}")
    print(f"BC 模仿学习训练")
    print(f"{'=' * 60}")
    print(f"数据: {data_path}")
    print(f"Epochs: {args.epochs}  Batch: {args.batch}  LR: {args.lr}")
    print(f"设备: {args.device}  ViT: {args.use_vit}")

    # ── 加载 LeRobotDataset ──
    print(f"\n[1/3] 加载数据集...")
    root = find_lerobot_root(data_path)
    repo_id = f"ur5_{args.task}"
    print(f"  LeRobot root: {root}")
    full_dataset = LeRobotRgbdTorchDataset(root, repo_id)
    print(f"  总帧数: {len(full_dataset)}")

    # 划分训练集和验证集
    n_val = max(1, int(len(full_dataset) * args.val_split))
    train_indices = list(range(len(full_dataset) - n_val))
    val_indices = list(range(len(full_dataset) - n_val, len(full_dataset)))
    print(f"  训练: {len(train_indices)}  验证: {len(val_indices)}")

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        num_workers=4, pin_memory=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch, shuffle=False,
        num_workers=2, pin_memory=True, prefetch_factor=2,
    )

    # ── 模型 ──
    print(f"\n[2/3] 创建模型...")
    actor = UR5MultimodalBackbone(model_cfg=model_cfg)
    actor.model_cfg = model_cfg
    
    # 打印模型配置
    print_model_config(model_cfg)

    start_epoch = 0
    if args.resume:
        print(f"  从 checkpoint 恢复: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        actor.load_state_dict(ckpt["actor_state_dict"])
        model_cfg = ckpt.get("model_cfg", model_cfg)
        start_epoch = ckpt.get("epoch", 0)
        print(f"  恢复 epoch: {start_epoch}")
    print(f"  模型参数总数: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"    观测编码器参数:")
    print(f"      视觉编码器参数: {sum(p.numel() for p in actor.visual_encoder.parameters()):,}")
    print(f"      状态编码器参数: {sum(p.numel() for p in actor.state_encoder.parameters()):,}")
    print(f"      任务编码器参数: {sum(p.numel() for p in actor.task_encoder.parameters()):,}")
    print(f"    特征融合器参数: {sum(p.numel() for p in actor.fusion.parameters()):,}")
    print(f"    策略 MLP 参数: {sum(p.numel() for p in actor.policy_mlp.parameters()):,}")


    # ── 训练 ──
    print(f"\n[3/3] 开始训练...")
    trainer = ImitationTrainer(
        actor=actor, train_loader=train_loader, val_loader=val_loader,
        learning_rate=args.lr, loss_type=args.loss, device=args.device,
        state_dropout=args.state_dropout,
    )
    stats = trainer.train(
        num_epochs=args.epochs, save_dir=output_dir, start_epoch=start_epoch,
        save_every=args.save_every,
        print_interval=max(1, args.epochs // 20),
    )

    final_path = output_dir / "best_actor.pt"
    trainer.save_actor(final_path)
    print(f"\n训练完成！最佳损失: {stats['best_val_loss']:.6f}")
    print(f"模型: {final_path}")


if __name__ == "__main__":
    main()