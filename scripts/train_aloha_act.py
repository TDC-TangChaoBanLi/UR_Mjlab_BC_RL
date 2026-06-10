#!/usr/bin/env python3
"""ALOHA ACT 训练脚本。

从 LeRobotDataset 加载专家数据，训练 DETRVAE 策略网络。

训练超参通过命令行指定，模型架构通过 --config 指定。
运行时自动将模型配置 + 训练参数写入输出目录。

示例:
  python scripts/train_aloha_act.py \\
      --data outputs/datasets/expert/pick_place/20260606_193958/ \\
      --epochs 200 --batch 32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ur_mjlab_bc_rl.models.policy.aloha_act_backbone import DETRVAE, build_detr_vae


# ═══════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════

def find_lerobot_root(data_path: Path) -> str:
    """在给定路径下查找 LeRobotDataset 根目录。"""
    if (data_path / "meta" / "info.json").exists():
        return str(data_path)
    for sub in sorted(data_path.iterdir()):
        if sub.is_dir() and (sub / "meta" / "info.json").exists():
            return str(sub)
    for sub in sorted(data_path.rglob("meta/info.json")):
        return str(sub.parent.parent)
    raise FileNotFoundError(f"未找到 LeRobotDataset（需要 meta/info.json）: {data_path}")


class ChunkedLeRobotDataset(Dataset):
    """从 LeRobotDataset 构造 action chunk。

    每个样本返回：
        camera:      [4, H, W] RGBD 图像
        state:       [state_dim] 仅取前 state_dim 维（默认 7，丢弃 last_action）
        action_chunk: [K, action_dim] 连续 K 帧动作
        is_pad:      [K] 标记哪些位置是 padding
    """

    def __init__(
        self,
        root: str | Path,
        repo_id: str,
        chunk_size: int = 10,
        action_dim: int = 7,
        state_dim: int = 7,
        task_id: int = 0,
    ) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self._lds = LeRobotDataset(repo_id=repo_id, root=str(Path(root).expanduser().resolve()))
        self._len = int(self._lds.num_frames)
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.task_id = int(task_id)

        # 获取 episode 边界（用于正确处理 chunk 边界）
        self._episode_boundaries = self._compute_episode_boundaries()
        self._episode_end_map = self._build_episode_end_map()  # O(1) 查找

    def _compute_episode_boundaries(self) -> list[int]:
        """计算每个 episode 的起始帧索引。"""
        boundaries = [0]
        try:
            eps = self._lds.episodes
            if eps is not None:
                for ep in eps:
                    boundaries.append(boundaries[-1] + ep["length"])
        except (AttributeError, KeyError):
            pass
        if len(boundaries) == 1:
            boundaries.append(self._len)
        return boundaries

    def _build_episode_end_map(self) -> list[int]:
        """预计算每帧对应的 episode 结束帧索引。 O(N) 一次性构建。"""
        end_map = [0] * self._len
        for i in range(len(self._episode_boundaries) - 1):
            start = self._episode_boundaries[i]
            end = self._episode_boundaries[i + 1]
            for j in range(start, min(end, self._len)):
                end_map[j] = end
        return end_map

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        f = self._lds[idx]

        # 图像
        rgb = f["observation.images.rgb"].float()
        depth = f["observation.images.depth"].float()
        if depth.ndim == 3:
            depth = depth[:1]
        camera = torch.cat([rgb, depth], dim=0)  # [4, H, W]

        # 状态：仅取前 state_dim 维（丢弃 last_action）
        full_state = f["observation.state"].float()  # [14]
        state = full_state[:self.state_dim]           # [7]

        # Action chunk — O(1) episode 边界查找
        action_chunk = torch.zeros(self.chunk_size, self.action_dim)
        is_pad = torch.ones(self.chunk_size, dtype=torch.bool)
        ep_end = self._episode_end_map[idx] if idx < len(self._episode_end_map) else self._len

        for k in range(self.chunk_size):
            t = idx + k
            if t < ep_end:
                f_k = self._lds[t]
                action_chunk[k] = f_k["action"].float()
                is_pad[k] = False

        return {
            "camera": camera,
            "state": state,
            "action_chunk": action_chunk,
            "is_pad": is_pad,
            "task": torch.tensor([self.task_id], dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════
# 训练循环
# ═══════════════════════════════════════════════════════════

def train_epoch(
    model: DETRVAE,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    kl_weight: float,
    device: str,
) -> float:
    """训练一个 epoch。"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        qpos = batch["state"].to(device)         # [B, 7]
        image = batch["camera"].to(device)       # [B, 4, H, W]
        action_chunk = batch["action_chunk"].to(device)  # [B, K, 7]
        is_pad = batch["is_pad"].to(device)      # [B, K]

        # Forward
        pred_chunk, mu, logvar = model(qpos, image, action_chunk, is_pad)

        # Loss
        loss_dict = model.compute_loss(pred_chunk, action_chunk, mu, logvar, is_pad, kl_weight)
        loss = loss_dict["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def validate(
    model: DETRVAE,
    val_loader: DataLoader,
    kl_weight: float,
    device: str,
) -> dict[str, float]:
    """验证。"""
    model.eval()
    total_l1 = 0.0
    total_kl = 0.0
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            qpos = batch["state"].to(device)
            image = batch["camera"].to(device)
            action_chunk = batch["action_chunk"].to(device)
            is_pad = batch["is_pad"].to(device)

            pred_chunk, mu, logvar = model(qpos, image, action_chunk, is_pad)
            loss_dict = model.compute_loss(pred_chunk, action_chunk, mu, logvar, is_pad, kl_weight)

            total_l1 += loss_dict["l1"].item()
            total_kl += loss_dict["kl"].item()
            total_loss += loss_dict["loss"].item()
            num_batches += 1

    n = max(num_batches, 1)
    return {"l1": total_l1 / n, "kl": total_kl / n, "loss": total_loss / n}


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ALOHA ACT 训练")
    parser.add_argument("--data", type=str, required=True, help="LeRobotDataset 路径")
    parser.add_argument("--task", type=str, default="pick_place", help="任务名称")
    parser.add_argument("--config", type=str, default="configs/model/aloha_act.yaml",
                        help="模型配置文件路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=8, help="批量大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--lr-backbone", type=float, default=1e-5, help="Backbone 学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--kl-weight", type=float, default=None, help="KL 散度权重（覆盖配置）")
    parser.add_argument("--chunk-size", type=int, default=None, help="动作分块大小 K（覆盖配置）")
    parser.add_argument("--state-dim", type=int, default=7, help="状态维度")
    parser.add_argument("--output", type=str, default="outputs/checkpoints", help="输出目录")
    parser.add_argument("--val-split", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复训练")
    parser.add_argument("--save-every", type=int, default=10, help="每 N epoch 保存")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_path = Path(args.data)
    output_dir = Path(args.output) / args.task / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载模型配置 ──
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"模型配置文件不存在: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # CLI 覆盖
    if args.chunk_size is not None:
        cfg["chunk_size"] = args.chunk_size
    if args.kl_weight is not None:
        cfg["kl_weight"] = args.kl_weight
    cfg["state_encoder"]["input_dim"] = args.state_dim

    # 保存配置到输出目录
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_dir / "aloha_act.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    # 保存训练参数
    train_args = {k: v for k, v in vars(args).items()
                  if k not in ("data", "task", "config", "output", "device", "resume")}
    with open(config_dir / "training_args.json", "w") as f:
        json.dump(train_args, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"ALOHA ACT 训练")
    print(f"{'=' * 60}")
    print(f"数据: {data_path}")
    print(f"模型配置: {config_path}")
    print(f"Epochs: {args.epochs}  Batch: {args.batch}  LR: {args.lr}")
    print(f"Chunk size: {cfg['chunk_size']}  KL weight: {cfg['kl_weight']}")
    print(f"设备: {args.device}")

    # ── 加载数据集 ──
    print(f"\n[1/3] 加载数据集...")
    root = find_lerobot_root(data_path)
    repo_id = f"ur5_{args.task}"
    print(f"  LeRobot root: {root}")

    full_dataset = ChunkedLeRobotDataset(
        root, repo_id,
        chunk_size=cfg["chunk_size"],
        action_dim=cfg["action_dim"],
        state_dim=args.state_dim,
    )
    print(f"  总帧数: {len(full_dataset)}")

    n_val = max(1, int(len(full_dataset) * args.val_split))
    train_indices = list(range(len(full_dataset) - n_val))
    val_indices = list(range(len(full_dataset) - n_val, len(full_dataset)))
    print(f"  训练: {len(train_indices)}  验证: {len(val_indices)}")

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        num_workers=16, pin_memory=True, prefetch_factor=2,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch, shuffle=False,
        num_workers=8, pin_memory=True, prefetch_factor=2,
        persistent_workers=True,
    )

    # ── 模型 ──
    print(f"\n[2/3] 创建模型...")
    model = build_detr_vae(cfg)
    model = model.to(args.device)

    start_epoch = 0
    best_val_loss = float("inf")
    best_state_dict = None

    if args.resume:
        print(f"  从 checkpoint 恢复: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  恢复 epoch: {start_epoch}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  可训练参数: {n_params:,} ({n_params / 1e6:.2f}M)")
    print(f"    CVAE Encoder:")
    print(f"      action sequence linear 参数: {sum(p.numel() for p in model.encoder_action_proj.parameters() if p.requires_grad):,}")
    print(f"      transformer 参数: {sum(p.numel() for p in model.cvae_encoder.parameters() if p.requires_grad):,}")
    print(f"    CVAE Decoder:")
    print(f"      visual encoder 参数: {sum(p.numel() for p in model.visual_encoder.parameters() if p.requires_grad):,}")
    print(f"      transformer encoder 参数: {sum(p.numel() for p in model.dec_encoder.parameters() if p.requires_grad):,}")
    print(f"      transformer decoder 参数: {sum(p.numel() for p in model.dec_decoder.parameters() if p.requires_grad):,}")

    # ── 优化器（与 ACT 一致：AdamW + 分离 backbone lr）──
    backbone_params = []
    other_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "visual_encoder" in n:
            backbone_params.append(p)
        else:
            other_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": other_params, "lr": args.lr},
        {"params": backbone_params, "lr": args.lr_backbone},
    ], weight_decay=args.weight_decay)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # ── 训练 ──
    print(f"\n[3/3] 开始训练...")
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, cfg["kl_weight"], args.device)
        val_metrics = validate(model, val_loader, cfg["kl_weight"], args.device)

        if (epoch + 1) % max(1, args.epochs // 20) == 0:
            print(f"Epoch {epoch + 1}/{start_epoch + args.epochs}: "
                  f"train_loss={train_loss:.6f}  "
                  f"val_l1={val_metrics['l1']:.6f}  "
                  f"val_kl={val_metrics['kl']:.6f}  "
                  f"val_loss={val_metrics['loss']:.6f}")

        # 保存最佳
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # 定期保存
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save({
                "model_type": "act",
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_cfg": cfg,
                "best_val_loss": best_val_loss,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    # 恢复最佳并保存
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    final_path = output_dir / "best_actor.pt"
    torch.save({
        "model_type": "act",
        "model_state_dict": model.state_dict(),
        "model_cfg": cfg,
        "best_val_loss": best_val_loss,
    }, final_path)
    print(f"\n训练完成！最佳验证损失: {best_val_loss:.6f}")
    print(f"模型: {final_path}")


if __name__ == "__main__":
    main()
