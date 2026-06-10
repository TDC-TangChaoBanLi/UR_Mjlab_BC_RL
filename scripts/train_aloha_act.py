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
    """
    从 LeRobotDataset 构造 ACT action chunk。

    每个样本返回：
        camera:       [4, H, W] RGBD 图像，只读取当前帧
        state:        [state_dim]
        action_chunk: [K, action_dim]
        is_pad:       [K]
        task:         标量 task id

    核心优化：
        1. __getitem__ 中只调用一次 self._lds[idx]
        2. action 和 state 提前缓存到 CPU 内存
        3. action_chunk 通过 tensor slicing 构造，不再循环读取 self._lds[t]
    """

    def __init__(
        self,
        root: str | Path,
        repo_id: str,
        chunk_size: int = 50,
        action_dim: int = 7,
        state_dim: int = 7,
        task_id: int = 0,
        cache_low_dim: bool = True,
    ) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self._lds = LeRobotDataset(
            repo_id=repo_id,
            root=str(Path(root).expanduser().resolve()),
        )

        self._len = int(self._lds.num_frames)
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.task_id = int(task_id)

        self._episode_boundaries = self._compute_episode_boundaries()
        self._episode_end_map = self._build_episode_end_map()

        self._arange_k = torch.arange(self.chunk_size)

        if cache_low_dim:
            self._actions = self._read_column_as_tensor("action")[:, : self.action_dim].contiguous()
            self._states = self._read_column_as_tensor("observation.state")[:, : self.state_dim].contiguous()
        else:
            raise ValueError("建议保持 cache_low_dim=True，否则会回到低效读取模式。")

    def _read_column_as_tensor(self, key: str) -> torch.Tensor:
        """
        从 LeRobotDataset 底层 HF dataset 中一次性读取低维列。
        action/state 很小，通常可以直接放入内存。
        """
        if not hasattr(self._lds, "hf_dataset"):
            raise AttributeError(
                "当前 LeRobotDataset 没有 hf_dataset 属性。"
                "需要根据你的 LeRobot 版本改成对应的底层 dataset 属性。"
            )

        col = self._lds.hf_dataset[key]

        try:
            arr = np.asarray(col, dtype=np.float32)
        except Exception:
            arr = np.stack(col).astype(np.float32)

        if arr.ndim == 1:
            arr = arr[:, None]

        return torch.from_numpy(arr)

    def _compute_episode_boundaries(self) -> list[int]:
        """
        计算每个 episode 的起始帧索引。
        返回形式：
            [0, ep0_end, ep1_end, ..., num_frames]
        """
        boundaries = [0]

        try:
            eps = self._lds.episodes
            if eps is not None:
                for ep in eps:
                    boundaries.append(boundaries[-1] + int(ep["length"]))
        except (AttributeError, KeyError, TypeError):
            pass

        if len(boundaries) == 1:
            boundaries.append(self._len)

        if boundaries[-1] != self._len:
            boundaries[-1] = self._len

        return boundaries

    def _build_episode_end_map(self) -> list[int]:
        """
        对每一帧预计算它所在 episode 的结束帧。
        这样 __getitem__ 中可以 O(1) 得到 ep_end。
        """
        end_map = [self._len] * self._len

        for i in range(len(self._episode_boundaries) - 1):
            start = self._episode_boundaries[i]
            end = self._episode_boundaries[i + 1]
            start = max(0, min(start, self._len))
            end = max(0, min(end, self._len))

            if start < end:
                end_map[start:end] = [end] * (end - start)

        return end_map

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # 只读取一次 LeRobotDataset。
        # 这里主要是为了拿当前帧图像。
        f = self._lds[idx]

        rgb = f["observation.images.rgb"]
        depth = f["observation.images.depth"]

        # rgb: [3, H, W]
        if rgb.dtype == torch.uint8:
            rgb = rgb.float().div_(255.0)
        else:
            rgb = rgb.to(torch.float32)

        # depth: 可能是 [H, W]、[1, H, W] 或 [C, H, W]
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        elif depth.ndim == 3:
            depth = depth[:1]
        else:
            raise ValueError(f"Unexpected depth shape: {depth.shape}")

        if depth.dtype == torch.uint8:
            depth = depth.float().div_(255.0)
        else:
            depth = depth.to(torch.float32)

        camera = torch.cat([rgb, depth], dim=0).contiguous()  # [4, H, W]

        # state 直接从缓存读取，不从 f 里再处理
        state = self._states[idx]

        # action chunk 直接切片，不再调用 self._lds[t]
        ep_end = self._episode_end_map[idx]
        valid_len = min(self.chunk_size, max(0, ep_end - idx))

        action_chunk = torch.zeros(
            self.chunk_size,
            self.action_dim,
            dtype=torch.float32,
        )

        if valid_len > 0:
            action_chunk[:valid_len] = self._actions[idx : idx + valid_len]

        is_pad = self._arange_k >= valid_len

        return {
            "camera": camera,
            "state": state,
            "action_chunk": action_chunk,
            "is_pad": is_pad,
            "task": torch.tensor(self.task_id, dtype=torch.long),
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
        num_workers=12, pin_memory=True, prefetch_factor=2,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch, shuffle=False,
        num_workers=6, pin_memory=True, prefetch_factor=2,
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

        # if (epoch + 1) % max(1, args.epochs // 20) == 0:
        if True:
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
