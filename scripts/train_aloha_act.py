#!/usr/bin/env python3
"""ALOHA ACT 训练脚本，支持单卡 / 双卡 / 多卡 DDP 训练。

从 LeRobotDataset 加载专家数据，训练 DETRVAE 策略网络。

单卡示例:
  python scripts/train_aloha_act.py \
      --data outputs/datasets/expert/pick_place/20260606_193958/ \
      --epochs 200 --batch 32

双卡示例:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 scripts/train_aloha_act.py \
      --data outputs/datasets/expert/pick_place/20260606_193958/ \
      --epochs 200 --batch 16 --num-workers 6

注意:
  DDP 下 --batch 表示每张 GPU 的 batch size。
  例如双卡 --batch 16，则全局 batch size = 16 * 2 = 32。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ur_mjlab_bc_rl.models.policy.aloha_act_backbone import DETRVAE, build_detr_vae


# 单机双卡默认不使用 IB/RoCE，避免 NCCL 误选 irdma/RoCE 网卡
os.environ.setdefault("NCCL_IB_DISABLE", "1")

# 你的日志里 cuMem host 有 warning，建议默认关掉
os.environ.setdefault("NCCL_CUMEM_HOST_ENABLE", "0")



# ═══════════════════════════════════════════════════════════
# DDP 工具函数
# ═══════════════════════════════════════════════════════════
def setup_distributed(args):
    """
    torchrun 启动时会自动设置：
    RANK, WORLD_SIZE, LOCAL_RANK
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.distributed = True
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(args.local_rank)
        args.device = torch.device(f"cuda:{args.local_rank}")

        # PyTorch 新版本支持 device_id，可以减少 NCCL 初始化时的设备猜测警告
        try:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                device_id=args.device,
            )
        except TypeError:
            # 兼容旧版本 PyTorch
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
            )

        # 这里可以直接不 barrier
        # 如果你确实想同步，则必须指定当前 rank 对应的 GPU
        # dist.barrier(device_ids=[args.local_rank])

    else:
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.device = torch.device(args.device)



def cleanup_distributed(args=None):
    if dist.is_available() and dist.is_initialized():
        if args is not None and getattr(args, "distributed", False):
            dist.barrier(device_ids=[args.local_rank])
        dist.destroy_process_group()


def is_main_process(args: argparse.Namespace) -> bool:
    """是否为主进程。只有主进程负责打印和保存文件。"""
    return (not getattr(args, "distributed", False)) or args.rank == 0


def rank0_print(args: argparse.Namespace, *values, **kwargs) -> None:
    """只在 rank0 打印。"""
    if is_main_process(args):
        print(*values, **kwargs)


def broadcast_object_from_rank0(obj, args: argparse.Namespace):
    """从 rank0 广播一个 Python 对象到所有进程。"""
    if not getattr(args, "distributed", False):
        return obj

    obj_list = [obj if args.rank == 0 else None]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]


def reduce_scalar(value: float, device: torch.device) -> float:
    """对所有进程上的标量取平均。"""
    if not (dist.is_available() and dist.is_initialized()):
        return float(value)

    t = torch.tensor(float(value), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())


def reduce_metrics(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    """对所有进程上的指标字典取平均。"""
    if not (dist.is_available() and dist.is_initialized()):
        return metrics

    return {k: reduce_scalar(v, device) for k, v in metrics.items()}


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """兼容 DDP 保存的 module.xxx 权重名。"""
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """恢复优化器后，把 optimizer state 中的 tensor 移到当前 GPU。"""
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def make_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    sampler,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """构造 DataLoader，兼容 num_workers=0 的情况。"""
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = True
    else:
        kwargs["persistent_workers"] = False

    return DataLoader(**kwargs)


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
            action_chunk[:valid_len] = self._actions[idx: idx + valid_len]

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
    model: DETRVAE | DDP,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    kl_weight: float,
    device: torch.device,
) -> float:
    """训练一个 epoch。"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        qpos = batch["state"].to(device, non_blocking=True)                # [B, 7]
        image = batch["camera"].to(device, non_blocking=True)              # [B, 4, H, W]
        action_chunk = batch["action_chunk"].to(device, non_blocking=True) # [B, K, 7]
        is_pad = batch["is_pad"].to(device, non_blocking=True)             # [B, K]

        # Forward
        pred_chunk, mu, logvar = model(qpos, image, action_chunk, is_pad)

        # Loss
        raw_model = model.module if isinstance(model, DDP) else model
        loss_dict = raw_model.compute_loss(
            pred_chunk,
            action_chunk,
            mu,
            logvar,
            is_pad,
            kl_weight,
        )
        loss = loss_dict["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: DETRVAE | DDP,
    val_loader: DataLoader,
    kl_weight: float,
    device: torch.device,
) -> dict[str, float]:
    """验证。"""
    model.eval()
    total_l1 = 0.0
    total_kl = 0.0
    total_loss = 0.0
    num_batches = 0

    raw_model = model.module if isinstance(model, DDP) else model

    for batch in val_loader:
        qpos = batch["state"].to(device, non_blocking=True)
        image = batch["camera"].to(device, non_blocking=True)
        action_chunk = batch["action_chunk"].to(device, non_blocking=True)
        is_pad = batch["is_pad"].to(device, non_blocking=True)

        pred_chunk, mu, logvar = model(qpos, image, action_chunk, is_pad)
        loss_dict = raw_model.compute_loss(
            pred_chunk,
            action_chunk,
            mu,
            logvar,
            is_pad,
            kl_weight,
        )

        total_l1 += float(loss_dict["l1"].item())
        total_kl += float(loss_dict["kl"].item())
        total_loss += float(loss_dict["loss"].item())
        num_batches += 1

    n = max(num_batches, 1)
    return {
        "l1": total_l1 / n,
        "kl": total_kl / n,
        "loss": total_loss / n,
    }


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="ALOHA ACT 训练")
    parser.add_argument("--data", type=str, required=True, help="LeRobotDataset 路径")
    parser.add_argument("--task", type=str, default="pick_place", help="任务名称")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model/aloha_act.yaml",
        help="模型配置文件路径",
    )
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=8, help="每张 GPU 的 batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--lr-backbone", type=float, default=1e-5, help="Backbone 学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--kl-weight", type=float, default=None, help="KL 散度权重（覆盖配置）")
    parser.add_argument("--chunk-size", type=int, default=None, help="动作分块大小 K（覆盖配置）")
    parser.add_argument("--state-dim", type=int, default=7, help="状态维度")
    parser.add_argument("--output", type=str, default="outputs/checkpoints", help="输出目录")
    parser.add_argument("--val-split", type=float, default=0.1, help="验证集比例")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="训练设备。DDP 模式下会被 LOCAL_RANK 自动覆盖。",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复训练")
    parser.add_argument("--save-every", type=int, default=10, help="每 N epoch 保存")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=6,
        help="每个 GPU 进程的 DataLoader worker 数。双卡时总 worker 数约为 num_workers * 2。",
    )
    parser.add_argument(
        "--find-unused-parameters",
        action="store_true",
        help="如果 DDP 报 unused parameters 错误，再开启此选项。默认关闭以获得更好性能。",
    )
    args = parser.parse_args()

    setup_distributed(args)

    try:
        # 每个 rank 用不同随机种子，避免 DataLoader / dropout 完全一致。
        torch.manual_seed(args.seed + args.rank)
        np.random.seed(args.seed + args.rank)
        if args.device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + args.rank)

        data_path = Path(args.data)

        # rank0 生成 run_name 后广播，避免多进程跨秒导致输出目录不一致。
        run_name = time.strftime("%Y%m%d_%H%M%S") if is_main_process(args) else None
        run_name = broadcast_object_from_rank0(run_name, args)
        output_dir = Path(args.output) / args.task / run_name

        if is_main_process(args):
            output_dir.mkdir(parents=True, exist_ok=True)

        if args.distributed:
            dist.barrier(device_ids=[args.local_rank])

        # ── 加载模型配置 ──
        config_path = PROJECT_ROOT / args.config
        if not config_path.exists():
            raise FileNotFoundError(f"模型配置文件不存在: {config_path}")
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # CLI 覆盖
        if args.chunk_size is not None:
            cfg["chunk_size"] = args.chunk_size
        if args.kl_weight is not None:
            cfg["kl_weight"] = args.kl_weight
        cfg["state_encoder"]["input_dim"] = args.state_dim

        # 保存配置到输出目录，只让 rank0 执行。
        if is_main_process(args):
            config_dir = output_dir / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)

            with open(config_dir / "aloha_act.yaml", "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

            train_args = {
                k: str(v) if isinstance(v, torch.device) else v
                for k, v in vars(args).items()
                if k not in ("data", "task", "config", "output", "resume")
            }
            with open(config_dir / "training_args.json", "w", encoding="utf-8") as f:
                json.dump(train_args, f, indent=2, default=str, ensure_ascii=False)

        rank0_print(args, f"\n{'=' * 60}")
        rank0_print(args, "ALOHA ACT 训练")
        rank0_print(args, f"{'=' * 60}")
        rank0_print(args, f"数据: {data_path}")
        rank0_print(args, f"模型配置: {config_path}")
        rank0_print(args, f"Epochs: {args.epochs}  Per-GPU Batch: {args.batch}  LR: {args.lr}")
        rank0_print(args, f"World size: {args.world_size}  Global Batch: {args.batch * args.world_size}")
        rank0_print(args, f"Chunk size: {cfg['chunk_size']}  KL weight: {cfg['kl_weight']}")
        rank0_print(args, f"设备: {args.device}")

        # ── 加载数据集 ──
        rank0_print(args, "\n[1/3] 加载数据集...")
        root = find_lerobot_root(data_path)
        repo_id = f"ur5_{args.task}"
        rank0_print(args, f"  LeRobot root: {root}")

        full_dataset = ChunkedLeRobotDataset(
            root,
            repo_id,
            chunk_size=cfg["chunk_size"],
            action_dim=cfg["action_dim"],
            state_dim=args.state_dim,
        )
        rank0_print(args, f"  总帧数: {len(full_dataset)}")

        n_val = max(1, int(len(full_dataset) * args.val_split))
        train_indices = list(range(len(full_dataset) - n_val))
        val_indices = list(range(len(full_dataset) - n_val, len(full_dataset)))
        rank0_print(args, f"  训练: {len(train_indices)}  验证: {len(val_indices)}")

        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

        if args.distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=True,
                drop_last=False,
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=False,
                drop_last=False,
            )
        else:
            train_sampler = None
            val_sampler = None

        pin_memory = args.device.type == "cuda"
        val_num_workers = 0 if args.num_workers == 0 else max(1, args.num_workers // 2)

        train_loader = make_dataloader(
            train_dataset,
            batch_size=args.batch,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        val_loader = make_dataloader(
            val_dataset,
            batch_size=args.batch,
            shuffle=False,
            sampler=val_sampler,
            num_workers=val_num_workers,
            pin_memory=pin_memory,
        )

        # ── 模型 ──
        rank0_print(args, "\n[2/3] 创建模型...")
        model = build_detr_vae(cfg)
        model = model.to(args.device)

        start_epoch = 0
        best_val_loss = float("inf")
        best_state_dict = None

        # 恢复模型参数。注意：要在 DDP 包装之前 load_state_dict。
        if args.resume:
            rank0_print(args, f"  从 checkpoint 恢复: {args.resume}")
            ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
            state_dict = strip_module_prefix(ckpt["model_state_dict"])
            model.load_state_dict(state_dict)
            start_epoch = ckpt.get("epoch", 0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            rank0_print(args, f"  恢复 epoch: {start_epoch}")

        # 包装 DDP。
        if args.distributed:
            model = DDP(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=args.find_unused_parameters,
            )

        raw_model = model.module if args.distributed else model

        n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        if is_main_process(args):
            print(f"  可训练参数: {n_params:,} ({n_params / 1e6:.2f}M)")
            print("    CVAE Encoder:")
            print(
                "      action sequence linear 参数: "
                f"{sum(p.numel() for p in raw_model.encoder_action_proj.parameters() if p.requires_grad):,}"
            )
            print(
                "      transformer 参数: "
                f"{sum(p.numel() for p in raw_model.cvae_encoder.parameters() if p.requires_grad):,}"
            )
            print("    CVAE Decoder:")
            print(
                "      visual encoder 参数: "
                f"{sum(p.numel() for p in raw_model.visual_encoder.parameters() if p.requires_grad):,}"
            )
            print(
                "      transformer encoder 参数: "
                f"{sum(p.numel() for p in raw_model.dec_encoder.parameters() if p.requires_grad):,}"
            )
            print(
                "      transformer decoder 参数: "
                f"{sum(p.numel() for p in raw_model.dec_decoder.parameters() if p.requires_grad):,}"
            )

        # ── 优化器（与 ACT 一致：AdamW + 分离 backbone lr）──
        backbone_params = []
        other_params = []
        for n, p in raw_model.named_parameters():
            if not p.requires_grad:
                continue
            if "visual_encoder" in n:
                backbone_params.append(p)
            else:
                other_params.append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": args.lr},
                {"params": backbone_params, "lr": args.lr_backbone},
            ],
            weight_decay=args.weight_decay,
        )

        # 恢复优化器。
        if args.resume:
            ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                move_optimizer_state_to_device(optimizer, args.device)

        # ── 训练 ──
        rank0_print(args, "\n[3/3] 开始训练...")
        for epoch in range(start_epoch, start_epoch + args.epochs):
            if args.distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                cfg["kl_weight"],
                args.device,
            )
            val_metrics = validate(
                model,
                val_loader,
                cfg["kl_weight"],
                args.device,
            )

            # 各进程 loss / metrics 取平均后再打印和保存。
            train_loss = reduce_scalar(train_loss, args.device)
            val_metrics = reduce_metrics(val_metrics, args.device)

            if is_main_process(args):
                print(
                    f"Epoch {epoch + 1}/{start_epoch + args.epochs}: "
                    f"train_loss={train_loss:.6f}  "
                    f"val_l1={val_metrics['l1']:.6f}  "
                    f"val_kl={val_metrics['kl']:.6f}  "
                    f"val_loss={val_metrics['loss']:.6f}"
                )

                # 保存最佳。
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    best_state_dict = {
                        k: v.detach().cpu().clone()
                        for k, v in raw_model.state_dict().items()
                    }

                # 定期保存。
                if (epoch + 1) % args.save_every == 0:
                    ckpt_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                    torch.save(
                        {
                            "model_type": "act",
                            "epoch": epoch + 1,
                            "model_state_dict": raw_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "model_cfg": cfg,
                            "best_val_loss": best_val_loss,
                            "world_size": args.world_size,
                            "global_batch_size": args.batch * args.world_size,
                        },
                        ckpt_path,
                    )
                    print(f"  Saved: {ckpt_path}")

            if args.distributed:
                dist.barrier(device_ids=[args.local_rank])

        # 恢复最佳并保存。只需要 rank0 保存最终模型。
        if is_main_process(args):
            if best_state_dict is not None:
                raw_model.load_state_dict(best_state_dict)

            final_path = output_dir / "best_actor.pt"
            torch.save(
                {
                    "model_type": "act",
                    "model_state_dict": raw_model.state_dict(),
                    "model_cfg": cfg,
                    "best_val_loss": best_val_loss,
                    "world_size": args.world_size,
                    "global_batch_size": args.batch * args.world_size,
                },
                final_path,
            )
            print(f"\n训练完成！最佳验证损失: {best_val_loss:.6f}")
            print(f"模型: {final_path}")

    finally:
        cleanup_distributed(args)


if __name__ == "__main__":
    main()




# NCCL_DEBUG=INFO NCCL_SHM_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 scripts/train_aloha_act.py \
#   --data outputs/datasets/expert/pick_place/20260606_193958/ \
#   --task pick_place \
#   --epochs 200 \
#   --batch 16 \
#   --num-workers 12



