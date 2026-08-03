#!/usr/bin/env python3
"""BiMFT 数据集流式诊断 — 用训练数据验证模型输出 action 与数据集 action 的差距。

以时间顺序流式读取数据集，复现训练时的 batch 构造（图像 [0,1]、depth 毫米），
跑模型前向计算 action loss；同时复现 eval_bimft.py 的错误预处理（图像 [0,255]）
做对照，验证图像范围不匹配是否是模型退化的根因。

用法:
    uv run python scripts/diag_bimft_dataset.py \
        --checkpoint outputs/train/2026-07-30/09-27-37_bimft/checkpoints/last/pretrained_model \
        --data outputs/datasets/expert/dual_pick_place/20260730_011257 \
        --n-samples 30 \
        --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from eval_bimft import load_bimft_policy  # noqa: E402

# 相机列表（与训练 config.input_features 一致）
CAMERAS = [
    "A_realsense_link_CAMERA",
    "B_realsense_link_CAMERA",
    "global_realsense_link_CAMERA",
]

# 初始关节角（scene_dual.yaml default_qpos + 夹爪 0，A 与 B 各一份）
INITIAL_ACTION = np.array(
    [0.0, -1.32, 1.32, -1.57, -1.57, 0.0, 0.0] * 2,
    dtype=np.float32,
)


def build_batch_from_frames(
    obs_frames: list[dict],
    action_frames: list[dict],
    *,
    device: str,
    image_scale: float,
) -> dict[str, torch.Tensor]:
    """把连续帧堆叠成训练格式 batch。

    obs_frames:  4 帧（t-3..t），每帧含图像/状态
    action_frames: 10 帧（t..t+9），每帧 action [3,14]
    image_scale: 图像缩放（1.0=训练[0,1]，255.0=复现 eval 的错误[0,255]）

    Returns:
        batch 含 observation.* / action / action_is_pad
    """
    B, T_obs, T_act = 1, len(obs_frames), len(action_frames)
    batch: dict[str, torch.Tensor] = {}

    # ── 图像 ──
    for cam in CAMERAS:
        rgb = torch.stack(
            [torch.as_tensor(f[f"observation.images.{cam}.rgb"]) * image_scale for f in obs_frames]
        )  # [T, 3, H, W]
        depth = torch.stack(
            [torch.as_tensor(f[f"observation.images.{cam}.depth"]) for f in obs_frames]
        )  # [T, 1, H, W]（mm，训练/评估一致）
        batch[f"observation.images.{cam}.rgb"] = rgb.unsqueeze(0).to(device)   # [1, T, 3, H, W]
        batch[f"observation.images.{cam}.depth"] = depth.unsqueeze(0).to(device)  # [1, T, 1, H, W]

    # ── 状态（[T, R, D]）──
    for sk in ["joint.position", "sensor.force", "sensor.torque"]:
        key = f"observation.state.{sk}"
        stacked = torch.stack([torch.as_tensor(f[key]) for f in obs_frames])  # [T, R, D]
        batch[key] = stacked.unsqueeze(0).to(device)  # [1, T, R, D]

    # ── action（[T_act, R, D] → [1, T_act, R, D]）──
    act = torch.stack([torch.as_tensor(f["action"]) for f in action_frames])  # [T_act, 3, 14]
    batch["action"] = act.unsqueeze(0).to(device)  # [1, T_act, 3, 14]
    batch["action_is_pad"] = torch.zeros(B, T_act, dtype=torch.bool, device=device)

    return batch


def collect_frames(ds, frame_idxs: list[int]) -> list[dict]:
    """按全局帧索引取一批观测字典（含图像，会解码视频）。"""
    return [ds[i] for i in frame_idxs]


def _to_array(v) -> np.ndarray:
    """把 parquet 中的 object 数组规范化为 float32 ndarray。"""
    v = np.asarray(v)
    if v.dtype == object:
        return np.stack(v).astype(np.float32)
    return v.astype(np.float32)


def load_parquet_frames(data_root: str) -> dict[str, dict[int, np.ndarray]]:
    """从 parquet 预加载 action / state（不涉及视频解码，快）。

    Returns:
        {"action": {global_idx: [3,14]}, "state.joint.position": {idx: [3,14]}, ...}
    """
    import glob
    import pandas as pd

    files = sorted(glob.glob(str(Path(data_root) / "data/chunk-*/file-*.parquet")))
    cols = [
        "index",
        "observation.state.joint.position",
        "observation.state.sensor.force",
        "observation.state.sensor.torque",
        "action",
    ]
    col_map = {
        "action": "action",
        "state.joint.position": "observation.state.joint.position",
        "state.sensor.force": "observation.state.sensor.force",
        "state.sensor.torque": "observation.state.sensor.torque",
    }
    out: dict[str, dict[int, np.ndarray]] = {k: {} for k in col_map}
    for f in files:
        df = pd.read_parquet(f, columns=cols)
        idxs = df["index"].astype(int).to_numpy()
        for key, col in col_map.items():
            for i, v in zip(idxs, df[col].to_numpy()):
                out[key][int(i)] = _to_array(v)
    return out


def run(args: argparse.Namespace) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"加载数据集: {args.data}")
    ds = LeRobotDataset(
        "ur5_dual_pick_place",
        root=args.data,
        video_backend="pyav",
    )
    print(f"  total_episodes={ds.num_episodes}  total_frames={ds.num_frames}")

    print(f"加载模型: {args.checkpoint}")
    policy = load_bimft_policy(args.checkpoint, args.device)

    print(f"预加载 parquet（action/state）: {args.data}")
    pf = load_parquet_frames(args.data)

    n_obs = policy.config.n_obs_steps   # 4
    horizon = policy.config.horizon     # 30
    act_delta = horizon // 3            # 10（每帧 3 采样）
    print(f"  n_obs_steps={n_obs} horizon={horizon} action_delta={act_delta}")

    # ── 统计量 ──
    losses_ok: list[float] = []   # 正确预处理 [0,1] 的 action loss
    losses_bad: list[float] = []  # 错误预处理 [0,255] 的 action loss
    mae_ok: list[float] = []      # 正确预处理下 预测 vs 真实 的 MAE
    mae_init: list[float] = []    # 真实 action 相对初始位置的 MAE（基线）
    pred_vs_init: list[float] = []  # 正确预处理下模型预测相对初始位置的距离

    n = 0
    eps = ds.meta.episodes
    for ep_idx in range(len(eps)):
        if n >= args.n_samples:
            break
        ep = eps[ep_idx]
        ep_from = int(ep["dataset_from_index"])
        ep_len = int(ep["length"])
        if ep_len < n_obs + act_delta:
            continue
        # 在 episode 内部等间隔采样起点（避开边界，保证有 4 帧历史 + 10 帧未来）
        for t_local in range(n_obs - 1, ep_len - act_delta, 15):
            if n >= args.n_samples:
                break
            idxs_obs = [ep_from + t_local - (n_obs - 1) + k for k in range(n_obs)]  # 4 帧
            idxs_act = [ep_from + t_local + k for k in range(act_delta)]            # 10 帧
            # obs 帧需要图像 → 解码视频；action/state 从 parquet 取（快）
            obs_frames = collect_frames(ds, idxs_obs)
            act_frames = [{"action": pf["action"][i]} for i in idxs_act]
            # 用 parquet 的 state 覆盖 obs_frames 里的 state（避免依赖 ds 解码）
            for k, i in enumerate(idxs_obs):
                obs_frames[k]["observation.state.joint.position"] = pf["state.joint.position"][i]
                obs_frames[k]["observation.state.sensor.force"] = pf["state.sensor.force"][i]
                obs_frames[k]["observation.state.sensor.torque"] = pf["state.sensor.torque"][i]

            # 正确预处理（训练口径）：图像 [0,1]
            batch_ok = build_batch_from_frames(obs_frames, act_frames, device=args.device, image_scale=1.0)
            loss_ok, info_ok = policy.forward(batch_ok)
            losses_ok.append(float(loss_ok.detach().cpu()))

            # 错误预处理（复现 eval_bimft.py）：图像 [0,255]
            batch_bad = build_batch_from_frames(obs_frames, act_frames, device=args.device, image_scale=255.0)
            loss_bad, _ = policy.forward(batch_bad)
            losses_bad.append(float(loss_bad.detach().cpu()))

            # 正确预处理下模型预测的 action chunk 与真实对比
            with torch.no_grad():
                chunk = policy.predict_action_chunk(batch_ok).cpu().numpy()[0]  # [30, 14]
                target = batch_ok["action"].cpu().numpy()[0].reshape(horizon, 14)  # [30, 14]
            mae_ok.append(float(np.mean(np.abs(chunk - target))))
            # 真实 action 相对初始位置的偏移（衡量任务是否要求运动）
            mae_init.append(float(np.mean(np.abs(target - INITIAL_ACTION))))
            # 模型预测相对初始位置的偏移（若≈0 说明模型输出退化到初始附近）
            pred_vs_init.append(float(np.mean(np.abs(chunk - INITIAL_ACTION))))

            n += 1

    # ── 输出报告 ──
    print("\n" + "=" * 70)
    print("流式验证结果（模型输出 action vs 数据集 action）")
    print("=" * 70)
    if losses_ok:
        print(f"\n[1] action loss（图像 [0,1] 训练口径）:      mean={np.mean(losses_ok):.4f}  "
              f"p50={np.median(losses_ok):.4f}")
        print(f"    action loss（图像 [0,255] 评估口径）:      mean={np.mean(losses_bad):.4f}  "
              f"p50={np.median(losses_bad):.4f}")
        print(f"    → 两口径 loss 比值: {np.mean(losses_bad)/max(np.mean(losses_ok),1e-9):.1f}x")
        print(f"\n[2] 预测 vs 真实 action 的 MAE（训练口径）: mean={np.mean(mae_ok):.4f} rad")
        print(f"    真实 action 相对初始位置的 MAE（任务运动量）: mean={np.mean(mae_init):.4f} rad")
        print(f"    模型预测相对初始位置的 MAE:            mean={np.mean(pred_vs_init):.4f} rad")
        print(f"\n    → 若 [2] 中真实 MAE 远大于 0 而预测 MAE 接近 0，说明模型退化为输出初始位置")
        print(f"    → 若 [1] 训练口径 loss 远小于评估口径 loss，说明图像范围不匹配是评估失败的根因")
    print("\n完成。")


def main() -> None:
    parser = argparse.ArgumentParser(description="BiMFT 数据集流式诊断")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="outputs/datasets/expert/dual_pick_place/20260730_011257")
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
