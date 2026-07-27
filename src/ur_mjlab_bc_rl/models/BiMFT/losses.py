"""BiMFT 损失函数."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    is_pad: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """带 padding mask 的 Huber 损失。

    Args:
        pred:   [B, K, D] 预测动作
        target: [B, K, D] 目标动作
        is_pad: [B, K]  True=padding, 不参与损失
        delta:  Huber loss delta 参数

    Returns:
        标量损失（对非 pad 位置取均值）
    """
    loss = F.huber_loss(pred, target, delta=delta, reduction="none")  # [B, K, D]
    valid = (~is_pad).unsqueeze(-1).to(loss.dtype)                    # [B, K, 1]
    total = (loss * valid).sum()
    count = valid.sum().clamp_min(1.0)
    return total / count


def smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    """动作时序平滑损失: 相邻帧差分的 L1 均值。

    Args:
        pred: [B, K, D] 预测动作（仅对关节部分使用，不包含夹爪）

    Returns:
        标量
    """
    velocity = pred[:, 1:] - pred[:, :-1]                            # [B, K-1, D]
    return velocity.abs().mean()


def bimft_total_loss(
    pred_left: torch.Tensor,
    pred_right: torch.Tensor,
    target_left: torch.Tensor,
    target_right: torch.Tensor,
    is_pad_left: torch.Tensor,
    is_pad_right: torch.Tensor,
    huber_delta: float = 1.0,
    smoothness_weight: float = 0.05,
    joint_dim: int = 6,
) -> dict[str, torch.Tensor]:
    """BiMFT 总损失 = 左臂 Huber + 右臂 Huber + λ * 平滑损失。

    Args:
        pred_left:     [B, K, 7]  左臂预测 (joint[6] + gripper[1])
        pred_right:    [B, K, 7]  右臂预测
        target_left:   [B, K, 7]  左臂目标
        target_right:  [B, K, 7]  右臂目标
        is_pad_left:   [B, K]     左臂 padding mask
        is_pad_right:  [B, K]     右臂 padding mask
        huber_delta:   Huber delta
        smoothness_weight: 平滑损失权重
        joint_dim:     关节维度（默认 6，夹爪不参与平滑）

    Returns:
        {"loss": total, "action_loss": action, "smooth_loss": smooth}
    """
    left_loss = masked_huber_loss(pred_left, target_left, is_pad_left, huber_delta)
    right_loss = masked_huber_loss(pred_right, target_right, is_pad_right, huber_delta)
    action_loss = left_loss + right_loss

    # 仅对关节部分做平滑（前 joint_dim 维），夹爪（二值）不参与
    smooth = smoothness_loss(pred_left[..., :joint_dim]) + smoothness_loss(pred_right[..., :joint_dim])

    total = action_loss + smoothness_weight * smooth

    return {
        "loss": total,
        "action_loss": action_loss,
        "smooth_loss": smooth,
        "left_loss": left_loss,
        "right_loss": right_loss,
    }
