"""BiMFT 编码器 — SingleArmEncoder 和 GlobalVisionEncoder。

SingleArmEncoder: 单臂完整编码，组合视觉+关节+力觉+slot_fusion+temporal
GlobalVisionEncoder: 全局相机视觉编码，仅视觉+temporal
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .rgbd_resnet import RGBDResNet18
from .state_encoders import HighRateModalityEncoder
from .slot_fusion import SlotMultimodalFusion
from .temporal_fusion import AnchoredTemporalFusion


# ═══════════════════════════════════════════════════════════
# 单臂编码器
# ═══════════════════════════════════════════════════════════

class SingleArmEncoder(nn.Module):
    """单臂完整编码器。

    输入:
      wrist_rgbd:       [B, T, 4, H, W]
      joint_gripper:    [B, T, R, joint_dim + gripper_dim]
      wrench:           [B, T, R, wrench_dim]

    输出:
      [B, N_v + 2, d_model]  — 当前帧的融合特征 (N_v 视觉 + joint_summary + force_summary)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        vision_grid_h: int,
        vision_grid_w: int,
        pretrained_rgb: bool,
        joint_encoder_layers: int,
        force_encoder_layers: int,
        slot_fusion_layers: int,
        arm_temporal_layers: int,
        joint_summary_tokens: int,
        force_summary_tokens: int,
        wrist_backbone: RGBDResNet18,
    ):
        super().__init__()
        self.d_model = d_model
        self.wrist_backbone = wrist_backbone

        self.joint_encoder = HighRateModalityEncoder(
            input_dim=7,  # joint_dim(6) + gripper_dim(1)
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            num_layers=joint_encoder_layers,
        )

        self.force_encoder = HighRateModalityEncoder(
            input_dim=6,  # wrench_dim(6)
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            num_layers=force_encoder_layers,
        )

        self.slot_fusion = SlotMultimodalFusion(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            num_layers=slot_fusion_layers,
            num_joint_queries=joint_summary_tokens,
            num_force_queries=force_summary_tokens,
        )

        self.temporal_fusion = AnchoredTemporalFusion(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            num_layers=arm_temporal_layers,
        )

    def forward(
        self,
        wrist_rgbd: torch.Tensor,
        joint_gripper: torch.Tensor,
        wrench: torch.Tensor,
        image_time_emb: torch.Tensor,         # [B, T, 1, D]
        joint_time_emb: torch.Tensor,        # [B, T, R_q, D]
        force_time_emb: torch.Tensor,        # [B, T, R_f, D]
        position_emb: torch.Tensor,           # [1, N_v, D]
        camera_emb: torch.Tensor,             # [1, 1, 1, D]
        arm_emb: torch.Tensor,                # [1, 1, 1, D]
        vision_modality_emb: torch.Tensor,    # [1, 1, 1, D]
        joint_modality_emb: torch.Tensor,     # [1, 1, 1, D]
        force_modality_emb: torch.Tensor,     # [1, 1, 1, D]
    ) -> torch.Tensor:
        b, t, c, h, w = wrist_rgbd.shape

        # ── 视觉编码 ──
        visual = self.wrist_backbone(
            wrist_rgbd.reshape(b * t, c, h, w)
        )                                              # [B*T, N_v, D]
        visual = visual.reshape(b, t, -1, self.d_model) # [B, T, N_v, D]

        visual = (
            visual
            + position_emb.unsqueeze(1)               # [1, 1, N_v, D]
            + image_time_emb                           # [B, T, 1, D]
            + camera_emb                               # [1, 1, 1, D]
            + arm_emb                                  # [1, 1, 1, D]
            + vision_modality_emb                      # [1, 1, 1, D]
        )

        # ── 关节编码 ──
        joint = self.joint_encoder(
            joint_gripper,
            joint_time_emb,
            arm_emb,
            joint_modality_emb,
        )                                              # [B, T, R_q, D]

        # ── 力觉编码 ──
        force = self.force_encoder(
            wrench,
            force_time_emb,
            arm_emb,
            force_modality_emb,
        )                                              # [B, T, R_f, D]

        # ── 逐槽位融合 ──
        nv = visual.shape[2]                           # N_v
        r_joint = joint.shape[2]                       # R_q
        r_force = force.shape[2]                       # R_f
        h_slot = self.slot_fusion(
            visual.reshape(b * t, nv, -1),
            joint.reshape(b * t, r_joint, -1),
            force.reshape(b * t, r_force, -1),
        )                                              # [B*T, N_v+N_q+N_f, D]
        n_a = nv + self.slot_fusion.joint_pool.queries.shape[1] + self.slot_fusion.force_pool.queries.shape[1]
        h_slot = h_slot.reshape(b, t, n_a, -1)         # [B, T, N_a, D]

        # ── 时序融合 ──
        return self.temporal_fusion(h_slot)            # [B, N_v+2, D]


# ═══════════════════════════════════════════════════════════
# 全局视觉编码器
# ═══════════════════════════════════════════════════════════

class GlobalVisionEncoder(nn.Module):
    """全局相机视觉编码器 — 仅视觉 + 时序融合，不包含关节和力觉。

    输入:  [B, T, 4, H, W]
    输出:  [B, N_v, D]
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        global_temporal_layers: int,
        backbone: RGBDResNet18,
    ):
        super().__init__()
        self.d_model = d_model
        self.backbone = backbone
        self.temporal = AnchoredTemporalFusion(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            num_layers=global_temporal_layers,
        )

    def forward(
        self,
        images: torch.Tensor,                    # [B, T, 4, H, W]
        image_time_emb: torch.Tensor,            # [B, T, 1, D]
        camera_emb: torch.Tensor,                # [1, 1, 1, D]
        modality_emb: torch.Tensor,              # [1, 1, 1, D]
        position_emb: torch.Tensor,              # [1, N_v, D]
    ) -> torch.Tensor:
        b, t, c, h, w = images.shape

        x = self.backbone(images.reshape(b * t, c, h, w))  # [B*T, N_v, D]
        x = x.reshape(b, t, -1, self.d_model)               # [B, T, N_v, D]

        x = (
            x
            + position_emb.unsqueeze(1)                     # [1, 1, N_v, D]
            + image_time_emb                                 # [B, T, 1, D]
            + camera_emb                                     # [1, 1, 1, D]
            + modality_emb                                   # [1, 1, 1, D]
        )

        return self.temporal(x)                              # [B, N_v, D]
