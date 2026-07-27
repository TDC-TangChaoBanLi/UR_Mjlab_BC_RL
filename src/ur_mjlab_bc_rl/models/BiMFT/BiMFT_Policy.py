"""BiMFT_Policy — Bimanual Multimodal Fusion Transformer Policy。

双臂 RGB-D + 力觉时序动作分块策略主网络。

架构:
  输入: 左右腕 RGB-D (4帧) + 全局 RGB-D (4帧) + 左右关节/夹爪 (4×3采样) + 左右力/力矩 (4×3采样)
  编码: SingleArmEncoder (共享权重) ×2 → H_L, H_R
        GlobalVisionEncoder → G
  融合: GlobalBimanualFusion → H'_L, H'_R, C
  解码: DualArmActionDecoder → action_L, action_R
  输出: 左右各 K×D_a 动作块
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict, NotRequired

import torch
import torch.nn as nn

from .rgbd_resnet import RGBDResNet18
from .embeddings import Factorized2DPositionEmbedding, ContinuousTimeEmbedding
from .encoders import SingleArmEncoder, GlobalVisionEncoder
from .bimanual_fusion import GlobalBimanualFusion
from .action_decoder import DualArmActionDecoder


# ═══════════════════════════════════════════════════════════
# 配置类
# ═══════════════════════════════════════════════════════════

@dataclass
class PolicyConfig:
    """BiMFT 策略网络配置（内部使用，对外 YAML 解析后填充）。"""

    # ── 输入 ──
    image_channels: int = 4
    image_height: int = 460
    image_width: int = 640
    vision_history: int = 4
    joint_high_rate: int = 3       # 每相机帧对应的关节采样数 R_q
    force_high_rate: int = 3       # 每相机帧对应的力觉采样数 R_f

    joint_dim: int = 6
    gripper_state_dim: int = 1
    wrench_dim: int = 6

    # ── 统一 token ──
    d_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1

    # ── 视觉 ──
    pretrained_rgb: bool = True
    vision_grid_h: int = 15
    vision_grid_w: int = 20
    share_global_backbone: bool = False

    # ── 高频状态编码 ──
    joint_encoder_layers: int = 2
    force_encoder_layers: int = 2

    # ── 单时刻多模态融合 ──
    slot_fusion_layers: int = 2
    joint_summary_tokens: int = 1   # N_q
    force_summary_tokens: int = 1   # N_f

    # ── 四帧时序 ──
    arm_temporal_layers: int = 2
    global_temporal_layers: int = 2

    # ── 全局视觉与双臂融合 ──
    arm_summary_tokens: int = 1
    global_queries_per_arm: int = 4
    coordination_queries: int = 8
    coordination_layers: int = 4
    reinjection_layers: int = 1

    # ── 动作解码 ──
    action_chunk_size: int = 30
    action_dim: int = 7
    action_decoder_layers: int = 4
    action_head_hidden_dim: int = 256

    # ── 损失 ──
    huber_delta: float = 1.0
    smoothness_weight: float = 0.05

    # ── 其他 ──
    max_parameters: int = 300_000_000

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) 必须被 n_heads ({self.n_heads}) 整除"
        )
        assert self.vision_history >= 2, "vision_history 必须 >= 2"


# ═══════════════════════════════════════════════════════════
# 输入/输出 TypedDict
# ═══════════════════════════════════════════════════════════

class PolicyBatch(TypedDict):
    """模型输入 batch 格式。

    标准 shape:
      left_wrist_rgbd:     [B, T_v, 4, H, W]
      right_wrist_rgbd:    [B, T_v, 4, H, W]
      global_rgbd:         [B, T_v, 4, H, W]
      left_joint_gripper:  [B, T_v, R, 7]
      right_joint_gripper: [B, T_v, R, 7]
      left_wrench:         [B, T_v, R, 6]
      right_wrench:        [B, T_v, R, 6]
      image_time_offsets:  [B, T_v]
      high_rate_time_offsets: [B, T_v, R]
    """
    left_wrist_rgbd: torch.Tensor
    right_wrist_rgbd: torch.Tensor
    global_rgbd: torch.Tensor
    left_joint_gripper: torch.Tensor
    right_joint_gripper: torch.Tensor
    left_wrench: torch.Tensor
    right_wrench: torch.Tensor
    image_time_offsets: torch.Tensor
    high_rate_time_offsets: torch.Tensor
    image_valid_mask: NotRequired[torch.Tensor]
    high_rate_valid_mask: NotRequired[torch.Tensor]


class PolicyOutput(TypedDict):
    """模型输出格式。"""
    left_action: torch.Tensor          # [B, K, D_a]
    right_action: torch.Tensor         # [B, K, D_a]
    left_action_features: torch.Tensor # [B, K, D]
    right_action_features: torch.Tensor # [B, K, D]
    diagnostics: dict[str, torch.Tensor]


# ═══════════════════════════════════════════════════════════
# BiMFT 主网络
# ═══════════════════════════════════════════════════════════

class BiMFT_Policy(nn.Module):
    """双臂多模态融合 Transformer 策略。

    参数共享:
      - wrist_backbone: 左右腕相机共用 RGBDResNet18
      - arm_encoder (SingleArmEncoder): 左右臂共用，通过 arm/camera embedding 区分
      - 动作查询: 左右独立
    """

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg

        # ── 视觉 backbone ──
        self.wrist_backbone = RGBDResNet18(
            d_model=cfg.d_model,
            grid_h=cfg.vision_grid_h,
            grid_w=cfg.vision_grid_w,
            pretrained=cfg.pretrained_rgb,
        )
        self.global_backbone = (
            self.wrist_backbone
            if cfg.share_global_backbone
            else RGBDResNet18(
                d_model=cfg.d_model,
                grid_h=cfg.vision_grid_h,
                grid_w=cfg.vision_grid_w,
                pretrained=cfg.pretrained_rgb,
            )
        )

        # ── 单臂编码器（左右共用） ──
        self.arm_encoder = SingleArmEncoder(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_ff=cfg.dim_feedforward,
            dropout=cfg.dropout,
            vision_grid_h=cfg.vision_grid_h,
            vision_grid_w=cfg.vision_grid_w,
            pretrained_rgb=cfg.pretrained_rgb,
            joint_encoder_layers=cfg.joint_encoder_layers,
            force_encoder_layers=cfg.force_encoder_layers,
            slot_fusion_layers=cfg.slot_fusion_layers,
            arm_temporal_layers=cfg.arm_temporal_layers,
            joint_summary_tokens=cfg.joint_summary_tokens,
            force_summary_tokens=cfg.force_summary_tokens,
            wrist_backbone=self.wrist_backbone,
        )

        # ── 全局视觉编码器 ──
        self.global_encoder = GlobalVisionEncoder(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_ff=cfg.dim_feedforward,
            dropout=cfg.dropout,
            global_temporal_layers=cfg.global_temporal_layers,
            backbone=self.global_backbone,
        )

        # ── 双臂全局融合 ──
        self.bimanual_fusion = GlobalBimanualFusion(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_ff=cfg.dim_feedforward,
            dropout=cfg.dropout,
            arm_summary_tokens=cfg.arm_summary_tokens,
            global_queries_per_arm=cfg.global_queries_per_arm,
            coordination_queries=cfg.coordination_queries,
            coordination_layers=cfg.coordination_layers,
            reinjection_layers=cfg.reinjection_layers,
        )

        # ── 动作解码器 ──
        self.action_decoder = DualArmActionDecoder(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_ff=cfg.dim_feedforward,
            dropout=cfg.dropout,
            num_layers=cfg.action_decoder_layers,
            chunk_size=cfg.action_chunk_size,
            action_dim=cfg.action_dim,
            action_head_hidden_dim=cfg.action_head_hidden_dim,
        )

        # ── Embeddings ──
        self.time_embedding = ContinuousTimeEmbedding(cfg.d_model)

        self.position_2d = Factorized2DPositionEmbedding(
            cfg.vision_grid_h, cfg.vision_grid_w, cfg.d_model
        )

        # camera: 0=left_wrist, 1=right_wrist, 2=global
        self.camera_embedding = nn.Embedding(3, cfg.d_model)
        # arm: 0=left, 1=right
        self.arm_embedding = nn.Embedding(2, cfg.d_model)
        # modality: 0=vision, 1=joint, 2=force, 3=coordination
        self.modality_embedding = nn.Embedding(4, cfg.d_model)

        self._check_parameter_budget()

    def _check_parameter_budget(self):
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if count > self.cfg.max_parameters:
            raise ValueError(
                f"Model has {count:,} trainable parameters, "
                f"exceeding limit {self.cfg.max_parameters:,}."
            )
        print(f"[BiMFT] Trainable parameters: {count:,}")

    def forward(self, batch: PolicyBatch) -> PolicyOutput:
        cfg = self.cfg

        # ── 1. 时间 embedding ──
        image_time_emb = self.time_embedding(batch["image_time_offsets"])
        # [B, T] → [B, T, D] → [B, T, 1, D]
        image_time_emb = image_time_emb.unsqueeze(2)

        high_rate_time_offsets = batch["high_rate_time_offsets"]
        # [B, T, R_max] → joint 和 force 分别取对应的子时间
        r_joint = batch["left_joint_gripper"].shape[2]
        r_force = batch["left_wrench"].shape[2]
        high_time_emb_joint = self.time_embedding(high_rate_time_offsets[:, :, :r_joint])
        high_time_emb_force = self.time_embedding(high_rate_time_offsets[:, :, :r_force])

        # ── 2. 位置 + 类别 embedding ──
        pos_emb = self.position_2d()  # [1, N_v, D]

        # camera: [1, 1, 1, D]
        cam_left = self.camera_embedding(torch.tensor([0], device=pos_emb.device)).view(1, 1, 1, -1)
        cam_right = self.camera_embedding(torch.tensor([1], device=pos_emb.device)).view(1, 1, 1, -1)
        cam_global = self.camera_embedding(torch.tensor([2], device=pos_emb.device)).view(1, 1, 1, -1)

        # arm: [1, 1, 1, D]
        arm_left = self.arm_embedding(torch.tensor([0], device=pos_emb.device)).view(1, 1, 1, -1)
        arm_right = self.arm_embedding(torch.tensor([1], device=pos_emb.device)).view(1, 1, 1, -1)

        # modality: [1, 1, 1, D]
        mod_vision = self.modality_embedding(torch.tensor([0], device=pos_emb.device)).view(1, 1, 1, -1)
        mod_joint = self.modality_embedding(torch.tensor([1], device=pos_emb.device)).view(1, 1, 1, -1)
        mod_force = self.modality_embedding(torch.tensor([2], device=pos_emb.device)).view(1, 1, 1, -1)
        mod_coord = self.modality_embedding(torch.tensor([3], device=pos_emb.device)).view(1, 1, 1, -1)

        # ── 3. 左右单臂编码 ──
        h_left = self.arm_encoder(
            wrist_rgbd=batch["left_wrist_rgbd"],
            joint_gripper=batch["left_joint_gripper"],
            wrench=batch["left_wrench"],
            image_time_emb=image_time_emb,
            joint_time_emb=high_time_emb_joint,
            force_time_emb=high_time_emb_force,
            position_emb=pos_emb,
            camera_emb=cam_left,
            arm_emb=arm_left,
            vision_modality_emb=mod_vision,
            joint_modality_emb=mod_joint,
            force_modality_emb=mod_force,
        )                                              # [B, N_a, D]

        h_right = self.arm_encoder(
            wrist_rgbd=batch["right_wrist_rgbd"],
            joint_gripper=batch["right_joint_gripper"],
            wrench=batch["right_wrench"],
            image_time_emb=image_time_emb,
            joint_time_emb=high_time_emb_joint,
            force_time_emb=high_time_emb_force,
            position_emb=pos_emb,
            camera_emb=cam_right,
            arm_emb=arm_right,
            vision_modality_emb=mod_vision,
            joint_modality_emb=mod_joint,
            force_modality_emb=mod_force,
        )                                              # [B, N_a, D]

        # ── 4. 全局视觉 ──
        global_tokens = self.global_encoder(
            images=batch["global_rgbd"],
            image_time_emb=image_time_emb,
            camera_emb=cam_global,
            modality_emb=mod_vision,
            position_emb=pos_emb,
        )                                              # [B, N_v, D]

        # ── 5. 双臂融合 ──
        h_left, h_right, coordination = self.bimanual_fusion(
            h_left, h_right, global_tokens
        )

        # ── 6. 双流同步动作解码 ──
        left_action, right_action, left_features, right_features = self.action_decoder(
            h_left, h_right, coordination
        )

        return {
            "left_action": left_action,
            "right_action": right_action,
            "left_action_features": left_features,
            "right_action_features": right_features,
            "diagnostics": {},
        }
