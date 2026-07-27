"""BiMFT 单时刻多模态融合 — SlotMultimodalFusion。

对单个相机时刻 (k) 的三路输入做双向 Cross-Attention 融合:
  vision [B, N_v, D] + joint [B, R, D] + force [B, R, D]
  → [B, N_v + 2, D]  (vision tokens + joint_summary + force_summary)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import AttentionPool, CrossAttentionBlock


# ═══════════════════════════════════════════════════════════
# 门控模块
# ═══════════════════════════════════════════════════════════

class FusionGate(nn.Module):
    """关节摘要 → 视觉门控 + 力觉门控。

    joint_summary [B, 1, D] → vision_gate [B, 1, 1], force_gate [B, 1, 1]
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, joint_summary: torch.Tensor):
        """joint_summary: [B, 1, D] → (vision_gate, force_gate) 各 [B, 1, 1]"""
        raw = self.net(joint_summary)               # [B, 1, 2]
        gates = torch.sigmoid(raw)
        vision_gate = gates[..., 0:1]               # [B, 1, 1]
        force_gate = gates[..., 1:2]                # [B, 1, 1]
        return vision_gate, force_gate


# ═══════════════════════════════════════════════════════════
# 单时刻多模态融合
# ═══════════════════════════════════════════════════════════

class SlotMultimodalFusion(nn.Module):
    """单相机时刻的多模态双向融合。

    流程:
      1. AttentionPool 池化 joint → joint_summary
      2. FusionGate 生成 vision/force 门控
      3. N 层双向 Cross-Attention:
         - vision reads [joint, force] (门控)
         - force reads vision (门控)
         ★ 使用 old_v / old_f 副本避免顺序偏置
      4. force 池化 → force_summary
      5. 输出 cat([vision, joint_summary, force_summary])
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        num_layers: int,
        num_joint_queries: int = 1,
        num_force_queries: int = 1,
    ):
        super().__init__()

        self.joint_pool = AttentionPool(d_model, n_heads, num_queries=num_joint_queries)
        self.force_pool = AttentionPool(d_model, n_heads, num_queries=num_force_queries)

        self.gate = FusionGate(d_model)

        self.vision_reads_state = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.force_reads_vision = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        vision: torch.Tensor,
        joint: torch.Tensor,
        force: torch.Tensor,
    ) -> torch.Tensor:
        """vision: [B, N_v, D], joint: [B, R_q, D], force: [B, R_f, D] → [B, N_v+N_q+N_f, D]"""
        joint_summary = self.joint_pool(joint)          # [B, N_q, D]
        # §9.2: pool joint_summary → single condition token for gate
        joint_cond = joint_summary.mean(dim=1, keepdim=True) if joint_summary.shape[1] > 1 else joint_summary
        vision_gate, force_gate = self.gate(joint_cond)

        v = vision
        f = force
        for v_block, f_block in zip(
            self.vision_reads_state,
            self.force_reads_vision,
            strict=True,
        ):
            old_v = v
            old_f = f

            # vision reads [joint, old_f]
            v = v_block(query=old_v, memory=torch.cat([joint, old_f], dim=1), gate=vision_gate)
            # force reads old_v
            f = f_block(query=old_f, memory=old_v, gate=force_gate)

        force_summary = self.force_pool(f)              # [B, N_f, D]

        return torch.cat([v, joint_summary, force_summary], dim=1)  # [B, N_v+N_q+N_f, D]
