"""BiMFT 双臂全局融合 — GlobalBimanualFusion。

组合以下子模块完成双臂与全局视觉的协调融合:
  - AttentionPool: 单臂摘要提取 (s_L, s_R)
  - ConditionedGlobalExtractor: 条件化全局查询 (G_L, G_R)
  - BimanualRelationExtractor: 双臂关系查询 (G_C)
  - TransformerEncoder: 紧凑协调序列编码
  - CoordinationReinjection: 协调信息回注入单臂特征
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import AttentionPool, CrossAttentionBlock


# ═══════════════════════════════════════════════════════════
# 条件化全局提取器
# ═══════════════════════════════════════════════════════════

class ConditionedGlobalExtractor(nn.Module):
    """用 arm_summary 条件化可学习 query，对 global_tokens 做 Cross-Attention。

    当 arm_summary 包含多个 token 时，先池化为 [B,1,D] 再条件注入。

    输出: [B, num_queries, d_model]
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        num_queries: int,
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, num_queries, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.condition_proj = nn.Linear(d_model, d_model)
        self.cross_attn = CrossAttentionBlock(d_model, n_heads, d_ff, dropout)

    def forward(
        self,
        arm_summary: torch.Tensor,   # [B, N_s, D]
        global_tokens: torch.Tensor, # [B, N_g, D]
    ) -> torch.Tensor:
        b = global_tokens.shape[0]
        # §12.2: c = Pool(S), 池化为单 token
        if arm_summary.shape[1] > 1:
            arm_cond = arm_summary.mean(dim=1, keepdim=True)  # [B, 1, D]
        else:
            arm_cond = arm_summary

        q = self.queries.expand(b, -1, -1)                   # [B, Q, D]
        q = q + self.condition_proj(arm_cond)                # 条件注入
        return self.cross_attn(q, global_tokens)              # [B, Q, D]


# ═══════════════════════════════════════════════════════════
# 双臂关系查询
# ═══════════════════════════════════════════════════════════

class BimanualRelationExtractor(nn.Module):
    """左右摘要联合条件化，cross-attend [H_L, H_R, global_tokens] 提取关系 query。

    输出: [B, coordination_queries, d_model]
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        num_queries: int,
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, num_queries, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.condition_proj = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.cross_attn = CrossAttentionBlock(d_model, n_heads, d_ff, dropout)

    def forward(
        self,
        h_left: torch.Tensor,        # [B, N, D]
        h_right: torch.Tensor,       # [B, N, D]
        global_tokens: torch.Tensor, # [B, G, D]
        s_left: torch.Tensor,        # [B, 1, D]
        s_right: torch.Tensor,       # [B, 1, D]
    ) -> torch.Tensor:
        b = h_left.shape[0]
        condition = torch.cat([s_left, s_right], dim=-1)      # [B, 1, 2D]
        condition = self.condition_proj(condition)             # [B, 1, D]

        q = self.queries.expand(b, -1, -1)                     # [B, Q, D]
        q = q + condition

        memory = torch.cat([h_left, h_right, global_tokens], dim=1)
        return self.cross_attn(q, memory)                      # [B, Q, D]


# ═══════════════════════════════════════════════════════════
# 协调信息回注入
# ═══════════════════════════════════════════════════════════

class CoordinationReinjection(nn.Module):
    """将协调 Transformer 输出 C 回注入左右单臂特征。

    左右各自通过门控 Cross-Attention 读取 coordination tokens。
    gate = sigmoid(Linear([s_L, s_R])) 各自独立。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        num_layers: int,
    ):
        super().__init__()

        self.left_gate_net = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.right_gate_net = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        self.left_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.right_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        h_left: torch.Tensor,        # [B, N, D]
        h_right: torch.Tensor,       # [B, N, D]
        s_left: torch.Tensor,        # [B, 1, D]
        s_right: torch.Tensor,       # [B, 1, D]
        coord: torch.Tensor,         # [B, N_c, D]
    ):
        pair = torch.cat([s_left, s_right], dim=-1)            # [B, 1, 2D]
        beta_left = self.left_gate_net(pair)                   # [B, 1, 1]
        beta_right = self.right_gate_net(pair)                 # [B, 1, 1]

        left = h_left
        right = h_right

        for l_layer, r_layer in zip(self.left_layers, self.right_layers, strict=True):
            left = l_layer(left, coord, gate=beta_left)
            right = r_layer(right, coord, gate=beta_right)

        return left, right


# ═══════════════════════════════════════════════════════════
# 双臂全局融合（顶层）
# ═══════════════════════════════════════════════════════════

class GlobalBimanualFusion(nn.Module):
    """双臂 + 全局视觉的协调融合模块。

    流程:
      s_L, s_R = AttentionPool(H_L), AttentionPool(H_R)
      G_L = ConditionedGlobalExtractor(s_L, G)
      G_R = ConditionedGlobalExtractor(s_R, G)
      G_C = BimanualRelationExtractor(H_L, H_R, G, s_L, s_R)
      compact = [s_L, s_R, G_L, G_R, G_C] → TransformerEncoder → C
      H'_L, H'_R = CoordinationReinjection(H_L, H_R, s_L, s_R, C)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        arm_summary_tokens: int,
        global_queries_per_arm: int,
        coordination_queries: int,
        coordination_layers: int,
        reinjection_layers: int,
    ):
        super().__init__()

        # 单臂摘要
        self.left_pool = AttentionPool(d_model, n_heads, arm_summary_tokens)
        self.right_pool = AttentionPool(d_model, n_heads, arm_summary_tokens)

        # 条件化全局查询
        self.left_global = ConditionedGlobalExtractor(
            d_model, n_heads, d_ff, dropout, global_queries_per_arm
        )
        self.right_global = ConditionedGlobalExtractor(
            d_model, n_heads, d_ff, dropout, global_queries_per_arm
        )

        # 双臂关系查询
        self.relation_queries = BimanualRelationExtractor(
            d_model, n_heads, d_ff, dropout, coordination_queries
        )

        # 协调 TransformerEncoder
        coord_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.coord_encoder = nn.TransformerEncoder(
            encoder_layer=coord_layer,
            num_layers=coordination_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        # 回注入
        self.reinject = CoordinationReinjection(
            d_model, n_heads, d_ff, dropout, reinjection_layers
        )

    def forward(
        self,
        h_left: torch.Tensor,        # [B, N, D]
        h_right: torch.Tensor,       # [B, N, D]
        global_tokens: torch.Tensor, # [B, G, D]
    ):
        # 单臂摘要
        s_left = self.left_pool(h_left)                    # [B, 1, D]
        s_right = self.right_pool(h_right)                 # [B, 1, D]

        # 条件化全局查询
        g_left = self.left_global(s_left, global_tokens)   # [B, Q, D]
        g_right = self.right_global(s_right, global_tokens) # [B, Q, D]

        # 双臂关系查询
        g_coord = self.relation_queries(
            h_left, h_right, global_tokens, s_left, s_right
        )                                                  # [B, C_q, D]

        # 紧凑协调序列
        compact = torch.cat([s_left, s_right, g_left, g_right, g_coord], dim=1)
        compact = self.coord_encoder(compact)              # [B, N_c, D]

        # 回注入
        h_left, h_right = self.reinject(h_left, h_right, s_left, s_right, compact)

        return h_left, h_right, compact
