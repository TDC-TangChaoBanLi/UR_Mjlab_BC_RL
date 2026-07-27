"""BiMFT 双流同步动作解码器。

DualActionDecoderLayer: 单层解码（Self-Attn → Local Cross-Attn → Coord Cross-Attn → Sync）
DualArmActionDecoder: 完整解码器（K 个动作查询 → N 层 → 动作头）
ActionMLPHead: 连续动作 MLP 头
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import SelfAttentionBlock, CrossAttentionBlock


# ═══════════════════════════════════════════════════════════
# Action MLP Head
# ═══════════════════════════════════════════════════════════

class ActionMLPHead(nn.Module):
    """连续动作 MLP 头: d_model → hidden → action_dim."""

    def __init__(self, d_model: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, K, d_model] → [B, K, action_dim]"""
        return self.net(x)


# ═══════════════════════════════════════════════════════════
# 双流同步动作解码层
# ═══════════════════════════════════════════════════════════

class DualActionDecoderLayer(nn.Module):
    """单层双流同步动作解码。

    流程 (左右对称):
      1. Self-Attention (各自独立)
      2. Cross-Attention → 本臂状态 (h_local)
      3. Cross-Attention → 协调信息 (coordination)
      4. Cross-Attention → 对侧臂动作查询 (同步)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()

        self.left_self = SelfAttentionBlock(d_model, n_heads, d_ff, dropout)
        self.right_self = SelfAttentionBlock(d_model, n_heads, d_ff, dropout)

        self.left_local = CrossAttentionBlock(d_model, n_heads, d_ff, dropout)
        self.right_local = CrossAttentionBlock(d_model, n_heads, d_ff, dropout)

        self.left_coord = CrossAttentionBlock(d_model, n_heads, d_ff, dropout)
        self.right_coord = CrossAttentionBlock(d_model, n_heads, d_ff, dropout)

        self.left_sync = CrossAttentionBlock(d_model, n_heads, d_ff, dropout)
        self.right_sync = CrossAttentionBlock(d_model, n_heads, d_ff, dropout)

    def forward(
        self,
        a_left: torch.Tensor,        # [B, K, D]
        a_right: torch.Tensor,       # [B, K, D]
        h_left: torch.Tensor,        # [B, N, D]
        h_right: torch.Tensor,       # [B, N, D]
        coordination: torch.Tensor,  # [B, N_c, D]
    ):
        # Self-Attention
        left = self.left_self(a_left)
        right = self.right_self(a_right)

        # Local Cross-Attention → 本臂状态
        left = self.left_local(left, h_left)
        right = self.right_local(right, h_right)

        # Coordination Cross-Attention
        left = self.left_coord(left, coordination)
        right = self.right_coord(right, coordination)

        # Sync: 互相关注（使用旧副本避免顺序偏置）
        old_left = left
        old_right = right

        left = self.left_sync(old_left, old_right)
        right = self.right_sync(old_right, old_left)

        return left, right


# ═══════════════════════════════════════════════════════════
# 完整动作解码器
# ═══════════════════════════════════════════════════════════

class DualArmActionDecoder(nn.Module):
    """双流同步动作解码器。

    使用左右独立可学习 action query + 共享动作位置编码，
    通过 N 层 DualActionDecoderLayer 解码 → ActionMLPHead 输出。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        num_layers: int,
        chunk_size: int,
        action_dim: int,
        action_head_hidden_dim: int,
    ):
        super().__init__()

        # 左右独立动作查询
        self.left_queries = nn.Parameter(torch.empty(1, chunk_size, d_model))
        self.right_queries = nn.Parameter(torch.empty(1, chunk_size, d_model))
        self.action_pos = nn.Parameter(torch.empty(1, chunk_size, d_model))

        for p in [self.left_queries, self.right_queries, self.action_pos]:
            nn.init.trunc_normal_(p, std=0.02)

        self.layers = nn.ModuleList([
            DualActionDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.left_norm = nn.LayerNorm(d_model)
        self.right_norm = nn.LayerNorm(d_model)

        self.left_head = ActionMLPHead(d_model, action_head_hidden_dim, action_dim)
        self.right_head = ActionMLPHead(d_model, action_head_hidden_dim, action_dim)

    def forward(
        self,
        h_left: torch.Tensor,        # [B, N, D]
        h_right: torch.Tensor,       # [B, N, D]
        coordination: torch.Tensor,  # [B, N_c, D]
    ):
        b = h_left.shape[0]

        left = (self.left_queries + self.action_pos).expand(b, -1, -1)   # [B, K, D]
        right = (self.right_queries + self.action_pos).expand(b, -1, -1) # [B, K, D]

        for layer in self.layers:
            left, right = layer(left, right, h_left, h_right, coordination)

        left_features = self.left_norm(left)
        right_features = self.right_norm(right)

        left_action = self.left_head(left_features)
        right_action = self.right_head(right_features)

        return left_action, right_action, left_features, right_features
