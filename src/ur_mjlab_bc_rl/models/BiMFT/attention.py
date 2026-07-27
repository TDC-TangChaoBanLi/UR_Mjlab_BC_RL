"""BiMFT 通用 Attention 模块。

直接使用 nn.MultiheadAttention 构造，不额外封装工厂函数。
所有模块使用 Pre-LayerNorm (norm_first 模式)。
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════
# FeedForward
# ═══════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    """双线性前馈网络: Linear → GELU → Dropout → Linear → Dropout."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═══════════════════════════════════════════════════════════
# Self-Attention Block (Pre-Norm)
# ═══════════════════════════════════════════════════════════

class SelfAttentionBlock(nn.Module):
    """Pre-Norm Self-Attention + FFN 残差块。

    流程: x → LayerNorm → MultiheadAttention(self) → +x → LayerNorm → FFN → +x
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normed = self.attn_norm(x)
        update, _ = self.attn(
            query=normed,
            key=normed,
            value=normed,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + update
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ═══════════════════════════════════════════════════════════
# Cross-Attention Block (Pre-Norm)
# ═══════════════════════════════════════════════════════════

class CrossAttentionBlock(nn.Module):
    """Pre-Norm Cross-Attention + FFN 残差块。

    流程: q → q_norm, kv → kv_norm → CrossAttention → +q (×gate) → FFN → +q

    支持 gate: 标量 [B,1,1] / token [B,N,1] / 通道 [B,1,D] 门控。
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.q_norm(query)
        kv = self.kv_norm(memory)

        update, _ = self.attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )

        if gate is not None:
            update = update * gate

        query = query + update
        query = query + self.ffn(self.ffn_norm(query))
        return query


# ═══════════════════════════════════════════════════════════
# Attention Pooling
# ═══════════════════════════════════════════════════════════

class AttentionPool(nn.Module):
    """可学习 query 的 attention 池化。

    用 num_queries 个可学习向量作为 query，对输入 x 做 Cross-Attention，
    输出 [B, num_queries, d_model]。
    """

    def __init__(self, d_model: int, n_heads: int, num_queries: int = 1):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, num_queries, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = x.shape[0]
        q = self.queries.expand(b, -1, -1)
        y, _ = self.attn(
            query=q,
            key=self.norm(x),
            value=self.norm(x),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return y
