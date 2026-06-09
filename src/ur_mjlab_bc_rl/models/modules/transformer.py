"""Transformer 工具（基于 PyTorch 内置模块的薄封装）。

提供：
- Transformer: DETR 风格 encoder-decoder 封装
  - 支持 prepend latent/proprio tokens
  - 支持 return_intermediate_dec（返回所有 decoder 层输出）
- get_sinusoid_encoding_table: 正弦位置编码表
- build_transformer: 构建函数

编码器/解码器层直接使用 nn.TransformerEncoderLayer / nn.TransformerDecoderLayer。
位置编码通过外部加法传入（与 DETR 的 with_pos_embed 行为等价）。
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════
# 位置编码
# ═══════════════════════════════════════════════════════════

def get_sinusoid_encoding_table(n_position: int, d_hid: int) -> torch.Tensor:
    """生成正弦位置编码表。

    Args:
        n_position: 序列长度
        d_hid: 隐藏维度

    Returns:
        [1, n_position, d_hid] 位置编码
    """

    def get_position_angle_vec(position):
        return [position / math.pow(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = torch.FloatTensor([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])

    return sinusoid_table.unsqueeze(0)


# ═══════════════════════════════════════════════════════════
# Transformer 封装
# ═══════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """DETR 风格 Transformer 封装。

    基于 nn.TransformerEncoder + 手动堆叠 nn.TransformerDecoderLayer。
    相比内置 nn.Transformer 的额外功能：
      1. return_intermediate_dec: 返回所有 decoder 层输出
      2. 支持在 encoder src 前 prepend latent/proprio tokens
      3. 自动处理 pos_embed 加法（加在 src 上再传入 encoder）

    位置编码策略：与 DETR with_pos_embed 等价——
    在外部将 pos_embed 加到 src 上，再传入 PyTorch 内置层。
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
        return_intermediate_dec: bool = False,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=normalize_before,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=normalize_before,
        )
        self.decoder_layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_decoder_layers)]
        )
        self.decoder_norm = nn.LayerNorm(d_model)

        self.return_intermediate_dec = return_intermediate_dec
        self.d_model = d_model
        self.nhead = nhead

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor],
        query_embed: torch.Tensor,
        pos_embed: torch.Tensor,
        latent_input: Optional[torch.Tensor] = None,
        proprio_input: Optional[torch.Tensor] = None,
        additional_pos_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播（batch_first 格式）。

        Args:
            src:         [B, N, d_model] 图像 token 序列
            mask:        [B, N] 或 None  src_key_padding_mask
            query_embed: [K, d_model] decoder 查询（充当 query_pos）
            pos_embed:   [B, N, d_model] 图像位置编码
            latent_input:[B, d_model] 潜变量投影（prepend 到 src 前）
            proprio_input: [B, d_model] 状态投影（prepend 到 src 前）
            additional_pos_embed: [2, d_model] latent/proprio 的位置编码

        Returns:
            [num_dec_layers, B, K, d_model] decoder 输出
        """
        bs = src.shape[0]

        # 位置编码加到 src 上
        src = src + pos_embed

        # Prepend latent + proprio tokens
        if latent_input is not None and proprio_input is not None:
            latent_token = latent_input.unsqueeze(1)
            proprio_token = proprio_input.unsqueeze(1)
            if additional_pos_embed is not None:
                add_pos = additional_pos_embed.unsqueeze(0).repeat(bs, 1, 1)
                latent_token = latent_token + add_pos[:, 0:1, :]
                proprio_token = proprio_token + add_pos[:, 1:2, :]
            src = torch.cat([latent_token, proprio_token, src], dim=1)

        # Encoder
        memory = self.encoder(src, src_key_padding_mask=mask)

        # Decoder（query_embed 作为 query_pos：由于 tgt 初始为 0，
        # tgt + query_pos 退化为 query_pos，与 DETR 行为一致）
        tgt = query_embed.unsqueeze(0).repeat(bs, 1, 1)  # [B, K, d_model]

        intermediate: list[torch.Tensor] = []
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, memory_key_padding_mask=mask)
            if self.return_intermediate_dec:
                intermediate.append(self.decoder_norm(tgt))

        if self.return_intermediate_dec:
            return torch.stack(intermediate)  # [num_layers, B, K, d_model]
        return self.decoder_norm(tgt).unsqueeze(0)


# ═══════════════════════════════════════════════════════════
# 构建函数
# ═══════════════════════════════════════════════════════════

def build_transformer(cfg: dict) -> Transformer:
    """根据配置构建 Transformer。

    Args:
        cfg: 字典，包含 hidden_dim, nheads, enc_layers, dec_layers,
             dim_feedforward, dropout, pre_norm

    Returns:
        Transformer 实例
    """
    return Transformer(
        d_model=cfg.get("hidden_dim", 256),
        nhead=cfg.get("nheads", 8),
        num_encoder_layers=cfg.get("enc_layers", 4),
        num_decoder_layers=cfg.get("dec_layers", 7),
        dim_feedforward=cfg.get("dim_feedforward", 2048),
        dropout=cfg.get("dropout", 0.1),
        activation="relu",
        normalize_before=cfg.get("pre_norm", False),
        return_intermediate_dec=True,
    )
