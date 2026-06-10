"""ALOHA ACT 策略网络（仿 ACT/DETRVAE 架构）。

包含完整的 Transformer 实现 + DETRVAE + 时间集成。

架构：
  Visual Encoder → input_proj → image tokens [B, N, d_model]
  State Encoder            → state token  [B, 1, d_model]

  CVAE Encoder (TransformerEncoder):
    [CLS, qpos, a_0...a_{K-1}] + sinusoidal PE → mu, logvar (z_dim)

  CVAE Decoder (Transformer encoder-decoder):
    Encoder: image_tokens + latent_token + state_token
    Decoder: K query_embed → cross-attend → action_head → [K, action_dim]

  Prior: N(0,I)  Inference: z=0
  Loss: L1 + kl_weight * KL
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vision.encoder_factory import build_visual_encoder
from ..state.encoder_factory import build_state_encoder
from ..specs import EncoderOutput


# ═══════════════════════════════════════════════════════════
# 位置编码
# ═══════════════════════════════════════════════════════════

def get_sinusoid_encoding_table(n_position: int, d_hid: int) -> torch.Tensor:
    """正弦位置编码表。 [n_position, d_hid]"""

    def get_position_angle_vec(pos):
        return [pos / math.pow(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    t = torch.FloatTensor([get_position_angle_vec(i) for i in range(n_position)])
    t[:, 0::2] = torch.sin(t[:, 0::2])
    t[:, 1::2] = torch.cos(t[:, 1::2])
    return t


def get_2d_sincos_pos_embed(h: int, w: int, embed_dim: int, device: torch.device) -> torch.Tensor:
    """2D 正弦-余弦位置编码。 [h*w, embed_dim]"""
    gh = torch.arange(h, dtype=torch.float32, device=device)
    gw = torch.arange(w, dtype=torch.float32, device=device)
    grid = torch.stack(torch.meshgrid(gh, gw, indexing="ij"), dim=-1)

    half = embed_dim // 2
    div = torch.exp(torch.arange(half, dtype=torch.float32, device=device) * (-math.log(10000.0) / half))

    pos = torch.zeros(h, w, embed_dim, device=device)
    pos[..., 0::2] = torch.sin(grid[..., 0:1] * div)
    pos[..., 1::2] = torch.cos(grid[..., 0:1] * div)
    pos[..., 0::2] += torch.sin(grid[..., 1:2] * div)
    pos[..., 1::2] += torch.cos(grid[..., 1:2] * div)
    return pos.reshape(h * w, embed_dim)


# ═══════════════════════════════════════════════════════════
# 配置解析（Transformer 参数 → Dict）
# ═══════════════════════════════════════════════════════════

_DEFAULT_XFM = {"d_model": 256, "nlayer": 4, "nhead": 8, "dim_feedforward": 2048,
                "dropout": 0.1, "activation": "relu", "norm_first": False}


def _parse_xfm_cfg(cfg: dict | None, defaults: dict | None = None) -> dict:
    """解析单个 transformer 子配置，填充默认值。"""
    c = cfg or {}
    d = defaults or _DEFAULT_XFM
    return {
        "d_model": c.get("d_model", d.get("d_model", 256)),
        "nhead": c.get("nhead", d.get("nhead", 8)),
        "nlayer": c.get("nlayer", d.get("nlayer", 4)),
        "dim_feedforward": c.get("dim_feedforward", d.get("dim_feedforward", 2048)),
        "dropout": c.get("dropout", d.get("dropout", 0.1)),
        "activation": c.get("activation", d.get("activation", "relu")),
        "norm_first": c.get("norm_first", d.get("norm_first", False)),
    }





# ═══════════════════════════════════════════════════════════
# CVAE
# ═══════════════════════════════════════════════════════════

def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """z = mu + std * eps"""
    return mu + logvar.div(2).exp() * torch.randn_like(logvar)


# ═══════════════════════════════════════════════════════════
# DETRVAE
# ═══════════════════════════════════════════════════════════

class DETRVAE(nn.Module):
    """DETR VAE 策略网络。

    - encoder: 从 action chunk 推断 z
    - decoder: 从 z + image + state 重建 action chunk
    - 推理: z=0（先验均值）
    """

    def __init__(
        self,
        visual_cfg: dict[str, Any],
        state_cfg: dict[str, Any],
        transformer_cfg: dict[str, Any],
        action_dim: int = 7,
        chunk_size: int = 10,
        z_dim: int = 32,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.z_dim = z_dim
        hidden_dim = transformer_cfg.get("encoder_encoder", {}).get("d_model", 256)

        # 1. 视觉编码器
        self.visual_encoder = build_visual_encoder(visual_cfg) # 每个通道展平后的视觉特征 [B, H*W, C]
        self.visual_proj = nn.Linear(self.visual_encoder.get_output_dim(), hidden_dim) # 将展平的维度映射到 Token 维度

        # 2. 状态编码器
        self.state_encoder = build_state_encoder(state_cfg)
        so_dim = self.state_encoder.get_output_dim()
        self.state_proj = nn.Linear(so_dim, hidden_dim) if so_dim != hidden_dim else nn.Identity()

        # 3. CVAE Encoder
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim) # 将 K 个 action 分别映射到 Token 维度
        self.encoder_joint_proj  = nn.Linear(state_cfg.get("input_dim", 7), hidden_dim)
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.register_buffer("pos_table", # action sequence position encoding
            get_sinusoid_encoding_table(chunk_size, hidden_dim).unsqueeze(0))
        # CVAE Encoder
        enc_enc = _parse_xfm_cfg(transformer_cfg.get("encoder_encoder"))
        cvae_enc_layer = nn.TransformerEncoderLayer(
            enc_enc["d_model"], enc_enc["nhead"], enc_enc["dim_feedforward"],
            enc_enc["dropout"], enc_enc["activation"], batch_first=True, norm_first=enc_enc["norm_first"]
        )
        self.cvae_encoder = nn.TransformerEncoder(cvae_enc_layer, num_layers=enc_enc["nlayer"])
        self.latent_proj = nn.Linear(hidden_dim, 2 * z_dim)

        # 4. CVAE Decoder

        # latent Linear Projection
        self.latent_out_proj = nn.Linear(z_dim, hidden_dim)

        # CVAE Decoder Encoder
        dec_enc = _parse_xfm_cfg(transformer_cfg.get("decoder_encoder"))
        dec_enc_layer = nn.TransformerEncoderLayer(
                dec_enc["d_model"], dec_enc["nhead"], dec_enc["dim_feedforward"],
                dec_enc["dropout"], dec_enc["activation"],
                batch_first=True, norm_first=dec_enc["norm_first"],
            )
        self.dec_encoder = nn.TransformerEncoder(dec_enc_layer, num_layers=dec_enc["nlayer"])

        # CVAE Decoder Decoder
        self.pose_embed_query = nn.Embedding(chunk_size, hidden_dim)
        dec_dec = _parse_xfm_cfg(transformer_cfg.get("decoder_decoder"))
        dec_dec_layer = nn.TransformerDecoderLayer(
            dec_dec["d_model"], dec_dec["nhead"], dec_dec["dim_feedforward"],
            dec_dec["dropout"], dec_dec["activation"],
            batch_first=True, norm_first=dec_dec["norm_first"],
        )
        self.dec_decoder = nn.TransformerDecoder(dec_dec_layer, num_layers=dec_dec["nlayer"])
        self.dec_decoder_norm = nn.LayerNorm(dec_enc["d_model"])

        self.action_head = nn.Linear(hidden_dim, action_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        is_pad: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        is_training = actions is not None
        bs = qpos.shape[0]

        # ── CVAE Encoder ──
        if is_training:
            pos_emb = self.pos_table.clone().detach().repeat(bs, 1, 1) # action sequence 位置编码
            act_emb = self.encoder_action_proj(actions) # action sequence Tokens
            qp_emb  = self.encoder_joint_proj(qpos).unsqueeze(1) # joint position Token
            cls_emb = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1) # CLS Token
            enc_in  = torch.cat([cls_emb, qp_emb, act_emb+pos_emb], dim=1)
            is_pad_full = torch.cat([
                torch.full((bs, 2), False, device=qpos.device), is_pad], dim=1) if is_pad is not None else None
            enc_out = self.cvae_encoder(enc_in, src_key_padding_mask=is_pad_full)
            li = self.latent_proj(enc_out[:, 0, :]) # 取 CLS Token 线性映射到 latent space
            mu, logvar = li[:, :self.z_dim], li[:, self.z_dim:]
            latent_input = self.latent_out_proj(reparametrize(mu, logvar))
        else:
            mu = logvar = None
            latent_input = self.latent_out_proj(torch.zeros(bs, self.z_dim, device=qpos.device))

        # ── CVAE Decoder──

        # ── 视觉 ──
        vout: EncoderOutput = self.visual_encoder(image)
        visual_tokens = self.visual_proj(vout.tokens) if vout.tokens is not None \
            else self.visual_proj(vout.get_vector()).unsqueeze(1)

        nt = visual_tokens.shape[1]
        h = w = int(math.sqrt(nt))
        if h * w == nt:
            image_pos = get_2d_sincos_pos_embed(h, w, visual_tokens.shape[-1], visual_tokens.device)
        else:
            image_pos = get_sinusoid_encoding_table(nt, visual_tokens.shape[-1]).to(visual_tokens.device)
        image_pos = image_pos.unsqueeze(0).repeat(bs, 1, 1)
        visual_tokens = visual_tokens + image_pos # 添加视觉位置编码 [B, N, d_model]

        # ── 状态 ──
        sout: EncoderOutput = self.state_encoder(qpos)
        state_feat = self.state_proj(sout.get_vector())
        state_feat = state_feat.unsqueeze(1)     # [B, 1, d_model]

        # —— 隐变量 ——
        latent_input = latent_input.unsqueeze(1)  # [B, 1, d_model]

        # Transformer Encoder: Self-Attention
        dec_src = torch.cat([visual_tokens, latent_input, state_feat], dim=1) # 拼接视觉、隐变量和状态特征 [B, N+2, d_model]
        memory = self.dec_encoder(dec_src) # 融合后的 Token 编存

        # Transformer Decoder: Cross-Attention
        tgt = self.pose_embed_query.weight.unsqueeze(0).repeat(bs, 1, 1)  # [B, K, d_model]
        hs = self.dec_decoder(tgt, memory)               # [B, K, d_model]
        hs = self.dec_decoder_norm(hs).unsqueeze(0)       # [1, B, K, d_model]

        return self.action_head(hs[-1]), mu, logvar

    def get_action(self, qpos: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(qpos, image)[0]

    def compute_loss(self, pred, target, mu, logvar, is_pad=None, kl_weight=10.0) -> dict:
        l1 = F.l1_loss(pred, target, reduction="none")
        if is_pad is not None:
            l1 = (l1 * ~is_pad.unsqueeze(-1)).mean()
        else:
            l1 = l1.mean()
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        return {"l1": l1, "kl": kl, "loss": l1 + kl_weight * kl}


def build_detr_vae(cfg: dict[str, Any]) -> DETRVAE:
    """从配置构建 DETRVAE。"""
    return DETRVAE(
        visual_cfg=cfg.get("visual_encoder", {}),
        state_cfg=cfg.get("state_encoder", {"input_dim": 7}),
        transformer_cfg=cfg.get("transformer", {}),
        action_dim=cfg.get("action_dim", 7),
        chunk_size=cfg.get("chunk_size", 10),
        z_dim=cfg.get("z_dim", 32),
    )


# ═══════════════════════════════════════════════════════════
# 时间集成
# ═══════════════════════════════════════════════════════════

class EnsembleBuffer:
    """ACT 时间集成缓冲区。

    指数加权累加 K 步预测，每步取加权平均后前移窗口。
    """

    def __init__(self, chunk_size: int = 10, action_dim: int = 7, decay: float = 0.1):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.decay = decay
        self.register()

    def register(self) -> None:
        self._ws = torch.zeros(self.chunk_size, self.action_dim)
        self._wc = torch.zeros(self.chunk_size)

    def add(self, chunk: torch.Tensor) -> None:
        d = self._ws.device
        w = torch.exp(-self.decay * torch.arange(self.chunk_size, dtype=torch.float32, device=d))
        self._ws += chunk.detach().to(d) * w.unsqueeze(-1)
        self._wc += w

    def get_action(self) -> torch.Tensor:
        a = self._ws[0] / (self._wc[0] + 1e-8)
        self._ws = torch.roll(self._ws, -1, 0)
        self._wc = torch.roll(self._wc, -1, 0)
        self._ws[-1] = 0.0
        self._wc[-1] = 0.0
        return a

    def to(self, device: torch.device) -> "EnsembleBuffer":
        self._ws = self._ws.to(device)
        self._wc = self._wc.to(device)
        return self