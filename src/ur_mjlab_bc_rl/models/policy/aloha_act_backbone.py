"""ALOHA ACT 策略网络（仿 ACT/DETRVAE 架构）。

严格仿照 https://github.com/tonyzhaozh/act 的 DETRVAE 实现：

架构：
  Visual Encoder (RGBDCNN, tokens 模式) → input_proj → image tokens [B, N, hidden_dim]
  State Encoder  (Linear/MLP)             → state token  [B, 1, hidden_dim]

  CVAE Encoder (TransformerEncoder, 4 layers):
    - encoder_action_proj:  Linear(action_dim → hidden_dim)
    - encoder_joint_proj:   Linear(state_dim → hidden_dim)
    - Input: [CLS, state, action_0...action_{K-1}] + sinusoidal PE
    - CLS output → mu, logvar (z_dim=32)

  CVAE Decoder (Transformer encoder-decoder, 4+7 layers):
    - latent_out_proj:  Linear(z_dim → hidden_dim)
    - Encoder: image_tokens + latent_token + state_token
    - Decoder: K query embeddings → cross-attend → action_head → [K, action_dim]

  Prior: N(0, I)
  Loss:  L1(pred, gt) + kl_weight * KL(N(μ,σ) || N(0,I))
  Inference: z=0 (prior mean) → decoder → action chunk
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vision.encoder_factory import build_visual_encoder
from ..state.encoder_factory import build_state_encoder
from ..modules.transformer import (
    Transformer,
    build_transformer,
    get_sinusoid_encoding_table,
)
from ..specs import EncoderOutput


# ═══════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════

def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """重参数化技巧：z = mu + std * eps"""
    std = logvar.div(2).exp()
    eps = torch.randn_like(std)
    return mu + std * eps


def get_2d_sincos_pos_embed(h: int, w: int, embed_dim: int, device: torch.device) -> torch.Tensor:
    """生成 2D 正弦-余弦位置编码。

    Args:
        h: 特征图高度
        w: 特征图宽度
        embed_dim: 嵌入维度
        device: 设备

    Returns:
        [h*w, embed_dim] 位置编码
    """
    grid_h = torch.arange(h, dtype=torch.float32, device=device)
    grid_w = torch.arange(w, dtype=torch.float32, device=device)
    grid = torch.stack(torch.meshgrid(grid_h, grid_w, indexing="ij"), dim=-1)  # [h, w, 2]

    half_dim = embed_dim // 2
    div_term = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * (-math.log(10000.0) / half_dim))

    pos = torch.zeros(h, w, embed_dim, device=device)
    pos[..., 0::2] = torch.sin(grid[..., 0:1] * div_term)
    pos[..., 1::2] = torch.cos(grid[..., 0:1] * div_term)
    pos[..., 0::2] += torch.sin(grid[..., 1:2] * div_term)  # combine row and col
    pos[..., 1::2] += torch.cos(grid[..., 1:2] * div_term)

    return pos.reshape(h * w, embed_dim)


# ═══════════════════════════════════════════════════════════
# DETRVAE 核心
# ═══════════════════════════════════════════════════════════

class DETRVAE(nn.Module):
    """DETR VAE 策略网络。

    与 ACT 原始 DETRVAE 对齐：
    - CVAE encoder 从 action chunk 推断潜变量 z
    - CVAE decoder 从 z + image + state 重建 action chunk
    - 推理时 z=0（先验均值）

    Args:
        visual_cfg:     视觉编码器配置（tokens 模式建议不用 head）
        state_cfg:      状态编码器配置（linear 或 mlp）
        transformer_cfg: Transformer 参数
        action_dim:     动作维度（7）
        chunk_size:     动作分块大小 K（默认 10）
        z_dim:          潜变量维度（默认 32）
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
        hidden_dim = transformer_cfg.get("hidden_dim", 256)

        # ══════════════════════════════════════════════════
        # 1. 视觉编码器（tokens 模式）
        # ══════════════════════════════════════════════════
        self.visual_encoder = build_visual_encoder(visual_cfg)
        visual_out_dim = self.visual_encoder.get_output_dim()
        self.input_proj = nn.Linear(visual_out_dim, hidden_dim)

        # ══════════════════════════════════════════════════
        # 2. 状态编码器（Decoder 侧）
        # ══════════════════════════════════════════════════
        self.state_encoder = build_state_encoder(state_cfg)
        state_out_dim = self.state_encoder.get_output_dim()
        if state_out_dim != hidden_dim:
            self.state_proj = nn.Linear(state_out_dim, hidden_dim)
        else:
            self.state_proj = nn.Identity()

        # ══════════════════════════════════════════════════
        # 3. CVAE Encoder
        #   Input: [CLS, qpos, a_0, a_1, ..., a_{K-1}]
        # ══════════════════════════════════════════════════
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
        self.encoder_joint_proj = nn.Linear(state_cfg.get("input_dim", 7), hidden_dim)
        self.cls_embed = nn.Embedding(1, hidden_dim)

        # Sinusoidal position encoding: CLS(1) + qpos(1) + action(K)
        pos_table = get_sinusoid_encoding_table(1 + 1 + chunk_size, hidden_dim)
        self.register_buffer("pos_table", pos_table)

        # Transformer encoder for CVAE（直接使用 PyTorch 内置模块）
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_cfg.get("nheads", 8),
            dim_feedforward=transformer_cfg.get("dim_feedforward", 2048),
            dropout=transformer_cfg.get("dropout", 0.1),
            activation="relu",
            batch_first=True,
            norm_first=transformer_cfg.get("pre_norm", False),
        )
        self.cvae_encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=transformer_cfg.get("enc_layers", 4),
        )

        # Latent projection
        self.latent_proj = nn.Linear(hidden_dim, 2 * z_dim)

        # ══════════════════════════════════════════════════
        # 4. CVAE Decoder
        # ══════════════════════════════════════════════════
        self.latent_out_proj = nn.Linear(z_dim, hidden_dim)
        self.query_embed = nn.Embedding(chunk_size, hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)  # latent + state tokens

        self.transformer = build_transformer(transformer_cfg)

        # 输出头
        self.action_head = nn.Linear(hidden_dim, action_dim)

        # 权重初始化
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ══════════════════════════════════════════════════════
    # Forward
    # ══════════════════════════════════════════════════════

    def forward(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        is_pad: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """前向传播。

        Args:
            qpos:   [B, state_dim] 当前关节状态
            image:  [B, 4, H, W] RGBD 图像
            actions: [B, K, action_dim] 动作分块（训练时提供，推理时为 None）
            is_pad:  [B, K] padding mask（True 表示填充位置）

        Returns:
            (pred_chunk [B, K, action_dim], mu [B, z_dim], logvar [B, z_dim])
            推理时 mu, logvar 为 None
        """
        is_training = actions is not None
        bs = qpos.shape[0]

        # ── 视觉特征 ──
        visual_out: EncoderOutput = self.visual_encoder(image)
        if visual_out.tokens is not None:
            # Tokens 模式：[B, N, C]
            visual_tokens = self.input_proj(visual_out.tokens)
        else:
            # Vector 模式（fallback）：[B, D] → [B, 1, hidden_dim]
            visual_vector = self.input_proj(visual_out.get_vector())
            visual_tokens = visual_vector.unsqueeze(1)

        # 2D 位置编码（基于 token 数量计算空间尺寸）
        n_tokens = visual_tokens.shape[1]
        h = w = int(math.sqrt(n_tokens))
        if h * w == n_tokens:
            # 正方形特征图
            image_pos = get_2d_sincos_pos_embed(h, w, visual_tokens.shape[-1], visual_tokens.device)
            image_pos = image_pos.unsqueeze(0).repeat(bs, 1, 1)  # [B, N, hidden_dim]
        else:
            # 非正方形回退：用 1D 正弦编码
            image_pos = get_sinusoid_encoding_table(n_tokens, visual_tokens.shape[-1])
            image_pos = image_pos.to(visual_tokens.device).repeat(bs, 1, 1)

        # ── 状态特征 ──
        state_out: EncoderOutput = self.state_encoder(qpos)
        state_feat = self.state_proj(state_out.get_vector())  # [B, hidden_dim]

        # ══════════════════════════════════════════════════
        # CVAE Encoder: 从 action chunk 推断 z
        # ══════════════════════════════════════════════════
        if is_training:
            # 投影 action 和 qpos
            action_embed = self.encoder_action_proj(actions)      # [B, K, hidden_dim]
            qpos_embed = self.encoder_joint_proj(qpos).unsqueeze(1)  # [B, 1, hidden_dim]
            cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)  # [B, 1, hidden_dim]

            # 拼接：[CLS, qpos, a_0, ..., a_{K-1}]，batch_first
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], dim=1)  # [B, 1+1+K, hidden_dim]

            # Position encoding：batch_first [B, 1+1+K, hidden_dim]
            pos_embed = self.pos_table.clone().detach().repeat(bs, 1, 1)  # [B, 1+1+K, hidden_dim]

            # Padding mask（CLS 和 qpos 不 mask）
            if is_pad is not None:
                cls_joint_is_pad = torch.full((bs, 2), False, device=qpos.device)
                is_pad_full = torch.cat([cls_joint_is_pad, is_pad], dim=1)  # [B, 2+K]
            else:
                is_pad_full = None

            # Encode（batch_first）：pos_embed 加到输入上再传入
            encoder_output = self.cvae_encoder(
                encoder_input + pos_embed,
                src_key_padding_mask=is_pad_full,
            )
            encoder_output = encoder_output[:, 0, :]  # CLS token output [B, hidden_dim]

            # Latent
            latent_info = self.latent_proj(encoder_output)  # [B, 2*z_dim]
            mu = latent_info[:, :self.z_dim]
            logvar = latent_info[:, self.z_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)  # [B, hidden_dim]
        else:
            # 推理：z = 0（先验均值）
            mu = logvar = None
            latent_sample = torch.zeros(bs, self.z_dim, device=qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        # ══════════════════════════════════════════════════
        # CVAE Decoder: z + image + state → action chunk
        # ══════════════════════════════════════════════════
        # Transformer decoder
        hs = self.transformer(
            src=visual_tokens,
            mask=None,
            query_embed=self.query_embed.weight,
            pos_embed=image_pos,
            latent_input=latent_input,
            proprio_input=state_feat,
            additional_pos_embed=self.additional_pos_embed.weight,
        )
        # hs: [num_dec_layers, B, K, hidden_dim]
        # 取最后一层
        a_hat = self.action_head(hs[-1])  # [B, K, action_dim]

        return a_hat, mu, logvar

    # ══════════════════════════════════════════════════════
    # 便捷方法
    # ══════════════════════════════════════════════════════

    def get_action(self, qpos: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """推理便捷方法：返回预测的 action chunk。

        Args:
            qpos:  [B, state_dim]
            image: [B, 4, H, W]

        Returns:
            [B, K, action_dim]
        """
        self.eval()
        with torch.no_grad():
            a_hat, _, _ = self.forward(qpos, image, actions=None)
        return a_hat

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        is_pad: Optional[torch.Tensor] = None,
        kl_weight: float = 10.0,
    ) -> dict[str, torch.Tensor]:
        """计算 ACT 损失：L1 + KL。

        Args:
            pred:   [B, K, action_dim] 预测
            target: [B, K, action_dim] 真值
            mu:     [B, z_dim]
            logvar: [B, z_dim]
            is_pad: [B, K] padding mask
            kl_weight: KL 散度权重

        Returns:
            {"l1": ..., "kl": ..., "loss": ...}
        """
        # L1 loss（mask padding）
        all_l1 = F.l1_loss(pred, target, reduction="none")
        if is_pad is not None:
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        else:
            l1 = all_l1.mean()

        # KL divergence: KL(N(μ,σ) || N(0,I))
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

        total_loss = l1 + kl_weight * kl

        return {"l1": l1, "kl": kl, "loss": total_loss}


# ═══════════════════════════════════════════════════════════
# 构建函数
# ═══════════════════════════════════════════════════════════

def build_detr_vae(cfg: dict[str, Any]) -> DETRVAE:
    """根据配置构建 DETRVAE 模型。

    Args:
        cfg: 顶层配置，需包含 visual_encoder, state_encoder,
             hidden_dim, z_dim, enc_layers, dec_layers, nheads,
             dim_feedforward, dropout, pre_norm, action_dim, chunk_size

    Returns:
        DETRVAE 实例
    """
    visual_cfg = cfg.get("visual_encoder", {})
    state_cfg = cfg.get("state_encoder", {"type": "linear", "input_dim": 7, "output_dim": 256})

    transformer_cfg = {
        "hidden_dim": cfg.get("hidden_dim", 256),
        "z_dim": cfg.get("z_dim", 32),
        "enc_layers": cfg.get("enc_layers", 4),
        "dec_layers": cfg.get("dec_layers", 7),
        "nheads": cfg.get("nheads", 8),
        "dim_feedforward": cfg.get("dim_feedforward", 2048),
        "dropout": cfg.get("dropout", 0.1),
        "pre_norm": cfg.get("pre_norm", False),
    }

    return DETRVAE(
        visual_cfg=visual_cfg,
        state_cfg=state_cfg,
        transformer_cfg=transformer_cfg,
        action_dim=cfg.get("action_dim", 7),
        chunk_size=cfg.get("chunk_size", 10),
        z_dim=cfg.get("z_dim", 32),
    )


# ═══════════════════════════════════════════════════════════
# 时间集成缓冲区
# ═══════════════════════════════════════════════════════════

class EnsembleBuffer:
    """ACT 时间集成缓冲区。

    维护对未来 K 步动作的加权预测。
    每一步：添加新预测（指数加权累加），取当前步的集成动作，前移窗口。

    用法:
        buf = EnsembleBuffer(chunk_size=10, action_dim=7)
        for step in range(max_steps):
            chunk = model.get_action(qpos, image)  # [1, K, 7]
            buf.add(chunk[0])  # [K, 7]
            action = buf.get_action()  # [7]
            env.step(action)

    Args:
        chunk_size: 动作分块大小 K
        action_dim: 单步动作维度
        decay: 指数衰减系数（越大衰减越快，0 表示均匀权重）
    """

    def __init__(
        self,
        chunk_size: int = 10,
        action_dim: int = 7,
        decay: float = 0.1,
    ) -> None:
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.decay = decay

        # 加权累加缓冲区 [K, action_dim]
        self.register()

    def register(self) -> None:
        """注册/重置缓冲区。"""
        self._weighted_sum = torch.zeros(self.chunk_size, self.action_dim)
        self._weight_sum = torch.zeros(self.chunk_size)

    def add(self, chunk: torch.Tensor) -> None:
        """添加新预测（指数加权）。

        Args:
            chunk: [K, action_dim] 新预测
        """
        dev = self._weighted_sum.device
        chunk = chunk.detach().to(dev)
        weights = torch.exp(-self.decay * torch.arange(self.chunk_size, dtype=torch.float32, device=dev))
        self._weighted_sum += chunk * weights.unsqueeze(-1)
        self._weight_sum += weights

    def get_action(self) -> torch.Tensor:
        """获取当前步的集成动作，并前移窗口。

        Returns:
            [action_dim] 当前步动作
        """
        # 当前步 = 加权平均
        action = self._weighted_sum[0] / (self._weight_sum[0] + 1e-8)

        # 前移窗口
        self._weighted_sum = torch.roll(self._weighted_sum, shifts=-1, dims=0)
        self._weight_sum = torch.roll(self._weight_sum, shifts=-1, dims=0)
        self._weighted_sum[-1] = 0.0
        self._weight_sum[-1] = 0.0

        return action

    def to(self, device: torch.device) -> "EnsembleBuffer":
        """移动到指定设备。"""
        self._weighted_sum = self._weighted_sum.to(device)
        self._weight_sum = self._weight_sum.to(device)
        return self
