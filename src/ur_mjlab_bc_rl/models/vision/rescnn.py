"""ResCNN 视觉编码器 — 残差 CNN，仅支持 stages 配置模式。

配置示例见 configs/model/multimodal.yaml。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..specs import EncoderOutput
from .base import VisualEncoderBase


# ═══════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "elu": nn.ELU,
}

_NORM_FNS: dict[str, type[nn.Module] | None] = {
    "batch_norm": nn.BatchNorm2d,
    "layer_norm": None,
    "none": None,
}


def _parse_norm_cfg(
    cfg: dict[str, Any],
    default: str = "none",
    defaults: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """从配置解析归一化参数。

    格式:
      group_norm: {num_groups: 8, eps: 1e-3}
      batch_norm: {eps: 1e-3, momentum: 0.1}
      layer_norm: true
      (无任何 *_norm key) → 使用 default

    Returns:
        (norm_name, norm_kwargs)
    """
    default_kw = defaults or {}
    for key in ("group_norm", "batch_norm", "layer_norm"):
        if key in cfg:
            val = cfg[key]
            if val is True or val is None:
                kw = dict(default_kw)
            elif isinstance(val, dict):
                kw = {**default_kw, **val}
            else:
                kw = dict(default_kw)
            return key, kw
    return default, dict(default_kw)


def _make_act(name: str, inplace: bool = False) -> nn.Module:
    """根据名称创建激活函数。"""
    cls = _ACTIVATIONS.get(name, nn.ReLU)
    if name == "relu":
        return cls(inplace=inplace)
    return cls()


def _make_norm(
    name: str,
    channels: int,
    *,
    eps: float = 1e-3,
    momentum: float = 0.1,
    num_groups: int = 8,
) -> nn.Module:
    """根据名称创建 2D 归一化层。"""
    eps = float(eps)
    momentum = float(momentum)
    num_groups = int(num_groups)
    if name == "batch_norm":
        return nn.BatchNorm2d(channels, eps=eps, momentum=momentum)
    if name == "group_norm":
        g = min(num_groups, channels)
        return nn.GroupNorm(g, channels, eps=eps)
    if name == "layer_norm":
        return nn.GroupNorm(1, channels, eps=eps)
    return nn.Identity()


# ═══════════════════════════════════════════════════════════
# 残差块
# ═══════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """BasicBlock 风格残差块：两个 3×3 卷积 + 残差连接。

    当 in_channels ≠ out_channels 或 stride ≠ 1 时，
    自动将 shortcut 替换为 1×1 卷积投影。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 3,
        activation: str = "relu",
        norm: str = "batch_norm",
        dropout: float = 0.0,
        eps: float = 1e-3,
        momentum: float = 0.1,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        _nk = dict(eps=eps, momentum=momentum, num_groups=num_groups)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                               padding=padding, bias=False)
        self.norm1 = _make_norm(norm, out_channels, **_nk)
        self.act1 = _make_act(activation)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1,
                               padding=padding, bias=False)
        self.norm2 = _make_norm(norm, out_channels, **_nk)
        self.act2 = _make_act(activation)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                _make_norm(norm, out_channels, **_nk),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += identity
        out = self.act2(out)
        return out


# ═══════════════════════════════════════════════════════════
# ResCNN 编码器
# ═══════════════════════════════════════════════════════════

class ResCNN(VisualEncoderBase):
    """ResNet 风格 CNN 视觉编码器。

    **唯一模式 — 残差 ResNet**（通过 stages 配置）：

        ResCNN(
            in_channels=4, image_size=(240, 320),
            stem_cfg={"channels": 64, "kernel": 7, "stride": 2, ...},
            stages=[{"channels": 64, "blocks": 2, "stride": 1}, ...],
            block_cfg={"kernel_size": 3, "activation": "relu", "norm": "group_norm"},
            head_cfg={"adaptive_avg_pool": {...}, "output": {"dim": 512}},
        )

    无 head_cfg → tokens 模式（输出空间特征 [B, N, C]）。
    有 head_cfg → vector 模式（输出池化向量 [B, D]）。
    """

    def __init__(
        self,
        in_channels: int = 4,
        image_size: tuple[int, int] = (128, 128),
        stem_cfg: dict[str, Any] | None = None,
        stages: list[dict[str, Any]] | None = None,
        block_cfg: dict[str, Any] | None = None,
        head_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        if stages is None:
            stages = [{"channels": 64, "blocks": 2, "stride": 1}]

        self.image_size = image_size

        # 解析 block_cfg（stages 通用参数）
        scfg = block_cfg or {}
        kernel_size = scfg.get("kernel_size", 3)
        activation = scfg.get("activation", "relu")
        dropout = scfg.get("dropout", 0.0)
        norm, norm_kwargs = _parse_norm_cfg(scfg, default="group_norm",
            defaults={"num_groups": 8, "eps": 1e-3})

        # 1. Stem
        self.stem, stem_ch = self._build_stem(stem_cfg, in_channels)

        # 2. Stages
        self.stages_mod, final_ch = self._build_stages(
            stages, stem_ch, kernel_size=kernel_size,
            activation=activation, norm=norm, dropout=dropout,
            norm_kwargs=norm_kwargs,
        )

        # 3. 计算最终特征图尺寸
        h, w = image_size
        fh, fw = self._compute_feature_size(h, w, stem_cfg, stages, kernel_size)

        # 4. Head
        self.head, self.output_dim, self._output_mode = self._build_head(
            head_cfg, final_ch, fh, fw,
        )

    # ── Stem ──────────────────────────────────────────

    def _build_stem(
        self, stem_cfg: dict[str, Any] | None, in_channels: int
    ) -> tuple[nn.Module, int]:
        """构建 Stem。返回 (module, output_channels)。

        格式:
          stem:
            channels: 64
            kernel: 7
            stride: 2
            activation: relu
            group_norm: {num_groups: 8, eps: 1e-3}
            max_pool: {kernel: 3, stride: 2}
        """
        if stem_cfg is None:
            return nn.Identity(), in_channels

        layers: list[nn.Module] = []
        ch = stem_cfg.get("channels", 64)
        kernel = stem_cfg.get("kernel", 7)
        stride = stem_cfg.get("stride", 2)
        padding = kernel // 2

        norm_name, norm_kw = _parse_norm_cfg(stem_cfg, default="none",
            defaults={"num_groups": 8, "eps": 1e-3})

        layers.append(
            nn.Conv2d(in_channels, ch, kernel, stride, padding=padding, bias=False)
        )
        if norm_name != "none":
            layers.append(_make_norm(norm_name, ch, **norm_kw))

        if stem_cfg.get("activation"):
            layers.append(_make_act(stem_cfg["activation"]))

        mp = stem_cfg.get("max_pool")
        if mp is not None:
            pk = mp.get("kernel", 3)
            ps = mp.get("stride", 2)
            layers.append(nn.MaxPool2d(pk, ps, padding=pk // 2))

        return nn.Sequential(*layers), ch

    # ── Stages ────────────────────────────────────────

    def _build_stages(
        self,
        stages: list[dict[str, Any]],
        in_channels: int,
        kernel_size: int = 3,
        activation: str = "relu",
        norm: str = "group_norm",
        dropout: float = 0.0,
        norm_kwargs: dict[str, Any] | None = None,
    ) -> tuple[nn.Module, int]:
        """构建残差阶段序列。返回 (module, final_channels)。"""
        nkw = norm_kwargs or {"num_groups": 8, "eps": 1e-3}
        seq: list[nn.Module] = []
        in_ch = in_channels

        for stage in stages:
            out_ch = stage["channels"]
            num_blocks = stage.get("blocks", 2)
            stage_stride = stage.get("stride", 1)

            for i in range(num_blocks):
                stride = stage_stride if i == 0 else 1
                seq.append(
                    ResidualBlock(
                        in_channels=in_ch, out_channels=out_ch,
                        stride=stride, kernel_size=kernel_size,
                        activation=activation, norm=norm, dropout=dropout,
                        **nkw,
                    )
                )
                in_ch = out_ch

        return nn.Sequential(*seq), in_ch

    # ── Feature size ──────────────────────────────────

    def _compute_feature_size(
        self,
        h: int, w: int,
        stem_cfg: dict[str, Any] | None,
        stages: list[dict[str, Any]],
        kernel_size: int = 3,
    ) -> tuple[int, int]:
        """预计算经过 stem + stages 后的空间尺寸。"""
        if stem_cfg is not None:
            k = stem_cfg.get("kernel", 7)
            s = stem_cfg.get("stride", 2)
            p = k // 2
            h = (h + 2 * p - k) // s + 1
            w = (w + 2 * p - k) // s + 1

            mp = stem_cfg.get("max_pool")
            if mp is not None:
                pk = mp.get("kernel", 3)
                ps = mp.get("stride", 2)
                pp = pk // 2
                h = (h + 2 * pp - pk) // ps + 1
                w = (w + 2 * pp - pk) // ps + 1

        for stage in stages:
            stride = stage.get("stride", 1)
            if stride > 1:
                p = kernel_size // 2
                h = (h + 2 * p - kernel_size) // stride + 1
                w = (w + 2 * p - kernel_size) // stride + 1

        return h, w

    # ── Head ──────────────────────────────────────────

    def _build_head(
        self,
        head_cfg: dict[str, Any] | None,
        final_channels: int,
        feat_h: int,
        feat_w: int,
    ) -> tuple[nn.Module, int, str]:
        """构建输出头。返回 (head_module, output_dim, mode)。

        无 head_cfg → tokens 模式；有 head_cfg → vector 模式。
          head:
            adaptive_avg_pool: {pool_size: [2, 2]}
            hidden: {dim: 512, activation: relu, layer_norm: true}
            output: {dim: 512, layer_norm: true, activation: none}
        """
        if head_cfg is None:
            return nn.Identity(), final_channels, "tokens"

        layers: list[nn.Module] = []

        pool_cfg = head_cfg.get("adaptive_avg_pool") or head_cfg.get("adaptive_max_pool")
        if pool_cfg is not None:
            pool_size = tuple(pool_cfg.get("pool_size", [feat_h, feat_w]))
            feat_h, feat_w = pool_size[0], pool_size[1]
            if "adaptive_avg_pool" in head_cfg:
                layers.append(nn.AdaptiveAvgPool2d(pool_size))
            else:
                layers.append(nn.AdaptiveMaxPool2d(pool_size))

        layers.append(nn.Flatten())
        in_dim = final_channels * feat_h * feat_w

        hidden_cfg = head_cfg.get("hidden")
        if hidden_cfg is not None:
            hdim = hidden_cfg.get("dim", in_dim)
            layers.append(nn.Linear(in_dim, hdim))
            if hidden_cfg.get("layer_norm", False):
                layers.append(nn.LayerNorm(hdim))
            act = hidden_cfg.get("activation")
            if act and act != "none":
                layers.append(_make_act(act))
            in_dim = hdim

        out_cfg = head_cfg.get("output", {})
        out_dim = out_cfg.get("dim", in_dim)
        layers.append(nn.Linear(in_dim, out_dim))
        if out_cfg.get("layer_norm", False):
            layers.append(nn.LayerNorm(out_dim))
        act = out_cfg.get("activation")
        if act and act != "none":
            layers.append(_make_act(act))

        return nn.Sequential(*layers), out_dim, "vector"

    # ══════════════════════════════════════════════════════
    # Forward
    # ══════════════════════════════════════════════════════

    def forward(self, rgbd: torch.Tensor) -> EncoderOutput:
        """处理 RGBD 图像。

        Args:
            rgbd: [B, 4, H, W] RGBD 图像

        Returns:
            EncoderOutput：vector 和/或 tokens
        """
        x = self.stem(rgbd)
        x = self.stages_mod(x)

        if self._output_mode == "vector":
            return EncoderOutput(vector=self.head(x), tokens=None)

        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        return EncoderOutput(vector=None, tokens=tokens)

    def get_output_dim(self) -> int:
        """返回输出维度。"""
        return self.output_dim
