"""RGBD CNN 视觉编码器。

支持两种模式：
1. 普通 CNN（向后兼容）：通过 hidden_dims 配置
2. 残差 ResNet（新）：通过 stages 配置，支持可选 stem / head / 残差连接

残差模式下，配置示例见 configs/model/multimodal.yaml。
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
    "layer_norm": None,   # 2D LayerNorm 用 GroupNorm(1, …) 近似
    "none": None,
}


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
    """根据名称创建 2D 归一化层。

    Args:
        name: 归一化类型 — "batch_norm", "group_norm", "layer_norm", "none"
        channels: 通道数
        eps: BatchNorm 的 epsilon（默认 1e-3，而非 PyTorch 默认 1e-5）
        momentum: BatchNorm 动量
        num_groups: GroupNorm 的分组数
    """
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
    自动将 shortcut 替换为 1×1 卷积投影，无需手动配置。
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        stride: 第一个卷积的步长（用于空间降采样）
        kernel_size: 卷积核大小
        activation: 激活函数名称
        norm: 归一化类型
        norm_eps: BatchNorm/GroupNorm 的 epsilon
        norm_momentum: BatchNorm 动量
        norm_num_groups: GroupNorm 分组数
        dropout: Dropout2d 概率
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 3,
        activation: str = "relu",
        norm: str = "batch_norm",
        norm_eps: float = 1e-3,
        norm_momentum: float = 0.1,
        norm_num_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2
        _nk = dict(eps=norm_eps, momentum=norm_momentum, num_groups=norm_num_groups)

        # Conv1 — 可能带 stride 降采样
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            padding=padding, bias=False,
        )
        self.norm1 = _make_norm(norm, out_channels, **_nk)
        self.act1 = _make_act(activation)

        # Conv2 — stride=1，保持尺寸
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, 1,
            padding=padding, bias=False,
        )
        self.norm2 = _make_norm(norm, out_channels, **_nk)
        self.act2 = _make_act(activation)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Shortcut — 自动判断是否需要 1×1 投影
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
# RGBD CNN 编码器（普通 + 残差双模式）
# ═══════════════════════════════════════════════════════════

class RGBDCNNEncoder(VisualEncoderBase):
    """RGBD CNN 视觉编码器。

    **普通 CNN 模式**（向后兼容）：
        RGBDCNNEncoder(in_channels=4, hidden_dims=[32,64,128], output_dim=256)

    **残差 ResNet 模式**（新）：
        RGBDCNNEncoder(
            in_channels=4, image_size=(240,320),
            stages=[{"channels":64,"blocks":2,"stride":1}, ...],
            head_cfg={"global_pool":"adaptive_avg","pool_size":[4,4],"hidden":[256],"output_dim":512},
        )
    """

    def __init__(
        self,
        in_channels: int = 4,
        image_size: tuple[int, int] = (128, 128),
        hidden_dims: list[int] | None = None,
        output_dim: int = 256,
        kernel_size: int = 3,
        pool_size: int = 2,
        # ── 残差模式参数 ──
        stem_cfg: dict[str, Any] | None = None,
        stages: list[dict[str, Any]] | None = None,
        block_cfg: dict[str, Any] | None = None,
        head_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self._is_residual = stages is not None

        if self._is_residual:
            self._build_residual(in_channels, stem_cfg, stages, block_cfg, head_cfg)
        else:
            self._build_plain(in_channels, hidden_dims, output_dim, kernel_size, pool_size)

    # ══════════════════════════════════════════════════════
    # 普通 CNN 模式（向后兼容）
    # ══════════════════════════════════════════════════════

    def _build_plain(
        self,
        in_channels: int,
        hidden_dims: list[int] | None,
        output_dim: int,
        kernel_size: int,
        pool_size: int,
    ) -> None:
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]

        self.output_dim = output_dim
        self._output_mode = "vector"

        layers = []
        in_dim = in_channels

        for out_dim in hidden_dims:
            layers.append(
                nn.Conv2d(
                    in_dim, out_dim,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_size))
            in_dim = out_dim

        self.conv_net = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        conv_output_dim = hidden_dims[-1] * 4 * 4

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    # ══════════════════════════════════════════════════════
    # 残差 ResNet 模式
    # ══════════════════════════════════════════════════════

    def _build_residual(
        self,
        in_channels: int,
        stem_cfg: dict[str, Any] | None,
        stages: list[dict[str, Any]],
        block_cfg: dict[str, Any] | None,
        head_cfg: dict[str, Any] | None,
    ) -> None:
        block_cfg = block_cfg or {}
        self._kernel_size = block_cfg.get("kernel_size", 3)
        self._activation = block_cfg.get("activation", "relu")
        self._norm = block_cfg.get("norm", "batch_norm")
        self._dropout = block_cfg.get("dropout", 0.0)
        self._norm_eps = block_cfg.get("norm_eps", 1e-3)
        self._norm_momentum = block_cfg.get("norm_momentum", 0.1)
        self._norm_num_groups = block_cfg.get("norm_num_groups", 8)

        # 1. Stem
        self.stem, stem_ch = self._build_stem(stem_cfg, in_channels)

        # 2. Stages
        self.stages, final_ch = self._build_stages(stages, stem_ch)

        # 3. 计算最终特征图尺寸
        h, w = self.image_size
        fh, fw = self._compute_feature_size(h, w, stem_cfg, stages)

        # 4. Head
        self._build_head(head_cfg, final_ch, fh, fw)

    def _build_stem(
        self, stem_cfg: dict[str, Any] | None, in_channels: int
    ) -> tuple[nn.Module, int]:
        """构建 Stem。返回 (module, output_channels)。"""
        if stem_cfg is None:
            return nn.Identity(), in_channels

        layers: list[nn.Module] = []
        ch = stem_cfg.get("channels", 64)
        kernel = stem_cfg.get("kernel", 7)
        stride = stem_cfg.get("stride", 2)
        padding = kernel // 2
        _nk = dict(eps=self._norm_eps, momentum=self._norm_momentum, num_groups=self._norm_num_groups)

        layers.append(
            nn.Conv2d(in_channels, ch, kernel, stride, padding=padding, bias=False)
        )
        layers.append(_make_norm(self._norm, ch, **_nk))
        layers.append(_make_act(self._activation))

        if stem_cfg.get("pool", False):
            pk = stem_cfg.get("pool_kernel", 3)
            ps = stem_cfg.get("pool_stride", 2)
            layers.append(nn.MaxPool2d(pk, ps, padding=pk // 2))

        return nn.Sequential(*layers), ch

    def _build_stages(
        self, stages: list[dict[str, Any]], in_channels: int
    ) -> tuple[nn.Module, int]:
        """构建残差阶段序列。返回 (module, final_channels)。"""
        seq: list[nn.Module] = []
        in_ch = in_channels
        _nk = dict(
            norm_eps=self._norm_eps,
            norm_momentum=self._norm_momentum,
            norm_num_groups=self._norm_num_groups,
        )

        for stage in stages:
            out_ch = stage["channels"]
            num_blocks = stage.get("blocks", 2)
            stage_stride = stage.get("stride", 1)

            for i in range(num_blocks):
                # 只有每阶段第一个 block 带降采样 stride
                stride = stage_stride if i == 0 else 1
                seq.append(
                    ResidualBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        stride=stride,
                        kernel_size=self._kernel_size,
                        activation=self._activation,
                        norm=self._norm,
                        dropout=self._dropout,
                        **_nk,
                    )
                )
                in_ch = out_ch

        return nn.Sequential(*seq), in_ch

    def _compute_feature_size(
        self,
        h: int, w: int,
        stem_cfg: dict[str, Any] | None,
        stages: list[dict[str, Any]],
    ) -> tuple[int, int]:
        """预计算经过 stem + stages 后的空间尺寸。"""
        # Stem
        if stem_cfg is not None:
            k = stem_cfg.get("kernel", 7)
            s = stem_cfg.get("stride", 2)
            p = k // 2
            h = (h + 2 * p - k) // s + 1
            w = (w + 2 * p - k) // s + 1

            if stem_cfg.get("pool", False):
                pk = stem_cfg.get("pool_kernel", 3)
                ps = stem_cfg.get("pool_stride", 2)
                pp = pk // 2
                h = (h + 2 * pp - pk) // ps + 1
                w = (w + 2 * pp - pk) // ps + 1

        # Stages — 只有每阶段首个 block 可能降采样
        for stage in stages:
            stride = stage.get("stride", 1)
            if stride > 1:
                k = self._kernel_size
                p = k // 2
                h = (h + 2 * p - k) // stride + 1
                w = (w + 2 * p - k) // stride + 1

        return h, w

    def _build_head(
        self,
        head_cfg: dict[str, Any] | None,
        final_channels: int,
        feat_h: int,
        feat_w: int,
    ) -> None:
        """构建输出头。

        - 无 head_cfg → tokens 模式：输出 [B, H×W, C] 空间特征
        - 有 head_cfg → vector 模式：Pool → Flatten → MLP → [B, D]
        """
        if head_cfg is None:
            self._output_mode = "tokens"
            self.output_dim = final_channels  # token 嵌入维度
            self.head = nn.Identity()
            return

        self._output_mode = "vector"
        layers: list[nn.Module] = []

        # 全局池化（可选）
        pool_type = head_cfg.get("global_pool")
        pool_size = head_cfg.get("pool_size", [4, 4])
        if pool_type == "adaptive_avg":
            layers.append(nn.AdaptiveAvgPool2d(tuple(pool_size)))
            feat_h, feat_w = pool_size[0], pool_size[1]
        elif pool_type == "adaptive_max":
            layers.append(nn.AdaptiveMaxPool2d(tuple(pool_size)))
            feat_h, feat_w = pool_size[0], pool_size[1]

        layers.append(nn.Flatten())
        in_dim = final_channels * feat_h * feat_w

        # MLP 隐藏层（可选）
        # 注意：当 in_dim 很大（如 4096）时，隐藏层会引入大量参数，
        # 容易导致梯度坍塌。若无特殊需求，建议省略 hidden 或使用较小值。
        hidden = head_cfg.get("hidden", [])
        activation = head_cfg.get("activation", "relu")
        for hdim in hidden:
            layers.append(nn.Linear(in_dim, hdim))
            # 在 ReLU 前加 LayerNorm 防止大量神经元死亡
            layers.append(nn.LayerNorm(hdim))
            layers.append(_make_act(activation))
            in_dim = hdim

        # 输出层（单 Linear，无激活函数）
        out_dim = head_cfg.get("output_dim", 256)
        layers.append(nn.Linear(in_dim, out_dim))

        # 输出归一化（防止 Linear 输出 scale 失控）
        if head_cfg.get("output_norm", True):
            layers.append(nn.LayerNorm(out_dim))

        self.output_dim = out_dim
        self.head = nn.Sequential(*layers)

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
        if self._is_residual:
            return self._forward_residual(rgbd)

        # 普通 CNN 路径
        x = self.conv_net(rgbd)
        x = self.adaptive_pool(x)
        vector = self.head(x)
        return EncoderOutput(vector=vector, tokens=None)

    def _forward_residual(self, rgbd: torch.Tensor) -> EncoderOutput:
        x = self.stem(rgbd)
        x = self.stages(x)

        if self._output_mode == "vector":
            vector = self.head(x)
            return EncoderOutput(vector=vector, tokens=None)

        # Tokens 模式：[B, C, H, W] → [B, H×W, C]
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        return EncoderOutput(vector=None, tokens=tokens)

    # ══════════════════════════════════════════════════════
    # Utility
    # ══════════════════════════════════════════════════════

    def get_output_dim(self) -> int:
        """返回输出维度。"""
        return self.output_dim

    def warmup_bn(
        self,
        dataloader: Any,
        num_batches: int = 50,
        device: str = "cpu",
    ) -> None:
        """对 BatchNorm 层进行预热，使用训练数据的 batch 统计值更新 running stats。

        在以下场景使用：
        - 训练完成后，BN running stats 偏离实际分布时
        - 加载旧 checkpoint 后，BN running stats 损坏时

        Args:
            dataloader: 训练数据加载器（迭代返回 {"camera": [B,4,H,W]} 的 dict）
            num_batches: 预热批次数
            device: 计算设备
        """
        if not self._is_residual:
            return

        # 切换到训练模式（BN 使用 batch stats 并更新 running stats）
        self.train()
        # 冻结所有可学习参数，只更新 BN running stats
        for param in self.parameters():
            param.requires_grad = False

        bn_count = 0
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                camera = batch["camera"] if isinstance(batch, dict) else batch
                if isinstance(camera, (list, tuple)):
                    camera = camera[0]
                camera = camera.to(device)
                self.to(device)
                _ = self.forward(camera)
                bn_count += 1

        # 恢复可学习参数状态
        for param in self.parameters():
            param.requires_grad = True

        # 恢复 eval 模式
        self.eval()

        # 报告
        if bn_count > 0:
            # 采样第一个 BN 的 running_var
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    print(
                        f"  BN warmup: {bn_count} batches, "
                        f"first BN running_var mean={m.running_var.mean():.4f}"
                    )
                    break
