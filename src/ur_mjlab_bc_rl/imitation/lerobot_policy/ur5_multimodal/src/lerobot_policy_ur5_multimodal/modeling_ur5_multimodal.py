"""UR5 Multimodal BC Policy — LeRobot PreTrainedPolicy 子类。

包装 UR5MultimodalBackbone，实现 LeRobot 要求的接口：
- forward(batch) → (loss, output_dict)
- select_action(batch) → action
- predict_action_chunk(batch) → chunk
- reset()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION

from .configuration_ur5_multimodal import UR5MultimodalConfig


# ── 模块级路径（不触发 mjlab 导入）─────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[7]  # → 项目根目录

# ── Task 映射 ────────────────────────────────────────
TASK_NAME_TO_ID: dict[str, int] = {
    "pick_place": 0,
    "push_t": 1,
    "peg_slot": 2,
    "peg_in_slot": 2,
}


# ── 工具函数 ────────────────────────────────────────

def _map_task_to_id(task_values: list[str] | torch.Tensor) -> torch.Tensor:
    """将 task 字符串列表转为 long tensor [B, 1]。"""
    if isinstance(task_values, torch.Tensor):
        return task_values.long().reshape(-1, 1)
    ids = [TASK_NAME_TO_ID.get(str(t), 0) for t in task_values]
    return torch.tensor(ids, dtype=torch.long).reshape(-1, 1)


def _build_rgbd_camera(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """从 LeRobot batch 构建 RGBD tensor [B, 4, H, W]."""
    rgb = batch["observation.images.rgb"]
    if rgb.dtype == torch.uint8:
        rgb = rgb.float() / 255.0

    depth = batch["observation.images.depth"]
    if depth.dtype == torch.uint8:
        depth = depth.float() / 255.0
    if depth.ndim == 4 and depth.shape[1] == 3:
        depth = depth[:, :1, :, :]
    elif depth.ndim == 3:
        depth = depth.unsqueeze(1)

    return torch.cat([rgb, depth], dim=1)


def _build_obs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """LeRobot batch → obs dict。"""
    # 推断 device
    device = batch["observation.state"].device
    rgbd = _build_rgbd_camera(batch)
    return {
        "camera": rgbd.to(device),
        "actor_state": batch["observation.state"],
        "task": _map_task_to_id(batch.get("task", ["pick_place"])).to(device),
    }


def _load_model_cfg(config: UR5MultimodalConfig) -> dict[str, Any]:
    """加载 YAML 模型配置，与 CLI 参数合并。"""
    cfg_path = _PROJECT_ROOT / config.model_cfg_path
    if cfg_path.exists():
        with open(cfg_path) as f:
            model_cfg = yaml.safe_load(f) or {}
    else:
        logging.warning(f"模型配置文件不存在: {cfg_path}，使用默认架构。")
        model_cfg = {}

    model_cfg.setdefault("visual_encoder", {})
    model_cfg["visual_encoder"]["type"] = config.visual_encoder_type
    model_cfg.setdefault("fusion", {})
    model_cfg["fusion"]["type"] = config.fusion_type
    model_cfg.setdefault("policy_mlp", {})
    model_cfg["policy_mlp"]["hidden_dims"] = list(config.policy_hidden_dims)
    model_cfg["policy_mlp"]["activation"] = config.policy_activation

    return model_cfg


class UR5MultimodalPolicy(PreTrainedPolicy):
    """UR5 多模态 BC 策略。

    内部使用 UR5MultimodalBackbone 作为策略网络，
    兼容 LeRobot 的训练和评估流程。
    """

    config_class = UR5MultimodalConfig
    name = "ur5_multimodal"

    def __init__(
        self,
        config: UR5MultimodalConfig,
        dataset_stats: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        config.validate_features()

        self.config = config
        self._loss_type = config.loss_type

        # 延迟导入：避免模块加载时触发 mjlab → matplotlib → numpy 链
        from ur_mjlab_bc_rl.models.Test_Multimodal.backbone import (
            UR5MultimodalBackbone,
        )

        model_cfg = _load_model_cfg(config)

        state_feat = config.input_features.get("observation.state")
        action_feat = config.output_features.get("action")
        if state_feat is not None:
            model_cfg.setdefault("state_encoder", {})
            model_cfg["state_encoder"]["input_dim"] = state_feat.shape[0]
        if action_feat is not None:
            model_cfg["action_dim"] = action_feat.shape[0]

        self.model = UR5MultimodalBackbone(model_cfg=model_cfg)

        if config.state_dropout > 0:
            self.model.set_modality_dropout(state_dropout=config.state_dropout)

    # ── LeRobot 必需接口 ─────────────────────────────

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        obs = _build_obs(batch)
        pred_action = self.model(obs, deterministic=True)
        target = batch[ACTION]

        # LeRobot stores actions as [B, T, action_dim]; squeeze T if single-step
        if target.ndim == 3 and target.shape[1] == 1:
            target = target.squeeze(1)

        if self._loss_type == "mse":
            loss = F.mse_loss(pred_action, target)
        elif self._loss_type == "l1":
            loss = F.l1_loss(pred_action, target)
        elif self._loss_type == "huber":
            loss = F.smooth_l1_loss(pred_action, target)
        else:
            loss = F.mse_loss(pred_action, target)

        return loss, {"action_loss": loss.item()}

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        obs = _build_obs(batch)
        return self.model(obs, deterministic=True)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        obs = _build_obs(batch)
        action = self.model(obs, deterministic=True)
        return action.unsqueeze(1)

    def reset(self) -> None:
        pass

    def get_optim_params(self) -> dict:
        return [{"params": self.parameters()}]
