"""多模态 Actor 骨干网络 — BC 和 PPO 共用。

UR5MultimodalBackbone 是网络核心，通过 build_actor() 构造。
BC 端直接使用，PPO 端通过 rsl_adapter.py 中的 UR5RslActorModel 封装。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..vision.encoder_factory import build_visual_encoder
from ..state.encoder_factory import build_state_encoder
from ..task.encoder_factory import build_task_encoder
from ..fusion.factory import build_fusion
from ..modules.mlp import MLP
from ..specs import EncoderOutput


def build_actor(
    architecture_cfg: dict[str, Any],
    output_dim: int,
) -> "UR5MultimodalBackbone":
    """BC 和 PPO 共用的 Actor 网络构造函数。

    architecture_cfg:
        visual_encoder  — {"type": "rescnn", "output_dim": 256, ...}
        state_encoder   — {"type": "mlp", "output_dim": 128, ...}
        task_encoder    — {"type": "embedding", "num_tasks": 3, ...}
        fusion          — {"type": "film", ...}
        policy_mlp      — {"hidden_dims": [512,256,128], "activation": "elu"}
    """
    visual_cfg = architecture_cfg.get("visual_encoder", {"type": "rescnn", "output_dim": 256})
    state_cfg = architecture_cfg.get("state_encoder", {"type": "mlp", "output_dim": 128})
    task_cfg = architecture_cfg.get("task_encoder", {"type": "embedding", "num_tasks": 3, "embedding_dim": 32, "output_dim": 64})
    fusion_cfg = architecture_cfg.get("fusion", {"type": "film"})
    policy_cfg = architecture_cfg.get("policy_mlp", {"hidden_dims": [512, 256, 128], "activation": "elu"})

    return UR5MultimodalBackbone(
        visual_cfg=visual_cfg,
        state_cfg=state_cfg,
        task_cfg=task_cfg,
        fusion_cfg=fusion_cfg,
        policy_cfg=policy_cfg,
        output_dim=output_dim,
    )


class UR5MultimodalBackbone(nn.Module):
    """BC 和 PPO 共用的多模态 Actor 网络核心。

    BC 端直接使用：
        actor = build_actor(cfg, output_dim=7)
        action = actor(obs, deterministic=True)
        action, log_prob = actor.act_training(obs)

    PPO 端通过 UR5RslActorModel 封装。

    输入: {"camera": [B,4,H,W], "actor_state": [B,D], "task": [B]}
    输出: [B, output_dim]
    """

    def __init__(
        self,
        visual_cfg: dict[str, Any] | None = None,
        state_cfg: dict[str, Any] | None = None,
        task_cfg: dict[str, Any] | None = None,
        fusion_cfg: dict[str, Any] | None = None,
        policy_cfg: dict[str, Any] | None = None,
        output_dim: int = 7,
        model_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        if model_cfg is not None:
            visual_cfg = model_cfg.get("visual_encoder", {"type": "rescnn", "output_dim": 256})
            state_cfg = model_cfg.get("state_encoder", {"type": "mlp", "output_dim": 128})
            task_cfg = model_cfg.get("task_encoder", {"type": "embedding", "num_tasks": 3, "embedding_dim": 32, "output_dim": 64})
            fusion_cfg = model_cfg.get("fusion", {"type": "film"})
            policy_cfg = model_cfg.get("policy_mlp", {"hidden_dims": [512, 256, 128], "activation": "elu"})
            output_dim = model_cfg.get("action_dim", output_dim)

        visual_cfg = visual_cfg or {"type": "rescnn", "output_dim": 256}
        state_cfg = state_cfg or {"type": "mlp", "input_dim": 14, "hidden_dims": [128, 128], "output_dim": 128}
        task_cfg = task_cfg or {"type": "embedding", "num_tasks": 3, "embedding_dim": 32, "output_dim": 64}
        fusion_cfg = fusion_cfg or {"type": "film"}
        policy_cfg = policy_cfg or {"hidden_dims": [512, 256, 128], "activation": "elu"}

        self.visual_encoder = build_visual_encoder(visual_cfg)
        self.state_encoder = build_state_encoder(state_cfg)
        self.task_encoder = build_task_encoder(task_cfg)
        self.fusion = build_fusion(fusion_cfg, self.visual_encoder.get_output_dim(), self.state_encoder.get_output_dim(), self.task_encoder.get_output_dim())
        self.policy_mlp = MLP(input_dim=self.fusion.get_output_dim(), hidden_dims=policy_cfg.get("hidden_dims", [512, 256, 128]), output_dim=output_dim, activation=policy_cfg.get("activation", "elu"))
        self.action_log_std = nn.Parameter(torch.zeros(output_dim))

        # 模态 dropout：训练时随机丢弃某个模态，强制网络依赖其他模态
        self._state_dropout_prob = 0.0
        self._visual_dropout_prob = 0.0

    def set_modality_dropout(
        self, state_dropout: float = 0.0, visual_dropout: float = 0.0
    ) -> None:
        """设置训练时的模态 dropout 概率。

        Args:
            state_dropout: 随机丢弃 state 特征的概率（强制依赖视觉）
            visual_dropout: 随机丢弃视觉特征的概率（强制依赖 state）
        """
        self._state_dropout_prob = float(state_dropout)
        self._visual_dropout_prob = float(visual_dropout)

    def forward(self, obs: dict[str, torch.Tensor], deterministic: bool = True) -> torch.Tensor:
        mean = self.get_action_mean(obs)
        if deterministic:
            return mean
        return mean + torch.exp(self.action_log_std) * torch.randn_like(mean)

    def act_training(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.get_action_mean(obs)
        std = torch.exp(self.action_log_std)
        action = mean + std * torch.randn_like(mean)
        var = std ** 2
        log_prob = -0.5 * (((action - mean) ** 2 / var) + torch.log(2 * torch.pi * var)).sum(dim=-1)
        return action, log_prob

    def get_action_mean(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        camera = obs["camera"]
        actor_state = obs["actor_state"]
        task = obs["task"].long().squeeze(-1)
        visual_out = self.visual_encoder(camera)
        state_out = self.state_encoder(actor_state)
        task_out = self.task_encoder(task)

        # 模态 dropout（仅训练时生效）：
        # 随机丢弃 state，强制网络学习使用视觉特征
        if self.training and self._state_dropout_prob > 0:
            if torch.rand(1, device=actor_state.device).item() < self._state_dropout_prob:
                state_out = EncoderOutput(vector=torch.zeros_like(state_out.get_vector()))
        if self.training and self._visual_dropout_prob > 0:
            if torch.rand(1, device=camera.device).item() < self._visual_dropout_prob:
                visual_out = EncoderOutput(vector=torch.zeros_like(visual_out.get_vector()))

        fused = self.fusion(visual_out, state_out, task_out)
        latent = fused.get_vector() if hasattr(fused, "get_vector") else fused
        return self.policy_mlp(latent)