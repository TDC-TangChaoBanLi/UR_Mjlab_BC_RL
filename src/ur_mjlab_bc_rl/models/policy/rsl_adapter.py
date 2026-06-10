"""RSL-RL 适配器 — 将 UR5MultimodalBackbone 封装为 mjlab PPO 可用模型。

提供:
  UR5RslActorModel  — 薄适配器 (nn.Module, RSL-RL 接口)
  UR5MultimodalModelCfg — 自定义配置类 (替代 RslRlModelCfg)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import HiddenState
from rsl_rl.utils import resolve_callable

from .multimodal_backbone import build_actor


class UR5RslActorModel(nn.Module):
    """RSL-RL 薄适配器 — 封装 UR5MultimodalBackbone + RSL-RL Distribution。

    通过 UR5MultimodalModelCfg.class_name 注入到 mjlab PPO 管线。
    内部使用 build_actor() 创建与 BC 完全相同的 UR5MultimodalBackbone。
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        architecture_cfg: dict[str, Any] | None = None,
        distribution_cfg: dict[str, Any] | None = None,
        obs_normalization: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        if architecture_cfg is None:
            raise ValueError("architecture_cfg is required")
        if distribution_cfg is None:
            raise ValueError("distribution_cfg is required")

        self.obs_groups = obs_groups.get(obs_set, [])

        dist_cfg = copy.deepcopy(distribution_cfg)
        dist_class = resolve_callable(dist_cfg.pop("class_name"))
        self.distribution = dist_class(output_dim, **dist_cfg)

        self.backbone = build_actor(architecture_cfg, output_dim=self.distribution.input_dim)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        actor_obs = self._extract_obs(obs)
        distribution_params = self.backbone.get_action_mean(actor_obs)

        if stochastic_output:
            self.distribution.update(distribution_params)
            return self.distribution.sample()
        return self.distribution.deterministic_output(distribution_params)

    def _extract_obs(self, obs: TensorDict) -> dict[str, torch.Tensor]:
        camera_groups = [g for g in self.obs_groups if g.startswith("camera")]
        state_groups = [g for g in self.obs_groups if g != "task" and g != "task_id" and not g.startswith("camera")]

        camera_list = []
        for g in camera_groups:
            cam = obs[g]
            if cam.dtype == torch.uint8:
                cam = cam.float() / 255.0
            camera_list.append(cam)
        camera = torch.cat(camera_list, dim=1) if camera_list else obs.get("camera", obs.get("camera_rgb"))
        actor_state = torch.cat([obs[g] for g in state_groups], dim=-1) if state_groups else obs.get("actor_state")
        task = obs.get("task", obs.get("task_id"))
        if task is None:
            task = torch.zeros(camera.shape[0], 1, device=camera.device)
        return {"camera": camera, "actor_state": actor_state, "task": task}

    # ── RSL-RL 接口 ──
    def update_normalization(self, obs: TensorDict) -> None: pass
    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None: pass
    def get_hidden_state(self) -> HiddenState: return None
    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None: pass

    @property
    def output_mean(self) -> torch.Tensor: return self.distribution.mean
    @property
    def output_std(self) -> torch.Tensor: return self.distribution.std
    @property
    def output_entropy(self) -> torch.Tensor: return self.distribution.entropy
    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]: return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)


@dataclass
class UR5MultimodalModelCfg:
    """UR5 多模态 Actor 的 RSL-RL 模型配置。

    替代 RslRlModelCfg 用于 actor，critic 仍使用标准 RslRlModelCfg。
    architecture_cfg 格式与 BC 的 model_cfg 完全一致。
    """

    class_name: str = "ur_mjlab_bc_rl.models.policy.rsl_adapter:UR5RslActorModel"
    architecture_cfg: dict[str, Any] = field(default_factory=lambda: {
        "visual_encoder": {"type": "rescnn", "output_dim": 256},
        "state_encoder": {"type": "mlp", "output_dim": 128},
        "task_encoder": {"type": "embedding", "num_tasks": 3, "embedding_dim": 32, "output_dim": 64},
        "fusion": {"type": "film"},
        "policy_mlp": {"hidden_dims": [512, 256, 128], "activation": "elu"},
    })
    distribution_cfg: dict[str, Any] | None = field(default_factory=lambda: {
        "class_name": "GaussianDistribution", "init_std": 0.2, "std_type": "scalar",
    })
    obs_normalization: bool = False
    hidden_dims: tuple[int, ...] = ()
    activation: str = "elu"
    cnn_cfg: dict[str, Any] | None = None
    rnn_type: str | None = None
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1
