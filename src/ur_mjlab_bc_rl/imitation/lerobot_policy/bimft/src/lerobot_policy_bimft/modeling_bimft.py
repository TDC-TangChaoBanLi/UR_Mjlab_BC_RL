"""BiMFT LeRobot Policy — PreTrainedPolicy 子类。

包装 BiMFT_Policy，实现 LeRobot 训练和推理接口。

LeRobot batch → PolicyBatch 映射:
  - 多相机 RGB + Depth → 合并为 RGB-D [B, T, 4, H, W]
  - observation.state → 拆分为左右 [B, T, R, 7]
  - 力/力矩从自定义 key 获取（observation.wrench_left / wrench_right）
  - 时间偏移从帧索引自动计算
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

from .configuration_bimft import BiMFTConfig
from .processor_bimft import make_bimft_pre_post_processors


# ── 项目根目录 ─────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[7]


# ── 工具函数 ────────────────────────────────────────

def _build_rgbd_from_batch(
    batch: dict[str, torch.Tensor],
    camera_key: str,
    n_obs_steps: int,
) -> torch.Tensor | None:
    """从 LeRobot batch 提取单路相机的 RGB-D tensor。

    LeRobot batch 中图像 key 格式:
      observation.images.{camera_key}.rgb      → [B T C H W] or [B C H W]
      observation.images.{camera_key}.depth    → [B T C H W] or [B C H W]

    返回: [B, T, 4, H, W] 或 None（相机不存在时）
    """
    rgb_key = f"observation.images.{camera_key}.rgb"
    depth_key = f"observation.images.{camera_key}.depth"

    if rgb_key not in batch:
        return None

    rgb = batch[rgb_key]
    depth = batch.get(depth_key)

    # 统一 dtype 并确保有时序维度
    if rgb.dtype == torch.uint8:
        rgb = rgb.float() / 255.0
    if rgb.ndim == 4:
        rgb = rgb.unsqueeze(1)  # [B, C, H, W] → [B, 1, C, H, W]

    if depth is not None:
        if depth.dtype == torch.uint8:
            depth = depth.float() / 255.0
        if depth.ndim == 4:
            depth = depth.unsqueeze(1)
        # 确保 depth 是单通道
        if depth.ndim == 5 and depth.shape[2] > 1:
            depth = depth[:, :, :1, :, :]
    else:
        # 无深度时用零填充
        depth = torch.zeros(
            rgb.shape[0], rgb.shape[1], 1, rgb.shape[3], rgb.shape[4],
            device=rgb.device, dtype=rgb.dtype,
        )

    # 扩展时序维度到 n_obs_steps: 如果只有 1 帧，同时 repeat rgb 和 depth
    actual_T = rgb.shape[1]
    if actual_T < n_obs_steps:
        repeat_times = n_obs_steps // actual_T
        rgb = rgb.repeat(1, repeat_times, 1, 1, 1)
        depth = depth.repeat(1, repeat_times, 1, 1, 1)

    return torch.cat([rgb, depth], dim=2)  # [B, T, 4, H, W]


def _split_left_right_state(
    state: torch.Tensor,
    left_dim: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """将合并的 state [..., 14] 拆分为 left [..., 7], right [..., 7].

    LeRobot 的 observation.state 前 7 维 = 左臂，后 7 维 = 右臂。
    """
    left = state[..., :left_dim]
    right = state[..., left_dim:]
    return left, right


def _build_policy_batch(
    batch: dict[str, torch.Tensor],
    config: BiMFTConfig,
) -> dict[str, torch.Tensor]:
    """LeRobot batch → BiMFT PolicyBatch。

    期望 LeRobot batch key (实际数据集格式):
      observation.images.A_realsense_link_CAMERA.rgb, .depth  → 左腕
      observation.images.B_realsense_link_CAMERA.rgb, .depth  → 右腕
      observation.images.global_realsense_link_CAMERA.rgb, .depth → 全局
      observation.state.joint.position → [B, T, 14] (左7 + 右7)
      observation.state.sensor.force    → [B, T, 6]  (可选, 左3 + 右3)
      observation.state.sensor.torque   → [B, T, 6]  (可选, 左3 + 右3)
      action                            → [B, horizon, 14]
      action_is_pad                     → [B, horizon] (可选)
    """
    # 自动检测 state key 格式
    state_key = None
    for key in ["observation.state", "observation.state.joint.position"]:
        if key in batch:
            state_key = key
            break
    if state_key is None:
        raise ValueError(f"未找到 state key, batch keys: {sorted(batch.keys())}")

    device = batch[state_key].device
    B = batch[state_key].shape[0]
    T = config.n_obs_steps
    R = 3  # high_rate_per_frame

    # ── 1. 三路相机 ──
    # 实际相机名称来自场景配置: A_realsense_link_CAMERA, B_realsense_link_CAMERA, global_realsense_link_CAMERA
    camera_names = [
        "A_realsense_link_CAMERA",
        "B_realsense_link_CAMERA",
        "global_realsense_link_CAMERA",
    ]

    left_rgbd = _build_rgbd_from_batch(batch, camera_names[0], T)
    right_rgbd = _build_rgbd_from_batch(batch, camera_names[1], T)
    global_rgbd = _build_rgbd_from_batch(batch, camera_names[2], T)

    if left_rgbd is None or right_rgbd is None or global_rgbd is None:
        # 回退: 尝试旧 key 格式
        left_rgbd = left_rgbd or _build_rgbd_from_batch(batch, "left_wrist", T)
        right_rgbd = right_rgbd or _build_rgbd_from_batch(batch, "right_wrist", T)
        global_rgbd = global_rgbd or _build_rgbd_from_batch(batch, "global", T)

    if left_rgbd is None or right_rgbd is None or global_rgbd is None:
        raise ValueError(
            f"未找到三路相机输入。可用相机 key: "
            f"{[k for k in batch if 'observation.images' in k]}"
        )

    # ── 2. 关节+夹爪状态 ──
    state = batch[state_key]  # [B, ?] — 形状取决于是否堆叠了 delta indices
    # 可能形状:
    #   [B, inner_T, D]       — 无 n_obs_steps 堆叠 (original)
    #   [B, T_obs, inner_T, D] — delta indices 堆叠后
    #   [B, T_obs, 1, inner_T, D] — delta indices + 维度展开

    # 规范化: squeeze 多余的 1 维（不 squeeze batch dim），确保最终是 [B, ...]
    # 可能的形状: [B, T_obs, 1, inner_T, D] → [B, T_obs, inner_T, D]
    if state.ndim == 5 and state.shape[2] == 1:
        state = state.squeeze(2)

    # 现在处理不同形状
    if state.ndim == 3:
        # [B, T_actual, D] → 对齐到 T
        actual_state_T = state.shape[1]
        if actual_state_T < T:
            pad = torch.zeros(B, T - actual_state_T, state.shape[-1], device=device, dtype=state.dtype)
            state = torch.cat([state, pad], dim=1)
        elif actual_state_T > T:
            state = state[:, :T, :]
        # 扩展为 [B, T, R, D]
        state = state.unsqueeze(2).expand(B, T, R, state.shape[-1])
    elif state.ndim == 4:
        # [B, T_obs, inner_T, D] — delta indices 堆叠后
        state = state[:, :T, :, :]  # 截断到 T
        R_actual = state.shape[2]
        if R_actual < R:
            state = state.unsqueeze(2).expand(B, T, R, state.shape[-1])

    # 拆分左右: 前 7 维 = 左臂, 后 7 维 = 右臂
    joint_dim_per_arm = state.shape[-1] // 2
    left_joint, right_joint = _split_left_right_state(state, left_dim=joint_dim_per_arm)

    # ── 3. 力/力矩 → wrench [B, actual_T, R, 6] ──
    force_key = "observation.state.sensor.force"
    torque_key = "observation.state.sensor.torque"

    def _build_wrench(f_key: str | None, t_key: str | None, start_idx: int) -> torch.Tensor:
        """组合 force(3) + torque(3) → wrench(6)，expands to [B, T_w, R, 6]."""
        force = batch.get(f_key) if f_key else None
        torque = batch.get(t_key) if t_key else None

        if force is not None and torque is not None:
            f_dim = force.shape[-1] // 2  # 3
            f_arm = force[..., start_idx * f_dim:(start_idx + 1) * f_dim]
            t_arm = torque[..., start_idx * f_dim:(start_idx + 1) * f_dim]
            wrench = torch.cat([f_arm, t_arm], dim=-1)  # [B, T_w, 6]
        elif force is not None:
            f_dim = force.shape[-1] // 2
            f_arm = force[..., start_idx * f_dim:(start_idx + 1) * f_dim]
            wrench = torch.cat([f_arm, torch.zeros_like(f_arm)], dim=-1)
        else:
            return torch.zeros(B, actual_state_T, R, 6, device=device)

        actual_T = wrench.shape[1]
        if wrench.ndim == 3:
            wrench = wrench.unsqueeze(2).expand(B, actual_T, R, 6)
        elif wrench.ndim == 4 and wrench.shape[2] != R:
            wrench = wrench.expand(B, actual_T, R, 6)
        return wrench

    left_wrench = _build_wrench(force_key, torque_key, start_idx=0)
    right_wrench = _build_wrench(force_key, torque_key, start_idx=1)

    # 对齐 T 维度
    for wrench in [left_wrench, right_wrench]:
        if hasattr(wrench, 'shape') and wrench.ndim >= 2:
            continue  # already handled in _build_wrench
    if left_wrench.shape[1] < T:
        pad = torch.zeros(B, T - left_wrench.shape[1], R, 6, device=device)
        left_wrench = torch.cat([left_wrench, pad], dim=1)
        right_wrench = torch.cat([right_wrench, pad], dim=1)
    elif left_wrench.shape[1] > T:
        left_wrench = left_wrench[:, :T, :, :]
        right_wrench = right_wrench[:, :T, :, :]

    # ── 4. 时间偏移 ──
    # 相机帧: 4 帧 @ 30Hz → 间隔 1/30 ≈ 0.0333s
    frame_interval = 1.0 / 30.0
    image_offsets = torch.linspace(
        -(T - 1) * frame_interval, 0.0, T,
        device=device,
    ).unsqueeze(0).expand(B, T)  # [B, T]

    # 高频采样: 每帧内 N 个 @ 90Hz → 间隔 1/90 ≈ 0.0111s
    # 使用 max(joint_high_rate, force_high_rate) 作为时间网格
    high_rate_interval = 1.0 / 90.0
    max_r = max(R, R)  # R 在函数顶部定义为 3
    high_rate_offsets = torch.linspace(
        -(T * R - 1) * high_rate_interval, 0.0, T * R,
        device=device,
    ).reshape(1, T, R).expand(B, T, R)  # [B, T, R]

    return {
        "left_wrist_rgbd": left_rgbd,
        "right_wrist_rgbd": right_rgbd,
        "global_rgbd": global_rgbd,
        "left_joint_gripper": left_joint,
        "right_joint_gripper": right_joint,
        "left_wrench": left_wrench,
        "right_wrench": right_wrench,
        "image_time_offsets": image_offsets,
        "high_rate_time_offsets": high_rate_offsets,
    }


def _load_bimft_config(config: BiMFTConfig) -> "PolicyConfig":
    """从 YAML 加载 PolicyConfig。

    延迟导入避免循环依赖。
    """
    from ur_mjlab_bc_rl.models.BiMFT.BiMFT_Policy import PolicyConfig

    cfg_path = _PROJECT_ROOT / config.model_cfg_path
    if cfg_path.exists():
        with open(cfg_path) as f:
            yaml_cfg = yaml.safe_load(f) or {}
    else:
        logging.warning(f"模型配置文件不存在: {cfg_path}，使用默认配置。")
        return PolicyConfig()

    inp = yaml_cfg.get("input", {})
    tok = yaml_cfg.get("token", {})
    vis = yaml_cfg.get("vision", {})
    st = yaml_cfg.get("state", {})
    sf = yaml_cfg.get("slot_fusion", {})
    tmp = yaml_cfg.get("temporal", {})
    bim = yaml_cfg.get("bimanual", {})
    act = yaml_cfg.get("action", {})
    los = yaml_cfg.get("loss", {})

    return PolicyConfig(
        image_channels=inp.get("image_channels", 4),
        image_height=inp.get("image_height", 460),
        image_width=inp.get("image_width", 640),
        vision_history=inp.get("vision_history", config.n_obs_steps),
        joint_high_rate=inp.get("joint_high_rate", 3),
        force_high_rate=inp.get("force_high_rate", 3),
        joint_dim=inp.get("joint_dim", 6),
        gripper_state_dim=inp.get("gripper_state_dim", 1),
        wrench_dim=inp.get("wrench_dim", 6),
        d_model=tok.get("d_model", 512),
        n_heads=tok.get("n_heads", 8),
        dim_feedforward=tok.get("dim_feedforward", 2048),
        dropout=tok.get("dropout", 0.1),
        pretrained_rgb=vis.get("pretrained_rgb", True),
        vision_grid_h=vis.get("grid_h", 15),
        vision_grid_w=vis.get("grid_w", 20),
        share_global_backbone=vis.get("share_global_backbone", False),
        joint_encoder_layers=st.get("joint_encoder_layers", 2),
        force_encoder_layers=st.get("force_encoder_layers", 2),
        slot_fusion_layers=sf.get("layers", 2),
        joint_summary_tokens=sf.get("joint_summary_tokens", 1),
        force_summary_tokens=sf.get("force_summary_tokens", 1),
        arm_temporal_layers=tmp.get("arm_layers", 2),
        global_temporal_layers=tmp.get("global_layers", 2),
        arm_summary_tokens=bim.get("arm_summary_tokens", 1),
        global_queries_per_arm=bim.get("global_queries_per_arm", 4),
        coordination_queries=bim.get("coordination_queries", 8),
        coordination_layers=bim.get("coordination_layers", 4),
        reinjection_layers=bim.get("reinjection_layers", 1),
        action_chunk_size=act.get("chunk_size", config.horizon),
        action_dim=act.get("action_dim", 7),
        action_decoder_layers=act.get("decoder_layers", 4),
        action_head_hidden_dim=act.get("head_hidden_dim", 256),
        huber_delta=los.get("huber_delta", 1.0),
        smoothness_weight=los.get("smoothness_weight", 0.05),
        max_parameters=yaml_cfg.get("max_parameters", 300_000_000),
    )


# ═══════════════════════════════════════════════════════════
# BiMFT LeRobot Policy
# ═══════════════════════════════════════════════════════════

class BiMFTPolicy(PreTrainedPolicy):
    """BiMFT 双臂多模态融合 Transformer 策略 — LeRobot 接口。

    使用 --policy.type=bimft 在 LeRobot 训练脚本中使用。
    """

    config_class = BiMFTConfig
    name = "bimft"

    def __init__(
        self,
        config: BiMFTConfig,
        dataset_stats: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        config.validate_features()

        self.config = config
        self._loss_type = config.loss_type

        # 延迟导入：避免模块加载时的循环依赖
        from ur_mjlab_bc_rl.models.BiMFT.BiMFT_Policy import BiMFT_Policy
        from ur_mjlab_bc_rl.models.BiMFT.losses import bimft_total_loss

        policy_cfg = _load_bimft_config(config)

        # 用 LeRobot 的 n_obs_steps / horizon 覆盖 YAML 中的值
        policy_cfg.vision_history = config.n_obs_steps
        policy_cfg.action_chunk_size = config.horizon

        # 从 output_features 推断 action_dim
        action_feat = config.output_features.get("action")
        if action_feat is not None:
            # shape 可能是 [3, 14] (3采样×14维) 或 [14] — 取最后一维
            total_action_dim = action_feat.shape[-1] if hasattr(action_feat, 'shape') else action_feat[1]
            policy_cfg.action_dim = total_action_dim // 2

        self.model = BiMFT_Policy(policy_cfg)
        self._bimft_loss_fn = bimft_total_loss
        self._dataset_stats = dataset_stats

    # ── LeRobot 必需接口 ─────────────────────────────

    def _save_pretrained(
        self, save_directory: Path, state_dict: dict[str, torch.Tensor] | None = None
    ) -> None:
        """保存模型权重、配置和 preprocessor/postprocessor pipeline。

        覆盖父类方法，额外保存:
          - policy_preprocessor.json  (归一化 + 设备转移)
          - policy_postprocessor.json (action 反归一化)
        """
        save_directory = Path(save_directory)
        super()._save_pretrained(save_directory, state_dict=state_dict)

        # 创建并保存 preprocessor / postprocessor
        # 注意: 必须始终包含 normalizer/unnormalizer 步骤，
        # 否则 lerobot resume 时的 processor 覆盖会失败。
        try:
            stats = self._dataset_stats if self._dataset_stats else {}
            preprocessor, postprocessor = make_bimft_pre_post_processors(
                config=self.config,
                dataset_stats=stats,
            )
            preprocessor.save_pretrained(
                save_directory, config_filename="policy_preprocessor.json"
            )
            postprocessor.save_pretrained(
                save_directory, config_filename="policy_postprocessor.json"
            )
        except Exception:
            logging.warning(
                "无法创建 preprocessor/postprocessor，跳过保存。"
                " resume 时需要手动创建空的 policy_preprocessor.json。"
            )

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        """训练前向: 计算损失。

        Returns:
            (loss, info_dict)
        """
        # LeRobot batch → PolicyBatch
        policy_batch = _build_policy_batch(batch, self.config)

        # 前向
        out = self.model(policy_batch)

        # 目标动作 — 数据集 key 已改为 "action"
        target = batch.get(ACTION)
        if target is None:
            logging.warning(f"action is None. Available: {sorted(batch.keys())}")
            return torch.tensor(0.0, requires_grad=True), {
                "action_loss": 0.0, "smooth_loss": 0.0,
                "left_loss": 0.0, "right_loss": 0.0,
            }

        # 确保 target 在正确设备上
        if target.device != policy_batch["left_wrist_rgbd"].device:
            target = target.to(policy_batch["left_wrist_rgbd"].device)

        # target: 处理 delta 堆叠维度 → reshape 为 [B, K, 14]
        K = self.model.cfg.action_chunk_size
        # squeeze 多余的 1 维: [B, 1, 1, 3, 14] → [B, 3, 14]
        #                 或 [B, 10, 1, 3, 14] → [B, 10, 3, 14]
        while target.ndim > 3:
            if target.shape[1] == 1:
                target = target.squeeze(1)
            else:
                break
        # 如果有多步 delta: [B, 10, 3, 14] → reshape → [B, 30, 14]
        if target.ndim == 4:
            T_delta, S = target.shape[1], target.shape[2]
            target = target.reshape(target.shape[0], T_delta * S, -1)  # [B, T_delta*S, 14]
        # 对齐到 K 步
        if target.shape[1] >= 3 and target.ndim == 3:
            target = target[:, -K:, :] if target.shape[1] > K else target
            if target.shape[1] < K:
                pad = target[:, -1:, :].expand(-1, K - target.shape[1], -1)
                target = torch.cat([target, pad], dim=1)
        elif target.ndim == 2:
            target = target.unsqueeze(1).expand(-1, K, -1)

        # 拆分左右目标: 前 action_dim = 左臂, 后 action_dim = 右臂
        action_dim = self.model.cfg.action_dim
        target_left = target[..., :action_dim]
        target_right = target[..., action_dim:]

        # padding mask — 对齐到 target 的 K 步
        is_pad = batch.get("action_is_pad")
        if is_pad is None:
            is_pad = torch.zeros(
                target.shape[0], target.shape[1],
                dtype=torch.bool, device=target.device,
            )
        elif is_pad.shape[1] < target.shape[1]:
            # 从 [B, 10] 扩展到 [B, 30]: 每行 3 个采样重复
            S = target.shape[1] // is_pad.shape[1]  # 3
            is_pad = is_pad.unsqueeze(-1).expand(-1, -1, S).reshape(target.shape[0], -1)

        # 计算 BiMFT 损失
        losses = self._bimft_loss_fn(
            pred_left=out["left_action"],
            pred_right=out["right_action"],
            target_left=target_left,
            target_right=target_right,
            is_pad_left=is_pad,
            is_pad_right=is_pad,
            huber_delta=self.model.cfg.huber_delta,
            smoothness_weight=self.model.cfg.smoothness_weight,
            joint_dim=self.model.cfg.joint_dim,
        )

        return losses["loss"], {
            "action_loss": losses["action_loss"].item(),
            "smooth_loss": losses["smooth_loss"].item(),
            "left_loss": losses["left_loss"].item(),
            "right_loss": losses["right_loss"].item(),
        }

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """推理: 返回当前步的动作（拼接左右）。"""
        self.eval()
        policy_batch = _build_policy_batch(batch, self.config)
        out = self.model(policy_batch)

        # 取动作块的第一步 → 拼接左右
        left = out["left_action"][:, 0, :]   # [B, D_a]
        right = out["right_action"][:, 0, :] # [B, D_a]
        return torch.cat([left, right], dim=-1)  # [B, 2*D_a]

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """返回完整动作块 [B, horizon, 2*D_a]（拼接左右）。"""
        self.eval()
        policy_batch = _build_policy_batch(batch, self.config)
        out = self.model(policy_batch)

        left = out["left_action"]        # [B, K, D_a]
        right = out["right_action"]      # [B, K, D_a]
        return torch.cat([left, right], dim=-1)  # [B, K, 2*D_a]

    def reset(self) -> None:
        """重置策略状态（BiMFT 无状态，无需操作）。"""
        pass

    def get_optim_params(self) -> dict:
        """返回优化器参数分组。

        预训练 ResNet backbone 使用较小学习率。
        """
        backbone_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name or "conv1" in name or "bn1" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        return [
            {"params": backbone_params, "lr": self.config.optimizer_lr * 0.1},
            {"params": other_params},
        ]
