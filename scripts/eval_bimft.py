#!/usr/bin/env python3
"""BiMFT 策略评估脚本 — MuJoCo 仿真 + 实时 Viewer 渲染。

用法:
    # 可视化推理（打开 MuJoCo viewer 窗口）
    python scripts/eval_bimft.py \
        --checkpoint outputs/train/2026-07-30/09-27-37_bimft/checkpoints/last/pretrained_model \
        --task dual_pick_place \
        --episodes 5

    # 无头推理（纯数据采集，无 viewer）
    python scripts/eval_bimft.py \
        --checkpoint ... --task dual_pick_place --no-render

时序说明:
  - 环境以 30Hz 运行（= 相机帧率），每步采集 R=3 高频状态采样
  - 策略需要 T=4 帧历史 → 4 帧图像 + 12 帧状态
  - 首 4 步用零填充预热，之后用滑动窗口
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


# ═══════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════

def load_bimft_policy(checkpoint_dir: str, device: str = "cpu"):
    """从 LeRobot 格式的 checkpoint 加载 BiMFT 策略。

    支持两种格式:
      1. LeRobot 训练输出: checkpoints/NNNNNN/ (含 pretrained_model/model.safetensors)
      2. 旧格式 .pt 文件
    """
    import json
    ckpt_path = Path(checkpoint_dir)

    # 查找 pretrained_model 目录
    if ckpt_path.is_dir():
        pt_model_dir = ckpt_path / "pretrained_model"
        if pt_model_dir.exists():
            ckpt_path = pt_model_dir

    print(f"加载 BiMFT 策略: {ckpt_path}")

    # ── 加载 config.json ──
    config_json = ckpt_path / "config.json"
    if config_json.exists():
        with open(config_json) as f:
            raw_cfg = json.load(f)
    else:
        raise FileNotFoundError(f"找不到 config.json: {config_json}")

    from ur_mjlab_bc_rl.lerobot_policy.bimft import BiMFTConfig, BiMFTPolicy

    # 手动构造 BiMFTConfig（避免 draccus 的类型解析问题）
    config = BiMFTConfig()
    # type 是只读属性，由 @register_subclass("bimft") 自动设置
    config.n_obs_steps = raw_cfg.get("n_obs_steps", 4)
    config.horizon = raw_cfg.get("horizon", 30)
    config.n_action_steps = raw_cfg.get("n_action_steps", 3)
    config.device = device
    config.use_amp = raw_cfg.get("use_amp", True)
    config.optimizer_lr = raw_cfg.get("optimizer_lr", 1e-4)
    config.optimizer_weight_decay = raw_cfg.get("optimizer_weight_decay", 1e-4)
    config.scheduler_warmup_steps = raw_cfg.get("scheduler_warmup_steps", 500)
    config.model_cfg_path = raw_cfg.get("model_cfg_path", "configs/model/bimft.yaml")
    config.loss_type = raw_cfg.get("loss_type", "huber")

    # 加载 input/output features（从保存的 JSON 恢复 PolicyFeature）
    from lerobot.configs.types import FeatureType, PolicyFeature
    input_features = {}
    for k, v in raw_cfg.get("input_features", {}).items():
        input_features[k] = PolicyFeature(
            type=FeatureType[v["type"]],
            shape=tuple(v["shape"]),
        )
    output_features = {}
    for k, v in raw_cfg.get("output_features", {}).items():
        output_features[k] = PolicyFeature(
            type=FeatureType[v["type"]],
            shape=tuple(v["shape"]),
        )
    config.input_features = input_features
    config.output_features = output_features
    config.normalization_mapping = raw_cfg.get("normalization_mapping", {})

    # ── 加载归一化参数 ──
    dataset_stats = None
    try:
        stats_files = [
            "policyprocessorpipeline_step_1_normalizer_processor.safetensors",
            "dataprocessorpipeline_step_1_normalizer_processor.safetensors",
        ]
        for sf in stats_files:
            sp = ckpt_path / sf
            if sp.exists():
                from safetensors.torch import load_file
                dataset_stats = load_file(str(sp))
                print(f"  ✓ 归一化参数加载: {sf}")
                break
    except Exception as e:
        print(f"  [WARN] 归一化参数加载失败: {e}")

    # ── 创建策略 ──
    policy = BiMFTPolicy(config, dataset_stats=dataset_stats)

    # ── 加载模型权重 ──
    model_path = ckpt_path / "model.safetensors"
    if model_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(model_path))
        # 移除 LeRobot "model." 前缀
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                cleaned[k[len("model."):]] = v
            else:
                cleaned[k] = v
        policy.model.load_state_dict(cleaned, strict=False)
        print(f"  ✓ 模型权重加载成功 ({len(cleaned)} 个参数)")
    else:
        raise FileNotFoundError(f"找不到模型权重: {model_path}")

    policy.to(device)
    policy.eval()
    return policy


# ═══════════════════════════════════════════════════════
# 观测缓冲器
# ═══════════════════════════════════════════════════════

class ObsBuffer:
    """滑动窗口观测缓冲器。

    维护最近 T=n_obs_steps 帧的观测，构建 LeRobot 格式的 batch。
    环境返回格式（gym key）→ 内部缓冲 → LeRobot batch（observation. 前缀）。

    首 T 帧用零填充（策略可能需要预热或忽略前几步）。
    """

    def __init__(self, n_obs_steps: int = 4):
        self._T = n_obs_steps
        self._buffer: deque[dict[str, np.ndarray]] = deque(maxlen=n_obs_steps)

    def add(self, obs: dict[str, np.ndarray]) -> None:
        """添加一帧观测。"""
        self._buffer.append(obs)

    def is_ready(self) -> bool:
        """缓冲器是否已满。"""
        return len(self._buffer) >= self._T  # 至少需要 T 帧

    def build_batch(self, device: str = "cpu") -> dict[str, torch.Tensor]:
        """构建 LeRobot 格式的 batch。

        gym obs key → LeRobot batch key:
          state.joint.position      → observation.state.joint.position
          images.A_...rgb           → observation.images.A_...rgb
          等等
        """
        if len(self._buffer) == 0:
            raise RuntimeError("ObsBuffer 为空")

        # 收集所有帧
        frames = list(self._buffer)

        # 如果帧数不足 T，用零填充前面
        while len(frames) < self._T:
            empty = {k: np.zeros_like(v) for k, v in frames[0].items()}
            frames.insert(0, empty)

        batch: dict[str, torch.Tensor] = {}

        # ── 状态 key 映射 ──
        state_keys_gym = ["state.joint.position", "state.sensor.force", "state.sensor.torque"]
        state_keys_lr = [
            "observation.state.joint.position",
            "observation.state.sensor.force",
            "observation.state.sensor.torque",
        ]
        for gk, lk in zip(state_keys_gym, state_keys_lr):
            if gk in frames[0]:
                stacked = np.stack([f[gk] for f in frames], axis=0)  # [T, R, D]
                batch[lk] = torch.from_numpy(stacked).float().unsqueeze(0).to(device)  # [1, T, R, D]

        # ── 图像 key 映射 ──
        # 注意: images.* 下既有 .rgb 也有 .depth，二者处理不同：
        #   - .rgb: env 返回 uint8 [0,255]，训练时 dataset 返回 float32 [0,1]，需 /255 对齐
        #   - .depth: env 返回 float32 毫米（已 clip+*1000），训练时也是毫米，无需缩放
        image_keys = [k for k in frames[0].keys() if k.startswith("images.")]
        for gk in image_keys:
            lk = f"observation.{gk}"  # images.X → observation.images.X
            stacked = np.stack([f[gk] for f in frames], axis=0)  # [T, C, H, W]
            if gk.endswith(".rgb"):
                stacked = stacked.astype(np.float32) / 255.0          # [0,1]，与训练一致
            else:
                stacked = stacked.astype(np.float32)                  # depth 毫米，保持原值
            batch[lk] = torch.from_numpy(stacked).unsqueeze(0).to(device)  # [1, T, C, H, W]

        return batch

    def clear(self) -> None:
        """清空缓冲器。"""
        self._buffer.clear()


# ═══════════════════════════════════════════════════════
# 时间集成（temporal ensemble，类似 ALOHA ACT）
# ═══════════════════════════════════════════════════════

class TemporalEnsembler:
    """动作块时间集成（90Hz 时间粒度，类似 ALOHA ACT）。

    策略每次推理输出未来 chunk 步（90Hz）的动作块；执行时对每个 90Hz 时刻，
    把所有历史推理中"预测到该时刻"的动作做指数加权平均，最新推理的权重
    最大（越近越可信），从而平滑动作、抑制抖动。

    权重: w_i ∝ exp(-decay * age_i)，age_i = 当前时刻 - 推理时刻（90Hz 步数）。
    """

    def __init__(self, chunk_size: int, action_dim: int, decay: float = 0.01):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.decay = decay
        self._infer_times: list[int] = []    # 每次推理的绝对 90Hz 时刻
        self._chunks: list[np.ndarray] = []  # 对应的动作块 [chunk_size, action_dim]

    def reset(self) -> None:
        self._infer_times = []
        self._chunks = []

    def add_chunk(self, t: int, chunk: np.ndarray) -> None:
        """在绝对 90Hz 时刻 t 记录一次推理输出的动作块 [chunk_size, action_dim]。"""
        self._infer_times.append(t)
        self._chunks.append(np.asarray(chunk, dtype=np.float32))

    def _get_action(self, t: int) -> np.ndarray:
        """返回绝对 90Hz 时刻 t 应执行的动作（历史预测的指数加权平均）。

        若还没有任何推理覆盖到时刻 t，返回全零。
        """
        preds: list[np.ndarray] = []
        ages: list[float] = []
        for it, chunk in zip(self._infer_times, self._chunks):
            idx = t - it
            if 0 <= idx < chunk.shape[0]:
                preds.append(chunk[idx])
                ages.append(float(t - it))
        if not preds:
            return np.zeros(self.action_dim, dtype=np.float32)
        preds_arr = np.stack(preds)                       # [N, action_dim]
        ages_arr = np.asarray(ages, dtype=np.float32)     # 越小越新
        weights = np.exp(-self.decay * ages_arr)          # 最新(age=0)权重最大
        weights = weights / weights.sum()
        return np.sum(weights[:, None] * preds_arr, axis=0).astype(np.float32)

    def get_actions(self, t: int, n: int) -> np.ndarray:
        """返回从绝对 90Hz 时刻 t 开始的 n 个连续动作 [n, action_dim]。

        供每个 env.step（= n_action_steps 个 90Hz 步）一次性取 n 个动作。
        """
        return np.stack([self._get_action(t + k) for k in range(n)])


# ═══════════════════════════════════════════════════════
# 主评估循环
# ═══════════════════════════════════════════════════════

def run_episode(
    env, policy, obs_buffer: ObsBuffer,
    device: str, max_steps: int, ensemble_decay: float = 0.01,
) -> int:
    """运行一条 episode。

    Returns:
        实际执行步数。
    """
    obs, _info = env.reset()
    obs_buffer.clear()
    obs_buffer.add(obs)

    # 预热阶段动作 = 初始关节角（reset 后 = default_qpos + 夹爪归零），
    # 避免预热期用零动作导致机械臂跳到 0 位、与初始姿态不一致。
    initial_action = env._collect_joint_state()

    # 每次执行的高频（90Hz）动作步数 = 1 个图像帧周期（1/30s）
    n_action_steps = int(getattr(policy.config, "n_action_steps", 3))

    # ALOHA ACT 式时间集成（90Hz 时间粒度）：每次推理输出整个动作块，
    # 每个 90Hz 时刻对历史预测做指数加权平均（最新推理权重最大）。
    ensembler = TemporalEnsembler(
        chunk_size=policy.config.horizon,
        action_dim=initial_action.shape[0],
        decay=ensemble_decay,
    )

    step_count = 0
    try:
        for step_count in range(max_steps):
            # 当前绝对 90Hz 时刻（每个 env.step = n_action_steps 个 90Hz 步）
            t90 = step_count * n_action_steps

            # ── 构建 batch → 策略推理 ──
            if obs_buffer.is_ready():
                batch = obs_buffer.build_batch(device)
                chunk_tensor = policy.predict_action_chunk(batch)  # [1, horizon, 14]
                chunk = chunk_tensor.cpu().numpy()[0]              # [horizon, 14]
                ensembler.add_chunk(t90, chunk)
                action = ensembler.get_actions(t90, n_action_steps)  # [n, 14]
            else:
                # 预热阶段：保持初始关节角（复制 n 份，每个 90Hz 步一致）
                action = np.tile(initial_action, (n_action_steps, 1))

            # ── 环境步进 ──
            obs, reward, terminated, truncated, _info = env.step(action)
            obs_buffer.add(obs)

            # ── MuJoCo viewer 渲染 ──
            env.render()

            if terminated or truncated:
                break
    finally:
        pass  # env 由调用方管理生命周期

    return step_count + 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BiMFT 策略 MuJoCo 仿真评估"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="LeRobot checkpoint 目录路径")
    parser.add_argument("--task", default="dual_pick_place",
                        help="任务名 (对应 configs/tasks/tasks_*.yaml)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="评估 episode 数")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="每 episode 最大步数")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="推理设备")
    parser.add_argument("--no-render", action="store_true",
                        help="禁用 MuJoCo viewer（无头模式）")
    parser.add_argument("--ensemble-decay", type=float, default=0.02,
                        help="时间集成指数衰减系数（越小越平滑，0 关闭加权=最新预测）")
    args = parser.parse_args()

    # ── 1. 加载策略 ──────────────────────────────────
    print(f"\n{'='*60}")
    print("BiMFT 策略评估")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"任务:       {args.task}")
    print(f"Episodes:   {args.episodes}")
    print(f"设备:       {args.device}")

    policy = load_bimft_policy(args.checkpoint, args.device)
    n_obs_steps = policy.config.n_obs_steps

    print(f"时序窗口:   n_obs_steps={n_obs_steps}")
    print(f"动作块:     horizon={policy.config.horizon}")
    print(f"时间集成:   decay={args.ensemble_decay}")

    # ── 2. 创建环境 ─────────────────────────────────
    render_mode = None if args.no_render else "human"
    if not args.no_render:
        print(f"\n启动 MuJoCo viewer（关闭窗口可提前终止）...")

    from ur_mjlab_bc_rl.lerobot_env import UR5eDualEnv
    env = UR5eDualEnv(
        task_name=args.task,
        render_mode=render_mode,
        max_steps=args.max_steps,
    )

    obs_buffer = ObsBuffer(n_obs_steps=n_obs_steps)

    # ── 3. 运行评估 ─────────────────────────────────
    lengths: list[int] = []
    try:
        for ep in range(args.episodes):
            steps = run_episode(
                env, policy, obs_buffer, args.device, args.max_steps,
                ensemble_decay=args.ensemble_decay,
            )
            lengths.append(steps)
            print(f"  Episode {ep+1}/{args.episodes}: {steps} steps")
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        env.close()

    # ── 4. 统计 ─────────────────────────────────────
    if lengths:
        print(f"\n{'='*60}")
        print(f"结果: avg_length={np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
        print(f"      min={min(lengths)}, max={max(lengths)}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
