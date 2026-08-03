# BiMFT — Bimanual Multimodal Fusion Transformer Policy

**双臂多模态融合 Transformer 策略** — LeRobot 训练框架集成。

---

## 快速开始

### 安装

```bash
uv pip install -e src/ur_mjlab_bc_rl/imitation/lerobot_policy/bimft/
```

### 训练

```bash
uv run lerobot-train \
    --policy.type=bimft \
    --policy.push_to_hub=false \
    --policy.use_amp=true \
    --dataset.repo_id=ur5_dual_pick_place \
    --dataset.root=outputs/datasets/expert/dual_pick_place/<timestamp> \
    --dataset.video_backend=pyav \
    --steps=1000 \
    --batch_size=8 \
    --tolerance=0.4
```

### 从 Checkpoint 继续训练

```bash
uv run lerobot-train \
    --policy.type=bimft \
    --policy.push_to_hub=false \
    --policy.use_amp=true \
    --dataset.repo_id=ur5_dual_pick_place \
    --dataset.root=outputs/datasets/expert/dual_pick_place/<timestamp> \
    --dataset.video_backend=pyav \
    --steps=200 \
    --batch_size=8 \
    --tolerance=0.4 \
    --resume=true
```

`--resume=true` 会自动加载 `outputs/train/<latest_timestamp>_bimft/checkpoints/last/pretrained_model` 中的优化器状态和学习率调度器，从上次中断的步数继续训练。如需指定特定 checkpoint：

```bash
uv run lerobot-train \
    --policy.type=bimft \
    ... \
    --resume=true \
    --checkpoint.output_dir=outputs/train/20260727_18-05-03_bimft/checkpoints/step_00000100
```

### 评估

```bash
uv run python scripts/eval_policy.py \
    --policy.type=bimft \
    --checkpoint outputs/train/<timestamp>_bimft/checkpoints/last/pretrained_model
```

---

## 参数说明

### 训练必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--policy.type=bimft` | 策略类型 | 固定值 |
| `--policy.push_to_hub=false` | 禁止自动推送 HuggingFace Hub | 本地训练用 false |
| `--dataset.repo_id` | 数据集 ID | `ur5_dual_pick_place` |
| `--dataset.root` | 数据集路径 | `outputs/datasets/expert/...` |
| `--dataset.video_backend=pyav` | 视频解码后端 | 环境 FFmpeg 兼容性 workaround |
| `--steps` | 训练步数 | `100` |
| `--batch_size` | batch 大小 | 16G 显存建议 `8`，24G 以上 `16` |

### 显存优化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--policy.use_amp` | `true` | BF16 混合精度，显存 ~40%↓ |
| `--batch_size` | `8` | 降低可线性减少显存 |

### Checkpoint 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--save_checkpoint` | `true` | 是否保存 checkpoint |
| `--save_freq` | `20000` | checkpoint 保存间隔（步数），设小值可更频繁保存 |
| `--resume` | `false` | 从上次 checkpoint 继续训练 |

### 日志与监控

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--log_freq` | `200` | 终端输出 loss 的间隔（步数），设 `1` 可每步输出 |
| `--wandb.enable` | `false` | 启用 WandB 实时可视化 loss 曲线 |
| `--wandb.project` | — | WandB 项目名，如 `ur5_dual_pick_place` |

终端输出示例（每 `--log_freq` 步）：

```
INFO step 10 action_loss=0.xxx smooth_loss=0.xxx left_loss=0.xxx right_loss=0.xxx
```

使用 WandB：

```bash
uv run lerobot-train \
    --policy.type=bimft \
    ... \
    --wandb.enable=true \
    --wandb.project=ur5_dual_pick_place
```

不使用 WandB 时，可将日志保存到文件：

```bash
uv run lerobot-train ... 2>&1 | tee training.log
```

### 模型架构参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--policy.n_obs_steps` | `4` | 输入视觉帧数（4 帧 ≈ 0.13s 历史） |
| `--policy.horizon` | `30` | 输出动作块长度（30 步 @ 90Hz） |
| `--policy.n_action_steps` | `3` | 每次执行的动作步数 |
| `--policy.model_cfg_path` | `configs/model/bimft.yaml` | 模型详细配置 YAML |

### 学习率参数

| 参数 | 默认值 |
|------|--------|
| `--policy.optimizer_lr` | `1e-4` |
| `--policy.optimizer_weight_decay` | `1e-4` |
| `--policy.scheduler_warmup_steps` | `500` |

---

## 模型配置

详细网络参数见 `configs/model/bimft.yaml`，关键可调参数：

| YAML key | 默认值 | 说明 |
|----------|--------|------|
| `token.d_model` | `512` | 统一 token 维度，增大 → 更多参数 |
| `token.n_heads` | `8` | 注意力头数 |
| `token.dim_feedforward` | `2048` | FFN 维度 |
| `input.vision_history` | `4` | 视觉历史帧数 |
| `input.joint_high_rate` | `3` | 每帧关节采样数 |
| `input.force_high_rate` | `3` | 每帧力觉采样数 |
| `action.chunk_size` | `30` | 动作块长度 |
| `bimanual.coordination_layers` | `4` | 双臂协调 Transformer 层数 |
| `action.decoder_layers` | `4` | 动作解码器层数 |

---

## 数据集格式要求

数据须包含以下字段（LeRobot 标准格式）：

| Key | Shape | 说明 |
|-----|-------|------|
| `observation.images.A_realsense_link_CAMERA.rgb` | `[T, 3, H, W]` | 左腕 RGB |
| `observation.images.A_realsense_link_CAMERA.depth` | `[T, 1, H, W]` | 左腕 Depth |
| `observation.images.B_realsense_link_CAMERA.rgb` | `[T, 3, H, W]` | 右腕 RGB |
| `observation.images.B_realsense_link_CAMERA.depth` | `[T, 1, H, W]` | 右腕 Depth |
| `observation.images.global_realsense_link_CAMERA.rgb` | `[T, 3, H, W]` | 全局 RGB |
| `observation.images.global_realsense_link_CAMERA.depth` | `[T, 1, H, W]` | 全局 Depth |
| `observation.state.joint.position` | `[T, 14]` | 左右关节 (左7+右7) |
| `observation.state.sensor.force` | `[T, 6]` | 左右力 (左3+右3) |
| `observation.state.sensor.torque` | `[T, 6]` | 左右力矩 (左3+右3) |
| `action` | `[T, 14]` | 动作 (左7+右7)，**必须用 key `action`** |


---

## 目录结构

```
models/BiMFT/
├── BiMFT_Policy.py    # 主网络 + PolicyConfig
├── attention.py       # Self/Cross-Attention, AttentionPool, FeedForward
├── embeddings.py      # 2D 位置编码 + 连续时间编码
├── rgbd_resnet.py     # 四通道 RGB-D ResNet-18
├── state_encoders.py  # 高频关节/力觉 Token 编码
├── slot_fusion.py     # 单时刻多模态融合 + 门控
├── temporal_fusion.py # 当前帧锚定时序融合
├── bimanual_fusion.py # 双臂全局协调融合
├── encoders.py        # 单臂编码器 + 全局视觉编码器
├── action_decoder.py  # 双流同步动作解码器
└── losses.py          # Huber + 平滑损失

imitation/lerobot_policy/bimft/
├── pyproject.toml
├── README.md
└── src/lerobot_policy_bimft/
    ├── __init__.py              # 插件注册入口
    ├── configuration_bimft.py   # BiMFTConfig
    ├── modeling_bimft.py        # BiMFTPolicy (LeRobot 接口)
    └── processor_bimft.py       # 预处理/后处理 pipeline
```

---

## 技术架构

参考 `models/BiMFT/结构.md` 获取完整数学符号化说明。

- **参数量**: 基线配置 ~197.5M
- **输入**: 3 路 RGB-D × 4 帧 + 关节/力觉 × 12 采样
- **输出**: 左右各 30 步连续动作块
- **核心模块**: RGB-D ResNet → 多模态融合 → 时序融合 → 双臂协调 → 动作解码
