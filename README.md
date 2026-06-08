
# UR_Mjlab_BC_RL

UR5 机械臂多任务操作学习框架 —— 纯 MuJoCo 专家数据生成 + PyTorch 模仿学习预训练 + MjLab PPO 强化学习微调。

## 支持的三个桌面操作任务

| 任务 ID | 任务名 | 描述 |
|---------|--------|------|
| 0 | Pick-and-Place | 抓取方块并放入盘子 |
| 1 | Push-T | 推动 T 形物体到目标位姿 |
| 2 | Peg-in-Slot | 抓取凸轴插入凹槽 |

## 训练流程

```text
┌────────────────────┐
│  Scripted Teacher  │  纯 MuJoCo 环境，3 个任务各自的自动化专家策略
│  自动采集专家数据  │
└────────┬───────────┘
         ↓
┌────────────────────┐
│  BC 模仿学习预训练 │  纯 PyTorch，多模态 Actor（RGBD + 状态 + 任务）
│  训练 UR5 Actor    │
└────────┬───────────┘
         ↓
┌────────────────────┐
│  PPO 强化学习微调  │  基于 MjLab/RSL-RL，加载 BC checkpoint
│  精细调优策略      │
└────────┬───────────┘
         ↓
┌────────────────────┐
│  部署 & 评估       │  MjLab play / eval 命令行
└────────────────────┘
```

## 项目结构

```
UR_Mjlab_BC_RL/
├── assets/mujoco/         # MuJoCo 模型文件（UR5, 物体, 场景）
├── configs/
│   ├── env/               # MjLab 环境 YAML 配置
│   ├── model/             # Actor 模型 YAML 配置
│   └── train/             # BC 和 PPO 训练 YAML 配置
├── scripts/
│   ├── collect_scripted_expert.py   # 自动采集专家数据
│   ├── collect_keyboard_expert.py   # 键盘交互采集
│   ├── train_imitation.py          # BC 预训练
│   ├── train_ppo_finetune.py       # PPO 微调
│   ├── replay_imitation_dataset.py # 数据回放
│   └── eval_policy.py              # 策略评估
├── src/ur_mjlab_bc_rl/
│   ├── models/             # 共享模型（Actor, Critic, 编码器, 融合模块）
│   ├── imitation/          # 专家数据生成 + BC 训练（不依赖 MjLab）
│   ├── reinforcement/      # BC→PPO 桥接 + checkpoint 工具
│   ├── cfg/                # MjLab 环境配置（观测/奖励/终止/事件）
│   └── env_cfg.py          # 3 个任务的 MjLab 环境注册
└── tests/                  # 47 个 pytest 单元测试
```

## 环境准备

```bash
# 安装依赖
uv sync
```

核心依赖：`torch`, `numpy`, `scipy`, `mjlab`（含 MuJoCo, RSL-RL）。

## 快速开始

### 1. 采集专家数据

```bash
# 自动采集（推荐）
python scripts/collect_scripted_expert.py --task pick_place --episodes 50

# 键盘交互采集
python scripts/collect_keyboard_expert.py --task pick_place
```

### 2. BC 模仿学习预训练

```bash
python scripts/train_imitation.py \
    --data outputs/datasets/expert/pick_place \
    --task pick_place \
    --epochs 100 --batch 64 --lr 1e-3
```

### 3. PPO 强化学习微调

```bash
# 从 BC checkpoint 初始化
python scripts/train_ppo_finetune.py \
    --task UR5-PickPlace \
    --bc-checkpoint outputs/checkpoints/pick_place/best_actor.pt \
    --num-envs 16 --headless
```

### 4. 评估与部署

```bash
# BC 策略评估
python scripts/eval_policy.py --task pick_place \
    --checkpoint outputs/checkpoints/pick_place/best_actor.pt --episodes 20

# PPO 策略交互（通过 MjLab）
mjlab play UR5-PickPlace \
    --checkpoint-file logs/rsl_rl/pick_place/model_0.pt \
    --viewer viser --num-envs 1
```

## 模型架构

`UR5MultimodalActor` 采用完全模块化设计：

```text
RGBD 图像 [B,4,H,W] ──→ VisualEncoder (CNN/ViT) ──┐
机器人状态 [B,27]    ──→ StateEncoder (MLP)      ──┼──→ Fusion ──→ PolicyMLP ──→ Action [B,7]
任务 ID [B]          ──→ TaskEncoder (Embedding)  ──┘       ↑
                                                        FiLM / Concat
```

- 视觉编码器：CNN 或 ViT（config 可切换）
- 融合方式：FiLM（特征调制）或 Concat（拼接）
- 17.5M 参数（CNN 配置）

## 运行测试

```bash
PYTHONPATH=src pytest tests/ -v -p no:launch_testing
```

47 个单元测试覆盖 Models / Imitation / Reinforcement 三大模块。

## License

MIT


