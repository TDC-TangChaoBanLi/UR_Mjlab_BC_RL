
# UR_Mjlab_BC_RL

UR5 机械臂多任务操作学习框架 —— 纯 MuJoCo 专家数据生成 + PyTorch 模仿学习预训练 + MjLab PPO 强化学习微调。

## 支持的三个桌面操作任务 (目前仅支持 Pick-and-Place 任务的训练与评估)

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
│   ├── train_aloha_act.py           # ALOHA ACT 训练
│   ├── train_ppo_finetune.py        # PPO 微调
│   ├── merge_datasets.py            # 数据集合并
│   ├── truncate_state_dataset.py    # 数据集截断
│   └── eval_policy.py               # 策略评估
├── src/ur_mjlab_bc_rl/
│   ├── models/             # 共享模型（Actor, Critic, 编码器, 融合模块）
│   ├── imitation/          # LeRobot 策略集成 + 数据采集 + 环境接口
│   │   ├── config_base.py          # 共享 PreTrainedConfig 基类
│   │   ├── policy_base.py          # 共享 PreTrainedPolicy 基类
│   │   ├── dataset/                # Episode 容器 + LeRobotDataset 写入
│   │   ├── mujoco_env/             # MuJoCo 仿真接口
│   │   ├── expert_generation/      # Scripted teacher
│   │   └── lerobot_policy/         # 各场景 LeRobot 策略包
│   │       └── ur5_multimodal/     # 单臂 RGBD BC 策略
│   ├── reinforcement/      # BC→PPO 桥接 + checkpoint 工具
│   ├── cfg/                # MjLab 环境配置（观测/奖励/终止/事件）
│   └── env_cfg.py          # 3 个任务的 MjLab 环境注册
└── tests/                  # 47 个 pytest 单元测试
```

## 环境准备

1. 安装 uv :
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2. 安装依赖
    ```bash
    uv sync
    ```

> info: 当前的依赖源被锁定为 `https://pypi.tuna.tsinghua.edu.cn/simple`。

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

使用 LeRobot 训练框架（支持单卡/多卡自动 DDP/FSDP）：

```bash
# 安装策略插件
pip install -e src/ur_mjlab_bc_rl/imitation/lerobot_policy/ur5_multimodal/

# 单卡训练
lerobot-train \
    --policy.type=ur5_multimodal \
    --dataset.repo_id=ur5_pick_place \
    --dataset.root=outputs/datasets/expert/pick_place/XXXXXX/ \
    --steps=5000 --batch_size=64

# 多卡训练（自动 DDP/FSDP）
accelerate launch --num_processes=4 lerobot-train \
    --policy.type=ur5_multimodal \
    --dataset.repo_id=ur5_pick_place \
    --dataset.root=outputs/datasets/expert/pick_place/XXXXXX/ \
    --steps=10000 --batch_size=32
```

2. Action Chunk with Transformer 模仿学习（仍在独立脚本中）：
    ```bash
    python scripts/train_aloha_act.py \
        --data outputs/datasets/expert/pick_place/XXXXXX/ \
        --epochs 100 --batch 32
    ```


### 3. PPO 强化学习微调 (尚未测试)

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
    --checkpoint outputs/checkpoints/pick_place/XXXXXX/best_actor.pt --episodes 20

# PPO 策略交互（通过 MjLab） (尚未测试)
mjlab play UR5-PickPlace \
    --checkpoint-file logs/rsl_rl/pick_place/model_0.pt \
    --viewer viser --num-envs 1
```

## 模型架构

`UR5MultimodalActor` 采用完全模块化设计：

```text
RGBD 图像 [B,4,H,W] ──→ VisualEncoder (CNN/ViT) ──┐
机器人状态 [B,7]    ──→ StateEncoder (MLP)      ──┼──→ Fusion ──→ PolicyMLP ──→ Action [B,7]
任务 ID [B]          ──→ TaskEncoder (Embedding)  ──┘       ↑
                                                        FiLM / Concat
```

- 视觉编码器：CNN 或 ViT（config 可切换）
- 融合方式：FiLM（特征调制）或 Concat（拼接）

## 运行测试

```bash
PYTHONPATH=src pytest tests/ -v -p no:launch_testing
```

47 个单元测试覆盖 Models / Imitation / Reinforcement 三大模块。

## License

MIT


