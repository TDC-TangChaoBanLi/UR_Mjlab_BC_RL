"""UR5 Multi-task BC + PPO RL — 基于 mjlab 框架.

注册任务:
- UR5-PickPlace (task 0)
- UR5-PushT (task 1)
- UR5-PegInSlot (task 2)

LeRobot 插件注册（通过 entry points 自动发现）:
- EnvConfig("ur5e_dual")
- PreTrainedConfig("bimft")
"""

# ── LeRobot 插件注册 ─────────────────────────────────
# 导入 env 模块触发 @EnvConfig.register_subclass("ur5e_dual")
from . import lerobot_env   # noqa: F401 — 注册 EnvConfig
# 导入 policy 模块触发 @PreTrainedConfig.register_subclass("bimft")
from . import lerobot_policy  # noqa: F401 — 注册 PreTrainedConfig

try:
    from mjlab.tasks.registry import register_mjlab_task
    _HAS_MJLAB = True
except ImportError:
    register_mjlab_task = None  # type: ignore[assignment]
    _HAS_MJLAB = False

from .reinforcement.env_cfg import (
    pick_place_env_cfg, pick_place_ppo_runner_cfg,
    push_t_env_cfg, push_t_ppo_runner_cfg,
    peg_in_slot_env_cfg, peg_in_slot_ppo_runner_cfg,
)

if _HAS_MJLAB:
    # Task 0: Pick-and-Place
    register_mjlab_task(
        task_id="UR5-PickPlace",
        env_cfg=pick_place_env_cfg(),
        play_env_cfg=pick_place_env_cfg(play=True),
        rl_cfg=pick_place_ppo_runner_cfg(),
    )

    # Task 1: Push-T
    register_mjlab_task(
        task_id="UR5-PushT",
        env_cfg=push_t_env_cfg(),
        play_env_cfg=push_t_env_cfg(play=True),
        rl_cfg=push_t_ppo_runner_cfg(),
    )

    # Task 2: Peg-in-Slot
    register_mjlab_task(
        task_id="UR5-PegInSlot",
        env_cfg=peg_in_slot_env_cfg(),
        play_env_cfg=peg_in_slot_env_cfg(play=True),
        rl_cfg=peg_in_slot_ppo_runner_cfg(),
    )


def main() -> None:
    print("Hello from ur-mjlab-bc-rl!")
