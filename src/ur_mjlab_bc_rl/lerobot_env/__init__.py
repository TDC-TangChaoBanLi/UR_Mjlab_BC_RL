"""环境模块 — Gymnasium 环境 + LeRobot EnvConfig 注册。

作为 LeRobot 第三方插件被发现:
  - 通过 lerobot_env/pyproject.toml 安装为独立包 lerobot_env_ur5e_dual
  - LeRobot 的 register_third_party_plugins() 按包名前缀 lerobot_env_ 发现并导入
  - 导入时触发 @EnvConfig.register_subclass("ur5e_dual") 注册

时序结构（匹配 BiMFT 策略网络）:
  - 图像: 3 路相机 RGB-D @ 30Hz, channel-first (C, H, W)
  - 状态: 每帧图像对应 R=3 个高频采样 @ 90Hz, shape (R, D)
  - LeRobot delta 堆叠 T=4 帧后:
    图像 → [B, T, C, H, W] = [B, 4, 3/1, 480, 640]
    状态 → [B, T, R, D] = [B, 4, 3, 14]
"""

from ur_mjlab_bc_rl.simulate.mujoco_wrapper import MujocoWrapper  # noqa: F401
from ur_mjlab_bc_rl.simulate.cameras import CameraSensor        # noqa: F401
from .ur5e_dual_env import UR5eDualEnv   # noqa: F401
from .lerobot_env_cfg import UR5eDualEnvConfig  # noqa: F401 — 触发注册
