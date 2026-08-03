"""lerobot_env_ur5e_dual — LeRobot 第三方环境插件。

被 LeRobot 的 register_third_party_plugins() 自动发现并导入，
触发 @EnvConfig.register_subclass("ur5e_dual") 注册。

通过独立 pyproject.toml (名称匹配 lerobot_env_ 前缀) 安装到 venv 中。
"""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this environment."
    )

# 从主包导入并触发注册
from ur_mjlab_bc_rl.lerobot_env.ur5e_dual_env import UR5eDualEnv  # noqa: F401
from ur_mjlab_bc_rl.lerobot_env.lerobot_env_cfg import UR5eDualEnvConfig  # noqa: F401

__all__ = ["UR5eDualEnv", "UR5eDualEnvConfig"]
