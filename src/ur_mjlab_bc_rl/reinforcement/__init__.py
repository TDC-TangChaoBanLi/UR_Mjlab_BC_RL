"""强化学习模块 —— MjLab PPO 微调。

职责：
- BC checkpoint → PPO 桥接
- PPO 微调配置
- 策略评估与部署

不依赖 imitation 模块（仅读取 checkpoint）。
"""

from .checkpoint_utils import load_bc_checkpoint as load_bc_checkpoint
from .checkpoint_utils import inspect_bc_checkpoint as inspect_bc_checkpoint

__all__ = [
    "load_bc_checkpoint",
    "inspect_bc_checkpoint",
]
