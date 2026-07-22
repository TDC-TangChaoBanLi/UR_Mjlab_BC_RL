"""UR5 Multimodal BC Policy — LeRobot 插件入口。

被 LeRobot 的 register_third_party_plugins() 自动发现并导入，
触发 @PreTrainedConfig.register_subclass 注册。
"""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    )

from .configuration_ur5_multimodal import UR5MultimodalConfig
from .modeling_ur5_multimodal import UR5MultimodalPolicy
from .processor_ur5_multimodal import make_ur5_multimodal_pre_post_processors

__all__ = [
    "UR5MultimodalConfig",
    "UR5MultimodalPolicy",
    "make_ur5_multimodal_pre_post_processors",
]
