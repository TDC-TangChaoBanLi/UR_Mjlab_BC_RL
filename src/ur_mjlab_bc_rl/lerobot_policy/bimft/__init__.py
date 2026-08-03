"""BiMFT Bimanual Multimodal Fusion Transformer Policy — LeRobot 插件入口。

被 LeRobot 的 register_third_party_plugins() 自动发现并导入，
触发 @PreTrainedConfig.register_subclass("bimft") 注册。
"""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    )

from .configuration_bimft import BiMFTConfig
from .modeling_bimft import BiMFTPolicy
from .processor_bimft import make_bimft_pre_post_processors

__all__ = [
    "BiMFTConfig",
    "BiMFTPolicy",
    "make_bimft_pre_post_processors",
]
