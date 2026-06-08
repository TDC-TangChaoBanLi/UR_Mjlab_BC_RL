"""模型模块."""

# 规格定义
from .specs import EncoderOutput as EncoderOutput
from .specs import ObsSpec as ObsSpec
from .specs import ActionSpec as ActionSpec
from .specs import TaskSpec as TaskSpec
from .specs import CriticObsSpec as CriticObsSpec

# 视觉编码器
from .vision import RGBDViT as RGBDViT
from .vision import RGBDCNNEncoder as RGBDCNNEncoder
from .vision import VisualEncoderBase as VisualEncoderBase
from .vision import build_visual_encoder as build_visual_encoder

# 状态编码器
from .state.base import StateEncoderBase as StateEncoderBase
from .state.mlp_state import MLPStateEncoder as MLPStateEncoder
from .state.encoder_factory import build_state_encoder as build_state_encoder

# 任务编码器
from .task.base import TaskEncoderBase as TaskEncoderBase
from .task.embedding_task import EmbeddingTaskEncoder as EmbeddingTaskEncoder
from .task.encoder_factory import build_task_encoder as build_task_encoder

# 融合模块
from .fusion.base import FusionBase as FusionBase
from .fusion.concat import ConcatFusion as ConcatFusion
from .fusion.film import FiLMFusion as FiLMFusion
from .fusion.factory import build_fusion as build_fusion

# 分布
from .distributions import GaussianDistribution as GaussianDistribution

# 策略
from .policy.multimodal_backbone import UR5MultimodalBackbone as UR5MultimodalBackbone
from .policy.rsl_adapter import UR5RslActorModel as UR5RslActorModel
from .policy.rsl_adapter import UR5MultimodalModelCfg as UR5MultimodalModelCfg

# 模块
from .modules import MLP as MLP

__all__ = [
    # 规格
    "EncoderOutput",
    "ObsSpec",
    "ActionSpec",
    "TaskSpec",
    "CriticObsSpec",
    # 视觉编码器
    "RGBDViT",
    "RGBDCNNEncoder",
    "VisualEncoderBase",
    "build_visual_encoder",
    # 状态编码器
    "StateEncoderBase",
    "MLPStateEncoder",
    "build_state_encoder",
    # 任务编码器
    "TaskEncoderBase",
    "EmbeddingTaskEncoder",
    "build_task_encoder",
    # 融合模块
    "FusionBase",
    "ConcatFusion",
    "FiLMFusion",
    "build_fusion",
    # 分布
    "GaussianDistribution",
    # 策略
    "UR5MultimodalBackbone",
    "UR5RslActorModel",
    "UR5MultimodalModelCfg",
    # 模块
    "MLP",
]
