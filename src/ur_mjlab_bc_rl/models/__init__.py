"""模型模块 — 从三个独立模型目录统一导出."""

# ══ 通用规格（models/ 根目录保留） ══
from .specs import EncoderOutput as EncoderOutput
from .specs import ObsSpec as ObsSpec
from .specs import ActionSpec as ActionSpec
from .specs import TaskSpec as TaskSpec
from .specs import CriticObsSpec as CriticObsSpec

# ══ 通用分布 ══
from .distributions import GaussianDistribution as GaussianDistribution

# ══ ALHAH_ACT 模型 ══
from .ALHAH_ACT.backbone import DETRVAE as DETRVAE
from .ALHAH_ACT.backbone import EnsembleBuffer as EnsembleBuffer
from .ALHAH_ACT.backbone import build_detr_vae as build_detr_vae
from .ALHAH_ACT.backbone import get_sinusoid_encoding_table as get_sinusoid_encoding_table
from .ALHAH_ACT.backbone import get_2d_sincos_pos_embed as get_2d_sincos_pos_embed

# ══ Test_Multimodal 模型 — 视觉编码器 ══
from .Test_Multimodal.vision_base import VisualEncoderBase as VisualEncoderBase
from .Test_Multimodal.rescnn import ResCNN as ResCNN
from .Test_Multimodal.vit import ViT as ViT
from .Test_Multimodal.vision_factory import build_visual_encoder as build_visual_encoder

# ══ Test_Multimodal 模型 — 状态编码器 ══
from .Test_Multimodal.state_base import StateEncoderBase as StateEncoderBase
from .Test_Multimodal.mlp_state import MLPStateEncoder as MLPStateEncoder
from .Test_Multimodal.state_factory import build_state_encoder as build_state_encoder

# ══ Test_Multimodal 模型 — 任务编码器 ══
from .Test_Multimodal.task_base import TaskEncoderBase as TaskEncoderBase
from .Test_Multimodal.embedding_task import EmbeddingTaskEncoder as EmbeddingTaskEncoder
from .Test_Multimodal.task_factory import build_task_encoder as build_task_encoder

# ══ Test_Multimodal 模型 — 融合模块 ══
from .Test_Multimodal.fusion_base import FusionBase as FusionBase
from .Test_Multimodal.concat import ConcatFusion as ConcatFusion
from .Test_Multimodal.film import FiLMFusion as FiLMFusion
from .Test_Multimodal.fusion_factory import build_fusion as build_fusion

# ══ Test_Multimodal 模型 — MLP ══
from .Test_Multimodal.mlp import MLP as MLP

# ══ Test_Multimodal 模型 — 策略 ══
from .Test_Multimodal.backbone import UR5MultimodalBackbone as UR5MultimodalBackbone
from .Test_Multimodal.backbone import build_actor as build_actor
from .Test_Multimodal.rsl_adapter import UR5RslActorModel as UR5RslActorModel
from .Test_Multimodal.rsl_adapter import UR5MultimodalModelCfg as UR5MultimodalModelCfg

# ══ BiMFT 模型 ══
from .BiMFT.BiMFT_Policy import BiMFT_Policy as BiMFT_Policy
from .BiMFT.BiMFT_Policy import PolicyConfig as BiMFTPolicyConfig

__all__ = [
    # 规格
    "EncoderOutput",
    "ObsSpec",
    "ActionSpec",
    "TaskSpec",
    "CriticObsSpec",
    # 分布
    "GaussianDistribution",
    # ALHAH_ACT
    "DETRVAE",
    "EnsembleBuffer",
    "build_detr_vae",
    "get_sinusoid_encoding_table",
    "get_2d_sincos_pos_embed",
    # 视觉编码器
    "ResCNN",
    "ViT",
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
    # MLP
    "MLP",
    # 策略
    "UR5MultimodalBackbone",
    "build_actor",
    "UR5RslActorModel",
    "UR5MultimodalModelCfg",
    # BiMFT
    "BiMFT_Policy",
    "BiMFTPolicyConfig",
]
