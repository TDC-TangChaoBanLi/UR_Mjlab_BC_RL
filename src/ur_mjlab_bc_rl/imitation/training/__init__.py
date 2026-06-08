"""训练模块。"""

from .losses import SquaredErrorLoss as SquaredErrorLoss
from .losses import L1Loss as L1Loss
from .losses import HuberLoss as HuberLoss
from .losses import build_loss as build_loss
from .trainer import ImitationTrainer as ImitationTrainer

__all__ = [
    "SquaredErrorLoss",
    "L1Loss",
    "HuberLoss",
    "build_loss",
    "ImitationTrainer",
]
