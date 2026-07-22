"""UR5 Multimodal BC Policy 预处理/后处理 pipeline。

LeRobot 训练循环在送入模型前会通过 preprocessor 处理 batch，
模型输出后通过 postprocessor 处理 action。
"""

from __future__ import annotations

from typing import Any

import torch

from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    NormalizerProcessorStep,
    UnnormalizerProcessorStep,
    DeviceProcessorStep,
    ProcessorStep,
)
from .configuration_ur5_multimodal import UR5MultimodalConfig


def make_ur5_multimodal_pre_post_processors(
    config: UR5MultimodalConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """创建预处理和后处理 pipeline。

    Preprocessor:
    1. NormalizerProcessorStep — 基于 dataset_stats 归一化 state / action
    2. DeviceProcessorStep       — 移到指定设备

    Postprocessor:
    1. UnnormalizerProcessorStep — action 反归一化

    Args:
        config: 策略配置
        dataset_stats: 数据集统计量，格式为 {feature_key: {mean, std, min, max}}

    Returns:
        (preprocessor, postprocessor)
    """
    pre_steps: list[ProcessorStep] = []

    # 归一化 step
    if dataset_stats is not None:
        pre_steps.append(
            NormalizerProcessorStep(
                stats=dataset_stats,
                features={
                    **config.input_features,
                    **config.output_features,
                },
                norm_map=config.normalization_mapping,
            )
        )

    # 设备 step
    pre_steps.append(DeviceProcessorStep(device=config.device))

    preprocessor = PolicyProcessorPipeline(steps=pre_steps)

    # 后处理：反归一化
    post_steps: list[ProcessorStep] = []
    if dataset_stats is not None:
        post_steps.append(
            UnnormalizerProcessorStep(
                stats=dataset_stats,
                features=config.output_features,
                norm_map=config.normalization_mapping,
            )
        )

    postprocessor = PolicyProcessorPipeline(steps=post_steps)

    return preprocessor, postprocessor
