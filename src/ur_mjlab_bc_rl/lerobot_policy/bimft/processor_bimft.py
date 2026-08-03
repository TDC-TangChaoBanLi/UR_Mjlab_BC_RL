"""BiMFT LeRobot Policy 预处理/后处理 pipeline。

创建归一化和设备转移的 processor pipeline。
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
    RenameObservationsProcessorStep,
    ProcessorStep,
)
from .configuration_bimft import BiMFTConfig


def make_bimft_pre_post_processors(
    config: BiMFTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """创建 BiMFT 预处理和后处理 pipeline。

    Preprocessor:
      1. RenameObservationsProcessorStep — 特征 key 重命名（默认空 map）
      2. NormalizerProcessorStep         — 基于 dataset_stats 归一化
      3. DeviceProcessorStep             — 移到指定设备

    Postprocessor:
      1. UnnormalizerProcessorStep — action 反归一化

    Args:
        config: BiMFT 策略配置
        dataset_stats: 数据集统计量

    Returns:
        (preprocessor, postprocessor) pipeline 对
    """
    pre_steps: list[ProcessorStep] = []

    # 特征重命名（lerobot resume 兼容，默认 rename_map 为空即不重命名）
    pre_steps.append(
        RenameObservationsProcessorStep(
            rename_map=config.normalization_mapping.get("rename_map", {}),
        )
    )

    # 归一化（始终包含，resume 覆盖时需要此步骤存在）
    pre_steps.append(
        NormalizerProcessorStep(
            stats=dataset_stats if dataset_stats else {},
            features={
                **config.input_features,
                **config.output_features,
            },
            norm_map=config.normalization_mapping,
        )
    )

    # 设备转移
    pre_steps.append(DeviceProcessorStep(device=config.device))

    preprocessor = PolicyProcessorPipeline(steps=pre_steps)

    # 后处理：反归一化（始终包含）
    post_steps: list[ProcessorStep] = []
    post_steps.append(
        UnnormalizerProcessorStep(
            stats=dataset_stats if dataset_stats else {},
            features=config.output_features,
            norm_map=config.normalization_mapping,
        )
    )

    postprocessor = PolicyProcessorPipeline(steps=post_steps)

    return preprocessor, postprocessor
