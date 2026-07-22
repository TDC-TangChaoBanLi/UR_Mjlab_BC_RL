"""仿真模块 — MuJoCo 环境 + Scripted Teacher + 数据采集。

本模块提供：
- env: MuJoCo 仿真接口（MujocoInterface, CameraSensor, ObservationCollector, ResetManager, MinkIK）
- teachers: Scripted 专家教师（PickPlaceTeacher, PushTTeacher, PegSlotTeacher）
- dataset_writer: LeRobot 数据集写入（Episode, LeRobotDatasetWriter, CollectionConfig）
- config_loader: 仿真参数加载
"""

from __future__ import annotations

from .env import (
    MujocoInterface,
    CameraSensor,
    ObservationCollector,
    convert_obs_to_model_input,
    flatten_state,
    DEFAULT_STATE_KEYS,
    MinkIK,
    ResetManager,
)
from .teachers import (
    Teacher,
    TeacherState,
    PickPlaceTeacher,
    PushTTeacher,
    PegSlotTeacher,
)
from .dataset_writer import (
    Episode,
    LeRobotDatasetWriter,
    LeRobotDatasetConfig,
    CollectionConfig,
)
from . import config_loader

__all__ = [
    # env
    "MujocoInterface",
    "CameraSensor",
    "ObservationCollector",
    "convert_obs_to_model_input",
    "flatten_state",
    "DEFAULT_STATE_KEYS",
    "MinkIK",
    "ResetManager",
    # teachers
    "Teacher",
    "TeacherState",
    "PickPlaceTeacher",
    "PushTTeacher",
    "PegSlotTeacher",
    # dataset_writer
    "Episode",
    "LeRobotDatasetWriter",
    "LeRobotDatasetConfig",
    "CollectionConfig",
    # utils
    "config_loader",
]
