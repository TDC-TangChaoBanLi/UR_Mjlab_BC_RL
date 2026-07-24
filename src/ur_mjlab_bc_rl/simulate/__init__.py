"""仿真模块 — MuJoCo 环境 + Scripted Teacher + 数据采集。"""

from .env import MujocoInterface, CameraSensor, ObservationCollector, MinkIK, ResetManager
from .teachers import PickPlaceTeacher, PushTTeacher, PegSlotTeacher
from .dataset_writer import Episode, LeRobotDatasetWriter, LeRobotDatasetConfig
from .config_loader import SceneConfig, load_scene_config, get_task_list
from .controllers import (
    Controller, ScriptedTeacherController, PolicyController,
)
from .simulation_manager import SimulationManager
