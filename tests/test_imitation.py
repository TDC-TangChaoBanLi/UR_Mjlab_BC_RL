"""Imitation / Simulate 模块单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class TestEpisode:
    """测试 Episode 数据类（simulate 模块）。"""

    def test_basic(self):
        from ur_mjlab_bc_rl.simulate.dataset_writer import Episode
        ep = Episode()
        assert len(ep) == 0
        ep.add({
            "state": {},
            "rgb": np.zeros((64, 64, 3), dtype=np.uint8),
            "depth": np.zeros((64, 64), dtype=np.float32),
            "task_id": 0,
        }, np.ones(7))
        assert len(ep) == 1
        assert ep.actions[0].shape == (7,)

    def test_multiple_frames(self):
        from ur_mjlab_bc_rl.simulate.dataset_writer import Episode
        ep = Episode()
        ep.add({
            "state": {}, "rgb": np.zeros((64, 64, 3), dtype=np.uint8),
            "depth": np.zeros((64, 64), dtype=np.float32), "task_id": 0,
        }, np.ones(7))
        ep.add({
            "state": {}, "rgb": np.ones((64, 64, 3), dtype=np.uint8),
            "depth": np.ones((64, 64), dtype=np.float32), "task_id": 1,
        }, np.zeros(7))
        assert len(ep) == 2
        assert np.allclose(ep.actions[0], np.ones(7))

    def test_iter_batches(self):
        from ur_mjlab_bc_rl.simulate.dataset_writer import Episode
        ep = Episode()
        for i in range(5):
            ep.add({
                "state": {}, "rgb": np.zeros((64, 64, 3), dtype=np.uint8),
                "depth": np.zeros((64, 64), dtype=np.float32), "task_id": 0,
            }, np.ones(7) * i)
        batches = list(ep.iter_batches(2))
        assert len(batches) == 3
        assert len(batches[0][0]) == 2
        assert len(batches[2][0]) == 1

    def test_clear(self):
        from ur_mjlab_bc_rl.simulate.dataset_writer import Episode
        ep = Episode()
        ep.add({
            "state": {}, "rgb": np.zeros((64, 64, 3), dtype=np.uint8),
            "depth": np.zeros((64, 64), dtype=np.float32), "task_id": 0,
        }, np.ones(7))
        ep.clear()
        assert len(ep) == 0

    def test_state_keys(self):
        from ur_mjlab_bc_rl.simulate.dataset_writer import DEFAULT_STATE_KEYS
        assert "arm_joint_pos" in DEFAULT_STATE_KEYS
        assert "gripper_pos" in DEFAULT_STATE_KEYS
        assert len(DEFAULT_STATE_KEYS) >= 2


class TestConfigLoader:
    """测试配置加载。"""

    def test_load_tasks(self):
        from ur_mjlab_bc_rl.simulate.config_loader import load_tasks
        tasks = load_tasks()
        assert "pick_place" in tasks
        assert tasks["pick_place"]["task_id"] == 0

    def test_get_sim_params(self):
        from ur_mjlab_bc_rl.simulate.config_loader import get_sim_params
        sim = get_sim_params()
        assert "physics_dt" in sim
        assert sim.get("physics_dt") == 0.001


class TestSimulateImports:
    """测试 simulate 模块导入。"""

    def test_env_imports(self):
        from ur_mjlab_bc_rl.simulate.env import (
            MujocoInterface, CameraSensor, ObservationCollector,
            ResetManager, MinkIK, convert_obs_to_model_input,
        )
        assert MujocoInterface is not None

    def test_teachers_imports(self):
        from ur_mjlab_bc_rl.simulate.teachers import (
            Teacher, PickPlaceTeacher, PushTTeacher, PegSlotTeacher,
        )
        assert Teacher is not None

    def test_dataset_writer_imports(self):
        from ur_mjlab_bc_rl.simulate.dataset_writer import (
            Episode, LeRobotDatasetWriter, LeRobotDatasetConfig,
            CollectionConfig,
        )
        assert CollectionConfig is not None

    def test_top_level_imports(self):
        from ur_mjlab_bc_rl.simulate import (
            MujocoInterface, CameraSensor, ObservationCollector,
            PickPlaceTeacher, Episode, LeRobotDatasetWriter,
            CollectionConfig, config_loader,
        )
        assert config_loader is not None
