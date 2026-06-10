"""Imitation 模块单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class TestEpisode:
    """测试 Episode 数据类。"""

    def test_basic(self):
        from ur_mjlab_bc_rl.imitation.dataset import Episode
        ep = Episode()
        assert len(ep) == 0
        ep.add({"state": {}, "rgb": np.zeros((64,64,3)), "depth": np.zeros((64,64)), "task_id": 0}, np.ones(7))
        assert len(ep) == 1
        assert ep.actions[0].shape == (7,)

    def test_to_from_dict(self):
        from ur_mjlab_bc_rl.imitation.dataset import Episode
        ep = Episode()
        ep.add({"state": {}, "rgb": np.zeros((64,64,3)), "depth": np.zeros((64,64)), "task_id": 0}, np.ones(7))
        ep.add({"state": {}, "rgb": np.ones((64,64,3)), "depth": np.ones((64,64)), "task_id": 1}, np.zeros(7))

        d = ep.to_dict()
        ep2 = Episode.from_dict(d)
        assert len(ep2) == 2
        assert np.allclose(ep2.actions[0], np.ones(7))


class TestLeRobotSaver:
    """测试 LeRobotSaver。"""

    def test_infer_dims(self, tmp_path):
        from ur_mjlab_bc_rl.imitation.dataset import Episode, LeRobotSaver
        ep = Episode()
        from collections import OrderedDict
        state = OrderedDict([("a", np.zeros(6)), ("b", np.ones(3))])
        ep.add({"state": state, "rgb": np.zeros((64,64,3), dtype=np.uint8), "depth": np.ones((64,64), dtype=np.float32), "task_id": 0}, np.ones(7))

        saver = LeRobotSaver()
        assert saver._infer_state_dim([ep]) == 9
        assert saver._infer_image_size([ep]) == (64, 64)
        assert saver._infer_action_dim([ep]) == 7


class TestImitationDataset:
    """测试 ImitationDataset。"""

    @pytest.fixture
    def episodes(self):
        eps = []
        for _ in range(5):
            obs_list = []
            act_list = []
            for _ in range(10):
                obs_list.append({
                    "camera": np.random.randn(4, 64, 64).astype(np.float32),
                    "actor_state": np.random.randn(27).astype(np.float32),
                    "task": np.array([0], dtype=np.int64),
                })
                act_list.append(np.random.randn(7).astype(np.float32))
            eps.append({
                "observations": obs_list,
                "actions": [a.tolist() for a in act_list],
                "rewards": [1.0] * 10,
                "dones": [False] * 9 + [True],
                "infos": [{}] * 10,
            })
        return eps

    def test_dataset_from_episodes(self, episodes):
        from ur_mjlab_bc_rl.imitation.dataset import ImitationDataset
        dataset = ImitationDataset(episodes)
        assert len(dataset) == 50

    def test_dataset_getitem(self, episodes):
        from ur_mjlab_bc_rl.imitation.dataset import ImitationDataset
        dataset = ImitationDataset(episodes)
        item = dataset[0]
        assert "camera" in item
        assert "actor_state" in item
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                assert not torch.isnan(v).any()


class TestLosses:
    """测试损失函数。"""

    def test_mse_loss(self):
        from ur_mjlab_bc_rl.imitation.training.losses import build_loss
        loss_fn = build_loss("mse")
        pred = torch.randn(4, 7)
        target = torch.randn(4, 7)
        loss = loss_fn(pred, target)
        assert loss >= 0

    def test_l1_loss(self):
        from ur_mjlab_bc_rl.imitation.training.losses import build_loss
        loss_fn = build_loss("l1")
        pred = torch.randn(4, 7)
        target = torch.randn(4, 7)
        loss = loss_fn(pred, target)
        assert loss >= 0


class TestImitationTrainer:
    """测试 ImitationTrainer。"""

    @pytest.fixture
    def trainer_setup(self):
        from ur_mjlab_bc_rl.models.policy.multimodal_backbone import UR5MultimodalBackbone
        from ur_mjlab_bc_rl.imitation.dataset import ImitationDataset
        from ur_mjlab_bc_rl.imitation.training import ImitationTrainer
        from torch.utils.data import DataLoader

        eps = []
        for _ in range(3):
            obs_list = [{"camera": np.random.randn(4, 64, 64).astype(np.float32), "actor_state": np.random.randn(27).astype(np.float32), "task": np.array([0])} for _ in range(20)]
            act_list = [np.random.randn(7).astype(np.float32) for _ in range(20)]
            eps.append({"observations": obs_list, "actions": [a.tolist() for a in act_list]})

        dataset = ImitationDataset(eps)
        loader = DataLoader(dataset, batch_size=8)

        actor = UR5MultimodalBackbone(model_cfg={
            "visual_encoder": {"type": "rescnn", "output_dim": 64},
            "state_encoder": {"type": "mlp", "input_dim": 27, "output_dim": 32},
            "task_encoder": {"type": "embedding", "num_tasks": 3, "embedding_dim": 8, "output_dim": 16},
            "fusion": {"type": "film"},
            "policy_mlp": {"hidden_dims": [128, 64]},
        })
        trainer = ImitationTrainer(actor, loader, lr=1e-3)
        return trainer

    def test_train_one_epoch(self, trainer_setup):
        loss = trainer_setup.train_epoch()
        assert loss > 0
