"""Reinforcement 模块单元测试。

测试 BC checkpoint 加载和 PPO 桥接。

使用 pytest 运行：
  pytest tests/test_reinforcement.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class TestCheckpointUtils:
    """测试 BC checkpoint 工具。"""

    @pytest.fixture
    def test_ckpt_path(self, tmp_path):
        """创建测试 checkpoint。"""
        from ur_mjlab_bc_rl.models.policy.multimodal_backbone import UR5MultimodalBackbone

        actor = UR5MultimodalBackbone(model_cfg={
            "visual_encoder": {"type": "rgbd_cnn", "output_dim": 64},
            "state_encoder": {"type": "mlp", "input_dim": 27, "output_dim": 32},
            "task_encoder": {"type": "embedding", "num_tasks": 3, "embedding_dim": 8, "output_dim": 16},
            "fusion": {"type": "film"},
            "policy_mlp": {"hidden_dims": [128, 64]},
        })
        actor.model_cfg = {"type": "ur5_multimodal"}

        path = tmp_path / "test_bc.pt"
        torch.save({
            "actor_state_dict": actor.state_dict(),
            "model_cfg": actor.model_cfg,
            "train_losses": [1.5, 1.0, 0.5],
        }, path)
        return path

    def test_load_bc_checkpoint(self, test_ckpt_path):
        from ur_mjlab_bc_rl.reinforcement.checkpoint_utils import load_bc_checkpoint
        ckpt = load_bc_checkpoint(test_ckpt_path)
        assert "actor_state_dict" in ckpt
        assert "model_cfg" in ckpt

    def test_inspect_bc_checkpoint(self, test_ckpt_path):
        from ur_mjlab_bc_rl.reinforcement.checkpoint_utils import inspect_bc_checkpoint
        info = inspect_bc_checkpoint(test_ckpt_path)
        assert "num_params" in info
        assert info["num_params"] > 0
        assert "file_size_mb" in info

    def test_load_nonexistent(self):
        from ur_mjlab_bc_rl.reinforcement.checkpoint_utils import load_bc_checkpoint
        with pytest.raises(FileNotFoundError):
            load_bc_checkpoint("/nonexistent/path.pt")


class TestBCToPPO:
    """测试 BC → PPO 桥接。"""

    @pytest.fixture
    def bc_ckpt(self, tmp_path):
        from ur_mjlab_bc_rl.models.policy.multimodal_backbone import UR5MultimodalBackbone

        actor = UR5MultimodalBackbone(model_cfg={
            "visual_encoder": {"type": "rgbd_cnn", "output_dim": 64},
            "state_encoder": {"type": "mlp", "input_dim": 27, "output_dim": 32},
            "task_encoder": {"type": "embedding", "num_tasks": 3, "embedding_dim": 8, "output_dim": 16},
            "fusion": {"type": "film"},
            "policy_mlp": {"hidden_dims": [128, 64]},
        })
        actor.model_cfg = {"type": "ur5_multimodal"}

        path = tmp_path / "test_bc.pt"
        torch.save({
            "actor_state_dict": actor.state_dict(),
            "model_cfg": actor.model_cfg,
        }, path)
        return path, actor

    def test_extract_policy_mlp(self, bc_ckpt):
        from ur_mjlab_bc_rl.reinforcement.checkpoint_utils import load_bc_checkpoint
        from ur_mjlab_bc_rl.reinforcement.bc_to_ppo import extract_policy_mlp_weights

        ckpt_path, _ = bc_ckpt
        ckpt = load_bc_checkpoint(ckpt_path)
        weights = extract_policy_mlp_weights(ckpt)
        assert len(weights) > 0
        for key in weights:
            assert "policy_mlp" not in key  # 已重命名

    def test_extract_encoder(self, bc_ckpt):
        from ur_mjlab_bc_rl.reinforcement.checkpoint_utils import load_bc_checkpoint
        from ur_mjlab_bc_rl.reinforcement.bc_to_ppo import extract_encoder_weights

        ckpt_path, _ = bc_ckpt
        ckpt = load_bc_checkpoint(ckpt_path)
        weights = extract_encoder_weights(ckpt)
        assert len(weights) > 0

    def test_create_ppo_init(self, bc_ckpt, tmp_path):
        from ur_mjlab_bc_rl.reinforcement.bc_to_ppo import create_ppo_init_checkpoint

        ckpt_path, _ = bc_ckpt
        output = tmp_path / "ppo_init.pt"
        create_ppo_init_checkpoint(ckpt_path, output, extract_mode="all")

        assert output.exists()
        ppo_ckpt = torch.load(output, map_location="cpu", weights_only=False)
        assert "policy_mlp_weights" in ppo_ckpt
        assert "encoder_weights" in ppo_ckpt
        assert "source" in ppo_ckpt
