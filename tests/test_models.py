"""Models 模块单元测试。

测试所有模型组件：specs, encoders, fusion, policy, distributions。

使用 pytest 运行：
  pytest tests/test_models.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class TestEncoderOutput:
    """测试 EncoderOutput 数据类。"""

    def test_vector_only(self):
        from ur_mjlab_bc_rl.models.specs import EncoderOutput

        v = torch.randn(4, 256)
        out = EncoderOutput(vector=v)
        assert out.get_vector().shape == (4, 256)
        # tokens 为 None 时 get_tokens 抛 ValueError
        with pytest.raises(ValueError):
            out.get_tokens()

    def test_tokens_only(self):
        from ur_mjlab_bc_rl.models.specs import EncoderOutput

        t = torch.randn(4, 64, 128)
        out = EncoderOutput(vector=None, tokens=t)
        # get_vector should pool tokens
        vec = out.get_vector()
        assert vec.shape == (4, 128)
        assert out.get_tokens().shape == (4, 64, 128)

    def test_both(self):
        from ur_mjlab_bc_rl.models.specs import EncoderOutput

        v = torch.randn(4, 256)
        t = torch.randn(4, 64, 128)
        out = EncoderOutput(vector=v, tokens=t)
        assert out.get_vector().shape == (4, 256)
        assert out.get_tokens().shape == (4, 64, 128)


class TestSpecs:
    """测试 ObsSpec, ActionSpec, TaskSpec。"""

    def test_obs_spec(self):
        from ur_mjlab_bc_rl.models.specs import ObsSpec

        spec = ObsSpec(
            camera_shape=(4, 128, 128),
            actor_state_dim=27,
        )
        assert spec.camera_shape == (4, 128, 128)
        assert spec.actor_state_dim == 27

    def test_action_spec(self):
        from ur_mjlab_bc_rl.models.specs import ActionSpec

        spec = ActionSpec(action_dim=7)
        assert spec.action_dim == 7

    def test_task_spec(self):
        from ur_mjlab_bc_rl.models.specs import TaskSpec

        spec = TaskSpec(num_tasks=3)
        assert spec.num_tasks == 3


class TestRGBDCNNEncoder:
    """测试 CNN 视觉编码器（普通模式 + 残差模式）。"""

    # ── 普通 CNN（向后兼容）─────────────────────────

    @pytest.fixture
    def encoder(self):
        from ur_mjlab_bc_rl.models.vision.rgbd_cnn import RGBDCNNEncoder
        return RGBDCNNEncoder(
            in_channels=4,
            image_size=(64, 64),
            hidden_dims=[32, 64, 128],
            output_dim=256,
        )

    def test_forward_plain(self, encoder):
        x = torch.randn(4, 4, 64, 64)
        out = encoder(x)
        assert out.vector.shape == (4, 256)
        assert out.tokens is None

    def test_variable_size_plain(self):
        """测试普通模式可变图像尺寸。"""
        from ur_mjlab_bc_rl.models.vision.rgbd_cnn import RGBDCNNEncoder
        encoder = RGBDCNNEncoder(
            in_channels=4,
            image_size=(128, 128),
            hidden_dims=[32, 64, 128],
            output_dim=256,
        )
        x1 = torch.randn(2, 4, 64, 64)
        x2 = torch.randn(2, 4, 128, 128)
        out1 = encoder(x1)
        out2 = encoder(x2)
        assert out1.vector.shape == (2, 256)
        assert out2.vector.shape == (2, 256)

    def test_get_output_dim_plain(self, encoder):
        assert encoder.get_output_dim() == 256

    # ── 残差 ResNet — vector 模式 ───────────────────

    @pytest.fixture
    def residual_encoder_vector(self):
        """240×320 → 512-dim vector 的残差 CNN。"""
        from ur_mjlab_bc_rl.models.vision.rgbd_cnn import RGBDCNNEncoder
        return RGBDCNNEncoder(
            in_channels=4,
            image_size=(240, 320),
            stages=[
                {"channels": 64, "blocks": 2, "stride": 1},
                {"channels": 128, "blocks": 2, "stride": 2},
                {"channels": 256, "blocks": 2, "stride": 2},
            ],
            stem_cfg={
                "channels": 64, "kernel": 7, "stride": 2,
                "pool": True, "pool_kernel": 3, "pool_stride": 2,
            },
            head_cfg={
                "global_pool": "adaptive_avg",
                "pool_size": [4, 4],
                "hidden": [256],
                "output_dim": 512,
            },
            block_cfg={"kernel_size": 3, "activation": "relu", "norm": "batch_norm"},
        )

    def test_forward_residual_vector(self, residual_encoder_vector):
        x = torch.randn(2, 4, 240, 320)
        out = residual_encoder_vector(x)
        assert out.vector.shape == (2, 512)
        assert out.tokens is None

    def test_get_output_dim_residual_vector(self, residual_encoder_vector):
        assert residual_encoder_vector.get_output_dim() == 512

    # ── 残差 ResNet — tokens 模式（无 head）─────────

    @pytest.fixture
    def residual_encoder_tokens(self):
        """无 head → 输出空间特征 tokens。"""
        from ur_mjlab_bc_rl.models.vision.rgbd_cnn import RGBDCNNEncoder
        return RGBDCNNEncoder(
            in_channels=4,
            image_size=(128, 128),
            stages=[
                {"channels": 64, "blocks": 2, "stride": 1},
                {"channels": 128, "blocks": 2, "stride": 2},
            ],
            stem_cfg={"channels": 64, "kernel": 7, "stride": 2, "pool": True},
            # 无 head_cfg → tokens 模式
        )

    def test_forward_residual_tokens(self, residual_encoder_tokens):
        x = torch.randn(2, 4, 128, 128)
        out = residual_encoder_tokens(x)
        assert out.vector is None
        assert out.tokens is not None
        # 128 → stem→64→32 → stage2→16  → 最终 16×16×128
        assert out.tokens.shape[0] == 2
        assert out.tokens.shape[-1] == 128  # 最后 stage 通道数

    def test_get_output_dim_tokens(self, residual_encoder_tokens):
        # tokens 模式下 output_dim = 最后 stage 通道数
        assert residual_encoder_tokens.get_output_dim() == 128

    # ── 残差 ResNet — 无 stem + 无 head ─────────────

    def test_residual_no_stem_no_head(self):
        """最简残差模式：无 stem + 无 head → tokens。"""
        from ur_mjlab_bc_rl.models.vision.rgbd_cnn import RGBDCNNEncoder
        encoder = RGBDCNNEncoder(
            in_channels=4,
            image_size=(64, 64),
            stages=[
                {"channels": 32, "blocks": 1, "stride": 1},
                {"channels": 64, "blocks": 1, "stride": 2},
            ],
            block_cfg={"kernel_size": 3, "activation": "relu", "norm": "batch_norm"},
        )
        x = torch.randn(2, 4, 64, 64)
        out = encoder(x)
        assert out.vector is None
        assert out.tokens is not None
        assert out.tokens.shape[-1] == 64

    # ── 残差 ResNet — 可变尺寸 ─────────────────────

    def test_residual_variable_size(self):
        """残差模式支持可变尺寸（无全局池化时直接 tokens）。"""
        from ur_mjlab_bc_rl.models.vision.rgbd_cnn import RGBDCNNEncoder
        encoder = RGBDCNNEncoder(
            in_channels=4,
            image_size=(128, 128),
            stages=[
                {"channels": 64, "blocks": 1, "stride": 2},
            ],
            head_cfg={
                "global_pool": "adaptive_avg",
                "pool_size": [4, 4],
                "hidden": [128],
                "output_dim": 256,
            },
        )
        # AdaptiveAvgPool 使不同输入尺寸均可用
        x1 = torch.randn(2, 4, 64, 64)
        x2 = torch.randn(2, 4, 128, 128)
        out1 = encoder(x1)
        out2 = encoder(x2)
        assert out1.vector.shape == (2, 256)
        assert out2.vector.shape == (2, 256)

    # ── 梯度流（残差连接不阻断梯度）───────────────

    def test_residual_gradient_flow(self, residual_encoder_vector):
        """验证残差连接下梯度可正常反向传播。"""
        x = torch.randn(2, 4, 240, 320, requires_grad=True)
        out = residual_encoder_vector(x)
        loss = out.vector.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    # ── 工厂方法 ────────────────────────────────────

    def test_factory_residual(self):
        """通过 build_visual_encoder 工厂创建残差 CNN。"""
        from ur_mjlab_bc_rl.models.vision.encoder_factory import build_visual_encoder
        cfg = {
            "type": "rgbd_cnn",
            "in_channels": 4,
            "image_size": [64, 64],
            "stages": [
                {"channels": 32, "blocks": 1, "stride": 1},
                {"channels": 64, "blocks": 1, "stride": 2},
            ],
            "head": {
                "global_pool": "adaptive_avg",
                "pool_size": [4, 4],
                "output_dim": 128,
            },
            "kernel_size": 3,
            "activation": "relu",
            "norm": "batch_norm",
        }
        encoder = build_visual_encoder(cfg)
        x = torch.randn(2, 4, 64, 64)
        out = encoder(x)
        assert out.vector.shape == (2, 128)

    def test_factory_plain_backward_compat(self):
        """工厂 — 无 stages 时回退到普通 CNN。"""
        from ur_mjlab_bc_rl.models.vision.encoder_factory import build_visual_encoder
        cfg = {
            "type": "rgbd_cnn",
            "in_channels": 4,
            "image_size": [64, 64],
            "hidden_dims": [32, 64],
            "output_dim": 128,
        }
        encoder = build_visual_encoder(cfg)
        x = torch.randn(2, 4, 64, 64)
        out = encoder(x)
        assert out.vector.shape == (2, 128)


class TestMLPStateEncoder:
    """测试 MLP 状态编码器。"""

    @pytest.fixture
    def encoder(self):
        from ur_mjlab_bc_rl.models.state.mlp_state import MLPStateEncoder
        return MLPStateEncoder(input_dim=27, hidden_dims=[128, 128], output_dim=128)

    def test_forward(self, encoder):
        x = torch.randn(4, 27)
        out = encoder(x)
        assert out.vector.shape == (4, 128)

    def test_get_output_dim(self, encoder):
        assert encoder.get_output_dim() == 128


class TestEmbeddingTaskEncoder:
    """测试 Embedding 任务编码器。"""

    @pytest.fixture
    def encoder(self):
        from ur_mjlab_bc_rl.models.task.embedding_task import EmbeddingTaskEncoder
        return EmbeddingTaskEncoder(num_tasks=3, embedding_dim=32, output_dim=64)

    def test_forward(self, encoder):
        x = torch.tensor([[0], [1], [2], [0]])
        out = encoder(x)
        assert out.vector.shape == (4, 64)

    def test_invalid_task(self, encoder):
        x = torch.tensor([[5]])  # 超出范围
        with pytest.raises(Exception):
            encoder(x)


class TestFiLMFusion:
    """测试 FiLM 融合。"""

    @pytest.fixture
    def fusion(self):
        from ur_mjlab_bc_rl.models.fusion.film import FiLMFusion
        return FiLMFusion(
            visual_dim=256,
            state_dim=128,
            task_dim=64,
        )

    def test_forward(self, fusion):
        from ur_mjlab_bc_rl.models.specs import EncoderOutput

        visual = EncoderOutput(vector=torch.randn(4, 256))
        state = EncoderOutput(vector=torch.randn(4, 128))
        task = EncoderOutput(vector=torch.randn(4, 64))

        out = fusion(visual, state, task)
        # FiLM 返回 torch.Tensor，不是 EncoderOutput
        assert out.shape == (4, 256 + 128)


class TestConcatFusion:
    """测试 Concat 融合。"""

    @pytest.fixture
    def fusion(self):
        from ur_mjlab_bc_rl.models.fusion.concat import ConcatFusion
        return ConcatFusion()

    def test_forward(self, fusion):
        from ur_mjlab_bc_rl.models.specs import EncoderOutput

        visual = EncoderOutput(vector=torch.randn(4, 256))
        state = EncoderOutput(vector=torch.randn(4, 128))
        task = EncoderOutput(vector=torch.randn(4, 64))

        out = fusion(visual, state, task)
        # Concat 返回 torch.Tensor，将所有向量拼接
        assert out.shape == (4, 256 + 128 + 64)


class TestGaussianDistribution:
    """测试高斯分布。"""

    @pytest.fixture
    def dist(self):
        from ur_mjlab_bc_rl.models.distributions import GaussianDistribution
        return GaussianDistribution(action_dim=7, log_std_init=0.0)

    def test_sample(self, dist):
        mean = torch.randn(4, 7)
        sample = dist.sample(mean)
        assert sample.shape == (4, 7)

    def test_log_prob(self, dist):
        mean = torch.randn(4, 7)
        action = torch.randn(4, 7)
        log_prob = dist.log_prob(action, mean)
        assert log_prob.shape == (4,)

    def test_entropy(self, dist):
        entropy = dist.entropy()
        assert entropy.shape == ()


class TestUR5MultimodalActor:
    """测试主 Actor 模型。"""

    @pytest.fixture
    def actor_cfg(self):
        return {
            "visual_encoder": {"type": "rgbd_cnn", "output_dim": 128},
            "state_encoder": {"type": "mlp", "input_dim": 27, "output_dim": 64},
            "task_encoder": {"type": "embedding", "num_tasks": 3, "embedding_dim": 16, "output_dim": 32},
            "fusion": {"type": "film"},
            "policy_mlp": {"hidden_dims": [256, 128]},
        }

    @pytest.fixture
    def actor(self, actor_cfg):
        from ur_mjlab_bc_rl.models.policy.multimodal_backbone import UR5MultimodalBackbone
        return UR5MultimodalBackbone(model_cfg=actor_cfg)

    def test_forward_inference(self, actor):
        obs = {
            "camera": torch.randn(4, 4, 64, 64),
            "actor_state": torch.randn(4, 27),
            "task": torch.randint(0, 3, (4, 1)),
        }
        action = actor(obs, deterministic=True)
        assert action.shape == (4, 7)

    def test_forward_training(self, actor):
        obs = {
            "camera": torch.randn(4, 4, 64, 64),
            "actor_state": torch.randn(4, 27),
            "task": torch.randint(0, 3, (4, 1)),
        }
        action = actor(obs, deterministic=False)
        assert action.shape == (4, 7)

    def test_act_inference(self, actor):
        obs = {
            "camera": torch.randn(2, 4, 64, 64),
            "actor_state": torch.randn(2, 27),
            "task": torch.randint(0, 3, (2, 1)),
        }
        action = actor.act_inference(obs)
        assert action.shape == (2, 7)

    def test_get_action_dist_params(self, actor):
        obs = {
            "camera": torch.randn(2, 4, 64, 64),
            "actor_state": torch.randn(2, 27),
            "task": torch.randint(0, 3, (2, 1)),
        }
        mean, std = actor.get_action_dist_params(obs)
        assert mean.shape == (2, 7)
        # std 是共享参数，形状为 [7]，不依赖 batch
        assert std.shape == (7,)

    def test_num_params(self, actor):
        n = sum(p.numel() for p in actor.parameters())
        assert n > 100_000  # 应至少有 10 万参数
        assert n < 100_000_000  # 但不超过 1 亿


class TestUR5RslActorModel:
    """测试 RSL-RL 适配器模型。"""

    @pytest.fixture
    def rsl_actor(self):
        from ur_mjlab_bc_rl.models.policy.rsl_adapter import UR5RslActorModel
        from tensordict import TensorDict

        bs = 2
        obs = TensorDict({
            "camera_rgb": torch.randn(bs, 3, 64, 64),
            "camera_depth": torch.randn(bs, 1, 64, 64),
            "joint_pos": torch.randn(bs, 6),
            "joint_vel": torch.randn(bs, 6),
            "task": torch.randint(0, 3, (bs, 1)),
        }, batch_size=bs)

        obs_groups = {
            "actor": ["camera_rgb", "camera_depth", "joint_pos", "joint_vel", "task"],
        }

        arch_cfg = {
            "visual_encoder": {"type": "rgbd_cnn", "output_dim": 128},
            "state_encoder": {"type": "mlp", "input_dim": 12, "output_dim": 64},
            "task_encoder": {"type": "embedding", "num_tasks": 3, "embedding_dim": 16, "output_dim": 32},
            "fusion": {"type": "concat"},
            "policy_mlp": {"hidden_dims": [256, 128]},
        }

        return UR5RslActorModel(
            obs=obs,
            obs_groups=obs_groups,
            obs_set="actor",
            output_dim=7,
            architecture_cfg=arch_cfg,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        )

    def test_forward_deterministic(self, rsl_actor):
        from tensordict import TensorDict
        obs = TensorDict({
            "camera_rgb": torch.randn(2, 3, 64, 64),
            "camera_depth": torch.randn(2, 1, 64, 64),
            "joint_pos": torch.randn(2, 6),
            "joint_vel": torch.randn(2, 6),
            "task": torch.randint(0, 3, (2, 1)),
        }, batch_size=2)
        out = rsl_actor(obs, stochastic_output=False)
        assert out.shape == (2, 7)

    def test_forward_stochastic(self, rsl_actor):
        from tensordict import TensorDict
        obs = TensorDict({
            "camera_rgb": torch.randn(2, 3, 64, 64),
            "camera_depth": torch.randn(2, 1, 64, 64),
            "joint_pos": torch.randn(2, 6),
            "joint_vel": torch.randn(2, 6),
            "task": torch.randint(0, 3, (2, 1)),
        }, batch_size=2)
        out = rsl_actor(obs, stochastic_output=True)
        assert out.shape == (2, 7)
        log_prob = rsl_actor.get_output_log_prob(out)
        assert log_prob.shape == (2,)

    def test_distribution_properties(self, rsl_actor):
        from tensordict import TensorDict
        obs = TensorDict({
            "camera_rgb": torch.randn(2, 3, 64, 64),
            "camera_depth": torch.randn(2, 1, 64, 64),
            "joint_pos": torch.randn(2, 6),
            "joint_vel": torch.randn(2, 6),
            "task": torch.randint(0, 3, (2, 1)),
        }, batch_size=2)
        _ = rsl_actor(obs, stochastic_output=True)
        assert rsl_actor.output_mean.shape == (2, 7)
        assert rsl_actor.output_std.shape == (2, 7)
        assert rsl_actor.output_entropy.shape == (2,)
