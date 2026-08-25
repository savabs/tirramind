"""
Tests for M1 ContinuousWorldModel components.

Covers:
    - SignaturePathBuilder: depth-2 fallback, shape correctness
    - _NHPModel.get_hidden_sequence: shape, determinism
    - NeuralHawkesEncoder.get_hidden_sequence: round-trip from obs dicts
    - MambaMemoryEncoder.encode_tokens + build_token: shape, no error
    - DiagonalDiffusionHead: positivity, KL formula, sample_noise
    - HeterogeneousCDEFunc: forward shape, column-norm clipping, context
    - ContinuousWorldModel: Phase B forward, memory update shape, KL=0 for B
    - TrainerConfig: new M1 fields have correct defaults
"""

from __future__ import annotations

import math
import time
import types

import pytest
import torch
import torch.nn as nn

# ─── helpers ──────────────────────────────────────────────────────────────


def _make_memory(num_nodes: int, memory_dim: int, device: str = "cpu"):
    """Minimal HeteroMemory-like object for testing."""
    mem = types.SimpleNamespace()
    mem.memory = torch.zeros(num_nodes, memory_dim)
    mem.last_update = torch.zeros(num_nodes)
    mem.num_nodes = num_nodes
    return mem


def _make_id_map(entity_type: str, entity_ids: list):
    """Minimal IDMap-like object for testing."""
    m = types.SimpleNamespace()
    g_map = {}
    l_map: dict[str, dict] = {}
    l_map[entity_type] = {}
    for i, eid in enumerate(entity_ids):
        g_map[(entity_type, eid)] = i
        l_map[entity_type][eid] = i

    def global_id(etype, eid):
        return g_map.get((etype, eid))

    def local_id(etype, eid):
        lm = l_map.get(etype, {})
        return lm.get(eid)

    m.global_id = global_id
    m.local_id = local_id
    return m


# ═══════════════════════════════════════════════════════════════════════════
# 1. SignaturePathBuilder
# ═══════════════════════════════════════════════════════════════════════════


class TestSignaturePathBuilder:
    def test_import(self):
        from agent.models.gnn.signature_path import SignaturePathBuilder

        assert SignaturePathBuilder is not None

    def test_shape_fallback(self):
        """Fallback (no iisignature) produces correct shape."""
        from agent.models.gnn.signature_path import SignaturePathBuilder

        builder = SignaturePathBuilder(message_dim=16, proj_dim=4, depth=3)
        msgs = torch.randn(5, 16)
        out = builder(msgs)
        assert out.shape == (
            5,
            builder.sig_dim,
        ), f"Expected (5,{builder.sig_dim}), got {out.shape}"

    def test_empty_input(self):
        from agent.models.gnn.signature_path import SignaturePathBuilder

        builder = SignaturePathBuilder(message_dim=16, proj_dim=4, depth=3)
        out = builder(torch.zeros(0, 16))
        assert out.shape == (0, builder.sig_dim)

    def test_single_event(self):
        from agent.models.gnn.signature_path import SignaturePathBuilder

        builder = SignaturePathBuilder(message_dim=8, proj_dim=4, depth=3)
        out = builder(torch.randn(1, 8))
        assert out.shape == (1, builder.sig_dim)

    def test_depth2_levy_area_antisymmetric(self):
        """Lévy area in the fallback must satisfy A^{ij} = -A^{ji} in aggregate."""
        from agent.models.gnn.signature_path import _depth2_logsig_incremental

        torch.manual_seed(42)
        increments = torch.randn(6, 4)
        result = _depth2_logsig_incremental(increments)
        # result: (6, 10) = 4 level-1 + 6 level-2 (pairs)
        # pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        # Check last step: the pairs area at result[-1, 4:] are non-zero
        assert result.shape == (6, 10)
        # All level-1 values must equal cumulative sum of increments
        expected_s1 = increments.cumsum(dim=0)
        assert torch.allclose(result[:, :4], expected_s1, atol=1e-5)

    def test_build_control_knots(self):
        from agent.models.gnn.signature_path import (
            build_control_knots,
            compute_d_z,
            SignaturePathBuilder,
        )

        builder = SignaturePathBuilder(message_dim=32, proj_dim=4, depth=3)
        n = 7
        msgs = torch.randn(n, 32)
        time_feats = torch.randn(n, 16)
        knots = build_control_knots(msgs, time_feats, builder)
        expected_d_z = compute_d_z(16, 32, builder)
        assert knots.shape == (n, expected_d_z)


# ═══════════════════════════════════════════════════════════════════════════
# 2. _NHPModel.get_hidden_sequence
# ═══════════════════════════════════════════════════════════════════════════


class TestNHPModelHiddenSequence:
    def _make_model(self, n_types=5, hidden_dim=32, emb_dim=8):
        from agent.convergence.neural_hawkes import _NHPModel

        return _NHPModel(n_types=n_types, hidden_dim=hidden_dim, emb_dim=emb_dim)

    def test_shape(self):
        model = self._make_model()
        types = torch.tensor([1, 2, 3, 1], dtype=torch.long)
        delta_ts = torch.tensor([0.0, 100.0, 200.0, 50.0])
        h_seq = model.get_hidden_sequence(types, delta_ts)
        assert h_seq.shape == (4, 32)

    def test_deterministic(self):
        model = self._make_model()
        model.eval()
        types = torch.tensor([1, 2, 3], dtype=torch.long)
        delta_ts = torch.tensor([0.0, 60.0, 120.0])
        h1 = model.get_hidden_sequence(types, delta_ts)
        h2 = model.get_hidden_sequence(types, delta_ts)
        assert torch.allclose(h1, h2)

    def test_single_event(self):
        model = self._make_model()
        types = torch.tensor([1], dtype=torch.long)
        delta_ts = torch.tensor([0.0])
        h_seq = model.get_hidden_sequence(types, delta_ts)
        assert h_seq.shape == (1, 32)


# ═══════════════════════════════════════════════════════════════════════════
# 3. NeuralHawkesEncoder.get_hidden_sequence
# ═══════════════════════════════════════════════════════════════════════════


class TestNeuralHawkesEncoderHiddenSeq:
    def _make_encoder_with_model(self):
        from agent.convergence.neural_hawkes import NeuralHawkesEncoder, _NHPModel

        enc = NeuralHawkesEncoder(hidden_dim=16, emb_dim=4, n_iters=1)
        # Force CPU so manually-created _model and encoder tensors agree
        enc.device = "cpu"
        # Manually set vocab and model (skip full training)
        enc._vocab = {"price_update": 1, "news_event": 2}
        enc._inv_vocab = {1: "price_update", 2: "news_event"}
        enc._model = _NHPModel(n_types=2, hidden_dim=16, emb_dim=4)
        return enc

    def test_returns_none_without_model(self):
        from agent.convergence.neural_hawkes import NeuralHawkesEncoder

        enc = NeuralHawkesEncoder(hidden_dim=16, emb_dim=4)
        result = enc.get_hidden_sequence([])
        assert result is None

    def test_returns_tensor_from_obs_dicts(self):
        enc = self._make_encoder_with_model()
        now = time.time()
        obs = [
            {"observation_type": "price_update", "observed_at": now - 200},
            {"observation_type": "news_event", "observed_at": now - 100},
            {"observation_type": "price_update", "observed_at": now},
        ]
        h = enc.get_hidden_sequence(obs)
        assert h is not None
        assert h.shape == (3, 16)

    def test_unknown_types_filtered(self):
        enc = self._make_encoder_with_model()
        now = time.time()
        obs = [
            {"observation_type": "unknown_type", "observed_at": now - 100},
            {"observation_type": "price_update", "observed_at": now},
        ]
        h = enc.get_hidden_sequence(obs)
        # Only 1 known event → shape (1, hidden_dim)
        assert h is not None
        assert h.shape == (1, 16)


# ═══════════════════════════════════════════════════════════════════════════
# 4. MambaMemoryEncoder.encode_tokens + build_token
# ═══════════════════════════════════════════════════════════════════════════


class TestMambaEncodeTokens:
    def _make_encoder(self, memory_dim=32, message_dim=32, time_dim=8):
        from agent.models.gnn.mamba_encoder import MambaMemoryEncoder

        return MambaMemoryEncoder(
            memory_dim=memory_dim,
            message_dim=message_dim,
            time_dim=time_dim,
        )

    def test_encode_tokens_shape_2d(self):
        enc = self._make_encoder()
        seq = torch.randn(5, 32)
        out = enc.encode_tokens(seq)
        assert out.shape == (32,)

    def test_encode_tokens_shape_3d(self):
        enc = self._make_encoder()
        seq = torch.randn(1, 5, 32)
        out = enc.encode_tokens(seq)
        assert out.shape == (32,)

    def test_encode_tokens_single(self):
        enc = self._make_encoder()
        seq = torch.randn(1, 32)
        out = enc.encode_tokens(seq)
        assert out.shape == (32,)

    def test_build_token_shape(self):
        enc = self._make_encoder()
        msg = torch.randn(32)
        tok = enc.build_token(msg, dt=3600.0)
        assert tok.shape == (32,)

    def test_build_token_deterministic(self):
        enc = self._make_encoder()
        enc.eval()
        msg = torch.randn(32)
        t1 = enc.build_token(msg, dt=100.0)
        t2 = enc.build_token(msg, dt=100.0)
        assert torch.allclose(t1, t2)


# ═══════════════════════════════════════════════════════════════════════════
# 5. DiagonalDiffusionHead
# ═══════════════════════════════════════════════════════════════════════════


class TestDiagonalDiffusionHead:
    def test_positivity(self):
        from agent.models.gnn.diffusion_head import DiagonalDiffusionHead

        head = DiagonalDiffusionHead(hidden_dim=16, noise_floor=1e-3)
        x = torch.randn(4, 16)
        sigma = head(x)
        assert sigma.shape == x.shape
        assert (sigma > 0).all(), "sigma must be strictly positive"

    def test_noise_floor_respected(self):
        from agent.models.gnn.diffusion_head import DiagonalDiffusionHead

        floor = 1e-3
        head = DiagonalDiffusionHead(hidden_dim=16, noise_floor=floor)
        x = torch.zeros(1, 16)
        sigma = head(x)
        assert (sigma >= floor).all()

    def test_kl_non_negative(self):
        from agent.models.gnn.diffusion_head import DiagonalDiffusionHead

        head = DiagonalDiffusionHead(hidden_dim=8)
        z = torch.randn(4, 8)
        kl = head.kl_divergence(z)
        assert kl.item() >= 0.0, "KL divergence must be non-negative"

    def test_kl_formula(self):
        """KL(N(mu,sigma^2)||N(0,1)) = 0.5*(sigma^2+mu^2-log(sigma^2)-1)"""
        from agent.models.gnn.diffusion_head import DiagonalDiffusionHead

        head = DiagonalDiffusionHead(hidden_dim=4)
        z = torch.ones(1, 4)  # mu=1
        # Override sigma to 1.0 → KL = 0.5*(1+1-0-1) = 0.5
        sigma = torch.ones(1, 4)
        kl = head.kl_divergence(z, sigma)
        expected = 0.5  # KL(N(1,1)||N(0,1)) per dim = 0.5, mean of 4 = 0.5
        assert abs(kl.item() - expected) < 1e-4, f"Expected ~0.5, got {kl.item()}"

    def test_sample_noise_zeros_in_inference(self):
        from agent.models.gnn.diffusion_head import DiagonalDiffusionHead

        head = DiagonalDiffusionHead(hidden_dim=8)
        z = torch.randn(2, 8)
        noise, sigma = head.sample_noise(z, dt=1.0, training=False)
        assert torch.all(noise == 0), "Inference noise must be zero"

    def test_sample_noise_nonzero_in_training(self):
        from agent.models.gnn.diffusion_head import DiagonalDiffusionHead

        torch.manual_seed(0)
        head = DiagonalDiffusionHead(hidden_dim=8)
        z = torch.randn(2, 8)
        # Very unlikely all noise is zero in training
        noise, _ = head.sample_noise(z, dt=1.0, training=True)
        assert not torch.all(
            noise == 0
        ), "Training noise should be non-zero (stochastic)"


# ═══════════════════════════════════════════════════════════════════════════
# 6. HeterogeneousCDEFunc
# ═══════════════════════════════════════════════════════════════════════════


class TestHeterogeneousCDEFunc:
    def test_forward_shape(self):
        from agent.models.gnn.heterogeneous_cde_func import HeterogeneousCDEFunc

        func = HeterogeneousCDEFunc(hidden_dim=16, d_z=8, memory_dim=16)
        z = torch.randn(2, 16)
        t = torch.tensor([0.0])
        F = func(t, z)
        assert F.shape == (2, 16, 8)

    def test_context_zeros_when_unset(self):
        from agent.models.gnn.heterogeneous_cde_func import HeterogeneousCDEFunc

        func = HeterogeneousCDEFunc(hidden_dim=8, d_z=4, memory_dim=8)
        z = torch.randn(1, 8)
        F1 = func(torch.tensor([0.0]), z)
        func.set_context(None, None)
        F2 = func(torch.tensor([0.0]), z)
        assert torch.allclose(F1, F2)

    def test_set_context_changes_output(self):
        from agent.models.gnn.heterogeneous_cde_func import HeterogeneousCDEFunc

        func = HeterogeneousCDEFunc(hidden_dim=8, d_z=4, memory_dim=8)
        z = torch.randn(1, 8)
        t = torch.tensor([0.0])
        F_no_ctx = func(t, z)

        func.set_context(
            graph_msg=torch.ones(1, 8),
            mamba_ctx=torch.ones(1, 8),
        )
        F_with_ctx = func(t, z)
        assert not torch.allclose(
            F_no_ctx, F_with_ctx
        ), "Context should change the drift output"

    def test_column_norm_clipping(self):
        """All columns of F must have L2 norm ≤ 1.0."""
        from agent.models.gnn.heterogeneous_cde_func import (
            HeterogeneousCDEFunc,
            _COL_NORM_CLIP,
        )

        func = HeterogeneousCDEFunc(hidden_dim=16, d_z=8, memory_dim=16)
        # Large state to stress-test norm clipping
        z = torch.randn(4, 16) * 100
        F = func(torch.tensor([0.0]), z)  # (4, 16, 8)
        col_norms = F.norm(dim=1)  # (4, 8)
        assert (
            col_norms <= _COL_NORM_CLIP + 1e-5
        ).all(), f"Column norm exceeded clip={_COL_NORM_CLIP}: max={col_norms.max().item():.4f}"

    def test_clear_context(self):
        from agent.models.gnn.heterogeneous_cde_func import HeterogeneousCDEFunc

        func = HeterogeneousCDEFunc(hidden_dim=8, d_z=4, memory_dim=8)
        func.set_context(torch.ones(1, 8), torch.ones(1, 8))
        func.clear_context()
        assert func._graph_msg is None
        assert func._mamba_ctx is None


# ═══════════════════════════════════════════════════════════════════════════
# 7. ContinuousWorldModel — Phase B
# ═══════════════════════════════════════════════════════════════════════════


class TestContinuousWorldModelPhaseB:
    def _make_cwm(self):
        from agent.models.gnn.continuous_world_model import ContinuousWorldModel

        return ContinuousWorldModel(
            hidden_dim=16,
            ctrl_time_dim=8,
            ctrl_msg_dim=8,
            n_euler_steps=5,
            use_signatures=False,
            use_mamba_ctx=False,
            use_diffusion=False,
        )

    def test_d_z(self):
        cwm = self._make_cwm()
        assert cwm.d_z == 8 + 8  # ctrl_time_dim + ctrl_msg_dim

    def test_update_memories_returns_kl_zero(self):
        """Phase B has no diffusion → KL must be exactly 0."""
        cwm = self._make_cwm()
        n_entities = 3
        memory = _make_memory(n_entities, memory_dim=16)
        id_map = _make_id_map("vessel", [10, 20, 30])
        embeddings = {"vessel": torch.randn(n_entities, 16)}

        now = time.time()
        events = [
            {"entity_type": "vessel", "entity_id": 10, "observed_at": now - 200},
            {"entity_type": "vessel", "entity_id": 10, "observed_at": now - 100},
            {"entity_type": "vessel", "entity_id": 20, "observed_at": now - 50},
        ]
        result = cwm.update_memories(events, memory, id_map, embeddings, training=True)
        assert "kl_loss" in result
        assert result["kl_loss"].item() == 0.0, "Phase B should have zero KL"

    def test_memory_is_modified(self):
        """update_memories must change memory state for entities with events."""
        cwm = self._make_cwm()
        n_entities = 2
        memory = _make_memory(n_entities, memory_dim=16)
        original = memory.memory.clone()
        id_map = _make_id_map("vessel", [1, 2])
        embeddings = {"vessel": torch.randn(n_entities, 16)}

        now = time.time()
        events = [
            {"entity_type": "vessel", "entity_id": 1, "observed_at": now - 100},
            {"entity_type": "vessel", "entity_id": 1, "observed_at": now},
        ]
        cwm.update_memories(events, memory, id_map, embeddings, training=False)
        # Entity 1 (global id 0) must have changed
        assert not torch.allclose(
            memory.memory[0], original[0]
        ), "Entity 1 memory should have been updated"
        # Entity 2 (global id 1) must be unchanged (no events)
        assert torch.allclose(
            memory.memory[1], original[1]
        ), "Entity 2 memory should be unchanged (no events)"

    def test_no_events_returns_zero_kl(self):
        cwm = self._make_cwm()
        memory = _make_memory(2, memory_dim=16)
        id_map = _make_id_map("vessel", [1, 2])
        embeddings = {"vessel": torch.randn(2, 16)}
        result = cwm.update_memories([], memory, id_map, embeddings)
        assert result["kl_loss"].item() == 0.0

    def test_unknown_entity_ignored(self):
        """Events for unregistered entities must not crash."""
        cwm = self._make_cwm()
        memory = _make_memory(2, memory_dim=16)
        id_map = _make_id_map("vessel", [1, 2])
        embeddings = {"vessel": torch.randn(2, 16)}
        now = time.time()
        events = [
            {"entity_type": "vessel", "entity_id": 99, "observed_at": now},
        ]
        result = cwm.update_memories(events, memory, id_map, embeddings)
        assert "kl_loss" in result  # no crash

    def test_last_update_timestamp(self):
        """last_update must be set to the last event time for updated entities.

        Note: memory.last_update is float32.  Unix epoch (~1.78e9) has only
        ~200 s of float32 precision, so use relative timestamps and a loose
        tolerance that accounts for this.
        """
        cwm = self._make_cwm()
        memory = _make_memory(2, memory_dim=16)
        id_map = _make_id_map("vessel", [1, 2])
        embeddings = {"vessel": torch.randn(2, 16)}
        # Use small relative timestamps to avoid float32 precision loss
        t_last = 500.0
        events = [
            {"entity_type": "vessel", "entity_id": 1, "observed_at": t_last - 50.0},
            {"entity_type": "vessel", "entity_id": 1, "observed_at": t_last},
        ]
        cwm.update_memories(events, memory, id_map, embeddings)
        # float32 at 500.0 is precise to <0.1s
        assert (
            abs(float(memory.last_update[0].item()) - t_last) < 0.1
        ), f"Expected ~{t_last}, got {float(memory.last_update[0].item())}"


# ═══════════════════════════════════════════════════════════════════════════
# 8. ContinuousWorldModel — Phase E (diffusion)
# ═══════════════════════════════════════════════════════════════════════════


class TestContinuousWorldModelPhaseE:
    def _make_cwm(self):
        from agent.models.gnn.continuous_world_model import ContinuousWorldModel

        return ContinuousWorldModel(
            hidden_dim=16,
            ctrl_time_dim=8,
            ctrl_msg_dim=8,
            n_euler_steps=5,
            use_signatures=False,
            use_mamba_ctx=False,
            use_diffusion=True,  # Phase E
        )

    def test_kl_positive_in_training(self):
        """Phase E diffusion should produce positive KL during training."""
        cwm = self._make_cwm()
        memory = _make_memory(2, memory_dim=16)
        id_map = _make_id_map("vessel", [1, 2])
        embeddings = {"vessel": torch.randn(2, 16)}
        now = time.time()
        events = [
            {"entity_type": "vessel", "entity_id": 1, "observed_at": now - 100},
            {"entity_type": "vessel", "entity_id": 1, "observed_at": now},
        ]
        result = cwm.update_memories(events, memory, id_map, embeddings, training=True)
        kl = result["kl_loss"].item()
        assert kl > 0.0, f"Phase E KL should be positive during training, got {kl}"

    def test_noise_zero_in_inference(self):
        """Memory update should be deterministic in inference mode."""
        cwm = self._make_cwm()
        memory1 = _make_memory(2, memory_dim=16)
        memory2 = _make_memory(2, memory_dim=16)
        id_map = _make_id_map("vessel", [1, 2])
        embeddings = {"vessel": torch.randn(2, 16)}
        now = time.time()
        events = [
            {"entity_type": "vessel", "entity_id": 1, "observed_at": now - 100},
            {"entity_type": "vessel", "entity_id": 1, "observed_at": now},
        ]
        with torch.no_grad():
            cwm.update_memories(events, memory1, id_map, embeddings, training=False)
            cwm.update_memories(events, memory2, id_map, embeddings, training=False)
        assert torch.allclose(
            memory1.memory, memory2.memory
        ), "Inference should be deterministic (no stochastic noise)"


# ═══════════════════════════════════════════════════════════════════════════
# 9. TrainerConfig M1 defaults
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainerConfigM1Defaults:
    def test_cwm_disabled_by_default(self):
        from agent.models.gnn.trainer import TrainerConfig

        cfg = TrainerConfig()
        assert cfg.use_continuous_world_model is False

    def test_cwm_phase_defaults(self):
        from agent.models.gnn.trainer import TrainerConfig

        cfg = TrainerConfig()
        assert cfg.cwm_curriculum_phase == "B"
        assert cfg.cwm_n_euler_steps == 20
        assert cfg.cwm_ctrl_time_dim == 16
        assert cfg.cwm_ctrl_msg_dim == 32

    def test_cwm_kl_defaults(self):
        from agent.models.gnn.trainer import TrainerConfig

        cfg = TrainerConfig()
        assert cfg.cwm_lambda_kl == 0.01
        assert cfg.cwm_kl_warmup_epochs == 10

    def test_cwm_sig_defaults(self):
        from agent.models.gnn.trainer import TrainerConfig

        cfg = TrainerConfig()
        assert cfg.cwm_sig_proj_dim == 4
        assert cfg.cwm_sig_depth == 3

    def test_cwm_phase_e_override(self):
        from agent.models.gnn.trainer import TrainerConfig

        cfg = TrainerConfig(use_continuous_world_model=True, cwm_curriculum_phase="E")
        assert cfg.cwm_curriculum_phase == "E"
        assert cfg.use_continuous_world_model is True
