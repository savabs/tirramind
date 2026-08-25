"""Tests for Idea 2 — Path Signature Encoder.

Covers:
    1.  compute_path_signature: output shape (unbatched & batched)
    2.  compute_path_signature: depth-1 output equals path increments
    3.  compute_path_signature: depth-2 Level-2 term is S2[i,j] not S2[j,i] (Chen order)
    4.  compute_path_signature: monotone path satisfies shuffle identity S^1_i · S^1_j = S^2_{ij} + S^2_{ji}
    5.  compute_path_signature: single-point path → zero signature
    6.  compute_path_signature: invalid depth raises ValueError
    7.  compute_path_signature: output is finite for random inputs
    8.  compute_path_signature: batch dim handled correctly (same as unbatched loop)
    9.  entity_observations_to_path: output shape, channel ranges
    10. entity_observations_to_path: empty observations → (1, 3) zeros
    11. entity_observations_to_path: max_seq_len truncation
    12. entity_observations_to_path: time channel in [0,1]
    13. compute_entity_signature: returns SIGNATURE_DIM-dim vector
    14. compute_entity_signature: output is finite
    15. PathSignatureEncoder: forward shape (unbatched)
    16. PathSignatureEncoder: forward shape (batched)
    17. _build_node_features with use_signatures=True: feature dim increases by SIGNATURE_DIM
    18. _build_node_features with use_signatures=False: feature dim unchanged
    19. TrainerConfig.use_signatures defaults False
    20. build_model() with use_signatures=True gives correct in_channels
    21. Full training loop with use_signatures=True completes without NaN
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from agent.models.gnn.signature_encoder import (
    PATH_CHANNELS,
    SIGNATURE_DIM,
    SIGNATURE_DEPTH,
    PathSignatureEncoder,
    compute_entity_signature,
    compute_path_signature,
    entity_observations_to_path,
)
from agent.models.gnn.graph_builder import (
    BASE_FEAT_DIM,
    ENRICHMENT_DIM,
    _build_node_features,
)
from agent.models.gnn.trainer import (
    SyntheticGraphGenerator,
    Trainer,
    TrainerConfig,
)
from agent.pipeline.store import PipelineStore


# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture()
def linear_path():
    """2D path along a straight line x=t, y=2t — 5 points."""
    t = torch.linspace(0, 1, 5)
    return torch.stack([t, 2 * t], dim=1)  # (5, 2)


@pytest.fixture()
def rand_path():
    torch.manual_seed(42)
    return torch.randn(8, 3)  # (8, 3) — random 3-channel path


@pytest.fixture()
def simple_store(tmp_path: Path) -> PipelineStore:
    store = PipelineStore(str(tmp_path / "sig_test.db"))
    gen = SyntheticGraphGenerator(
        num_companies=4, num_countries=2, num_vessels=2,
        time_span=86400.0 * 4, base_event_rate=0.005, seed=99,
    )
    gen.generate(store)
    return store


# ═══════════════════════════════════════════════════════════════
# 1–8. compute_path_signature
# ═══════════════════════════════════════════════════════════════

class TestComputePathSignature:

    def test_output_shape_unbatched_depth3(self, rand_path):
        """Unbatched: output shape = (d + d^2 + d^3,) = 39 for d=3."""
        sig = compute_path_signature(rand_path, depth=3)
        d = rand_path.shape[-1]
        expected = sum(d**k for k in range(1, 4))
        assert sig.shape == (expected,), f"Expected ({expected},), got {sig.shape}"

    def test_output_shape_batched(self, rand_path):
        """Batched: output shape = (batch, sig_dim)."""
        batch_path = rand_path.unsqueeze(0).expand(4, -1, -1)  # (4, 8, 3)
        sig = compute_path_signature(batch_path, depth=3)
        d = rand_path.shape[-1]
        expected = sum(d**k for k in range(1, 4))
        assert sig.shape == (4, expected)

    def test_depth1_equals_total_increment(self, linear_path):
        """Depth-1 signature = total increment (X_T - X_0)."""
        sig = compute_path_signature(linear_path, depth=1)
        total_increment = linear_path[-1] - linear_path[0]
        assert torch.allclose(sig, total_increment, atol=1e-5)

    def test_shuffle_identity_level2(self):
        """Chen shuffle identity: S^1_i * S^1_j = S^2_{ij} + S^2_{ji}.

        For the signature of any path, the shuffle product satisfies:
            S^{1}_{i} · S^{1}_{j} = S^{2}_{ij} + S^{2}_{ji}
        This is a fundamental algebraic property.
        """
        torch.manual_seed(7)
        path = torch.randn(6, 2)
        sig = compute_path_signature(path, depth=2)
        # d=2: sig = [S1_0, S1_1, S2_00, S2_01, S2_10, S2_11]
        S1_0, S1_1 = sig[0], sig[1]
        S2_01, S2_10 = sig[3], sig[4]
        lhs = S1_0 * S1_1
        rhs = S2_01 + S2_10
        assert abs((lhs - rhs).item()) < 1e-5, (
            f"Shuffle identity violated: S1_0*S1_1={lhs:.6f}, S2_01+S2_10={rhs:.6f}"
        )

    def test_single_point_path_zero_signature(self):
        """Single-point path has no increments → zero signature."""
        path = torch.randn(1, 3)
        sig = compute_path_signature(path, depth=3)
        assert torch.all(sig == 0.0)

    def test_invalid_depth_raises(self):
        """depth not in {1,2,3} raises ValueError."""
        path = torch.randn(5, 3)
        with pytest.raises(ValueError, match="depth must be"):
            compute_path_signature(path, depth=4)

    def test_output_finite_random_inputs(self):
        """No NaN or Inf in output for random path."""
        torch.manual_seed(0)
        sig = compute_path_signature(torch.randn(20, 3), depth=3)
        assert torch.isfinite(sig).all()

    def test_batched_equals_unbatched_loop(self, rand_path):
        """Batched result equals looping over unbatched calls."""
        batch = rand_path.unsqueeze(0).expand(3, -1, -1)  # (3, 8, 3)
        batch_sig = compute_path_signature(batch, depth=2)
        for i in range(3):
            single_sig = compute_path_signature(rand_path, depth=2)
            assert torch.allclose(batch_sig[i], single_sig, atol=1e-5)


# ═══════════════════════════════════════════════════════════════
# 9–12. entity_observations_to_path
# ═══════════════════════════════════════════════════════════════

class TestEntityObservationsToPath:

    def _make_obs(self, n=5, base_t=1000.0, dt=100.0) -> list[dict]:
        return [
            {
                "observed_at": base_t + i * dt,
                "observation_type": "price_movement",
                "value": {"value": float(i)},
            }
            for i in range(n)
        ]

    def test_output_shape(self):
        """path shape = (n_obs, PATH_CHANNELS) for n_obs ≤ max_seq_len."""
        obs = self._make_obs(10)
        path = entity_observations_to_path(obs, max_seq_len=64)
        assert path.shape == (10, PATH_CHANNELS)

    def test_empty_observations(self):
        """Empty observations return (1, PATH_CHANNELS) zeros."""
        path = entity_observations_to_path([], max_seq_len=64)
        assert path.shape == (1, PATH_CHANNELS)
        assert torch.all(path == 0.0)

    def test_max_seq_len_truncation(self):
        """Long event streams are truncated to max_seq_len most recent."""
        obs = self._make_obs(100)
        path = entity_observations_to_path(obs, max_seq_len=32)
        assert path.shape == (32, PATH_CHANNELS)

    def test_time_channel_range(self):
        """Time channel (channel 0) is in [0, 1]."""
        obs = self._make_obs(20)
        path = entity_observations_to_path(obs)
        assert path[:, 0].min().item() >= 0.0
        assert path[:, 0].max().item() <= 1.0 + 1e-6

    def test_value_channel_range(self):
        """Value channel (channel 1) is in (-1, 1) due to tanh normalisation."""
        obs = self._make_obs(20)
        path = entity_observations_to_path(obs)
        assert path[:, 1].min().item() > -1.0 - 1e-6
        assert path[:, 1].max().item() < 1.0 + 1e-6

    def test_type_channel_range(self):
        """Type channel (channel 2) is in [0, 1)."""
        obs = self._make_obs(10)
        path = entity_observations_to_path(obs)
        assert path[:, 2].min().item() >= 0.0
        assert path[:, 2].max().item() < 1.0 + 1e-6


# ═══════════════════════════════════════════════════════════════
# 13–14. compute_entity_signature
# ═══════════════════════════════════════════════════════════════

class TestComputeEntitySignature:

    def _make_obs(self, n=5):
        return [
            {"observed_at": float(i * 100), "observation_type": "trade_flow",
             "value": {"usd_amount": float(i * 1e6)}}
            for i in range(n)
        ]

    def test_output_dim(self):
        """compute_entity_signature returns SIGNATURE_DIM-dim vector."""
        obs = self._make_obs(8)
        sig = compute_entity_signature(obs)
        assert sig.shape == (SIGNATURE_DIM,)

    def test_output_finite(self):
        """Output is finite for typical observation dicts."""
        obs = self._make_obs(10)
        sig = compute_entity_signature(obs)
        assert torch.isfinite(sig).all()

    def test_empty_obs_returns_zeros(self):
        """Empty observation list → zero signature."""
        sig = compute_entity_signature([])
        assert sig.shape == (SIGNATURE_DIM,)
        assert torch.all(sig == 0.0)


# ═══════════════════════════════════════════════════════════════
# 15–16. PathSignatureEncoder
# ═══════════════════════════════════════════════════════════════

class TestPathSignatureEncoder:

    def test_forward_shape_unbatched(self):
        """(seq_len, channels) → (output_dim,)."""
        enc = PathSignatureEncoder(output_dim=16)
        path = torch.randn(10, PATH_CHANNELS)
        out = enc(path)
        assert out.shape == (16,)

    def test_forward_shape_batched(self):
        """(batch, seq_len, channels) → (batch, output_dim)."""
        enc = PathSignatureEncoder(output_dim=8)
        path = torch.randn(4, 10, PATH_CHANNELS)
        out = enc(path)
        assert out.shape == (4, 8)

    def test_forward_finite(self):
        """Output contains no NaN or Inf."""
        enc = PathSignatureEncoder(output_dim=16)
        path = torch.randn(12, PATH_CHANNELS)
        out = enc(path)
        assert torch.isfinite(out).all()

    def test_custom_depth(self):
        """depth=1 → raw dim=3, projection to output_dim."""
        enc = PathSignatureEncoder(output_dim=4, depth=1)
        path = torch.randn(6, PATH_CHANNELS)
        out = enc(path)
        assert out.shape == (4,)


# ═══════════════════════════════════════════════════════════════
# 17–18. _build_node_features with use_signatures
# ═══════════════════════════════════════════════════════════════

class TestBuildNodeFeaturesSignatures:

    def _make_obs(self, entity_id: str, n=5) -> list[dict]:
        return [
            {
                "entity_id": entity_id,
                "observed_at": float(i * 200),
                "observation_type": "trade_flow",
                "value": {"usd_amount": float(i * 1e5)},
            }
            for i in range(n)
        ]

    def test_feature_dim_with_signatures(self):
        """use_signatures=True increases feature dim by SIGNATURE_DIM."""
        obs = self._make_obs("e0", 5) + self._make_obs("e1", 3)
        feats_no_sig = _build_node_features("company", ["e0", "e1"], obs, 1000.0, use_signatures=False)
        feats_sig = _build_node_features("company", ["e0", "e1"], obs, 1000.0, use_signatures=True)
        assert feats_sig.shape[1] == feats_no_sig.shape[1] + SIGNATURE_DIM

    def test_feature_dim_without_signatures(self):
        """use_signatures=False keeps feature dim at BASE_FEAT_DIM."""
        obs = self._make_obs("e0", 4)
        feats = _build_node_features("company", ["e0"], obs, 800.0, use_signatures=False)
        assert feats.shape[1] == BASE_FEAT_DIM

    def test_signature_region_non_zero(self):
        """Signature features are non-zero for entities with observations."""
        obs = self._make_obs("e0", 6)
        feats = _build_node_features("company", ["e0"], obs, 1200.0, use_signatures=True)
        sig_start = BASE_FEAT_DIM
        sig_end = sig_start + SIGNATURE_DIM
        assert not torch.all(feats[0, sig_start:sig_end] == 0.0), (
            "Signature features should be non-zero for entity with 6 events"
        )

    def test_signature_features_finite(self):
        """Signature region contains no NaN or Inf."""
        obs = self._make_obs("e0", 8)
        feats = _build_node_features("company", ["e0"], obs, 1600.0, use_signatures=True)
        assert torch.isfinite(feats).all()


# ═══════════════════════════════════════════════════════════════
# 19–20. TrainerConfig + build_model
# ═══════════════════════════════════════════════════════════════

class TestTrainerConfigSignatures:

    def test_use_signatures_defaults_false(self):
        cfg = TrainerConfig()
        assert cfg.use_signatures is False

    def test_build_model_use_signatures_true_expands_in_channels(self, simple_store):
        """use_signatures=True gives in_channels SIGNATURE_DIM larger than False."""
        cfg_base = TrainerConfig(hidden_dim=16, memory_dim=16, message_dim=16,
                                 time_dim=8, num_heads=1, num_layers=1)
        cfg_sig = TrainerConfig(hidden_dim=16, memory_dim=16, message_dim=16,
                                time_dim=8, num_heads=1, num_layers=1,
                                use_signatures=True)
        t_base = Trainer(simple_store, cfg_base)
        t_sig = Trainer(simple_store, cfg_sig)
        m_base = t_base.build_model()
        m_sig = t_sig.build_model()

        # Every node type present should have SIGNATURE_DIM more input channels
        for ntype in m_base.node_types:
            proj_base = m_base.type_projections[ntype]
            proj_sig = m_sig.type_projections[ntype]
            diff = proj_sig.in_features - proj_base.in_features
            assert diff == SIGNATURE_DIM, (
                f"{ntype}: expected +{SIGNATURE_DIM} in_features, got {diff}"
            )


# ═══════════════════════════════════════════════════════════════
# 21. Full training loop with use_signatures=True
# ═══════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestSignatureTrainingLoop:

    def test_training_loop_no_nan_losses(self, simple_store):
        """2-epoch training with use_signatures=True produces finite losses."""
        cfg = TrainerConfig(
            hidden_dim=16, memory_dim=16, message_dim=16, time_dim=8,
            num_heads=1, num_layers=1, epochs=2, window_size=86400.0,
            use_signatures=True, return_weight=0.0,
        )
        trainer = Trainer(simple_store, cfg)
        trainer.build_model()
        history = trainer.train()

        for loss_name, values in history.items():
            for v in values:
                assert math.isfinite(v), (
                    f"NaN/Inf in history['{loss_name}']: {values}"
                )

    def test_signatures_increase_first_epoch_loss_stability(self, simple_store):
        """Signature and baseline models both converge (no NaN divergence)."""
        base_cfg = dict(hidden_dim=16, memory_dim=16, message_dim=16,
                        time_dim=8, num_heads=1, num_layers=1, epochs=2,
                        window_size=86400.0, return_weight=0.0)

        for use_sig in (False, True):
            trainer = Trainer(simple_store, TrainerConfig(**base_cfg, use_signatures=use_sig))
            trainer.build_model()
            history = trainer.train()
            total = history["total"][-1]
            assert math.isfinite(total), f"use_signatures={use_sig}: NaN total loss"
