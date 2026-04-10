"""Tests for Phase 12b: Time2Vec + TemporalEncoder.

Covers:
    - Time2Vec output shape, gradient flow, periodic behavior, edge cases
    - TemporalEncoder: varying history, empty, single obs, truncation
    - Batch encoding
"""

from __future__ import annotations

import math

import pytest
import torch

from agent.models.gnn.graph_builder import OBSERVATION_TYPES
from agent.models.gnn.temporal import TemporalEncoder, Time2Vec


# ═══════════════════════════════════════════════════════════════
# Time2Vec tests
# ═══════════════════════════════════════════════════════════════


class TestTime2Vec:
    def test_output_shape_scalar(self):
        t2v = Time2Vec(out_features=16)
        out = t2v(torch.tensor(1000.0))
        assert out.shape == (1, 16)

    def test_output_shape_1d(self):
        t2v = Time2Vec(out_features=8)
        out = t2v(torch.tensor([100.0, 200.0, 300.0]))
        assert out.shape == (3, 8)

    def test_output_shape_2d(self):
        t2v = Time2Vec(out_features=8)
        out = t2v(torch.tensor([[100.0], [200.0]]))
        assert out.shape == (2, 8)

    def test_output_shape_single_feature(self):
        """out_features=1 means only the linear component."""
        t2v = Time2Vec(out_features=1)
        out = t2v(torch.tensor([1.0, 2.0]))
        assert out.shape == (2, 1)

    def test_invalid_out_features(self):
        with pytest.raises(ValueError, match="out_features must be >= 1"):
            Time2Vec(out_features=0)

    def test_gradient_flow(self):
        t2v = Time2Vec(out_features=8)
        t = torch.tensor([100.0], requires_grad=True)
        out = t2v(t)
        loss = out.sum()
        loss.backward()
        # Gradient should flow to all parameters
        assert t2v.omega.grad is not None
        assert t2v.phi.grad is not None
        assert t2v.w_linear.grad is not None
        assert t2v.b_linear.grad is not None

    def test_zero_timestamp(self):
        t2v = Time2Vec(out_features=8)
        out = t2v(torch.tensor(0.0))
        assert out.shape == (1, 8)
        assert torch.isfinite(out).all()

    def test_negative_timestamp(self):
        t2v = Time2Vec(out_features=8)
        out = t2v(torch.tensor(-1000.0))
        assert out.shape == (1, 8)
        assert torch.isfinite(out).all()

    def test_very_large_timestamp(self):
        t2v = Time2Vec(out_features=8)
        out = t2v(torch.tensor(1e10))  # ~317 years in seconds
        assert out.shape == (1, 8)
        assert torch.isfinite(out).all()

    def test_periodic_components_bounded(self):
        """Periodic components (sin) should be in [-1, 1]."""
        t2v = Time2Vec(out_features=16)
        t = torch.linspace(0, 1e6, 100)
        out = t2v(t)
        # Columns 1: are periodic (sin), should be in [-1, 1]
        periodic = out[:, 1:]
        assert periodic.min().item() >= -1.0 - 1e-6
        assert periodic.max().item() <= 1.0 + 1e-6

    def test_different_timestamps_different_outputs(self):
        t2v = Time2Vec(out_features=8)
        out1 = t2v(torch.tensor([100.0]))
        out2 = t2v(torch.tensor([200.0]))
        assert not torch.allclose(out1, out2)

    def test_batch_consistency(self):
        """Batch of timestamps gives same result as individual calls."""
        t2v = Time2Vec(out_features=8)
        t2v.eval()
        t1 = torch.tensor([100.0])
        t2 = torch.tensor([200.0])
        batch = torch.tensor([100.0, 200.0])
        with torch.no_grad():
            out_batch = t2v(batch)
            out1 = t2v(t1)
            out2 = t2v(t2)
        assert torch.allclose(out_batch[0], out1.squeeze(0))
        assert torch.allclose(out_batch[1], out2.squeeze(0))

    def test_very_close_timestamps(self):
        """Timestamps differing by microseconds should produce similar outputs."""
        t2v = Time2Vec(out_features=8)
        t2v.eval()
        with torch.no_grad():
            out1 = t2v(torch.tensor([1000.0]))
            out2 = t2v(torch.tensor([1000.000001]))
        # Should be very close but not identical
        assert torch.allclose(out1, out2, atol=1e-3)


# ═══════════════════════════════════════════════════════════════
# TemporalEncoder tests
# ═══════════════════════════════════════════════════════════════


class TestTemporalEncoder:
    def test_output_dim(self):
        enc = TemporalEncoder(time_dim=16)
        expected = len(OBSERVATION_TYPES) + 4 + 16
        assert enc.output_dim == expected

    def test_empty_history(self):
        enc = TemporalEncoder(time_dim=8)
        out = enc.forward([], current_time=5000.0)
        assert out.shape == (enc.output_dim,)
        assert torch.isfinite(out).all()
        # All type counts should be 0
        assert out[: len(OBSERVATION_TYPES)].sum().item() == 0.0

    def test_single_observation(self):
        enc = TemporalEncoder(time_dim=8)
        obs = [{"observed_at": 1000.0, "observation_type": "btc_transfer"}]
        out = enc.forward(obs, current_time=2000.0)
        assert out.shape == (enc.output_dim,)
        # btc_transfer count should be 1
        idx = OBSERVATION_TYPES.index("btc_transfer")
        assert out[idx].item() == 1.0
        # Inter-event stats should be zero (only 1 obs)
        num_types = len(OBSERVATION_TYPES)
        assert out[num_types : num_types + 4].sum().item() == 0.0

    def test_multiple_observations(self):
        enc = TemporalEncoder(time_dim=8)
        obs = [
            {"observed_at": 1000.0, "observation_type": "insider_trade"},
            {"observed_at": 2000.0, "observation_type": "insider_trade"},
            {"observed_at": 3000.0, "observation_type": "geopolitical_event"},
        ]
        out = enc.forward(obs, current_time=4000.0)
        # insider_trade count = 2
        idx_it = OBSERVATION_TYPES.index("insider_trade")
        assert out[idx_it].item() == 2.0
        # geopolitical_event count = 1
        idx_ge = OBSERVATION_TYPES.index("geopolitical_event")
        assert out[idx_ge].item() == 1.0

    def test_inter_event_stats(self):
        enc = TemporalEncoder(time_dim=8)
        obs = [
            {"observed_at": 100.0, "observation_type": "insider_trade"},
            {"observed_at": 200.0, "observation_type": "insider_trade"},
            {"observed_at": 400.0, "observation_type": "insider_trade"},
        ]
        out = enc.forward(obs, current_time=500.0)
        num_types = len(OBSERVATION_TYPES)
        mean_dt = out[num_types].item()
        min_dt = out[num_types + 2].item()
        max_dt = out[num_types + 3].item()
        # Deltas: [100, 200], mean = 150
        assert mean_dt == pytest.approx(150.0)
        assert min_dt == pytest.approx(100.0)
        assert max_dt == pytest.approx(200.0)

    def test_max_history_truncation(self):
        enc = TemporalEncoder(time_dim=8, max_history=3)
        # Create 10 observations — should keep last 3
        obs = [
            {"observed_at": float(i), "observation_type": "port_call"}
            for i in range(10)
        ]
        out = enc.forward(obs, current_time=10.0)
        # Count should be 3 (truncated)
        idx = OBSERVATION_TYPES.index("port_call")
        assert out[idx].item() == 3.0

    def test_unsorted_input(self):
        """Input doesn't need to be sorted — encoder sorts internally."""
        enc = TemporalEncoder(time_dim=8)
        obs = [
            {"observed_at": 3000.0, "observation_type": "insider_trade"},
            {"observed_at": 1000.0, "observation_type": "insider_trade"},
            {"observed_at": 2000.0, "observation_type": "insider_trade"},
        ]
        out = enc.forward(obs, current_time=4000.0)
        num_types = len(OBSERVATION_TYPES)
        # Δt from last obs (3000) → current (4000) = 1000
        # The time2vec portion should encode this
        assert out.shape == (enc.output_dim,)
        assert torch.isfinite(out).all()

    def test_unknown_obs_type_ignored(self):
        enc = TemporalEncoder(time_dim=8)
        obs = [
            {"observed_at": 1000.0, "observation_type": "alien_signal"},
        ]
        out = enc.forward(obs, current_time=2000.0)
        # All type counts should be 0
        assert out[: len(OBSERVATION_TYPES)].sum().item() == 0.0

    def test_missing_observed_at(self):
        enc = TemporalEncoder(time_dim=8)
        obs = [{"observation_type": "insider_trade"}]  # no observed_at
        out = enc.forward(obs, current_time=2000.0)
        assert out.shape == (enc.output_dim,)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self):
        enc = TemporalEncoder(time_dim=8)
        obs = [
            {"observed_at": 100.0, "observation_type": "btc_transfer"},
            {"observed_at": 200.0, "observation_type": "btc_transfer"},
        ]
        out = enc.forward(obs, current_time=300.0)
        loss = out.sum()
        loss.backward()
        # Time2Vec params should have gradients
        assert enc.time2vec.omega.grad is not None

    def test_custom_obs_types(self):
        enc = TemporalEncoder(time_dim=4, obs_types=["alpha", "beta"])
        assert enc.output_dim == 2 + 4 + 4  # 2 types + 4 inter-event + 4 time
        obs = [
            {"observed_at": 100.0, "observation_type": "alpha"},
            {"observed_at": 200.0, "observation_type": "beta"},
        ]
        out = enc.forward(obs, current_time=300.0)
        assert out[0].item() == 1.0  # alpha count
        assert out[1].item() == 1.0  # beta count


# ═══════════════════════════════════════════════════════════════
# TemporalEncoder.encode_batch tests
# ═══════════════════════════════════════════════════════════════


class TestTemporalEncoderBatch:
    def test_empty_batch(self):
        enc = TemporalEncoder(time_dim=8)
        result = enc.encode_batch({}, [], current_time=1000.0)
        assert result.shape == (0, enc.output_dim)

    def test_batch_shape(self):
        enc = TemporalEncoder(time_dim=8)
        obs_by_entity = {
            "e1": [{"observed_at": 100.0, "observation_type": "btc_transfer"}],
            "e2": [{"observed_at": 200.0, "observation_type": "port_call"}],
        }
        result = enc.encode_batch(obs_by_entity, ["e1", "e2"], current_time=300.0)
        assert result.shape == (2, enc.output_dim)

    def test_missing_entity_gets_zeros(self):
        enc = TemporalEncoder(time_dim=8)
        obs_by_entity = {
            "e1": [{"observed_at": 100.0, "observation_type": "btc_transfer"}],
        }
        result = enc.encode_batch(obs_by_entity, ["e1", "e2"], current_time=300.0)
        assert result.shape == (2, enc.output_dim)
        # e2 has no obs — type counts should be 0
        assert result[1, : len(OBSERVATION_TYPES)].sum().item() == 0.0

    def test_batch_matches_individual(self):
        enc = TemporalEncoder(time_dim=8)
        enc.eval()
        obs_by_entity = {
            "e1": [{"observed_at": 100.0, "observation_type": "btc_transfer"}],
            "e2": [{"observed_at": 200.0, "observation_type": "port_call"}],
        }
        with torch.no_grad():
            batch = enc.encode_batch(obs_by_entity, ["e1", "e2"], current_time=300.0)
            ind1 = enc.forward(obs_by_entity["e1"], current_time=300.0)
            ind2 = enc.forward(obs_by_entity["e2"], current_time=300.0)
        assert torch.allclose(batch[0], ind1)
        assert torch.allclose(batch[1], ind2)

    def test_all_same_type(self):
        """All observations of same type — no inter-type diversity."""
        enc = TemporalEncoder(time_dim=4)
        obs = [
            {"observed_at": float(i * 100), "observation_type": "port_call"}
            for i in range(5)
        ]
        result = enc.encode_batch({"e1": obs}, ["e1"], current_time=500.0)
        assert result.shape == (1, enc.output_dim)
        idx = OBSERVATION_TYPES.index("port_call")
        assert result[0, idx].item() == 5.0
