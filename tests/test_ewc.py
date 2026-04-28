"""Phase 46: EWC (Elastic Weight Consolidation) test suite.

Tests ewc.py (EWCState, compute_fisher, ewc_penalty) and Trainer EWC integration
(steps 46.1–46.6: Fisher computation, save/load round-trip, online_update).

Reference: Kirkpatrick et al. 2017, arXiv:1612.00796.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from agent.models.gnn.ewc import EWCState, compute_fisher, ewc_penalty
from agent.models.gnn.trainer import (
    InjectedPattern,
    SyntheticGraphGenerator,
    Trainer,
    TrainerConfig,
)
from agent.pipeline.store import PipelineStore


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture()
def store_with_data(tmp_path: Path) -> PipelineStore:
    """PipelineStore populated with 7 days of synthetic observations."""
    store = PipelineStore(str(tmp_path / "test.db"))
    gen = SyntheticGraphGenerator(
        num_companies=4,
        num_countries=2,
        num_vessels=2,
        num_wallets=3,
        time_span=86400.0 * 7,
        base_event_rate=0.005,
        seed=42,
    )
    gen.generate(store)
    return store


@pytest.fixture()
def mini_cfg() -> TrainerConfig:
    """Minimal TrainerConfig for fast tests."""
    return TrainerConfig(
        hidden_dim=16,
        memory_dim=16,
        message_dim=16,
        time_dim=8,
        num_heads=1,
        num_layers=1,
        epochs=1,
        window_size=86400.0,
        ewc_lambda=500.0,
        online_batch_threshold=5,
    )


@pytest.fixture()
def trained_trainer(store_with_data: PipelineStore, mini_cfg: TrainerConfig) -> Trainer:
    """A Trainer that has been built and trained (EWC state populated)."""
    trainer = Trainer(store_with_data, mini_cfg)
    trainer.build_model()
    trainer.train()
    return trainer


# ── Helper ────────────────────────────────────────────────────


def _tiny_linear() -> nn.Linear:
    """Two-param linear layer for isolated ewc.py tests."""
    m = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        m.weight.fill_(1.0)
    return m


# ═══════════════════════════════════════════════════════════════
# 1. EWCState dataclass
# ═══════════════════════════════════════════════════════════════


def test_ewc_state_creation() -> None:
    """EWCState stores fisher/anchor/lambda and defaults for timestamps."""
    fisher = {"w": torch.ones(3)}
    anchor = {"w": torch.zeros(3)}
    state = EWCState(fisher=fisher, anchor=anchor, lambda_=1000.0)

    assert state.fisher is fisher
    assert state.anchor is anchor
    assert state.lambda_ == 1000.0
    assert state.last_update_ts <= time.time()
    assert state.obs_count_at_update == 0


# ═══════════════════════════════════════════════════════════════
# 2. compute_fisher
# ═══════════════════════════════════════════════════════════════


def test_compute_fisher_returns_dict() -> None:
    """compute_fisher returns a dict with one non-negative tensor per named param."""
    model = _tiny_linear()
    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))

    def loss_fn() -> torch.Tensor:
        return nn.functional.cross_entropy(model(x), y)

    diag = compute_fisher(model, loss_fn, n_samples=1)

    assert isinstance(diag, dict)
    assert len(diag) == len(list(model.named_parameters()))
    for name, val in diag.items():
        assert isinstance(val, torch.Tensor)
        assert (val >= 0).all(), f"Fisher diagonal has negative value for '{name}'"


def test_compute_fisher_invalid_n_samples_raises() -> None:
    """compute_fisher raises ValueError for n_samples < 1."""
    model = _tiny_linear()

    with pytest.raises(ValueError, match="n_samples"):
        compute_fisher(model, lambda: model(torch.randn(2, 4)).sum(), n_samples=0)


def test_compute_fisher_non_scalar_loss_raises() -> None:
    """compute_fisher raises RuntimeError when the loss closure returns a non-scalar."""
    model = _tiny_linear()

    with pytest.raises(RuntimeError, match="scalar"):
        compute_fisher(model, lambda: model(torch.randn(2, 4)), n_samples=1)


# ═══════════════════════════════════════════════════════════════
# 3. ewc_penalty
# ═══════════════════════════════════════════════════════════════


def test_ewc_penalty_zero_when_no_drift() -> None:
    """ewc_penalty is 0 when current weights exactly equal anchor weights."""
    model = _tiny_linear()
    state = EWCState(
        fisher={"weight": torch.ones_like(model.weight.data)},
        anchor={"weight": model.weight.data.clone()},
        lambda_=1000.0,
    )

    penalty = ewc_penalty(model, state)

    assert penalty.item() == pytest.approx(0.0, abs=1e-6)


def test_ewc_penalty_positive_when_drift() -> None:
    """ewc_penalty is > 0 when weights have drifted from anchor."""
    model = _tiny_linear()
    anchor_val = model.weight.data.clone()

    # Drift the model weights
    with torch.no_grad():
        model.weight.add_(1.0)

    state = EWCState(
        fisher={"weight": torch.ones_like(anchor_val)},
        anchor={"weight": anchor_val},
        lambda_=1000.0,
    )

    penalty = ewc_penalty(model, state)

    assert penalty.item() > 0.0


def test_ewc_penalty_shape_mismatch_returns_zero() -> None:
    """ewc_penalty returns 0 and does not raise when anchor shape mismatches model."""
    model = nn.Linear(4, 3, bias=False)
    state = EWCState(
        fisher={"weight": torch.ones(2, 4)},  # wrong shape: (2,4) vs (3,4)
        anchor={"weight": torch.zeros(2, 4)},
        lambda_=1000.0,
    )

    penalty = ewc_penalty(model, state)

    assert penalty.item() == pytest.approx(0.0, abs=1e-6)


# ═══════════════════════════════════════════════════════════════
# 4. Trainer.train() populates EWC state
# ═══════════════════════════════════════════════════════════════


def test_trainer_has_ewc_state_after_train(trained_trainer: Trainer) -> None:
    """After train(), _ewc_state is populated with fisher, anchor, and correct lambda."""
    state = trained_trainer._ewc_state

    assert state is not None, "_ewc_state should not be None after train()"
    assert len(state.fisher) > 0
    assert len(state.anchor) == len(state.fisher)
    assert state.lambda_ == pytest.approx(500.0)
    # Fisher diagonals must be non-negative
    for name, f in state.fisher.items():
        assert (f >= 0).all(), f"Negative Fisher value for param '{name}'"


# ═══════════════════════════════════════════════════════════════
# 5. save_model / load_model round-trips
# ═══════════════════════════════════════════════════════════════


def test_save_load_roundtrip_with_ewc(
    trained_trainer: Trainer,
    store_with_data: PipelineStore,
    tmp_path: Path,
) -> None:
    """EWC state survives save_model → load_model with tensor-exact Fisher values."""
    model_path = str(tmp_path / "model.pt")
    trained_trainer.save_model(model_path)

    loaded = Trainer.load_model(model_path, store_with_data)

    assert loaded._ewc_state is not None
    orig = trained_trainer._ewc_state
    assert set(loaded._ewc_state.fisher.keys()) == set(orig.fisher.keys())
    assert loaded._ewc_state.lambda_ == pytest.approx(orig.lambda_)
    for k in orig.fisher:
        diff = (orig.fisher[k] - loaded._ewc_state.fisher[k]).abs().max()
        assert diff.item() < 1e-5, f"Fisher tensor mismatch for '{k}': max diff {diff}"


def test_save_load_roundtrip_without_ewc(
    store_with_data: PipelineStore,
    mini_cfg: TrainerConfig,
    tmp_path: Path,
) -> None:
    """A checkpoint saved without EWC state loads with _ewc_state = None (backward compat)."""
    model_path = str(tmp_path / "model_no_ewc.pt")

    # Build and save a model without triggering EWC (0 epochs → no fisher)
    cfg_no_ewc = TrainerConfig(
        hidden_dim=16,
        memory_dim=16,
        message_dim=16,
        time_dim=8,
        num_heads=1,
        num_layers=1,
        epochs=0,
    )
    trainer = Trainer(store_with_data, cfg_no_ewc)
    trainer.build_model()
    trainer.save_model(model_path)

    # Manually remove EWC keys to simulate pre-Phase-46 checkpoint
    import torch as _torch

    ckpt = _torch.load(model_path, weights_only=False)
    for k in (
        "ewc_fisher",
        "ewc_anchor",
        "ewc_lambda",
        "ewc_last_update_ts",
        "ewc_obs_count_at_update",
    ):
        ckpt.pop(k, None)
    _torch.save(ckpt, model_path)

    loaded = Trainer.load_model(model_path, store_with_data)

    assert (
        loaded._ewc_state is None
    ), "Pre-Phase-46 checkpoint must load with _ewc_state = None"


# ═══════════════════════════════════════════════════════════════
# 6. online_update
# ═══════════════════════════════════════════════════════════════


def test_online_update_returns_correct_keys(
    trained_trainer: Trainer,
    store_with_data: PipelineStore,
) -> None:
    """online_update returns dict with loss_new, loss_ewc, loss_total, n_events."""
    all_obs = store_with_data.query_all_observations()
    all_obs.sort(key=lambda o: o.get("observed_at", 0.0))
    batch = all_obs[-10:]

    result = trained_trainer.online_update(batch)

    assert set(result.keys()) == {"loss_new", "loss_ewc", "loss_total", "n_events"}
    assert result["n_events"] == pytest.approx(10.0)


def test_online_update_loss_non_negative(
    trained_trainer: Trainer,
    store_with_data: PipelineStore,
) -> None:
    """online_update: all loss values are non-negative and total = new + ewc."""
    all_obs = store_with_data.query_all_observations()
    all_obs.sort(key=lambda o: o.get("observed_at", 0.0))
    batch = all_obs[-10:]

    result = trained_trainer.online_update(batch)

    assert result["loss_new"] >= 0.0
    assert result["loss_ewc"] >= 0.0
    assert result["loss_total"] == pytest.approx(
        result["loss_new"] + result["loss_ewc"], abs=1e-4
    )


def test_online_update_updates_bookkeeping(
    trained_trainer: Trainer,
    store_with_data: PipelineStore,
) -> None:
    """online_update increments obs_count_at_update and refreshes last_update_ts."""
    all_obs = store_with_data.query_all_observations()
    batch = all_obs[-10:]
    before_ts = trained_trainer._ewc_state.last_update_ts

    trained_trainer.online_update(batch)

    assert trained_trainer._ewc_state.obs_count_at_update > 0
    assert trained_trainer._ewc_state.last_update_ts >= before_ts


# ── Guards ────────────────────────────────────────────────────


def test_online_update_no_model_raises(
    store_with_data: PipelineStore,
    mini_cfg: TrainerConfig,
) -> None:
    """online_update raises RuntimeError when model has not been built/trained."""
    trainer = Trainer(store_with_data, mini_cfg)  # no build_model / train

    with pytest.raises(RuntimeError, match="train()"):
        trainer.online_update([{"entity_id": "x"}])


def test_online_update_no_ewc_state_raises(
    trained_trainer: Trainer,
    store_with_data: PipelineStore,
) -> None:
    """online_update raises RuntimeError when EWC state is None (Fisher not computed)."""
    trained_trainer._ewc_state = None

    with pytest.raises(RuntimeError):
        trained_trainer.online_update([{"entity_id": "x"}])


def test_online_update_empty_events_raises(trained_trainer: Trainer) -> None:
    """online_update raises RuntimeError when given an empty event list."""
    with pytest.raises(RuntimeError, match="non-empty"):
        trained_trainer.online_update([])
