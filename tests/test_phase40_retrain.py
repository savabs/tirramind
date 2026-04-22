"""Tests for Phase 40: Real Data Model Refresh — retrain script edge cases.

Covers:
    Empty store → graceful error
    Single observation type → trains without crash
    No entity links → contrastive loss = 0, other losses train normally
    Epochs=0 → returns empty history
    Model save/load round-trip → weights match
    Evaluation on empty test split → returns zero metrics gracefully
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest
import torch

from agent.models.gnn.graph_builder import ENTITY_TYPES, OBSERVATION_TYPES
from agent.models.gnn.trainer import (
    Trainer,
    TrainerConfig,
    evaluate,
)
from agent.pipeline.store import PipelineStore


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def empty_store():
    """PipelineStore with no data."""
    return PipelineStore(db_path=":memory:")


@pytest.fixture
def single_type_store():
    """Store with only instrument entities and instrument_daily observations."""
    store = PipelineStore(db_path=":memory:")
    t0 = 1_700_000_000.0
    # Create 5 instruments
    for i in range(5):
        store.register_entity(
            entity_type="instrument",
            canonical_name=f"Instrument {i}",
            entity_id=f"inst_{i}",
        )
    # Create daily observations spanning 30 days
    for day in range(30):
        for i in range(5):
            store.store_entity_observation(
                entity_id=f"inst_{i}",
                source_tool="test",
                observed_at=t0 + day * 86400 + i * 100,
                observation_type="instrument_daily",
                value=100.0 + day * 0.5 + i,
            )
    return store


@pytest.fixture
def no_links_store():
    """Store with entities and observations but no entity links."""
    store = PipelineStore(db_path=":memory:")
    t0 = 1_700_000_000.0
    for i in range(3):
        store.register_entity(
            entity_type="company",
            canonical_name=f"Company {i}",
            entity_id=f"co_{i}",
        )
        store.register_entity(
            entity_type="instrument",
            canonical_name=f"Instrument {i}",
            entity_id=f"inst_{i}",
        )
    for day in range(20):
        for i in range(3):
            store.store_entity_observation(
                entity_id=f"inst_{i}",
                source_tool="test",
                observed_at=t0 + day * 86400 + i * 100,
                observation_type="instrument_daily",
                value=50.0 + day,
            )
            store.store_entity_observation(
                entity_id=f"co_{i}",
                source_tool="test",
                observed_at=t0 + day * 86400 + 50000 + i * 100,
                observation_type="earnings_release",
                value=1.0,
            )
    # No links added — contrastive loss should be 0
    return store


@pytest.fixture
def linked_store():
    """Store with entities, observations, and links for round-trip testing."""
    store = PipelineStore(db_path=":memory:")
    t0 = 1_700_000_000.0
    for i in range(4):
        store.register_entity(
            entity_type="instrument",
            canonical_name=f"Instrument {i}",
            entity_id=f"inst_{i}",
        )
    store.register_entity(
        entity_type="country",
        canonical_name="United States",
        entity_id="us",
    )
    # Link instruments to country
    for i in range(4):
        store.link_entities(
            entity_id_a=f"inst_{i}",
            entity_id_b="us",
            link_type="located_in",
            source="test",
            confidence=0.9,
        )
    # Observations spanning 30 days
    for day in range(30):
        for i in range(4):
            store.store_entity_observation(
                entity_id=f"inst_{i}",
                source_tool="test",
                observed_at=t0 + day * 86400 + i * 100,
                observation_type="instrument_daily",
                value=100.0 + day,
            )
        store.store_entity_observation(
            entity_id="us",
            source_tool="test",
            observed_at=t0 + day * 86400 + 50000,
            observation_type="geopolitical_event",
            value=0.5,
        )
    return store


# ═══════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════


class TestRetrainEdgeCases:
    """Edge case tests for the retrain workflow."""

    def test_empty_store_no_crash(self, empty_store):
        """Empty store should not crash build_model, but train() has no data."""
        config = TrainerConfig(epochs=1, hidden_dim=16, memory_dim=16, message_dim=16)
        trainer = Trainer(empty_store, config)
        trainer.build_model()
        history = trainer.train()
        # No windows → no training → empty loss lists
        assert history["total"] == [] or all(v == 0.0 for v in history["total"])

    def test_single_obs_type_trains(self, single_type_store):
        """Training on a single observation type should work fine."""
        config = TrainerConfig(epochs=2, hidden_dim=16, memory_dim=16, message_dim=16)
        trainer = Trainer(single_type_store, config)
        trainer.build_model()
        history = trainer.train()
        assert len(history["total"]) == 2
        # obs_type loss should be nonzero (learning to predict instrument_daily)
        assert any(v > 0 for v in history["obs_type"])

    def test_no_links_contrastive_zero(self, no_links_store):
        """Without entity links, contrastive loss should be zero."""
        config = TrainerConfig(epochs=2, hidden_dim=16, memory_dim=16, message_dim=16)
        trainer = Trainer(no_links_store, config)
        trainer.build_model()
        history = trainer.train()
        assert len(history["total"]) == 2
        # Contrastive loss should be 0 with no links
        assert all(v == 0.0 for v in history["contrastive"])

    def test_zero_epochs_empty_history(self, linked_store):
        """epochs=0 should return empty history without crashing."""
        config = TrainerConfig(epochs=0, hidden_dim=16, memory_dim=16, message_dim=16)
        trainer = Trainer(linked_store, config)
        trainer.build_model()
        history = trainer.train()
        for key in ("total", "obs_type", "time_delta", "contrastive", "value"):
            assert history[key] == []

    def test_model_save_load_roundtrip(self, linked_store):
        """Saved model should load with identical weights."""
        config = TrainerConfig(epochs=1, hidden_dim=16, memory_dim=16, message_dim=16)
        trainer = Trainer(linked_store, config)
        trainer.build_model()
        trainer.train()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            trainer.save_model(path)

            # Load into a new Trainer
            loaded_trainer = Trainer.load_model(path, linked_store)
            loaded_model = loaded_trainer.model

            # Compare parameters
            orig_params = dict(trainer.model.named_parameters())
            loaded_params = dict(loaded_model.named_parameters())
            assert set(orig_params.keys()) == set(loaded_params.keys())
            for name in orig_params:
                torch.testing.assert_close(
                    orig_params[name],
                    loaded_params[name],
                    msg=f"Parameter {name} differs after save/load",
                )
        finally:
            Path(path).unlink(missing_ok=True)

    def test_evaluate_empty_test_split(self, empty_store):
        """Evaluation on a store with no data returns zero metrics."""
        config = TrainerConfig(epochs=0, hidden_dim=16, memory_dim=16, message_dim=16)
        trainer = Trainer(empty_store, config)
        trainer.build_model()

        metrics = evaluate(trainer.model, empty_store, config, split="test")
        assert metrics["obs_type_acc_top1"] == 0.0
        assert metrics["obs_type_acc_top5"] == 0.0
        assert metrics["time_delta_mae"] == 0.0
        assert metrics["num_predictions"] == 0

    def test_evaluate_val_split(self, linked_store):
        """Evaluation on val split should return valid metrics."""
        config = TrainerConfig(epochs=1, hidden_dim=16, memory_dim=16, message_dim=16)
        trainer = Trainer(linked_store, config)
        trainer.build_model()
        trainer.train()

        metrics = evaluate(trainer.model, linked_store, config, split="val")
        assert 0.0 <= metrics["obs_type_acc_top1"] <= 1.0
        assert 0.0 <= metrics["obs_type_acc_top5"] <= 1.0
        assert metrics["time_delta_mae"] >= 0.0

    def test_auto_tune_loss_weights(self, linked_store):
        """auto_tune_loss_weights should produce learned weights."""
        config = TrainerConfig(
            epochs=2,
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            auto_tune_loss_weights=True,
        )
        trainer = Trainer(linked_store, config)
        trainer.build_model()
        trainer.train()

        weights = trainer.effective_loss_weights()
        assert "obs_type" in weights
        assert "time_delta" in weights
        assert "contrastive" in weights
        assert "value" in weights
        # Learned weights should be positive
        for k, v in weights.items():
            assert v > 0, f"Learned weight for {k} should be positive, got {v}"
