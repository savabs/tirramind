"""Phase 19a: GNN real-data training tests.

Tests Trainer.infer(), save_model(), load_model() and smoke tests
self-supervised training on synthetic (but structurally real) data.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

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
    """PipelineStore populated with synthetic entity data (5 types, links, obs)."""
    store = PipelineStore(str(tmp_path / "test.db"))
    gen = SyntheticGraphGenerator(
        num_companies=4,
        num_countries=2,
        num_vessels=2,
        num_wallets=3,
        time_span=86400.0 * 7,  # 7 days
        base_event_rate=0.005,
        seed=123,
        patterns=[
            InjectedPattern(
                source_type="company",
                source_obs_type="insider_trade",
                target_type="country",
                target_obs_type="geopolitical_event",
                via_edge="headquartered_in",
                lag_seconds=3600.0,
                lag_jitter=300.0,
            ),
        ],
    )
    gen.generate(store)
    return store


@pytest.fixture()
def empty_store(tmp_path: Path) -> PipelineStore:
    """Completely empty PipelineStore."""
    return PipelineStore(str(tmp_path / "empty.db"))


@pytest.fixture()
def small_config() -> TrainerConfig:
    """Minimal config for fast tests."""
    return TrainerConfig(
        hidden_dim=16,
        memory_dim=16,
        message_dim=16,
        time_dim=8,
        num_heads=1,
        num_layers=1,
        learning_rate=1e-3,
        epochs=2,
        window_size=86400.0,  # 1 day
    )


# ═══════════════════════════════════════════════════════════════
# 19a.1: Trainer.infer()
# ═══════════════════════════════════════════════════════════════


class TestTrainerInfer:
    """Tests for Trainer.infer() embedding extraction."""

    def test_infer_returns_embeddings_and_idmap(self, store_with_data: PipelineStore, small_config: TrainerConfig):
        trainer = Trainer(store_with_data, small_config)
        embeddings, id_map = trainer.infer()

        assert isinstance(embeddings, dict)
        assert len(embeddings) > 0
        for ntype, emb in embeddings.items():
            assert isinstance(emb, torch.Tensor)
            assert emb.ndim == 2
            assert emb.shape[1] == small_config.hidden_dim

    def test_infer_without_explicit_build_model(self, store_with_data: PipelineStore, small_config: TrainerConfig):
        """infer() should auto-build model if not built yet."""
        trainer = Trainer(store_with_data, small_config)
        assert trainer._model is None
        embeddings, id_map = trainer.infer()
        assert trainer._model is not None
        assert len(embeddings) > 0

    def test_infer_with_until_excludes_future(self, store_with_data: PipelineStore, small_config: TrainerConfig):
        trainer = Trainer(store_with_data, small_config)
        trainer.build_model()

        # Full graph
        emb_full, id_full = trainer.infer()

        # Only first 2 days
        emb_early, id_early = trainer.infer(until=86400.0 * 2)

        # Early graph should have same or fewer entities
        assert id_early.num_nodes <= id_full.num_nodes

    def test_infer_produces_no_grad_tensors(self, store_with_data: PipelineStore, small_config: TrainerConfig):
        trainer = Trainer(store_with_data, small_config)
        embeddings, _ = trainer.infer()
        for emb in embeddings.values():
            assert not emb.requires_grad

    def test_infer_deterministic(self, store_with_data: PipelineStore, small_config: TrainerConfig):
        trainer = Trainer(store_with_data, small_config)
        trainer.build_model()
        emb1, _ = trainer.infer()
        emb2, _ = trainer.infer()
        for ntype in emb1:
            assert torch.allclose(emb1[ntype], emb2[ntype], atol=1e-6)

    def test_infer_empty_store_returns_empty(self, empty_store: PipelineStore, small_config: TrainerConfig):
        trainer = Trainer(empty_store, small_config)
        embeddings, id_map = trainer.infer()
        assert embeddings == {}

    def test_infer_covers_all_entity_types_with_data(self, store_with_data: PipelineStore, small_config: TrainerConfig):
        """All entity types that have entities should appear in embeddings."""
        trainer = Trainer(store_with_data, small_config)
        embeddings, id_map = trainer.infer()
        entities = store_with_data.query_all_entities()
        types_in_store = {e["entity_type"] for e in entities}
        for t in types_in_store:
            assert t in embeddings, f"Missing embeddings for entity type {t}"


# ═══════════════════════════════════════════════════════════════
# 19a.2: Training smoke test
# ═══════════════════════════════════════════════════════════════


class TestTrainerSmokeTest:
    """Smoke tests for self-supervised training on synthetic data."""

    def test_train_completes_without_error(self, store_with_data: PipelineStore, small_config: TrainerConfig):
        trainer = Trainer(store_with_data, small_config)
        trainer.build_model()
        history = trainer.train()

        assert "total" in history
        assert len(history["total"]) == small_config.epochs

    def test_train_loss_dict_has_all_keys(self, store_with_data: PipelineStore, small_config: TrainerConfig):
        trainer = Trainer(store_with_data, small_config)
        trainer.build_model()
        history = trainer.train()

        for key in ("total", "obs_type", "time_delta", "contrastive"):
            assert key in history
            assert isinstance(history[key], list)

    def test_infer_after_training_works(self, store_with_data: PipelineStore, small_config: TrainerConfig):
        """Embeddings should be extractable after training."""
        trainer = Trainer(store_with_data, small_config)
        trainer.build_model()
        trainer.train()
        embeddings, id_map = trainer.infer()
        assert len(embeddings) > 0

    def test_train_embeddings_differ_from_untrained(self, store_with_data: PipelineStore, small_config: TrainerConfig):
        """Training should change embeddings (not identical to random init)."""
        trainer = Trainer(store_with_data, small_config)
        trainer.build_model()

        emb_before, _ = trainer.infer()
        before_flat = torch.cat([v.flatten() for v in emb_before.values()])

        trainer.train()
        emb_after, _ = trainer.infer()
        after_flat = torch.cat([v.flatten() for v in emb_after.values()])

        # They should differ after training (very unlikely to be identical)
        assert not torch.allclose(before_flat, after_flat, atol=1e-6)


# ═══════════════════════════════════════════════════════════════
# 19a.3: save_model / load_model
# ═══════════════════════════════════════════════════════════════


class TestModelPersistence:
    """Tests for save_model() and load_model() round-trip."""

    def test_save_creates_file(
        self,
        store_with_data: PipelineStore,
        small_config: TrainerConfig,
        tmp_path: Path,
    ):
        trainer = Trainer(store_with_data, small_config)
        trainer.build_model()

        model_path = tmp_path / "model.pt"
        trainer.save_model(model_path)
        assert model_path.exists()
        assert model_path.stat().st_size > 0

    def test_save_creates_parent_dirs(
        self,
        store_with_data: PipelineStore,
        small_config: TrainerConfig,
        tmp_path: Path,
    ):
        trainer = Trainer(store_with_data, small_config)
        trainer.build_model()

        model_path = tmp_path / "deep" / "nested" / "model.pt"
        trainer.save_model(model_path)
        assert model_path.exists()

    def test_save_without_model_raises(
        self,
        store_with_data: PipelineStore,
        small_config: TrainerConfig,
        tmp_path: Path,
    ):
        trainer = Trainer(store_with_data, small_config)
        with pytest.raises(RuntimeError, match="No model"):
            trainer.save_model(tmp_path / "model.pt")

    def test_load_recovers_model(
        self,
        store_with_data: PipelineStore,
        small_config: TrainerConfig,
        tmp_path: Path,
    ):
        trainer = Trainer(store_with_data, small_config)
        trainer.build_model()

        model_path = tmp_path / "model.pt"
        trainer.save_model(model_path)

        loaded = Trainer.load_model(model_path, store_with_data)
        assert loaded._model is not None
        assert loaded.config.hidden_dim == small_config.hidden_dim

    def test_load_produces_identical_embeddings(
        self,
        store_with_data: PipelineStore,
        small_config: TrainerConfig,
        tmp_path: Path,
    ):
        trainer = Trainer(store_with_data, small_config)
        trainer.build_model()
        emb_original, _ = trainer.infer()

        model_path = tmp_path / "model.pt"
        trainer.save_model(model_path)

        loaded = Trainer.load_model(model_path, store_with_data)
        emb_loaded, _ = loaded.infer()

        for ntype in emb_original:
            assert ntype in emb_loaded
            assert torch.allclose(emb_original[ntype], emb_loaded[ntype], atol=1e-5), f"Embeddings differ for {ntype}"

    def test_load_nonexistent_raises(self, store_with_data: PipelineStore, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            Trainer.load_model(tmp_path / "nonexistent.pt", store_with_data)

    def test_save_load_after_training(
        self,
        store_with_data: PipelineStore,
        small_config: TrainerConfig,
        tmp_path: Path,
    ):
        """Trained model should round-trip correctly."""
        trainer = Trainer(store_with_data, small_config)
        trainer.build_model()
        trainer.train()

        emb_trained, _ = trainer.infer()
        model_path = tmp_path / "trained.pt"
        trainer.save_model(model_path)

        loaded = Trainer.load_model(model_path, store_with_data)
        emb_loaded, _ = loaded.infer()

        for ntype in emb_trained:
            assert torch.allclose(emb_trained[ntype], emb_loaded[ntype], atol=1e-5)

    def test_loaded_model_can_continue_training(
        self,
        store_with_data: PipelineStore,
        small_config: TrainerConfig,
        tmp_path: Path,
    ):
        """Loaded model should be trainable (optimizer exists)."""
        trainer = Trainer(store_with_data, small_config)
        trainer.build_model()

        model_path = tmp_path / "model.pt"
        trainer.save_model(model_path)

        loaded = Trainer.load_model(model_path, store_with_data)
        history = loaded.train()
        assert len(history["total"]) == small_config.epochs

    def test_load_restores_concat_head_from_checkpoint_weights(
        self,
        tmp_path: Path,
    ):
        """State-dict inference restores concat head flags on legacy checkpoints."""
        from agent.models.gnn.trainer import _het_tgn_kwargs_from_checkpoint

        config = TrainerConfig(use_concat_head=False)
        checkpoint = {
            "in_channels": {"instrument": 14, "company": 10},
            "model_state_dict": {
                "return_concat_head.0.weight": torch.zeros(8, 142),
            },
        }
        kw = _het_tgn_kwargs_from_checkpoint(checkpoint, config)
        assert kw["use_concat_head"] is True
        assert kw["instrument_raw_dim"] == 14
        assert config.use_concat_head is True
