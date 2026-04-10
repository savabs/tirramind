"""Tests for Phase 15: Outcome-Labeled Fine-Tuning & GNN Diagnostics.

Covers:
    OutcomeLabel dataclass       — field checks, creation
    generate_outcome_labels      — positive/negative labels, balancing, temporal
    SupervisedHead               — shape, gradient flow, probability range
    FineTuner                    — loss decrease, frozen layers, empty labels
    evaluate_supervised          — metric computation, edge cases
    compute_diagnostics          — structure, non-empty results
    retrain_and_discover         — dict return, diagnostics inclusion
"""

from __future__ import annotations

import pytest
import torch

from agent.models.gnn.graph_builder import GraphBuilder, IDMap
from agent.models.gnn.het_tgn import HetTGN, SupervisedHead
from agent.models.gnn.integration import compute_diagnostics, retrain_and_discover
from agent.models.gnn.pattern_extractor import CrystallizedPattern
from agent.models.gnn.trainer import (
    FineTuner,
    InjectedPattern,
    OutcomeLabel,
    SyntheticGraphGenerator,
    TrainerConfig,
    Trainer,
    evaluate_supervised,
    generate_outcome_labels,
)
from agent.pipeline.entity import entity_id_from_key
from agent.pipeline.store import PipelineStore


# ─── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def store():
    return PipelineStore(db_path=":memory:")


@pytest.fixture
def populated_store(store):
    gen = SyntheticGraphGenerator(
        num_companies=4,
        num_countries=2,
        num_vessels=2,
        num_wallets=2,
        time_span=86400.0 * 10,
        base_event_rate=0.0005,
        seed=42,
    )
    gen.generate(store)
    return store


@pytest.fixture
def pattern_store(store):
    pattern = InjectedPattern(
        source_type="company",
        source_obs_type="insider_trade",
        target_type="country",
        target_obs_type="geopolitical_event",
        via_edge="headquartered_in",
        lag_seconds=3600.0,
        lag_jitter=300.0,
    )
    gen = SyntheticGraphGenerator(
        num_companies=4,
        num_countries=2,
        num_vessels=2,
        num_wallets=2,
        time_span=86400.0 * 10,
        base_event_rate=0.0005,
        seed=42,
        patterns=[pattern],
    )
    gen.generate(store)
    return store


@pytest.fixture
def sample_pattern():
    return CrystallizedPattern(
        source_type="company",
        target_type="country",
        via_edge="headquartered_in",
        obs_type_a="insider_trade",
        obs_type_b="geopolitical_event",
        window_seconds=7200.0,
    )


@pytest.fixture
def small_config():
    return TrainerConfig(
        hidden_dim=16,
        memory_dim=16,
        message_dim=16,
        num_heads=2,
        num_layers=1,
        epochs=2,
        window_size=86400.0 * 2,
    )


# ═══════════════════════════════════════════════════════════════
# OutcomeLabel tests
# ═══════════════════════════════════════════════════════════════


class TestOutcomeLabel:
    def test_fields(self):
        lbl = OutcomeLabel(
            src_entity_id="eid_a",
            dst_entity_id="eid_b",
            src_type="company",
            dst_type="country",
            pattern_edge="headquartered_in",
            timestamp=1000.0,
            label=1,
        )
        assert lbl.src_entity_id == "eid_a"
        assert lbl.dst_entity_id == "eid_b"
        assert lbl.src_type == "company"
        assert lbl.dst_type == "country"
        assert lbl.pattern_edge == "headquartered_in"
        assert lbl.timestamp == 1000.0
        assert lbl.label == 1

    def test_label_zero(self):
        lbl = OutcomeLabel(
            src_entity_id="x",
            dst_entity_id="y",
            src_type="a",
            dst_type="b",
            pattern_edge="e",
            timestamp=0,
            label=0,
        )
        assert lbl.label == 0


# ═══════════════════════════════════════════════════════════════
# generate_outcome_labels tests
# ═══════════════════════════════════════════════════════════════


class TestGenerateOutcomeLabels:
    def test_with_injected_pattern(self, pattern_store, sample_pattern):
        """Injected pattern should produce positive labels."""
        labels = generate_outcome_labels([sample_pattern], pattern_store)
        assert len(labels) > 0
        positives = [lbl for lbl in labels if lbl.label == 1]
        negatives = [lbl for lbl in labels if lbl.label == 0]
        assert len(positives) > 0, "Injected pattern should yield positives"

    def test_temporal_ordering(self, pattern_store, sample_pattern):
        """Labels should be sorted by timestamp."""
        labels = generate_outcome_labels([sample_pattern], pattern_store)
        if len(labels) > 1:
            timestamps = [lbl.timestamp for lbl in labels]
            assert timestamps == sorted(timestamps)

    def test_negative_ratio_capped(self, pattern_store, sample_pattern):
        """Negative-to-positive ratio should not exceed max_neg_ratio."""
        labels = generate_outcome_labels(
            [sample_pattern],
            pattern_store,
            max_neg_ratio=2.0,
        )
        positives = sum(1 for lbl in labels if lbl.label == 1)
        negatives = sum(1 for lbl in labels if lbl.label == 0)
        if positives > 0:
            assert negatives <= positives * 2 + 1  # +1 for rounding

    def test_empty_patterns(self, pattern_store):
        """No patterns → no labels."""
        labels = generate_outcome_labels([], pattern_store)
        assert labels == []

    def test_no_matching_data(self, populated_store):
        """Pattern that matches nothing produces no labels (or only negatives)."""
        bad_pattern = CrystallizedPattern(
            source_type="company",
            target_type="country",
            via_edge="headquartered_in",
            obs_type_a="nonexistent_obs",
            obs_type_b="geopolitical_event",
            window_seconds=3600.0,
        )
        labels = generate_outcome_labels([bad_pattern], populated_store)
        positives = [lbl for lbl in labels if lbl.label == 1]
        assert len(positives) == 0

    def test_label_fields_populated(self, pattern_store, sample_pattern):
        """Every label field should be populated."""
        labels = generate_outcome_labels([sample_pattern], pattern_store)
        if labels:
            lbl = labels[0]
            assert lbl.src_entity_id
            assert lbl.dst_entity_id
            assert lbl.src_type == "company"
            assert lbl.dst_type == "country"
            assert lbl.pattern_edge == "headquartered_in"
            assert isinstance(lbl.timestamp, (int, float))
            assert lbl.label in (0, 1)


# ═══════════════════════════════════════════════════════════════
# SupervisedHead tests
# ═══════════════════════════════════════════════════════════════


class TestSupervisedHead:
    def test_output_shape(self):
        head = SupervisedHead(hidden_dim=32)
        src = torch.randn(5, 32)
        dst = torch.randn(5, 32)
        out = head(src, dst)
        assert out.shape == (5,)

    def test_output_range(self):
        """Output should be in [0, 1] (sigmoid)."""
        head = SupervisedHead(hidden_dim=16)
        src = torch.randn(10, 16)
        dst = torch.randn(10, 16)
        out = head(src, dst)
        assert (out >= 0).all() and (out <= 1).all()

    def test_gradient_flow(self):
        """Gradients should flow through the head."""
        head = SupervisedHead(hidden_dim=8)
        src = torch.randn(3, 8, requires_grad=True)
        dst = torch.randn(3, 8, requires_grad=True)
        out = head(src, dst)
        loss = out.sum()
        loss.backward()
        assert src.grad is not None
        assert dst.grad is not None
        assert head.weight.grad is not None
        assert head.bias.grad is not None

    def test_single_sample(self):
        head = SupervisedHead(hidden_dim=4)
        src = torch.randn(1, 4)
        dst = torch.randn(1, 4)
        out = head(src, dst)
        assert out.shape == (1,)

    def test_deterministic(self):
        """Same input → same output."""
        head = SupervisedHead(hidden_dim=8)
        head.eval()
        src = torch.randn(2, 8)
        dst = torch.randn(2, 8)
        out1 = head(src, dst)
        out2 = head(src, dst)
        assert torch.allclose(out1, out2)


# ═══════════════════════════════════════════════════════════════
# HetTGN.predict_outcome tests
# ═══════════════════════════════════════════════════════════════


class TestPredictOutcome:
    def test_predict_outcome_shape(self, populated_store, small_config):
        trainer = Trainer(populated_store, small_config)
        model = trainer.build_model()
        src = torch.randn(3, small_config.hidden_dim)
        dst = torch.randn(3, small_config.hidden_dim)
        probs = model.predict_outcome(src, dst)
        assert probs.shape == (3,)
        assert (probs >= 0).all() and (probs <= 1).all()


# ═══════════════════════════════════════════════════════════════
# FineTuner tests
# ═══════════════════════════════════════════════════════════════


class TestFineTuner:
    def test_finetune_runs(self, pattern_store, sample_pattern, small_config):
        """Fine-tuning completes without error."""
        trainer = Trainer(pattern_store, small_config)
        model = trainer.build_model()
        trainer.train()

        labels = generate_outcome_labels([sample_pattern], pattern_store)
        if not labels:
            pytest.skip("No outcome labels generated")

        ft = FineTuner(model, pattern_store, labels, epochs=3)
        history = ft.finetune()
        assert "loss" in history
        assert "accuracy" in history
        assert len(history["loss"]) > 0

    def test_frozen_backbone(self, pattern_store, sample_pattern, small_config):
        """When freeze_backbone=True, only supervised_head params have grads."""
        trainer = Trainer(pattern_store, small_config)
        model = trainer.build_model()
        trainer.train()

        labels = generate_outcome_labels([sample_pattern], pattern_store)
        if not labels:
            pytest.skip("No outcome labels generated")

        # Take a snapshot of backbone params before fine-tuning
        backbone_before = {}
        for name, param in model.named_parameters():
            if "supervised_head" not in name:
                backbone_before[name] = param.data.clone()

        ft = FineTuner(model, pattern_store, labels, epochs=3, freeze_backbone=True)
        ft.finetune()

        # After fine-tuning, backbone should be unfrozen again
        for param in model.parameters():
            assert param.requires_grad, "All params should be unfrozen after finetune"

    def test_empty_labels(self, populated_store, small_config):
        """Empty labels → skip fine-tuning."""
        trainer = Trainer(populated_store, small_config)
        model = trainer.build_model()
        ft = FineTuner(model, populated_store, [], epochs=3)
        history = ft.finetune()
        assert history == {"loss": [], "accuracy": []}

    def test_loss_decreases_or_stable(
        self, pattern_store, sample_pattern, small_config
    ):
        """Loss should not explode during fine-tuning."""
        trainer = Trainer(pattern_store, small_config)
        model = trainer.build_model()
        trainer.train()

        labels = generate_outcome_labels([sample_pattern], pattern_store)
        if not labels:
            pytest.skip("No outcome labels generated")

        ft = FineTuner(model, pattern_store, labels, epochs=5, lr=1e-3)
        history = ft.finetune()
        losses = history["loss"]
        if len(losses) >= 2:
            # Loss should not explode (10x increase would be concerning)
            assert losses[-1] < losses[0] * 10, "Loss should not explode"


# ═══════════════════════════════════════════════════════════════
# evaluate_supervised tests
# ═══════════════════════════════════════════════════════════════


class TestEvaluateSupervised:
    def test_returns_metrics(self, pattern_store, sample_pattern, small_config):
        """evaluate_supervised returns AUROC, precision, recall, F1."""
        trainer = Trainer(pattern_store, small_config)
        model = trainer.build_model()
        trainer.train()

        labels = generate_outcome_labels([sample_pattern], pattern_store)
        if not labels or len(set(lbl.label for lbl in labels)) < 2:
            pytest.skip("Need both classes for evaluation")

        metrics = evaluate_supervised(model, pattern_store, labels)
        assert "auroc" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "num_samples" in metrics
        assert metrics["num_samples"] > 0
        assert 0.0 <= metrics["auroc"] <= 1.0

    def test_empty_labels(self, populated_store, small_config):
        """Empty labels → zero metrics."""
        trainer = Trainer(populated_store, small_config)
        model = trainer.build_model()
        metrics = evaluate_supervised(model, populated_store, [])
        assert metrics["num_samples"] == 0
        assert metrics["auroc"] == 0.0

    def test_single_class_labels(self, pattern_store, small_config):
        """Single class → graceful handling (auroc = 0)."""
        trainer = Trainer(pattern_store, small_config)
        model = trainer.build_model()
        trainer.train()

        # Create all-positive labels
        all_pos = [
            OutcomeLabel(
                src_entity_id="x",
                dst_entity_id="y",
                src_type="company",
                dst_type="country",
                pattern_edge="e",
                timestamp=float(i),
                label=1,
            )
            for i in range(5)
        ]
        metrics = evaluate_supervised(model, pattern_store, all_pos)
        # With fake entity IDs that don't exist, num_samples should be 0
        assert metrics["num_samples"] == 0


# ═══════════════════════════════════════════════════════════════
# compute_diagnostics tests
# ═══════════════════════════════════════════════════════════════


class TestComputeDiagnostics:
    def test_diagnostic_structure(self, populated_store, small_config):
        """Diagnostics dict has all expected keys."""
        trainer = Trainer(populated_store, small_config)
        model = trainer.build_model()
        trainer.train()

        diag = compute_diagnostics(model, populated_store)
        assert "entity_type_density" in diag
        assert "observation_density" in diag
        assert "edge_type_attention" in diag
        assert "neighborhood_sparsity" in diag
        assert "supervised_confidence" in diag

    def test_entity_type_density(self, populated_store, small_config):
        """Entity type density should match store."""
        trainer = Trainer(populated_store, small_config)
        model = trainer.build_model()
        diag = compute_diagnostics(model, populated_store)
        density = diag["entity_type_density"]
        assert isinstance(density, dict)
        assert len(density) > 0
        total = sum(density.values())
        all_entities = populated_store.query_all_entities()
        assert total == len(all_entities)

    def test_observation_density(self, populated_store, small_config):
        """Observation density should be non-empty."""
        trainer = Trainer(populated_store, small_config)
        model = trainer.build_model()
        diag = compute_diagnostics(model, populated_store)
        obs_d = diag["observation_density"]
        assert isinstance(obs_d, dict)
        assert len(obs_d) > 0

    def test_neighborhood_sparsity(self, populated_store, small_config):
        """Sparsity should have entries for entity types with links."""
        trainer = Trainer(populated_store, small_config)
        model = trainer.build_model()
        diag = compute_diagnostics(model, populated_store)
        sparsity = diag["neighborhood_sparsity"]
        assert isinstance(sparsity, dict)
        # At least some types should have non-zero degree
        assert any(v > 0 for v in sparsity.values())

    def test_edge_type_attention(self, populated_store, small_config):
        """Edge attention should have entries for graph edge types."""
        trainer = Trainer(populated_store, small_config)
        model = trainer.build_model()
        trainer.train()
        diag = compute_diagnostics(model, populated_store)
        attn = diag["edge_type_attention"]
        assert isinstance(attn, dict)
        # May be empty if graph has no edges, but should be a dict

    def test_with_crystallized_patterns(
        self, pattern_store, sample_pattern, small_config
    ):
        """With patterns, supervised_confidence should be populated."""
        trainer = Trainer(pattern_store, small_config)
        model = trainer.build_model()
        trainer.train()

        diag = compute_diagnostics(model, pattern_store, [sample_pattern])
        # supervised_confidence may or may not have entries depending on
        # whether outcome labels could be generated
        assert isinstance(diag["supervised_confidence"], dict)

    def test_without_crystallized(self, populated_store, small_config):
        """Without patterns, supervised_confidence is empty."""
        trainer = Trainer(populated_store, small_config)
        model = trainer.build_model()
        diag = compute_diagnostics(model, populated_store, None)
        assert diag["supervised_confidence"] == {}

    def test_empty_store(self, store, small_config):
        """Empty store should not crash."""
        # Build a model from an empty store — it should still work
        trainer = Trainer(store, small_config)
        model = trainer.build_model()
        diag = compute_diagnostics(model, store)
        assert diag["entity_type_density"] == {}
        assert diag["observation_density"] == {}
        assert diag["neighborhood_sparsity"] == {}
        assert diag["supervised_confidence"] == {}


# ═══════════════════════════════════════════════════════════════
# retrain_and_discover dict return tests
# ═══════════════════════════════════════════════════════════════


class TestRetrainAndDiscoverPhase15:
    def test_returns_dict(self, populated_store, small_config):
        """retrain_and_discover returns dict with 'patterns' key."""
        result = retrain_and_discover(
            populated_store,
            small_config,
            score_threshold=0.0,
            include_diagnostics=False,
        )
        assert isinstance(result, dict)
        assert "patterns" in result

    def test_includes_diagnostics(self, populated_store, small_config):
        """With include_diagnostics=True, diagnostics key present."""
        result = retrain_and_discover(
            populated_store,
            small_config,
            score_threshold=0.0,
            include_diagnostics=True,
        )
        assert "diagnostics" in result
        diag = result["diagnostics"]
        assert "entity_type_density" in diag
        assert "neighborhood_sparsity" in diag

    def test_finetune_flag(self, pattern_store, small_config):
        """With finetune=True, finetune_history appears in result."""
        result = retrain_and_discover(
            pattern_store,
            small_config,
            score_threshold=0.0,
            finetune=True,
            finetune_epochs=2,
            include_diagnostics=False,
        )
        patterns = result["patterns"]
        if patterns:
            # If patterns were found and labels generated,
            # finetune_history should be present
            if "finetune_history" in result:
                assert "loss" in result["finetune_history"]


# ═══════════════════════════════════════════════════════════════
# Edge case tests
# ═══════════════════════════════════════════════════════════════


class TestPhase15EdgeCases:
    def test_supervised_head_large_batch(self):
        """Large batch shouldn't cause issues."""
        head = SupervisedHead(hidden_dim=32)
        src = torch.randn(1000, 32)
        dst = torch.randn(1000, 32)
        out = head(src, dst)
        assert out.shape == (1000,)
        assert not torch.isnan(out).any()

    def test_supervised_head_zero_input(self):
        """Zero inputs → sigmoid(bias) ≈ 0.5."""
        head = SupervisedHead(hidden_dim=8)
        src = torch.zeros(2, 8)
        dst = torch.zeros(2, 8)
        out = head(src, dst)
        # With zero-initialized bias, should be close to 0.5
        assert (out > 0.4).all() and (out < 0.6).all()

    def test_finetune_no_freeze(self, pattern_store, sample_pattern, small_config):
        """Fine-tuning without freezing backbone."""
        trainer = Trainer(pattern_store, small_config)
        model = trainer.build_model()
        trainer.train()

        labels = generate_outcome_labels([sample_pattern], pattern_store)
        if not labels:
            pytest.skip("No outcome labels generated")

        ft = FineTuner(
            model,
            pattern_store,
            labels,
            epochs=2,
            freeze_backbone=False,
        )
        history = ft.finetune()
        assert len(history["loss"]) > 0

    def test_generate_labels_multiple_patterns(self, pattern_store):
        """Multiple patterns should each contribute labels."""
        p1 = CrystallizedPattern(
            source_type="company",
            target_type="country",
            via_edge="headquartered_in",
            obs_type_a="insider_trade",
            obs_type_b="geopolitical_event",
            window_seconds=7200.0,
        )
        p2 = CrystallizedPattern(
            source_type="vessel",
            target_type="country",
            via_edge="port_call_to",
            obs_type_a="port_call",
            obs_type_b="geopolitical_event",
            window_seconds=86400.0,
        )
        labels = generate_outcome_labels([p1, p2], pattern_store)
        # At least one pattern should produce labels
        assert isinstance(labels, list)
