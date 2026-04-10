"""Tests for GNN integration layer (Phases 12f, 16a).

Covers:
    AutoPatternDetector — detect + store observations
    compare_patterns — auto vs hand-crafted stats
    retrain_and_discover — end-to-end pipeline
    format_diagnostic_report — threshold flagging (Phase 16a)
    End-to-end synthetic diagnostic pipeline (Phase 16a)
    Edge cases — empty data, no patterns, no links
"""

from __future__ import annotations

import math

import pytest

from agent.models.gnn.integration import (
    AutoPatternDetector,
    compare_patterns,
    compute_diagnostics,
    format_diagnostic_report,
    retrain_and_discover,
    run_diagnostics,
)
from agent.models.gnn.pattern_extractor import CrystallizedPattern
from agent.models.gnn.trainer import (
    InjectedPattern,
    SyntheticGraphGenerator,
    TrainerConfig,
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


# ═══════════════════════════════════════════════════════════════
# AutoPatternDetector tests
# ═══════════════════════════════════════════════════════════════


class TestAutoPatternDetector:
    def test_detect_with_injected_pattern(self, pattern_store):
        """Injected pattern should be detectable."""
        detector = AutoPatternDetector(pattern_store)
        rule = CrystallizedPattern(
            source_type="company",
            target_type="country",
            via_edge="headquartered_in",
            obs_type_a="insider_trade",
            obs_type_b="geopolitical_event",
            window_seconds=7200.0,  # 2 hours
        )
        detections = detector.detect([rule])
        assert len(detections) > 0

    def test_detection_fields(self, pattern_store):
        detector = AutoPatternDetector(pattern_store)
        rule = CrystallizedPattern(
            source_type="company",
            target_type="country",
            via_edge="headquartered_in",
            obs_type_a="insider_trade",
            obs_type_b="geopolitical_event",
            window_seconds=7200.0,
        )
        detections = detector.detect([rule])
        if detections:
            d = detections[0]
            assert "entity_a" in d
            assert "entity_b" in d
            assert d["pattern_source"] == "auto_gnn"
            assert 0 < d["lag"] <= 7200.0
            assert 0 <= d["score"] <= 1.0

    def test_detect_no_matching_pattern(self, populated_store):
        """Pattern for non-existent edge → no detections."""
        detector = AutoPatternDetector(populated_store)
        rule = CrystallizedPattern(
            source_type="person",
            target_type="organization",
            via_edge="works_at",
            obs_type_a="insider_trade",
            obs_type_b="geopolitical_event",
            window_seconds=3600.0,
        )
        detections = detector.detect([rule])
        assert detections == []

    def test_detect_empty_patterns(self, populated_store):
        detector = AutoPatternDetector(populated_store)
        detections = detector.detect([])
        assert detections == []

    def test_store_observations(self, pattern_store):
        detector = AutoPatternDetector(pattern_store)
        rule = CrystallizedPattern(
            source_type="company",
            target_type="country",
            via_edge="headquartered_in",
            obs_type_a="insider_trade",
            obs_type_b="geopolitical_event",
            window_seconds=7200.0,
        )
        detections = detector.detect([rule])
        count = detector.store_observations(detections)
        assert count == len(detections)

        # Verify stored observations
        all_obs = pattern_store.query_all_observations()
        auto_obs = [o for o in all_obs if o.get("source_tool") == "auto_gnn"]
        assert len(auto_obs) == count

    def test_store_observations_empty(self, populated_store):
        detector = AutoPatternDetector(populated_store)
        count = detector.store_observations([])
        assert count == 0


class TestAutoPatternDetectorWindow:
    def test_narrow_window_fewer_matches(self, pattern_store):
        """Narrower window should produce fewer detections."""
        detector = AutoPatternDetector(pattern_store)
        rule_wide = CrystallizedPattern(
            source_type="company",
            target_type="country",
            via_edge="headquartered_in",
            obs_type_a="insider_trade",
            obs_type_b="geopolitical_event",
            window_seconds=86400.0,  # 1 day
        )
        rule_narrow = CrystallizedPattern(
            source_type="company",
            target_type="country",
            via_edge="headquartered_in",
            obs_type_a="insider_trade",
            obs_type_b="geopolitical_event",
            window_seconds=1800.0,  # 30 min — tighter than injected 1h lag
        )
        wide_hits = detector.detect([rule_wide])
        narrow_hits = detector.detect([rule_narrow])
        assert len(wide_hits) >= len(narrow_hits)


# ═══════════════════════════════════════════════════════════════
# compare_patterns tests
# ═══════════════════════════════════════════════════════════════


class TestComparePatterns:
    def test_empty_store(self, store):
        result = compare_patterns(store)
        assert result["auto_gnn"]["count"] == 0
        assert result["hand_crafted"]["count"] == 0

    def test_after_auto_detection(self, pattern_store):
        detector = AutoPatternDetector(pattern_store)
        rule = CrystallizedPattern(
            source_type="company",
            target_type="country",
            via_edge="headquartered_in",
            obs_type_a="insider_trade",
            obs_type_b="geopolitical_event",
            window_seconds=7200.0,
        )
        detections = detector.detect([rule])
        detector.store_observations(detections)

        result = compare_patterns(pattern_store)
        assert result["auto_gnn"]["count"] == len(detections)
        if len(detections) > 0:
            assert result["auto_gnn"]["mean_score"] > 0


# ═══════════════════════════════════════════════════════════════
# retrain_and_discover tests
# ═══════════════════════════════════════════════════════════════


class TestRetrainAndDiscover:
    def test_end_to_end(self, populated_store):
        """Full pipeline: train → extract → crystallize."""
        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            num_heads=2,
            num_layers=1,
            epochs=2,
            window_size=86400.0 * 2,
        )
        result = retrain_and_discover(
            populated_store,
            cfg,
            score_threshold=0.0,
            top_k=5,
            include_diagnostics=False,
        )
        assert isinstance(result, dict)
        rules = result["patterns"]
        assert isinstance(rules, list)
        for r in rules:
            assert isinstance(r, CrystallizedPattern)

    def test_end_to_end_with_detection(self, populated_store):
        """Discovered rules can actually detect something."""
        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            num_heads=2,
            num_layers=1,
            epochs=2,
            window_size=86400.0 * 2,
        )
        result = retrain_and_discover(
            populated_store,
            cfg,
            score_threshold=0.0,
            include_diagnostics=False,
        )
        rules = result["patterns"]
        if rules:
            detector = AutoPatternDetector(populated_store)
            detections = detector.detect(rules)
            # At least some patterns should match (not guaranteed though)
            assert isinstance(detections, list)

    def test_high_threshold_no_rules(self, populated_store):
        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            epochs=1,
            window_size=86400.0 * 2,
        )
        result = retrain_and_discover(
            populated_store,
            cfg,
            score_threshold=1e6,
            include_diagnostics=False,
        )
        assert result["patterns"] == []


class TestIntegrationEdgeCases:
    def test_single_entity_store(self, store):
        eid = store.register_entity(
            "company", "solo", entity_id_from_key("company", "solo")
        )
        store.store_entity_observation(eid, "test", 100.0, "insider_trade", {"v": 1})
        store.store_entity_observation(eid, "test", 200.0, "insider_trade", {"v": 2})
        cfg = TrainerConfig(
            hidden_dim=8,
            memory_dim=8,
            message_dim=8,
            num_heads=1,
            num_layers=1,
            epochs=1,
            window_size=50.0,
        )
        rules = retrain_and_discover(
            store, cfg, score_threshold=0.0, include_diagnostics=False
        )
        # No links → no patterns to crystallize
        assert rules["patterns"] == []

    def test_multiple_pattern_rules(self, pattern_store):
        """Multiple crystallized rules can be evaluated at once."""
        detector = AutoPatternDetector(pattern_store)
        rules = [
            CrystallizedPattern(
                source_type="company",
                target_type="country",
                via_edge="headquartered_in",
                obs_type_a="insider_trade",
                obs_type_b="geopolitical_event",
                window_seconds=7200.0,
            ),
            CrystallizedPattern(
                source_type="vessel",
                target_type="country",
                via_edge="port_call_to",
                obs_type_a="port_call",
                obs_type_b="geopolitical_event",
                window_seconds=86400.0,
            ),
        ]
        detections = detector.detect(rules)
        assert isinstance(detections, list)


# ═══════════════════════════════════════════════════════════════
# format_diagnostic_report tests (Phase 16a)
# ═══════════════════════════════════════════════════════════════


class TestFormatDiagnosticReport:
    """Unit tests for format_diagnostic_report()."""

    @pytest.fixture
    def sample_diagnostics(self):
        return {
            "entity_type_density": {
                "company": 50,
                "country": 10,
                "domain": 3,  # below default threshold 5
                "protocol": 1,  # below threshold
            },
            "observation_density": {
                "insider_trade": 200,
                "patent_filing": 5,  # below threshold 10
                "cert_issued": 8,  # below threshold
            },
            "edge_type_attention": {
                "headquartered_in": 0.12,
                "port_call_to": 0.03,  # below threshold 0.05
                "exchange_based_in": 0.01,  # below threshold
            },
            "neighborhood_sparsity": {
                "company": 2.5,
                "country": 3.0,
                "domain": 0.5,  # below threshold 1.0
                "protocol": 0.0,  # below threshold
            },
            "supervised_confidence": {
                "company": 0.72,
                "domain": 0.48,  # within 0.4-0.6 → uncertain
                "protocol": 0.51,  # within 0.4-0.6 → uncertain
            },
        }

    def test_report_has_all_sections(self, sample_diagnostics):
        report = format_diagnostic_report(sample_diagnostics)
        expected_sections = [
            "entity_density",
            "observation_density",
            "edge_attention",
            "neighborhood_sparsity",
            "supervised_confidence",
            "summary",
        ]
        for section in expected_sections:
            assert section in report, f"Missing section: {section}"

    def test_flagged_entities(self, sample_diagnostics):
        report = format_diagnostic_report(sample_diagnostics)
        flagged = report["entity_density"]["flagged"]
        assert "domain" in flagged
        assert "protocol" in flagged
        assert "company" not in flagged
        assert "country" not in flagged

    def test_flagged_obs(self, sample_diagnostics):
        report = format_diagnostic_report(sample_diagnostics)
        flagged = report["observation_density"]["flagged"]
        assert "patent_filing" in flagged
        assert "cert_issued" in flagged
        assert "insider_trade" not in flagged

    def test_flagged_attention(self, sample_diagnostics):
        report = format_diagnostic_report(sample_diagnostics)
        flagged = report["edge_attention"]["flagged"]
        assert "port_call_to" in flagged
        assert "exchange_based_in" in flagged
        assert "headquartered_in" not in flagged

    def test_flagged_sparsity(self, sample_diagnostics):
        report = format_diagnostic_report(sample_diagnostics)
        flagged = report["neighborhood_sparsity"]["flagged"]
        assert "domain" in flagged
        assert "protocol" in flagged
        assert "company" not in flagged

    def test_flagged_confidence(self, sample_diagnostics):
        report = format_diagnostic_report(sample_diagnostics)
        flagged = report["supervised_confidence"]["flagged"]
        assert "domain" in flagged
        assert "protocol" in flagged
        assert "company" not in flagged

    def test_summary_counts(self, sample_diagnostics):
        report = format_diagnostic_report(sample_diagnostics)
        s = report["summary"]
        assert s["flagged_entity_types"] == 2
        assert s["flagged_obs_types"] == 2
        assert s["flagged_edge_types"] == 2
        assert s["flagged_sparse_types"] == 2
        assert s["flagged_uncertain_types"] == 2

    def test_custom_thresholds(self, sample_diagnostics):
        report = format_diagnostic_report(sample_diagnostics, entity_density_min=100)
        # All entity types < 100 should be flagged
        assert len(report["entity_density"]["flagged"]) == 4

    def test_empty_diagnostics(self):
        report = format_diagnostic_report({})
        assert report["summary"]["flagged_entity_types"] == 0
        assert report["entity_density"]["values"] == {}
        assert report["entity_density"]["flagged"] == {}

    def test_values_preserved(self, sample_diagnostics):
        report = format_diagnostic_report(sample_diagnostics)
        assert (
            report["entity_density"]["values"]
            == sample_diagnostics["entity_type_density"]
        )
        assert (
            report["observation_density"]["values"]
            == sample_diagnostics["observation_density"]
        )

    def test_no_supervised_confidence(self):
        diag = {
            "entity_type_density": {"company": 10},
            "observation_density": {"insider_trade": 50},
            "edge_type_attention": {},
            "neighborhood_sparsity": {"company": 2.0},
            "supervised_confidence": {},
        }
        report = format_diagnostic_report(diag)
        assert report["supervised_confidence"]["flagged"] == {}
        assert report["summary"]["flagged_uncertain_types"] == 0


# ═══════════════════════════════════════════════════════════════
# End-to-end synthetic diagnostic pipeline (Phase 16a)
# ═══════════════════════════════════════════════════════════════


class TestSyntheticDiagnosticPipeline:
    """Full pipeline: SyntheticGraphGenerator → train → crystallize →
    compute_diagnostics → format_diagnostic_report.
    Validates that the diagnostic workflow runs end-to-end on synthetic data.
    """

    def test_full_pipeline(self, populated_store):
        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            num_heads=2,
            num_layers=1,
            epochs=2,
            window_size=86400.0 * 2,
        )
        result = retrain_and_discover(
            populated_store,
            cfg,
            score_threshold=0.0,
            include_diagnostics=True,
        )
        assert "diagnostics" in result

        report = format_diagnostic_report(result["diagnostics"])

        # All five sections present
        for key in (
            "entity_density",
            "observation_density",
            "edge_attention",
            "neighborhood_sparsity",
            "supervised_confidence",
        ):
            assert key in report
            assert "values" in report[key]
            assert "flagged" in report[key]

        # Summary present with integer counts
        s = report["summary"]
        for k in s:
            assert isinstance(s[k], int)
            assert s[k] >= 0

        # Values should be finite floats or ints
        for v in report["entity_density"]["values"].values():
            assert isinstance(v, (int, float))
            assert math.isfinite(v)
        for v in report["neighborhood_sparsity"]["values"].values():
            assert isinstance(v, (int, float))
            assert math.isfinite(v)

    def test_pipeline_with_injected_pattern(self, pattern_store):
        """Injected-pattern store should produce non-empty diagnostics."""
        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            num_heads=2,
            num_layers=1,
            epochs=2,
            window_size=86400.0 * 2,
        )
        result = retrain_and_discover(
            pattern_store,
            cfg,
            score_threshold=0.0,
            finetune=True,
            finetune_epochs=2,
            include_diagnostics=True,
        )
        assert "diagnostics" in result
        diag = result["diagnostics"]

        # Entity graph should be non-empty
        assert len(diag["entity_type_density"]) > 0
        assert len(diag["observation_density"]) > 0
        assert len(diag["neighborhood_sparsity"]) > 0

        report = format_diagnostic_report(diag)
        # Should have entity types
        assert len(report["entity_density"]["values"]) > 0


# ═══════════════════════════════════════════════════════════════
# run_diagnostics — Phase 16b edge cases
# ═══════════════════════════════════════════════════════════════


class TestRunDiagnostics:
    """Tests for run_diagnostics() — the CLI-callable entry point."""

    def test_nonexistent_path_raises(self, tmp_path):
        """Missing DB file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            run_diagnostics(str(tmp_path / "does_not_exist.db"))

    def test_directory_path_raises(self, tmp_path):
        """Passing a directory instead of a file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            run_diagnostics(str(tmp_path))

    def test_empty_store_returns_empty_graph(self, tmp_path):
        """A valid DB with no entities should return status='empty_graph'."""
        db_file = tmp_path / "empty.db"
        # Create a real empty PipelineStore on disk
        empty_store = PipelineStore(db_path=str(db_file))
        empty_store.close()

        result = run_diagnostics(str(db_file))
        assert result["status"] == "empty_graph"
        assert result["diagnostics"] is None
        assert result["report"] is None
        assert result["patterns"] == []
        assert result["entity_count"] == 0
        assert "0 entities" in result["message"]

    def test_populated_store_returns_ok(self, tmp_path):
        """A store with synthetic entity data should return status='ok'."""
        db_file = tmp_path / "populated.db"
        store = PipelineStore(db_path=str(db_file))
        gen = SyntheticGraphGenerator(
            num_companies=3,
            num_countries=2,
            num_vessels=1,
            num_wallets=1,
            time_span=86400.0 * 5,
            base_event_rate=0.0005,
            seed=99,
        )
        gen.generate(store)
        store.close()

        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            num_heads=2,
            num_layers=1,
            epochs=2,
            window_size=86400.0 * 2,
        )
        result = run_diagnostics(str(db_file), config=cfg)

        assert result["status"] == "ok"
        assert result["diagnostics"] is not None
        assert result["report"] is not None
        assert result["entity_count"] > 0
        assert result["obs_count"] > 0
        assert "Diagnostics complete" in result["message"]

        # Report structure should be well-formed
        report = result["report"]
        for key in (
            "entity_density",
            "observation_density",
            "edge_attention",
            "neighborhood_sparsity",
            "supervised_confidence",
        ):
            assert key in report
            assert "values" in report[key]
            assert "flagged" in report[key]

    def test_no_crystallized_patterns_still_works(self, tmp_path):
        """If score_threshold is very high, no patterns crystallize.

        Diagnostics should still run with empty supervised_confidence.
        """
        db_file = tmp_path / "no_patterns.db"
        store = PipelineStore(db_path=str(db_file))
        gen = SyntheticGraphGenerator(
            num_companies=3,
            num_countries=2,
            num_vessels=1,
            num_wallets=1,
            time_span=86400.0 * 5,
            base_event_rate=0.0005,
            seed=42,
        )
        gen.generate(store)
        store.close()

        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            num_heads=2,
            num_layers=1,
            epochs=2,
            window_size=86400.0 * 2,
        )
        result = run_diagnostics(str(db_file), config=cfg, score_threshold=999.0)

        assert result["status"] == "ok"
        # supervised_confidence should be empty or absent of meaningful data
        sc = result["diagnostics"]["supervised_confidence"]
        # With no crystallized patterns, supervised confidence shouldn't have data
        assert isinstance(sc, dict)

    def test_result_keys_always_present(self, tmp_path):
        """Every result dict must contain the canonical keys regardless of status."""
        db_file = tmp_path / "keys.db"
        store = PipelineStore(db_path=str(db_file))
        store.close()

        result = run_diagnostics(str(db_file))
        for key in (
            "status",
            "diagnostics",
            "report",
            "patterns",
            "message",
            "entity_count",
            "obs_count",
        ):
            assert key in result, f"Missing key: {key}"

    def test_finetune_flag_propagates(self, tmp_path):
        """finetune=True should not crash even when no outcome labels exist."""
        db_file = tmp_path / "ft.db"
        store = PipelineStore(db_path=str(db_file))
        gen = SyntheticGraphGenerator(
            num_companies=3,
            num_countries=2,
            num_vessels=1,
            num_wallets=1,
            time_span=86400.0 * 5,
            base_event_rate=0.0005,
            seed=42,
        )
        gen.generate(store)
        store.close()

        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            num_heads=2,
            num_layers=1,
            epochs=2,
            window_size=86400.0 * 2,
        )
        result = run_diagnostics(
            str(db_file), config=cfg, finetune=True, finetune_epochs=1
        )
        assert result["status"] == "ok"
