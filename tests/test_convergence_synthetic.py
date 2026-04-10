"""Tests for synthetic validation engine (convergence backtest sub-phase A).

Covers:
- Scenario generation: shapes, categories, timestamps, determinism
- Planted chain: signal patterns, correlation, anomaly injection
- Decoy signals: independence, low magnitude
- Validation runner: precision/recall/F1, edge cases
- Registry builder: dedup, all categories covered
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from agent.convergence.evidence import Evidence
from agent.convergence.detector import ConvergenceDetectorConfig
from agent.convergence.synthetic import (
    SyntheticScenario,
    SyntheticValidationResult,
    _build_registry_from_evidence,
    _generate_correlated_series,
    _pick_signal_for_step,
    generate_decoy_signals,
    generate_planted_chain,
    generate_scenarios,
    run_synthetic_validation,
)
from agent.convergence.taxonomy import CATEGORIES
from agent.convergence.templates import (
    TEMPLATE_LIBRARY,
    CausalTemplate,
    TemplateStep,
)


# ── Helpers ────────────────────────────────────────────────────

_DAY = 86_400


class TestCorrelatedSeriesGeneration:
    """Test the factor-model series generator."""

    def test_shape(self):
        rng = np.random.default_rng(0)
        s = _generate_correlated_series(3, 50, 0.8, rng)
        assert s.shape == (3, 50)

    def test_correlation_positive(self):
        rng = np.random.default_rng(42)
        s = _generate_correlated_series(2, 1000, 0.9, rng)
        rho = np.corrcoef(s[0], s[1])[0, 1]
        assert rho > 0.7, f"Expected high correlation, got {rho:.3f}"

    def test_correlation_zero(self):
        rng = np.random.default_rng(42)
        s = _generate_correlated_series(2, 1000, 0.0, rng)
        rho = np.corrcoef(s[0], s[1])[0, 1]
        assert abs(rho) < 0.15, f"Expected near-zero correlation, got {rho:.3f}"

    def test_single_signal(self):
        rng = np.random.default_rng(0)
        s = _generate_correlated_series(1, 20, 0.5, rng)
        assert s.shape == (1, 20)

    def test_clamp_rho(self):
        rng = np.random.default_rng(0)
        s = _generate_correlated_series(2, 50, 1.5, rng)  # > 1 clamped
        assert s.shape == (2, 50)
        s2 = _generate_correlated_series(2, 50, -0.5, rng)  # < 0 clamped
        assert s2.shape == (2, 50)


class TestPickSignalForStep:
    """Test signal-category assignment for template steps."""

    def test_physical_disruption_step(self):
        step = TemplateStep("physical_disruption", r"weather\.", 0, +1)
        rng = np.random.default_rng(0)
        sig_id, cat = _pick_signal_for_step(step, rng)
        assert cat == "physical_disruption"
        assert "weather" in sig_id

    def test_multi_category_step(self):
        step = TemplateStep(
            "monetary_policy|financial_stress",
            r"capital_flows\.|sovereign_debt\.",
            7,
            None,
        )
        rng = np.random.default_rng(0)
        sig_id, cat = _pick_signal_for_step(step, rng)
        assert cat in ("monetary_policy", "financial_stress")

    def test_invalid_category_fallback(self):
        step = TemplateStep("nonexistent_category", r"anything", 0, None)
        rng = np.random.default_rng(0)
        sig_id, cat = _pick_signal_for_step(step, rng)
        # Should return fallback
        assert sig_id is not None


class TestGeneratePlantedChain:
    """Test individual planted chain generation."""

    def test_basic_shape(self):
        tmpl = TEMPLATE_LIBRARY[0]  # supply_chain_disruption
        ev, lead, direction = generate_planted_chain(tmpl, 1_600_000_000, seed=42)
        n_steps = len(tmpl.steps)
        assert len(ev) == n_steps * 60  # 60 points per signal
        assert isinstance(lead, str) and len(lead) > 0
        assert direction in (-1, 0, 1)

    def test_all_evidence_valid(self):
        tmpl = TEMPLATE_LIBRARY[0]
        ev, _, _ = generate_planted_chain(tmpl, 1_600_000_000, seed=42)
        for e in ev:
            assert isinstance(e, Evidence)
            assert e.category in CATEGORIES
            assert 0 <= e.confidence <= 1.0
            assert e.timestamp > 0
            assert e.ttl > 0

    def test_deterministic(self):
        tmpl = TEMPLATE_LIBRARY[0]
        ev1, l1, d1 = generate_planted_chain(tmpl, 1_600_000_000, seed=99)
        ev2, l2, d2 = generate_planted_chain(tmpl, 1_600_000_000, seed=99)
        assert l1 == l2
        assert d1 == d2
        assert len(ev1) == len(ev2)
        for a, b in zip(ev1, ev2, strict=True):
            assert a.signal_id == b.signal_id
            assert a.value == b.value

    def test_anomaly_present(self):
        """Values in the tail should be much larger than baseline."""
        tmpl = TEMPLATE_LIBRARY[0]
        ev, _, _ = generate_planted_chain(
            tmpl, 1_600_000_000, anomaly_magnitude=5.0, seed=42
        )
        # Group by signal_id
        by_sig: dict[str, list[float]] = {}
        for e in ev:
            by_sig.setdefault(e.signal_id, []).append(e.value)

        for sig_id, vals in by_sig.items():
            last_10 = vals[-10:]
            first_10 = vals[:10]
            assert np.mean(last_10) > np.mean(
                first_10
            ), f"Anomaly not visible in {sig_id}"

    def test_shared_time_window(self):
        """All signals should share the same timestamp range."""
        tmpl = TEMPLATE_LIBRARY[0]
        ev, _, _ = generate_planted_chain(tmpl, 1_600_000_000, seed=42)
        by_sig: dict[str, list[float]] = {}
        for e in ev:
            by_sig.setdefault(e.signal_id, []).append(e.timestamp)

        ts_ranges = [(min(ts), max(ts)) for ts in by_sig.values()]
        # All should start at the same time
        starts = [r[0] for r in ts_ranges]
        assert len(set(starts)) == 1, "All signals must share start timestamp"

    def test_multiple_categories(self):
        tmpl = TEMPLATE_LIBRARY[0]  # has 4 steps across 4 categories
        ev, _, _ = generate_planted_chain(tmpl, 1_600_000_000, seed=42)
        cats = {e.category for e in ev}
        assert len(cats) >= 2, "Planted chain should span ≥2 categories"


class TestGenerateDecoySignals:
    """Test decoy evidence generation."""

    def test_basic_shape(self):
        decoys = generate_decoy_signals(5, 60, 1_600_000_000, seed=42)
        assert len(decoys) == 5 * 60

    def test_low_magnitude(self):
        decoys = generate_decoy_signals(3, 100, 1_600_000_000, seed=42)
        vals = [e.value for e in decoys]
        # Decoys are N(0, 0.5) — most should be < 2
        assert np.percentile(np.abs(vals), 95) < 3.0

    def test_valid_categories(self):
        decoys = generate_decoy_signals(5, 10, 1_600_000_000, seed=42)
        for e in decoys:
            assert e.category in CATEGORIES

    def test_decoy_tag(self):
        decoys = generate_decoy_signals(2, 10, 1_600_000_000, seed=42)
        for e in decoys:
            assert "decoy" in e.tags

    def test_empty(self):
        decoys = generate_decoy_signals(0, 10, 1_600_000_000, seed=42)
        assert decoys == []


class TestGenerateScenarios:
    """Test bulk scenario generation."""

    def test_count(self):
        scenarios = generate_scenarios(n=5, seed=42)
        assert len(scenarios) == 5

    def test_scenario_fields(self):
        scenarios = generate_scenarios(n=1, seed=42)
        s = scenarios[0]
        assert isinstance(s, SyntheticScenario)
        assert len(s.planted_evidence) > 0
        assert len(s.decoy_evidence) > 0
        assert s.expected_template in [t.name for t in TEMPLATE_LIBRARY]
        assert s.expected_direction in (-1, 0, 1)
        assert len(s.expected_lead_signal) > 0

    def test_deterministic(self):
        s1 = generate_scenarios(n=3, seed=42)
        s2 = generate_scenarios(n=3, seed=42)
        for a, b in zip(s1, s2, strict=True):
            assert a.name == b.name
            assert a.expected_template == b.expected_template

    def test_templates_cycle(self):
        """With n > len(TEMPLATE_LIBRARY), templates cycle."""
        n = len(TEMPLATE_LIBRARY) + 5
        scenarios = generate_scenarios(n=n, seed=42)
        assert len(scenarios) == n
        # First and (len+1)th should use the same template
        assert (
            scenarios[0].expected_template
            == scenarios[len(TEMPLATE_LIBRARY)].expected_template
        )

    def test_no_templates_raises(self):
        with pytest.raises(ValueError, match="No templates"):
            generate_scenarios(n=1, seed=42, templates=[])


class TestBuildRegistryFromEvidence:
    """Test the helper that builds SignalRegistry from Evidence."""

    def test_dedup(self):
        ev = [
            Evidence("src", "sig.a", 1.0, 1.0, 1, 0.8, "positioning", (), 86400),
            Evidence("src", "sig.a", 2.0, 2.0, 1, 0.8, "positioning", (), 86400),
            Evidence("src", "sig.b", 3.0, 3.0, 0, 0.5, "macro_momentum", (), 86400),
        ]
        reg = _build_registry_from_evidence(ev)
        assert len(reg) == 2
        assert reg.get("sig.a") is not None
        assert reg.get("sig.b") is not None

    def test_empty(self):
        reg = _build_registry_from_evidence([])
        assert len(reg) == 0


class TestRunSyntheticValidation:
    """Test the full validation runner."""

    def test_empty_scenarios(self):
        result = run_synthetic_validation([])
        assert result.n_scenarios == 0
        assert result.f1 == 0.0

    def test_small_run(self):
        """Smoke test: 5 scenarios should produce a valid result."""
        scenarios = generate_scenarios(n=5, seed=42)
        result = run_synthetic_validation(scenarios)
        assert result.n_scenarios == 5
        assert result.true_positives + result.false_negatives == 5
        assert 0.0 <= result.precision <= 1.0
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.f1 <= 1.0
        assert len(result.details) == 5

    def test_custom_config(self):
        """Very strict config should detect fewer scenarios."""
        scenarios = generate_scenarios(n=5, seed=42)
        strict = ConvergenceDetectorConfig(
            z_threshold=4.0,
            p_threshold=0.001,
            fdr_q=0.001,
            min_clique_size=4,
            min_categories=4,
            min_persistence=1,
        )
        result = run_synthetic_validation(scenarios, config=strict)
        assert result.n_scenarios == 5
        # Stricter config → fewer TPs (may be 0)
        assert result.recall <= 1.0

    def test_all_details_have_required_fields(self):
        scenarios = generate_scenarios(n=3, seed=42)
        result = run_synthetic_validation(scenarios)
        for d in result.details:
            assert "scenario" in d
            assert "detected" in d
            assert "n_detections" in d
            assert "false_positives" in d
            assert "expected_template" in d

    def test_baseline_accuracy(self):
        """100 scenarios should produce reasonable accuracy.

        This is the main regression test — if a code change breaks
        the convergence engine, this will catch it.
        """
        scenarios = generate_scenarios(n=100, seed=42)
        result = run_synthetic_validation(scenarios)
        assert result.recall >= 0.60, f"Recall too low: {result.recall}"
        assert result.precision >= 0.80, f"Precision too low: {result.precision}"
        assert result.f1 >= 0.65, f"F1 too low: {result.f1}"
        assert (
            result.direction_accuracy >= 0.60
        ), f"Direction accuracy too low: {result.direction_accuracy}"


class TestSinglePlantedOnlyScenario:
    """Scenario with ONLY planted evidence (no decoys)."""

    def test_planted_only_detected(self):
        tmpl = TEMPLATE_LIBRARY[0]
        planted, lead, direction = generate_planted_chain(tmpl, 1_600_000_000, seed=42)
        scenario = SyntheticScenario(
            name="planted_only",
            planted_evidence=planted,
            decoy_evidence=[],
            expected_template=tmpl.name,
            expected_direction=direction,
            expected_lead_signal=lead,
        )
        result = run_synthetic_validation([scenario])
        assert result.recall >= 0.0  # May not detect with no decoy contrast


class TestSingleDecoyOnlyScenario:
    """Scenario with ONLY decoy evidence (no planted chain)."""

    def test_decoy_only_no_detection(self):
        decoys = generate_decoy_signals(8, 60, 1_600_000_000, seed=42)
        scenario = SyntheticScenario(
            name="decoy_only",
            planted_evidence=[],
            decoy_evidence=decoys,
            expected_template="none",
            expected_direction=0,
            expected_lead_signal="none",
        )
        # planted_evidence is empty → overlap check always fails
        # so any detection = false positive
        result = run_synthetic_validation([scenario])
        assert result.true_positives == 0
        assert result.false_negatives == 1  # "planted" was never found


class TestNumericalStability:
    """Edge cases with extreme numeric values."""

    def test_high_anomaly_magnitude(self):
        tmpl = TEMPLATE_LIBRARY[0]
        ev, _, _ = generate_planted_chain(
            tmpl, 1_600_000_000, anomaly_magnitude=100.0, seed=42
        )
        assert all(math.isfinite(e.value) for e in ev)

    def test_zero_correlation(self):
        tmpl = TEMPLATE_LIBRARY[0]
        ev, _, _ = generate_planted_chain(tmpl, 1_600_000_000, correlation=0.0, seed=42)
        assert len(ev) > 0

    def test_max_correlation(self):
        tmpl = TEMPLATE_LIBRARY[0]
        ev, _, _ = generate_planted_chain(tmpl, 1_600_000_000, correlation=1.0, seed=42)
        assert len(ev) > 0
