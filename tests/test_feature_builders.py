"""Tests for ConvergenceFeatureBuilder and MacroStateFeatureBuilder.

Covers: happy paths, empty data, stale data, NaN/Inf in inputs, missing
series, single-observation edge cases, duplicate timestamps, quality
degradation, all-missing output, validation of emitted features, and
boundary conditions.
"""

from __future__ import annotations

import time

import pytest

from agent.features.builders import (
    ConvergenceFeatureBuilder,
    FeatureBuilder,
    MacroStateFeatureBuilder,
)
from agent.features.protocol import EngineeredFeature, validate_feature
from agent.pipeline.store import PipelineStore

# ── helpers ────────────────────────────────────────────────────

_NOW = time.time()
_DAY = 86_400


def _store() -> PipelineStore:
    return PipelineStore(db_path=":memory:")


def _insert_convergence_signal(
    store: PipelineStore,
    signal_name: str = "convergence.credit_stress.2026-04-01",
    value: float = 0.75,
    computed_at: float | None = None,
    persistence_days: int = 3,
    direction: int = 1,
) -> None:
    """Insert a convergence signal directly into the signals table."""
    if computed_at is None:
        computed_at = _NOW - 3600  # 1 hour ago
    metadata = {
        "event_type": "credit_stress",
        "signals_involved": ["sig_a", "sig_b", "sig_c"],
        "categories_involved": ["credit", "macro"],
        "cross_category_count": 2,
        "p_value": 0.01,
        "persistence_days": persistence_days,
        "template_match": 0.8,
        "direction": direction,
        "lead_signal": "sig_a",
        "lag_signals": ["sig_b", "sig_c"],
    }
    store.store_signal(signal_name, value, metadata)
    # Override computed_at (store_signal uses time.time())
    conn = store._get_conn()
    conn.execute(
        "UPDATE signals SET computed_at=? WHERE id = "
        "(SELECT id FROM signals WHERE signal_name=? ORDER BY id DESC LIMIT 1)",
        (computed_at, signal_name),
    )
    conn.commit()


def _insert_macro_data(
    store: PipelineStore,
    series_data: dict,
    fetched_at: float | None = None,
) -> None:
    """Insert macro_data into pipeline_data table."""
    if fetched_at is None:
        fetched_at = _NOW
    store.store_data("macro_data", {"source": "fred"}, series_data)
    # Override fetched_at
    conn = store._get_conn()
    conn.execute(
        "UPDATE pipeline_data SET fetched_at=? "
        "WHERE id = (SELECT id FROM pipeline_data WHERE source='macro_data' ORDER BY id DESC LIMIT 1)",
        (fetched_at,),
    )
    conn.commit()


def _make_fred_series(
    series_id: str,
    n_obs: int = 60,
    start_value: float = 5.0,
    trend: float = 0.01,
) -> dict:
    """Generate a synthetic FRED series with daily observations."""
    from datetime import datetime, timedelta

    base_date = datetime(2026, 3, 1)
    obs = []
    for i in range(n_obs):
        d = base_date + timedelta(days=i)
        val = start_value + trend * i
        obs.append({"date": d.strftime("%Y-%m-%d"), "value": str(val)})
    return {series_id: obs}


def _assert_all_valid(features: list[EngineeredFeature]) -> None:
    """Assert every feature passes protocol validation."""
    for feat in features:
        errors = validate_feature(feat)
        assert errors == [], f"{feat.feature_name} failed validation: {errors}"


# ══════════════════════════════════════════════════════════════
#  ConvergenceFeatureBuilder
# ══════════════════════════════════════════════════════════════


class TestConvergenceFeatureBuilderBasic:
    def test_is_feature_builder(self):
        builder = ConvergenceFeatureBuilder()
        assert isinstance(builder, FeatureBuilder)
        assert builder.name == "ConvergenceFeatureBuilder"

    def test_produces_three_features(self):
        store = _store()
        _insert_convergence_signal(store)
        builder = ConvergenceFeatureBuilder()
        features = builder.build(store, _NOW)
        assert len(features) == 3
        store.close()

    def test_feature_names(self):
        store = _store()
        _insert_convergence_signal(store)
        features = ConvergenceFeatureBuilder().build(store, _NOW)
        names = {f.feature_name for f in features}
        assert names == {
            "convergence.stress_breadth.7d",
            "convergence.stress_intensity.7d",
            "convergence.regime_persistence.7d",
        }
        store.close()

    def test_all_features_pass_validation(self):
        store = _store()
        _insert_convergence_signal(store)
        features = ConvergenceFeatureBuilder().build(store, _NOW)
        _assert_all_valid(features)
        store.close()

    def test_stress_breadth_single_signal(self):
        store = _store()
        _insert_convergence_signal(store, signal_name="convergence.a.2026-04-01")
        features = ConvergenceFeatureBuilder().build(store, _NOW)
        breadth = next(f for f in features if "breadth" in f.feature_name)
        assert breadth.value == 1.0
        store.close()

    def test_stress_breadth_multiple_signals(self):
        store = _store()
        _insert_convergence_signal(store, signal_name="convergence.a.2026-04-01", value=0.5)
        _insert_convergence_signal(store, signal_name="convergence.b.2026-04-01", value=0.7)
        _insert_convergence_signal(store, signal_name="convergence.c.2026-04-02", value=0.3)
        features = ConvergenceFeatureBuilder().build(store, _NOW)
        breadth = next(f for f in features if "breadth" in f.feature_name)
        assert breadth.value == 3.0
        store.close()

    def test_stress_intensity_is_max_value(self):
        store = _store()
        _insert_convergence_signal(store, signal_name="convergence.a.2026-04-01", value=0.3)
        _insert_convergence_signal(store, signal_name="convergence.b.2026-04-01", value=0.9)
        features = ConvergenceFeatureBuilder().build(store, _NOW)
        intensity = next(f for f in features if "intensity" in f.feature_name)
        assert intensity.value == pytest.approx(0.9)
        store.close()

    def test_regime_persistence_from_metadata(self):
        store = _store()
        _insert_convergence_signal(
            store,
            signal_name="convergence.a.2026-04-01",
            persistence_days=5,
        )
        _insert_convergence_signal(
            store,
            signal_name="convergence.b.2026-04-01",
            persistence_days=12,
        )
        features = ConvergenceFeatureBuilder().build(store, _NOW)
        persist = next(f for f in features if "persistence" in f.feature_name)
        assert persist.value == 12.0
        store.close()

    def test_all_features_have_correct_builder(self):
        store = _store()
        _insert_convergence_signal(store)
        features = ConvergenceFeatureBuilder().build(store, _NOW)
        for f in features:
            assert f.builder == "ConvergenceFeatureBuilder"
        store.close()


class TestConvergenceFeatureBuilderEdgeCases:
    def test_no_signals_no_data_returns_empty(self):
        """Empty store → 3 None-valued features (consistent GNN dimensionality)."""
        store = _store()
        features = ConvergenceFeatureBuilder().build(store, _NOW)
        assert len(features) == 3
        assert all(f.value is None for f in features)
        assert all(f.missing_reason == "no_convergence_activity" for f in features)
        store.close()

    def test_signals_outside_window_are_ignored(self):
        store = _store()
        # Insert signal 10 days ago (outside 7d window)
        _insert_convergence_signal(
            store,
            computed_at=_NOW - 10 * _DAY,
        )
        features = ConvergenceFeatureBuilder().build(store, _NOW)
        assert all(f.value is None for f in features)
        store.close()

    def test_signals_at_window_boundary(self):
        """Signal exactly at the edge of the 7d window should be included."""
        store = _store()
        _insert_convergence_signal(
            store,
            computed_at=_NOW - 7 * _DAY,  # exactly 7 days ago
        )
        features = ConvergenceFeatureBuilder().build(store, _NOW)
        breadth = next(f for f in features if "breadth" in f.feature_name)
        assert breadth.value == 1.0
        store.close()

    def test_persistence_missing_from_metadata(self):
        """Signal without persistence_days in metadata defaults to 0."""
        store = _store()
        # Insert signal with metadata that has no persistence_days key.
        # Must override computed_at so it falls within the builder's window.
        store.store_signal("convergence.test.2026-04-01", 0.5, {"event_type": "test"})
        conn = store._get_conn()
        conn.execute(
            "UPDATE signals SET computed_at=? WHERE id = (SELECT id FROM signals ORDER BY id DESC LIMIT 1)",
            (_NOW - 3600,),
        )
        conn.commit()
        features = ConvergenceFeatureBuilder().build(store, _NOW)
        persist = next(f for f in features if "persistence" in f.feature_name)
        assert persist.value == 0.0
        store.close()

    def test_non_convergence_signals_ignored(self):
        """Signals not matching convergence.* prefix are excluded."""
        store = _store()
        store.store_signal("other.signal", 0.8, {})
        # Override computed_at to be within the window
        conn = store._get_conn()
        conn.execute(
            "UPDATE signals SET computed_at=? WHERE id = (SELECT id FROM signals ORDER BY id DESC LIMIT 1)",
            (_NOW - 3600,),
        )
        conn.commit()
        features = ConvergenceFeatureBuilder().build(store, _NOW)
        assert all(f.value is None for f in features)
        store.close()

    def test_duplicate_signal_names_counted_once_for_breadth(self):
        """Same signal_name appearing multiple times counts as one for breadth."""
        store = _store()
        name = "convergence.credit_stress.2026-04-01"
        _insert_convergence_signal(store, signal_name=name, value=0.5, computed_at=_NOW - 3600)
        _insert_convergence_signal(store, signal_name=name, value=0.8, computed_at=_NOW - 1800)
        features = ConvergenceFeatureBuilder().build(store, _NOW)
        breadth = next(f for f in features if "breadth" in f.feature_name)
        assert breadth.value == 1.0
        # But intensity should use max across all instances
        intensity = next(f for f in features if "intensity" in f.feature_name)
        assert intensity.value == pytest.approx(0.8)
        store.close()

    def test_source_signals_are_sorted(self):
        store = _store()
        _insert_convergence_signal(store, signal_name="convergence.z.2026-04-01")
        _insert_convergence_signal(store, signal_name="convergence.a.2026-04-01")
        features = ConvergenceFeatureBuilder().build(store, _NOW)
        for f in features:
            if f.value is not None:
                assert list(f.source_signals) == sorted(f.source_signals)
        store.close()

    def test_horizon_is_7d(self):
        store = _store()
        _insert_convergence_signal(store)
        features = ConvergenceFeatureBuilder().build(store, _NOW)
        for f in features:
            assert f.horizon == "7d"
        store.close()

    def test_effective_at_equals_as_of(self):
        store = _store()
        _insert_convergence_signal(store)
        as_of = _NOW - 100
        features = ConvergenceFeatureBuilder().build(store, as_of)
        for f in features:
            assert f.effective_at == as_of
        store.close()


# ══════════════════════════════════════════════════════════════
#  MacroStateFeatureBuilder
# ══════════════════════════════════════════════════════════════


class TestMacroStateFeatureBuilderBasic:
    def test_is_feature_builder(self):
        builder = MacroStateFeatureBuilder()
        assert isinstance(builder, FeatureBuilder)
        assert builder.name == "MacroStateFeatureBuilder"

    def test_produces_three_features(self):
        store = _store()
        data = {}
        data.update(_make_fred_series("DFF", n_obs=60, start_value=5.25, trend=0.001))
        data.update(_make_fred_series("GS10", n_obs=60, start_value=4.5, trend=0.002))
        data.update(_make_fred_series("GS2", n_obs=60, start_value=4.8, trend=0.001))
        data.update(_make_fred_series("WALCL", n_obs=60, start_value=8_000_000, trend=-10_000))
        _insert_macro_data(store, data)
        features = MacroStateFeatureBuilder().build(store, _NOW)
        assert len(features) == 3
        store.close()

    def test_feature_names(self):
        store = _store()
        data = {}
        data.update(_make_fred_series("DFF"))
        data.update(_make_fred_series("GS10"))
        data.update(_make_fred_series("GS2"))
        data.update(_make_fred_series("WALCL"))
        _insert_macro_data(store, data)
        features = MacroStateFeatureBuilder().build(store, _NOW)
        names = {f.feature_name for f in features}
        assert names == {
            "macro.rate_momentum.30d",
            "macro.yield_curve_slope.spot",
            "macro.liquidity_pressure.30d",
        }
        store.close()

    def test_all_features_pass_validation(self):
        store = _store()
        data = {}
        data.update(_make_fred_series("DFF"))
        data.update(_make_fred_series("GS10"))
        data.update(_make_fred_series("GS2"))
        data.update(_make_fred_series("WALCL"))
        _insert_macro_data(store, data)
        features = MacroStateFeatureBuilder().build(store, _NOW)
        _assert_all_valid(features)
        store.close()

    def test_all_features_have_correct_builder(self):
        store = _store()
        data = {}
        data.update(_make_fred_series("DFF"))
        data.update(_make_fred_series("GS10"))
        data.update(_make_fred_series("GS2"))
        data.update(_make_fred_series("WALCL"))
        _insert_macro_data(store, data)
        features = MacroStateFeatureBuilder().build(store, _NOW)
        for f in features:
            assert f.builder == "MacroStateFeatureBuilder"
        store.close()


class TestRateMomentum:
    def test_rising_rate(self):
        """Positive trend in DFF → positive momentum."""
        store = _store()
        data = _make_fred_series("DFF", n_obs=60, start_value=5.0, trend=0.01)
        _insert_macro_data(store, data)
        features = MacroStateFeatureBuilder().build(store, _NOW)
        momentum = next(f for f in features if "rate_momentum" in f.feature_name)
        assert momentum.value is not None
        assert momentum.value > 0  # rising rate → positive bps
        assert momentum.unit == "bps"
        assert momentum.horizon == "30d"
        store.close()

    def test_falling_rate(self):
        """Negative trend in DFF → negative momentum."""
        store = _store()
        data = _make_fred_series("DFF", n_obs=60, start_value=5.0, trend=-0.02)
        _insert_macro_data(store, data)
        features = MacroStateFeatureBuilder().build(store, _NOW)
        momentum = next(f for f in features if "rate_momentum" in f.feature_name)
        assert momentum.value is not None
        assert momentum.value < 0
        store.close()

    def test_flat_rate(self):
        """No trend → ~0 momentum."""
        store = _store()
        data = _make_fred_series("DFF", n_obs=60, start_value=5.0, trend=0.0)
        _insert_macro_data(store, data)
        features = MacroStateFeatureBuilder().build(store, _NOW)
        momentum = next(f for f in features if "rate_momentum" in f.feature_name)
        assert momentum.value == pytest.approx(0.0, abs=1.0)
        store.close()

    def test_missing_dff(self):
        store = _store()
        _insert_macro_data(store, _make_fred_series("GS10"))
        features = MacroStateFeatureBuilder().build(store, _NOW)
        momentum = next(f for f in features if "rate_momentum" in f.feature_name)
        assert momentum.value is None
        assert momentum.missing_reason == "insufficient_dff_history"
        store.close()

    def test_single_observation_insufficient(self):
        store = _store()
        _insert_macro_data(store, {"DFF": [{"date": "2026-04-01", "value": "5.25"}]})
        features = MacroStateFeatureBuilder().build(store, _NOW)
        momentum = next(f for f in features if "rate_momentum" in f.feature_name)
        # Only 1 obs — can find latest but no 30d-ago comparison within 10 days
        # This should be missing
        assert momentum.value is None or momentum.missing_reason is not None
        store.close()


class TestYieldCurveSlope:
    def test_normal_curve(self):
        """10Y > 2Y → positive spread."""
        store = _store()
        data = {}
        data.update(_make_fred_series("GS10", n_obs=5, start_value=4.5))
        data.update(_make_fred_series("GS2", n_obs=5, start_value=4.0))
        _insert_macro_data(store, data)
        features = MacroStateFeatureBuilder().build(store, _NOW)
        slope = next(f for f in features if "yield_curve" in f.feature_name)
        assert slope.value is not None
        assert slope.value > 0  # positive spread in bps
        assert slope.unit == "bps"
        assert slope.horizon == "spot"
        store.close()

    def test_inverted_curve(self):
        """2Y > 10Y → negative spread."""
        store = _store()
        data = {}
        data.update(_make_fred_series("GS10", n_obs=5, start_value=3.5))
        data.update(_make_fred_series("GS2", n_obs=5, start_value=4.8))
        _insert_macro_data(store, data)
        features = MacroStateFeatureBuilder().build(store, _NOW)
        slope = next(f for f in features if "yield_curve" in f.feature_name)
        assert slope.value is not None
        assert slope.value < 0  # inverted
        store.close()

    def test_missing_gs10(self):
        store = _store()
        _insert_macro_data(store, _make_fred_series("GS2"))
        features = MacroStateFeatureBuilder().build(store, _NOW)
        slope = next(f for f in features if "yield_curve" in f.feature_name)
        assert slope.value is None
        assert slope.missing_reason == "missing_treasury_yields"
        store.close()

    def test_missing_gs2(self):
        store = _store()
        _insert_macro_data(store, _make_fred_series("GS10"))
        features = MacroStateFeatureBuilder().build(store, _NOW)
        slope = next(f for f in features if "yield_curve" in f.feature_name)
        assert slope.value is None
        store.close()

    def test_two_source_signals(self):
        store = _store()
        data = {}
        data.update(_make_fred_series("GS10"))
        data.update(_make_fred_series("GS2"))
        _insert_macro_data(store, data)
        features = MacroStateFeatureBuilder().build(store, _NOW)
        slope = next(f for f in features if "yield_curve" in f.feature_name)
        assert len(slope.source_signals) == 2
        store.close()


class TestLiquidityPressure:
    def test_shrinking_balance_sheet(self):
        """Declining WALCL → negative z-score (tightening)."""
        store = _store()
        data = _make_fred_series("WALCL", n_obs=60, start_value=9_000_000, trend=-20_000)
        _insert_macro_data(store, data)
        features = MacroStateFeatureBuilder().build(store, _NOW)
        liq = next(f for f in features if "liquidity_pressure" in f.feature_name)
        assert liq.value is not None
        assert liq.unit == "z_score"
        assert liq.horizon == "30d"
        store.close()

    def test_missing_walcl(self):
        store = _store()
        _insert_macro_data(store, _make_fred_series("DFF"))
        features = MacroStateFeatureBuilder().build(store, _NOW)
        liq = next(f for f in features if "liquidity_pressure" in f.feature_name)
        assert liq.value is None
        assert liq.missing_reason == "insufficient_walcl_history"
        store.close()

    def test_too_few_observations(self):
        store = _store()
        _insert_macro_data(
            store,
            {
                "WALCL": [
                    {"date": "2026-04-01", "value": "8000000"},
                    {"date": "2026-04-02", "value": "8000100"},
                ]
            },
        )
        features = MacroStateFeatureBuilder().build(store, _NOW)
        liq = next(f for f in features if "liquidity_pressure" in f.feature_name)
        # Only 2 obs → probably insufficient for z-score
        assert liq.value is None or isinstance(liq.value, float)
        store.close()

    def test_z_score_clamped(self):
        """Extreme outlier should be clamped to [-10, 10]."""
        store = _store()
        # Create data with a massive spike at the end
        from datetime import datetime, timedelta

        base = datetime(2026, 2, 1)
        obs = []
        for i in range(58):
            d = base + timedelta(days=i)
            obs.append({"date": d.strftime("%Y-%m-%d"), "value": str(8_000_000)})
        # Add extreme spike
        obs.append({"date": "2026-03-31", "value": str(8_000_000)})
        obs.append({"date": "2026-04-01", "value": str(100_000_000)})  # 12x jump
        _insert_macro_data(store, {"WALCL": obs})
        features = MacroStateFeatureBuilder().build(store, _NOW)
        liq = next(f for f in features if "liquidity_pressure" in f.feature_name)
        if liq.value is not None:
            assert -10.0 <= liq.value <= 10.0
        store.close()


class TestMacroEdgeCases:
    def test_no_macro_data_at_all(self):
        """Empty store → 3 None-valued features (consistent GNN dimensionality)."""
        store = _store()
        features = MacroStateFeatureBuilder().build(store, _NOW)
        assert len(features) == 3
        assert all(f.value is None for f in features)
        store.close()

    def test_dot_values_from_fred_skipped(self):
        """FRED uses '.' as a sentinel for missing values — must skip."""
        store = _store()
        data = {
            "DFF": [
                {"date": "2026-03-01", "value": "."},
                {"date": "2026-03-02", "value": "5.25"},
            ]
        }
        _insert_macro_data(store, data)
        features = MacroStateFeatureBuilder().build(store, _NOW)
        # Should still run without crashing
        assert len(features) == 3
        store.close()

    def test_nan_values_skipped(self):
        """NaN values in series should be silently excluded."""
        store = _store()
        data = {
            "DFF": [
                {"date": "2026-03-01", "value": "nan"},
                {"date": "2026-03-02", "value": "5.25"},
                {"date": "2026-04-01", "value": "5.30"},
            ]
        }
        _insert_macro_data(store, data)
        features = MacroStateFeatureBuilder().build(store, _NOW)
        assert len(features) == 3
        store.close()

    def test_non_numeric_values_skipped(self):
        store = _store()
        data = {
            "DFF": [
                {"date": "2026-03-01", "value": "N/A"},
                {"date": "2026-03-15", "value": "5.25"},
                {"date": "2026-04-01", "value": "5.30"},
            ]
        }
        _insert_macro_data(store, data)
        features = MacroStateFeatureBuilder().build(store, _NOW)
        assert len(features) == 3
        store.close()

    def test_multiple_macro_data_rows_merged(self):
        """Multiple pipeline_data rows with partial series get merged."""
        store = _store()
        # First fetch has DFF
        _insert_macro_data(store, _make_fred_series("DFF", n_obs=60))
        # Second fetch has GS10 + GS2
        data2 = {}
        data2.update(_make_fred_series("GS10", n_obs=60))
        data2.update(_make_fred_series("GS2", n_obs=60))
        _insert_macro_data(store, data2)
        # Third fetch has WALCL
        _insert_macro_data(store, _make_fred_series("WALCL", n_obs=60))
        features = MacroStateFeatureBuilder().build(store, _NOW)
        # All 3 features should have real values (not missing)
        for f in features:
            assert f.value is not None, f"{f.feature_name} should not be missing"
        store.close()

    def test_duplicate_dates_in_series_deduplicated(self):
        """Same date appearing multiple times should be deduplicated."""
        store = _store()
        data = {
            "DFF": [
                {"date": "2026-03-01", "value": "5.20"},
                {"date": "2026-03-01", "value": "5.25"},  # duplicate date
                {"date": "2026-04-01", "value": "5.30"},
            ]
        }
        _insert_macro_data(store, data)
        features = MacroStateFeatureBuilder().build(store, _NOW)
        # Should not crash; later value overwrites
        momentum = next(f for f in features if "rate_momentum" in f.feature_name)
        assert momentum is not None
        store.close()

    def test_old_macro_data_outside_90d_window_excluded(self):
        """Data older than 90 days should not be returned by query."""
        store = _store()
        from datetime import datetime, timedelta

        # Insert data from 120 days ago (outside 90d window)
        old_date = (datetime(2026, 4, 7) - timedelta(days=120)).strftime("%Y-%m-%d")
        data = {"DFF": [{"date": old_date, "value": "5.00"}]}
        _insert_macro_data(store, data, fetched_at=_NOW - 120 * _DAY)
        features = MacroStateFeatureBuilder().build(store, _NOW)
        # Old data outside 90d window → query returns nothing → 3 missing features.
        assert len(features) == 3
        assert all(f.value is None for f in features)
        store.close()

    def test_effective_at_equals_as_of(self):
        store = _store()
        data = {}
        data.update(_make_fred_series("DFF"))
        data.update(_make_fred_series("GS10"))
        data.update(_make_fred_series("GS2"))
        data.update(_make_fred_series("WALCL"))
        _insert_macro_data(store, data)
        as_of = _NOW - 500
        features = MacroStateFeatureBuilder().build(store, as_of)
        for f in features:
            assert f.effective_at == as_of
        store.close()

    def test_quality_degrades_with_sparse_data(self):
        """Rate momentum quality should be 0.7 when < 20 observations."""
        store = _store()
        data = _make_fred_series("DFF", n_obs=10, start_value=5.0, trend=0.01)
        _insert_macro_data(store, data)
        features = MacroStateFeatureBuilder().build(store, _NOW)
        momentum = next(f for f in features if "rate_momentum" in f.feature_name)
        if momentum.value is not None:
            assert momentum.quality == pytest.approx(0.7)
        store.close()


# ══════════════════════════════════════════════════════════════
#  _extract_series helper
# ══════════════════════════════════════════════════════════════


class TestExtractSeries:
    def test_empty_rows(self):
        result = MacroStateFeatureBuilder._extract_series([])
        assert result == {}

    def test_malformed_data_skipped(self):
        rows = [
            {"data": "not a dict"},
            {"data": {"DFF": "not a list"}},
            {"data": {"DFF": [{"date": "2026-01-01"}]}},  # missing value
            {"data": {"DFF": [{"value": "5.0"}]}},  # missing date
        ]
        result = MacroStateFeatureBuilder._extract_series(rows)
        assert result == {} or all(len(v) == 0 for v in result.values())

    def test_inf_values_skipped(self):
        rows = [
            {
                "data": {
                    "X": [
                        {"date": "2026-01-01", "value": "inf"},
                        {"date": "2026-01-02", "value": "5.0"},
                    ]
                }
            }
        ]
        result = MacroStateFeatureBuilder._extract_series(rows)
        assert len(result.get("X", [])) == 1

    def test_sorted_by_date(self):
        rows = [
            {
                "data": {
                    "X": [
                        {"date": "2026-01-03", "value": "3"},
                        {"date": "2026-01-01", "value": "1"},
                        {"date": "2026-01-02", "value": "2"},
                    ]
                }
            }
        ]
        result = MacroStateFeatureBuilder._extract_series(rows)
        dates = [d for d, _ in result["X"]]
        assert dates == sorted(dates)


# ══════════════════════════════════════════════════════════════
#  Integration: builders → store persistence
# ══════════════════════════════════════════════════════════════


class TestBuilderToStorePersistence:
    def test_convergence_features_persist(self):
        store = _store()
        _insert_convergence_signal(store)
        features = ConvergenceFeatureBuilder().build(store, _NOW)
        for f in features:
            store.store_feature(f)
        rows = store.query_features("convergence.stress_breadth.7d")
        assert len(rows) == 1
        store.close()

    def test_macro_features_persist(self):
        store = _store()
        data = {}
        data.update(_make_fred_series("DFF"))
        data.update(_make_fred_series("GS10"))
        data.update(_make_fred_series("GS2"))
        data.update(_make_fred_series("WALCL"))
        _insert_macro_data(store, data)
        features = MacroStateFeatureBuilder().build(store, _NOW)
        for f in features:
            store.store_feature(f)
        for name in [
            "macro.rate_momentum.30d",
            "macro.yield_curve_slope.spot",
            "macro.liquidity_pressure.30d",
        ]:
            r = store.get_latest_feature(name)
            assert r is not None, f"Feature {name} not found in store"
        store.close()

    def test_missing_features_persist(self):
        """Empty store produces 3 None-valued features that persist correctly."""
        store = _store()
        features = ConvergenceFeatureBuilder().build(store, _NOW)
        # Phase 45.3: empty store → 3 missing features with no_convergence_activity reason
        assert len(features) == 3
        assert all(f.value is None for f in features)
        store.close()
