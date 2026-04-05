"""Tests for convergence signal emission + pipeline store integration (Phase 7c-D.3).

Covers: ConvergenceSignal construction + to_metadata_dict, format_signal_name
(valid/invalid), emit_signals (round-trip store→query→compare, empty list,
store failure), from_detection_result factory.
"""

from __future__ import annotations

import json

import pytest

from agent.convergence.detector import DetectionResult
from agent.convergence.graph import ConvergenceClique
from agent.convergence.signals import (
    ConvergenceSignal,
    emit_signals,
    format_signal_name,
    from_detection_result,
)
from agent.pipeline.store import PipelineStore

# ── Helpers ────────────────────────────────────────────────────

_TS = 1_700_000_000.0


def _make_store() -> PipelineStore:
    return PipelineStore(":memory:")


def _sample_signal(**overrides) -> ConvergenceSignal:
    """Construct a ConvergenceSignal with sensible defaults."""
    defaults = dict(
        signal_name="convergence.supply_chain_disruption.2025-11-15",
        computed_at=_TS,
        value=0.72,
        event_type="supply_chain_disruption",
        signals_involved=["cftc.crude.mm", "ais.baltic.count", "pmi.de.mfg"],
        categories_involved=["positioning", "physical_flow", "macro_momentum"],
        cross_category_count=3,
        p_value=0.003,
        persistence_days=2,
        template_match=0.85,
        direction=1,
        lead_signal="cftc.crude.mm",
        lag_signals=["ais.baltic.count", "pmi.de.mfg"],
    )
    defaults.update(overrides)
    return ConvergenceSignal(**defaults)


def _sample_detection_result(**overrides) -> DetectionResult:
    """Build a DetectionResult for factory tests."""
    clique = ConvergenceClique(
        signals=["sig_a", "sig_b", "sig_c"],
        categories=["positioning", "macro_momentum"],
        edges=[("sig_a", "sig_b", 1.0), ("sig_a", "sig_c", 0.8)],
        score=0.6,
        p_values=[0.01, 0.02],
    )
    defaults = dict(
        clique=clique,
        event_type="supply_chain_disruption",
        template_match=0.75,
        boosted_score=0.825,
        lead_signal="sig_a",
        lag_signals=["sig_b", "sig_c"],
        template_result=None,
    )
    defaults.update(overrides)
    return DetectionResult(**defaults)


# ═══════════════════════════════════════════════════════════════
# format_signal_name
# ═══════════════════════════════════════════════════════════════


class TestFormatSignalName:
    def test_basic(self):
        name = format_signal_name("supply_chain_disruption", "2025-11-15")
        assert name == "convergence.supply_chain_disruption.2025-11-15"

    def test_unknown_pattern(self):
        name = format_signal_name("unknown_pattern", "2025-01-01")
        assert name == "convergence.unknown_pattern.2025-01-01"

    def test_rejects_spaces(self):
        with pytest.raises(ValueError, match="alphanumeric"):
            format_signal_name("supply chain", "2025-01-01")

    def test_rejects_dots(self):
        with pytest.raises(ValueError, match="alphanumeric"):
            format_signal_name("supply.chain", "2025-01-01")

    def test_rejects_slashes(self):
        with pytest.raises(ValueError, match="alphanumeric"):
            format_signal_name("../../etc/passwd", "2025-01-01")

    def test_rejects_empty_event_type(self):
        with pytest.raises(ValueError):
            format_signal_name("", "2025-01-01")

    def test_rejects_special_chars_in_date(self):
        with pytest.raises(ValueError, match="unsafe"):
            format_signal_name("test", "2025/01/01")

    def test_allows_underscores(self):
        name = format_signal_name("credit_stress_cascade", "2025-12-31")
        assert "credit_stress_cascade" in name


# ═══════════════════════════════════════════════════════════════
# ConvergenceSignal construction + to_metadata_dict
# ═══════════════════════════════════════════════════════════════


class TestConvergenceSignal:
    def test_construction(self):
        sig = _sample_signal()
        assert sig.value == 0.72
        assert sig.cross_category_count == 3
        assert sig.direction == 1

    def test_to_metadata_dict_keys(self):
        sig = _sample_signal()
        meta = sig.to_metadata_dict()
        expected_keys = {
            "event_type",
            "signals_involved",
            "categories_involved",
            "cross_category_count",
            "p_value",
            "persistence_days",
            "template_match",
            "direction",
            "lead_signal",
            "lag_signals",
        }
        assert set(meta.keys()) == expected_keys

    def test_metadata_is_json_serializable(self):
        sig = _sample_signal()
        meta = sig.to_metadata_dict()
        # Should not raise
        serialized = json.dumps(meta)
        deserialized = json.loads(serialized)
        assert deserialized["event_type"] == "supply_chain_disruption"
        assert deserialized["p_value"] == 0.003
        assert deserialized["signals_involved"] == sig.signals_involved

    def test_metadata_values_match(self):
        sig = _sample_signal()
        meta = sig.to_metadata_dict()
        assert meta["event_type"] == sig.event_type
        assert meta["signals_involved"] == sig.signals_involved
        assert meta["categories_involved"] == sig.categories_involved
        assert meta["cross_category_count"] == sig.cross_category_count
        assert meta["p_value"] == sig.p_value
        assert meta["persistence_days"] == sig.persistence_days
        assert meta["template_match"] == sig.template_match
        assert meta["direction"] == sig.direction
        assert meta["lead_signal"] == sig.lead_signal
        assert meta["lag_signals"] == sig.lag_signals

    def test_default_lag_signals(self):
        sig = ConvergenceSignal(
            signal_name="convergence.test.2025-01-01",
            computed_at=_TS,
            value=0.5,
            event_type="test",
            signals_involved=["a"],
            categories_involved=["positioning"],
            cross_category_count=1,
            p_value=0.05,
            persistence_days=1,
            template_match=0.0,
            direction=1,
            lead_signal="a",
        )
        assert sig.lag_signals == []


# ═══════════════════════════════════════════════════════════════
# emit_signals — round-trip pipeline store integration
# ═══════════════════════════════════════════════════════════════


class TestEmitSignals:
    def test_emit_single(self):
        store = _make_store()
        sig = _sample_signal()
        count = emit_signals([sig], store)
        assert count == 1

    def test_emit_empty(self):
        store = _make_store()
        count = emit_signals([], store)
        assert count == 0

    def test_emit_multiple(self):
        store = _make_store()
        sigs = [
            _sample_signal(signal_name="convergence.a.2025-01-01", value=0.5),
            _sample_signal(signal_name="convergence.b.2025-01-02", value=0.8),
            _sample_signal(signal_name="convergence.c.2025-01-03", value=0.3),
        ]
        count = emit_signals(sigs, store)
        assert count == 3

    def test_round_trip_query(self):
        """Store → query → verify fields match."""
        store = _make_store()
        sig = _sample_signal()
        emit_signals([sig], store)

        rows = store.query_signals(sig.signal_name)
        assert len(rows) == 1

        row = rows[0]
        assert row["signal_name"] == sig.signal_name
        assert row["value"] == sig.value

        meta = row.get("metadata")
        assert meta is not None
        assert meta["event_type"] == "supply_chain_disruption"
        assert meta["signals_involved"] == sig.signals_involved
        assert meta["p_value"] == sig.p_value
        assert meta["template_match"] == sig.template_match
        assert meta["direction"] == sig.direction
        assert meta["lead_signal"] == sig.lead_signal
        assert meta["lag_signals"] == sig.lag_signals

    def test_metadata_round_trip_categories(self):
        store = _make_store()
        sig = _sample_signal()
        emit_signals([sig], store)

        rows = store.query_signals(sig.signal_name)
        meta = rows[0]["metadata"]
        assert meta["categories_involved"] == sig.categories_involved
        assert meta["cross_category_count"] == sig.cross_category_count
        assert meta["persistence_days"] == sig.persistence_days


# ═══════════════════════════════════════════════════════════════
# from_detection_result factory
# ═══════════════════════════════════════════════════════════════


class TestFromDetectionResult:
    def test_basic_conversion(self):
        dr = _sample_detection_result()
        sig = from_detection_result(dr, persistence_count=3, as_of=_TS)

        assert sig.signal_name.startswith("convergence.supply_chain_disruption.")
        assert sig.computed_at == _TS
        assert sig.value == dr.boosted_score
        assert sig.event_type == "supply_chain_disruption"
        assert sig.signals_involved == ["sig_a", "sig_b", "sig_c"]
        assert set(sig.categories_involved) == {"macro_momentum", "positioning"}
        assert sig.cross_category_count == 2
        assert sig.persistence_days == 3
        assert sig.template_match == 0.75
        assert sig.lead_signal == "sig_a"
        assert sig.lag_signals == ["sig_b", "sig_c"]

    def test_unknown_pattern(self):
        dr = _sample_detection_result(event_type="unknown_pattern")
        sig = from_detection_result(dr, as_of=_TS)
        assert sig.event_type == "unknown_pattern"
        assert "unknown_pattern" in sig.signal_name

    def test_no_lead_signal(self):
        dr = _sample_detection_result(lead_signal=None)
        sig = from_detection_result(dr, as_of=_TS)
        assert sig.lead_signal == ""

    def test_empty_lag_signals(self):
        dr = _sample_detection_result(lag_signals=[])
        sig = from_detection_result(dr, as_of=_TS)
        assert sig.lag_signals == []

    def test_default_as_of(self):
        """When as_of is None, uses current time."""
        dr = _sample_detection_result()
        sig = from_detection_result(dr)
        assert sig.computed_at > 0

    def test_full_round_trip(self):
        """DetectionResult → ConvergenceSignal → store → query → compare."""
        store = _make_store()
        dr = _sample_detection_result()
        sig = from_detection_result(dr, persistence_count=2, as_of=_TS)

        emit_signals([sig], store)
        rows = store.query_signals(sig.signal_name)
        assert len(rows) == 1

        meta = rows[0]["metadata"]
        assert meta["event_type"] == "supply_chain_disruption"
        assert meta["template_match"] == 0.75
        assert meta["persistence_days"] == 2
