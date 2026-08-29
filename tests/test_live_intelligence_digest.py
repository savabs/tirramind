"""Tests for the live intelligence digest — real math on real stored data."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agent.pipeline.store import PipelineStore
from scripts.live_intelligence_digest import (
    _SCORABLE,
    _changepoint_flag,
    _extract_series_multi,
    _zscore_anomaly,
    build_digest,
)


@pytest.fixture
def store(tmp_path):
    import time

    s = PipelineStore(str(tmp_path / "pipeline.db"))
    cur = s._conn.cursor()
    # seed a real series: 50 pts flat, then a spike at the end
    for i in range(50):
        val = 100.0 + (50.0 if i >= 45 else 0.0)  # flat then spike
        cur.execute(
            "insert into entity_observations "
            "(entity_id, source_tool, observed_at, ingested_at, "
            " observation_type, depth_level, value_json) "
            "values (?, 'sovereign_debt', ?, ?, 'sovereign_yield', 1, ?)",
            ("test_entity", 1000 + i * 100, time.time(), json.dumps({"yield_pct": val})),
        )
    s._conn.commit()
    return s


def test_zscore_flags_spike():
    vals = [100.0] * 45 + [150.0, 155.0, 160.0, 165.0, 170.0]
    z = _zscore_anomaly(vals)
    assert z is not None
    assert z > 2.0  # spike is anomalous vs prior


def test_zscore_flat_returns_none():
    assert _zscore_anomaly([100.0] * 50) is None  # no variance → no anomaly
    assert _zscore_anomaly([1.0, 2.0]) is None  # too short


def test_changepoint_detected_on_step():
    import numpy as np

    rng = np.random.default_rng(0)
    # noisy series so a run length can build, then a step change
    vals = list(100.0 + rng.normal(0, 0.5, 60))
    vals += list(150.0 + rng.normal(0, 0.5, 20))
    assert _changepoint_flag(vals) is True


def test_build_digest_flags_real_series(store):
    report = build_digest(store, top_n=10)
    # our seeded spike entity should be flagged via z-score (>= 2)
    flagged = [f for f in report["digest"] if f["entity_id"] == "test_entity"]
    assert flagged, "seeded spike not flagged"
    assert abs(flagged[0]["zscore"]) > 2.0


def test_build_digest_one_bad_source_does_not_kill_the_others(store):
    """A single source's query blowing up (locked DB, missing table, ...)
    must not take the other healthy sources down with it."""
    real_extract = __import__("scripts.live_intelligence_digest", fromlist=["_extract_series"])._extract_series

    def _flaky(store_, source, obs_type, fields):
        if source == "sovereign_debt":
            raise RuntimeError("simulated: database is locked")
        return real_extract(store_, source, obs_type, fields)

    with patch("scripts.live_intelligence_digest._extract_series", side_effect=_flaky):
        report = build_digest(store, top_n=10)

    assert "sovereign_debt" in report["sources_failed"]
    assert "sovereign_debt" not in report["sources_ok"]
    # every other source was still attempted (they legitimately have no rows
    # in this fixture's DB, so they end up in sources_ok with 0 series).
    assert set(report["sources_ok"]) == set(_SCORABLE) - {"sovereign_debt"}


def test_polymarket_removed_structurally_dead():
    """polymarket/market_probability tops out at 15 points/entity across all
    1,493 entities — a hard ceiling below the 20-point z-score floor, not a
    quiet period (measured on the real pipeline.db, 2026-08-29). Scanning it
    nightly and counting it toward surface_scored/sources_ok reports a
    working source that mathematically cannot ever produce a finding."""
    assert "polymarket" not in _SCORABLE


def test_sovereign_debt_kept_not_structurally_dead():
    """Unlike polymarket, sovereign_debt has no hard ceiling — 5 of 13
    entities clear the 20-point floor and compute real variance (measured
    |z| up to 1.78). Currently quiet, not incapable; kept in _SCORABLE."""
    assert "sovereign_debt" in _SCORABLE


def test_extract_series_multi_unions_base_and_newer_types(tmp_path):
    """The baseline-correctness fix: a source can carry the SAME field under
    more than one observation_type (instrument_daily, long history, vs.
    instrument_volatility, ~50 points). The union must be longer than either
    type alone, and on an overlapping timestamp the LATER type in the tuple
    (the superseding collector) must win."""
    import time

    s = PipelineStore(str(tmp_path / "pipeline.db"))
    cur = s._conn.cursor()
    base_ts = 1_000_000
    # base/superseded type: 40 points, one field only ("realized_vol_20d")
    for i in range(40):
        cur.execute(
            "insert into entity_observations "
            "(entity_id, source_tool, observed_at, ingested_at, "
            " observation_type, depth_level, value_json) "
            "values (?, 'instrument_universe', ?, ?, 'instrument_daily', 1, ?)",
            ("ent1", base_ts + i * 100, time.time(), json.dumps({"realized_vol_20d": 10.0})),
        )
    # newer type: 10 points, overlapping the last 5 timestamps of the base
    # type with a DIFFERENT value — the newer type must win there.
    for i in range(35, 45):
        cur.execute(
            "insert into entity_observations "
            "(entity_id, source_tool, observed_at, ingested_at, "
            " observation_type, depth_level, value_json) "
            "values (?, 'instrument_universe', ?, ?, 'instrument_volatility', 1, ?)",
            ("ent1", base_ts + i * 100, time.time(), json.dumps({"realized_vol_20d": 99.0})),
        )
    s._conn.commit()

    type_specs = (
        ("instrument_daily", ("realized_vol_20d",)),
        ("instrument_volatility", ("realized_vol_20d",)),  # listed later -> wins on overlap
    )
    series = _extract_series_multi(s, "instrument_universe", type_specs)
    values, latest_ts, timestamps, obs_type = series[("ent1", "realized_vol_20d")]

    # union length: 40 base + 10 newer - 5 overlapping duplicates = 45 points,
    # not 40 (base alone) and not 10 (newer alone).
    assert len(values) == 45
    # overlapping timestamp took the NEWER type's value (99.0), not the base
    # type's (10.0).
    overlap_idx = timestamps.index(base_ts + 35 * 100)
    assert values[overlap_idx] == 99.0
    # the latest point came from the newer type.
    assert obs_type == "instrument_volatility"


def test_build_digest_all_sources_failing_raises_not_silently_empty():
    """An empty digest must always mean 'genuinely checked, nothing
    anomalous' — never 'failed to check anything.' If every source errors,
    this must raise rather than return a report indistinguishable from a
    real quiet week."""

    def _always_fails(store_, source, obs_type, fields):
        raise RuntimeError("simulated total outage")

    fake_store = object()
    with patch("scripts.live_intelligence_digest._extract_series", side_effect=_always_fails):
        with pytest.raises(RuntimeError, match="all .* scorable sources failed"):
            build_digest(fake_store, top_n=10)
