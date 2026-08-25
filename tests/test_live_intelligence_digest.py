"""Tests for the live intelligence digest — real math on real stored data."""

from __future__ import annotations

import json

import pytest

from agent.pipeline.store import PipelineStore
from scripts.live_intelligence_digest import _changepoint_flag, _zscore_anomaly, build_digest


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
