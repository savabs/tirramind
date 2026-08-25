"""Tests for the Intelligence Brief delivery layer."""

from __future__ import annotations

import pytest

from agent.delivery.brief_deliverer import BriefDeliverer


def _sample_brief() -> dict:
    return {
        "brief_type": "intelligence",
        "contract_opportunities": [
            {"award_id": "X1", "recipient": "Co", "agency": "VA",
             "amount_usd": 40000.0, "expected_value_usd": 20000.0,
             "p_win": 0.75, "is_long_tail": True}
        ],
        "live_anomalies": [
            {"source": "cftc", "observation_type": "futures_positioning",
             "field": "mm_net", "zscore": -3.0, "changepoint": True}
        ],
    }


@pytest.fixture
def render_md():
    return lambda brief: f"# {brief['brief_type']}"


def test_deliver_writes_files(tmp_path, render_md):
    d = BriefDeliverer(out_dir=str(tmp_path / "del"), render_md=render_md)
    rec = d.deliver(_sample_brief())
    assert (tmp_path / "del" / "intelligence_brief.json").exists()
    assert (tmp_path / "del" / "intelligence_brief.md").exists()
    assert rec.n_contracts == 1
    assert rec.n_anomalies == 1
    assert rec.duration_ms >= 0


def test_delivery_log_append_and_latest(tmp_path, render_md):
    d = BriefDeliverer(out_dir=str(tmp_path / "del"), render_md=render_md)
    d.deliver(_sample_brief())
    d.deliver(_sample_brief())
    assert len(d.records()) == 2
    latest = d.latest()
    assert latest is not None and latest.n_contracts == 1


def test_delivery_record_roundtrip(tmp_path, render_md):
    d = BriefDeliverer(out_dir=str(tmp_path / "del"), render_md=render_md)
    d.deliver(_sample_brief())
    # reload from disk (new instance) preserves records
    d2 = BriefDeliverer(out_dir=str(tmp_path / "del"), render_md=render_md)
    assert len(d2.records()) == 1
    assert d2.latest().checksum == d.latest().checksum


def test_status_shape(tmp_path, render_md):
    d = BriefDeliverer(out_dir=str(tmp_path / "del"), render_md=render_md)
    s = d.status()
    assert "out_dir" in s and "total_deliveries" in s and "latest" in s
    assert s["total_deliveries"] == 0
    d.deliver(_sample_brief())
    assert d.status()["total_deliveries"] == 1


def test_deliver_without_renderer_still_writes_json(tmp_path):
    d = BriefDeliverer(out_dir=str(tmp_path / "del"), render_md=None)
    rec = d.deliver(_sample_brief())
    assert rec.n_contracts == 1
    assert (tmp_path / "del" / "intelligence_brief.json").exists()


def test_latest_returns_newest_not_oldest(tmp_path, render_md):
    """Regression: latest() must return the most recent delivery, not the oldest (newest-first list)."""
    import time as _time
    d = BriefDeliverer(out_dir=str(tmp_path / "del"), render_md=render_md)
    first = d.deliver(_sample_brief())
    _time.sleep(0.01)  # ensure distinct timestamps
    second = d.deliver(_sample_brief())
    assert d.latest() is not None
    assert d.latest().delivered_at >= first.delivered_at
    assert d.latest().checksum == second.checksum
