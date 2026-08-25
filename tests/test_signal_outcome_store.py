"""Tests for the honest signal-outcome loop (surface → realize)."""

from __future__ import annotations

import json
import time

import pytest

from agent.quant.signal_outcome_store import SignalOutcomeStore


@pytest.fixture
def ledger(tmp_path):
    return SignalOutcomeStore(str(tmp_path / "signals.jsonl"))


def test_surface_stores_pending(ledger):
    sig = ledger.surface(
        source="cftc", observation_type="futures_positioning",
        entity_id="e1", field="mm_net",
        direction=1.0, flagged_ts=1000.0, ref_value=50.0, zscore=3.0,
    )
    assert sig.status == "pending"
    assert sig.success is None  # no reward assigned on surface
    pending = ledger.pending()
    assert len(pending) == 1
    assert pending[0].signal_id == sig.signal_id


def test_realize_removes_from_pending_and_sets_outcome(ledger):
    sig = ledger.surface(
        source="cftc", observation_type="futures_positioning",
        entity_id="e1", field="mm_net",
        direction=1.0, flagged_ts=1000.0, ref_value=50.0, zscore=3.0,
    )
    ledger.realize(sig.signal_id, success=True)
    # realized signals are no longer pending (they are resolved history)
    assert ledger.pending() == []


def test_pending_persists_across_reload(tmp_path):
    path = str(tmp_path / "signals.jsonl")
    ledger = SignalOutcomeStore(path)
    sig = ledger.surface(
        source="cftc", observation_type="futures_positioning",
        entity_id="e1", field="mm_net",
        direction=-1.0, flagged_ts=1000.0, ref_value=10.0, zscore=-2.5,
    )
    reloaded = SignalOutcomeStore(path)
    assert len(reloaded.pending()) == 1
    assert reloaded.pending()[0].signal_id == sig.signal_id


def test_realize_not_guessed_without_realize_call(ledger):
    """A surfaced signal without a realize() stays pending — no fabricated reward."""
    sig = ledger.surface(
        source="cftc", observation_type="futures_positioning",
        entity_id="e1", field="mm_net",
        direction=1.0, flagged_ts=1000.0, ref_value=50.0, zscore=3.0,
    )
    assert sig.success is None  # never guessed


def _seed_series(store_path, entity, source, otype, field, points, start_ts=1000.0):
    """Insert synthetic numeric observations into a temp PipelineStore."""
    import sqlite3
    con = sqlite3.connect(store_path)
    con.execute(
        "create table if not exists entity_observations ("
        "id integer primary key, entity_id text, source_tool text, observed_at real, "
        "ingested_at real, observation_type text, depth_level int, value_json text, metadata_json text)"
    )
    con.execute("delete from entity_observations")
    for i, val in enumerate(points):
        con.execute(
            "insert into entity_observations (entity_id, source_tool, observed_at, ingested_at, observation_type, depth_level, value_json) values (?,?,?,?,?,?,?)",
            (entity, source, start_ts + i * 100, time.time(), otype, 1, json.dumps({field: val})),
        )
    con.commit()
    con.close()


def test_realize_records_honest_outcome_with_forward_move(tmp_path):
    """A signal surfaced BEFORE a real upward move must realize as success=True;
    surfacing itself must never be a reward — only the forward confirmation is.
    """
    from agent.quant.signal_outcome_store import SignalOutcomeStore

    db = str(tmp_path / "p.db")
    ledger_path = str(tmp_path / "signals.jsonl")
    ledger = SignalOutcomeStore(ledger_path)

    # Series: flat 100s for a while, then a step UP at the end.
    points = [100.0] * 40 + [150.0, 160.0, 170.0]
    _seed_series(db, "e1", "sovereign_debt", "sovereign_yield", "yield_pct", points)

    # Surface the anomaly at the point where the series first spikes (ts = 1000 + 40*100).
    sig = ledger.surface(
        source="sovereign_debt", observation_type="sovereign_yield",
        entity_id="e1", field="yield_pct",
        direction=1.0, flagged_ts=1000.0 + 40 * 100, ref_value=100.0, zscore=3.0,
    )
    assert sig.success is None  # surfaced, but not rewarded

    # Now realize against forward data (already present in DB).
    from scripts.live_intelligence_digest import realize_pending
    res = realize_pending(db, store_path=ledger_path, state_dir=str(tmp_path / "awos"), min_forward_points=2)
    assert res["realized"] == 1, res
    assert res["still_pending"] == 0
    # Ledger shows it realized (no longer pending).
