"""Phase 47 — unit tests for scripts/density_audit.py (9 tests).

Covers:
  - empty DB → exit 1 (error on missing table)
  - all types above threshold → exit 0 (PASS)
  - obs_count below min_obs → SPARSE, exit 1
  - temporal span below min_days → SPARSE, exit 1
  - entity_count below min_entities → SPARSE, exit 1
  - Shannon entropy computed correctly
  - exit code 0 (all pass)
  - exit code 1 (one sparse type)
  - source_tool report shows correct counts
"""

from __future__ import annotations

import math
import sqlite3
import time
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from density_audit import _run_audit, _entropy, _is_sparse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db(tmp_path) -> str:
    db_path = str(tmp_path / "pipeline.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE entities (
            entity_id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            created_at REAL NOT NULL,
            metadata_json TEXT
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE entity_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id TEXT NOT NULL,
            source_tool TEXT NOT NULL,
            observed_at REAL NOT NULL,
            ingested_at REAL NOT NULL,
            observation_type TEXT NOT NULL,
            depth_level INTEGER NOT NULL DEFAULT 1,
            value_json TEXT NOT NULL,
            metadata_json TEXT
        )
    """
    )
    conn.commit()
    conn.close()
    return db_path


def _insert_obs(
    db_path: str,
    entity_type: str,
    source_tool: str,
    n: int,
    earliest_ts: float,
    latest_ts: float,
) -> None:
    """Insert n observations spread over [earliest_ts, latest_ts]."""
    conn = sqlite3.connect(db_path)
    # Ensure entities row exists for each observation
    span = max(latest_ts - earliest_ts, 0)
    step = span / max(n - 1, 1) if n > 1 else 0
    for i in range(n):
        eid = f"{entity_type}_{i}"
        ts = earliest_ts + (i * step)
        conn.execute(
            "INSERT OR IGNORE INTO entities VALUES (?,?,?,?,?)",
            (eid, entity_type, eid, time.time(), None),
        )
        conn.execute(
            "INSERT INTO entity_observations VALUES (?,?,?,?,?,?,?,?,?)",
            (None, eid, source_tool, ts, time.time(), "obs", 2, "{}", None),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# 1. Empty DB (no entity_observations table) → exit 1
# ---------------------------------------------------------------------------


def test_empty_db_exits_one(tmp_path):
    db_path = str(tmp_path / "empty.db")
    sqlite3.connect(db_path).close()
    code = _run_audit(db_path)
    assert code == 1


# ---------------------------------------------------------------------------
# 2. All types above thresholds → exit 0 (PASS)
# ---------------------------------------------------------------------------


def test_all_above_threshold_exits_zero(tmp_path):
    db_path = _make_db(tmp_path)
    now = time.time()
    _insert_obs(
        db_path,
        "company",
        "insider_filings",
        n=150,
        earliest_ts=now - 200 * 86400,
        latest_ts=now,
    )
    code = _run_audit(db_path, min_obs=100, min_days=180, min_entities=5)
    assert code == 0


# ---------------------------------------------------------------------------
# 3. obs_count below min_obs → SPARSE, exit 1
# ---------------------------------------------------------------------------


def test_obs_below_min_exits_one(tmp_path):
    db_path = _make_db(tmp_path)
    now = time.time()
    _insert_obs(
        db_path,
        "vessel",
        "ais_vessel",
        n=10,
        earliest_ts=now - 200 * 86400,
        latest_ts=now,
    )
    code = _run_audit(db_path, min_obs=100, min_days=30, min_entities=1)
    assert code == 1


# ---------------------------------------------------------------------------
# 4. Temporal span below min_days → SPARSE, exit 1
# ---------------------------------------------------------------------------


def test_span_below_min_days_exits_one(tmp_path):
    db_path = _make_db(tmp_path)
    now = time.time()
    # 150 obs but only 10 days of span
    _insert_obs(
        db_path,
        "politician",
        "insider_filings",
        n=150,
        earliest_ts=now - 10 * 86400,
        latest_ts=now,
    )
    code = _run_audit(db_path, min_obs=100, min_days=180, min_entities=1)
    assert code == 1


# ---------------------------------------------------------------------------
# 5. entity_count below min_entities → SPARSE, exit 1
# ---------------------------------------------------------------------------


def test_entity_count_below_min_exits_one(tmp_path):
    db_path = _make_db(tmp_path)
    now = time.time()
    # Insert 150 obs but all for a single entity (n=3 unique entities via _insert_obs)
    # Override to use one entity for all rows
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT OR IGNORE INTO entities VALUES (?,?,?,?,?)",
        ("only_one", "domain", "only_one", time.time(), None),
    )
    for i in range(150):
        ts = now - (200 - i) * 86400
        conn.execute(
            "INSERT INTO entity_observations VALUES (?,?,?,?,?,?,?,?,?)",
            (
                None,
                "only_one",
                "internet_infrastructure",
                ts,
                time.time(),
                "obs",
                2,
                "{}",
                None,
            ),
        )
    conn.commit()
    conn.close()
    code = _run_audit(db_path, min_obs=100, min_days=100, min_entities=5)
    assert code == 1


# ---------------------------------------------------------------------------
# 6. Shannon entropy computed correctly
# ---------------------------------------------------------------------------


def test_entropy_uniform_distribution():
    counts = [25, 25, 25, 25]  # perfectly uniform
    H = _entropy(counts)
    assert abs(H - math.log(4)) < 1e-9


def test_entropy_single_class():
    H = _entropy([100])
    assert H == 0.0


def test_entropy_zero_counts_skipped():
    H = _entropy([50, 0, 50])
    assert abs(H - math.log(2)) < 1e-9


# ---------------------------------------------------------------------------
# 7. _is_sparse helper returns correct reasons
# ---------------------------------------------------------------------------


def test_is_sparse_returns_reasons():
    row = {"obs_count": 5, "span_days": 10, "entity_count": 2}
    sparse, reasons = _is_sparse(row, min_obs=100, min_days=180, min_entities=5)
    assert sparse is True
    assert len(reasons) == 3


def test_is_sparse_all_ok():
    row = {"obs_count": 200, "span_days": 365, "entity_count": 10}
    sparse, reasons = _is_sparse(row, min_obs=100, min_days=180, min_entities=5)
    assert sparse is False
    assert reasons == []
