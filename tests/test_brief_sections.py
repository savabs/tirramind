"""Tests for the three new $29-tier Brief sections: insider sell intent
(form144/sell_intent), credit stress (creditor_filings/creditor_filing), and
the week-over-week anomaly diff.

Ground truth these tests encode (measured against the real DB 2026-08-29,
see scripts/brief_sections.py module docstring for the full write-up):
  - form144/sell_intent is CROSS-SECTIONAL (max 9 points/entity) -- these
    tests never exercise a z-score path for it.
  - creditor_filings has only 16 rows ever; a given week can be genuinely
    empty and the section must say so, not disappear.
  - the archive can have same-day duplicate files (overwrites) and can be
    completely absent (first-ever run) -- both must degrade gracefully.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import pytest

from agent.pipeline.store import PipelineStore
from scripts.brief_sections import (
    attach_sections,
    compute_wow_diff,
    fetch_credit_stress,
    fetch_insider_sell_intent,
    render_credit_stress_section,
    render_insider_sell_intent_section,
    render_wow_diff_section,
)

# A fixed "now" so week-boundary math is deterministic across test runs.
# 2026-08-26 is a Wednesday in ISO week 2026-W35 (2026-08-24 .. 2026-08-30).
_NOW = datetime(2026, 8, 26, 12, 0, 0, tzinfo=UTC)
_THIS_WEEK_TS = datetime(2026, 8, 25, 18, 0, 0, tzinfo=UTC).timestamp()
_LAST_WEEK_TS = datetime(2026, 8, 17, 18, 0, 0, tzinfo=UTC).timestamp()


def _insert_observation(store, entity_id, source_tool, obs_type, observed_at, value):
    cur = store._conn.cursor()
    cur.execute(
        "insert into entity_observations "
        "(entity_id, source_tool, observed_at, ingested_at, observation_type, depth_level, value_json) "
        "values (?, ?, ?, ?, ?, 1, ?)",
        (entity_id, source_tool, observed_at, time.time(), obs_type, json.dumps(value)),
    )
    store._conn.commit()


def _insert_entity(store, entity_id, entity_type, canonical_name):
    cur = store._conn.cursor()
    cur.execute(
        "insert or ignore into entities (entity_id, entity_type, canonical_name, created_at) values (?, ?, ?, ?)",
        (entity_id, entity_type, canonical_name, time.time()),
    )
    store._conn.commit()


@pytest.fixture
def store(tmp_path):
    return PipelineStore(str(tmp_path / "pipeline.db"))


# ---------------------------------------------------------------------------
# fetch_insider_sell_intent
# ---------------------------------------------------------------------------


def test_insider_sell_intent_filters_zero_and_non_immediate(store):
    """Only dollar_value > 0 AND urgency == 'immediate' rows count -- the
    other 638/1,195 measured rows on the real DB are zero/blank and would
    understate every issuer if included."""
    _insert_entity(store, "filer-1", "person", "Jane Doe")
    _insert_observation(
        store,
        "filer-1",
        "form144",
        "sell_intent",
        _THIS_WEEK_TS,
        {
            "ticker": "ZZZ",
            "company": "Zero Corp",
            "dollar_value": 0.0,
            "urgency": "immediate",
            "relationship": "Officer",
        },
    )
    _insert_observation(
        store,
        "filer-1",
        "form144",
        "sell_intent",
        _THIS_WEEK_TS + 1,
        {
            "ticker": "YYY",
            "company": "Not Urgent Inc",
            "dollar_value": 500_000.0,
            "urgency": "unknown",
            "relationship": "Director",
        },
    )
    result = fetch_insider_sell_intent(store, now=_NOW)
    assert result["qualifying_count"] == 0
    assert result["top_issuers"] == []
    assert result["total_rows_this_week"] == 2


def test_insider_sell_intent_aggregates_to_issuer_week(store):
    """Cross-sectional aggregation: two filings for the same ticker in the
    same week sum into one issuer-week total, ranked above a smaller one."""
    _insert_entity(store, "filer-a", "person", "Alice Officer")
    _insert_entity(store, "filer-b", "person", "Bob Officer")
    _insert_entity(store, "filer-c", "person", "Cara Director")
    _insert_observation(
        store,
        "filer-a",
        "form144",
        "sell_intent",
        _THIS_WEEK_TS,
        {
            "ticker": "BIG",
            "company": "Big Corp",
            "dollar_value": 100_000_000.0,
            "urgency": "immediate",
            "relationship": "Officer",
        },
    )
    _insert_observation(
        store,
        "filer-b",
        "form144",
        "sell_intent",
        _THIS_WEEK_TS + 10,
        {
            "ticker": "BIG",
            "company": "Big Corp",
            "dollar_value": 50_000_000.0,
            "urgency": "immediate",
            "relationship": "Officer",
        },
    )
    _insert_observation(
        store,
        "filer-c",
        "form144",
        "sell_intent",
        _THIS_WEEK_TS + 20,
        {
            "ticker": "SMALL",
            "company": "Small Corp",
            "dollar_value": 1_000_000.0,
            "urgency": "immediate",
            "relationship": "Director",
        },
    )
    result = fetch_insider_sell_intent(store, now=_NOW)
    assert result["qualifying_count"] == 3
    assert result["total_declared_usd"] == pytest.approx(151_000_000.0)
    top = result["top_issuers"]
    assert top[0]["ticker"] == "BIG"
    assert top[0]["total_declared_usd"] == pytest.approx(150_000_000.0)
    assert top[0]["filing_count"] == 2
    # largest single filing under BIG is the $100M one, filed by Alice
    assert top[0]["largest_single_filing_usd"] == pytest.approx(100_000_000.0)
    assert top[0]["largest_single_filer_name"] == "Alice Officer"
    assert top[1]["ticker"] == "SMALL"


def test_insider_sell_intent_excludes_prior_week(store):
    _insert_entity(store, "filer-old", "person", "Old Filer")
    _insert_observation(
        store,
        "filer-old",
        "form144",
        "sell_intent",
        _LAST_WEEK_TS,
        {
            "ticker": "OLD",
            "company": "Old Corp",
            "dollar_value": 9_000_000.0,
            "urgency": "immediate",
            "relationship": "Officer",
        },
    )
    result = fetch_insider_sell_intent(store, now=_NOW)
    assert result["total_rows_this_week"] == 0
    assert result["qualifying_count"] == 0


def test_insider_sell_intent_dedupes_reingested_rows(store):
    """Same (entity_id, observed_at) inserted twice (re-ingestion without an
    upsert, the documented 52.8%-duplicate behavior) must count once."""
    _insert_entity(store, "filer-dup", "person", "Dup Filer")
    payload = {
        "ticker": "DUP",
        "company": "Dup Corp",
        "dollar_value": 10_000_000.0,
        "urgency": "immediate",
        "relationship": "Officer",
    }
    _insert_observation(store, "filer-dup", "form144", "sell_intent", _THIS_WEEK_TS, payload)
    _insert_observation(
        store, "filer-dup", "form144", "sell_intent", _THIS_WEEK_TS, payload
    )  # re-ingest, same observed_at
    result = fetch_insider_sell_intent(store, now=_NOW)
    assert result["qualifying_count"] == 1
    assert result["top_issuers"][0]["total_declared_usd"] == pytest.approx(10_000_000.0)


# ---------------------------------------------------------------------------
# fetch_credit_stress
# ---------------------------------------------------------------------------


def test_credit_stress_renders_item_codes_with_labels(store):
    _insert_entity(store, "issuer-1", "company", "acme corp acme cik 0001234567")
    _insert_observation(
        store,
        "issuer-1",
        "creditor_filings",
        "creditor_filing",
        _THIS_WEEK_TS,
        {
            "cik": "0001234567",
            "form": "8-K",
            "file_date": "2026-08-25",
            "items": ["1.01", "2.03", "9.01"],
            "is_stress_signal": True,
        },
    )
    result = fetch_credit_stress(store, now=_NOW)
    assert result["filing_count"] == 1
    f = result["filings"][0]
    assert f["entity_name"] == "Acme Corp Acme"  # " cik ..." suffix stripped, title-cased
    assert f["items"] == ["1.01", "2.03", "9.01"]
    assert f["item_labels"] == [
        "Entry into a Material Definitive Agreement",
        "Creation of a Direct Financial Obligation",
        "Financial Statements and Exhibits",
    ]


def test_credit_stress_empty_week_is_honest_not_omitted(store):
    """No rows at all in the store -- fetch must still return a dict (never
    None) so the renderer can say so honestly, matching the requirement that
    a genuinely quiet week must not make the whole section vanish."""
    result = fetch_credit_stress(store, now=_NOW)
    assert result["filing_count"] == 0
    assert result["filings"] == []

    brief = {"credit_stress": result}
    lines = render_credit_stress_section(brief)
    assert lines is not None
    text = "\n".join(lines)
    assert "No credit-stress 8-K filings this week" in text


def test_credit_stress_excludes_other_weeks(store):
    _insert_entity(store, "issuer-2", "company", "other corp othr cik 0009999999")
    _insert_observation(
        store,
        "issuer-2",
        "creditor_filings",
        "creditor_filing",
        _LAST_WEEK_TS,
        {"cik": "0009999999", "form": "8-K", "file_date": "2026-08-17", "items": ["2.03"], "is_stress_signal": True},
    )
    result = fetch_credit_stress(store, now=_NOW)
    assert result["filing_count"] == 0


# ---------------------------------------------------------------------------
# render_* section contract: None input -> None (never renders a broken
# section on missing data), populated input -> real content.
# ---------------------------------------------------------------------------


def test_insider_render_returns_none_when_key_absent():
    assert render_insider_sell_intent_section({}) is None
    assert render_insider_sell_intent_section({"insider_sell_intent": None}) is None


def test_credit_stress_render_returns_none_when_key_absent():
    assert render_credit_stress_section({}) is None
    assert render_credit_stress_section({"credit_stress": None}) is None


def test_wow_diff_render_returns_none_when_key_absent():
    assert render_wow_diff_section({}) is None


# ---------------------------------------------------------------------------
# compute_wow_diff / render_wow_diff_section
# ---------------------------------------------------------------------------


def _write_archive_edition(archive_dir, edition_date, anomalies):
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / f"intelligence_brief_{edition_date}.json").write_text(
        json.dumps({"edition_date": edition_date, "live_anomalies": anomalies}), encoding="utf-8"
    )


def test_wow_diff_first_edition_when_archive_missing(tmp_path):
    result = compute_wow_diff(str(tmp_path / "does_not_exist"), "2026-08-29", [])
    assert result["first_edition"] is True
    lines = render_wow_diff_section({"wow_diff": result})
    assert "No prior edition on record" in "\n".join(lines)


def test_wow_diff_first_edition_when_archive_empty(tmp_path):
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    result = compute_wow_diff(str(archive_dir), "2026-08-29", [])
    assert result["first_edition"] is True


def test_wow_diff_ignores_same_day_files(tmp_path):
    """Same-day reruns overwrite the one file for that date -- a file dated
    the SAME as the current edition must never be treated as 'prior', even
    though it physically exists on disk (this is the exact scenario the
    real index.jsonl shows: three 2026-08-27 checksums, one file)."""
    archive_dir = tmp_path / "archive"
    _write_archive_edition(archive_dir, "2026-08-29", [{"source": "cftc", "entity_id": "e1", "field": "mm_net"}])
    result = compute_wow_diff(str(archive_dir), "2026-08-29", [])
    assert result["first_edition"] is True  # no STRICTLY earlier file exists


def test_wow_diff_new_resolved_held(tmp_path):
    archive_dir = tmp_path / "archive"
    prior_anomalies = [
        {"source": "cftc", "entity_id": "e1", "field": "mm_net", "entity_name": "Cotton", "zscore": 2.5},
        {"source": "cftc", "entity_id": "e2", "field": "mm_net", "entity_name": "Corn", "zscore": -3.0},
    ]
    _write_archive_edition(archive_dir, "2026-08-22", prior_anomalies)

    current_anomalies = [
        {"source": "cftc", "entity_id": "e1", "field": "mm_net", "entity_name": "Cotton", "zscore": 2.6},  # held
        {"source": "cftc", "entity_id": "e3", "field": "mm_net", "entity_name": "Wheat", "zscore": 4.0},  # new
        # e2/Corn dropped out -> resolved
    ]
    result = compute_wow_diff(str(archive_dir), "2026-08-29", current_anomalies)
    assert result["first_edition"] is False
    assert result["prior_edition_date"] == "2026-08-22"
    assert [a["entity_id"] for a in result["new"]] == ["e3"]
    assert [a["entity_id"] for a in result["resolved"]] == ["e2"]
    assert [a["entity_id"] for a in result["held"]] == ["e1"]

    md = "\n".join(render_wow_diff_section({"wow_diff": result}))
    assert "1 new" in md
    assert "1 resolved" in md
    assert "1 held" in md
    assert "Wheat" in md
    assert "Corn" in md
    assert "Cotton" in md


def test_wow_diff_picks_most_recent_of_multiple_prior_editions(tmp_path):
    archive_dir = tmp_path / "archive"
    _write_archive_edition(archive_dir, "2026-08-15", [{"source": "cftc", "entity_id": "old", "field": "mm_net"}])
    _write_archive_edition(archive_dir, "2026-08-22", [{"source": "cftc", "entity_id": "recent", "field": "mm_net"}])
    result = compute_wow_diff(str(archive_dir), "2026-08-29", [])
    assert result["prior_edition_date"] == "2026-08-22"


# ---------------------------------------------------------------------------
# attach_sections integration + THE REGRESSION TEST: everything must still
# render when live_anomalies is FORCED EMPTY (the structural blocker the
# core agent fixed in intelligence_brief.render_markdown).
# ---------------------------------------------------------------------------


def test_attach_sections_populates_all_three_keys(store, tmp_path):
    db_path = str(tmp_path / "pipeline.db")
    # attach_sections opens its own PipelineStore(db_path); seed through a
    # store pointed at the same file.
    seeded = PipelineStore(db_path)
    _insert_entity(seeded, "filer-x", "person", "X Officer")
    _insert_observation(
        seeded,
        "filer-x",
        "form144",
        "sell_intent",
        _THIS_WEEK_TS,
        {
            "ticker": "XYZ",
            "company": "XYZ Corp",
            "dollar_value": 5_000_000.0,
            "urgency": "immediate",
            "relationship": "Officer",
        },
    )
    _insert_entity(seeded, "issuer-y", "company", "y corp yco cik 0001111111")
    _insert_observation(
        seeded,
        "issuer-y",
        "creditor_filings",
        "creditor_filing",
        _THIS_WEEK_TS,
        {"cik": "0001111111", "form": "8-K", "file_date": "2026-08-25", "items": ["2.03"], "is_stress_signal": True},
    )

    brief = {"edition_date": "2026-08-26", "live_anomalies": []}
    brief = attach_sections(brief, db_path=db_path, archive_dir=str(tmp_path / "no_archive"), now=_NOW)

    assert brief["insider_sell_intent"]["qualifying_count"] == 1
    assert brief["credit_stress"]["filing_count"] == 1
    assert brief["wow_diff"]["first_edition"] is True


def test_all_three_sections_render_when_live_anomalies_forced_empty(tmp_path):
    """THE regression test for the core agent's Step 1 fix: render_markdown
    must never let an empty live_anomalies list silently drop sections
    registered after it. This drives the full real intelligence_brief
    render path (not a hand-built brief dict) with live_anomalies forced to
    [] and a populated DB for the two new data sections plus a prior
    archive edition for the diff -- all three new sections, plus the
    honest empty-anomalies note, must appear in the same render.
    """
    from scripts.intelligence_brief import render_markdown

    db_path = str(tmp_path / "pipeline.db")
    seeded = PipelineStore(db_path)
    _insert_entity(seeded, "filer-z", "person", "Z Director")
    _insert_observation(
        seeded,
        "filer-z",
        "form144",
        "sell_intent",
        _THIS_WEEK_TS,
        {
            "ticker": "ZCO",
            "company": "Z Corp",
            "dollar_value": 7_000_000.0,
            "urgency": "immediate",
            "relationship": "Director",
        },
    )
    _insert_entity(seeded, "issuer-w", "company", "w corp wco cik 0002222222")
    _insert_observation(
        seeded,
        "issuer-w",
        "creditor_filings",
        "creditor_filing",
        _THIS_WEEK_TS,
        {
            "cik": "0002222222",
            "form": "8-K",
            "file_date": "2026-08-25",
            "items": ["1.01", "2.03"],
            "is_stress_signal": True,
        },
    )

    archive_dir = tmp_path / "archive"
    _write_archive_edition(
        archive_dir,
        "2026-08-19",
        [{"source": "cftc", "entity_id": "gone", "field": "mm_net", "entity_name": "Resolved Thing"}],
    )

    from scripts.brief_sections import attach_sections

    brief = {
        "brief_type": "intelligence",
        "edition_date": "2026-08-26",
        "edition_id": "tirra-brief-test",
        "methodology": "m",
        "live_anomalies": [],  # FORCED EMPTY
        "anomalies_note": "No anomalies crossed the z-score threshold across any scored source this edition.",
    }
    brief = attach_sections(brief, db_path=db_path, archive_dir=str(archive_dir), now=_NOW)

    md = render_markdown(brief)

    # The quiet-week anomalies note must still be present (core agent's fix).
    assert "none currently flagged" in md or "No anomalies crossed" in md
    # All three new sections must have rendered anyway.
    assert "## Week-over-Week — Anomaly Changes" in md
    assert "Resolved Thing" in md
    assert "## Insider Sell Intent — Form 144" in md
    assert "Z Corp" in md
    assert "## Credit Stress — 8-K Filings" in md
    assert "Entry into a Material Definitive Agreement" in md

    print("\n" + "=" * 80)
    print("RENDERED BRIEF WITH live_anomalies FORCED EMPTY:")
    print("=" * 80)
    print(md)
