"""Tests for the intelligence brief (positioning & flow anomalies only).

Contract Opportunities was cut (niche decision, 2026-08-27): P(win) was a
hardcoded, never-learned Beta prior for every row and USASpending
`spending_by_award` only returns already-awarded contracts — nothing
biddable. See docs/specs/nineteen_dollar_tier_spec.md Step 2.
"""

from __future__ import annotations

from unittest.mock import patch

from scripts.intelligence_brief import _SECTION_RENDERERS, build_brief, render_markdown


def test_build_brief_has_no_contract_opportunities():
    with patch("scripts.intelligence_brief.fetch_report") as m_anom:
        m_anom.return_value = {"digest": [{"source": "cftc", "zscore": 3.0, "changepoint": True}]}
        brief = build_brief()
    assert "contract_opportunities" not in brief
    assert brief["live_anomalies"][0]["source"] == "cftc"


def test_build_brief_ignores_deprecated_contract_kwargs():
    """Old callers (deliver_brief.py, tirra_engine.py) still pass these — must not raise."""
    with patch("scripts.intelligence_brief.fetch_report") as m_anom:
        m_anom.return_value = {"digest": []}
        brief = build_brief(
            contracts_limit=10,
            anomalies_limit=8,
            learner_path=".tirra_opportunities/win_learner.jsonl",
            db_path=".tirra_pipeline/pipeline.db",
            max_contract_rows=5,
        )
    assert brief["brief_type"] == "intelligence"


def test_build_brief_has_edition_identity():
    with patch("scripts.intelligence_brief.fetch_report") as m_anom:
        m_anom.return_value = {"digest": []}
        brief = build_brief()
    assert brief["edition_date"]  # UTC date, non-empty
    assert brief["edition_id"].startswith(f"tirra-brief-{brief['edition_date']}-")


def test_build_brief_edition_id_stable_for_same_content_same_day():
    with patch("scripts.intelligence_brief.fetch_report") as m_anom:
        m_anom.return_value = {"digest": [{"source": "cftc", "entity_id": "e1", "field": "mm_net", "zscore": 3.0}]}
        b1 = build_brief()
        b2 = build_brief()
    assert b1["edition_id"] == b2["edition_id"]


def test_build_brief_edition_id_changes_with_content():
    with patch("scripts.intelligence_brief.fetch_report") as m_anom:
        m_anom.return_value = {"digest": [{"source": "cftc", "entity_id": "e1", "field": "mm_net", "zscore": 3.0}]}
        b1 = build_brief()
        m_anom.return_value = {"digest": [{"source": "cftc", "entity_id": "e2", "field": "mm_net", "zscore": -2.5}]}
        b2 = build_brief()
    assert b1["edition_id"] != b2["edition_id"]


def test_build_brief_surfaces_degraded_sources_honestly():
    """A partial outage (some sources unreachable) must be disclosed, not
    silently rendered identically to a full, clean, all-quiet edition."""
    with patch("scripts.intelligence_brief.fetch_report") as m_anom:
        m_anom.return_value = {
            "digest": [{"source": "cftc", "entity_id": "e1", "field": "mm_net", "zscore": 3.0}],
            "surface_scored": 5,
            "sources_failed": ["defi_flows", "polymarket"],
        }
        brief = build_brief()
    assert brief["degraded_sources"] == ["defi_flows", "polymarket"]
    assert "2 of 5" in brief["degraded_note"]
    md = render_markdown(brief)
    assert "2 of 5" in md


def test_render_markdown_has_no_contract_section_and_shows_edition():
    brief = {
        "edition_date": "2026-08-27",
        "edition_id": "tirra-brief-2026-08-27-deadbeefcafe",
        "methodology": "test methodology",
        "live_anomalies": [
            {
                "source": "cftc",
                "observation_type": "futures_positioning",
                "entity_id": "abc123",
                "entity_name": "COTTON NO. 2 - ICE FUTURES U.S.",
                "field": "open_interest",
                "zscore": 3.14,
                "direction": "up",
                "changepoint": True,
                "changepoint_weeks_ago": 6.0,
                "n_points": 169,
                "latest_value": 361871.0,
            }
        ],
    }
    md = render_markdown(brief)
    assert "Contract Opportunities" not in md
    assert "2026-08-27" in md
    assert "tirra-brief-2026-08-27-deadbeefcafe" in md
    assert "COTTON NO. 2 - ICE FUTURES U.S." in md
    assert "open interest" in md
    assert "z=+3.14" in md
    assert "169-point baseline" in md
    assert "6 weeks ago" in md


def test_render_markdown_groups_by_source_with_headings():
    brief = {
        "edition_date": "2026-08-27",
        "edition_id": "x",
        "methodology": "m",
        "live_anomalies": [
            {
                "source": "cftc",
                "observation_type": "futures_positioning",
                "entity_id": "e1",
                "entity_name": "Cotton",
                "field": "mm_net",
                "zscore": 3.0,
                "direction": "up",
                "changepoint": False,
                "changepoint_weeks_ago": None,
                "n_points": 169,
                "latest_value": 1.0,
            },
            {
                "source": "defi_flows",
                "observation_type": "tvl_change",
                "entity_id": "e2",
                "entity_name": "Some Pool",
                "field": "tvl_usd",
                "zscore": -2.5,
                "direction": "down",
                "changepoint": False,
                "changepoint_weeks_ago": None,
                "n_points": 3000,
                "latest_value": 1.0,
            },
        ],
    }
    md = render_markdown(brief)
    # More than one source rendered — a single-source edition is the exact
    # failure mode this brief previously shipped (100% cftc).
    assert "CFTC" in md
    assert "DeFi Flows" in md
    assert md.index("CFTC") < md.index("Cotton")
    assert md.index("DeFi Flows") < md.index("Some Pool")


def test_render_markdown_empty_anomalies_is_honest_not_missing():
    brief = {"edition_date": "2026-08-27", "edition_id": "x", "methodology": "m", "live_anomalies": []}
    md = render_markdown(brief)
    assert "none currently flagged" in md
    assert "2026-08-27" in md


def test_render_markdown_skips_malformed_finding_without_crashing():
    """A single malformed row (non-numeric zscore) must not take the whole
    render — and therefore the whole weekly delivery — down with it."""
    brief = {
        "edition_date": "2026-08-27",
        "edition_id": "x",
        "methodology": "m",
        "live_anomalies": [
            {"source": "cftc", "entity_id": "bad", "field": "mm_net", "zscore": None},
            {
                "source": "cftc",
                "entity_id": "e2",
                "entity_name": "Cotton",
                "field": "mm_net",
                "zscore": 3.0,
                "direction": "up",
                "n_points": 169,
            },
        ],
    }
    md = render_markdown(brief)  # must not raise
    assert "Cotton" in md
    assert "z=+3.00" in md


def test_render_markdown_sections_compose_on_a_quiet_week():
    """The structural blocker (fixed 2026-08-27): a section registered after
    the anomalies section must still render even when live_anomalies is
    empty. Previously the empty-anomalies branch `return`ed immediately and
    silently dropped everything after it — precisely the quiet week when a
    second section matters most.
    """
    brief = {
        "edition_date": "2026-08-27",
        "edition_id": "x",
        "methodology": "m",
        "live_anomalies": [],
    }

    def _extra_section(_brief):
        return ["## Some Later Section", "", "content that must survive a quiet week", ""]

    _SECTION_RENDERERS.append(_extra_section)
    try:
        md = render_markdown(brief)
    finally:
        _SECTION_RENDERERS.remove(_extra_section)

    assert "none currently flagged" in md
    assert "Some Later Section" in md
    assert "content that must survive a quiet week" in md
    # order preserved: registration order is render order
    assert md.index("Positioning & Flow Anomalies") < md.index("Some Later Section")


def test_render_markdown_section_returning_none_is_skipped_not_broken():
    """A section contract: returning None (or []) means 'nothing to render
    this edition' and must not raise or leave stray blank sections."""
    brief = {"edition_date": "2026-08-27", "edition_id": "x", "methodology": "m", "live_anomalies": []}

    def _quiet_section(_brief):
        return None

    _SECTION_RENDERERS.append(_quiet_section)
    try:
        md = render_markdown(brief)  # must not raise
    finally:
        _SECTION_RENDERERS.remove(_quiet_section)
    assert "none currently flagged" in md


def test_render_markdown_missing_source_key_falls_back_to_unknown():
    brief = {
        "edition_date": "2026-08-27",
        "edition_id": "x",
        "methodology": "m",
        "live_anomalies": [{"entity_id": "e1", "field": "mm_net", "zscore": 2.5, "n_points": 30}],
    }
    md = render_markdown(brief)  # must not raise on missing "source"
    assert "z=+2.50" in md
