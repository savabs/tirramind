"""Tests for the intelligence brief (positioning & flow anomalies only).

Contract Opportunities was cut (niche decision, 2026-08-27): P(win) was a
hardcoded, never-learned Beta prior for every row and USASpending
`spending_by_award` only returns already-awarded contracts — nothing
biddable. See docs/specs/nineteen_dollar_tier_spec.md Step 2.
"""

from __future__ import annotations

from unittest.mock import patch

from scripts.intelligence_brief import build_brief, render_markdown


def test_build_brief_has_no_contract_opportunities():
    with patch("scripts.intelligence_brief.fetch_anomalies") as m_anom:
        m_anom.return_value = [{"source": "cftc", "zscore": 3.0, "changepoint": True}]
        brief = build_brief()
    assert "contract_opportunities" not in brief
    assert brief["live_anomalies"][0]["source"] == "cftc"


def test_build_brief_ignores_deprecated_contract_kwargs():
    """Old callers (deliver_brief.py, tirra_engine.py) still pass these — must not raise."""
    with patch("scripts.intelligence_brief.fetch_anomalies") as m_anom:
        m_anom.return_value = []
        brief = build_brief(
            contracts_limit=10,
            anomalies_limit=8,
            learner_path=".tirra_opportunities/win_learner.jsonl",
            db_path=".tirra_pipeline/pipeline.db",
            max_contract_rows=5,
        )
    assert brief["brief_type"] == "intelligence"


def test_build_brief_has_edition_identity():
    with patch("scripts.intelligence_brief.fetch_anomalies") as m_anom:
        m_anom.return_value = []
        brief = build_brief()
    assert brief["edition_date"]  # UTC date, non-empty
    assert brief["edition_id"].startswith(f"tirra-brief-{brief['edition_date']}-")


def test_build_brief_edition_id_stable_for_same_content_same_day():
    with patch("scripts.intelligence_brief.fetch_anomalies") as m_anom:
        m_anom.return_value = [{"source": "cftc", "entity_id": "e1", "field": "mm_net", "zscore": 3.0}]
        b1 = build_brief()
        b2 = build_brief()
    assert b1["edition_id"] == b2["edition_id"]


def test_build_brief_edition_id_changes_with_content():
    with patch("scripts.intelligence_brief.fetch_anomalies") as m_anom:
        m_anom.return_value = [{"source": "cftc", "entity_id": "e1", "field": "mm_net", "zscore": 3.0}]
        b1 = build_brief()
        m_anom.return_value = [{"source": "cftc", "entity_id": "e2", "field": "mm_net", "zscore": -2.5}]
        b2 = build_brief()
    assert b1["edition_id"] != b2["edition_id"]


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
