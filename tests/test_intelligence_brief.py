"""Tests for the fused intelligence brief (system contracts + anomalies)."""

from __future__ import annotations

from unittest.mock import patch

from agent.quant.contract_opportunity import WinProbabilityLearner
from scripts.intelligence_brief import build_brief, fetch_opportunities, render_markdown

_SAMPLE_AWARDS = [
    {"award_id": "T1", "recipient": "Small Co", "agency": "VA", "description": "janitorial",
     "amount_usd": 40000.0, "start_date": "2026-08-01", "award_type": None},
    {"award_id": "T2", "recipient": "Big Co", "agency": "DoD", "description": "arms",
     "amount_usd": 900000.0, "start_date": "2026-08-01", "award_type": None},
    {"award_id": "T3", "recipient": "Mid Co", "agency": "USDA", "description": "permits",
     "amount_usd": 60000.0, "start_date": "2026-08-01", "award_type": None},
]


def _ok_result():
    from agent.tools.base import ToolResult
    return ToolResult(success=True, output="n awards", data={"awards": _SAMPLE_AWARDS})


def test_fetch_opportunities_long_tail_first(tmp_path):
    k = tmp_path  # capture to avoid unused
    learner_path = str(tmp_path / "win.jsonl")
    with patch("scripts.intelligence_brief.GovContractsTool") as MockTool:
        MockTool.return_value.execute.return_value = _ok_result()
        rows = fetch_opportunities(limit=3, learner_path=learner_path, max_rows=3)
    assert len(rows) == 3
    # long-tail (small) contracts sort before the large one
    assert rows[0]["is_long_tail"] is True
    assert rows[-1]["is_long_tail"] is False
    # T3 (60k) has higher EV than T1 (40k); both long-tail so sort by EV desc
    assert rows[0]["award_id"] == "T3"
    assert rows[1]["award_id"] == "T1"
    assert "expected_value_usd" in rows[0]
    assert "p_win" in rows[0]


def test_fetch_opportunities_learned_pwin(tmp_path):
    learner_path = str(tmp_path / "win.jsonl")
    learner = WinProbabilityLearner(learner_path)
    # VA wins often in real histories -> higher learned P(win)
    for i in range(10):
        learner.record(f"a{i}", "VA", 40000.0, realized_success=(i < 8))

    with patch("scripts.intelligence_brief.GovContractsTool") as MockTool:
        MockTool.return_value.execute.return_value = _ok_result()
        rows = fetch_opportunities(limit=3, learner_path=learner_path, max_rows=3)
    # VA contract should have P(win) above the Beta prior baseline (0.5)
    va = [r for r in rows if r["award_id"] == "T1"][0]
    assert va["p_win"] > 0.6  # 8W/2L posterior mean = 0.75


def test_build_brief_structure(tmp_path):
    with patch("scripts.intelligence_brief.fetch_opportunities") as m_fetch, \
         patch("scripts.intelligence_brief.fetch_anomalies") as m_anom:
        m_fetch.return_value = [{"award_id": "X", "expected_value_usd": 100.0}]
        m_anom.return_value = [{"source": "cftc", "zscore": 3.0, "changepoint": True}]
        brief = build_brief()
    assert "contract_opportunities" in brief
    assert "live_anomalies" in brief
    assert brief["contract_opportunities"][0]["award_id"] == "X"
    assert brief["live_anomalies"][0]["source"] == "cftc"


def test_render_markdown_contains_sections():
    brief = {
        "contract_opportunities": [
            {"award_id": "T1", "recipient": "Small Co", "agency": "VA",
             "amount_usd": 40000.0, "expected_value_usd": 20000.0,
             "p_win": 0.75, "description": "janitorial", "is_long_tail": True}
        ],
        "live_anomalies": [
            {"source": "cftc", "observation_type": "futures_positioning",
             "field": "mm_net", "zscore": -3.0, "changepoint": True}
        ],
    }
    md = render_markdown(brief)
    assert "Contract Opportunities" in md
    assert "Live Anomalies" in md
    assert "T1" in md
    assert "z=-3.00" in md
