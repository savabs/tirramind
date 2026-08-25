"""Edge tests for agent.quant.ghost_brief."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.quant.ghost_brief import (
    DRAFT_BRIEFS_DIR,
    alert_to_brief_markdown,
    brief_path_for_alert,
    update_brief_outcome,
    write_brief_for_alert,
)


@pytest.fixture
def sample_alert() -> dict:
    return {
        "alert_id": "2026-06-10_MP-1_EIA_REGIME_CFTC_004",
        "micro_playground": "MP-1",
        "chain_template": "eia_regime_cftc",
        "nodes": [
            {
                "entity": "US crude stocks ex-SPR",
                "obs": "energy_supply/petroleum_inventory/weekly_change",
                "z": -2.19,
                "value": -7974,
                "source_url": "https://www.eia.gov/petroleum/supply/weekly/",
                "observed_at": "2026-05-29T00:00:00Z",
            },
            {
                "entity": "WTI-PHYSICAL",
                "obs": "cftc/futures_positioning/mm_net",
                "z": -1.34,
                "value": 90765,
                "source_url": "https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm",
                "observed_at": "2026-06-02T00:00:00Z",
            },
        ],
        "issued_at": "2026-06-10T08:00:00Z",
        "readout_instrument": "CL=F",
        "evaluation_window_days": 5,
        "chain_score": 1.71,
        "outcome": None,
    }


class TestAlertToBrief:
    def test_markdown_has_required_sections(self, sample_alert: dict) -> None:
        """Draft brief must include frontmatter and core sections."""
        md = alert_to_brief_markdown(sample_alert)
        assert "---" in md
        assert "## What happened" in md
        assert "## Why it matters" in md
        assert "## What to watch next" in md
        assert "## Sources" in md
        assert "DRAFT" in md
        assert "Chain Brief #4" in md
        assert "eia_regime_cftc" in md

    def test_pending_outcome_footer(self, sample_alert: dict) -> None:
        """Unresolved alerts show resolve instructions."""
        md = alert_to_brief_markdown(sample_alert)
        assert "resolve_ghost_alert.py" in md

    def test_resolved_outcome_footer(self, sample_alert: dict) -> None:
        """Resolved alerts show return in footer."""
        sample_alert["outcome"] = {
            "direction": "up",
            "return_pct": 4.19,
            "notes": "2-session return",
        }
        md = alert_to_brief_markdown(sample_alert)
        assert "+4.19%" in md
        assert "Resolved" in md


class TestWriteBrief:
    def test_write_brief_creates_file(self, sample_alert: dict, tmp_path: Path) -> None:
        """write_brief_for_alert writes markdown to briefs dir."""
        out = write_brief_for_alert(sample_alert, briefs_dir=tmp_path)
        assert out.exists()
        assert out.name.endswith("_CHAIN_BRIEF_004.md")

    def test_brief_path_matches_alert_seq(self, sample_alert: dict) -> None:
        """Draft brief filename sequence matches alert_id suffix."""
        path = brief_path_for_alert(sample_alert, draft=True)
        assert path.name == "2026-06-10_MP-1_CHAIN_BRIEF_004.md"
        assert "draft" in str(path)


class TestUpdateOutcome:
    def test_update_brief_outcome_replaces_footer(
        self, sample_alert: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Resolution updates the outcome line in draft and publish briefs."""
        draft_dir = tmp_path / "draft"
        pub_dir = tmp_path / "pub"
        monkeypatch.setattr("agent.quant.ghost_brief.DRAFT_BRIEFS_DIR", draft_dir)
        monkeypatch.setattr("agent.quant.ghost_brief.PUBLISH_BRIEFS_DIR", pub_dir)
        write_brief_for_alert(sample_alert, briefs_dir=draft_dir, draft=True)
        write_brief_for_alert(sample_alert, briefs_dir=pub_dir, draft=False, overwrite=True)
        sample_alert["outcome"] = {
            "direction": "down",
            "return_pct": -4.15,
            "notes": "5-session return CL=F",
        }
        paths = update_brief_outcome(sample_alert)
        assert len(paths) == 2
        for path in paths:
            text = path.read_text(encoding="utf-8")
            assert "-4.15%" in text
            assert "resolve_ghost_alert.py" not in text


class TestRealArchiveShape:
    def test_load_real_alert_json(self) -> None:
        """Existing archive alert produces valid brief without error."""
        path = Path("ghost_archive/alerts/2026-06-09_MP-1_EIA_REGIME_CFTC_001.json")
        if not path.exists():
            pytest.skip("archive alert not in workspace")
        alert = json.loads(path.read_text(encoding="utf-8"))
        md = alert_to_brief_markdown(alert)
        assert alert["alert_id"] in md
