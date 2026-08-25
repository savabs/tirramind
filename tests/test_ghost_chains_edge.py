"""Edge tests for agent.quant.ghost_chains."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from agent.quant.ghost_chains import (
    _cftc_contract_for_ticker,
    _country_entity_ids,
    _load_ais_daily_counts,
    chain_to_alert,
    load_chain_template,
    match_chain,
    rolling_zscore,
)


class TestRollingZscore:
    def test_flat_series_returns_zero(self) -> None:
        s = np.ones(20)
        assert rolling_zscore(s, 10, 10) == 0.0

    def test_spike_positive_z(self) -> None:
        s = np.array([1.0, 1.1, 0.9, 1.0, 1.2, 0.8, 1.1, 0.9, 1.0, 1.1, 5.0])
        z = rolling_zscore(s, 10, 10)
        assert z > 2.0


class TestLoadTemplate:
    def test_load_mp1_template(self) -> None:
        path = Path("templates/ghost_chains/mp1/ais_eia_cftc.yaml")
        if not path.exists():
            pytest.skip("template not in workspace")
        tmpl = load_chain_template(path)
        assert tmpl.id == "ais_eia_cftc"
        assert tmpl.micro_playground == "MP-1"
        assert len(tmpl.nodes) == 3


class TestChainMatch:
    def _make_db(self) -> sqlite3.Connection:
        con = sqlite3.connect(":memory:")
        con.executescript(
            """
            CREATE TABLE entities (
                entity_id TEXT PRIMARY KEY,
                entity_type TEXT,
                canonical_name TEXT,
                metadata_json TEXT
            );
            CREATE TABLE entity_observations (
                entity_id TEXT,
                source_tool TEXT,
                observed_at TEXT,
                observation_type TEXT,
                value_json TEXT
            );
            CREATE TABLE entity_links (
                entity_id_a TEXT, entity_id_b TEXT, link_type TEXT
            );
            """
        )
        return con

    def test_match_fails_on_empty_db(self) -> None:
        path = Path("templates/ghost_chains/mp1/eia_regime_cftc.yaml")
        if not path.exists():
            pytest.skip("template not in workspace")
        tmpl = load_chain_template(path)
        con = self._make_db()
        assert match_chain(con, tmpl) is None
        con.close()

    def test_chain_to_alert_shape(self) -> None:
        from agent.quant.ghost_chains import ChainMatch, ChainTemplate, NodeMatch

        node = NodeMatch(
            node_id="a",
            entity="WTI",
            obs="cftc/futures_positioning/mm_net",
            z=2.5,
            value=100.0,
            observed_at="2026-06-01T00:00:00Z",
            source_url="https://www.cftc.gov/",
        )
        tmpl = ChainTemplate(
            id="test",
            micro_playground="MP-1",
            description="t",
            readout_instrument="CL=F",
            evaluation_window_days=21,
            max_chain_lag_days=14,
            min_chain_score=2.0,
            nodes=(),
        )
        m = ChainMatch(
            template=tmpl,
            nodes=[node, node],
            chain_score=2.5,
            issued_at="2026-06-08T08:00:00Z",
        )
        alert = chain_to_alert(m)
        assert alert["outcome"] is None
        assert len(alert["nodes"]) == 2
        assert alert["micro_playground"] == "MP-1"


class TestCftcMapping:
    def test_cl_prefers_wti_physical_entity(self) -> None:
        db = Path(".tirra_pipeline/pipeline.db")
        if not db.exists():
            pytest.skip("pipeline.db not available")
        con = sqlite3.connect(str(db))
        eid = _cftc_contract_for_ticker(con, "CL=F")
        con.close()
        assert eid is not None
        con = sqlite3.connect(str(db))
        row = con.execute(
            "SELECT canonical_name FROM entities WHERE entity_id=?", (eid,)
        ).fetchone()
        con.close()
        assert row is not None
        assert "WTI-PHYSICAL" in row[0]


class TestGdeltCountries:
    def test_iso2_resolves_to_canonical_country_names(self) -> None:
        db = Path(".tirra_pipeline/pipeline.db")
        if not db.exists():
            pytest.skip("pipeline.db not available")
        con = sqlite3.connect(str(db))
        eids = _country_entity_ids(con, ("SA", "RU"))
        con.close()
        assert len(eids) >= 2


class TestAisDailySeries:
    def _make_db(self) -> sqlite3.Connection:
        con = sqlite3.connect(":memory:")
        con.execute(
            """
            CREATE TABLE entity_observations (
                entity_id TEXT,
                source_tool TEXT,
                observed_at TEXT,
                observation_type TEXT,
                value_json TEXT
            )
            """
        )
        return con

    def test_prefers_proxy_over_vessel_position_spam(self) -> None:
        con = self._make_db()
        for i in range(10):
            day = f"2026-05-{10 + i:02d}"
            ts = f"{day}T12:00:00+00:00"
            con.execute(
                "INSERT INTO entity_observations VALUES (?,?,?,?,?)",
                (
                    "area",
                    "ais_vessel",
                    ts,
                    "baltic_activity_proxy",
                    json.dumps({"tanker_count": 10 + i, "series_count": 10 + i}),
                ),
            )
        for _ in range(200):
            con.execute(
                "INSERT INTO entity_observations VALUES (?,?,?,?,?)",
                (
                    "v",
                    "ais_vessel",
                    "2026-06-09T12:00:00+00:00",
                    "vessel_position",
                    "{}",
                ),
            )
        times, values, label = _load_ais_daily_counts(con)
        con.close()
        assert len(times) == 10
        assert values[0] == 10.0
        assert values[-1] == 19.0
        assert "proxy" in label.lower()
