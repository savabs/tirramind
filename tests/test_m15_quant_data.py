"""M15 ingest: options chain, dividends, rate curve persistence."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agent.tools.dividend_data import ingest_dividends
from agent.tools.m15_universe import OPTIONS_MUST, all_options_tickers
from agent.tools.options_chain import ingest_options_chains, summarize_chain
from agent.tools.sovereign_debt import SovereignDebtTool


def test_summarize_chain_atm_iv():
    calls = pd.DataFrame(
        {
            "strike": [90.0, 100.0, 110.0],
            "impliedVolatility": [0.2, 0.18, 0.22],
            "openInterest": [10, 100, 5],
        }
    )
    puts = pd.DataFrame(
        {
            "strike": [90.0, 100.0, 110.0],
            "impliedVolatility": [0.25, 0.19, 0.21],
            "openInterest": [50, 80, 20],
        }
    )
    s = summarize_chain("SPY", "2026-06-20", calls, puts, spot=100.0)
    assert s["underlying"] == "SPY"
    assert s["atm_call_iv"] == pytest.approx(0.18, rel=1e-3)
    assert s["put_call_oi_ratio"] == pytest.approx(150 / 115, rel=1e-2)


def test_ingest_options_chains_mocked(tmp_path):
    store = MagicMock()
    summary = {
        "underlying": "SPY",
        "expiry": "2026-06-20",
        "spot": 500.0,
        "n_calls": 10,
        "n_puts": 10,
        "atm_call_iv": 0.15,
        "atm_put_iv": 0.16,
        "put_call_oi_ratio": 1.1,
        "total_open_interest": 1000.0,
        "fetched_at": "2026-06-03",
    }
    with patch(
        "agent.tools.options_chain.fetch_chain_snapshot", return_value=summary
    ):
        out = ingest_options_chains(store, tickers=["SPY"], include_should=False)
    assert out["stored"] == 1
    assert store.store_entity_observation.called


def test_ingest_dividends_mocked(tmp_path):
    store = MagicMock()
    idx = pd.to_datetime(["2025-12-19", "2026-03-20"], utc=True)
    div = pd.Series([1.99, 1.80], index=idx)
    mock_ticker = MagicMock()
    mock_ticker.dividends = div
    with patch("yfinance.Ticker", return_value=mock_ticker):
        out = ingest_dividends(store, tickers=["SPY"], include_should=False)
    assert out["observations_stored"] == 2


def test_sovereign_us_yields_persists_full_month(tmp_path):
    from agent.pipeline.store import PipelineStore

    db = tmp_path / "t.db"
    store = PipelineStore(str(db))
    tool = SovereignDebtTool(pipeline_store=store)
    data = {
        "records": [
            {
                "date": "2026-03-01",
                "yields": {"2y": 4.0, "10y": 4.5},
                "curve_2s10s": 0.5,
                "curve_3m10y": None,
            },
            {
                "date": "2026-03-02",
                "yields": {"2y": 4.1, "10y": 4.6},
                "curve_2s10s": 0.5,
                "curve_3m10y": None,
            },
        ]
    }
    counts = tool._persist_entities(data, "us_yields")
    assert counts["sovereign_yield_obs"] == 2
    obs = store.query_all_observations()
    assert all(o["observation_type"] == "sovereign_yield" for o in obs)
    assert obs[0]["value"]["yields"]["10y"] == 4.5
    store.close()


def test_options_must_subset():
    assert "SPY" in OPTIONS_MUST
    assert len(all_options_tickers(include_should=False)) >= 2
