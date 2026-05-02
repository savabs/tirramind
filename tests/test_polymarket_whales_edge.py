"""
TirraMind — Polymarket Whale Tracking: Comprehensive Edge Case Tests

Covers:
  - fetch_recent_trades: normal, empty, error, micro-market filter, malformed data
  - index_trades: dedup, empty upstream, missing fields
  - track_resolutions: normal, dedup, ambiguous prices, error
  - score_wallets: Bayesian accuracy, profit factor, composite formula, edge cases
  - detect_signals: whale alerts, consensus, contrarian, thresholds, cold start
  - DAG structure: validation, schedule, dependencies
  - Agent tool: all 4 modes, cold start, validation
  - Integration: CLI registration
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.pipeline.dags.whale_tracking import (
    _CONSENSUS_MIN_WALLETS,
    _MICRO_RE,
    _WHALE_ALERT_MIN_USDC,
    _resolve_winner,
    _to_float,
    build_whale_scoring_dag,
    build_whale_tracking_dag,
    detect_signals,
    fetch_recent_trades,
    index_trades,
    score_wallets,
    track_resolutions,
)
from agent.pipeline.store import PipelineStore
from agent.tools.polymarket_whales import PolymarketWhalesTool

# ═══════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def mem_store(tmp_path: Path) -> PipelineStore:
    """In-memory PipelineStore for tests."""
    return PipelineStore(":memory:")


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """Temp DB path for functions that create their own store."""
    return str(tmp_path / "test.db")


def _make_trade(
    tx_hash: str = "0xabc123",
    wallet: str = "0xwallet1",
    condition_id: str = "0xcond1",
    side: str = "BUY",
    size: float = 100.0,
    price: float = 0.65,
    outcome: str = "Yes",
    outcome_index: int = 0,
    title: str = "Will X happen?",
    slug: str = "will-x-happen",
    timestamp: float | None = None,
) -> dict:
    """Factory for trade dicts matching data-api format."""
    ts = timestamp or time.time()
    return {
        "transactionHash": tx_hash,
        "proxyWallet": wallet,
        "conditionId": condition_id,
        "side": side,
        "size": str(size),
        "price": str(price),
        "timestamp": str(ts),
        "outcome": outcome,
        "outcomeIndex": outcome_index,
        "title": title,
        "slug": slug,
        "eventSlug": "event-x",
        "name": "",
        "pseudonym": "Trader123",
    }


def _make_parsed_trade(**kwargs: Any) -> dict:
    """Factory for parsed trade dicts (as returned by fetch_recent_trades)."""
    defaults = {
        "tx_hash": "0xabc123",
        "wallet": "0xwallet1",
        "condition_id": "0xcond1",
        "side": "BUY",
        "size": 100.0,
        "price": 0.65,
        "timestamp": time.time(),
        "outcome": "Yes",
        "outcome_index": 0,
        "title": "Will X happen?",
        "slug": "will-x-happen",
        "event_slug": "event-x",
        "usdc_value": 65.0,
        "name": "",
        "pseudonym": "Trader123",
    }
    defaults.update(kwargs)
    return defaults


def _gamma_resolved_event(
    condition_id: str = "0xcond1",
    winning_index: int = 0,
    title: str = "Did X happen?",
) -> dict:
    """Factory for Gamma API resolved event."""
    prices = ["0.000001", "0.999999"] if winning_index == 1 else ["0.999999", "0.000001"]
    return {
        "title": title,
        "markets": [
            {
                "conditionId": condition_id,
                "question": title,
                "slug": "did-x-happen",
                "outcomePrices": json.dumps(prices),
                "outcomes": json.dumps(["Yes", "No"]),
            }
        ],
    }


# ═══════════════════════════════════════════════════════════════
#  _to_float helper
# ═══════════════════════════════════════════════════════════════


class TestToFloat:
    def test_int(self):
        assert _to_float(42) == 42.0

    def test_float(self):
        assert _to_float(3.14) == 3.14

    def test_string_number(self):
        assert _to_float("100.5") == 100.5

    def test_none(self):
        assert _to_float(None) == 0.0

    def test_empty_string(self):
        assert _to_float("") == 0.0

    def test_nan(self):
        assert _to_float(float("nan")) == 0.0

    def test_garbage(self):
        assert _to_float("not_a_number") == 0.0

    def test_negative(self):
        assert _to_float("-5.5") == -5.5


# ═══════════════════════════════════════════════════════════════
#  _resolve_winner helper
# ═══════════════════════════════════════════════════════════════


class TestResolveWinner:
    def test_yes_wins(self):
        mkt = {"outcomePrices": '["0.999999", "0.000001"]'}
        assert _resolve_winner(mkt) == 0

    def test_no_wins(self):
        mkt = {"outcomePrices": '["0.000001", "0.999999"]'}
        assert _resolve_winner(mkt) == 1

    def test_ambiguous_prices(self):
        """Prices not near 0 or 1 → None (unresolved)."""
        mkt = {"outcomePrices": '["0.55", "0.45"]'}
        assert _resolve_winner(mkt) is None

    def test_empty_prices(self):
        assert _resolve_winner({"outcomePrices": ""}) is None

    def test_no_prices_key(self):
        assert _resolve_winner({}) is None

    def test_malformed_json(self):
        assert _resolve_winner({"outcomePrices": "not json"}) is None

    def test_single_outcome(self):
        assert _resolve_winner({"outcomePrices": '["0.999"]'}) is None

    def test_three_outcomes(self):
        mkt = {"outcomePrices": '["0.000001", "0.999999", "0.000001"]'}
        assert _resolve_winner(mkt) == 1

    def test_list_not_string(self):
        """outcomePrices already parsed as list."""
        mkt = {"outcomePrices": [0.999999, 0.000001]}
        assert _resolve_winner(mkt) == 0

    def test_barely_resolved(self):
        """Price at exactly 0.91 should pass the > 0.9 threshold."""
        mkt = {"outcomePrices": '["0.91", "0.09"]'}
        assert _resolve_winner(mkt) == 0

    def test_below_threshold(self):
        """Price at 0.89 should NOT pass."""
        mkt = {"outcomePrices": '["0.89", "0.11"]'}
        assert _resolve_winner(mkt) is None


# ═══════════════════════════════════════════════════════════════
#  _MICRO_RE pattern
# ═══════════════════════════════════════════════════════════════


class TestMicroPattern:
    def test_matches_standard(self):
        assert _MICRO_RE.search("Bitcoin Up or Down on June 1?")

    def test_matches_case_insensitive(self):
        assert _MICRO_RE.search("ETH UP OR DOWN")

    def test_no_match_normal(self):
        assert not _MICRO_RE.search("Will Trump win the election?")

    def test_no_match_partial(self):
        assert not _MICRO_RE.search("Up and Down")


# ═══════════════════════════════════════════════════════════════
#  fetch_recent_trades
# ═══════════════════════════════════════════════════════════════


class TestFetchRecentTrades:
    @patch("agent.pipeline.dags.whale_tracking.httpx.Client")
    def test_normal_fetch(self, mock_client_cls):
        """Happy path: 3 trades, 1 micro filtered."""
        raw = [
            _make_trade(tx_hash="0x1", title="Real market question"),
            _make_trade(tx_hash="0x2", title="BTC Up or Down in 15 min?"),
            _make_trade(tx_hash="0x3", title="Another real market"),
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = raw
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = fetch_recent_trades({}, {})
        assert result["count"] == 2
        assert len(result["trades"]) == 2
        # Verify micro-market was filtered
        titles = [t["title"] for t in result["trades"]]
        assert all("Up or Down" not in t for t in titles)

    @patch("agent.pipeline.dags.whale_tracking.httpx.Client")
    def test_skip_micro_disabled(self, mock_client_cls):
        """skip_micro=False keeps all trades."""
        raw = [
            _make_trade(tx_hash="0x1", title="BTC Up or Down?"),
            _make_trade(tx_hash="0x2", title="Real market"),
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = raw
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = fetch_recent_trades({"skip_micro": False}, {})
        assert result["count"] == 2

    @patch("agent.pipeline.dags.whale_tracking.httpx.Client")
    def test_empty_response(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = fetch_recent_trades({}, {})
        assert result["count"] == 0
        assert result["trades"] == []
        assert "error" not in result

    @patch("agent.pipeline.dags.whale_tracking.httpx.Client")
    def test_http_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock())
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = fetch_recent_trades({}, {})
        assert result["count"] == 0
        assert "error" in result

    @patch("agent.pipeline.dags.whale_tracking.httpx.Client")
    def test_timeout(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.ReadTimeout("timed out")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = fetch_recent_trades({}, {})
        assert result["count"] == 0
        assert "error" in result

    @patch("agent.pipeline.dags.whale_tracking.httpx.Client")
    def test_non_list_response(self, mock_client_cls):
        """API returns unexpected type (dict instead of list)."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": "something"}
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = fetch_recent_trades({}, {})
        assert result["count"] == 0
        assert "error" in result

    @patch("agent.pipeline.dags.whale_tracking.httpx.Client")
    def test_missing_tx_hash_skipped(self, mock_client_cls):
        """Trades without tx_hash are skipped."""
        raw = [
            _make_trade(tx_hash="0x1"),
            {**_make_trade(tx_hash=""), "transactionHash": ""},
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = raw
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = fetch_recent_trades({}, {})
        assert result["count"] == 1

    @patch("agent.pipeline.dags.whale_tracking.httpx.Client")
    def test_missing_wallet_skipped(self, mock_client_cls):
        raw = [{**_make_trade(tx_hash="0x1"), "proxyWallet": ""}]
        mock_resp = MagicMock()
        mock_resp.json.return_value = raw
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = fetch_recent_trades({}, {})
        assert result["count"] == 0

    @patch("agent.pipeline.dags.whale_tracking.httpx.Client")
    def test_usdc_value_computed(self, mock_client_cls):
        raw = [_make_trade(tx_hash="0x1", size=200.0, price=0.5)]
        mock_resp = MagicMock()
        mock_resp.json.return_value = raw
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = fetch_recent_trades({}, {})
        assert result["trades"][0]["usdc_value"] == 100.0

    @patch("agent.pipeline.dags.whale_tracking.httpx.Client")
    def test_string_numeric_fields(self, mock_client_cls):
        """data-api returns size/price/timestamp as strings."""
        raw = [_make_trade(tx_hash="0x1", size=50.5, price=0.75)]
        mock_resp = MagicMock()
        mock_resp.json.return_value = raw
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = fetch_recent_trades({}, {})
        trade = result["trades"][0]
        assert isinstance(trade["size"], float)
        assert isinstance(trade["price"], float)
        assert isinstance(trade["timestamp"], float)


# ═══════════════════════════════════════════════════════════════
#  index_trades
# ═══════════════════════════════════════════════════════════════


class TestIndexTrades:
    def test_normal_indexing(self, db_path):
        upstream = {
            "fetch_recent_trades": {
                "trades": [
                    _make_parsed_trade(tx_hash="0x1"),
                    _make_parsed_trade(tx_hash="0x2"),
                ],
                "count": 2,
            }
        }
        result = index_trades({"db_path": db_path}, upstream)
        assert result["indexed"] == 2
        assert result["duplicates"] == 0
        assert result["total_seen"] == 2

        # Verify stored in DB
        store = PipelineStore(db_path)
        rows = store.query_data("pm_trades", limit=10)
        assert len(rows) == 2
        store.close()

    def test_dedup_within_batch(self, db_path):
        """Same tx_hash twice in one batch → only one stored."""
        upstream = {
            "fetch_recent_trades": {
                "trades": [
                    _make_parsed_trade(tx_hash="0x1"),
                    _make_parsed_trade(tx_hash="0x1"),
                ],
                "count": 2,
            }
        }
        result = index_trades({"db_path": db_path}, upstream)
        assert result["indexed"] == 1
        assert result["duplicates"] == 1

    def test_dedup_across_runs(self, db_path):
        """Trades from previous run are not re-indexed."""
        upstream = {
            "fetch_recent_trades": {
                "trades": [_make_parsed_trade(tx_hash="0xexist")],
                "count": 1,
            }
        }
        # First run
        index_trades({"db_path": db_path}, upstream)
        # Second run
        result = index_trades({"db_path": db_path}, upstream)
        assert result["indexed"] == 0
        assert result["duplicates"] == 1

    def test_empty_upstream(self, db_path):
        result = index_trades({"db_path": db_path}, {})
        assert result == {"indexed": 0, "duplicates": 0, "total_seen": 0}

    def test_empty_trades(self, db_path):
        upstream = {"fetch_recent_trades": {"trades": [], "count": 0}}
        result = index_trades({"db_path": db_path}, upstream)
        assert result == {"indexed": 0, "duplicates": 0, "total_seen": 0}

    def test_error_upstream(self, db_path):
        """Upstream had an error → empty trades."""
        upstream = {"fetch_recent_trades": {"trades": [], "count": 0, "error": "timeout"}}
        result = index_trades({"db_path": db_path}, upstream)
        assert result["indexed"] == 0

    def test_trade_fields_stored(self, db_path):
        """Verify all trade fields are preserved in DB."""
        trade = _make_parsed_trade(
            tx_hash="0xfull",
            wallet="0xwallet99",
            condition_id="0xcond99",
            side="SELL",
            size=500.0,
            price=0.3,
            outcome="No",
        )
        upstream = {"fetch_recent_trades": {"trades": [trade], "count": 1}}
        index_trades({"db_path": db_path}, upstream)

        store = PipelineStore(db_path)
        rows = store.query_data("pm_trades", limit=1)
        assert len(rows) == 1
        stored = rows[0]["data"]
        assert stored["wallet"] == "0xwallet99"
        assert stored["side"] == "SELL"
        assert stored["size"] == 500.0
        store.close()


# ═══════════════════════════════════════════════════════════════
#  track_resolutions
# ═══════════════════════════════════════════════════════════════


class TestTrackResolutions:
    @patch("agent.pipeline.dags.whale_tracking.httpx.Client")
    def test_normal_resolutions(self, mock_client_cls, db_path):
        events = [
            _gamma_resolved_event("0xcond1", winning_index=0),
            _gamma_resolved_event("0xcond2", winning_index=1),
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = events
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = track_resolutions({"db_path": db_path}, {})
        assert result["resolved"] == 2
        assert result["new"] == 2

    @patch("agent.pipeline.dags.whale_tracking.httpx.Client")
    def test_dedup_resolutions(self, mock_client_cls, db_path):
        events = [_gamma_resolved_event("0xcond1", winning_index=0)]
        mock_resp = MagicMock()
        mock_resp.json.return_value = events
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        track_resolutions({"db_path": db_path}, {})
        result = track_resolutions({"db_path": db_path}, {})
        assert result["new"] == 0

    @patch("agent.pipeline.dags.whale_tracking.httpx.Client")
    def test_ambiguous_prices_skipped(self, mock_client_cls, db_path):
        """Markets with ambiguous prices (not near 0/1) are skipped."""
        events = [
            {
                "title": "Ambiguous",
                "markets": [
                    {
                        "conditionId": "0xcond1",
                        "question": "Ambiguous",
                        "outcomePrices": '["0.5", "0.5"]',
                    }
                ],
            }
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = events
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = track_resolutions({"db_path": db_path}, {})
        assert result["resolved"] == 0
        assert result["new"] == 0

    @patch("agent.pipeline.dags.whale_tracking.httpx.Client")
    def test_http_error(self, mock_client_cls, db_path):
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.ConnectError("connection failed")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = track_resolutions({"db_path": db_path}, {})
        assert result["resolved"] == 0
        assert "error" in result

    @patch("agent.pipeline.dags.whale_tracking.httpx.Client")
    def test_missing_condition_id_skipped(self, mock_client_cls, db_path):
        events = [
            {
                "title": "Test",
                "markets": [
                    {
                        "conditionId": "",
                        "outcomePrices": '["0.999", "0.001"]',
                    }
                ],
            }
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = events
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = track_resolutions({"db_path": db_path}, {})
        assert result["new"] == 0


# ═══════════════════════════════════════════════════════════════
#  score_wallets
# ═══════════════════════════════════════════════════════════════


def _seed_trades_and_resolutions(
    db_path: str,
    wallets: int = 1,
    trades_per_wallet: int = 10,
    correct_pct: float = 0.7,
) -> None:
    """Seed DB with trades and matching resolutions for testing."""
    store = PipelineStore(db_path)
    now = time.time()

    for w_idx in range(wallets):
        wallet = f"0xwallet{w_idx:04d}"
        for t_idx in range(trades_per_wallet):
            cid = f"0xcond_{w_idx}_{t_idx}"
            is_correct = t_idx < int(trades_per_wallet * correct_pct)
            # If correct: BUY on outcome_index=0, winning_index=0
            # If wrong: BUY on outcome_index=0, winning_index=1
            trade = {
                "tx_hash": f"0xtx_{w_idx}_{t_idx}",
                "wallet": wallet,
                "condition_id": cid,
                "side": "BUY",
                "size": 100.0,
                "price": 0.6,
                "timestamp": now - (trades_per_wallet - t_idx) * 3600,
                "outcome": "Yes",
                "outcome_index": 0,
                "title": f"Test market {t_idx}",
                "slug": f"test-{t_idx}",
                "event_slug": "test-event",
                "usdc_value": 60.0,
                "name": "",
                "pseudonym": "",
            }
            store.store_data("pm_trades", {"tx_hash": trade["tx_hash"], "wallet": wallet, "condition_id": cid}, trade)

            # Store resolution
            winning_idx = 0 if is_correct else 1
            store.store_data(
                "pm_resolutions",
                {"condition_id": cid},
                {"condition_id": cid, "winning_index": winning_idx, "title": f"Test {t_idx}"},
            )

    store.close()


class TestScoreWallets:
    def test_normal_scoring(self, db_path):
        _seed_trades_and_resolutions(db_path, wallets=3, trades_per_wallet=10, correct_pct=0.7)
        result = score_wallets({"db_path": db_path}, {})
        assert result["scored"] == 3
        assert len(result["top_10"]) == 3

    def test_no_resolutions(self, db_path):
        # Store trades but no resolutions
        store = PipelineStore(db_path)
        for i in range(5):
            store.store_data(
                "pm_trades",
                {"tx_hash": f"0x{i}", "wallet": "0xw1", "condition_id": f"0xc{i}"},
                {
                    "tx_hash": f"0x{i}",
                    "wallet": "0xw1",
                    "condition_id": f"0xc{i}",
                    "side": "BUY",
                    "size": 10,
                    "price": 0.5,
                    "timestamp": time.time(),
                    "outcome": "Yes",
                    "outcome_index": 0,
                    "usdc_value": 5,
                },
            )
        store.close()

        result = score_wallets({"db_path": db_path}, {})
        assert result["scored"] == 0
        assert "no resolutions" in result.get("note", "")

    def test_below_min_threshold(self, db_path):
        """Wallet with fewer than MIN_RESOLVED_TRADES is not scored."""
        _seed_trades_and_resolutions(db_path, wallets=1, trades_per_wallet=3)
        result = score_wallets({"db_path": db_path}, {})
        assert result["scored"] == 0

    def test_bayesian_accuracy_all_correct(self, db_path):
        """All 10 correct → Bayesian = (10+1)/(10+2) = 0.9167."""
        _seed_trades_and_resolutions(db_path, wallets=1, trades_per_wallet=10, correct_pct=1.0)
        result = score_wallets({"db_path": db_path}, {})
        assert result["scored"] == 1
        # Can't check top_10 directly since it's truncated, check DB
        store = PipelineStore(db_path)
        rows = store.query_data("pm_wallet_scores", limit=1)
        wallets = rows[0]["data"]["wallets"]
        assert len(wallets) == 1
        assert abs(wallets[0]["accuracy"] - 11 / 12) < 0.01
        store.close()

    def test_bayesian_accuracy_all_wrong(self, db_path):
        """All 10 wrong → Bayesian = (0+1)/(10+2) = 0.0833."""
        _seed_trades_and_resolutions(db_path, wallets=1, trades_per_wallet=10, correct_pct=0.0)
        result = score_wallets({"db_path": db_path}, {})
        store = PipelineStore(db_path)
        rows = store.query_data("pm_wallet_scores", limit=1)
        wallets = rows[0]["data"]["wallets"]
        assert abs(wallets[0]["accuracy"] - 1 / 12) < 0.01
        store.close()

    def test_profit_factor_no_losses(self, db_path):
        """All wins → losing_pnl=0 → profit_factor clamped to avoid div/0."""
        _seed_trades_and_resolutions(db_path, wallets=1, trades_per_wallet=10, correct_pct=1.0)
        result = score_wallets({"db_path": db_path}, {})
        store = PipelineStore(db_path)
        rows = store.query_data("pm_wallet_scores", limit=1)
        wallets = rows[0]["data"]["wallets"]
        # profit_factor = winning_pnl / 0.01 → very large number
        assert wallets[0]["profit_factor"] > 1000
        store.close()

    def test_recency_decay(self, db_path):
        """Wallet with old trades should have lower recency."""
        store = PipelineStore(db_path)
        now = time.time()
        old_ts = now - 60 * 86400  # 60 days ago

        for i in range(10):
            cid = f"0xold_cond_{i}"
            trade = {
                "tx_hash": f"0xold_{i}",
                "wallet": "0xold_wallet",
                "condition_id": cid,
                "side": "BUY",
                "size": 100,
                "price": 0.6,
                "timestamp": old_ts,
                "outcome": "Yes",
                "outcome_index": 0,
                "usdc_value": 60,
            }
            store.store_data(
                "pm_trades", {"tx_hash": f"0xold_{i}", "wallet": "0xold_wallet", "condition_id": cid}, trade
            )
            store.store_data("pm_resolutions", {"condition_id": cid}, {"condition_id": cid, "winning_index": 0})
        store.close()

        result = score_wallets({"db_path": db_path}, {})
        store = PipelineStore(db_path)
        rows = store.query_data("pm_wallet_scores", limit=1)
        wallets = rows[0]["data"]["wallets"]
        # At 60 days, recency = exp(-ln(2)/30 * 60) = exp(-2*ln(2)) = 0.25
        assert abs(wallets[0]["recency"] - 0.25) < 0.05
        store.close()

    def test_composite_formula(self, db_path):
        """Verify composite = accuracy * log(1+vol) * recency * sqrt(markets)."""
        _seed_trades_and_resolutions(db_path, wallets=1, trades_per_wallet=10, correct_pct=0.7)
        result = score_wallets({"db_path": db_path}, {})
        store = PipelineStore(db_path)
        rows = store.query_data("pm_wallet_scores", limit=1)
        w = rows[0]["data"]["wallets"][0]

        expected = w["accuracy"] * math.log(1 + w["total_volume"]) * w["recency"] * math.sqrt(w["markets"])
        assert abs(w["composite"] - round(expected, 4)) < 0.01
        store.close()

    def test_empty_db(self, db_path):
        result = score_wallets({"db_path": db_path}, {})
        assert result["scored"] == 0

    def test_sell_side_correct(self, db_path):
        """SELL on non-winning outcome is correct."""
        store = PipelineStore(db_path)
        now = time.time()
        wallet = "0xseller"

        for i in range(10):
            cid = f"0xsell_cond_{i}"
            trade = {
                "tx_hash": f"0xsell_{i}",
                "wallet": wallet,
                "condition_id": cid,
                "side": "SELL",
                "size": 100,
                "price": 0.6,
                "timestamp": now,
                "outcome": "No",
                "outcome_index": 1,
                "usdc_value": 60,
            }
            store.store_data("pm_trades", {"tx_hash": f"0xsell_{i}", "wallet": wallet, "condition_id": cid}, trade)
            # outcome_index=1 (No), winning_index=0 (Yes) → SELL on non-winner = correct
            store.store_data("pm_resolutions", {"condition_id": cid}, {"condition_id": cid, "winning_index": 0})
        store.close()

        result = score_wallets({"db_path": db_path}, {})
        store = PipelineStore(db_path)
        rows = store.query_data("pm_wallet_scores", limit=1)
        wallets = rows[0]["data"]["wallets"]
        assert wallets[0]["correct"] == 10
        store.close()


# ═══════════════════════════════════════════════════════════════
#  detect_signals
# ═══════════════════════════════════════════════════════════════


def _seed_whale_scenario(db_path: str) -> None:
    """Seed DB with scored wallets + recent trades for signal detection."""
    store = PipelineStore(db_path)
    now = time.time()

    # Create wallet scores (top 50 wallets)
    wallets = []
    for i in range(60):
        wallets.append(
            {
                "wallet": f"0xwhale{i:04d}",
                "composite": 100.0 - i,
                "accuracy": 0.8,
                "correct": 8,
                "total_resolved": 10,
                "profit_factor": 2.0,
                "total_volume": 50000.0,
                "markets": 20,
                "recency": 0.95,
                "latest_trade_ts": now,
            }
        )
    store.store_data("pm_wallet_scores", {"batch": "latest"}, {"wallets": wallets, "scored_at": now})

    # Create recent trades:

    # Whale alert: top wallet, big trade
    big_trade = _make_parsed_trade(
        tx_hash="0xbig",
        wallet="0xwhale0000",
        condition_id="0xmarket_a",
        side="BUY",
        size=5000,
        price=0.5,
        title="Big Market",
    )
    big_trade["usdc_value"] = 2500.0
    big_trade["timestamp"] = now - 100
    store.store_data(
        "pm_trades", {"tx_hash": "0xbig", "wallet": "0xwhale0000", "condition_id": "0xmarket_a"}, big_trade
    )

    # Consensus: 4 top wallets on same side of same market
    for i in range(4):
        t = _make_parsed_trade(
            tx_hash=f"0xcons{i}",
            wallet=f"0xwhale{i:04d}",
            condition_id="0xmarket_b",
            side="BUY",
            size=100,
            price=0.6,
            title="Consensus Market",
        )
        t["usdc_value"] = 60.0
        t["timestamp"] = now - 200
        store.store_data(
            "pm_trades",
            {"tx_hash": f"0xcons{i}", "wallet": f"0xwhale{i:04d}", "condition_id": "0xmarket_b"},
            t,
        )

    # Contrarian: top wallet buys at low price
    ct = _make_parsed_trade(
        tx_hash="0xcontrarian",
        wallet="0xwhale0001",
        condition_id="0xmarket_c",
        side="BUY",
        size=200,
        price=0.15,
        title="Underdog Market",
    )
    ct["usdc_value"] = 30.0
    ct["timestamp"] = now - 300
    store.store_data(
        "pm_trades",
        {"tx_hash": "0xcontrarian", "wallet": "0xwhale0001", "condition_id": "0xmarket_c"},
        ct,
    )

    store.close()


class TestDetectSignals:
    def test_whale_alert(self, db_path):
        _seed_whale_scenario(db_path)
        result = detect_signals({"db_path": db_path}, {})
        assert len(result["whale_alerts"]) >= 1
        # The $2500 trade by whale0000 should trigger
        alerts = [a for a in result["whale_alerts"] if a["wallet"] == "0xwhale0000"]
        assert len(alerts) == 1
        assert alerts[0]["usdc_value"] >= _WHALE_ALERT_MIN_USDC

    def test_consensus_signal(self, db_path):
        _seed_whale_scenario(db_path)
        result = detect_signals({"db_path": db_path}, {})
        consensus = [c for c in result["consensus"] if c["condition_id"] == "0xmarket_b"]
        assert len(consensus) == 1
        assert consensus[0]["wallet_count"] >= _CONSENSUS_MIN_WALLETS

    def test_contrarian_signal(self, db_path):
        _seed_whale_scenario(db_path)
        result = detect_signals({"db_path": db_path}, {})
        contrarian = [c for c in result["contrarian"] if c["condition_id"] == "0xmarket_c"]
        assert len(contrarian) == 1

    def test_no_scores_cold_start(self, db_path):
        """No wallet scores → returns empty with note."""
        store = PipelineStore(db_path)
        store.close()  # empty DB
        result = detect_signals({"db_path": db_path}, {})
        assert result["signals_emitted"] == 0
        assert "no wallet scores" in result.get("note", "")

    def test_below_whale_threshold(self, db_path):
        """Trade by top wallet but below $1000 → no whale alert."""
        store = PipelineStore(db_path)
        now = time.time()
        wallets = [
            {
                "wallet": "0xtop",
                "composite": 99,
                "accuracy": 0.8,
                "correct": 8,
                "total_resolved": 10,
                "profit_factor": 2,
                "total_volume": 5000,
                "markets": 5,
                "recency": 1,
                "latest_trade_ts": now,
            }
        ]
        store.store_data("pm_wallet_scores", {"batch": "latest"}, {"wallets": wallets, "scored_at": now})

        small_trade = _make_parsed_trade(tx_hash="0xsmall", wallet="0xtop", usdc_value=500)
        small_trade["timestamp"] = now
        store.store_data("pm_trades", {"tx_hash": "0xsmall", "wallet": "0xtop", "condition_id": "0xc1"}, small_trade)
        store.close()

        result = detect_signals({"db_path": db_path}, {})
        assert len(result["whale_alerts"]) == 0

    def test_consensus_below_threshold(self, db_path):
        """Only 2 wallets on same side → no consensus signal."""
        store = PipelineStore(db_path)
        now = time.time()
        wallets = [
            {
                "wallet": f"0xw{i}",
                "composite": 50 - i,
                "accuracy": 0.7,
                "correct": 7,
                "total_resolved": 10,
                "profit_factor": 1.5,
                "total_volume": 3000,
                "markets": 10,
                "recency": 0.9,
                "latest_trade_ts": now,
            }
            for i in range(10)
        ]
        store.store_data("pm_wallet_scores", {"batch": "latest"}, {"wallets": wallets, "scored_at": now})

        for i in range(2):  # Only 2, below threshold of 3
            t = _make_parsed_trade(tx_hash=f"0xt{i}", wallet=f"0xw{i}", condition_id="0xmkt1")
            t["timestamp"] = now
            store.store_data("pm_trades", {"tx_hash": f"0xt{i}", "wallet": f"0xw{i}", "condition_id": "0xmkt1"}, t)
        store.close()

        result = detect_signals({"db_path": db_path}, {})
        assert len(result["consensus"]) == 0

    def test_contrarian_not_triggered_mid_price(self, db_path):
        """Price at 0.5 is not contrarian (neither < 0.3 nor > 0.7)."""
        store = PipelineStore(db_path)
        now = time.time()
        wallets = [
            {
                "wallet": "0xtop2",
                "composite": 80,
                "accuracy": 0.8,
                "correct": 8,
                "total_resolved": 10,
                "profit_factor": 2,
                "total_volume": 5000,
                "markets": 5,
                "recency": 1,
                "latest_trade_ts": now,
            }
        ]
        store.store_data("pm_wallet_scores", {"batch": "latest"}, {"wallets": wallets, "scored_at": now})

        t = _make_parsed_trade(tx_hash="0xmid", wallet="0xtop2", price=0.5, side="BUY")
        t["timestamp"] = now
        store.store_data("pm_trades", {"tx_hash": "0xmid", "wallet": "0xtop2", "condition_id": "0xmkt"}, t)
        store.close()

        result = detect_signals({"db_path": db_path}, {})
        assert len(result["contrarian"]) == 0

    def test_signals_stored_in_db(self, db_path):
        """Verify signals are persisted via store.store_signal."""
        _seed_whale_scenario(db_path)
        detect_signals({"db_path": db_path}, {})

        store = PipelineStore(db_path)
        alerts = store.query_signals("pm_whale_alert", limit=10)
        consensus = store.query_signals("pm_consensus", limit=10)
        contrarian = store.query_signals("pm_contrarian", limit=10)
        assert len(alerts) >= 1
        assert len(consensus) >= 1
        assert len(contrarian) >= 1
        store.close()


# ═══════════════════════════════════════════════════════════════
#  DAG Structure
# ═══════════════════════════════════════════════════════════════


class TestDagStructure:
    def test_whale_tracking_valid(self):
        dag = build_whale_tracking_dag()
        errors = dag.validate()
        assert errors == []

    def test_whale_scoring_valid(self):
        dag = build_whale_scoring_dag()
        errors = dag.validate()
        assert errors == []

    def test_tracking_schedule(self):
        dag = build_whale_tracking_dag()
        assert dag.schedule == "*/15 * * * *"

    def test_scoring_schedule(self):
        dag = build_whale_scoring_dag()
        assert dag.schedule == "0 6 * * *"

    def test_tracking_node_count(self):
        dag = build_whale_tracking_dag()
        assert len(dag.nodes) == 3

    def test_scoring_node_count(self):
        dag = build_whale_scoring_dag()
        assert len(dag.nodes) == 2

    def test_tracking_dependencies(self):
        dag = build_whale_tracking_dag()
        assert dag.nodes["index_trades"].depends_on == ["fetch_recent_trades"]
        assert dag.nodes["detect_signals"].depends_on == ["index_trades"]
        assert dag.nodes["fetch_recent_trades"].depends_on == []

    def test_scoring_dependencies(self):
        dag = build_whale_scoring_dag()
        assert dag.nodes["score_wallets"].depends_on == ["track_resolutions"]
        assert dag.nodes["track_resolutions"].depends_on == []

    def test_tracking_topo_sort(self):
        dag = build_whale_tracking_dag()
        layers = dag.topo_sort()
        assert len(layers) == 3
        assert layers[0] == ["fetch_recent_trades"]
        assert layers[1] == ["index_trades"]
        assert layers[2] == ["detect_signals"]

    def test_scoring_topo_sort(self):
        dag = build_whale_scoring_dag()
        layers = dag.topo_sort()
        assert len(layers) == 2
        assert layers[0] == ["track_resolutions"]
        assert layers[1] == ["score_wallets"]

    def test_custom_db_path(self):
        dag = build_whale_tracking_dag(db_path="/tmp/custom.db")
        assert dag.nodes["index_trades"].params["db_path"] == "/tmp/custom.db"
        assert dag.nodes["detect_signals"].params["db_path"] == "/tmp/custom.db"

    def test_store_result_disabled(self):
        dag = build_whale_tracking_dag()
        for node in dag.nodes.values():
            assert node.store_result is False

    def test_nodes_use_callables(self):
        """All whale dag nodes use FunctionOperator (callable), not string (ToolOperator)."""
        dag = build_whale_tracking_dag()
        for node in dag.nodes.values():
            assert callable(node.operator)


# ═══════════════════════════════════════════════════════════════
#  Agent Tool — PolymarketWhalesTool
# ═══════════════════════════════════════════════════════════════


class TestPolymarketWhalesTool:
    def test_invalid_mode(self):
        tool = PolymarketWhalesTool(db_path=":memory:")
        result = tool.execute(mode="nonexistent")
        assert not result.success
        assert "Invalid mode" in result.output

    def test_tool_name(self):
        tool = PolymarketWhalesTool()
        assert tool.name == "polymarket_whales"

    def test_openai_schema(self):
        tool = PolymarketWhalesTool()
        schema = tool.to_openai_tool()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "polymarket_whales"
        assert "mode" in schema["function"]["parameters"]["properties"]
        assert "mode" in schema["function"]["parameters"]["required"]

    def test_top_wallets_empty(self):
        """Empty DB → cold start fallback (mocked)."""
        tool = PolymarketWhalesTool(db_path=":memory:")
        with patch("agent.tools.polymarket_whales.httpx.Client") as mock_cls:
            mock_resp = MagicMock()
            mock_resp.json.return_value = []
            mock_resp.raise_for_status = MagicMock()
            mock_client = MagicMock()
            mock_client.get.return_value = mock_resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_cls.return_value = mock_client

            result = tool.execute(mode="top_wallets")
            assert result.success

    def test_top_wallets_with_data(self, db_path):
        store = PipelineStore(db_path)
        wallets = [
            {
                "wallet": f"0xw{i}",
                "composite": 50 - i,
                "accuracy": 0.8,
                "correct": 8,
                "total_resolved": 10,
                "profit_factor": 2,
                "total_volume": 5000,
                "markets": 10,
                "recency": 0.9,
                "latest_trade_ts": time.time(),
            }
            for i in range(5)
        ]
        store.store_data("pm_wallet_scores", {"batch": "latest"}, {"wallets": wallets, "scored_at": time.time()})
        store.close()

        tool = PolymarketWhalesTool(db_path=db_path)
        result = tool.execute(mode="top_wallets", limit=3)
        assert result.success
        assert len(result.data["wallets"]) == 3

    def test_wallet_detail_no_wallet(self):
        tool = PolymarketWhalesTool(db_path=":memory:")
        result = tool.execute(mode="wallet_detail")
        assert not result.success
        assert "wallet address" in result.output.lower()

    def test_wallet_detail_bad_format(self):
        tool = PolymarketWhalesTool(db_path=":memory:")
        result = tool.execute(mode="wallet_detail", wallet="not_hex")
        assert not result.success

    def test_wallet_detail_not_found(self, db_path):
        tool = PolymarketWhalesTool(db_path=db_path)
        result = tool.execute(mode="wallet_detail", wallet="0xnotfound")
        assert not result.success
        assert "not found" in result.output.lower()

    def test_wallet_detail_with_data(self, db_path):
        store = PipelineStore(db_path)
        wallets = [
            {
                "wallet": "0xfound",
                "composite": 80,
                "accuracy": 0.85,
                "correct": 8,
                "total_resolved": 10,
                "profit_factor": 3,
                "total_volume": 10000,
                "markets": 15,
                "recency": 0.95,
                "latest_trade_ts": time.time(),
            }
        ]
        store.store_data("pm_wallet_scores", {"batch": "latest"}, {"wallets": wallets, "scored_at": time.time()})
        trade = _make_parsed_trade(tx_hash="0xt1", wallet="0xfound")
        store.store_data("pm_trades", {"tx_hash": "0xt1", "wallet": "0xfound", "condition_id": "0xc1"}, trade)
        store.close()

        tool = PolymarketWhalesTool(db_path=db_path)
        result = tool.execute(mode="wallet_detail", wallet="0xfound")
        assert result.success
        assert result.data["score"]["composite"] == 80

    def test_market_whales_no_market(self):
        tool = PolymarketWhalesTool(db_path=":memory:")
        result = tool.execute(mode="market_whales")
        assert not result.success
        assert "market" in result.output.lower()

    def test_market_whales_no_trades(self, db_path):
        tool = PolymarketWhalesTool(db_path=db_path)
        result = tool.execute(mode="market_whales", market="0xnonexistent")
        assert result.success
        assert result.data["trades"] == []

    def test_market_whales_by_title(self, db_path):
        store = PipelineStore(db_path)
        trade = _make_parsed_trade(tx_hash="0xm1", wallet="0xwhale1", title="Trump wins 2024?")
        store.store_data("pm_trades", {"tx_hash": "0xm1", "wallet": "0xwhale1", "condition_id": "0xc1"}, trade)
        wallets = [
            {
                "wallet": "0xwhale1",
                "composite": 50,
                "accuracy": 0.7,
                "correct": 7,
                "total_resolved": 10,
                "profit_factor": 1.5,
                "total_volume": 3000,
                "markets": 5,
                "recency": 0.9,
                "latest_trade_ts": time.time(),
            }
        ]
        store.store_data("pm_wallet_scores", {"batch": "latest"}, {"wallets": wallets, "scored_at": time.time()})
        store.close()

        tool = PolymarketWhalesTool(db_path=db_path)
        result = tool.execute(mode="market_whales", market="trump")
        assert result.success
        assert len(result.data["whales"]) == 1

    def test_recent_signals_empty(self, db_path):
        tool = PolymarketWhalesTool(db_path=db_path)
        result = tool.execute(mode="recent_signals")
        assert result.success
        assert result.data["signals"] == []

    def test_recent_signals_with_data(self, db_path):
        store = PipelineStore(db_path)
        store.store_signal("pm_whale_alert", 5000.0, {"wallet": "0xw1", "side": "BUY", "market": "Test"})
        store.store_signal("pm_consensus", 0.85, {"wallet_count": 5, "side": "BUY", "market": "Test2"})
        store.close()

        tool = PolymarketWhalesTool(db_path=db_path)
        result = tool.execute(mode="recent_signals")
        assert result.success
        assert len(result.data["signals"]) == 2

    def test_recent_signals_filter(self, db_path):
        store = PipelineStore(db_path)
        store.store_signal("pm_whale_alert", 5000.0, {"wallet": "0xw1"})
        store.store_signal("pm_consensus", 0.85, {"wallet_count": 5})
        store.close()

        tool = PolymarketWhalesTool(db_path=db_path)
        result = tool.execute(mode="recent_signals", signal_type="whale_alert")
        assert result.success
        assert all(s["type"] == "whale_alert" for s in result.data["signals"])

    def test_recent_signals_invalid_type(self, db_path):
        tool = PolymarketWhalesTool(db_path=db_path)
        result = tool.execute(mode="recent_signals", signal_type="invalid")
        assert not result.success
        assert "Invalid signal_type" in result.output

    def test_limit_clamping(self, db_path):
        tool = PolymarketWhalesTool(db_path=db_path)
        # limit=0 → clamped to 1
        # We can't easily test the internal clamping without data,
        # but at least it shouldn't crash
        result = tool.execute(mode="recent_signals", limit=0)
        assert result.success

    def test_limit_max_clamping(self, db_path):
        tool = PolymarketWhalesTool(db_path=db_path)
        result = tool.execute(mode="recent_signals", limit=9999)
        assert result.success


# ═══════════════════════════════════════════════════════════════
#  Cold Start Fallback
# ═══════════════════════════════════════════════════════════════


class TestColdStart:
    @patch("agent.tools.polymarket_whales.httpx.Client")
    def test_cold_start_live_trades(self, mock_client_cls):
        """Cold start fetches live trades and aggregates by volume."""
        raw = [
            {"proxyWallet": "0xbigw", "size": "1000", "price": "0.6", "transactionHash": "0x1"},
            {"proxyWallet": "0xbigw", "size": "500", "price": "0.8", "transactionHash": "0x2"},
            {"proxyWallet": "0xsmallw", "size": "10", "price": "0.5", "transactionHash": "0x3"},
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = raw
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        tool = PolymarketWhalesTool(db_path=":memory:")
        result = tool.execute(mode="top_wallets")
        assert result.success
        assert result.data["cold_start"] is True
        assert result.data["wallets"][0]["wallet"] == "0xbigw"

    @patch("agent.tools.polymarket_whales.httpx.Client")
    def test_cold_start_api_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.ConnectError("down")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        tool = PolymarketWhalesTool(db_path=":memory:")
        result = tool.execute(mode="top_wallets")
        assert not result.success
        assert "fallback failed" in result.output.lower()


# ═══════════════════════════════════════════════════════════════
#  DAG Registration
# ═══════════════════════════════════════════════════════════════


class TestDagRegistration:
    def test_get_default_dags_includes_whale(self):
        from agent.pipeline.dags import get_default_dags

        dags = get_default_dags(tool_registry=None)
        names = [d.name for d in dags]
        assert "whale_tracking" in names
        assert "whale_scoring" in names

    def test_registry_loads_whale_dags(self):
        from agent.pipeline.registry import DAGRegistry

        registry = DAGRegistry()
        registry.load_defaults(tool_registry=None)
        assert registry.get("whale_tracking") is not None
        assert registry.get("whale_scoring") is not None


# ═══════════════════════════════════════════════════════════════
#  CLI Registration
# ═══════════════════════════════════════════════════════════════


class TestCLIRegistration:
    def test_tool_in_registry(self):
        from agent.cli import build_tool_registry

        registry = build_tool_registry()
        tool = registry.get("polymarket_whales")
        assert tool is not None
        assert tool.name == "polymarket_whales"

    def test_tool_count_increased(self):
        """Registry should have polymarket_whales alongside polymarket."""
        from agent.cli import build_tool_registry

        registry = build_tool_registry()
        names = registry.list_names()
        assert "polymarket" in names
        assert "polymarket_whales" in names
