"""Tests for Phase 24f — Paper Trade Launch (E2E Integration + Alerts).

Covers:
    - Full chain smoke test: daily_collection → gnn → scoring → sac → emit
      (with mocked yfinance + synthetic entity data)
    - Alert conditions: concentration, drawdown, Sharpe, edge decay
    - Schedule verification: inference DAG fires after all upstream DAGs
    - portfolio_weights and paper_trade_pnl tables populated after chain execution
"""

from __future__ import annotations

import math
import time
from datetime import date, timedelta, timezone; UTC = timezone.utc
from pathlib import Path

import numpy as np
import pytest

from agent.pipeline.dags.inference import (
    _DRAWDOWN_THRESHOLD,
    _SHARPE_THRESHOLD,
    _check_concentration,
    _check_drawdown,
    _check_edge_decay,
    _check_sharpe,
    _emit_portfolio,
    build_inference_dag,
)
from agent.pipeline.store import PipelineStore

# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path: Path) -> PipelineStore:
    """In-memory store for testing."""
    return PipelineStore(str(tmp_path / "test.db"))


def _register_instruments(store: PipelineStore, tickers: list[str]) -> None:
    """Register instrument entities in the store."""
    for t in tickers:
        store.register_entity("instrument", f"{t} Fund", t)


def _store_daily_returns(
    store: PipelineStore,
    tickers: list[str],
    as_of: str,
    returns: dict[str, float] | None = None,
) -> None:
    """Store daily_return observations for given tickers on a date."""
    from datetime import datetime

    ts = datetime.fromisoformat(as_of).replace(hour=16, tzinfo=UTC).timestamp()

    rng = np.random.default_rng(42)
    for t in tickers:
        lr = returns.get(t, float(rng.standard_normal() * 0.01)) if returns else float(rng.standard_normal() * 0.01)
        store.store_entity_observation(t, "instrument_universe", ts, "daily_return", {"log_return": lr})


def _seed_pnl_history(
    store: PipelineStore,
    n_days: int = 40,
    daily_return: float = -0.002,
    start_date: str = "2026-02-01",
) -> None:
    """Seed N days of paper_trade_pnl history with slight noise."""
    rng = np.random.default_rng(123)
    d = date.fromisoformat(start_date)
    cumulative = 0.0
    for i in range(n_days):
        day_str = (d + timedelta(days=i)).isoformat()
        # Add small noise so std > 0 (avoids zero-variance guard)
        ret = daily_return + rng.standard_normal() * abs(daily_return) * 0.1
        cumulative += ret
        store.store_paper_pnl(
            date=day_str,
            portfolio_return=ret,
            benchmark_return=0.0001,
            cumulative_return=cumulative,
        )


# ──────────────────────────────────────────────────────────────
# Schedule Verification (24f.1)
# ──────────────────────────────────────────────────────────────


class TestScheduleVerification:
    """Verify the inference DAG schedule fires after upstream DAGs."""

    def test_inference_dag_schedule(self) -> None:
        """Inference DAG is scheduled at 19:45 UTC weekdays."""
        dag = build_inference_dag()
        assert dag.schedule == "45 19 * * 1-5"

    def test_fires_after_all_upstream_dags(self) -> None:
        """19:45 > max(all other DAG schedules)."""
        from agent.pipeline.dags.adversarial_scan import build_adversarial_scan_dag
        from agent.pipeline.dags.daily_collection import build_daily_collection_dag
        from agent.pipeline.dags.rl_training import build_rl_training_dag
        from agent.pipeline.dags.world_model_update import build_world_model_dag

        # These are the latest upstream DAGs
        assert build_rl_training_dag().schedule == "30 19 * * 1-5"
        assert build_world_model_dag().schedule == "30 19 * * 1-5"

        # Inference at 19:45 is after all of them
        inference = build_inference_dag()
        # Parse hour:minute from cron
        parts = inference.schedule.split()
        minute, hour = int(parts[0]), int(parts[1])
        inference_time = hour * 60 + minute  # 19*60+45 = 1185

        for dag_builder in [
            build_daily_collection_dag,
            build_rl_training_dag,
            build_world_model_dag,
            build_adversarial_scan_dag,
        ]:
            dag = dag_builder()
            if dag.schedule and "*/" not in dag.schedule:
                dparts = dag.schedule.split()
                dm, dh = int(dparts[0]), int(dparts[1])
                dag_time = dh * 60 + dm
                assert inference_time > dag_time, (
                    f"Inference at {hour}:{minute:02d} not after {dag.name} at {dh}:{dm:02d}"
                )


# ──────────────────────────────────────────────────────────────
# Alert Condition Tests (24f.2)
# ──────────────────────────────────────────────────────────────


class TestConcentrationAlert:
    """Test: single instrument weight > 30% → WARNING."""

    def test_no_alert_under_threshold(self) -> None:
        weights = {"SPY": 0.25, "AGG": 0.25, "GLD": 0.25, "TLT": 0.25}
        alerts = _check_concentration(weights)
        assert len(alerts) == 0

    def test_alert_above_threshold(self) -> None:
        weights = {"SPY": 0.50, "AGG": 0.25, "GLD": 0.25}
        alerts = _check_concentration(weights)
        assert len(alerts) == 1
        assert alerts[0]["type"] == "concentration"
        assert alerts[0]["level"] == "WARNING"
        assert alerts[0]["ticker"] == "SPY"
        assert alerts[0]["weight"] == 0.50

    def test_negative_weight_concentration(self) -> None:
        """Large short positions also trigger."""
        weights = {"SPY": -0.40, "AGG": 0.10}
        alerts = _check_concentration(weights)
        assert len(alerts) == 1
        assert alerts[0]["ticker"] == "SPY"

    def test_exactly_at_threshold(self) -> None:
        """Exactly 30% should NOT trigger (must exceed)."""
        weights = {"SPY": 0.30}
        alerts = _check_concentration(weights)
        assert len(alerts) == 0

    def test_multiple_concentrated(self) -> None:
        """Multiple instruments above threshold → multiple alerts."""
        weights = {"SPY": 0.40, "AGG": 0.35, "GLD": 0.25}
        alerts = _check_concentration(weights)
        assert len(alerts) == 2
        tickers = {a["ticker"] for a in alerts}
        assert tickers == {"SPY", "AGG"}


class TestDrawdownAlert:
    """Test: drawdown > 5% from peak → WARNING."""

    def test_no_alert_when_no_history(self, store: PipelineStore) -> None:
        alerts = _check_drawdown(store, "2026-04-13")
        assert len(alerts) == 0

    def test_no_alert_under_threshold(self, store: PipelineStore) -> None:
        """Small drawdown should not trigger."""
        # Positive history → peak wealth > 1, small drawdown
        for i in range(10):
            d = (date(2026, 4, 1) + timedelta(days=i)).isoformat()
            store.store_paper_pnl(d, 0.001, 0.0, 0.001 * (i + 1))
        # Last day has slight loss but still within 5%
        store.store_paper_pnl("2026-04-12", -0.001, 0.0, 0.001 * 10 - 0.001)
        alerts = _check_drawdown(store, "2026-04-12")
        assert len(alerts) == 0

    def test_alert_exceeds_threshold(self, store: PipelineStore) -> None:
        """Build history with >5% drawdown → should trigger."""
        # Build up to a peak, then large drop
        cumulative = 0.0
        for i in range(20):
            d = (date(2026, 3, 1) + timedelta(days=i)).isoformat()
            cumulative += 0.005
            store.store_paper_pnl(d, 0.005, 0.0, cumulative)
        # Peak cumulative = 0.10 → wealth = exp(0.10) ≈ 1.105
        # Now drop it significantly
        for i in range(10):
            d = (date(2026, 3, 21) + timedelta(days=i)).isoformat()
            cumulative -= 0.02
            store.store_paper_pnl(d, -0.02, 0.0, cumulative)
        # cumulative = 0.10 - 0.20 = -0.10 → wealth ≈ 0.905
        # Drawdown = (1.105 - 0.905) / 1.105 ≈ 18.1% > 5%
        alerts = _check_drawdown(store, (date(2026, 3, 30)).isoformat())
        assert len(alerts) == 1
        assert alerts[0]["type"] == "drawdown"
        assert alerts[0]["level"] == "WARNING"
        assert alerts[0]["drawdown"] > _DRAWDOWN_THRESHOLD


class TestSharpeAlert:
    """Test: cumulative Sharpe < -0.5 at ≥30 calendar days → CRITICAL."""

    def test_no_alert_under_30_days(self, store: PipelineStore) -> None:
        """Fewer than 30 calendar days → no check."""
        for i in range(20):
            d = (date(2026, 4, 1) + timedelta(days=i)).isoformat()
            store.store_paper_pnl(d, -0.01, 0.0, -0.01 * (i + 1))
        alerts = _check_sharpe(store, "2026-04-20")
        assert len(alerts) == 0

    def test_no_alert_acceptable_sharpe(self, store: PipelineStore) -> None:
        """Positive returns → good Sharpe → no alert."""
        cumulative = 0.0
        for i in range(60):
            d = (date(2026, 2, 1) + timedelta(days=i)).isoformat()
            cumulative += 0.001
            store.store_paper_pnl(d, 0.001, 0.0, cumulative)
        alerts = _check_sharpe(store, (date(2026, 2, 1) + timedelta(days=59)).isoformat())
        assert len(alerts) == 0

    def test_alert_bad_sharpe_after_30_days(self, store: PipelineStore) -> None:
        """Consistently negative returns for 40+ days → terrible Sharpe → CRITICAL."""
        _seed_pnl_history(store, n_days=45, daily_return=-0.005, start_date="2026-02-01")
        today = (date(2026, 2, 1) + timedelta(days=44)).isoformat()
        alerts = _check_sharpe(store, today)
        assert len(alerts) == 1
        assert alerts[0]["type"] == "sharpe"
        assert alerts[0]["level"] == "CRITICAL"
        assert alerts[0]["annualised_sharpe"] < _SHARPE_THRESHOLD

    def test_zero_variance_returns(self, store: PipelineStore) -> None:
        """Constant returns → zero std → no alert (avoid div by zero)."""
        cumulative = 0.0
        for i in range(40):
            d = (date(2026, 2, 1) + timedelta(days=i)).isoformat()
            cumulative += 0.001
            store.store_paper_pnl(d, 0.001, 0.0, cumulative)
        alerts = _check_sharpe(store, (date(2026, 2, 1) + timedelta(days=39)).isoformat())
        # std ≈ 0 → skip
        assert len(alerts) == 0


class TestEdgeDecayAlert:
    """Test: edge decay flag on held instruments → WARNING."""

    def test_no_flags_no_alert(self, store: PipelineStore) -> None:
        _register_instruments(store, ["SPY"])
        weights = {"SPY": 0.5}
        alerts = _check_edge_decay(store, weights, "2026-04-13")
        assert len(alerts) == 0

    def test_edge_decay_on_held_instrument(self, store: PipelineStore) -> None:
        """Adversarial flag on a held ticker → WARNING."""
        _register_instruments(store, ["SPY"])
        # Store an adversarial flag observation
        store.store_entity_observation(
            "SPY",
            "adversarial_scanner",
            time.time(),
            "adversarial_flag",
            {"flag_type": "edge_decay", "severity": 0.8, "confidence": 0.9},
        )
        weights = {"SPY": 0.3}
        alerts = _check_edge_decay(store, weights, "2026-04-13")
        assert len(alerts) == 1
        assert alerts[0]["type"] == "edge_decay"
        assert alerts[0]["ticker"] == "SPY"

    def test_edge_decay_on_non_held_instrument(self, store: PipelineStore) -> None:
        """Adversarial flag on a ticker NOT held → no alert."""
        _register_instruments(store, ["SPY", "AGG"])
        store.store_entity_observation(
            "AGG",
            "adversarial_scanner",
            time.time(),
            "adversarial_flag",
            {"flag_type": "edge_decay", "severity": 0.7, "confidence": 0.8},
        )
        weights = {"SPY": 0.5}  # Only holding SPY
        alerts = _check_edge_decay(store, weights, "2026-04-13")
        assert len(alerts) == 0

    def test_non_edge_decay_flag_ignored(self, store: PipelineStore) -> None:
        """Adversarial flags of type != edge_decay are not reported."""
        _register_instruments(store, ["SPY"])
        store.store_entity_observation(
            "SPY",
            "adversarial_scanner",
            time.time(),
            "adversarial_flag",
            {"flag_type": "vpin_spike", "severity": 0.9, "confidence": 0.95},
        )
        weights = {"SPY": 0.5}
        alerts = _check_edge_decay(store, weights, "2026-04-13")
        assert len(alerts) == 0

    def test_zero_weight_not_held(self, store: PipelineStore) -> None:
        """Instruments with zero weight are not 'held'."""
        _register_instruments(store, ["SPY"])
        store.store_entity_observation(
            "SPY",
            "adversarial_scanner",
            time.time(),
            "adversarial_flag",
            {"flag_type": "edge_decay", "severity": 0.8, "confidence": 0.9},
        )
        weights = {"SPY": 0.0}
        alerts = _check_edge_decay(store, weights, "2026-04-13")
        assert len(alerts) == 0


# ──────────────────────────────────────────────────────────────
# E2E Integration Smoke Test (24f.3)
# ──────────────────────────────────────────────────────────────


class TestE2EIntegration:
    """Full chain: load_models → gnn → sac → emit_portfolio.

    Uses mocked GNN and SAC to avoid torch dependency, and synthetic
    entity data in PipelineStore.
    """

    def _setup_chain(
        self,
        store: PipelineStore,
        tickers: list[str],
        today: str = "2026-04-13",
        yesterday: str = "2026-04-12",
        yesterday_weights: dict[str, float] | None = None,
        today_returns: dict[str, float] | None = None,
    ) -> tuple[str, str]:
        """Register instruments, store yesterday's weights and today's returns."""
        _register_instruments(store, tickers)

        if yesterday_weights:
            store.store_portfolio_weights(yesterday, yesterday_weights)

        if today_returns:
            _store_daily_returns(store, tickers, today, today_returns)

        return today, yesterday

    def test_full_chain_produces_weights_and_pnl(self, tmp_path: Path) -> None:
        """End-to-end: chain produces portfolio_weights and paper_trade_pnl entries."""
        db_path = str(tmp_path / "e2e.db")
        store = PipelineStore(db_path)
        tickers = ["SPY", "AGG", "GLD"]
        today, yesterday = self._setup_chain(
            store,
            tickers,
            yesterday_weights={"SPY": 0.5, "AGG": 0.3, "GLD": 0.2},
            today_returns={"SPY": 0.01, "AGG": -0.005, "GLD": 0.003},
        )
        store.close()

        # Simulate: load_models says both models are ready
        load_result = {
            "status": "ready",
            "has_gnn": True,
            "has_sac": True,
            "gnn_model_path": "/fake/model.pt",
            "sac_config": {"state_dim": 100, "action_dim": 3},
        }

        # Simulate: gnn_inference returns surprises
        gnn_result = {
            "status": "completed",
            "instrument_surprises": {
                "SPY": [0.1, 0.2, 0.3, 0.4, 0.5],
                "AGG": [0.05, 0.1, 0.15, 0.2, 0.25],
                "GLD": [0.3, 0.1, 0.2, 0.1, 0.4],
            },
            "entity_count": 50,
        }

        # Simulate: sac_inference returns weights
        sac_result = {
            "status": "completed",
            "weights": {"SPY": 0.4, "AGG": 0.35, "GLD": 0.25},
            "instrument_tickers": tickers,
        }

        # Run emit_portfolio
        result = _emit_portfolio(
            params={
                "db_path": db_path,
                "as_of_date": today,
                "yesterday_date": yesterday,
            },
            upstream={
                "load_models": load_result,
                "gnn_inference": gnn_result,
                "sac_inference": sac_result,
            },
        )

        assert result["status"] == "completed"
        assert result["pnl_computed"] is True
        assert result["n_instruments"] == 3
        assert "portfolio_return" in result
        assert "benchmark_return" in result
        assert "alerts" in result

        # Verify tables have data
        store2 = PipelineStore(db_path)
        weights = store2.query_portfolio_weights(today)
        assert len(weights) == 3
        assert weights["SPY"] == pytest.approx(0.4)

        pnl = store2.query_paper_pnl(start_date=today, end_date=today)
        assert len(pnl) == 1
        assert math.isfinite(pnl[0]["portfolio_return"])
        store2.close()

    def test_chain_with_no_previous_weights(self, tmp_path: Path) -> None:
        """First day: no yesterday weights → weights stored, P&L skipped."""
        db_path = str(tmp_path / "e2e_first.db")
        store = PipelineStore(db_path)
        tickers = ["SPY", "AGG"]
        today, yesterday = self._setup_chain(store, tickers)
        store.close()

        sac_result = {
            "status": "completed",
            "weights": {"SPY": 0.6, "AGG": 0.4},
            "instrument_tickers": tickers,
        }

        result = _emit_portfolio(
            params={
                "db_path": db_path,
                "as_of_date": today,
                "yesterday_date": yesterday,
            },
            upstream={"sac_inference": sac_result},
        )

        assert result["status"] == "completed"
        assert result["pnl_computed"] is False
        assert result["reason"] == "no_previous_weights"

        # But weights ARE stored
        store2 = PipelineStore(db_path)
        weights = store2.query_portfolio_weights(today)
        assert "SPY" in weights
        store2.close()

    def test_chain_with_sac_skipped(self) -> None:
        """If SAC inference was skipped, emit_portfolio skips too."""
        result = _emit_portfolio(
            params={},
            upstream={"sac_inference": {"status": "skipped", "reason": "no_sac_model"}},
        )
        assert result["status"] == "skipped"

    def test_pnl_dot_product_correctness(self, tmp_path: Path) -> None:
        """P&L = dot(yesterday_weights, today_returns), verified numerically."""
        db_path = str(tmp_path / "dot.db")
        store = PipelineStore(db_path)
        tickers = ["A", "B", "C"]

        yesterday_weights = {"A": 0.5, "B": 0.3, "C": 0.2}
        today_returns = {"A": 0.01, "B": -0.02, "C": 0.03}
        expected_pnl = 0.5 * 0.01 + 0.3 * (-0.02) + 0.2 * 0.03  # = 0.005 - 0.006 + 0.006 = 0.005

        today, yesterday = self._setup_chain(
            store,
            tickers,
            yesterday_weights=yesterday_weights,
            today_returns=today_returns,
        )
        store.close()

        sac_result = {
            "status": "completed",
            "weights": {"A": 0.4, "B": 0.4, "C": 0.2},  # today's new weights
            "instrument_tickers": tickers,
        }

        result = _emit_portfolio(
            params={
                "db_path": db_path,
                "as_of_date": today,
                "yesterday_date": yesterday,
            },
            upstream={"sac_inference": sac_result},
        )

        assert result["pnl_computed"] is True
        assert result["portfolio_return"] == pytest.approx(expected_pnl, abs=1e-10)

    def test_alerts_returned_in_result(self, tmp_path: Path) -> None:
        """Concentration alert triggers when weight > 30%."""
        db_path = str(tmp_path / "alert.db")
        store = PipelineStore(db_path)
        tickers = ["SPY", "AGG"]
        today, yesterday = self._setup_chain(store, tickers)
        store.close()

        # SPY weight > 30% → concentration alert
        sac_result = {
            "status": "completed",
            "weights": {"SPY": 0.80, "AGG": 0.20},
            "instrument_tickers": tickers,
        }

        result = _emit_portfolio(
            params={
                "db_path": db_path,
                "as_of_date": today,
                "yesterday_date": yesterday,
            },
            upstream={"sac_inference": sac_result},
        )

        assert result["status"] == "completed"
        assert len(result["alerts"]) >= 1
        conc_alerts = [a for a in result["alerts"] if a["type"] == "concentration"]
        assert len(conc_alerts) == 1
        assert conc_alerts[0]["ticker"] == "SPY"

    def test_cumulative_return_chaining(self, tmp_path: Path) -> None:
        """Cumulative return accumulates across consecutive days."""
        db_path = str(tmp_path / "chain.db")
        store = PipelineStore(db_path)
        tickers = ["SPY"]
        _register_instruments(store, tickers)

        # Day 1: seed with known weights
        store.store_portfolio_weights("2026-04-10", {"SPY": 1.0})
        _store_daily_returns(store, tickers, "2026-04-11", {"SPY": 0.01})

        # Store day 1 weights + compute P&L
        store.store_portfolio_weights("2026-04-11", {"SPY": 1.0})
        store.store_paper_pnl("2026-04-11", 0.01, 0.01, 0.01)

        # Day 2
        _store_daily_returns(store, tickers, "2026-04-12", {"SPY": -0.005})
        store.close()

        sac_result = {
            "status": "completed",
            "weights": {"SPY": 1.0},
            "instrument_tickers": tickers,
        }

        result = _emit_portfolio(
            params={
                "db_path": db_path,
                "as_of_date": "2026-04-12",
                "yesterday_date": "2026-04-11",
            },
            upstream={"sac_inference": sac_result},
        )

        assert result["pnl_computed"] is True
        assert result["portfolio_return"] == pytest.approx(-0.005, abs=1e-10)
        # Cumulative = 0.01 + (-0.005) = 0.005
        assert result["cumulative_return"] == pytest.approx(0.005, abs=1e-10)


# ──────────────────────────────────────────────────────────────
# Alert Integration with emit_portfolio
# ──────────────────────────────────────────────────────────────


class TestAlertIntegration:
    """Test that alerts are integrated into the emit_portfolio flow."""

    def test_drawdown_alert_in_full_chain(self, tmp_path: Path) -> None:
        """With enough negative P&L history, DrawdownAlert triggers."""
        db_path = str(tmp_path / "dd.db")
        store = PipelineStore(db_path)
        tickers = ["SPY"]
        _register_instruments(store, tickers)

        # Build up a peak, then crash
        cumulative = 0.0
        for i in range(15):
            d = (date(2026, 3, 1) + timedelta(days=i)).isoformat()
            cumulative += 0.005
            store.store_paper_pnl(d, 0.005, 0.0, cumulative)
        for i in range(15):
            d = (date(2026, 3, 16) + timedelta(days=i)).isoformat()
            cumulative -= 0.015
            store.store_paper_pnl(d, -0.015, 0.0, cumulative)

        # Now run emit_portfolio for today
        today = "2026-03-31"
        yesterday = "2026-03-30"
        store.store_portfolio_weights(yesterday, {"SPY": 0.5})
        _store_daily_returns(store, tickers, today, {"SPY": -0.01})
        store.close()

        result = _emit_portfolio(
            params={
                "db_path": db_path,
                "as_of_date": today,
                "yesterday_date": yesterday,
            },
            upstream={
                "sac_inference": {
                    "status": "completed",
                    "weights": {"SPY": 0.5},
                    "instrument_tickers": tickers,
                }
            },
        )

        assert result["status"] == "completed"
        dd_alerts = [a for a in result.get("alerts", []) if a["type"] == "drawdown"]
        assert len(dd_alerts) == 1

    def test_sharpe_alert_in_full_chain(self, tmp_path: Path) -> None:
        """With 40+ days of bad returns, Sharpe alert triggers."""
        db_path = str(tmp_path / "sh.db")
        store = PipelineStore(db_path)
        tickers = ["SPY"]
        _register_instruments(store, tickers)

        _seed_pnl_history(store, n_days=45, daily_return=-0.005, start_date="2026-02-01")

        today = (date(2026, 2, 1) + timedelta(days=45)).isoformat()
        yesterday = (date(2026, 2, 1) + timedelta(days=44)).isoformat()
        store.store_portfolio_weights(yesterday, {"SPY": 0.5})
        _store_daily_returns(store, tickers, today, {"SPY": -0.005})
        store.close()

        result = _emit_portfolio(
            params={
                "db_path": db_path,
                "as_of_date": today,
                "yesterday_date": yesterday,
            },
            upstream={
                "sac_inference": {
                    "status": "completed",
                    "weights": {"SPY": 0.5},
                    "instrument_tickers": tickers,
                }
            },
        )

        assert result["status"] == "completed"
        sharpe_alerts = [a for a in result.get("alerts", []) if a["type"] == "sharpe"]
        assert len(sharpe_alerts) == 1
        assert sharpe_alerts[0]["level"] == "CRITICAL"

    def test_edge_decay_alert_in_full_chain(self, tmp_path: Path) -> None:
        """Edge decay flag on held instrument → alert in emit_portfolio."""
        db_path = str(tmp_path / "ed.db")
        store = PipelineStore(db_path)
        tickers = ["SPY"]
        _register_instruments(store, tickers)

        # Store edge decay flag
        store.store_entity_observation(
            "SPY",
            "adversarial_scanner",
            time.time(),
            "adversarial_flag",
            {"flag_type": "edge_decay", "severity": 0.85, "confidence": 0.9},
        )

        today = "2026-04-13"
        yesterday = "2026-04-12"
        store.store_portfolio_weights(yesterday, {"SPY": 0.5})
        _store_daily_returns(store, tickers, today, {"SPY": 0.002})
        store.close()

        result = _emit_portfolio(
            params={
                "db_path": db_path,
                "as_of_date": today,
                "yesterday_date": yesterday,
            },
            upstream={
                "sac_inference": {
                    "status": "completed",
                    "weights": {"SPY": 0.5},
                    "instrument_tickers": tickers,
                }
            },
        )

        assert result["status"] == "completed"
        ed_alerts = [a for a in result.get("alerts", []) if a["type"] == "edge_decay"]
        assert len(ed_alerts) == 1
        assert ed_alerts[0]["ticker"] == "SPY"

    def test_no_alerts_healthy_portfolio(self, tmp_path: Path) -> None:
        """Healthy portfolio with small weights and positive returns → no alerts."""
        db_path = str(tmp_path / "healthy.db")
        store = PipelineStore(db_path)
        tickers = ["SPY", "AGG", "GLD", "TLT"]
        _register_instruments(store, tickers)

        today = "2026-04-13"
        yesterday = "2026-04-12"
        store.store_portfolio_weights(yesterday, {"SPY": 0.25, "AGG": 0.25, "GLD": 0.25, "TLT": 0.25})
        _store_daily_returns(
            store,
            tickers,
            today,
            {"SPY": 0.005, "AGG": 0.002, "GLD": 0.003, "TLT": 0.001},
        )
        store.close()

        result = _emit_portfolio(
            params={
                "db_path": db_path,
                "as_of_date": today,
                "yesterday_date": yesterday,
            },
            upstream={
                "sac_inference": {
                    "status": "completed",
                    "weights": {"SPY": 0.25, "AGG": 0.25, "GLD": 0.25, "TLT": 0.25},
                    "instrument_tickers": tickers,
                }
            },
        )

        assert result["status"] == "completed"
        assert result["pnl_computed"] is True
        assert len(result.get("alerts", [])) == 0
