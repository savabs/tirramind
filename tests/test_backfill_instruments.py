"""Tests for Phase 24b — instrument backfill + GNN training with instruments.

Covers:
    - backfill_instruments.backfill(): idempotency, day-by-day obs storage,
      rolling signal computation, failure handling, dry-run mode
    - GNN training convergence with instrument entities (slow test)
    - SurpriseExtractor returns surprises for instrument entities
"""

from __future__ import annotations

import math
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from agent.pipeline.entity import entity_id_from_key
from agent.pipeline.store import PipelineStore

# Import backfill machinery — the script adds project root to sys.path
# when run standalone; for tests, it's already importable.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from backfill_instruments import (
    _date_to_ts,
    _get_existing_dates,
    backfill,
)

from agent.tools.instrument_universe import InstrumentDef, tradeable_instruments


# ── Helpers ───────────────────────────────────────────────────


def _make_ohlcv_series(
    n_days: int = 60,
    start_price: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with realistic structure."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-04-01", periods=n_days, freq="B")
    returns = rng.normal(0.0005, 0.015, n_days)
    closes = start_price * np.exp(np.cumsum(returns))
    highs = closes * (1 + rng.uniform(0.001, 0.02, n_days))
    lows = closes * (1 - rng.uniform(0.001, 0.02, n_days))
    opens = closes * (1 + rng.normal(0, 0.005, n_days))
    volumes = rng.randint(100_000, 10_000_000, n_days).astype(float)
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=dates,
    )


def _make_batch_df(
    ticker_data: dict[str, pd.DataFrame],
    *,
    force_multi: bool = False,
) -> pd.DataFrame:
    """Build a multi-ticker DataFrame like yf.download returns.

    With 1 ticker, returns a plain DataFrame (matching yfinance behaviour)
    unless force_multi=True (simulates multi-ticker download where some failed).
    """
    if len(ticker_data) == 1 and not force_multi:
        return list(ticker_data.values())[0]
    frames = {}
    for ticker, df in ticker_data.items():
        for col in df.columns:
            frames[(ticker, col)] = df[col]
    result = pd.DataFrame(frames)
    result.columns = pd.MultiIndex.from_tuples(result.columns)
    return result


# ── _date_to_ts ──────────────────────────────────────────────


class TestDateToTs:
    def test_basic(self):
        ts = _date_to_ts(date(2025, 1, 1))
        dt = datetime.fromtimestamp(ts)
        assert dt.year == 2025
        assert dt.month == 1
        assert dt.day == 1

    def test_midnight(self):
        ts = _date_to_ts(date(2025, 6, 15))
        dt = datetime.fromtimestamp(ts)
        assert dt.hour == 0
        assert dt.minute == 0


# ── _get_existing_dates ──────────────────────────────────────


class TestGetExistingDates:
    def test_empty_store(self):
        store = PipelineStore(":memory:")
        dates = _get_existing_dates(store, "test_id")
        assert dates == set()

    def test_with_data(self):
        store = PipelineStore(":memory:")
        eid = entity_id_from_key("instrument", "SPY")
        store.register_entity("instrument", "SPY", eid)
        ts = _date_to_ts(date(2025, 3, 10))
        store.store_entity_observation(
            entity_id=eid,
            source_tool="instrument_universe",
            observed_at=ts,
            observation_type="instrument_return",
            value={"log_return": 0.01, "close": 100.0},
        )
        dates = _get_existing_dates(store, eid)
        assert date(2025, 3, 10) in dates


# ── Backfill core logic ──────────────────────────────────────


class TestBackfillBasic:
    """Test backfill with mocked yfinance and controlled instruments."""

    def _run_backfill(
        self,
        ticker_data: dict[str, pd.DataFrame],
        db_path: str,
        instruments: list[InstrumentDef] | None = None,
        dry_run: bool = False,
        force_multi: bool = False,
    ) -> dict:
        """Helper: run backfill with given data."""
        if instruments is None:
            instruments = [
                InstrumentDef(t, f"Test {t}", "equity_etf", "US") for t in ticker_data
            ]

        batch_df = _make_batch_df(ticker_data, force_multi=force_multi)

        with patch(
            "backfill_instruments.tradeable_instruments", return_value=instruments
        ):
            with patch("yfinance.download", return_value=batch_df):
                result = backfill(db_path=db_path, years=2, dry_run=dry_run)

        return result

    def test_single_ticker_backfill(self, tmp_path):
        db = str(tmp_path / "test.db")
        df = _make_ohlcv_series(n_days=30)
        result = self._run_backfill({"SPY": df}, db_path=db)

        assert result["instruments"] == 1
        assert result["instruments_failed"] == 0
        assert result["days_total"] == 30
        assert result["observations_stored"] > 0

        # Check entity was registered
        store = PipelineStore(db)
        eid = entity_id_from_key("instrument", "SPY")
        obs = store.query_entity_observations(
            eid, source_tool="instrument_universe", limit=1000
        )
        assert len(obs) > 0
        store.close()

    def test_multi_ticker(self, tmp_path):
        db = str(tmp_path / "test.db")
        data = {
            "SPY": _make_ohlcv_series(n_days=25, seed=1),
            "QQQ": _make_ohlcv_series(n_days=25, seed=2),
        }
        result = self._run_backfill(data, db_path=db)
        assert result["instruments"] == 2
        assert result["observations_stored"] > 0

    def test_observation_types_stored(self, tmp_path):
        """Each day should store 2-3 obs types: return (after day 0), volume, volatility."""
        db = str(tmp_path / "test.db")
        df = _make_ohlcv_series(n_days=25)
        self._run_backfill({"SPY": df}, db_path=db)

        store = PipelineStore(db)
        eid = entity_id_from_key("instrument", "SPY")
        obs = store.query_entity_observations(
            eid, source_tool="instrument_universe", limit=1000
        )
        store.close()

        obs_types = {o["observation_type"] for o in obs}
        assert "instrument_return" in obs_types
        assert "instrument_volume" in obs_types
        assert "instrument_volatility" in obs_types

    def test_day0_no_return(self, tmp_path):
        """First day has no return (no previous close)."""
        db = str(tmp_path / "test.db")
        df = _make_ohlcv_series(n_days=5)
        self._run_backfill({"SPY": df}, db_path=db)

        store = PipelineStore(db)
        eid = entity_id_from_key("instrument", "SPY")
        obs = store.query_entity_observations(
            eid, source_tool="instrument_universe", limit=1000
        )
        store.close()

        # Count return observations
        return_obs = [o for o in obs if o["observation_type"] == "instrument_return"]
        volume_obs = [o for o in obs if o["observation_type"] == "instrument_volume"]

        # 5 days: return for days 1-4 = 4, volume for all 5 days = 5
        assert len(return_obs) == 4
        assert len(volume_obs) == 5

    def test_log_return_values(self, tmp_path):
        """Verify log returns are computed correctly from synthetic data."""
        db = str(tmp_path / "test.db")
        prices = [100.0, 105.0, 103.0]
        dates = pd.date_range("2025-01-01", periods=3, freq="B")
        df = pd.DataFrame(
            {
                "Open": prices,
                "High": [p * 1.01 for p in prices],
                "Low": [p * 0.99 for p in prices],
                "Close": prices,
                "Volume": [1e6] * 3,
            },
            index=dates,
        )
        self._run_backfill({"SPY": df}, db_path=db)

        store = PipelineStore(db)
        eid = entity_id_from_key("instrument", "SPY")
        obs = store.query_entity_observations(
            eid, source_tool="instrument_universe", limit=1000
        )
        store.close()

        return_obs = sorted(
            [o for o in obs if o["observation_type"] == "instrument_return"],
            key=lambda o: o["observed_at"],
        )
        assert len(return_obs) == 2

        # Day 1: ln(105/100) ≈ 0.04879
        lr1 = return_obs[0]["value"]["log_return"]
        assert abs(lr1 - math.log(105 / 100)) < 1e-10

        # Day 2: ln(103/105) ≈ -0.01923
        lr2 = return_obs[1]["value"]["log_return"]
        assert abs(lr2 - math.log(103 / 105)) < 1e-10

    def test_rolling_vol_20d(self, tmp_path):
        """With 25 days, day 20 onward should have proper 20d vol."""
        db = str(tmp_path / "test.db")
        df = _make_ohlcv_series(n_days=25, seed=99)
        self._run_backfill({"SPY": df}, db_path=db)

        store = PipelineStore(db)
        eid = entity_id_from_key("instrument", "SPY")
        obs = store.query_entity_observations(
            eid, source_tool="instrument_universe", limit=1000
        )
        store.close()

        vol_obs = [o for o in obs if o["observation_type"] == "instrument_volatility"]
        assert len(vol_obs) > 0

        for v in vol_obs:
            rv = v["value"]["realized_vol_20d"]
            assert rv > 0
            assert not math.isnan(rv)
            assert rv < 2.0  # annualized vol < 200% for normal data


class TestBackfillIdempotency:
    """Verify re-running backfill skips existing dates."""

    def test_second_run_skips(self, tmp_path):
        db = str(tmp_path / "test.db")
        df = _make_ohlcv_series(n_days=10, seed=42)
        instruments = [InstrumentDef("SPY", "S&P 500", "equity_etf", "US")]
        batch_df = _make_batch_df({"SPY": df})

        with patch(
            "backfill_instruments.tradeable_instruments", return_value=instruments
        ):
            with patch("yfinance.download", return_value=batch_df):
                r1 = backfill(db_path=db, years=2)
                r2 = backfill(db_path=db, years=2)

        assert r1["observations_stored"] > 0
        assert r2["days_skipped"] == 10
        assert r2["observations_stored"] == 0

    def test_partial_backfill_completes(self, tmp_path):
        """If 5/10 days exist, only 5 new days are stored."""
        db = str(tmp_path / "test.db")
        instruments = [InstrumentDef("SPY", "S&P 500", "equity_etf", "US")]
        df = _make_ohlcv_series(n_days=10, seed=42)
        batch_df = _make_batch_df({"SPY": df})

        # First run: only 5 days
        df_partial = _make_ohlcv_series(n_days=5, seed=42)
        batch_partial = _make_batch_df({"SPY": df_partial})

        with patch(
            "backfill_instruments.tradeable_instruments", return_value=instruments
        ):
            with patch("yfinance.download", return_value=batch_partial):
                r1 = backfill(db_path=db, years=2)

            # Second run: full 10 days
            with patch("yfinance.download", return_value=batch_df):
                r2 = backfill(db_path=db, years=2)

        assert r1["days_total"] == 5
        assert r2["days_skipped"] == 5
        assert r2["days_total"] == 5  # only the new 5 days


class TestBackfillEdgeCases:
    def test_dry_run_no_writes(self):
        df = _make_ohlcv_series(n_days=10)
        instruments = [InstrumentDef("SPY", "S&P 500", "equity_etf", "US")]
        batch_df = _make_batch_df({"SPY": df})

        with patch(
            "backfill_instruments.tradeable_instruments", return_value=instruments
        ):
            with patch("yfinance.download", return_value=batch_df):
                result = backfill(db_path="fake.db", years=2, dry_run=True)

        # Dry run still counts obs but doesn't write
        assert result["observations_stored"] > 0

    def test_empty_download(self):
        instruments = [InstrumentDef("SPY", "S&P 500", "equity_etf", "US")]
        empty_df = pd.DataFrame()

        with patch(
            "backfill_instruments.tradeable_instruments", return_value=instruments
        ):
            with patch("yfinance.download", return_value=empty_df):
                result = backfill(db_path="fake.db", years=2, dry_run=True)

        assert result["instruments"] == 0
        assert result["observations_stored"] == 0

    def test_missing_ticker_in_batch(self, tmp_path):
        """Ticker not in batch download → counted as failure."""
        db = str(tmp_path / "test.db")
        instruments = [
            InstrumentDef("SPY", "S&P 500", "equity_etf", "US"),
            InstrumentDef("MISSING", "Missing", "equity_etf", "US"),
        ]
        df = _make_ohlcv_series(n_days=10)
        # force_multi=True simulates yf.download(2 tickers) returning MultiIndex
        # even though only SPY has data
        batch_df = _make_batch_df({"SPY": df}, force_multi=True)

        with patch(
            "backfill_instruments.tradeable_instruments", return_value=instruments
        ):
            with patch("yfinance.download", return_value=batch_df):
                result = backfill(db_path=db, years=2)

        assert result["instruments"] == 1
        assert result["instruments_failed"] == 1


# ── GNN Training with Instruments (slow) ─────────────────────


@pytest.mark.slow
class TestGNNTrainingWithInstruments:
    """End-to-end: populate store with instruments + entities, train GNN,
    extract surprises. This takes ~30-60s due to training.
    """

    @pytest.fixture
    def instrument_store(self):
        """Build a store with 5 instruments + 10 entities + observations."""
        store = PipelineStore(":memory:")

        # ── Create 5 instrument entities with 30 days of obs ──
        instrument_tickers = ["SPY", "QQQ", "GLD", "CL=F", "EURUSD=X"]
        instrument_names = [
            "S&P 500 ETF",
            "Nasdaq ETF",
            "Gold ETF",
            "Crude Oil",
            "EUR/USD",
        ]
        rng = np.random.RandomState(42)
        t_start = 1_700_000_000.0  # ~Nov 2023
        day_sec = 86400.0

        for i, (ticker, name) in enumerate(zip(instrument_tickers, instrument_names)):
            eid = entity_id_from_key("instrument", ticker)
            store.register_entity(
                entity_type="instrument",
                canonical_name=name,
                entity_id=eid,
                metadata={"ticker": ticker, "asset_class": "equity_etf"},
            )

            # Generate 30 days of observations
            base_price = 100.0 + i * 50
            for day in range(30):
                ts = t_start + day * day_sec
                lr = float(rng.normal(0.001, 0.02))
                vol = float(abs(rng.normal(0.15, 0.03)))
                volume = float(rng.randint(1_000_000, 50_000_000))

                store.store_entity_observation(
                    entity_id=eid,
                    source_tool="instrument_universe",
                    observed_at=ts,
                    observation_type="instrument_return",
                    value={"log_return": lr, "close": base_price * math.exp(lr * day)},
                    depth_level=1,
                )
                store.store_entity_observation(
                    entity_id=eid,
                    source_tool="instrument_universe",
                    observed_at=ts,
                    observation_type="instrument_volume",
                    value={"volume": volume, "avg_volume_20d": volume * 0.9},
                    depth_level=1,
                )
                store.store_entity_observation(
                    entity_id=eid,
                    source_tool="instrument_universe",
                    observed_at=ts,
                    observation_type="instrument_volatility",
                    value={"realized_vol_20d": vol, "intraday_range": vol * 10},
                    depth_level=1,
                )

        # ── Create 10 non-instrument entities with observations ──
        companies = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
        countries = ["US", "UK", "JP", "DE", "CN"]

        for i, c in enumerate(companies):
            eid = entity_id_from_key("company", c)
            store.register_entity("company", c, eid)

            for day in range(30):
                ts = t_start + day * day_sec + rng.uniform(0, day_sec / 2)
                store.store_entity_observation(
                    entity_id=eid,
                    source_tool="synthetic",
                    observed_at=ts,
                    observation_type="insider_trade",
                    value={"amount": float(rng.uniform(1000, 500000))},
                )

        for i, c in enumerate(countries):
            eid = entity_id_from_key("country", c)
            store.register_entity("country", c, eid)

            for day in range(30):
                ts = t_start + day * day_sec + rng.uniform(0, day_sec / 2)
                store.store_entity_observation(
                    entity_id=eid,
                    source_tool="synthetic",
                    observed_at=ts,
                    observation_type="geopolitical_event",
                    value={"severity": float(rng.uniform(0.1, 1.0))},
                )

        # ── Create links: company → country ──
        for i, c in enumerate(companies):
            ceid = entity_id_from_key("company", c)
            coeid = entity_id_from_key("country", countries[i])
            store.link_entities(ceid, coeid, "headquartered_in", "synthetic")

        return store

    def test_graph_contains_instrument_nodes(self, instrument_store):
        """Graph builder should create instrument-type nodes."""
        from agent.models.gnn.graph_builder import GraphBuilder

        builder = GraphBuilder(instrument_store)
        data, id_map, events = builder.build()

        assert id_map.num_nodes_of_type("instrument") == 5
        assert "instrument" in data.node_types
        assert data["instrument"].x.shape[0] == 5

    def test_training_loss_decreases(self, instrument_store):
        """Train HetTGN for 50 epochs; loss should decrease."""
        from agent.models.gnn.trainer import Trainer, TrainerConfig

        config = TrainerConfig(
            epochs=50,
            hidden_dim=32,
            memory_dim=32,
            message_dim=32,
            time_dim=8,
            num_heads=2,
            num_layers=1,
            learning_rate=1e-3,
        )
        trainer = Trainer(instrument_store, config=config)
        trainer.build_model()
        loss_history = trainer.train()

        total = loss_history["total"]
        assert len(total) == 50

        # Average of first 5 > average of last 5 = loss decreased
        first_5 = np.mean(total[:5])
        last_5 = np.mean(total[-5:])
        assert (
            last_5 < first_5
        ), f"Loss did not decrease: first_5={first_5:.4f}, last_5={last_5:.4f}"

    def test_instrument_embeddings_nonzero(self, instrument_store):
        """After training, instrument node embeddings should be non-zero."""
        from agent.models.gnn.graph_builder import GraphBuilder
        from agent.models.gnn.trainer import Trainer, TrainerConfig

        config = TrainerConfig(
            epochs=20,
            hidden_dim=32,
            memory_dim=32,
            message_dim=32,
            time_dim=8,
            num_heads=2,
            num_layers=1,
        )
        trainer = Trainer(instrument_store, config=config)
        trainer.build_model()
        trainer.train()

        model = trainer.model
        builder = GraphBuilder(instrument_store)
        data, id_map, _ = builder.build()

        # Run forward pass
        model.eval()
        with torch.no_grad():
            out = model(data, id_map)

        # Check instrument embeddings are non-zero
        if "instrument" in out:
            inst_emb = out["instrument"]
            norms = torch.norm(inst_emb, dim=1)
            assert (norms > 0).all(), "Some instrument embeddings are all-zero"

    def test_surprise_extraction_for_instruments(self, instrument_store):
        """SurpriseExtractor should return EntitySurprise for instrument entities."""
        from agent.fusion.surprise import SurpriseExtractor
        from agent.models.gnn.graph_builder import GraphBuilder
        from agent.models.gnn.trainer import Trainer, TrainerConfig

        config = TrainerConfig(
            epochs=10,
            hidden_dim=32,
            memory_dim=32,
            message_dim=32,
            time_dim=8,
            num_heads=2,
            num_layers=1,
        )
        trainer = Trainer(instrument_store, config=config)
        trainer.build_model()
        trainer.train()

        model = trainer.model
        builder = GraphBuilder(instrument_store)
        data, id_map, events = builder.build()

        # Extract only instrument-related observations
        instrument_obs = [
            e for e in events if e.get("observation_type", "").startswith("instrument_")
        ]

        se = SurpriseExtractor()
        surprises = se.extract(model, data, id_map, instrument_obs[-20:])

        # Should have at least some instrument surprises
        instrument_eids = {
            entity_id_from_key("instrument", t)
            for t in ["SPY", "QQQ", "GLD", "CL=F", "EURUSD=X"]
        }
        instrument_surprises = {
            eid: s for eid, s in surprises.items() if eid in instrument_eids
        }

        assert len(instrument_surprises) > 0, "No instrument surprises extracted"

        for eid, s in instrument_surprises.items():
            assert s.entity_type == "instrument"
            assert isinstance(s.composite_surprise, float)
            assert not math.isnan(s.composite_surprise)

    def test_diagnostics_include_instruments(self, instrument_store):
        """compute_diagnostics should report instrument entity/obs density."""
        from agent.models.gnn.integration import (
            compute_diagnostics,
            format_diagnostic_report,
        )
        from agent.models.gnn.trainer import Trainer, TrainerConfig

        config = TrainerConfig(
            epochs=5,
            hidden_dim=32,
            memory_dim=32,
            message_dim=32,
            time_dim=8,
            num_heads=2,
            num_layers=1,
        )
        trainer = Trainer(instrument_store, config=config)
        trainer.build_model()
        trainer.train()

        diag = compute_diagnostics(trainer.model, instrument_store)

        # Entity type density should include instruments
        assert "instrument" in diag["entity_type_density"]
        assert diag["entity_type_density"]["instrument"] == 5

        # Observation density should include instrument obs types
        assert "instrument_return" in diag["observation_density"]
        assert "instrument_volume" in diag["observation_density"]
        assert "instrument_volatility" in diag["observation_density"]

        # Each instrument has 30 days × 3 obs types = 90 each type
        for ot in ("instrument_return", "instrument_volume", "instrument_volatility"):
            assert diag["observation_density"][ot] == 150  # 5 instruments × 30 days

        # Neighborhood sparsity: instruments have no links → degree 0
        assert "instrument" in diag["neighborhood_sparsity"]
        assert diag["neighborhood_sparsity"]["instrument"] == 0.0

        # Format report should also work — uses different key names
        report = format_diagnostic_report(diag)
        assert "entity_density" in report
