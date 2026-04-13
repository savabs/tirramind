"""Tests for agent/tools/instrument_universe.py — Phase 24a."""

from __future__ import annotations

import math
import time
from datetime import date
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from agent.tools.instrument_universe import (
    INSTRUMENTS,
    InstrumentDef,
    _entity_id,
    ingest_daily_prices,
    instruments_by_class,
    run_instrument_ingest,
    tradeable_instruments,
    ticker_to_instrument,
)


# ── Helpers ───────────────────────────────────────────────────


def _make_store() -> MagicMock:
    store = MagicMock()
    store.register_entity = MagicMock(side_effect=lambda **kw: kw["entity_id"])
    store.store_entity_observation = MagicMock(return_value=1)
    store.close = MagicMock()
    return store


def _make_ohlcv_df(
    closes: list[float],
    volumes: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame like yfinance returns."""
    n = len(closes)
    if volumes is None:
        volumes = [1_000_000.0] * n
    if highs is None:
        highs = [c * 1.01 for c in closes]
    if lows is None:
        lows = [c * 0.99 for c in closes]
    dates = pd.date_range("2026-03-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        },
        index=dates,
    )


def _make_batch_download(ticker_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a DataFrame like yf.download returns.

    Single ticker → flat columns (matching yfinance behavior).
    Multiple tickers → MultiIndex columns grouped by ticker.
    """
    if len(ticker_data) == 1:
        return list(ticker_data.values())[0]
    frames = {}
    for ticker, df in ticker_data.items():
        for col in df.columns:
            frames[(ticker, col)] = df[col]
    result = pd.DataFrame(frames)
    result.columns = pd.MultiIndex.from_tuples(result.columns)
    return result


# ── InstrumentDef dataclass ──────────────────────────────────


class TestInstrumentDef:
    def test_creation(self):
        inst = InstrumentDef("AAPL", "Apple Inc", "equity_etf", "US")
        assert inst.ticker == "AAPL"
        assert inst.is_tradeable is True

    def test_frozen(self):
        inst = InstrumentDef("SPY", "S&P 500", "equity_etf", "US")
        with pytest.raises(AttributeError):
            inst.ticker = "QQQ"  # type: ignore

    def test_not_tradeable(self):
        inst = InstrumentDef("^VIX", "VIX", "vol", "US", is_tradeable=False)
        assert inst.is_tradeable is False


# ── INSTRUMENTS constant ─────────────────────────────────────


class TestInstruments:
    def test_total_count(self):
        assert len(INSTRUMENTS) == 90

    def test_tradeable_count(self):
        tradeable = [i for i in INSTRUMENTS if i.is_tradeable]
        assert len(tradeable) == 89

    def test_non_tradeable_is_vix(self):
        non_tradeable = [i for i in INSTRUMENTS if not i.is_tradeable]
        assert len(non_tradeable) == 1
        assert non_tradeable[0].ticker == "^VIX"

    def test_unique_tickers(self):
        tickers = [i.ticker for i in INSTRUMENTS]
        assert len(tickers) == len(set(tickers))

    def test_all_have_asset_class(self):
        valid_classes = {
            "commodity_future",
            "fx",
            "equity_index",
            "equity_etf",
            "sector_etf",
            "fixed_income",
            "vol",
            "crypto",
        }
        for inst in INSTRUMENTS:
            assert (
                inst.asset_class in valid_classes
            ), f"{inst.ticker}: invalid asset_class={inst.asset_class}"

    def test_all_have_region(self):
        valid_regions = {"US", "Europe", "Asia", "LatAm", "Pacific", "Global", "EM"}
        for inst in INSTRUMENTS:
            assert (
                inst.region in valid_regions
            ), f"{inst.ticker}: invalid region={inst.region}"


# ── Asset class counts ───────────────────────────────────────


class TestAssetClassCounts:
    def test_commodity_futures(self):
        assert len(instruments_by_class("commodity_future")) == 20

    def test_fx(self):
        assert len(instruments_by_class("fx")) == 15

    def test_equity_index(self):
        assert len(instruments_by_class("equity_index")) == 4

    def test_equity_etf(self):
        assert len(instruments_by_class("equity_etf")) == 21

    def test_sector_etf(self):
        assert len(instruments_by_class("sector_etf")) == 15

    def test_fixed_income(self):
        assert len(instruments_by_class("fixed_income")) == 10

    def test_vol(self):
        assert len(instruments_by_class("vol")) == 3

    def test_crypto(self):
        assert len(instruments_by_class("crypto")) == 2

    def test_unknown_class_empty(self):
        assert instruments_by_class("nonexistent") == []


# ── Helper functions ─────────────────────────────────────────


class TestHelpers:
    def test_tradeable_instruments_excludes_vix(self):
        result = tradeable_instruments()
        assert len(result) == 89
        tickers = {i.ticker for i in result}
        assert "^VIX" not in tickers

    def test_ticker_to_instrument_lookup(self):
        lookup = ticker_to_instrument()
        assert len(lookup) == 90
        assert lookup["SPY"].name == "S&P 500 ETF"
        assert lookup["CL=F"].asset_class == "commodity_future"
        assert lookup["^VIX"].is_tradeable is False

    def test_entity_id_deterministic(self):
        id1 = _entity_id("SPY")
        id2 = _entity_id("SPY")
        assert id1 == id2
        assert len(id1) == 16

    def test_entity_id_differs_by_ticker(self):
        assert _entity_id("SPY") != _entity_id("QQQ")


# ── ingest_daily_prices ──────────────────────────────────────


class TestIngestBasic:
    """Test ingest with mocked yfinance download."""

    def _run_ingest(
        self, ticker_data: dict[str, pd.DataFrame]
    ) -> tuple[dict, MagicMock]:
        """Helper: run ingest with given per-ticker data."""
        store = _make_store()
        batch_df = _make_batch_download(ticker_data)

        with patch(
            "agent.tools.instrument_universe.tradeable_instruments"
        ) as mock_trad:
            # Only return instruments matching our test tickers
            mock_trad.return_value = [
                InstrumentDef(t, f"Test {t}", "equity_etf", "US") for t in ticker_data
            ]
            with patch("yfinance.download", return_value=batch_df):
                result = ingest_daily_prices(store, as_of=date(2026, 4, 1))

        return result, store

    def test_single_ticker_registered(self):
        df = _make_ohlcv_df([100.0, 105.0])
        result, store = self._run_ingest({"SPY": df})

        assert result["instruments_fetched"] == 1
        assert result["instruments_failed"] == []
        store.register_entity.assert_called_once()
        call_kw = store.register_entity.call_args.kwargs
        assert call_kw["entity_type"] == "instrument"
        assert call_kw["canonical_name"] == "Test SPY"

    def test_log_return_computation(self):
        # close yesterday=100, today=105 → ln(105/100) ≈ 0.04879
        df = _make_ohlcv_df([100.0, 105.0])
        _, store = self._run_ingest({"SPY": df})

        obs_calls = store.store_entity_observation.call_args_list
        return_call = [
            c
            for c in obs_calls
            if c.kwargs.get("observation_type") == "instrument_return"
        ]
        assert len(return_call) == 1
        log_ret = return_call[0].kwargs["value"]["log_return"]
        assert abs(log_ret - math.log(105 / 100)) < 1e-10

    def test_volume_observation_stored(self):
        df = _make_ohlcv_df([100.0, 102.0], volumes=[500_000, 750_000])
        _, store = self._run_ingest({"SPY": df})

        obs_calls = store.store_entity_observation.call_args_list
        vol_calls = [
            c
            for c in obs_calls
            if c.kwargs.get("observation_type") == "instrument_volume"
        ]
        assert len(vol_calls) == 1
        assert vol_calls[0].kwargs["value"]["volume"] == 750_000.0

    def test_volatility_with_enough_data(self):
        # 25 data points → enough for 20d vol
        closes = [100.0 + i * 0.5 for i in range(25)]
        df = _make_ohlcv_df(closes)
        _, store = self._run_ingest({"SPY": df})

        obs_calls = store.store_entity_observation.call_args_list
        vol_calls = [
            c
            for c in obs_calls
            if c.kwargs.get("observation_type") == "instrument_volatility"
        ]
        assert len(vol_calls) == 1
        realized_vol = vol_calls[0].kwargs["value"]["realized_vol_20d"]
        assert realized_vol > 0
        assert not math.isnan(realized_vol)

    def test_multi_ticker(self):
        data = {
            "SPY": _make_ohlcv_df([100.0, 101.0]),
            "QQQ": _make_ohlcv_df([200.0, 202.0]),
        }
        result, store = self._run_ingest(data)
        assert result["instruments_fetched"] == 2
        assert store.register_entity.call_count == 2

    def test_three_obs_per_ticker_with_enough_data(self):
        # With 25 data points: return + volume + volatility = 3 obs
        closes = [100.0 + i * 0.5 for i in range(25)]
        df = _make_ohlcv_df(closes)
        result, store = self._run_ingest({"SPY": df})
        assert result["observations_stored"] == 3


class TestIngestEdgeCases:
    def _run_ingest_single(
        self, df: pd.DataFrame, ticker: str = "SPY"
    ) -> tuple[dict, MagicMock]:
        store = _make_store()
        batch_df = _make_batch_download({ticker: df})
        with patch(
            "agent.tools.instrument_universe.tradeable_instruments"
        ) as mock_trad:
            mock_trad.return_value = [
                InstrumentDef(ticker, f"Test {ticker}", "equity_etf", "US")
            ]
            with patch("yfinance.download", return_value=batch_df):
                result = ingest_daily_prices(store, as_of=date(2026, 4, 1))
        return result, store

    def test_single_day_no_return(self):
        """Only 1 close → can't compute return → only volume obs stored."""
        df = _make_ohlcv_df([100.0])
        result, store = self._run_ingest_single(df)
        assert result["instruments_fetched"] == 1
        obs_calls = store.store_entity_observation.call_args_list
        return_calls = [
            c
            for c in obs_calls
            if c.kwargs.get("observation_type") == "instrument_return"
        ]
        assert len(return_calls) == 0  # no return obs
        vol_calls = [
            c
            for c in obs_calls
            if c.kwargs.get("observation_type") == "instrument_volume"
        ]
        assert len(vol_calls) == 1  # volume still stored

    def test_nan_close_rows_dropped(self):
        """Rows with NaN close should be dropped."""
        df = _make_ohlcv_df([100.0, float("nan"), 102.0])
        result, store = self._run_ingest_single(df)
        assert result["instruments_fetched"] == 1
        obs_calls = store.store_entity_observation.call_args_list
        return_calls = [
            c
            for c in obs_calls
            if c.kwargs.get("observation_type") == "instrument_return"
        ]
        assert len(return_calls) == 1

    def test_zero_volume(self):
        """Zero volume should be stored, not skipped."""
        df = _make_ohlcv_df([100.0, 101.0], volumes=[0, 0])
        result, store = self._run_ingest_single(df)
        obs_calls = store.store_entity_observation.call_args_list
        vol_calls = [
            c
            for c in obs_calls
            if c.kwargs.get("observation_type") == "instrument_volume"
        ]
        assert len(vol_calls) == 1
        assert vol_calls[0].kwargs["value"]["volume"] == 0.0

    def test_empty_df_counts_as_failure(self):
        """Completely empty DataFrame for a single ticker → 100% failure → RuntimeError."""
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        with pytest.raises(RuntimeError, match=">50%"):
            self._run_ingest_single(df)

    def test_insufficient_vol_history(self):
        """< 20 data points → vol computed from available data."""
        closes = [100.0 + i for i in range(5)]
        df = _make_ohlcv_df(closes)
        result, store = self._run_ingest_single(df)
        obs_calls = store.store_entity_observation.call_args_list
        vol_calls = [
            c
            for c in obs_calls
            if c.kwargs.get("observation_type") == "instrument_volatility"
        ]
        assert len(vol_calls) == 1
        assert vol_calls[0].kwargs["value"]["realized_vol_20d"] > 0

    def test_two_datapoints_vol_not_stored(self):
        """Exactly 2 closes → 1 return → code needs >= 2 returns for vol → NaN → not stored."""
        df = _make_ohlcv_df([100.0, 100.0])  # no change
        result, store = self._run_ingest_single(df)
        obs_calls = store.store_entity_observation.call_args_list
        vol_calls = [
            c
            for c in obs_calls
            if c.kwargs.get("observation_type") == "instrument_volatility"
        ]
        # 1 return → < 2 returns → realized_vol = NaN → not stored
        assert len(vol_calls) == 0


class TestIngestFailureRate:
    def test_over_50_percent_failure_raises(self):
        """If >50% of tickers fail, should raise RuntimeError."""
        store = _make_store()

        # Create batch where ticker is missing from columns
        good_df = _make_ohlcv_df([100.0, 101.0])
        batch = _make_batch_download({"GOOD": good_df})

        with patch(
            "agent.tools.instrument_universe.tradeable_instruments"
        ) as mock_trad:
            # 3 tickers but only 1 in the download → 2/3 fail = 66% > 50%
            mock_trad.return_value = [
                InstrumentDef("GOOD", "Good", "equity_etf", "US"),
                InstrumentDef("BAD1", "Bad1", "equity_etf", "US"),
                InstrumentDef("BAD2", "Bad2", "equity_etf", "US"),
            ]
            with patch("yfinance.download", return_value=batch):
                with pytest.raises(RuntimeError, match=">50%"):
                    ingest_daily_prices(store, as_of=date(2026, 4, 1))

    def test_download_exception_raises(self):
        """Full download failure raises RuntimeError."""
        store = _make_store()
        with patch(
            "agent.tools.instrument_universe.tradeable_instruments"
        ) as mock_trad:
            mock_trad.return_value = [InstrumentDef("SPY", "SPY", "equity_etf", "US")]
            with patch("yfinance.download", side_effect=Exception("API down")):
                with pytest.raises(RuntimeError, match="failed entirely"):
                    ingest_daily_prices(store, as_of=date(2026, 4, 1))


class TestIngestEntityRegistration:
    def test_entity_type_is_instrument(self):
        store = _make_store()
        df = _make_ohlcv_df([100.0, 101.0])
        batch = _make_batch_download({"SPY": df})

        with patch(
            "agent.tools.instrument_universe.tradeable_instruments"
        ) as mock_trad:
            mock_trad.return_value = [
                InstrumentDef("SPY", "S&P 500 ETF", "equity_etf", "US")
            ]
            with patch("yfinance.download", return_value=batch):
                ingest_daily_prices(store, as_of=date(2026, 4, 1))

        kw = store.register_entity.call_args.kwargs
        assert kw["entity_type"] == "instrument"
        assert kw["entity_id"] == _entity_id("SPY")
        assert kw["metadata"]["ticker"] == "SPY"
        assert kw["metadata"]["asset_class"] == "equity_etf"
        assert kw["metadata"]["region"] == "US"

    def test_observation_source_tool(self):
        store = _make_store()
        df = _make_ohlcv_df([100.0, 101.0])
        batch = _make_batch_download({"SPY": df})

        with patch(
            "agent.tools.instrument_universe.tradeable_instruments"
        ) as mock_trad:
            mock_trad.return_value = [
                InstrumentDef("SPY", "S&P 500 ETF", "equity_etf", "US")
            ]
            with patch("yfinance.download", return_value=batch):
                ingest_daily_prices(store, as_of=date(2026, 4, 1))

        for call in store.store_entity_observation.call_args_list:
            assert call.kwargs["source_tool"] == "instrument_universe"


# ── DAG callback ─────────────────────────────────────────────


class TestRunInstrumentIngest:
    def test_callback_opens_and_closes_store(self):
        with patch("agent.pipeline.store.PipelineStore") as MockStore:
            mock_store_inst = MagicMock()
            MockStore.return_value = mock_store_inst
            with patch(
                "agent.tools.instrument_universe.ingest_daily_prices"
            ) as mock_ingest:
                mock_ingest.return_value = {
                    "instruments_fetched": 0,
                    "instruments_failed": [],
                    "observations_stored": 0,
                }
                run_instrument_ingest({"db_path": "/tmp/test.db"}, {})

            MockStore.assert_called_once_with("/tmp/test.db")
            mock_store_inst.close.assert_called_once()


# ── Graph builder integration ────────────────────────────────


class TestGraphBuilderIntegration:
    def test_instrument_in_entity_types(self):
        from agent.models.gnn.graph_builder import ENTITY_TYPES

        assert "instrument" in ENTITY_TYPES

    def test_instrument_obs_types(self):
        from agent.models.gnn.graph_builder import OBSERVATION_TYPES

        assert "instrument_return" in OBSERVATION_TYPES
        assert "instrument_volume" in OBSERVATION_TYPES
        assert "instrument_volatility" in OBSERVATION_TYPES

    def test_enrichment_dim_updated(self):
        from agent.models.gnn.graph_builder import ENRICHMENT_DIM, OBSERVATION_TYPES

        # ENRICHMENT_DIM should account for obs_type_dist of length == len(OBSERVATION_TYPES)
        # Formula: cusum(1) + hawkes(1) + event_study(1) + bocpd(1) +
        #          value_var(1) + value_min(1) + value_max(1) + value_iqr(1) +
        #          num_tools(1) + obs_type_dist(len(OBS_TYPES))
        expected = 9 + len(OBSERVATION_TYPES)
        assert (
            ENRICHMENT_DIM == expected
        ), f"ENRICHMENT_DIM={ENRICHMENT_DIM} != expected {expected}"

    def test_base_feat_dim(self):
        from agent.models.gnn.graph_builder import BASE_FEAT_DIM, ENTITY_TYPES

        assert BASE_FEAT_DIM == len(ENTITY_TYPES) + 3


# ── Daily collection DAG integration ─────────────────────────


class TestDailyCollectionDAG:
    def test_dag_has_fetch_instruments_node(self):
        from agent.pipeline.dags.daily_collection import build_daily_collection_dag

        dag = build_daily_collection_dag()
        assert "fetch_instruments" in dag.nodes

    def test_fetch_instruments_no_deps(self):
        from agent.pipeline.dags.daily_collection import build_daily_collection_dag

        dag = build_daily_collection_dag()
        node = dag.nodes["fetch_instruments"]
        assert node.depends_on == []

    def test_dag_still_valid(self):
        from agent.pipeline.dags.daily_collection import build_daily_collection_dag

        dag = build_daily_collection_dag()
        errors = dag.validate()
        assert errors == [], f"DAG validation errors: {errors}"
