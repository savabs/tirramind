"""Tests for agent/tools/instrument_universe.py — Phase 24a."""

from __future__ import annotations

import math
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agent.tools.instrument_universe import (
    INSTRUMENTS,
    InstrumentDef,
    _entity_id,
    _persist_instrument_links,
    cftc_code_to_ticker,
    ingest_daily_prices,
    instruments_by_class,
    run_instrument_ingest,
    ticker_to_instrument,
    tradeable_instruments,
)

# ── Helpers ───────────────────────────────────────────────────


def _make_store() -> MagicMock:
    store = MagicMock()
    store.register_entity = MagicMock(side_effect=lambda **kw: kw["entity_id"])
    store.store_entity_observation = MagicMock(return_value=1)
    store.link_entities = MagicMock(return_value=1)
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
            assert inst.asset_class in valid_classes, f"{inst.ticker}: invalid asset_class={inst.asset_class}"

    def test_all_have_region(self):
        valid_regions = {"US", "Europe", "Asia", "LatAm", "Pacific", "Global", "EM"}
        for inst in INSTRUMENTS:
            assert inst.region in valid_regions, f"{inst.ticker}: invalid region={inst.region}"


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

    def _run_ingest(self, ticker_data: dict[str, pd.DataFrame]) -> tuple[dict, MagicMock]:
        """Helper: run ingest with given per-ticker data."""
        store = _make_store()
        batch_df = _make_batch_download(ticker_data)

        with patch("agent.tools.instrument_universe.tradeable_instruments") as mock_trad:
            # Only return instruments matching our test tickers
            mock_trad.return_value = [InstrumentDef(t, f"Test {t}", "equity_etf", "US") for t in ticker_data]
            with patch("yfinance.download", return_value=batch_df):
                result = ingest_daily_prices(store, as_of=date(2026, 4, 1))

        return result, store

    def test_single_ticker_registered(self):
        df = _make_ohlcv_df([100.0, 105.0])
        result, store = self._run_ingest({"SPY": df})

        assert result["instruments_fetched"] == 1
        assert result["instruments_failed"] == []
        # Find the instrument entity registration among all register_entity calls
        # (Phase 25 _persist_instrument_links also registers issuer/country entities)
        inst_calls = [c for c in store.register_entity.call_args_list if c.kwargs.get("entity_type") == "instrument"]
        assert len(inst_calls) == 1
        assert inst_calls[0].kwargs["canonical_name"] == "Test SPY"

    def test_log_return_computation(self):
        # close yesterday=100, today=105 → ln(105/100) ≈ 0.04879
        df = _make_ohlcv_df([100.0, 105.0])
        _, store = self._run_ingest({"SPY": df})

        obs_calls = store.store_entity_observation.call_args_list
        return_call = [c for c in obs_calls if c.kwargs.get("observation_type") == "instrument_return"]
        assert len(return_call) == 1
        log_ret = return_call[0].kwargs["value"]["log_return"]
        assert abs(log_ret - math.log(105 / 100)) < 1e-10

    def test_volume_observation_stored(self):
        df = _make_ohlcv_df([100.0, 102.0], volumes=[500_000, 750_000])
        _, store = self._run_ingest({"SPY": df})

        obs_calls = store.store_entity_observation.call_args_list
        vol_calls = [c for c in obs_calls if c.kwargs.get("observation_type") == "instrument_volume"]
        assert len(vol_calls) == 1
        assert vol_calls[0].kwargs["value"]["volume"] == 750_000.0

    def test_volatility_with_enough_data(self):
        # 25 data points → enough for 20d vol
        closes = [100.0 + i * 0.5 for i in range(25)]
        df = _make_ohlcv_df(closes)
        _, store = self._run_ingest({"SPY": df})

        obs_calls = store.store_entity_observation.call_args_list
        vol_calls = [c for c in obs_calls if c.kwargs.get("observation_type") == "instrument_volatility"]
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
        inst_calls = [c for c in store.register_entity.call_args_list if c.kwargs.get("entity_type") == "instrument"]
        assert len(inst_calls) == 2

    def test_three_obs_per_ticker_with_enough_data(self):
        # With 25 data points: return + volume + volatility = 3 obs
        closes = [100.0 + i * 0.5 for i in range(25)]
        df = _make_ohlcv_df(closes)
        result, store = self._run_ingest({"SPY": df})
        assert result["observations_stored"] == 3


class TestIngestEdgeCases:
    def _run_ingest_single(self, df: pd.DataFrame, ticker: str = "SPY") -> tuple[dict, MagicMock]:
        store = _make_store()
        batch_df = _make_batch_download({ticker: df})
        with patch("agent.tools.instrument_universe.tradeable_instruments") as mock_trad:
            mock_trad.return_value = [InstrumentDef(ticker, f"Test {ticker}", "equity_etf", "US")]
            with patch("yfinance.download", return_value=batch_df):
                result = ingest_daily_prices(store, as_of=date(2026, 4, 1))
        return result, store

    def test_single_day_no_return(self):
        """Only 1 close → can't compute return → only volume obs stored."""
        df = _make_ohlcv_df([100.0])
        result, store = self._run_ingest_single(df)
        assert result["instruments_fetched"] == 1
        obs_calls = store.store_entity_observation.call_args_list
        return_calls = [c for c in obs_calls if c.kwargs.get("observation_type") == "instrument_return"]
        assert len(return_calls) == 0  # no return obs
        vol_calls = [c for c in obs_calls if c.kwargs.get("observation_type") == "instrument_volume"]
        assert len(vol_calls) == 1  # volume still stored

    def test_nan_close_rows_dropped(self):
        """Rows with NaN close should be dropped."""
        df = _make_ohlcv_df([100.0, float("nan"), 102.0])
        result, store = self._run_ingest_single(df)
        assert result["instruments_fetched"] == 1
        obs_calls = store.store_entity_observation.call_args_list
        return_calls = [c for c in obs_calls if c.kwargs.get("observation_type") == "instrument_return"]
        assert len(return_calls) == 1

    def test_zero_volume(self):
        """Zero volume should be stored, not skipped."""
        df = _make_ohlcv_df([100.0, 101.0], volumes=[0, 0])
        result, store = self._run_ingest_single(df)
        obs_calls = store.store_entity_observation.call_args_list
        vol_calls = [c for c in obs_calls if c.kwargs.get("observation_type") == "instrument_volume"]
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
        vol_calls = [c for c in obs_calls if c.kwargs.get("observation_type") == "instrument_volatility"]
        assert len(vol_calls) == 1
        assert vol_calls[0].kwargs["value"]["realized_vol_20d"] > 0

    def test_two_datapoints_vol_not_stored(self):
        """Exactly 2 closes → 1 return → code needs >= 2 returns for vol → NaN → not stored."""
        df = _make_ohlcv_df([100.0, 100.0])  # no change
        result, store = self._run_ingest_single(df)
        obs_calls = store.store_entity_observation.call_args_list
        vol_calls = [c for c in obs_calls if c.kwargs.get("observation_type") == "instrument_volatility"]
        # 1 return → < 2 returns → realized_vol = NaN → not stored
        assert len(vol_calls) == 0


class TestIngestFailureRate:
    def test_over_50_percent_failure_raises(self):
        """If >50% of tickers fail, should raise RuntimeError."""
        store = _make_store()

        # Create batch where ticker is missing from columns
        good_df = _make_ohlcv_df([100.0, 101.0])
        batch = _make_batch_download({"GOOD": good_df})

        with patch("agent.tools.instrument_universe.tradeable_instruments") as mock_trad:
            # 3 tickers but only 1 in the download → 2/3 fail = 66% > 50%
            mock_trad.return_value = [
                InstrumentDef("GOOD", "Good", "equity_etf", "US"),
                InstrumentDef("BAD1", "Bad1", "equity_etf", "US"),
                InstrumentDef("BAD2", "Bad2", "equity_etf", "US"),
            ]
            with patch("yfinance.download", return_value=batch), pytest.raises(RuntimeError, match=">50%"):
                ingest_daily_prices(store, as_of=date(2026, 4, 1))

    def test_download_exception_raises(self):
        """Full download failure raises RuntimeError."""
        store = _make_store()
        with patch("agent.tools.instrument_universe.tradeable_instruments") as mock_trad:
            mock_trad.return_value = [InstrumentDef("SPY", "SPY", "equity_etf", "US")]
            with patch("yfinance.download", side_effect=Exception("API down")):
                with pytest.raises(RuntimeError, match="failed entirely"):
                    ingest_daily_prices(store, as_of=date(2026, 4, 1))


class TestIngestEntityRegistration:
    def test_entity_type_is_instrument(self):
        store = _make_store()
        df = _make_ohlcv_df([100.0, 101.0])
        batch = _make_batch_download({"SPY": df})

        with patch("agent.tools.instrument_universe.tradeable_instruments") as mock_trad:
            mock_trad.return_value = [InstrumentDef("SPY", "S&P 500 ETF", "equity_etf", "US")]
            with patch("yfinance.download", return_value=batch):
                ingest_daily_prices(store, as_of=date(2026, 4, 1))

        kw_list = [
            c.kwargs for c in store.register_entity.call_args_list if c.kwargs.get("entity_type") == "instrument"
        ]
        assert len(kw_list) == 1
        kw = kw_list[0]
        assert kw["entity_type"] == "instrument"
        assert kw["entity_id"] == _entity_id("SPY")
        assert kw["metadata"]["ticker"] == "SPY"
        assert kw["metadata"]["asset_class"] == "equity_etf"
        assert kw["metadata"]["region"] == "US"

    def test_observation_source_tool(self):
        store = _make_store()
        df = _make_ohlcv_df([100.0, 101.0])
        batch = _make_batch_download({"SPY": df})

        with patch("agent.tools.instrument_universe.tradeable_instruments") as mock_trad:
            mock_trad.return_value = [InstrumentDef("SPY", "S&P 500 ETF", "equity_etf", "US")]
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
            with patch("agent.tools.instrument_universe.ingest_daily_prices") as mock_ingest:
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
        assert expected == ENRICHMENT_DIM, f"ENRICHMENT_DIM={ENRICHMENT_DIM} != expected {expected}"

    def test_base_feat_dim(self):
        from agent.models.gnn.graph_builder import BASE_FEAT_DIM, ENTITY_TYPES

        assert len(ENTITY_TYPES) + 3 == BASE_FEAT_DIM


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


# ── Phase 25: Cross-domain link persistence tests ────────────


class TestCftcCodeToTicker:
    """Tests for cftc_code_to_ticker() helper."""

    def test_returns_dict(self):
        result = cftc_code_to_ticker()
        assert isinstance(result, dict)

    def test_all_values_are_tickers(self):
        ticker_set = {i.ticker for i in INSTRUMENTS}
        for code, ticker in cftc_code_to_ticker().items():
            assert ticker in ticker_set, f"cftc_code {code} maps to unknown ticker {ticker}"

    def test_all_keys_are_strings(self):
        for code in cftc_code_to_ticker():
            assert isinstance(code, str)
            assert len(code) > 0

    def test_known_mapping(self):
        mapping = cftc_code_to_ticker()
        assert mapping.get("088691") == "GC=F"  # Gold
        assert mapping.get("023651") == "NG=F"  # Natural Gas

    def test_instruments_without_cftc_excluded(self):
        mapping = cftc_code_to_ticker()
        # SPY is an ETF — no CFTC code
        assert "SPY" not in mapping.values() or any(i.cftc_code for i in INSTRUMENTS if i.ticker == "SPY")

    def test_no_duplicate_codes(self):
        codes = [i.cftc_code for i in INSTRUMENTS if i.cftc_code]
        assert len(codes) == len(set(codes)), "Duplicate CFTC codes in INSTRUMENTS"


class TestPersistInstrumentLinks:
    """Tests for _persist_instrument_links() Phase 25 link creation."""

    def test_returns_count_dict(self):
        store = _make_store()
        result = _persist_instrument_links(store)
        assert isinstance(result, dict)
        assert "tracks_issuer" in result
        assert "inst_country" in result
        assert "issuer_country" in result

    def test_creates_issuer_links(self):
        store = _make_store()
        result = _persist_instrument_links(store)
        # At least some ETFs have issuers
        assert result["tracks_issuer"] > 0

    def test_creates_country_links(self):
        store = _make_store()
        result = _persist_instrument_links(store)
        # Most instruments have country
        assert result["inst_country"] > 0

    def test_creates_issuer_country_links(self):
        store = _make_store()
        result = _persist_instrument_links(store)
        assert result["issuer_country"] > 0

    def test_register_entity_called_for_issuers(self):
        store = _make_store()
        _persist_instrument_links(store)
        company_calls = [c for c in store.register_entity.call_args_list if c.kwargs.get("entity_type") == "company"]
        assert len(company_calls) > 0

    def test_register_entity_called_for_countries(self):
        store = _make_store()
        _persist_instrument_links(store)
        country_calls = [c for c in store.register_entity.call_args_list if c.kwargs.get("entity_type") == "country"]
        assert len(country_calls) > 0

    def test_issuer_dedup(self):
        """Multiple instruments from same issuer should register only once."""
        store = _make_store()
        _persist_instrument_links(store)
        company_calls = [c for c in store.register_entity.call_args_list if c.kwargs.get("entity_type") == "company"]
        registered_names = [c.kwargs["canonical_name"] for c in company_calls]
        assert len(registered_names) == len(set(registered_names)), (
            f"Duplicate issuer registrations: {registered_names}"
        )

    def test_link_entities_called_for_tracks_issuer(self):
        store = _make_store()
        _persist_instrument_links(store)
        issuer_links = [c for c in store.link_entities.call_args_list if c.kwargs.get("link_type") == "tracks_issuer"]
        assert len(issuer_links) > 0

    def test_link_entities_called_for_located_in(self):
        store = _make_store()
        _persist_instrument_links(store)
        located_links = [c for c in store.link_entities.call_args_list if c.kwargs.get("link_type") == "located_in"]
        assert len(located_links) > 0

    def test_link_confidence_is_one(self):
        """Deterministic metadata links should have confidence=1.0."""
        store = _make_store()
        _persist_instrument_links(store)
        for call in store.link_entities.call_args_list:
            assert call.kwargs["confidence"] == 1.0

    def test_link_source_is_instrument_universe(self):
        store = _make_store()
        _persist_instrument_links(store)
        for call in store.link_entities.call_args_list:
            assert call.kwargs["source"] == "instrument_universe"

    def test_no_links_for_instrument_without_issuer_or_country(self):
        """Instruments with no issuer and no country produce no links."""
        # VIX has no issuer and no country:
        vix = next((i for i in INSTRUMENTS if i.ticker == "^VIX"), None)
        if vix is None:
            pytest.skip("VIX not in INSTRUMENTS")
        # VIX should not appear in any link call
        store = _make_store()
        _persist_instrument_links(store)
        link_tickers = []
        for call in store.link_entities.call_args_list:
            md = call.kwargs.get("metadata") or {}
            if "ticker" in md:
                link_tickers.append(md["ticker"])
        # If VIX has no issuer and no country, it shouldn't appear
        if not vix.issuer and not vix.country:
            assert "^VIX" not in link_tickers

    def test_link_entities_returns_zero_skipped(self):
        """When link_entities returns 0 (duplicate), count stays at 0."""
        store = _make_store()
        store.link_entities = MagicMock(return_value=0)
        result = _persist_instrument_links(store)
        assert result["tracks_issuer"] == 0
        assert result["inst_country"] == 0
        assert result["issuer_country"] == 0

    def test_idempotent_link_creation(self):
        """Running twice produces the same structure."""
        store = _make_store()
        r1 = _persist_instrument_links(store)
        store.register_entity.reset_mock()
        store.link_entities.reset_mock()
        r2 = _persist_instrument_links(store)
        assert r1 == r2

    def test_country_entity_ids_deterministic(self):
        """Country entity IDs are deterministic from country code."""
        from agent.pipeline.entity import entity_id_from_key

        store = _make_store()
        _persist_instrument_links(store)
        country_calls = [c for c in store.register_entity.call_args_list if c.kwargs.get("entity_type") == "country"]
        for call in country_calls:
            expected = entity_id_from_key("country", call.kwargs["canonical_name"])
            assert call.kwargs["entity_id"] == expected


class TestInstrumentMetadataEnrichment:
    """Verify enriched InstrumentDef fields."""

    def test_all_etfs_have_issuer(self):
        etfs = [i for i in INSTRUMENTS if "etf" in i.asset_class]
        for etf in etfs:
            assert etf.issuer is not None, f"{etf.ticker} is ETF but has no issuer"

    def test_most_instruments_have_country(self):
        with_country = [i for i in INSTRUMENTS if i.country]
        # At least 70% should have country (commodity futures are exchange-traded, not country-specific)
        assert len(with_country) >= len(INSTRUMENTS) * 0.7

    def test_cftc_codes_on_commodity_futures(self):
        commodities = instruments_by_class("commodity_futures")
        with_code = [i for i in commodities if i.cftc_code]
        # Most commodity futures should have CFTC code
        assert len(with_code) >= len(commodities) * 0.7

    def test_no_cftc_code_on_etfs(self):
        etfs = [i for i in INSTRUMENTS if "etf" in i.asset_class]
        for etf in etfs:
            assert etf.cftc_code is None, f"ETF {etf.ticker} shouldn't have CFTC code"

    def test_optional_fields_default_none(self):
        bare = InstrumentDef("TEST", "Test", "equity_etf", "US")
        assert bare.issuer is None
        assert bare.country is None
        assert bare.cftc_code is None

    def test_country_codes_are_iso(self):
        """Country codes should be 2-letter ISO codes or short identifiers."""
        for inst in INSTRUMENTS:
            if inst.country:
                assert len(inst.country) <= 6, f"{inst.ticker} country {inst.country!r} too long"
                assert inst.country == inst.country.upper(), f"{inst.ticker} country {inst.country!r} not uppercase"


# ── Phase 27: FX two-country metadata tests ──────────────────


class TestFXTwoCountryMetadata:
    """27.1 — explicit base_country/quote_country on FX instruments."""

    def test_all_fx_have_base_country(self):
        fx = instruments_by_class("fx")
        for inst in fx:
            assert inst.base_country is not None, f"{inst.ticker} is FX but has no base_country"

    def test_all_fx_have_quote_country(self):
        fx = instruments_by_class("fx")
        for inst in fx:
            assert inst.quote_country is not None, f"{inst.ticker} is FX but has no quote_country"

    def test_base_and_quote_differ(self):
        """Base and quote should be different countries for every FX pair."""
        fx = instruments_by_class("fx")
        for inst in fx:
            assert inst.base_country != inst.quote_country, (
                f"{inst.ticker}: base_country == quote_country ({inst.base_country})"
            )

    def test_base_country_codes_are_iso(self):
        fx = instruments_by_class("fx")
        for inst in fx:
            assert inst.base_country == inst.base_country.upper()
            assert 2 <= len(inst.base_country) <= 6

    def test_quote_country_codes_are_iso(self):
        fx = instruments_by_class("fx")
        for inst in fx:
            assert inst.quote_country == inst.quote_country.upper()
            assert 2 <= len(inst.quote_country) <= 6

    def test_non_fx_have_no_base_quote(self):
        """Non-FX instruments should not have base_country/quote_country."""
        non_fx = [i for i in INSTRUMENTS if i.asset_class != "fx"]
        for inst in non_fx:
            assert inst.base_country is None, f"{inst.ticker} ({inst.asset_class}) should not have base_country"
            assert inst.quote_country is None, f"{inst.ticker} ({inst.asset_class}) should not have quote_country"

    def test_fx_country_field_still_present(self):
        """Backward compat: all FX pairs still have the legacy country field."""
        fx = instruments_by_class("fx")
        for inst in fx:
            assert inst.country is not None, f"{inst.ticker} is FX but country is None"

    def test_fx_country_is_one_of_pair_sides(self):
        """The legacy country field should match either base or quote."""
        fx = instruments_by_class("fx")
        for inst in fx:
            assert inst.country in (inst.base_country, inst.quote_country), (
                f"{inst.ticker}: country={inst.country} not in ({inst.base_country}, {inst.quote_country})"
            )

    def test_known_pair_mappings(self):
        """Spot-check specific deterministic pair → country mappings."""
        lookup = ticker_to_instrument()
        # EUR/USD
        assert lookup["EURUSD=X"].base_country == "EU"
        assert lookup["EURUSD=X"].quote_country == "US"
        # USD/JPY
        assert lookup["USDJPY=X"].base_country == "US"
        assert lookup["USDJPY=X"].quote_country == "JP"
        # GBP/JPY (cross, no USD side)
        assert lookup["GBPJPY=X"].base_country == "GB"
        assert lookup["GBPJPY=X"].quote_country == "JP"
        # USD/ZAR (EM)
        assert lookup["USDZAR=X"].base_country == "US"
        assert lookup["USDZAR=X"].quote_country == "ZA"

    def test_new_fields_default_none(self):
        """New InstrumentDef without FX fields defaults to None."""
        bare = InstrumentDef("TEST", "Test", "equity_etf", "US")
        assert bare.base_country is None
        assert bare.quote_country is None

    def test_frozen_fx_fields(self):
        """FX country fields are frozen (immutable)."""
        inst = InstrumentDef(
            "TEST=X",
            "Test FX",
            "fx",
            "Global",
            base_country="US",
            quote_country="JP",
        )
        with pytest.raises(AttributeError):
            inst.base_country = "GB"  # type: ignore
        with pytest.raises(AttributeError):
            inst.quote_country = "GB"  # type: ignore

    def test_fx_pair_count_unchanged(self):
        """Exactly 15 FX instruments after metadata enrichment."""
        assert len(instruments_by_class("fx")) == 15


# ── Phase 27: FX link persistence tests ──────────────────────


class TestFXLinkPersistence:
    """27.2 — _persist_instrument_links creates fx_base_country/fx_quote_country."""

    def test_returns_fx_link_counts(self):
        store = _make_store()
        result = _persist_instrument_links(store)
        assert "fx_base_country" in result
        assert "fx_quote_country" in result

    def test_fx_base_country_links_created(self):
        store = _make_store()
        result = _persist_instrument_links(store)
        # All 15 FX pairs have base_country → 15 links
        assert result["fx_base_country"] == 15

    def test_fx_quote_country_links_created(self):
        store = _make_store()
        result = _persist_instrument_links(store)
        # All 15 FX pairs have quote_country → 15 links
        assert result["fx_quote_country"] == 15

    def test_fx_links_use_correct_link_types(self):
        store = _make_store()
        _persist_instrument_links(store)
        link_types = {c.kwargs["link_type"] for c in store.link_entities.call_args_list}
        assert "fx_base_country" in link_types
        assert "fx_quote_country" in link_types

    def test_fx_links_point_to_country_entities(self):
        """FX base/quote links should target country entity IDs."""

        store = _make_store()
        _persist_instrument_links(store)
        fx_link_calls = [
            c
            for c in store.link_entities.call_args_list
            if c.kwargs["link_type"] in ("fx_base_country", "fx_quote_country")
        ]
        # Collect all target entity IDs from FX links
        target_eids = {c.kwargs["entity_id_b"] for c in fx_link_calls}
        # All should be valid country entity IDs
        registered_country_eids = {
            c.kwargs["entity_id"]
            for c in store.register_entity.call_args_list
            if c.kwargs.get("entity_type") == "country"
        }
        assert target_eids.issubset(registered_country_eids)

    def test_fx_links_confidence_is_one(self):
        """Deterministic FX country links have confidence=1.0."""
        store = _make_store()
        _persist_instrument_links(store)
        fx_calls = [
            c
            for c in store.link_entities.call_args_list
            if c.kwargs["link_type"] in ("fx_base_country", "fx_quote_country")
        ]
        for call in fx_calls:
            assert call.kwargs["confidence"] == 1.0

    def test_fx_links_source(self):
        store = _make_store()
        _persist_instrument_links(store)
        fx_calls = [
            c
            for c in store.link_entities.call_args_list
            if c.kwargs["link_type"] in ("fx_base_country", "fx_quote_country")
        ]
        for call in fx_calls:
            assert call.kwargs["source"] == "instrument_universe"

    def test_fx_links_include_ticker_metadata(self):
        store = _make_store()
        _persist_instrument_links(store)
        fx_calls = [
            c
            for c in store.link_entities.call_args_list
            if c.kwargs["link_type"] in ("fx_base_country", "fx_quote_country")
        ]
        for call in fx_calls:
            assert "ticker" in call.kwargs["metadata"]

    def test_non_fx_no_fx_links(self):
        """Non-FX instruments should not produce fx_base/quote links."""
        store = _make_store()
        _persist_instrument_links(store)
        fx_calls = [
            c
            for c in store.link_entities.call_args_list
            if c.kwargs["link_type"] in ("fx_base_country", "fx_quote_country")
        ]
        fx_tickers = {i.ticker for i in INSTRUMENTS if i.asset_class == "fx"}
        for call in fx_calls:
            assert call.kwargs["metadata"]["ticker"] in fx_tickers

    def test_eurusd_links_to_both_eu_and_us(self):
        """EUR/USD should create links to both EU and US country nodes."""
        from agent.pipeline.entity import entity_id_from_key

        store = _make_store()
        _persist_instrument_links(store)
        eurusd_eid = _entity_id("EURUSD=X")
        eu_eid = entity_id_from_key("country", "EU")
        us_eid = entity_id_from_key("country", "US")

        fx_calls = [
            c
            for c in store.link_entities.call_args_list
            if c.kwargs.get("metadata", {}).get("ticker") == "EURUSD=X"
            and c.kwargs["link_type"] in ("fx_base_country", "fx_quote_country")
        ]
        targets = {c.kwargs["entity_id_b"] for c in fx_calls}
        assert eu_eid in targets, "EUR/USD missing link to EU"
        assert us_eid in targets, "EUR/USD missing link to US"

    def test_gbpjpy_cross_links_no_usd(self):
        """GBP/JPY (cross pair) should link to GB and JP, not US."""
        from agent.pipeline.entity import entity_id_from_key

        store = _make_store()
        _persist_instrument_links(store)
        gb_eid = entity_id_from_key("country", "GB")
        jp_eid = entity_id_from_key("country", "JP")

        gbpjpy_fx_calls = [
            c
            for c in store.link_entities.call_args_list
            if c.kwargs.get("metadata", {}).get("ticker") == "GBPJPY=X"
            and c.kwargs["link_type"] in ("fx_base_country", "fx_quote_country")
        ]
        targets = {c.kwargs["entity_id_b"] for c in gbpjpy_fx_calls}
        assert gb_eid in targets
        assert jp_eid in targets
        assert len(targets) == 2  # exactly two distinct country targets

    def test_idempotent_with_fx_links(self):
        """Running _persist_instrument_links twice produces same counts."""
        store = _make_store()
        r1 = _persist_instrument_links(store)
        store.register_entity.reset_mock()
        store.link_entities.reset_mock()
        r2 = _persist_instrument_links(store)
        assert r1 == r2

    def test_fx_link_returns_zero_skipped(self):
        """When link_entities returns 0 (duplicate), FX counts stay 0."""
        store = _make_store()
        store.link_entities = MagicMock(return_value=0)
        result = _persist_instrument_links(store)
        assert result["fx_base_country"] == 0
        assert result["fx_quote_country"] == 0

    def test_legacy_located_in_still_created_for_fx(self):
        """FX instruments with a country field still get located_in links."""
        store = _make_store()
        _persist_instrument_links(store)
        fx_tickers = {i.ticker for i in INSTRUMENTS if i.asset_class == "fx"}
        located_in_calls = [
            c
            for c in store.link_entities.call_args_list
            if c.kwargs["link_type"] == "located_in" and c.kwargs.get("metadata", {}).get("ticker") in fx_tickers
        ]
        # All 15 FX pairs have country → 15 located_in links
        assert len(located_in_calls) == 15

    def test_existing_non_fx_links_unchanged(self):
        """Phase 25 link counts for non-FX instruments are not affected."""
        store = _make_store()
        result = _persist_instrument_links(store)
        # tracks_issuer and issuer_country should be > 0 (ETFs have issuers)
        assert result["tracks_issuer"] > 0
        assert result["issuer_country"] > 0
