"""Global liquidity composite — fetch, align, normalize, compute."""

from __future__ import annotations

import pandas as pd

from agent.tools.macro_data import MacroDataTool
from agent.tools.market_data import MarketDataTool


class LiquidityComposite:
    """Builds a global liquidity composite from central bank balance sheet data."""

    def __init__(
        self,
        macro_tool: MacroDataTool,
        market_tool: MarketDataTool,
    ) -> None:
        self.macro = macro_tool
        self.market = market_tool

    # FRED series → column name mapping
    _US_SERIES = {
        "WALCL": "walcl",  # Fed balance sheet (millions USD, weekly Wed)
        "WTREGEN": "wtregen",  # Treasury General Account (millions USD, weekly Wed)
        "RRPONTSYD": "rrp",  # Overnight reverse repo (BILLIONS USD, daily)
        "M2SL": "m2",  # M2 money supply (billions USD, monthly)
    }

    def fetch_us(self, start: str, end: str) -> pd.DataFrame:
        """Fetch core US liquidity series from FRED.

        Returns DataFrame with DatetimeIndex and columns: walcl, wtregen, rrp, m2.
        All values in millions USD.
        """
        series_ids = ",".join(self._US_SERIES.keys())
        result = self.macro.execute(series_id=series_ids, start_date=start, end_date=end)
        if not result.success:
            raise ValueError(f"FRED fetch failed: {result.output}")

        frames: dict[str, pd.Series] = {}
        for fred_id, col_name in self._US_SERIES.items():
            obs = result.data.get(fred_id, [])
            if not obs:
                raise ValueError(f"No data returned for FRED series {fred_id}")
            dates = pd.to_datetime([o["date"] for o in obs])
            values = [float(o["value"]) for o in obs]
            s = pd.Series(values, index=dates, name=col_name)

            # Unit normalization: RRPONTSYD is in billions, everything else in millions
            if fred_id == "RRPONTSYD":
                s = s * 1000  # billions → millions
            # M2SL is in billions → convert to millions
            if fred_id == "M2SL":
                s = s * 1000  # billions → millions

            frames[col_name] = s

        df = pd.DataFrame(frames)
        df.index.name = "date"
        df = df.sort_index()
        return df

    # Global central bank FRED series
    # NOTE: BOE excluded — no reliable free FRED source for BOE total assets
    # (BOEBSTAUKA is annual % of GDP, UKASSETS discontinued 2014). BOE is ~5% of
    # major CB mass, so 3-CB composite (Fed+ECB+BOJ) covers ~95%.
    _GLOBAL_SERIES = {
        "ECBASSETSW": "ecb",  # ECB total assets (millions EUR, weekly)
        "JPNASSETS": "boj",  # BOJ total assets (100 millions JPY, monthly)
    }

    # FX tickers for USD conversion
    _FX_TICKERS = {
        "EURUSD=X": "eurusd",
        "JPY=X": "usdjpy",  # USD/JPY
    }

    def fetch_global(self, start: str, end: str) -> pd.DataFrame:
        """Fetch global central bank series (ECB, BOJ) + FX, merged with US.

        Returns DataFrame with US columns plus ecb_usd, boj_usd.
        All values in millions USD.
        """
        # Start with US data
        us = self.fetch_us(start, end)

        # Fetch global CB balance sheets from FRED
        global_ids = ",".join(self._GLOBAL_SERIES.keys())
        cb_result = self.macro.execute(series_id=global_ids, start_date=start, end_date=end)
        if not cb_result.success:
            raise ValueError(f"Global FRED fetch failed: {cb_result.output}")

        cb_frames: dict[str, pd.Series] = {}
        for fred_id, col_name in self._GLOBAL_SERIES.items():
            obs = cb_result.data.get(fred_id, [])
            if not obs:
                raise ValueError(f"No data returned for FRED series {fred_id}")
            dates = pd.to_datetime([o["date"] for o in obs])
            values = [float(o["value"]) for o in obs]
            cb_frames[col_name] = pd.Series(values, index=dates, name=col_name)

        # Fetch FX rates for USD conversion
        fx_tickers = ",".join(self._FX_TICKERS.keys())
        fx_result = self.market.execute(tickers=fx_tickers, period="max", interval="1d")

        fx_frames: dict[str, pd.Series] = {}
        for ticker, col_name in self._FX_TICKERS.items():
            bars = fx_result.data.get(ticker, [])
            if not bars:
                raise ValueError(f"No FX data for {ticker}")
            dates = pd.to_datetime([b["Date"] for b in bars], utc=True).tz_localize(None)
            values = [b["Close"] for b in bars]
            fx_frames[col_name] = pd.Series(values, index=dates, name=col_name)

        # Combine CB and FX into one frame, align on daily grid, forward-fill
        combined = pd.DataFrame({**cb_frames, **fx_frames})
        full_idx = pd.date_range(combined.index.min(), combined.index.max(), freq="D")
        combined = combined.reindex(full_idx).ffill()

        # Convert to USD millions
        # ECB: millions EUR * EURUSD rate = millions USD
        combined["ecb_usd"] = combined["ecb"] * combined["eurusd"]
        # BOJ: 100 millions JPY → millions JPY (* 100), then / USDJPY = millions USD
        combined["boj_usd"] = (combined["boj"] * 100) / combined["usdjpy"]

        # Merge global USD columns into US data
        global_cols = combined[["ecb_usd", "boj_usd"]]
        merged = us.join(global_cols, how="outer")
        merged.index.name = "date"
        merged = merged.sort_index()
        return merged

    def compute(self, raw: pd.DataFrame, *, global_: bool = False) -> pd.Series:
        """Compute detrended liquidity composite from raw data.

        Steps:
        1. Align all series to weekly Wednesday grid (forward-fill across frequencies).
        2. Compute net liquidity level.
        3. Detrend: first difference → rolling z-score (52-week window).

        Returns z-scored ΔLiquidity series (NaNs during warm-up period dropped).
        """
        weekly = self._align_weekly(raw)

        global_cols = ["ecb_usd", "boj_usd"]
        if global_ and all(c in weekly.columns for c in global_cols):
            net = weekly["walcl"] - weekly["wtregen"] - weekly["rrp"] + weekly["ecb_usd"] + weekly["boj_usd"]
        else:
            net = weekly["walcl"] - weekly["wtregen"] - weekly["rrp"]

        # Detrend: first difference removes level, z-score normalizes scale
        delta = net.diff()
        z_window = 52  # 1-year lookback
        rolling_mean = delta.rolling(window=z_window, min_periods=z_window).mean()
        rolling_std = delta.rolling(window=z_window, min_periods=z_window).std()
        z_scored = (delta - rolling_mean) / rolling_std

        z_scored.name = "liquidity_zscore"
        return z_scored.dropna()

    @staticmethod
    def _align_weekly(raw: pd.DataFrame) -> pd.DataFrame:
        """Resample mixed-frequency data to weekly Wednesday grid via forward-fill.

        Handles mixed frequencies (daily RRP, weekly WALCL/WTREGEN, monthly M2):
        - Reindex to complete daily grid
        - Forward-fill each column independently
        - RRP filled with 0 before series start (per spec: unavailable before 2013)
        - Resample to weekly Wednesday (last known value)
        """
        # Create complete daily grid
        full_idx = pd.date_range(raw.index.min(), raw.index.max(), freq="D")
        daily = raw.reindex(full_idx)

        # RRP: fill with 0 before series starts (RRPONTSYD began ~2013)
        if "rrp" in daily.columns:
            first_valid = daily["rrp"].first_valid_index()
            if first_valid is not None:
                daily.loc[daily.index < first_valid, "rrp"] = 0.0

        # Forward-fill each column
        daily = daily.ffill()

        # Resample to weekly Wednesday, taking last value in each week
        weekly = daily.resample("W-WED").last()

        # Only require the core formula columns (walcl, wtregen, rrp)
        required = ["walcl", "wtregen", "rrp"]
        present_required = [c for c in required if c in weekly.columns]
        return weekly.dropna(subset=present_required)
