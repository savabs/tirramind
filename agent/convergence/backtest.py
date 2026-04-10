"""Historical macro backtest for the convergence engine.

Uses FRED time series with decades of history to build Evidence objects,
runs the ConvergenceDetector at each weekly step, and feeds the detection
signal into the WalkForward backtester to evaluate predictive value against
SPY / TLT / GLD returns.

Architecture:
    1. ``FredSeriesConfig`` — maps a FRED series to a signal_id + category
    2. ``HistoricalEvidenceBuilder`` — converts FRED DataFrames → Evidence
    3. ``precompute_convergence_scores()`` — runs the detector at every step
    4. ``ConvergenceRiskOffStrategy`` — binary exit on stress convergence
    5. ``ConvergenceDirectionalStrategy`` — proportional score × direction
    6. ``run_macro_backtest()`` — orchestrates everything

No look-ahead: all Evidence is generated via ``as_of`` filtering.
"""

from __future__ import annotations

import bisect
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from agent.convergence.detector import (
    ConvergenceDetector,
    ConvergenceDetectorConfig,
    DetectionResult,
)
from agent.convergence.evidence import Evidence
from agent.convergence.taxonomy import SignalMeta, SignalRegistry
from agent.pipeline.store import PipelineStore
from agent.quant.backtest import BacktestResult, Strategy, WalkForward
from agent.quant.scoring import block_bootstrap_ci, score_returns

log = logging.getLogger(__name__)

_DAY = 86_400
_WEEK = 7 * _DAY
_DEFAULT_TARGETS = ["SPY", "TLT", "GLD"]
_DEFAULT_BOOTSTRAP_COUNT = 1000
_FAST_BOOTSTRAP_COUNT = 200
_FAST_START_YEAR = 2018


def _get_fred_api_key() -> str:
    """Return the configured FRED API key.

    Prefers the repo-standard ``TIRRA_FRED_API_KEY`` but also accepts
    ``FRED_API_KEY`` for direct CLI use.
    """
    return os.environ.get("TIRRA_FRED_API_KEY", "") or os.environ.get(
        "FRED_API_KEY", ""
    )


def _is_placeholder_api_key(api_key: str) -> bool:
    """Return True when the configured API key is an obvious placeholder."""
    normalized = api_key.strip().lower()
    return normalized in {
        "",
        "your-key-here",
        "your_api_key_here",
        "your-fred-api-key",
        "replace-me",
        "changeme",
    }


# ── FRED→Evidence Mapping ──────────────────────────────────────


@dataclass(frozen=True)
class FredSeriesConfig:
    """Describes how a single FRED series maps to convergence Evidence."""

    fred_id: str
    """FRED series code (e.g. 'DFF', 'WALCL')."""

    signal_id: str
    """Convergence signal_id (e.g. 'rate_monitor.fed.rate')."""

    category: str
    """Taxonomy category."""

    direction_rule: str
    """How to derive direction from the value:
    'delta_pos_up'   → Δ > 0 ⇒ +1
    'delta_pos_down' → Δ > 0 ⇒ -1
    'level_below_50' → value < 50 ⇒ +1
    'delta_neg_up'   → Δ < 0 ⇒ +1
    """

    frequency: str = "weekly"
    """Observation frequency for the registry."""


# The 13 FRED series from the spec.
FRED_SERIES: list[FredSeriesConfig] = [
    FredSeriesConfig(
        "WALCL", "central_bank.fed.assets", "monetary_policy", "delta_pos_up", "weekly"
    ),
    FredSeriesConfig(
        "RRPONTSYD",
        "central_bank.fed.rrp",
        "monetary_policy",
        "delta_pos_down",
        "daily",
    ),
    FredSeriesConfig(
        "DFF", "rate_monitor.fed.rate", "monetary_policy", "delta_pos_down", "daily"
    ),
    FredSeriesConfig(
        "DGS10", "sovereign_debt.us.10y", "financial_stress", "delta_pos_up", "daily"
    ),
    FredSeriesConfig(
        "T10Y2Y",
        "sovereign_debt.us.curve",
        "financial_stress",
        "level_below_0",
        "daily",
    ),
    FredSeriesConfig(
        "BAMLH0A0HYM2",
        "creditor.us.hy_spread",
        "financial_stress",
        "delta_pos_up",
        "daily",
    ),
    FredSeriesConfig(
        "UNRATE", "jobs.us.unemployment", "macro_momentum", "delta_pos_up", "monthly"
    ),
    FredSeriesConfig(
        "ICSA", "jobs.us.claims", "macro_momentum", "delta_pos_up", "weekly"
    ),
    # NAPM/ISM is not a durable free programmatic input. Keep the legacy
    # macro-momentum slot stable for template compatibility, but source it from
    # the maintained US leading index series instead.
    FredSeriesConfig(
        "USSLIND", "pmi.us.manufacturing", "macro_momentum", "level_below_0", "monthly"
    ),
    FredSeriesConfig(
        "UMCSENT", "consumer.us.sentiment", "macro_momentum", "delta_neg_up", "monthly"
    ),
    FredSeriesConfig(
        "PERMIT", "building.us.permits", "macro_momentum", "delta_neg_up", "monthly"
    ),
    FredSeriesConfig(
        "CPIAUCSL", "cpi.us.headline", "macro_momentum", "delta_pos_up", "monthly"
    ),
    FredSeriesConfig(
        "M2SL", "monetary.us.m2", "monetary_policy", "delta_pos_up", "monthly"
    ),
]


def _apply_direction_rule(
    rule: str,
    value: float,
    prev_value: float | None,
) -> int:
    """Return +1, -1, or 0 based on the direction rule."""
    if prev_value is None:
        return 0
    delta = value - prev_value
    if rule == "delta_pos_up":
        return 1 if delta > 0 else (-1 if delta < 0 else 0)
    if rule == "delta_pos_down":
        return -1 if delta > 0 else (1 if delta < 0 else 0)
    if rule == "delta_neg_up":
        return 1 if delta < 0 else (-1 if delta > 0 else 0)
    if rule == "level_below_50":
        return 1 if value < 50.0 else (-1 if value > 50.0 else 0)
    if rule == "level_below_0":
        return 1 if value < 0.0 else (-1 if value > 0.0 else 0)
    return 0


# ── Historical Evidence Builder ────────────────────────────────


class HistoricalEvidenceBuilder:
    """Converts FRED time series DataFrames into Evidence objects.

    Parameters
    ----------
    fred_data : dict[str, list[tuple[float, float]]]
        ``{fred_id: [(unix_ts, value), ...]}`` sorted by timestamp.
        Timestamps are observation dates converted to unix epoch.
    series_configs : list[FredSeriesConfig] | None
        Which FRED series to use.  Defaults to ``FRED_SERIES``.
    """

    def __init__(
        self,
        fred_data: dict[str, list[tuple[float, float]]],
        series_configs: list[FredSeriesConfig] | None = None,
    ) -> None:
        self._configs = {c.fred_id: c for c in (series_configs or FRED_SERIES)}
        # Pre-sort each series by timestamp.
        self._data: dict[str, list[tuple[float, float]]] = {}
        for fid, points in fred_data.items():
            if fid in self._configs:
                self._data[fid] = sorted(points, key=lambda p: p[0])

    def build_evidence(self, as_of: float) -> list[Evidence]:
        """Generate Evidence from FRED data available up to *as_of*.

        Strict no-look-ahead: only observations with ``timestamp <= as_of``.
        Uses a 52-week lookback window (only recent data is relevant).
        """
        lookback_start = as_of - 365 * _DAY
        all_evidence: list[Evidence] = []

        for fred_id, points in self._data.items():
            cfg = self._configs[fred_id]

            # Filter to [lookback_start, as_of].
            relevant = [(ts, v) for ts, v in points if lookback_start <= ts <= as_of]
            if len(relevant) < 2:
                continue

            all_evidence.extend(self._build_series_evidence(fred_id, cfg, relevant))

        return all_evidence

    def build_all_evidence(self) -> list[Evidence]:
        """Pre-build Evidence for ALL observations across the full date range.

        Returns evidence sorted by timestamp.  Callers slice by ``as_of``
        using :func:`bisect.bisect_right` on timestamps for no-look-ahead.

        Uses vectorized cumulative statistics — O(n) per series instead of
        the O(n²) per-step approach.
        """
        all_evidence: list[Evidence] = []

        for fred_id, points in self._data.items():
            if len(points) < 2:
                continue
            cfg = self._configs[fred_id]
            all_evidence.extend(self._build_series_evidence(fred_id, cfg, points))

        all_evidence.sort(key=lambda e: e.timestamp)
        return all_evidence

    @staticmethod
    def _build_series_evidence(
        fred_id: str,
        cfg: FredSeriesConfig,
        points: list[tuple[float, float]],
    ) -> list[Evidence]:
        """Build Evidence list for one FRED series with vectorized z-scores.

        Replaces the O(k²) inner loop with O(k) cumulative statistics.
        """
        n = len(points)
        if n < 2:
            return []

        values = np.array([v for _, v in points], dtype=np.float64)

        # Cumulative running mean and std for confidence z-score.
        # At index i, we want mean/std of values[0..i-1] (exclusive).
        cumsum = np.cumsum(values)
        cumsum2 = np.cumsum(values**2)

        source = f"fred.{fred_id.lower()}"
        tags = (f"fred:{fred_id}",)
        evidence: list[Evidence] = []
        prev_value: float | None = None

        for i, (ts, value) in enumerate(points):
            direction = _apply_direction_rule(cfg.direction_rule, value, prev_value)

            # Confidence from z-score against prior observations.
            if i >= 3:
                # mean and std of values[0..i-1]
                count = i  # number of prior values
                mean_v = cumsum[i - 1] / count
                var_v = cumsum2[i - 1] / count - mean_v**2
                # Bessel correction: sample variance = var * n / (n-1)
                if count > 1:
                    var_v = var_v * count / (count - 1)
                std_v = np.sqrt(max(var_v, 0.0))
                if std_v > 0:
                    z = abs((value - mean_v) / std_v)
                    confidence = min(z / 4.0, 1.0)
                else:
                    confidence = 0.3
            else:
                confidence = 0.3

            evidence.append(
                Evidence(
                    source=source,
                    signal_id=cfg.signal_id,
                    timestamp=ts,
                    value=value,
                    direction=direction,
                    confidence=confidence,
                    category=cfg.category,
                    tags=tags,
                    ttl=90 * _DAY,
                )
            )
            prev_value = value

        return evidence

    def build_registry(self) -> SignalRegistry:
        """Create a SignalRegistry for the FRED-backed signals."""
        registry = SignalRegistry()
        for fred_id, cfg in self._configs.items():
            if fred_id in self._data and self._data[fred_id]:
                meta = SignalMeta(
                    signal_id=cfg.signal_id,
                    source=f"fred.{fred_id.lower()}",
                    category=cfg.category,
                    frequency=cfg.frequency,
                    direction_semantics=f"FRED {fred_id} via {cfg.direction_rule}",
                )
                registry.register(meta)
        return registry


# ── FRED Data Fetcher ──────────────────────────────────────────


def _parse_fred_response(
    fred_id: str,
    raw_data: list[dict[str, str]],
) -> list[tuple[float, float]]:
    """Parse FRED API response to ``[(unix_ts, value)]``.

    Skips entries with missing/invalid values (FRED uses '.' for NA).
    """
    points: list[tuple[float, float]] = []
    for row in raw_data:
        date_str = row.get("date", "")
        val_str = row.get("value", "")
        if not date_str or val_str in ("", "."):
            continue
        try:
            val = float(val_str)
        except (ValueError, TypeError):
            continue
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        points.append((dt.timestamp(), val))
    return sorted(points, key=lambda p: p[0])


def fetch_fred_history(
    series_ids: list[str] | None = None,
    start_date: str = "2005-01-01",
    end_date: str | None = None,
    *,
    cache_dir: str | Path = ".tirra_cache/fred",
    force_refresh: bool = False,
) -> dict[str, list[tuple[float, float]]]:
    """Fetch FRED history for given series, with local file caching.

    Uses the ``MacroDataTool`` when available and FRED_API_KEY is set.
    Falls back to cached data if the API is unreachable.

    Parameters
    ----------
    series_ids : list[str] | None
        FRED codes.  Defaults to all 13 in ``FRED_SERIES``.
    start_date, end_date : str
        Date range in YYYY-MM-DD.  end_date defaults to today.
    cache_dir : path
        Local cache directory for raw JSON responses.
    force_refresh : bool
        If True, bypass cache and re-fetch.

    Returns
    -------
    dict mapping fred_id → [(unix_ts, value)].
    """
    ids = series_ids or [c.fred_id for c in FRED_SERIES]
    end_date = end_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    result: dict[str, list[tuple[float, float]]] = {}

    for fred_id in ids:
        cache_file = cache_path / f"{fred_id}_{start_date}_{end_date}.json"

        # Try cache first.
        if cache_file.exists() and not force_refresh:
            try:
                raw = json.loads(cache_file.read_text())
                parsed = _parse_fred_response(fred_id, raw)
                if parsed:
                    result[fred_id] = parsed
                    log.debug("Cache hit for %s (%d points)", fred_id, len(parsed))
                    continue
            except (json.JSONDecodeError, KeyError):
                pass  # Corrupt cache, re-fetch.

        # Fetch via MacroDataTool.
        try:
            from agent.tools.macro_data import MacroDataTool

            api_key = _get_fred_api_key()
            if _is_placeholder_api_key(api_key):
                log.warning(
                    "FRED API key not set; using cached data only for %s",
                    fred_id,
                )
                continue

            tool = MacroDataTool(fred_api_key=api_key)
            tool_result = tool.execute(
                series_id=fred_id,
                source="fred",
                start_date=start_date,
                end_date=end_date,
            )
            if tool_result.success and fred_id in tool_result.data:
                raw_data = tool_result.data[fred_id]
                # Cache the raw response.
                cache_file.write_text(json.dumps(raw_data))
                parsed = _parse_fred_response(fred_id, raw_data)
                result[fred_id] = parsed
                log.info("Fetched %s: %d points", fred_id, len(parsed))
            else:
                log.warning("FRED fetch failed for %s: %s", fred_id, tool_result.output)
        except Exception:
            log.exception("Failed to fetch %s from FRED", fred_id)

    return result


# ── Pre-compute convergence scores ─────────────────────────────


@dataclass
class StepScore:
    """Convergence score at a single timestep."""

    timestamp: float
    score: float = 0.0
    direction: int = 0
    n_cliques: int = 0
    best_template: str = ""
    template_match: float = 0.0


def precompute_convergence_scores(
    fred_data: dict[str, list[tuple[float, float]]],
    timestamps: np.ndarray,
    config: ConvergenceDetectorConfig | None = None,
    series_configs: list[FredSeriesConfig] | None = None,
    step_score_cache: dict[float, StepScore] | None = None,
) -> list[StepScore]:
    """Run the ConvergenceDetector at every timestamp to produce score array.

    Optimised path:
    1. Pre-build ALL evidence once (vectorized z-scores).
    2. At each step, bisect-slice evidence for the lookback window.
    3. Reuse a single detector instance (preserves persistence history).
    4. No per-step PipelineStore — evidence passed directly.

    Parameters
    ----------
    fred_data : dict of FRED series data (from fetch_fred_history).
    timestamps : 1-D array of unix epoch timestamps (weekly).
    config : detector config overrides.
    series_configs : FRED→Evidence mapping overrides.
    step_score_cache : shared cache keyed by timestamp.

    Returns
    -------
    list[StepScore] aligned with *timestamps*.
    """
    builder = HistoricalEvidenceBuilder(fred_data, series_configs)
    registry = builder.build_registry()
    cfg = config or ConvergenceDetectorConfig(
        min_clique_size=2,
        min_categories=2,
        min_persistence=1,
        lookback_days=365,
        corr_window=12,
        baseline_window=52,
    )

    # ── Pre-build all evidence once ────────────────────────────
    all_evidence = builder.build_all_evidence()
    all_ev_timestamps = [ev.timestamp for ev in all_evidence]

    # ── Single detector + dummy store ──────────────────────────
    dummy_store = PipelineStore(db_path=":memory:")
    detector = ConvergenceDetector(dummy_store, registry, cfg)

    scores: list[StepScore] = []
    n = len(timestamps)
    lookback_seconds = cfg.lookback_days * _DAY

    for i, ts in enumerate(timestamps):
        if step_score_cache is not None and ts in step_score_cache:
            scores.append(step_score_cache[ts])
            continue

        if (i + 1) % 50 == 0:
            log.info("Convergence scoring step %d/%d", i + 1, n)

        # Bisect-slice: evidence with timestamp in [ts - lookback, ts].
        lookback_start = ts - lookback_seconds
        lo = bisect.bisect_left(all_ev_timestamps, lookback_start)
        hi = bisect.bisect_right(all_ev_timestamps, ts)
        evidence = all_evidence[lo:hi]

        if len(evidence) < 4:
            score = StepScore(timestamp=ts)
            if step_score_cache is not None:
                step_score_cache[ts] = score
            scores.append(score)
            continue

        # Inject evidence directly — skip store round-trip.
        _ev_snapshot = evidence
        detector._load_evidence = lambda _since, _until, _ev=_ev_snapshot: _ev  # type: ignore[assignment]
        detections = detector.detect(as_of=ts)

        if detections:
            best = max(detections, key=lambda d: d.boosted_score)
            score = StepScore(
                timestamp=ts,
                score=best.boosted_score,
                direction=best.direction,
                n_cliques=len(detections),
                best_template=best.event_type,
                template_match=best.template_match,
            )
        else:
            score = StepScore(timestamp=ts)

        if step_score_cache is not None:
            step_score_cache[ts] = score
        scores.append(score)

    dummy_store.close()
    return scores


# ── Strategy implementations ───────────────────────────────────


class ConvergenceRiskOffStrategy(Strategy):
    """Binary risk-off: exit when stress convergence detected.

    Reads pre-computed convergence scores from ``test_extra``:
    - ``conv_score``: np.ndarray of boosted_score values
    - ``conv_direction``: np.ndarray of direction values (+1/-1/0)

    Weight = 0 (cash) when score > threshold and direction = +1 (stress).
    Weight = 1 (full exposure) otherwise.
    """

    def __init__(self, score_threshold: float = 0.3) -> None:
        self._threshold = score_threshold

    @property
    def name(self) -> str:
        return "convergence_risk_off"

    def generate_weights(
        self,
        train_returns: np.ndarray,
        test_length: int,
        *,
        train_extra: dict[str, Any] | None = None,
        test_extra: dict[str, Any] | None = None,
    ) -> np.ndarray:
        weights = np.ones(test_length)
        if test_extra is None:
            return weights

        scores = test_extra.get("conv_score")
        directions = test_extra.get("conv_direction")
        if scores is None or directions is None:
            return weights

        for i in range(test_length):
            if i < len(scores) and scores[i] > self._threshold and directions[i] >= 1:
                weights[i] = 0.0

        return weights


class ConvergenceDirectionalStrategy(Strategy):
    """Proportional positioning: weight = 1 - score × direction_sign.

    When stress detected (direction=+1), reduce exposure proportionally.
    When relief detected (direction=-1), maintain full exposure.
    """

    def __init__(self, scale: float = 1.0) -> None:
        self._scale = scale

    @property
    def name(self) -> str:
        return "convergence_directional"

    def generate_weights(
        self,
        train_returns: np.ndarray,
        test_length: int,
        *,
        train_extra: dict[str, Any] | None = None,
        test_extra: dict[str, Any] | None = None,
    ) -> np.ndarray:
        weights = np.ones(test_length)
        if test_extra is None:
            return weights

        scores = test_extra.get("conv_score")
        directions = test_extra.get("conv_direction")
        if scores is None or directions is None:
            return weights

        for i in range(test_length):
            if i >= len(scores):
                break
            # direction=+1 (stress) → reduce; direction=-1 (relief) → hold
            if directions[i] >= 1:
                adjustment = min(scores[i] * self._scale, 1.0)
                weights[i] = max(1.0 - adjustment, 0.0)

        return weights


class ConvergenceTemplateStrategy(Strategy):
    """Only act when template match confidence is high (>0.5)."""

    def __init__(
        self, match_threshold: float = 0.5, score_threshold: float = 0.3
    ) -> None:
        self._match_threshold = match_threshold
        self._score_threshold = score_threshold

    @property
    def name(self) -> str:
        return "convergence_template"

    def generate_weights(
        self,
        train_returns: np.ndarray,
        test_length: int,
        *,
        train_extra: dict[str, Any] | None = None,
        test_extra: dict[str, Any] | None = None,
    ) -> np.ndarray:
        weights = np.ones(test_length)
        if test_extra is None:
            return weights

        scores = test_extra.get("conv_score")
        directions = test_extra.get("conv_direction")
        matches = test_extra.get("conv_template_match")
        if scores is None or directions is None or matches is None:
            return weights

        for i in range(test_length):
            if i >= len(scores):
                break
            if (
                scores[i] > self._score_threshold
                and directions[i] >= 1
                and matches[i] > self._match_threshold
            ):
                weights[i] = 0.0

        return weights


# ── Macro Backtest Orchestrator ────────────────────────────────


@dataclass
class MacroBacktestResult:
    """Full results of a macro backtest run."""

    strategies: dict[str, BacktestResult]
    """Per-strategy WalkForward results."""

    convergence_scores: list[StepScore]
    """Pre-computed convergence scores at each step."""

    sharpe_diffs: dict[str, dict[str, Any]]
    """Bootstrap CI for Sharpe difference vs buy-and-hold."""

    n_detections: int
    """Total timesteps where convergence was detected."""

    detection_rate: float
    """Fraction of timesteps with non-zero convergence score."""


def _fetch_target_returns(
    ticker: str,
    start_date: str,
    end_date: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Fetch weekly log returns and timestamps for a ticker.

    Returns
    -------
    (timestamps, log_returns) — both 1-D arrays aligned weekly.
    """
    try:
        import pandas as pd
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "yfinance required for macro backtest: pip install yfinance"
        ) from e

    def _extract_close_series(frame: pd.DataFrame) -> pd.Series:
        if isinstance(frame.columns, pd.MultiIndex):
            top_level = set(frame.columns.get_level_values(0))
            price_col = "Adj Close" if "Adj Close" in top_level else "Close"
            series_or_frame = frame.xs(price_col, axis=1, level=0)
        else:
            price_col = "Adj Close" if "Adj Close" in frame.columns else "Close"
            if price_col not in frame.columns:
                raise ValueError(
                    f"No close column available for {ticker}; columns={list(frame.columns)}"
                )
            series_or_frame = frame[price_col]

        if isinstance(series_or_frame, pd.DataFrame):
            if ticker in series_or_frame.columns:
                series = series_or_frame[ticker]
            else:
                series = series_or_frame.iloc[:, 0]
        else:
            series = series_or_frame
        return series.dropna()

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1wk",
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for {ticker} ({start_date} to {end_date})")

    close = _extract_close_series(df)

    log_ret = np.log(close / close.shift(1)).dropna().values.astype(np.float64)

    # Convert index to unix timestamps.
    dates = close.index[1:]  # skip first NaN from diff
    timestamps = np.array(
        [d.timestamp() for d in dates.to_pydatetime()],
        dtype=np.float64,
    )

    return timestamps, log_ret


def run_macro_backtest(
    start_year: int = 2010,
    end_year: int = 2025,
    targets: list[str] | None = None,
    min_train: int = 52,
    test_size: int = 12,
    bootstrap_count: int = _DEFAULT_BOOTSTRAP_COUNT,
    fred_data: dict[str, list[tuple[float, float]]] | None = None,
    detector_config: ConvergenceDetectorConfig | None = None,
) -> dict[str, MacroBacktestResult]:
    """Run the full macro backtest pipeline.

    Parameters
    ----------
    start_year, end_year : int
        Calendar year range.
    targets : list of ticker symbols.
        Defaults to ['SPY', 'TLT', 'GLD'].
    min_train : int
        Minimum training weeks for WalkForward.
    test_size : int
        Test window size in weeks.
    fred_data : pre-fetched FRED data (skip fetch if provided).
    detector_config : override detector configuration.

    Returns
    -------
    dict mapping ticker → MacroBacktestResult.
    """
    targets = targets or list(_DEFAULT_TARGETS)
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    # Step 1: Fetch FRED history.
    if fred_data is None:
        log.info("Fetching FRED history...")
        fred_data = fetch_fred_history(start_date=start_date, end_date=end_date)

    if not fred_data:
        raise ValueError(
            "No FRED data available. Set FRED_API_KEY or provide fred_data."
        )

    results: dict[str, MacroBacktestResult] = {}
    shared_step_score_cache: dict[float, StepScore] = {}

    for ticker in targets:
        log.info("=== Backtesting %s ===", ticker)

        # Step 2: Fetch target returns.
        try:
            timestamps, log_returns = _fetch_target_returns(
                ticker, start_date, end_date
            )
        except (ImportError, ValueError) as e:
            log.error("Cannot fetch %s: %s", ticker, e)
            continue

        n_weeks = len(log_returns)
        log.info("%s: %d weekly observations", ticker, n_weeks)

        # Step 3: Pre-compute convergence scores at each weekly step.
        log.info("Pre-computing convergence scores for %s...", ticker)
        step_scores = precompute_convergence_scores(
            fred_data,
            timestamps,
            config=detector_config,
            step_score_cache=shared_step_score_cache,
        )

        conv_score_arr = np.array([s.score for s in step_scores])
        conv_dir_arr = np.array([s.direction for s in step_scores])
        conv_tmatch_arr = np.array([s.template_match for s in step_scores])

        n_detected = int(np.sum(conv_score_arr > 0))
        detection_rate = n_detected / max(n_weeks, 1)
        log.info(
            "Detection rate: %d/%d (%.1f%%)", n_detected, n_weeks, detection_rate * 100
        )

        # Step 4: Build extra arrays for WalkForward.
        extra = {
            "conv_score": conv_score_arr,
            "conv_direction": conv_dir_arr,
            "conv_template_match": conv_tmatch_arr,
        }

        # Step 5: Run WalkForward for each strategy.
        wf = WalkForward(
            min_train=min_train,
            test_size=test_size,
            periods_per_year=52,
        )

        strategies: list[Strategy] = [
            ConvergenceRiskOffStrategy(score_threshold=0.3),
            ConvergenceDirectionalStrategy(scale=1.0),
            ConvergenceTemplateStrategy(match_threshold=0.5),
        ]

        bt_results: dict[str, BacktestResult] = {}
        for strat in strategies:
            log.info("Running WalkForward: %s on %s", strat.name, ticker)
            bt = wf.run(strat, log_returns, extra=extra)
            bt_results[strat.name] = bt
            sharpe = bt.aggregate_metrics.get("sharpe", float("nan"))
            log.info("  %s Sharpe: %.3f", strat.name, sharpe)

        # Step 6: Buy-and-hold benchmark.
        from agent.quant.backtest import BuyAndHoldStrategy

        bh_result = wf.run(BuyAndHoldStrategy(), log_returns, extra=extra)
        bt_results["buy_and_hold"] = bh_result
        bh_sharpe = bh_result.aggregate_metrics.get("sharpe", float("nan"))
        log.info("  Buy & Hold Sharpe: %.3f", bh_sharpe)

        # Step 7: Bootstrap CI for Sharpe difference.
        sharpe_diffs: dict[str, dict[str, Any]] = {}
        bh_sharpe_val = bh_result.aggregate_metrics.get("sharpe", 0.0)
        for sname, sresult in bt_results.items():
            if sname == "buy_and_hold":
                continue
            try:
                strat_sharpe_val = sresult.aggregate_metrics.get("sharpe", 0.0)
                sharpe_diff_point = strat_sharpe_val - bh_sharpe_val

                # Bootstrap the strategy Sharpe to get a CI.
                from agent.quant.scoring import sharpe_ratio

                _sharpe_fn = lambda r: sharpe_ratio(
                    r, periods_per_year=52
                )  # noqa: E731
                point_est, ci_lo, ci_hi = block_bootstrap_ci(
                    sresult.all_test_returns,
                    _sharpe_fn,
                    n_bootstrap=bootstrap_count,
                    block_length=4,
                )
                sharpe_diffs[sname] = {
                    "sharpe_diff": float(sharpe_diff_point),
                    "strategy_sharpe": float(strat_sharpe_val),
                    "benchmark_sharpe": float(bh_sharpe_val),
                    "strategy_sharpe_ci_lo": float(ci_lo),
                    "strategy_sharpe_ci_hi": float(ci_hi),
                }
                log.info(
                    "  %s: Sharpe=%.3f [%.3f, %.3f] (diff vs B&H: %.3f)",
                    sname,
                    point_est,
                    ci_lo,
                    ci_hi,
                    sharpe_diff_point,
                )
            except Exception:
                log.exception("Bootstrap failed for %s", sname)
                sharpe_diffs[sname] = {}

        results[ticker] = MacroBacktestResult(
            strategies=bt_results,
            convergence_scores=step_scores,
            sharpe_diffs=sharpe_diffs,
            n_detections=n_detected,
            detection_rate=detection_rate,
        )

    return results


def _resolve_macro_runtime(
    *,
    start_year: int,
    end_year: int,
    targets: list[str],
    bootstrap_count: int | None,
    fast_mode: bool,
) -> tuple[int, int, list[str], int]:
    """Resolve CLI runtime knobs for macro backtests.

    Fast mode is a reduced-cost preset for development runs. It only shrinks the
    default date range and default target set; explicit user overrides are kept.
    """
    resolved_start_year = start_year
    resolved_end_year = end_year
    resolved_targets = list(targets)
    resolved_bootstrap = (
        bootstrap_count if bootstrap_count is not None else _DEFAULT_BOOTSTRAP_COUNT
    )

    if not fast_mode:
        return (
            resolved_start_year,
            resolved_end_year,
            resolved_targets,
            resolved_bootstrap,
        )

    if bootstrap_count is None:
        resolved_bootstrap = _FAST_BOOTSTRAP_COUNT
    if resolved_targets == _DEFAULT_TARGETS:
        resolved_targets = ["SPY"]
    if start_year == 2010 and end_year == 2025:
        resolved_start_year = _FAST_START_YEAR

    return (
        resolved_start_year,
        resolved_end_year,
        resolved_targets,
        resolved_bootstrap,
    )


# ── Baseline Persistence ───────────────────────────────────────


def save_baseline(
    results: dict[str, MacroBacktestResult],
    path: str | Path = "docs/baselines/convergence_backtest_baseline.json",
) -> None:
    """Persist baseline metrics for phase-gate comparison."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    baseline: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    for ticker, res in results.items():
        ticker_data: dict[str, Any] = {
            "n_detections": res.n_detections,
            "detection_rate": res.detection_rate,
        }
        for sname, bt in res.strategies.items():
            ticker_data[sname] = {
                k: float(v) if isinstance(v, (float, np.floating)) else v
                for k, v in bt.aggregate_metrics.items()
                if not isinstance(v, np.ndarray)
            }
        if res.sharpe_diffs:
            ticker_data["sharpe_diffs"] = {
                k: {
                    kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                    for kk, vv in v.items()
                }
                for k, v in res.sharpe_diffs.items()
            }
        baseline[ticker] = ticker_data

    Path(path).write_text(json.dumps(baseline, indent=2, default=str))
    log.info("Baseline saved to %s", path)


def validate_against_baseline(
    results: dict[str, MacroBacktestResult],
    baseline_path: str | Path = "docs/baselines/convergence_backtest_baseline.json",
    max_degradation: float = 0.10,
) -> tuple[bool, list[str]]:
    """Compare current results against saved baseline.

    Returns (passed, list_of_failures).
    A metric fails if it degrades by more than *max_degradation* relative.
    """
    bp = Path(baseline_path)
    if not bp.exists():
        return True, ["No baseline found — first run, auto-pass."]

    baseline = json.loads(bp.read_text())
    failures: list[str] = []

    for ticker, res in results.items():
        if ticker not in baseline:
            continue
        bl = baseline[ticker]

        for sname, bt in res.strategies.items():
            if sname not in bl:
                continue
            for metric in ("sharpe", "sortino", "max_drawdown"):
                current = bt.aggregate_metrics.get(metric)
                bl_val = bl[sname].get(metric)
                if current is None or bl_val is None:
                    continue
                # For max_drawdown, higher (less negative) is better.
                if metric == "max_drawdown":
                    if (
                        bl_val != 0
                        and (current - bl_val) / abs(bl_val) > max_degradation
                    ):
                        failures.append(
                            f"{ticker}/{sname}/{metric}: "
                            f"{bl_val:.4f} → {current:.4f} "
                            f"(degraded > {max_degradation*100:.0f}%)"
                        )
                else:
                    if (
                        bl_val != 0
                        and (bl_val - current) / abs(bl_val) > max_degradation
                    ):
                        failures.append(
                            f"{ticker}/{sname}/{metric}: "
                            f"{bl_val:.4f} → {current:.4f} "
                            f"(degraded > {max_degradation*100:.0f}%)"
                        )

    passed = len(failures) == 0
    return passed, failures


# ── CLI Entry Point ────────────────────────────────────────────


def main() -> None:
    """CLI for convergence backtest operations."""
    import argparse

    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    if load_dotenv is not None:
        load_dotenv(Path.cwd() / ".env")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Convergence Engine Backtest")
    parser.add_argument(
        "--synthetic", action="store_true", help="Run synthetic validation only"
    )
    parser.add_argument("--macro", action="store_true", help="Run macro backtest only")
    parser.add_argument(
        "--validate", action="store_true", help="Compare against saved baseline"
    )
    parser.add_argument(
        "--save-baseline", action="store_true", help="Save results as new baseline"
    )
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--targets", nargs="+", default=list(_DEFAULT_TARGETS))
    parser.add_argument(
        "--bootstrap-count",
        type=int,
        default=None,
        help="Override bootstrap resample count for macro backtests.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run a reduced-cost macro preset for development iterations.",
    )
    args = parser.parse_args()

    if args.synthetic:
        from agent.convergence.synthetic import (
            generate_scenarios,
            run_synthetic_validation,
        )

        log.info("Running synthetic validation (100 scenarios)...")
        scenarios = generate_scenarios(n=100, seed=42)
        result = run_synthetic_validation(scenarios)
        print(f"\n{'='*60}")
        print(f"Synthetic Validation Results")
        print(f"{'='*60}")
        print(f"  Scenarios: {result.n_scenarios}")
        print(
            f"  TP={result.true_positives}  FN={result.false_negatives}  FP={result.false_positives}"
        )
        print(f"  Precision: {result.precision:.4f}")
        print(f"  Recall:    {result.recall:.4f}")
        print(f"  F1:        {result.f1:.4f}")
        print(f"  Template accuracy:  {result.template_accuracy:.4f}")
        print(f"  Direction accuracy: {result.direction_accuracy:.4f}")
        return

    if args.macro or args.validate or args.save_baseline:
        start_year, end_year, targets, bootstrap_count = _resolve_macro_runtime(
            start_year=args.start_year,
            end_year=args.end_year,
            targets=args.targets,
            bootstrap_count=args.bootstrap_count,
            fast_mode=args.fast,
        )
        results = run_macro_backtest(
            start_year=start_year,
            end_year=end_year,
            targets=targets,
            bootstrap_count=bootstrap_count,
        )

        for ticker, res in results.items():
            print(f"\n{'='*60}")
            print(f"  {ticker} — Detection rate: {res.detection_rate:.1%}")
            for sname, bt in res.strategies.items():
                m = bt.aggregate_metrics
                print(
                    f"  {sname}: Sharpe={m.get('sharpe', 'N/A'):.3f}  "
                    f"MDD={m.get('max_drawdown', 'N/A'):.3f}"
                )

        if args.save_baseline:
            save_baseline(results)

        if args.validate:
            passed, failures = validate_against_baseline(results)
            if passed:
                print("\n✓ PHASE GATE PASSED")
            else:
                print("\n✗ PHASE GATE FAILED")
                for f in failures:
                    print(f"  - {f}")
            raise SystemExit(0 if passed else 1)

        return

    parser.print_help()


if __name__ == "__main__":
    main()
