"""
Tool: FINRA Short Volume & Short Interest

Two data modes from FINRA's free public API:
  1. short_volume  — Reg SHO daily short volume ratio by ticker (daily, T+0/T+1)
  2. short_interest — Consolidated bi-monthly short interest (2-month lag)

Signals: short ratio deviation from baseline (anomaly), days-to-cover
squeeze risk, building/covering detection, cross-asset credit-equity divergence.

Data source: https://api.finra.org (zero cost, no API key, public data).
"""

from __future__ import annotations

import logging
import statistics
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

try:
    from agent.pipeline.entity import entity_id_from_key, normalize_company_name
except ImportError:  # pragma: no cover
    entity_id_from_key = None  # type: ignore[assignment]
    normalize_company_name = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

_USER_AGENT = "TirraMind/0.1 (research; https://github.com/tirramind)"
_BASE_URL = "https://api.finra.org/data/group/otcMarket/name"
_REG_SHO_ENDPOINT = f"{_BASE_URL}/regShoDaily"
_SHORT_INTEREST_ENDPOINT = f"{_BASE_URL}/consolidatedShortInterest"
_REQUEST_DELAY = 0.12  # seconds between requests
_MAX_PAGES = 6  # 6 × 5000 = 30,000 records max for scan mode
_PAGE_SIZE = 5000


class FinraShortVolumeTool(Tool):
    """FINRA Reg SHO short volume and consolidated short interest."""

    def __init__(
        self,
        cache: DataCache | None = None,
        *,
        pipeline_store: PipelineStore | None = None,
    ) -> None:
        self._cache = cache
        self._store = pipeline_store

    # ------------------------------------------------------------------
    # Entity persistence (L2)
    # ------------------------------------------------------------------

    def _persist_entities(self, records: list[dict[str, Any]]) -> None:
        """Register company entities and store short-interest observations."""
        if self._store is None or entity_id_from_key is None:
            return
        if not records:
            return
        try:
            self._persist_entities_inner(records)
        except Exception:
            log.exception("Entity persistence failed (non-fatal)")

    def _persist_entities_inner(self, records: list[dict[str, Any]]) -> None:
        assert self._store is not None  # noqa: S101
        store = self._store

        seen: set[str] = set()
        for rec in records:
            ticker = (rec.get("ticker") or rec.get("symbol") or "").strip().upper()
            if not ticker:
                continue

            eid = entity_id_from_key("company", ticker)

            if eid not in seen:
                seen.add(eid)
                store.register_entity(
                    entity_type="company",
                    canonical_name=ticker,
                    entity_id=eid,
                    metadata={"source": "finra", "ticker": ticker},
                )

            # Determine timestamp from date field or settlement_date
            date_str = rec.get("date") or rec.get("settlement_date") or ""
            try:
                ts = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC).timestamp()
            except (ValueError, TypeError):
                ts = datetime.now(tz=UTC).timestamp()

            # Build value dict from available fields
            value: dict[str, Any] = {}
            for key in (
                "short_ratio",
                "total_volume",
                "short_volume",
                "exempt_volume",
                "zscore",
                "trend",
                "is_anomaly",
                "days_to_cover",
                "current_short_position",
                "previous_short_position",
                "change_percent",
                "avg_daily_volume",
                "facility_count",
            ):
                if key in rec:
                    value[key] = rec[key]

            if not value:
                continue

            store.store_entity_observation(
                entity_id=eid,
                source_tool="finra_short_volume",
                observed_at=ts,
                observation_type="short_interest",
                depth_level=2,
                value=value,
            )

    @property
    def name(self) -> str:
        return "finra_short_volume"

    @property
    def description(self) -> str:
        return (
            "Fetch FINRA short selling data. Two modes: "
            "'short_volume' for daily Reg SHO short volume ratio by ticker "
            "(near-real-time, covers off-exchange/dark pool volume), or "
            "'short_interest' for bi-monthly accumulated short positions "
            "(~2-month lag, shows days-to-cover and position changes). "
            "Can scan for top short-ratio anomalies across all tickers."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["short_volume", "short_interest"],
                    "description": "Data mode: 'short_volume' for daily Reg SHO ratios, 'short_interest' for bi-monthly positions",
                },
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker (e.g. AAPL). Omit for scan mode in short_volume.",
                },
                "date": {
                    "type": "string",
                    "description": "Date YYYY-MM-DD. Default: most recent trading day.",
                },
                "days_back": {
                    "type": "integer",
                    "description": "Number of trading days to fetch for trend analysis (1-20, default 5). Only for short_volume mode.",
                },
                "min_total_volume": {
                    "type": "integer",
                    "description": "Minimum total volume to include in scan results (default 100000). Filters out illiquid noise.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return in scan mode (default 20).",
                },
            },
            "required": ["mode"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        mode = kwargs.get("mode", "")
        if mode not in ("short_volume", "short_interest"):
            return ToolResult(False, f"Invalid mode '{mode}'. Use 'short_volume' or 'short_interest'.")

        ticker = (kwargs.get("ticker") or "").strip().upper() or None
        date_str = kwargs.get("date") or ""
        days_back = max(1, min(20, int(kwargs.get("days_back", 5))))
        min_vol = max(0, int(kwargs.get("min_total_volume", 100_000)))
        limit = max(1, min(100, int(kwargs.get("limit", 20))))

        # Parse or default the date
        target_date = self._parse_date(date_str)
        if target_date is None:
            return ToolResult(False, f"Invalid date format: '{date_str}'. Use YYYY-MM-DD.")

        try:
            if mode == "short_volume":
                return self._execute_short_volume(ticker, target_date, days_back, min_vol, limit)
            else:
                return self._execute_short_interest(ticker, target_date)
        except httpx.TimeoutException:
            return ToolResult(False, "FINRA API request timed out. Try again later.")
        except Exception as exc:
            log.exception("FINRA tool error")
            return ToolResult(False, f"Error: {exc}")

    # ------------------------------------------------------------------
    # Short Volume Mode
    # ------------------------------------------------------------------

    def _execute_short_volume(
        self,
        ticker: str | None,
        target_date: datetime,
        days_back: int,
        min_vol: int,
        limit: int,
    ) -> ToolResult:
        """Fetch daily short volume for one ticker or scan all."""
        if ticker:
            return self._short_volume_ticker(ticker, target_date, days_back)
        else:
            return self._short_volume_scan(target_date, min_vol, limit)

    def _short_volume_ticker(
        self,
        ticker: str,
        target_date: datetime,
        days_back: int,
    ) -> ToolResult:
        """Multi-day short volume for a single ticker."""
        dates = self._trading_dates(target_date, days_back)
        daily_data: list[dict] = []

        for d in dates:
            date_str = d.strftime("%Y-%m-%d")
            records = self._fetch_reg_sho(date_str, ticker=ticker)
            if not records:
                continue
            agg = self._aggregate_facilities(records)
            if ticker in agg:
                entry = agg[ticker]
                entry["date"] = date_str
                daily_data.append(entry)

        if not daily_data:
            return ToolResult(
                True,
                f"No Reg SHO data found for {ticker} in the last {days_back} trading days from {target_date.strftime('%Y-%m-%d')}.",
                data={"ticker": ticker, "records": []},
            )

        # Compute signals
        signals = self._compute_volume_signals(daily_data)

        # Format output
        lines = [f"## FINRA Short Volume: {ticker}", ""]
        for entry in daily_data:
            flag = ""
            if entry.get("is_anomaly"):
                flag = " ⚠ ANOMALY" if entry["short_ratio"] > entry.get("avg_ratio", 0.5) else " ↓ COVERING"
            lines.append(
                f"  {entry['date']}  total={entry['total_volume']:>14,.0f}  "
                f"short={entry['short_volume']:>14,.0f}  "
                f"ratio={entry['short_ratio']:.1%}  "
                f"exempt={entry.get('exempt_volume', 0):>8,.0f}{flag}"
            )

        lines.append("")
        if signals:
            lines.append(f"Short ratio (latest): {signals['latest_ratio']:.1%}")
            if signals.get("avg_ratio") is not None:
                lines.append(f"Short ratio ({len(daily_data)}-day avg): {signals['avg_ratio']:.1%}")
            if signals.get("zscore") is not None:
                lines.append(f"Z-score: {signals['zscore']:+.2f}")
            lines.append(f"Trend: {signals.get('trend', 'n/a')}")
            if signals.get("is_anomaly"):
                lines.append(f"⚠ ANOMALY DETECTED: short ratio z-score {signals['zscore']:+.2f}")

        # Persist entities (L2)
        for entry in daily_data:
            entry.setdefault("ticker", ticker)
        self._persist_entities(daily_data)

        return ToolResult(
            True,
            "\n".join(lines),
            data={
                "ticker": ticker,
                "records": daily_data,
                "signals": signals,
            },
        )

    def _short_volume_scan(
        self,
        target_date: datetime,
        min_vol: int,
        limit: int,
    ) -> ToolResult:
        """Scan all tickers for a single day, rank by short ratio."""
        date_str = target_date.strftime("%Y-%m-%d")
        all_records: list[dict] = []

        # Paginate to get all records
        for page in range(_MAX_PAGES):
            records = self._fetch_reg_sho(date_str, ticker=None, offset=page * _PAGE_SIZE)
            if not records:
                break
            all_records.extend(records)
            if len(records) < _PAGE_SIZE:
                break
            import time

            time.sleep(_REQUEST_DELAY)

        if not all_records:
            return ToolResult(
                True,
                f"No Reg SHO data found for {date_str}. May be a weekend/holiday.",
                data={"date": date_str, "results": []},
            )

        # Aggregate across facilities
        agg = self._aggregate_facilities(all_records)

        # Filter by min volume and compute ratio
        filtered = []
        for sym, v in agg.items():
            if v["total_volume"] >= min_vol and v["total_volume"] > 0:
                filtered.append({"ticker": sym, **v})

        # Sort by short ratio descending
        filtered.sort(key=lambda x: x["short_ratio"], reverse=True)
        top = filtered[:limit]

        lines = [f"## FINRA Short Volume Scan: {date_str}", ""]
        lines.append(f"Total tickers (min vol {min_vol:,}): {len(filtered)}")
        lines.append(f"Total records fetched: {len(all_records)}")
        lines.append("")

        for i, entry in enumerate(top, 1):
            lines.append(
                f"  {i:>3}. {entry['ticker']:8s}  "
                f"total={entry['total_volume']:>14,.0f}  "
                f"short={entry['short_volume']:>14,.0f}  "
                f"ratio={entry['short_ratio']:.1%}  "
                f"exempt={entry.get('exempt_volume', 0):>8,.0f}"
            )

        # Persist entities (L2)
        self._persist_entities(top)

        return ToolResult(
            True,
            "\n".join(lines),
            data={
                "date": date_str,
                "total_tickers": len(filtered),
                "results": top,
            },
        )

    # ------------------------------------------------------------------
    # Short Interest Mode
    # ------------------------------------------------------------------

    def _execute_short_interest(
        self,
        ticker: str | None,
        target_date: datetime,
    ) -> ToolResult:
        if not ticker:
            return ToolResult(False, "short_interest mode requires a ticker parameter.")

        # Search backwards for the most recent settlement date
        records = self._fetch_short_interest_recent(ticker, target_date)

        if not records:
            return ToolResult(
                True,
                f"No short interest data found for {ticker}. Data is bi-monthly with ~2 month lag.",
                data={"ticker": ticker, "records": []},
            )

        # Compute signals
        latest = records[0]
        signals: dict[str, Any] = {
            "current_short_position": latest.get("currentShortPositionQuantity", 0),
            "previous_short_position": latest.get("previousShortPositionQuantity", 0),
            "change_percent": latest.get("changePercent", 0),
            "days_to_cover": latest.get("daysToCoverQuantity", 0),
            "avg_daily_volume": latest.get("averageDailyVolumeQuantity", 0),
            "settlement_date": latest.get("settlementDate", ""),
        }
        signals["squeeze_risk"] = (signals["days_to_cover"] or 0) > 5.0
        signals["building_short"] = (signals["change_percent"] or 0) > 15.0
        signals["covering"] = (signals["change_percent"] or 0) < -15.0

        # Format output
        lines = [f"## FINRA Short Interest: {ticker}", ""]

        for rec in records:
            dtc = rec.get("daysToCoverQuantity", 0) or 0
            chg = rec.get("changePercent", 0) or 0
            si = rec.get("currentShortPositionQuantity", 0) or 0
            vol = rec.get("averageDailyVolumeQuantity", 0) or 0
            flag = ""
            if dtc > 5.0:
                flag = " ⚠ SQUEEZE RISK"
            elif chg > 15.0:
                flag = " ↑ BUILDING"
            elif chg < -15.0:
                flag = " ↓ COVERING"
            lines.append(
                f"  {rec.get('settlementDate', '?')}  "
                f"SI={si:>14,}  chg={chg:>+6.1f}%  "
                f"DTC={dtc:.1f}  ADV={vol:>12,}{flag}"
            )

        lines.append("")
        if signals["squeeze_risk"]:
            lines.append(f"⚠ SQUEEZE RISK: days-to-cover = {signals['days_to_cover']:.1f}")
        if signals["building_short"]:
            lines.append(f"↑ SHORT BUILDING: +{signals['change_percent']:.1f}% from previous period")
        if signals["covering"]:
            lines.append(f"↓ SHORT COVERING: {signals['change_percent']:.1f}% from previous period")

        # Persist entities (L2)
        si_records = [self._si_record_to_dict(r) for r in records]
        self._persist_entities(si_records)

        return ToolResult(
            True,
            "\n".join(lines),
            data={
                "ticker": ticker,
                "records": si_records,
                "signals": signals,
            },
        )

    # ------------------------------------------------------------------
    # API Fetching
    # ------------------------------------------------------------------

    def _fetch_reg_sho(
        self,
        date: str,
        ticker: str | None = None,
        offset: int = 0,
    ) -> list[dict]:
        """Fetch Reg SHO daily short volume from FINRA API."""
        cache_params = {"date": date, "ticker": ticker or "ALL", "offset": offset}
        if self._cache:
            cached = self._cache.get("finra_regsho", cache_params)
            if cached is not None:
                return cached

        filters = [
            {
                "fieldName": "tradeReportDate",
                "fieldValue": date,
                "compareType": "EQUAL",
            },
        ]
        if ticker:
            filters.append(
                {
                    "fieldName": "securitiesInformationProcessorSymbolIdentifier",
                    "fieldValue": ticker,
                    "compareType": "EQUAL",
                }
            )

        body: dict[str, Any] = {
            "fields": [
                "securitiesInformationProcessorSymbolIdentifier",
                "totalParQuantity",
                "shortParQuantity",
                "shortExemptParQuantity",
                "reportingFacilityCode",
            ],
            "compareFilters": filters,
            "limit": _PAGE_SIZE,
            "offset": offset,
        }

        r = self._api_post(_REG_SHO_ENDPOINT, body)
        if r is None:
            return []

        if self._cache and r:
            self._cache.put("finra_regsho", cache_params, r)
        return r

    def _fetch_short_interest(self, ticker: str, settlement_date: str) -> list[dict]:
        """Fetch consolidated short interest for a ticker on a settlement date."""
        cache_params = {"ticker": ticker, "settlement_date": settlement_date}
        if self._cache:
            cached = self._cache.get("finra_si", cache_params)
            if cached is not None:
                return cached

        body: dict[str, Any] = {
            "fields": [
                "symbolCode",
                "settlementDate",
                "currentShortPositionQuantity",
                "previousShortPositionQuantity",
                "changePercent",
                "daysToCoverQuantity",
                "averageDailyVolumeQuantity",
                "marketClassCode",
                "issueName",
            ],
            "compareFilters": [
                {
                    "fieldName": "symbolCode",
                    "fieldValue": ticker,
                    "compareType": "EQUAL",
                },
                {
                    "fieldName": "settlementDate",
                    "fieldValue": settlement_date,
                    "compareType": "EQUAL",
                },
            ],
            "limit": 5,
        }

        r = self._api_post(_SHORT_INTEREST_ENDPOINT, body)
        if r is None:
            return []

        if self._cache and r:
            self._cache.put("finra_si", cache_params, r)
        return r

    def _fetch_short_interest_recent(
        self,
        ticker: str,
        target_date: datetime,
        max_lookback: int = 90,
    ) -> list[dict]:
        """Search backwards for the most recent settlement dates with data."""
        # Standard FINRA SI settlement dates: 15th and end of month
        candidates = self._si_settlement_dates(target_date, max_lookback)
        results: list[dict] = []

        for sd in candidates:
            records = self._fetch_short_interest(ticker, sd)
            if records:
                results.extend(records)
                if len(results) >= 4:  # get 2-4 most recent periods
                    break
            import time

            time.sleep(_REQUEST_DELAY)

        return results

    def _api_post(self, endpoint: str, body: dict) -> list[dict] | None:
        """POST to FINRA API, handle errors."""
        headers = {
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        try:
            r = httpx.post(endpoint, headers=headers, json=body, timeout=20)
        except httpx.TimeoutException:
            log.warning("FINRA API timeout: %s", endpoint)
            raise
        except httpx.HTTPError as exc:
            log.warning("FINRA HTTP error: %s", exc)
            return None

        if r.status_code == 204:
            return []
        if r.status_code == 429:
            log.warning("FINRA rate limited (429)")
            return None
        if r.status_code == 400:
            log.warning("FINRA bad request: %s", r.text[:200])
            return None
        if r.status_code >= 500:
            log.warning("FINRA server error %d: %s", r.status_code, r.text[:200])
            return None
        if r.status_code != 200:
            log.warning("FINRA unexpected status %d", r.status_code)
            return None

        try:
            data = r.json()
        except Exception:
            log.warning("FINRA response not JSON")
            return None

        if not isinstance(data, list):
            log.warning("FINRA response not a list")
            return None

        return data

    # ------------------------------------------------------------------
    # Aggregation & Signal Logic
    # ------------------------------------------------------------------

    def _aggregate_facilities(self, records: list[dict]) -> dict[str, dict]:
        """Aggregate Reg SHO records across reporting facilities into per-ticker totals."""
        by_ticker: dict[str, dict] = {}
        for rec in records:
            sym = rec.get("securitiesInformationProcessorSymbolIdentifier", "")
            if not sym:
                continue
            if sym not in by_ticker:
                by_ticker[sym] = {
                    "total_volume": 0.0,
                    "short_volume": 0.0,
                    "exempt_volume": 0.0,
                    "facility_count": 0,
                }
            by_ticker[sym]["total_volume"] += _safe_float(rec.get("totalParQuantity", 0))
            by_ticker[sym]["short_volume"] += _safe_float(rec.get("shortParQuantity", 0))
            by_ticker[sym]["exempt_volume"] += _safe_float(rec.get("shortExemptParQuantity", 0))
            by_ticker[sym]["facility_count"] += 1

        # Compute ratios
        for sym, v in by_ticker.items():
            v["short_ratio"] = v["short_volume"] / v["total_volume"] if v["total_volume"] > 0 else 0.0

        return by_ticker

    def _compute_volume_signals(self, daily_data: list[dict]) -> dict[str, Any]:
        """Compute trend and anomaly signals from multi-day short volume data."""
        if not daily_data:
            return {}

        ratios = [d["short_ratio"] for d in daily_data if "short_ratio" in d]
        latest = daily_data[0]
        signals: dict[str, Any] = {
            "latest_ratio": latest.get("short_ratio", 0),
            "latest_date": latest.get("date", ""),
        }

        if len(ratios) >= 2:
            avg = statistics.mean(ratios)
            signals["avg_ratio"] = avg

            if len(ratios) >= 3:
                stdev = statistics.stdev(ratios)
                if stdev > 0:
                    zscore = (ratios[0] - avg) / stdev
                    signals["zscore"] = round(zscore, 3)
                    signals["is_anomaly"] = abs(zscore) > 1.5

                    # Propagate anomaly flag to daily entries
                    for entry in daily_data:
                        r = entry.get("short_ratio", 0)
                        entry["avg_ratio"] = avg
                        if stdev > 0:
                            entry_z = (r - avg) / stdev
                            entry["is_anomaly"] = abs(entry_z) > 1.5
                        else:
                            entry["is_anomaly"] = False
                else:
                    signals["zscore"] = 0.0
                    signals["is_anomaly"] = False
            else:
                signals["zscore"] = None
                signals["is_anomaly"] = False

            # Trend: compare first half vs second half of ratios
            mid = len(ratios) // 2
            first_half = statistics.mean(ratios[mid:]) if ratios[mid:] else 0
            second_half = statistics.mean(ratios[:mid]) if ratios[:mid] else 0
            diff = second_half - first_half
            if abs(diff) < 0.02:
                signals["trend"] = "flat"
            elif diff > 0:
                signals["trend"] = "rising"
            else:
                signals["trend"] = "falling"
        else:
            signals["avg_ratio"] = None
            signals["zscore"] = None
            signals["is_anomaly"] = False
            signals["trend"] = "insufficient_data"

        return signals

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_date(date_str: str) -> datetime | None:
        """Parse YYYY-MM-DD or default to today."""
        if not date_str:
            return datetime.now()
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return None

    @staticmethod
    def _trading_dates(end: datetime, count: int) -> list[datetime]:
        """Generate `count` candidate trading dates going backwards from `end`.
        Skips weekends. Returns most recent first.
        """
        dates = []
        current = end
        while len(dates) < count:
            # weekday: 0=Mon, 5=Sat, 6=Sun
            if current.weekday() < 5:
                dates.append(current)
            current -= timedelta(days=1)
        return dates

    @staticmethod
    def _si_settlement_dates(target: datetime, lookback_days: int = 90) -> list[str]:
        """Generate candidate SI settlement dates (15th and end-of-month) going backwards."""
        candidates = []
        d = target
        end = target - timedelta(days=lookback_days)
        while d >= end:
            # End of month — use last day
            if d.month == 12:
                eom = datetime(d.year + 1, 1, 1) - timedelta(days=1)
            else:
                eom = datetime(d.year, d.month + 1, 1) - timedelta(days=1)
            if eom <= target:
                eom_str = eom.strftime("%Y-%m-%d")
                if eom_str not in candidates:
                    candidates.append(eom_str)

            # Mid-month (15th)
            mid = datetime(d.year, d.month, 15)
            if mid <= target:
                mid_str = mid.strftime("%Y-%m-%d")
                if mid_str not in candidates:
                    candidates.append(mid_str)

            # Go back one month
            if d.month == 1:
                d = datetime(d.year - 1, 12, 15)
            else:
                d = datetime(d.year, d.month - 1, 15)

        # Sort most recent first
        candidates.sort(reverse=True)
        return candidates

    @staticmethod
    def _si_record_to_dict(rec: dict) -> dict:
        """Normalize a short interest record to clean dict."""
        return {
            "symbol": rec.get("symbolCode", ""),
            "settlement_date": rec.get("settlementDate", ""),
            "current_short_position": rec.get("currentShortPositionQuantity", 0),
            "previous_short_position": rec.get("previousShortPositionQuantity", 0),
            "change_percent": rec.get("changePercent", 0),
            "days_to_cover": rec.get("daysToCoverQuantity", 0),
            "avg_daily_volume": rec.get("averageDailyVolumeQuantity", 0),
            "market_class": rec.get("marketClassCode", ""),
            "issue_name": rec.get("issueName", ""),
        }


def _safe_float(val: Any) -> float:
    """Convert to float safely (FINRA returns fractional par quantities)."""
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0
