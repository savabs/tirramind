"""
Tool: Energy Supply Monitor — EIA API v2

US petroleum stocks, supply & disposition (weekly), and rig counts
(monthly) from the Energy Information Administration.

Source:
  EIA Open Data API v2  — https://api.eia.gov/v2/
  Auth: DEMO_KEY works; TIRRA_EIA_API_KEY env var for higher limits.

Modes:
  petroleum_stocks — Weekly crude oil, gasoline, and distillate
                     ending stocks.  Inventory = physical supply buffer.
  rig_count        — Monthly rotary rigs in operation (EIA series).
                     Leading indicator for production 6-12 months out.
  petroleum_supply — Weekly supply & disposition summary including
                     crude stocks, production indicators.

Signal theory:
  - Crude stocks declining 3+ consecutive weeks = supply tightening
  - Gasoline stocks drop into summer = seasonal price spike risk
  - Rig count declining >10% over 3 months = future production drop
  - SPR stocks at historic lows = reduced emergency buffer
  - Week-over-week inventory change >5M barrels = surprise event

Market relevance:
  Petroleum inventory → crude/gasoline/heating oil prices, refinery
  margins, energy sector earnings, inflation expectations (energy CPI),
  central bank policy (oil price pass-through to headline inflation),
  geopolitical risk premium (supply disruption vulnerability).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

try:
    from agent.pipeline.entity import entity_id_from_key
except ImportError:  # pragma: no cover
    entity_id_from_key = None  # type: ignore[misc, assignment]

UTC = timezone.utc

log = logging.getLogger(__name__)

_UA = "TirraMind/0.1"
_TIMEOUT = 20
_CACHE_TTL = 7200  # 2 hours — weekly data updates on Wednesdays
_CACHE_TTL_RIG = 21600  # 6 hours — rig count is monthly

_EIA_BASE = "https://api.eia.gov/v2"

VALID_MODES = {"petroleum_stocks", "rig_count", "petroleum_supply"}

# ── Key EIA series codes ────────────────────────────────────────

STOCK_SERIES: dict[str, str] = {
    "crude_excl_spr": "WCESTUS1",
    "gasoline_total": "WGTSTUS1",
    "distillate": "WDISTUS1",
    "spr": "WPRSTUS1",
}

SUPPLY_SERIES: dict[str, str] = {
    "crude_excl_spr": "WCESTUS1",
    "crude_production": "WCRFPUS2",
    "crude_imports": "WCEIMUS2",
    "refinery_inputs": "WGIRIUS2",
}


def _get_api_key() -> str:
    """Get EIA API key from env or fall back to DEMO_KEY."""
    return os.environ.get("TIRRA_EIA_API_KEY", "DEMO_KEY")


class EnergySupplyTool(Tool):
    """Monitor US energy supply via EIA petroleum stocks and rig counts."""

    def __init__(
        self,
        cache: DataCache | None = None,
        *,
        pipeline_store: PipelineStore | None = None,
    ) -> None:
        self._cache = cache
        self._store = pipeline_store

    @property
    def name(self) -> str:
        return "energy_supply"

    @property
    def description(self) -> str:
        return (
            "Monitor US energy supply — weekly petroleum stocks (crude, gasoline, "
            "distillate, SPR), monthly rig counts, and petroleum supply & disposition "
            "from EIA API v2. Detects inventory surprises, supply tightening, and "
            "production leading indicators."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": sorted(VALID_MODES),
                    "description": (
                        "petroleum_stocks: weekly crude/gasoline/distillate/SPR stocks. "
                        "petroleum_supply: weekly supply & disposition indicators. "
                        "rig_count: monthly US rotary rig count."
                    ),
                },
                "series": {
                    "type": "string",
                    "description": (
                        "Filter to specific series. "
                        "For petroleum_stocks: 'crude_excl_spr', 'gasoline_total', "
                        "'distillate', 'spr'. "
                        "For petroleum_supply: 'crude_excl_spr', 'crude_production', "
                        "'crude_imports', 'refinery_inputs'. "
                        "Omit for all series in the mode."
                    ),
                },
                "weeks": {
                    "type": "integer",
                    "description": "Number of recent weeks of data (default: 12, max: 52). For rig_count this is months.",
                },
            },
            "required": ["mode"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        mode = kwargs.get("mode", "")
        if mode not in VALID_MODES:
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Must be one of: {sorted(VALID_MODES)}",
            )

        weeks = kwargs.get("weeks")
        days_back = kwargs.get("days_back")
        if weeks is None and days_back:
            weeks = max(1, min(int(days_back) // 7, 52))
        weeks = min(weeks or 12, 52)
        series_filter = (kwargs.get("series") or "").strip().lower()

        if mode == "petroleum_stocks":
            return self._handle_stocks(series_filter, weeks)
        elif mode == "rig_count":
            return self._handle_rig_count(weeks)
        else:
            return self._handle_supply(series_filter, weeks)

    # ── Mode handlers ───────────────────────────────────────

    def _handle_stocks(self, series_filter: str, weeks: int) -> ToolResult:
        if series_filter:
            if series_filter not in STOCK_SERIES:
                return ToolResult(
                    success=False,
                    output=(f"Invalid series '{series_filter}'. Must be one of: {sorted(STOCK_SERIES)}"),
                )
            series_map = {series_filter: STOCK_SERIES[series_filter]}
        else:
            series_map = STOCK_SERIES

        return self._fetch_petroleum_data(
            "petroleum/stoc/wstk/data/",
            series_map,
            weeks,
            "petroleum_stocks",
            _CACHE_TTL,
        )

    def _handle_supply(self, series_filter: str, weeks: int) -> ToolResult:
        if series_filter:
            if series_filter not in SUPPLY_SERIES:
                return ToolResult(
                    success=False,
                    output=(f"Invalid series '{series_filter}'. Must be one of: {sorted(SUPPLY_SERIES)}"),
                )
            series_map = {series_filter: SUPPLY_SERIES[series_filter]}
        else:
            series_map = SUPPLY_SERIES

        return self._fetch_petroleum_data(
            "petroleum/sum/sndw/data/",
            series_map,
            weeks,
            "petroleum_supply",
            _CACHE_TTL,
        )

    def _handle_rig_count(self, months: int) -> ToolResult:
        cache_key = f"energy:rig_count:{months}"
        if self._cache:
            hit = self._cache.get("energy_supply", {"key": cache_key})
            if hit is not None:
                return ToolResult(success=True, output=hit["output"], data=hit["data"])

        url = f"{_EIA_BASE}/natural-gas/enr/drill/data/"
        params = {
            "api_key": _get_api_key(),
            "frequency": "monthly",
            "data[0]": "value",
            "length": str(months),
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
        }

        records, err = _fetch_eia(url, params)
        if err:
            return ToolResult(success=False, output=err)

        parsed = _parse_eia_records(records)
        signals = _compute_rig_signals(parsed)
        summary = _format_rig_summary(parsed, signals, months)

        result_data = {
            "records": parsed,
            "count": len(parsed),
            "months": months,
            "signals": signals,
        }

        if self._cache:
            self._cache.put(
                "energy_supply",
                {"key": cache_key},
                {"output": summary, "data": result_data},
            )

        try:
            self._persist_rig_count(parsed)
        except Exception:
            log.exception("Energy supply rig_count persistence failed (non-fatal)")

        return ToolResult(success=True, output=summary, data=result_data)

    def _fetch_petroleum_data(
        self,
        endpoint: str,
        series_map: dict[str, str],
        weeks: int,
        label: str,
        cache_ttl: int,
    ) -> ToolResult:
        """Fetch petroleum data for one or more series."""
        all_records: dict[str, list[dict]] = {}
        all_signals: dict[str, dict] = {}

        for name, series_code in series_map.items():
            cache_key = f"energy:{label}:{name}:{weeks}"
            if self._cache:
                hit = self._cache.get("energy_supply", {"key": cache_key})
                if hit is not None:
                    all_records[name] = hit["data"]["records"]
                    all_signals[name] = hit["data"]["signals"]
                    continue

            url = f"{_EIA_BASE}/{endpoint}"
            params = {
                "api_key": _get_api_key(),
                "frequency": "weekly",
                "data[0]": "value",
                "facets[series][]": series_code,
                "length": str(weeks),
                "sort[0][column]": "period",
                "sort[0][direction]": "desc",
            }

            records, err = _fetch_eia(url, params)
            if err:
                return ToolResult(success=False, output=f"{name}: {err}")

            parsed = _parse_eia_records(records)
            signals = _compute_stock_signals(parsed, name)

            all_records[name] = parsed
            all_signals[name] = signals

            if self._cache:
                self._cache.put(
                    "energy_supply",
                    {"key": cache_key},
                    {"output": "", "data": {"records": parsed, "signals": signals}},
                )

        summary = _format_petroleum_summary(all_records, all_signals, label, weeks)

        result_data = {
            "series": all_records,
            "signals": all_signals,
            "label": label,
            "weeks": weeks,
        }

        try:
            self._persist_petroleum_series(label, all_records)
        except Exception:
            log.exception("Energy supply petroleum persistence failed (non-fatal)")

        return ToolResult(success=True, output=summary, data=result_data)

    # ── L2 persistence (US country entity) ──────────────────────────

    def _us_country_entity_id(self) -> str | None:
        if entity_id_from_key is None:
            return None
        return entity_id_from_key("country", "US")

    def _persist_petroleum_series(
        self, label: str, all_records: dict[str, list[dict]]
    ) -> int:
        if self._store is None or entity_id_from_key is None:
            return 0
        return self._persist_petroleum_series_inner(label, all_records)

    def _persist_petroleum_series_inner(
        self, label: str, all_records: dict[str, list[dict]]
    ) -> int:
        assert self._store is not None
        eid = self._us_country_entity_id()
        if not eid:
            return 0
        self._store.register_entity(
            entity_type="country",
            canonical_name="United States",
            entity_id=eid,
        )
        n = 0
        for series_name, records in all_records.items():
            for rec in records:
                period = (rec.get("period") or "").strip()
                if not period:
                    continue
                self._store.store_entity_observation(
                    entity_id=eid,
                    source_tool="energy_supply",
                    observed_at=_eia_period_to_ts(period),
                    observation_type="petroleum_inventory",
                    depth_level=2,
                    value={
                        "mode": label,
                        "series": series_name,
                        "period": period,
                        "value": rec.get("value"),
                        "units": rec.get("units"),
                        "product": rec.get("product"),
                        "process": rec.get("process"),
                    },
                )
                n += 1
        if n:
            log.info("Energy supply L2: %d petroleum_inventory observations", n)
        return n

    def _persist_rig_count(self, records: list[dict]) -> int:
        if self._store is None or entity_id_from_key is None:
            return 0
        eid = self._us_country_entity_id()
        if not eid:
            return 0
        self._store.register_entity(
            entity_type="country",
            canonical_name="United States",
            entity_id=eid,
        )
        n = 0
        for rec in records:
            period = (rec.get("period") or "").strip()
            if not period:
                continue
            self._store.store_entity_observation(
                entity_id=eid,
                source_tool="energy_supply",
                observed_at=_eia_period_to_ts(period),
                observation_type="rig_count",
                depth_level=2,
                value={
                    "mode": "rig_count",
                    "period": period,
                    "value": rec.get("value"),
                    "units": rec.get("units"),
                },
            )
            n += 1
        if n:
            log.info("Energy supply L2: %d rig_count observations", n)
        return n


# ── EIA fetch (module-level for testability) ────────────────────


def _fetch_eia(
    url: str,
    params: dict,
) -> tuple[list[dict], str | None]:
    """Fetch from EIA API v2. Returns (data_list, error_or_None)."""
    try:
        with httpx.Client(
            timeout=_TIMEOUT,
            headers={"User-Agent": _UA},
        ) as client:
            resp = client.get(url, params=params)
    except httpx.TimeoutException:
        return [], "EIA API request timed out."
    except httpx.HTTPError as exc:
        return [], f"HTTP error: {exc}"

    if resp.status_code == 429:
        return [], "EIA API rate limit reached. Retry later."
    if resp.status_code == 404:
        return [], "EIA API endpoint not found (404)."
    if resp.status_code != 200:
        return [], f"EIA API returned HTTP {resp.status_code}"

    try:
        body = resp.json()
    except Exception:
        return [], "Failed to parse EIA API response."

    # Check for EIA error response
    if "error" in body:
        return [], f"EIA error: {body['error']}"

    data = body.get("response", {}).get("data", [])
    return data, None


def _parse_eia_records(raw: list[dict]) -> list[dict]:
    """Parse EIA API data into normalized records."""
    records = []
    for entry in raw:
        value_str = entry.get("value")
        value = _safe_float(value_str)
        if value is None:
            continue

        records.append(
            {
                "period": entry.get("period", ""),
                "area": entry.get("area-name", ""),
                "product": entry.get("product-name", ""),
                "process": entry.get("process-name", ""),
                "series": entry.get("series", ""),
                "series_description": entry.get("series-description", ""),
                "value": value,
                "units": entry.get("units", ""),
            }
        )

    # Sort chronologically (EIA returns newest-first)
    records.sort(key=lambda r: r["period"])
    return records


def _eia_period_to_ts(period: str) -> float:
    """EIA weekly YYYY-MM-DD or monthly YYYY-MM → unix timestamp."""
    period = period.strip()
    try:
        if len(period) >= 10:
            dt = datetime.strptime(period[:10], "%Y-%m-%d")
        else:
            dt = datetime.strptime(period[:7], "%Y-%m").replace(day=1)
        return dt.replace(tzinfo=UTC).timestamp()
    except ValueError:
        return datetime.now(UTC).timestamp()


def _safe_float(val: Any) -> float | None:
    """Convert to float, handling None/empty/non-numeric."""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ── Signal computation ──────────────────────────────────────────


def _compute_stock_signals(records: list[dict], name: str) -> dict:
    """Compute signals for a petroleum stock series."""
    if not records:
        return {"status": "NO_DATA", "alert": None}

    values = [r["value"] for r in records]
    latest = values[-1]
    avg = sum(values) / len(values)
    peak = max(values)
    trough = min(values)

    signals: dict[str, Any] = {
        "latest_value": latest,
        "period_average": round(avg, 2),
        "period_peak": peak,
        "period_trough": trough,
        "total_weeks": len(values),
        "units": records[-1].get("units", "") if records else "",
    }

    # Week-over-week change
    if len(values) >= 2:
        wow_change = latest - values[-2]
        signals["wow_change"] = round(wow_change, 1)
        signals["wow_pct"] = round(100 * wow_change / values[-2], 2) if values[-2] != 0 else None
    else:
        signals["wow_change"] = None
        signals["wow_pct"] = None

    # Consecutive direction counter
    consecutive = 0
    direction = None
    for i in range(len(values) - 1, 0, -1):
        diff = values[i] - values[i - 1]
        if diff > 0:
            if direction is None:
                direction = "build"
            if direction == "build":
                consecutive += 1
            else:
                break
        elif diff < 0:
            if direction is None:
                direction = "draw"
            if direction == "draw":
                consecutive += 1
            else:
                break
        else:
            break

    signals["consecutive_weeks"] = consecutive
    signals["direction"] = direction

    # Alert thresholds
    wow = signals.get("wow_change")
    if wow is not None:
        if abs(wow) > 5000:  # >5M barrel change
            signals["alert"] = f"SURPRISE — {wow:+,.0f}K barrel {'build' if wow > 0 else 'draw'}"
        elif direction == "draw" and consecutive >= 3:
            signals["alert"] = f"TIGHTENING — {consecutive} consecutive weekly draws"
        elif direction == "build" and consecutive >= 3:
            signals["alert"] = f"BUILDING — {consecutive} consecutive weekly builds"
        else:
            signals["alert"] = None
    else:
        signals["alert"] = None

    return signals


def _compute_rig_signals(records: list[dict]) -> dict:
    """Compute signals for rig count data."""
    if not records:
        return {"status": "NO_DATA", "alert": None}

    values = [r["value"] for r in records]
    latest = values[-1]
    avg = sum(values) / len(values)
    peak = max(values)

    signals: dict[str, Any] = {
        "latest_value": latest,
        "period_average": round(avg, 2),
        "period_peak": peak,
        "total_months": len(values),
    }

    # Month-over-month change
    if len(values) >= 2:
        mom_change = latest - values[-2]
        signals["mom_change"] = round(mom_change, 1)
        signals["mom_pct"] = round(100 * mom_change / values[-2], 2) if values[-2] != 0 else None
    else:
        signals["mom_change"] = None
        signals["mom_pct"] = None

    # 3-month trend
    if len(values) >= 3:
        three_mo_ago = values[-3]
        if three_mo_ago > 0:
            three_mo_change_pct = round(100 * (latest - three_mo_ago) / three_mo_ago, 1)
            signals["three_month_change_pct"] = three_mo_change_pct
            if three_mo_change_pct < -10:
                signals["alert"] = f"WARNING — rig count down {abs(three_mo_change_pct)}% over 3 months"
            elif three_mo_change_pct > 10:
                signals["alert"] = f"NOTICE — rig count up {three_mo_change_pct}% over 3 months"
            else:
                signals["alert"] = None
        else:
            signals["three_month_change_pct"] = None
            signals["alert"] = None
    else:
        signals["three_month_change_pct"] = None
        signals["alert"] = None

    # Trend direction
    if len(values) >= 6:
        recent_3 = values[-3:]
        prior_3 = values[-6:-3]
        recent_avg = sum(recent_3) / len(recent_3)
        prior_avg = sum(prior_3) / len(prior_3)
        if prior_avg > 0:
            ratio = recent_avg / prior_avg
            if ratio > 1.05:
                signals["trend"] = "EXPANDING"
            elif ratio < 0.95:
                signals["trend"] = "CONTRACTING"
            else:
                signals["trend"] = "STABLE"
        else:
            signals["trend"] = "NO_BASELINE"
    else:
        signals["trend"] = "INSUFFICIENT_DATA"

    return signals


# ── Formatting ──────────────────────────────────────────────────


def _format_petroleum_summary(
    all_records: dict[str, list[dict]],
    all_signals: dict[str, dict],
    label: str,
    weeks: int,
) -> str:
    """Format petroleum stock or supply summary."""
    title = "Petroleum Stocks" if label == "petroleum_stocks" else "Petroleum Supply"
    lines = [f"EIA {title} — Last {weeks} weeks"]

    for name, records in all_records.items():
        signals = all_signals.get(name, {})
        lines.append("")
        desc = records[0].get("series_description", name) if records else name
        lines.append(f"  {desc}")

        if not records:
            lines.append("    No data")
            continue

        latest = signals.get("latest_value", "N/A")
        units = signals.get("units", "")
        lines.append(
            f"    Latest: {latest:,.0f} {units}" if isinstance(latest, (int, float)) else f"    Latest: {latest}"
        )
        avg = signals.get("period_average", "N/A")
        lines.append(f"    Average: {avg:,.0f}" if isinstance(avg, (int, float)) else f"    Average: {avg}")

        wow = signals.get("wow_change")
        wow_pct = signals.get("wow_pct")
        if wow is not None:
            pct_str = f" ({wow_pct:+.2f}%)" if isinstance(wow_pct, (int, float)) else ""
            lines.append(f"    WoW change: {wow:+,.0f}{pct_str}")

        direction = signals.get("direction")
        consec = signals.get("consecutive_weeks", 0)
        if direction and consec > 0:
            lines.append(f"    Direction: {consec} consecutive {direction}s")

        alert = signals.get("alert")
        if alert:
            lines.append(f"    ⚠ {alert}")

    return "\n".join(lines)


def _format_rig_summary(
    records: list[dict],
    signals: dict,
    months: int,
) -> str:
    """Format rig count summary."""
    lines = [f"EIA Rig Count — Last {months} months"]
    lines.append(f"Records: {len(records)}")

    if not records:
        lines.append("No rig count data available.")
        return "\n".join(lines)

    lines.append(f"Latest: {signals.get('latest_value', 'N/A'):.0f} rigs")
    lines.append(f"Period average: {signals.get('period_average', 'N/A'):.0f}")
    lines.append(f"Period peak: {signals.get('period_peak', 'N/A'):.0f}")

    mom = signals.get("mom_change")
    if mom is not None:
        lines.append(f"MoM change: {mom:+.0f} ({signals.get('mom_pct', 0):+.1f}%)")

    three_mo = signals.get("three_month_change_pct")
    if three_mo is not None:
        lines.append(f"3-month change: {three_mo:+.1f}%")

    trend = signals.get("trend", "N/A")
    lines.append(f"Trend: {trend}")

    alert = signals.get("alert")
    if alert:
        lines.append(f"⚠ {alert}")

    # Recent values
    lines.append("")
    lines.append("Recent data:")
    for r in records[-6:]:
        lines.append(f"  {r['period']}: {r['value']:.0f} rigs")

    return "\n".join(lines)
