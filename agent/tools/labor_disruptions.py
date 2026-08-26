"""
Tool: Labor Disruptions Monitor — BLS Work Stoppages

US major work stoppages data from the Bureau of Labor Statistics.
Tracks workers involved and idle days lost for strikes/lockouts
involving 1,000+ workers.  Monthly data, free, no authentication.

Source:
  BLS Public Data API v2  — https://api.bls.gov/publicAPI/v2/timeseries/data/
  Series WSU001: Workers involved (thousands) in major work stoppages
  Series WSU002: Days idle (thousands) during month

Modes:
  work_stoppages — Workers involved in major stoppages per month.
                   Spike = new large strike; sustained high = protracted dispute.
  idle_days      — Working days lost per month.
                   High idle-days / low workers = long-duration dispute.
  overview       — Both series together with derived signals.

Signal theory:
  - Workers spike >500K in a month = major sector-wide disruption
  - Days idle rising while workers flat = strikes dragging on (entrenched)
  - Workers/idle ratio shift = changing strike character
  - Multi-month elevated workers = contagion across sectors
  - Sudden drop to zero after prolonged stoppage = settlement → supply recovery

Market relevance:
  Work stoppages → production output (autos, logistics, healthcare),
  wage inflation pressure, earnings revisions for affected sectors,
  consumer spending shifts, political/policy response probability.
"""

from __future__ import annotations

import logging
from datetime import UTC

UTC = UTC
from typing import Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

_UA = "TirraMind/0.1"
_TIMEOUT = 15
_CACHE_TTL = 21600  # 6 hours — BLS monthly data changes infrequently

_BLS_BASE = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# BLS series IDs for major work stoppages
SERIES_WORKERS = "WSU001"  # Workers involved (thousands)
SERIES_IDLE = "WSU002"  # Days idle (thousands)

VALID_MODES = {"work_stoppages", "idle_days", "overview"}

# Default year range
_DEFAULT_SPAN = 4  # years back


class LaborDisruptionsTool(Tool):
    """Monitor US labor disruptions via BLS work stoppages data."""

    def __init__(self, cache: DataCache | None = None) -> None:
        self._cache = cache

    @property
    def name(self) -> str:
        return "labor_disruptions"

    @property
    def description(self) -> str:
        return (
            "Monitor US labor disruptions — major work stoppages (1,000+ workers), "
            "workers involved, and days idle from BLS monthly data. "
            "Detects strike surges, prolonged disputes, and cross-sector contagion."
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
                        "idle_days: days lost per month. "
                        "overview: both workers + idle days with signals. "
                        "work_stoppages: workers involved per month."
                    ),
                },
                "start_year": {
                    "type": "integer",
                    "description": "Start year (default: 4 years ago). Min 1993.",
                },
                "end_year": {
                    "type": "integer",
                    "description": "End year (default: current year).",
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

        from datetime import datetime

        now_year = datetime.now(UTC).year
        end_year = kwargs.get("end_year") or now_year
        start_year = kwargs.get("start_year") or (end_year - _DEFAULT_SPAN)

        start_year = max(start_year, 1993)
        end_year = min(end_year, now_year + 1)

        if start_year > end_year:
            return ToolResult(
                success=False,
                output=f"start_year ({start_year}) cannot be after end_year ({end_year}).",
            )

        if mode == "work_stoppages":
            return self._handle_single_series(
                SERIES_WORKERS,
                start_year,
                end_year,
                "workers",
            )
        elif mode == "idle_days":
            return self._handle_single_series(
                SERIES_IDLE,
                start_year,
                end_year,
                "idle_days",
            )
        else:
            return self._handle_overview(start_year, end_year)

    # ── Mode handlers ───────────────────────────────────────

    def _handle_single_series(
        self,
        series_id: str,
        start: int,
        end: int,
        label: str,
    ) -> ToolResult:
        cache_key = f"labor:{label}:{start}-{end}"
        if self._cache:
            hit = self._cache.get("labor_disruptions", {"key": cache_key})
            if hit is not None:
                return ToolResult(
                    success=True,
                    output=hit["output"],
                    data=hit["data"],
                )

        records, err = _fetch_bls_series(series_id, start, end)
        if err:
            return ToolResult(success=False, output=err)

        signals = _compute_single_signals(records, label)
        summary = _format_single_summary(records, signals, label, start, end)

        result_data = {
            "records": records,
            "count": len(records),
            "series": series_id,
            "label": label,
            "start_year": start,
            "end_year": end,
            "signals": signals,
        }

        if self._cache:
            self._cache.put(
                "labor_disruptions",
                {"key": cache_key},
                {"output": summary, "data": result_data},
            )

        return ToolResult(success=True, output=summary, data=result_data)

    def _handle_overview(self, start: int, end: int) -> ToolResult:
        cache_key = f"labor:overview:{start}-{end}"
        if self._cache:
            hit = self._cache.get("labor_disruptions", {"key": cache_key})
            if hit is not None:
                return ToolResult(
                    success=True,
                    output=hit["output"],
                    data=hit["data"],
                )

        workers, err_w = _fetch_bls_series(SERIES_WORKERS, start, end)
        if err_w:
            return ToolResult(success=False, output=f"Workers series: {err_w}")

        idle, err_i = _fetch_bls_series(SERIES_IDLE, start, end)
        if err_i:
            return ToolResult(success=False, output=f"Idle-days series: {err_i}")

        signals = _compute_overview_signals(workers, idle)
        summary = _format_overview_summary(workers, idle, signals, start, end)

        result_data = {
            "workers": workers,
            "idle_days": idle,
            "workers_count": len(workers),
            "idle_count": len(idle),
            "start_year": start,
            "end_year": end,
            "signals": signals,
        }

        if self._cache:
            self._cache.put(
                "labor_disruptions",
                {"key": cache_key},
                {"output": summary, "data": result_data},
            )

        return ToolResult(success=True, output=summary, data=result_data)


# ── BLS fetch (module-level for testability) ────────────────────


def _fetch_bls_series(
    series_id: str,
    start_year: int,
    end_year: int,
) -> tuple[list[dict], str | None]:
    """Fetch BLS time series. Returns (records, error_string_or_None)."""
    payload = {
        "seriesid": [series_id],
        "startyear": str(start_year),
        "endyear": str(end_year),
    }

    try:
        with httpx.Client(
            timeout=_TIMEOUT,
            headers={"User-Agent": _UA},
        ) as client:
            resp = client.post(_BLS_BASE, json=payload)
    except httpx.TimeoutException:
        return [], "BLS API request timed out."
    except httpx.HTTPError as exc:
        return [], f"HTTP error: {exc}"

    if resp.status_code == 429:
        return [], "BLS API rate limit reached. Retry later."
    if resp.status_code != 200:
        return [], f"BLS API returned HTTP {resp.status_code}"

    try:
        body = resp.json()
    except Exception:
        return [], "Failed to parse BLS API response."

    status = body.get("status", "")
    if status != "REQUEST_SUCCEEDED":
        msg = "; ".join(body.get("message", ["Unknown error"]))
        return [], f"BLS request failed: {msg}"

    series_list = body.get("Results", {}).get("series", [])
    if not series_list:
        return [], f"No series returned for {series_id}."

    raw_data = series_list[0].get("data", [])
    return _parse_bls_records(raw_data, series_id), None


def _parse_bls_records(raw: list, series_id: str) -> list[dict]:
    """Parse BLS time series data into normalized records."""
    records = []
    for entry in raw:
        year = entry.get("year", "")
        period = entry.get("period", "")
        period_name = entry.get("periodName", "")
        value_str = entry.get("value", "")

        value = _safe_float(value_str)
        if value is None:
            continue

        footnotes = []
        for fn in entry.get("footnotes", []):
            code = fn.get("code", "")
            text = fn.get("text", "")
            if code or text:
                footnotes.append(f"{code}: {text}" if code else text)

        records.append(
            {
                "year": year,
                "period": period,
                "period_name": period_name,
                "value": value,
                "preliminary": any(
                    "P" in fn.get("code", "") for fn in entry.get("footnotes", []) if isinstance(fn, dict)
                ),
                "series_id": series_id,
                "footnotes": footnotes,
            }
        )

    # Sort chronologically (BLS returns newest-first)
    records.sort(key=lambda r: (r["year"], r["period"]))
    return records


def _safe_float(val: Any) -> float | None:
    """Convert to float, returning None on failure."""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ── Signal computation ──────────────────────────────────────────


def _compute_single_signals(records: list[dict], label: str) -> dict:
    """Compute signals from a single BLS series."""
    if not records:
        return {"status": "NO_DATA", "alert": None}

    values = [r["value"] for r in records]
    latest = values[-1]
    avg = sum(values) / len(values)
    peak = max(values)

    # Count non-zero months
    active_months = sum(1 for v in values if v > 0)

    signals: dict[str, Any] = {
        "latest_value": latest,
        "period_average": round(avg, 2),
        "period_peak": peak,
        "total_months": len(values),
        "active_months": active_months,
        "active_pct": round(100 * active_months / len(values), 1) if values else 0,
    }

    # Alert thresholds
    if label == "workers":
        if latest > 500:
            signals["alert"] = "CRITICAL — >500K workers in stoppages"
        elif latest > 100:
            signals["alert"] = "WARNING — >100K workers in stoppages"
        elif latest > 0:
            signals["alert"] = "NOTICE — active work stoppages"
        else:
            signals["alert"] = None
    elif label == "idle_days":
        if latest > 10000:
            signals["alert"] = "CRITICAL — >10M days idle this month"
        elif latest > 1000:
            signals["alert"] = "WARNING — >1M days idle this month"
        elif latest > 0:
            signals["alert"] = "NOTICE — ongoing idle time"
        else:
            signals["alert"] = None
    else:
        signals["alert"] = None

    # Trend: compare latest 6 months avg to prior 6 months avg
    if len(values) >= 12:
        recent = values[-6:]
        prior = values[-12:-6]
        recent_avg = sum(recent) / len(recent)
        prior_sum = sum(prior)
        prior_avg = prior_sum / len(prior)
        if prior_avg > 0:
            signals["trend_ratio"] = round(recent_avg / prior_avg, 2)
            if recent_avg > prior_avg * 1.5:
                signals["trend"] = "ESCALATING"
            elif recent_avg > prior_avg * 1.1:
                signals["trend"] = "RISING"
            elif recent_avg < prior_avg * 0.5:
                signals["trend"] = "DECLINING"
            else:
                signals["trend"] = "STABLE"
        else:
            signals["trend_ratio"] = None
            signals["trend"] = "NEW_ACTIVITY" if recent_avg > 0 else "QUIET"
    else:
        signals["trend"] = "INSUFFICIENT_DATA"
        signals["trend_ratio"] = None

    return signals


def _compute_overview_signals(
    workers: list[dict],
    idle: list[dict],
) -> dict:
    """Compute combined signals from workers + idle-days series."""
    w_signals = _compute_single_signals(workers, "workers")
    i_signals = _compute_single_signals(idle, "idle_days")

    # Intensity: idle days per worker (thousands/thousands)
    latest_w = w_signals.get("latest_value", 0)
    latest_i = i_signals.get("latest_value", 0)

    if latest_w and latest_w > 0:
        intensity = round(latest_i / latest_w, 2)
    else:
        intensity = None

    # Consecutive active months
    w_values = [r["value"] for r in workers]
    consecutive = 0
    for v in reversed(w_values):
        if v > 0:
            consecutive += 1
        else:
            break

    combined_alert = w_signals.get("alert") or i_signals.get("alert")

    return {
        "workers": w_signals,
        "idle_days": i_signals,
        "intensity_ratio": intensity,
        "consecutive_active_months": consecutive,
        "combined_alert": combined_alert,
    }


# ── Formatting ──────────────────────────────────────────────────


def _format_single_summary(
    records: list[dict],
    signals: dict,
    label: str,
    start: int,
    end: int,
) -> str:
    """Format a single-series summary."""
    unit = "thousands of workers" if label == "workers" else "thousands of days idle"
    lines = [f"BLS Work Stoppages — {label} ({start}-{end})"]
    lines.append(f"Records: {len(records)} months")

    if not records:
        lines.append("No data available for this period.")
        return "\n".join(lines)

    lines.append(f"Latest: {signals.get('latest_value', 'N/A')} ({unit})")
    lines.append(f"Period average: {signals.get('period_average', 'N/A')}")
    lines.append(f"Period peak: {signals.get('period_peak', 'N/A')}")
    lines.append(
        f"Active months: {signals.get('active_months', 0)}"
        f"/{signals.get('total_months', 0)} "
        f"({signals.get('active_pct', 0)}%)"
    )

    trend = signals.get("trend", "N/A")
    lines.append(f"Trend: {trend}")

    alert = signals.get("alert")
    if alert:
        lines.append(f"⚠ {alert}")

    # Show last 6 records
    lines.append("")
    lines.append("Recent data:")
    for r in records[-6:]:
        prelim = " (P)" if r.get("preliminary") else ""
        lines.append(f"  {r['year']}-{r['period']}: {r['value']}{prelim}")

    return "\n".join(lines)


def _format_overview_summary(
    workers: list[dict],
    idle: list[dict],
    signals: dict,
    start: int,
    end: int,
) -> str:
    """Format combined overview summary."""
    lines = [f"BLS Work Stoppages — Overview ({start}-{end})"]

    w_sig = signals.get("workers", {})
    i_sig = signals.get("idle_days", {})

    lines.append(f"Workers involved: {len(workers)} months of data")
    lines.append(f"  Latest: {w_sig.get('latest_value', 'N/A')}K workers")
    lines.append(f"  Trend: {w_sig.get('trend', 'N/A')}")

    lines.append(f"Days idle: {len(idle)} months of data")
    lines.append(f"  Latest: {i_sig.get('latest_value', 'N/A')}K days")
    lines.append(f"  Trend: {i_sig.get('trend', 'N/A')}")

    intensity = signals.get("intensity_ratio")
    if intensity is not None:
        lines.append(f"Strike intensity (idle/workers ratio): {intensity}")

    consec = signals.get("consecutive_active_months", 0)
    if consec > 0:
        lines.append(f"Consecutive active months: {consec}")

    alert = signals.get("combined_alert")
    if alert:
        lines.append(f"⚠ {alert}")

    return "\n".join(lines)
