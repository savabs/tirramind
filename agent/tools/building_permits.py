"""
Tool: Building Permits / Construction — Real Estate Cycle Detector

FRED API:    https://api.stlouisfed.org/fred/  (free key)
Census/BLS:  Various permit + construction series (via FRED)

Building permits are a leading indicator of economic activity.  Permits
crash 12-18 months before recessions.  They represent irreversible
financial commitments — developers must pay fees, hire architects, and
secure financing before a permit is issued.

Regional divergence matters: permits booming in the Sun Belt while
crashing in the Midwest = migration-driven expansion, not organic growth.

Single-family vs multi-family divergence signals credit conditions:
multi-family holds up longer (institutional capital) while single-family
crashes first (consumer credit sensitivity).

Modes
-----
permits       National building permit trends (total, single-family,
              multi-family). Month-over-month and year-over-year changes.

regional      Regional breakdown (Northeast, Midwest, South, West).
              Detects geographic divergence in construction cycles.

housing_starts  Housing starts vs permits ratio — high ratio = confident
                builders; low ratio = permits pulled but not started (cold feet).

Signal theory:
  - Permits falling 3+ consecutive months = recession warning
  - Single-family permits falling while multi-family holds = credit tightening
  - Permits/starts ratio declining = builder confidence dropping
  - Regional divergence = migration + capital reallocation signal
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_UA = "TirraMind/0.1 (building-permits-tool)"
_TIMEOUT = 20
_CACHE_TTL = 14400  # 4 hours — monthly data

VALID_MODES = frozenset({"permits", "regional", "housing_starts"})

# National permit series
_PERMIT_SERIES: dict[str, str] = {
    "PERMIT": "Total Building Permits (SA, thousands)",
    "PERMIT1": "Single-Family Permits (SA, thousands)",
    "PERMITNSA": "Total Building Permits (NSA, thousands)",
}

# Regional permit series (SA = seasonally adjusted)
_REGIONAL_SERIES: dict[str, str] = {
    "PERIMT1NE": "Northeast — Single-Family Permits",
    "PERIMT1MW": "Midwest — Single-Family Permits",
    "PERIMT1SO": "South — Single-Family Permits",
    "PERIMT1WS": "West — Single-Family Permits",
    "PERMITNE": "Northeast — Total Permits",
    "PERMITMW": "Midwest — Total Permits",
    "PERMITS": "South — Total Permits",
    "PERMITW": "West — Total Permits",
}

# Housing starts series for starts-to-permits ratio
_STARTS_SERIES: dict[str, str] = {
    "HOUST": "Housing Starts (SA, thousands)",
    "HOUST1F": "Single-Family Starts (SA, thousands)",
    "PERMIT": "Total Permits (SA, thousands)",
    "PERMIT1": "Single-Family Permits (SA, thousands)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_fred(
    series_id: str,
    api_key: str,
    *,
    limit: int = 24,
) -> list[dict[str, str]]:
    """Fetch recent FRED observations in descending date order."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": str(limit),
    }
    try:
        with httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": _UA}) as client:
            resp = client.get(_FRED_BASE, params=params)
            if resp.status_code != 200:
                log.warning("FRED HTTP %d for %s", resp.status_code, series_id)
                return []
            data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("FRED error for %s: %s", series_id, exc)
        return []

    observations = data.get("observations", [])
    return [
        {"date": obs["date"], "value": obs["value"]}
        for obs in observations
        if obs.get("value") not in (".", "", None)
    ]


def _latest(series: list[dict[str, str]]) -> tuple[str, float | None]:
    """Get latest (date, value) from a desc-sorted series."""
    if not series:
        return ("N/A", None)
    try:
        return (series[0]["date"], float(series[0]["value"]))
    except (ValueError, KeyError):
        return (series[0].get("date", "N/A"), None)


def _pct_change(series: list[dict[str, str]], offset: int) -> float | None:
    """Compute % change between latest and offset-th observation."""
    if len(series) <= offset:
        return None
    try:
        current = float(series[0]["value"])
        previous = float(series[offset]["value"])
    except (ValueError, KeyError, IndexError):
        return None
    if previous == 0:
        return None
    return ((current - previous) / previous) * 100


def _trend_direction(series: list[dict[str, str]], n: int = 3) -> str:
    """Simple trend from last n observations (desc order)."""
    vals = []
    for obs in series[:n + 1]:
        try:
            vals.append(float(obs["value"]))
        except (ValueError, KeyError):
            continue
    if len(vals) < 2:
        return "insufficient data"
    if vals[0] > vals[-1] * 1.05:
        return "rising"
    if vals[0] < vals[-1] * 0.95:
        return "falling"
    return "stable"


def _consecutive_declines(series: list[dict[str, str]]) -> int:
    """Count consecutive months of decline from most recent."""
    count = 0
    for i in range(len(series) - 1):
        try:
            current = float(series[i]["value"])
            previous = float(series[i + 1]["value"])
        except (ValueError, KeyError):
            break
        if current < previous:
            count += 1
        else:
            break
    return count


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class BuildingPermitsTool(Tool):
    """Monitor US building permits and housing construction cycle."""

    name = "building_permits"
    description = (
        "Track US building permits, housing starts, and construction activity. "
        "Leading economic indicator: permits crash 12-18 months before "
        "recessions. Regional and single/multi-family breakdowns reveal "
        "credit conditions and migration patterns."
    )
    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": sorted(VALID_MODES),
                "description": (
                    "housing_starts: starts-to-permits ratio (builder confidence). "
                    "permits: national permit trends (total + single-family). "
                    "regional: permits by Census region (NE, MW, S, W)."
                ),
            },
            "months": {
                "type": "integer",
                "description": "Months of data (default 24, max 120).",
            },
        },
        "required": ["mode"],
    }

    def __init__(
        self,
        *,
        fred_api_key: str = "",
        cache: DataCache | None = None,
    ) -> None:
        self._api_key = fred_api_key
        self._cache = cache

    def execute(self, **kwargs: Any) -> ToolResult:
        mode = (kwargs.get("mode") or "").strip().lower()
        if mode not in VALID_MODES:
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Use: {', '.join(sorted(VALID_MODES))}",
            )

        if not self._api_key:
            return ToolResult(
                success=False,
                output=(
                    "FRED API key required for building_permits. "
                    "Set TIRRA_FRED_API_KEY in .env."
                ),
            )

        months = min(max(int(kwargs.get("months", 24)), 1), 120)

        if mode == "permits":
            return self._permits(months=months)
        if mode == "regional":
            return self._regional(months=months)
        return self._housing_starts(months=months)

    # ── permits mode ─────────────────────────────────────────────────

    def _permits(self, *, months: int) -> ToolResult:
        cache_key = f"permits_{months}"
        if self._cache:
            cached = self._cache.get("building_permits", cache_key)
            if cached is not None:
                return self._format_permits(cached, months, from_cache=True)

        results: dict[str, list[dict[str, str]]] = {}
        for sid in _PERMIT_SERIES:
            results[sid] = _fetch_fred(sid, self._api_key, limit=months)

        if self._cache:
            self._cache.set("building_permits", cache_key, results, ttl=_CACHE_TTL)

        return self._format_permits(results, months)

    def _format_permits(
        self,
        results: dict[str, list[dict[str, str]]],
        months: int,
        *,
        from_cache: bool = False,
    ) -> ToolResult:
        tag = " (cached)" if from_cache else ""
        lines = [f"US Building Permits{tag}", ""]

        summary: dict[str, Any] = {}
        for sid, label in _PERMIT_SERIES.items():
            obs = results.get(sid, [])
            date, value = _latest(obs)
            mom = _pct_change(obs, 1)
            yoy = _pct_change(obs, 12)
            trend = _trend_direction(obs)
            declines = _consecutive_declines(obs)

            val_str = f"{value:,.1f}K" if value is not None else "N/A"
            mom_str = f"{mom:+.1f}% MoM" if mom is not None else ""
            yoy_str = f"{yoy:+.1f}% YoY" if yoy is not None else ""

            lines.append(f"  {label}: {val_str} ({date})")
            if mom_str or yoy_str:
                lines.append(f"    {mom_str}  {yoy_str}  trend: {trend}")
            if declines >= 3:
                lines.append(f"    ⚠ {declines} consecutive months of decline")

            summary[sid] = {
                "label": label,
                "latest_date": date,
                "latest_value": value,
                "mom_pct": mom,
                "yoy_pct": yoy,
                "trend": trend,
                "consecutive_declines": declines,
            }

        # Single-family vs total ratio
        total_val = summary.get("PERMIT", {}).get("latest_value")
        sf_val = summary.get("PERMIT1", {}).get("latest_value")
        if total_val and sf_val and total_val > 0:
            sf_share = (sf_val / total_val) * 100
            mf_share = 100 - sf_share
            lines.append("")
            lines.append(
                f"  Single-family share: {sf_share:.0f}% | Multi-family: {mf_share:.0f}%"
            )

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "mode": "permits",
                "summary": summary,
                "series": results,
            },
        )

    # ── regional mode ────────────────────────────────────────────────

    def _regional(self, *, months: int) -> ToolResult:
        cache_key = f"regional_{months}"
        if self._cache:
            cached = self._cache.get("building_permits", cache_key)
            if cached is not None:
                return self._format_regional(cached, from_cache=True)

        results: dict[str, list[dict[str, str]]] = {}
        for sid in _REGIONAL_SERIES:
            results[sid] = _fetch_fred(sid, self._api_key, limit=months)

        if self._cache:
            self._cache.set("building_permits", cache_key, results, ttl=_CACHE_TTL)

        return self._format_regional(results)

    def _format_regional(
        self,
        results: dict[str, list[dict[str, str]]],
        *,
        from_cache: bool = False,
    ) -> ToolResult:
        tag = " (cached)" if from_cache else ""
        lines = [f"US Building Permits — Regional Breakdown{tag}", ""]

        summary: dict[str, Any] = {}
        for sid, label in _REGIONAL_SERIES.items():
            obs = results.get(sid, [])
            date, value = _latest(obs)
            trend = _trend_direction(obs)
            yoy = _pct_change(obs, 12)

            val_str = f"{value:,.1f}K" if value is not None else "N/A"
            yoy_str = f"({yoy:+.1f}% YoY)" if yoy is not None else ""
            lines.append(f"  {label}: {val_str} {yoy_str} — {trend}")

            summary[sid] = {
                "label": label,
                "latest_value": value,
                "yoy_pct": yoy,
                "trend": trend,
            }

        # Detect regional divergence
        region_vals: dict[str, float] = {}
        for sid, s in summary.items():
            if "Total" in (s.get("label") or "") and s.get("latest_value") is not None:
                region_name = s["label"].split(" — ")[0]
                region_vals[region_name] = s["latest_value"]

        if len(region_vals) >= 2:
            strongest = max(region_vals, key=lambda k: region_vals[k])
            weakest = min(region_vals, key=lambda k: region_vals[k])
            if region_vals[weakest] > 0:
                ratio = region_vals[strongest] / region_vals[weakest]
                lines.append("")
                lines.append(
                    f"  Strongest: {strongest} ({region_vals[strongest]:,.1f}K) | "
                    f"Weakest: {weakest} ({region_vals[weakest]:,.1f}K) — "
                    f"ratio: {ratio:.1f}x"
                )

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "mode": "regional",
                "summary": summary,
                "series": results,
            },
        )

    # ── housing_starts mode ──────────────────────────────────────────

    def _housing_starts(self, *, months: int) -> ToolResult:
        cache_key = f"starts_{months}"
        if self._cache:
            cached = self._cache.get("building_permits", cache_key)
            if cached is not None:
                return self._format_starts(cached, from_cache=True)

        results: dict[str, list[dict[str, str]]] = {}
        for sid in _STARTS_SERIES:
            results[sid] = _fetch_fred(sid, self._api_key, limit=months)

        if self._cache:
            self._cache.set("building_permits", cache_key, results, ttl=_CACHE_TTL)

        return self._format_starts(results)

    def _format_starts(
        self,
        results: dict[str, list[dict[str, str]]],
        *,
        from_cache: bool = False,
    ) -> ToolResult:
        tag = " (cached)" if from_cache else ""
        lines = [f"Housing Starts vs Permits{tag}", ""]

        summary: dict[str, Any] = {}
        for sid, label in _STARTS_SERIES.items():
            obs = results.get(sid, [])
            date, value = _latest(obs)
            trend = _trend_direction(obs)
            val_str = f"{value:,.1f}K" if value is not None else "N/A"
            lines.append(f"  {label}: {val_str} ({date}) — {trend}")
            summary[sid] = {
                "label": label,
                "latest_date": date,
                "latest_value": value,
                "trend": trend,
            }

        # Compute starts/permits ratio (builder confidence)
        starts_val = summary.get("HOUST", {}).get("latest_value")
        permits_val = summary.get("PERMIT", {}).get("latest_value")
        sf_starts = summary.get("HOUST1F", {}).get("latest_value")
        sf_permits = summary.get("PERMIT1", {}).get("latest_value")

        if starts_val and permits_val and permits_val > 0:
            ratio = starts_val / permits_val
            lines.append("")
            lines.append(
                f"  Starts/Permits ratio: {ratio:.2f} "
                f"(>0.95 = confident builders, <0.80 = cautious)"
            )
            summary["starts_permits_ratio"] = ratio

        if sf_starts and sf_permits and sf_permits > 0:
            sf_ratio = sf_starts / sf_permits
            lines.append(
                f"  SF Starts/Permits: {sf_ratio:.2f}"
            )
            summary["sf_starts_permits_ratio"] = sf_ratio

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "mode": "housing_starts",
                "summary": summary,
                "series": results,
            },
        )
