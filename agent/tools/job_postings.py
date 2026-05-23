"""
Tool: Job Postings / Hiring Intent — Labor Market Signal Detector

BLS JOLTS (via FRED):  https://api.stlouisfed.org/fred/  (free key)
BLS Public API:        https://api.bls.gov/publicAPI/v2/  (free, no auth)

Companies commit real money when they hire. JOLTS data (Job Openings, Quits,
Hires, Layoffs) is a leading indicator. The quits-to-layoffs ratio signals
worker confidence vs employer stress. Sector-level breakdowns reveal which
parts of the economy are expanding or contracting.

The BLS Public API provides sector-level JOLTS data without requiring a FRED
key. If TIRRA_FRED_API_KEY is set, FRED is used for faster access and wider
series.

Modes
-----
jolts       JOLTS headline data: job openings, quits, hires, layoffs.
            Uses FRED API if key available, else BLS direct.

sector      Sector-level JOLTS breakdown (by NAICS super-sector).
            Uses BLS Public API for sector series.

labor_market  Composite labor market overview: JOLTS + unemployment rate +
              initial claims. Requires FRED API key.

Signal theory:
  - Quits rising = workers confident (late-cycle), quits falling = workers scared
  - Openings/unemployed ratio > 1.5 = extremely tight, < 1.0 = recession
  - Layoffs spike = imminent recession signal
  - Sector divergence: tech hiring down + healthcare up = rotation signal
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
_BLS_BASE = "https://api.bls.gov/publicAPI/v2/timeseries/data"
_UA = "TirraMind/0.1 (job-postings-tool)"
_TIMEOUT = 20
_CACHE_TTL = 14400  # 4 hours — JOLTS is monthly, changes slowly

VALID_MODES = frozenset({"jolts", "sector", "labor_market"})

# FRED series IDs for headline JOLTS
_JOLTS_SERIES: dict[str, str] = {
    "JTSJOL": "Job Openings (thousands)",
    "JTSQUL": "Quits (thousands)",
    "JTSHIL": "Hires (thousands)",
    "JTSLDR": "Layoffs & Discharges (thousands)",
}

# BLS series IDs for sector-level JOLTS (Total Nonfarm sub-sectors)
_SECTOR_SERIES: dict[str, str] = {
    "JTS000000000000000JOL": "Total Nonfarm — Openings",
    "JTS100000000000000JOL": "Mining & Logging — Openings",
    "JTS200000000000000JOL": "Construction — Openings",
    "JTS300000000000000JOL": "Manufacturing — Openings",
    "JTS400000000000000JOL": "Trade, Transport, Utilities — Openings",
    "JTS510000000000000JOL": "Information — Openings",
    "JTS510099000000000JOL": "Financial Activities — Openings",
    "JTS600000000000000JOL": "Professional & Business Services — Openings",
    "JTS700000000000000JOL": "Education & Health Services — Openings",
    "JTS800000000000000JOL": "Leisure & Hospitality — Openings",
    "JTS900000000000000JOL": "Government — Openings",
}

# FRED series for composite labor market view
_LABOR_SERIES: dict[str, str] = {
    "JTSJOL": "Job Openings (thousands)",
    "JTSQUL": "Quits (thousands)",
    "JTSHIL": "Hires (thousands)",
    "JTSLDR": "Layoffs & Discharges (thousands)",
    "UNRATE": "Unemployment Rate (%)",
    "ICSA": "Initial Claims (weekly)",
    "PAYEMS": "Total Nonfarm Payrolls (thousands)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_fred_series(
    series_id: str,
    api_key: str,
    *,
    limit: int = 24,
) -> list[dict[str, str]]:
    """Fetch recent FRED observations. Returns list of {date, value}."""
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
        {"date": obs["date"], "value": obs["value"]} for obs in observations if obs.get("value") not in (".", "", None)
    ]


def _fetch_bls_series(
    series_ids: list[str],
    *,
    start_year: int,
    end_year: int,
) -> dict[str, list[dict[str, str]]]:
    """Fetch multiple BLS series via POST. Returns {series_id: [{year, period, value}]}."""
    payload = {
        "seriesid": series_ids,
        "startyear": str(start_year),
        "endyear": str(end_year),
    }
    try:
        with httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": _UA}) as client:
            resp = client.post(_BLS_BASE, json=payload)
            if resp.status_code != 200:
                log.warning("BLS HTTP %d", resp.status_code)
                return {}
            data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("BLS error: %s", exc)
        return {}

    if data.get("status") != "REQUEST_SUCCEEDED":
        log.warning("BLS request failed: %s", data.get("message"))
        return {}

    result: dict[str, list[dict[str, str]]] = {}
    for series in data.get("Results", {}).get("series", []):
        sid = series.get("seriesID", "")
        observations = []
        for obs in series.get("data", []):
            year = obs.get("year", "")
            period = obs.get("period", "")
            value = obs.get("value", "")
            if value and period.startswith("M"):
                month = period[1:].zfill(2)
                observations.append(
                    {
                        "date": f"{year}-{month}-01",
                        "value": value,
                    }
                )
        result[sid] = observations
    return result


def _latest_value(series: list[dict[str, str]]) -> tuple[str, float | None]:
    """Get latest (date, value) from a series."""
    if not series:
        return ("N/A", None)
    latest = series[0]
    try:
        return (latest["date"], float(latest["value"]))
    except (ValueError, KeyError):
        return (latest.get("date", "N/A"), None)


def _compute_trend(series: list[dict[str, str]], n: int = 3) -> str:
    """Compute simple trend from last n observations."""
    vals = []
    for obs in series[: n + 1]:
        try:
            vals.append(float(obs["value"]))
        except (ValueError, KeyError):
            continue
    if len(vals) < 2:
        return "insufficient data"
    # vals[0] is most recent (desc order)
    if vals[0] > vals[-1] * 1.05:
        return "rising"
    if vals[0] < vals[-1] * 0.95:
        return "falling"
    return "stable"


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class JobPostingsTool(Tool):
    """Monitor labor market signals via JOLTS and BLS data."""

    name = "job_postings"
    description = (
        "Track US labor market signals: JOLTS job openings, quits, hires, "
        "layoffs. Sector-level breakdown reveals which parts of the economy "
        "are expanding/contracting. Composite view includes unemployment "
        "rate and initial claims."
    )
    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": sorted(VALID_MODES),
                "description": (
                    "jolts: headline JOLTS data (openings, quits, hires, layoffs). "
                    "labor_market: composite overview (JOLTS + unemployment + claims). "
                    "sector: sector-level JOLTS breakdown."
                ),
            },
            "months": {
                "type": "integer",
                "description": "Number of recent months to retrieve (default 12, max 60).",
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

        months = min(max(int(kwargs.get("months", 12)), 1), 60)

        if mode == "jolts":
            return self._jolts(months=months)
        if mode == "sector":
            return self._sector(months=months)
        return self._labor_market(months=months)

    # ── jolts mode ───────────────────────────────────────────────────

    def _jolts(self, *, months: int) -> ToolResult:
        cache_key = f"jolts_{months}"
        if self._cache:
            cached = self._cache.get("job_postings", cache_key)
            if cached is not None:
                return self._format_jolts(cached, from_cache=True)

        if not self._api_key:
            # Fallback to BLS direct API
            return self._jolts_via_bls(months=months, cache_key=cache_key)

        results: dict[str, list[dict[str, str]]] = {}
        for series_id in _JOLTS_SERIES:
            obs = _fetch_fred_series(series_id, self._api_key, limit=months)
            results[series_id] = obs

        if self._cache:
            self._cache.set("job_postings", cache_key, results, ttl=_CACHE_TTL)

        return self._format_jolts(results)

    def _jolts_via_bls(self, *, months: int, cache_key: str) -> ToolResult:
        """JOLTS via BLS Public API (no key needed)."""
        import datetime

        now = datetime.datetime.now(UTC)
        end_year = now.year
        start_year = max(end_year - (months // 12) - 1, end_year - 10)

        series_ids = list(_JOLTS_SERIES.keys())
        # BLS uses different IDs than FRED for JOLTS
        bls_ids = [
            "JTS000000000000000JOL",  # Openings
            "JTS000000000000000QUL",  # Quits
            "JTS000000000000000HIL",  # Hires
            "JTS000000000000000LDL",  # Layoffs
        ]
        bls_to_fred = dict(zip(bls_ids, series_ids))

        raw = _fetch_bls_series(bls_ids, start_year=start_year, end_year=end_year)

        results: dict[str, list[dict[str, str]]] = {}
        for bls_id, fred_id in bls_to_fred.items():
            obs = raw.get(bls_id, [])
            results[fred_id] = sorted(obs, key=lambda x: x.get("date", ""), reverse=True)[:months]

        if self._cache:
            self._cache.set("job_postings", cache_key, results, ttl=_CACHE_TTL)

        return self._format_jolts(results)

    def _format_jolts(self, results: dict[str, list[dict[str, str]]], *, from_cache: bool = False) -> ToolResult:
        tag = " (cached)" if from_cache else ""
        lines = [f"JOLTS Labor Market Summary{tag}", ""]

        summary: dict[str, Any] = {}
        for series_id, label in _JOLTS_SERIES.items():
            obs = results.get(series_id, [])
            date, value = _latest_value(obs)
            trend = _compute_trend(obs)
            val_str = f"{value:,.0f}K" if value is not None else "N/A"
            lines.append(f"  {label}: {val_str} ({date}) — trend: {trend}")
            summary[series_id] = {
                "label": label,
                "latest_date": date,
                "latest_value": value,
                "trend": trend,
            }

        # Compute quits/layoffs ratio (worker confidence indicator)
        quits_val = summary.get("JTSQUL", {}).get("latest_value")
        layoffs_val = summary.get("JTSLDR", {}).get("latest_value")
        if quits_val and layoffs_val and layoffs_val > 0:
            ratio = quits_val / layoffs_val
            lines.append("")
            lines.append(f"  Quits/Layoffs ratio: {ratio:.2f} (>2.0 = strong confidence, <1.5 = stress)")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "mode": "jolts",
                "summary": summary,
                "series": results,
            },
        )

    # ── sector mode ──────────────────────────────────────────────────

    def _sector(self, *, months: int) -> ToolResult:
        cache_key = f"sector_{months}"
        if self._cache:
            cached = self._cache.get("job_postings", cache_key)
            if cached is not None:
                return self._format_sector(cached, from_cache=True)

        import datetime

        now = datetime.datetime.now(UTC)
        end_year = now.year
        start_year = max(end_year - (months // 12) - 1, end_year - 3)

        series_ids = list(_SECTOR_SERIES.keys())
        raw = _fetch_bls_series(series_ids, start_year=start_year, end_year=end_year)

        results: dict[str, list[dict[str, str]]] = {}
        for sid in series_ids:
            obs = raw.get(sid, [])
            results[sid] = sorted(obs, key=lambda x: x.get("date", ""), reverse=True)[:months]

        if self._cache:
            self._cache.set("job_postings", cache_key, results, ttl=_CACHE_TTL)

        return self._format_sector(results)

    def _format_sector(self, results: dict[str, list[dict[str, str]]], *, from_cache: bool = False) -> ToolResult:
        tag = " (cached)" if from_cache else ""
        lines = [f"JOLTS Sector-Level Job Openings{tag}", ""]

        summary: dict[str, Any] = {}
        for series_id, label in _SECTOR_SERIES.items():
            obs = results.get(series_id, [])
            date, value = _latest_value(obs)
            trend = _compute_trend(obs)
            val_str = f"{value:,.0f}K" if value is not None else "N/A"
            lines.append(f"  {label}: {val_str} ({date}) — {trend}")
            summary[series_id] = {
                "label": label,
                "latest_value": value,
                "trend": trend,
            }

        # Identify strongest/weakest sectors
        ranked = [(sid, s) for sid, s in summary.items() if s.get("latest_value") is not None]
        ranked.sort(key=lambda x: x[1]["latest_value"] or 0, reverse=True)

        if len(ranked) >= 2:
            lines.append("")
            top = ranked[0][1]
            bottom = ranked[-1][1]
            lines.append(f"  Strongest: {top['label']} ({top['latest_value']:,.0f}K)")
            lines.append(f"  Weakest: {bottom['label']} ({bottom['latest_value']:,.0f}K)")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "mode": "sector",
                "summary": summary,
                "series": results,
            },
        )

    # ── labor_market mode ────────────────────────────────────────────

    def _labor_market(self, *, months: int) -> ToolResult:
        if not self._api_key:
            return ToolResult(
                success=False,
                output=(
                    "FRED API key required for labor_market mode. "
                    "Set TIRRA_FRED_API_KEY in .env. "
                    "Use 'jolts' mode (no key needed) for basic JOLTS data."
                ),
            )

        cache_key = f"labor_market_{months}"
        if self._cache:
            cached = self._cache.get("job_postings", cache_key)
            if cached is not None:
                return self._format_labor_market(cached, from_cache=True)

        results: dict[str, list[dict[str, str]]] = {}
        for series_id in _LABOR_SERIES:
            obs = _fetch_fred_series(series_id, self._api_key, limit=months)
            results[series_id] = obs

        if self._cache:
            self._cache.set("job_postings", cache_key, results, ttl=_CACHE_TTL)

        return self._format_labor_market(results)

    def _format_labor_market(self, results: dict[str, list[dict[str, str]]], *, from_cache: bool = False) -> ToolResult:
        tag = " (cached)" if from_cache else ""
        lines = [f"Composite Labor Market Overview{tag}", ""]

        summary: dict[str, Any] = {}
        for series_id, label in _LABOR_SERIES.items():
            obs = results.get(series_id, [])
            date, value = _latest_value(obs)
            trend = _compute_trend(obs)
            if series_id == "UNRATE":
                val_str = f"{value:.1f}%" if value is not None else "N/A"
            elif series_id == "ICSA":
                val_str = f"{value:,.0f}" if value is not None else "N/A"
            else:
                val_str = f"{value:,.0f}K" if value is not None else "N/A"
            lines.append(f"  {label}: {val_str} ({date}) — {trend}")
            summary[series_id] = {
                "label": label,
                "latest_date": date,
                "latest_value": value,
                "trend": trend,
            }

        # Market tightness: openings / (unemployed workers)
        # unemployed ≈ payrolls * (UNRATE / 100) is rough approximation
        openings = summary.get("JTSJOL", {}).get("latest_value")
        unrate = summary.get("UNRATE", {}).get("latest_value")
        payrolls = summary.get("PAYEMS", {}).get("latest_value")
        if openings and unrate and payrolls and unrate > 0:
            unemployed = payrolls * (unrate / 100)
            tightness = openings / unemployed if unemployed > 0 else 0
            lines.append("")
            lines.append(f"  Market tightness (openings/unemployed): {tightness:.2f} (>1.5 = very tight, <0.8 = loose)")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "mode": "labor_market",
                "summary": summary,
                "series": results,
            },
        )
