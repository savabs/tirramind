"""
Tool: Migration & Refugee Flows Monitor — UNHCR + World Bank

Global displacement tracking and remittance flow monitoring from two
authoritative free sources.

Sources:
  UNHCR Refugee Statistics API v1  — population stocks, asylum decisions
    by country of origin/asylum.  Annual data, no authentication.
  World Bank Open Data  — personal remittances received (indicator
    BX.TRF.PWKR.CD.DT).  Annual data, no authentication.

Modes:
  displacement  — UNHCR: refugee, IDP, asylum seeker, stateless populations
                  by country of asylum or origin.  Tracks stock levels.
  asylum        — UNHCR: asylum decision outcomes (recognized, rejected,
                  closed) by country pair.  Acceptance rate = policy signal.
  remittances   — World Bank: personal remittances received (current US$)
                  by country.  Tracks diaspora economics and labor migration.

Signal theory:
  - Displacement stock surge (>20% YoY) in a country of asylum = regional crisis
  - Asylum acceptance rate drop = policy tightening / political shift
  - Remittance surge = new diaspora wave or crisis-driven transfers
  - Remittance collapse = banking sanctions, corridor disruption, or source-
    country recession
  - IDP stock spike + refugee outflow = conflict escalation (displacement
    moves from internal to cross-border)

Market relevance:
  Mass displacement → housing/rental pressure in host countries, labor market
  shifts, political destabilization (anti-immigrant policy), humanitarian spend,
  remittance corridor economics (Western Union, Wise), origin-country GDP
  collapse, host-country fiscal burden.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

_UA = "TirraMind/0.1"
_TIMEOUT = 20
_CACHE_TTL = 43200  # 12 hours — annual data changes infrequently

_UNHCR_BASE = "https://api.unhcr.org/population/v1"
_WB_BASE = "https://api.worldbank.org/v2"

WB_REMITTANCE_INDICATOR = "BX.TRF.PWKR.CD.DT"

VALID_MODES = {"displacement", "asylum", "remittances"}


class MigrationFlowsTool(Tool):
    """Monitor global migration and refugee flows via UNHCR + World Bank."""

    def __init__(self, cache: DataCache | None = None) -> None:
        self._cache = cache

    @property
    def name(self) -> str:
        return "migration_flows"

    @property
    def description(self) -> str:
        return (
            "Monitor global migration and refugee flows — UNHCR displacement "
            "stocks (refugees, IDPs, asylum seekers, stateless), asylum decision "
            "outcomes, and World Bank remittance inflows. Detects displacement "
            "surges, policy shifts, and diaspora economic disruptions."
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
                        "asylum: asylum decision outcomes by country. "
                        "displacement: refugee/IDP/stateless populations. "
                        "remittances: personal remittance inflows (World Bank)."
                    ),
                },
                "country": {
                    "type": "string",
                    "description": (
                        "ISO 3166-1 alpha-3 code for displacement/asylum (e.g. "
                        "'TUR', 'DEU', 'SYR'). ISO alpha-2 for remittances "
                        "(e.g. 'PH', 'MX', 'IN'). Omit for global aggregates."
                    ),
                },
                "year": {
                    "type": "integer",
                    "description": "Year to query (default: most recent available).",
                },
                "role": {
                    "type": "string",
                    "enum": ["asylum", "origin"],
                    "description": (
                        "For displacement/asylum: 'asylum' = country hosting "
                        "refugees, 'origin' = country people fled from. "
                        "Default: 'asylum'."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default: 20, max: 100).",
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

        country = (kwargs.get("country") or "").strip().upper()
        year = kwargs.get("year")
        limit = min(kwargs.get("limit") or 20, 100)

        if mode == "displacement":
            return self._handle_displacement(country, year, kwargs.get("role", "asylum"), limit)
        elif mode == "asylum":
            return self._handle_asylum(country, year, kwargs.get("role", "asylum"), limit)
        else:
            return self._handle_remittances(country, year, limit)

    # ── Mode handlers ───────────────────────────────────────

    def _handle_displacement(
        self, country: str, year: int | None, role: str, limit: int,
    ) -> ToolResult:
        cache_key = f"migration:displacement:{country}:{year}:{role}:{limit}"
        if self._cache:
            hit = self._cache.get(cache_key)
            if hit is not None:
                return ToolResult(success=True, output=hit["output"], data=hit["data"])

        params: dict[str, Any] = {"limit": limit, "page": 1}
        if year:
            params["year"] = year
        if country:
            if role == "origin":
                params["coo"] = country
            else:
                params["coa"] = country

        records, err = _fetch_unhcr(f"{_UNHCR_BASE}/population/", params)
        if err:
            return ToolResult(success=False, output=err)

        parsed = _parse_population_records(records)
        signals = _compute_displacement_signals(parsed)
        summary = _format_displacement_summary(parsed, signals, country, year, role)

        result_data = {
            "records": parsed,
            "count": len(parsed),
            "country": country,
            "year": year,
            "role": role,
            "signals": signals,
        }

        if self._cache:
            self._cache.set(cache_key, {"output": summary, "data": result_data}, ttl=_CACHE_TTL)

        return ToolResult(success=True, output=summary, data=result_data)

    def _handle_asylum(
        self, country: str, year: int | None, role: str, limit: int,
    ) -> ToolResult:
        cache_key = f"migration:asylum:{country}:{year}:{role}:{limit}"
        if self._cache:
            hit = self._cache.get(cache_key)
            if hit is not None:
                return ToolResult(success=True, output=hit["output"], data=hit["data"])

        params: dict[str, Any] = {"limit": limit, "page": 1}
        if year:
            params["year"] = year
        if country:
            if role == "origin":
                params["coo"] = country
            else:
                params["coa"] = country

        records, err = _fetch_unhcr(f"{_UNHCR_BASE}/asylum-decisions/", params)
        if err:
            return ToolResult(success=False, output=err)

        parsed = _parse_asylum_records(records)
        signals = _compute_asylum_signals(parsed)
        summary = _format_asylum_summary(parsed, signals, country, year, role)

        result_data = {
            "records": parsed,
            "count": len(parsed),
            "country": country,
            "year": year,
            "role": role,
            "signals": signals,
        }

        if self._cache:
            self._cache.set(cache_key, {"output": summary, "data": result_data}, ttl=_CACHE_TTL)

        return ToolResult(success=True, output=summary, data=result_data)

    def _handle_remittances(
        self, country: str, year: int | None, limit: int,
    ) -> ToolResult:
        if not country:
            return ToolResult(
                success=False,
                output="'country' is required for remittances mode. Use ISO alpha-2 (e.g. 'PH', 'MX').",
            )
        if len(country) < 2 or len(country) > 3:
            return ToolResult(
                success=False,
                output=f"Invalid country code '{country}'. Must be 2-3 characters.",
            )

        from datetime import datetime, timezone
        now_year = datetime.now(timezone.utc).year
        end_year = year or now_year
        start_year = end_year - 9  # 10 years of data

        cache_key = f"migration:remittances:{country}:{start_year}-{end_year}"
        if self._cache:
            hit = self._cache.get(cache_key)
            if hit is not None:
                return ToolResult(success=True, output=hit["output"], data=hit["data"])

        records, err = _fetch_wb_remittances(country, start_year, end_year, limit)
        if err:
            return ToolResult(success=False, output=err)

        signals = _compute_remittance_signals(records)
        summary = _format_remittance_summary(records, signals, country, start_year, end_year)

        result_data = {
            "records": records,
            "count": len(records),
            "country": country,
            "start_year": start_year,
            "end_year": end_year,
            "signals": signals,
        }

        if self._cache:
            self._cache.set(cache_key, {"output": summary, "data": result_data}, ttl=_CACHE_TTL)

        return ToolResult(success=True, output=summary, data=result_data)


# ── UNHCR fetch (module-level for testability) ──────────────────


def _fetch_unhcr(
    url: str, params: dict,
) -> tuple[list[dict], str | None]:
    """Fetch from UNHCR API. Returns (items, error_or_None)."""
    try:
        with httpx.Client(
            timeout=_TIMEOUT, headers={"User-Agent": _UA},
        ) as client:
            resp = client.get(url, params=params)
    except httpx.TimeoutException:
        return [], "UNHCR API request timed out."
    except httpx.HTTPError as exc:
        return [], f"HTTP error: {exc}"

    if resp.status_code == 429:
        return [], "UNHCR API rate limit reached. Retry later."
    if resp.status_code != 200:
        return [], f"UNHCR API returned HTTP {resp.status_code}"

    try:
        body = resp.json()
    except Exception:
        return [], "Failed to parse UNHCR API response."

    items = body.get("items", [])
    return items, None


def _safe_int(val: Any) -> int:
    """Convert UNHCR values to int. Handles int, str '0', str '-', None."""
    if val is None or val == "-" or val == "":
        return 0
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def _parse_population_records(items: list[dict]) -> list[dict]:
    """Parse UNHCR population items into normalized records."""
    records = []
    for item in items:
        records.append({
            "year": _safe_int(item.get("year")),
            "coo": item.get("coo", ""),
            "coo_name": item.get("coo_name", ""),
            "coo_iso": item.get("coo_iso", ""),
            "coa": item.get("coa", ""),
            "coa_name": item.get("coa_name", ""),
            "coa_iso": item.get("coa_iso", ""),
            "refugees": _safe_int(item.get("refugees")),
            "asylum_seekers": _safe_int(item.get("asylum_seekers")),
            "returned_refugees": _safe_int(item.get("returned_refugees")),
            "idps": _safe_int(item.get("idps")),
            "returned_idps": _safe_int(item.get("returned_idps")),
            "stateless": _safe_int(item.get("stateless")),
            "ooc": _safe_int(item.get("ooc")),
            "oip": _safe_int(item.get("oip")),
            "hst": _safe_int(item.get("hst")),
        })
    return records


def _parse_asylum_records(items: list[dict]) -> list[dict]:
    """Parse UNHCR asylum-decisions items into normalized records."""
    records = []
    for item in items:
        records.append({
            "year": _safe_int(item.get("year")),
            "coo": item.get("coo", ""),
            "coo_name": item.get("coo_name", ""),
            "coo_iso": item.get("coo_iso", ""),
            "coa": item.get("coa", ""),
            "coa_name": item.get("coa_name", ""),
            "coa_iso": item.get("coa_iso", ""),
            "procedure_type": item.get("procedure_type", ""),
            "dec_level": item.get("dec_level", ""),
            "dec_recognized": _safe_int(item.get("dec_recognized")),
            "dec_other": _safe_int(item.get("dec_other")),
            "dec_rejected": _safe_int(item.get("dec_rejected")),
            "dec_closed": _safe_int(item.get("dec_closed")),
            "dec_total": _safe_int(item.get("dec_total")),
        })
    return records


# ── World Bank fetch ────────────────────────────────────────────


def _fetch_wb_remittances(
    country: str, start_year: int, end_year: int, limit: int,
) -> tuple[list[dict], str | None]:
    """Fetch remittance data from World Bank."""
    url = f"{_WB_BASE}/country/{country}/indicator/{WB_REMITTANCE_INDICATOR}"
    params = {
        "format": "json",
        "date": f"{start_year}:{end_year}",
        "per_page": str(min(limit, 100)),
    }

    try:
        with httpx.Client(
            timeout=_TIMEOUT, headers={"User-Agent": _UA},
        ) as client:
            resp = client.get(url, params=params)
    except httpx.TimeoutException:
        return [], "World Bank API request timed out."
    except httpx.HTTPError as exc:
        return [], f"HTTP error: {exc}"

    if resp.status_code == 429:
        return [], "World Bank API rate limit reached. Retry later."
    if resp.status_code != 200:
        return [], f"World Bank API returned HTTP {resp.status_code}"

    try:
        body = resp.json()
    except Exception:
        return [], "Failed to parse World Bank API response."

    if not isinstance(body, list) or len(body) < 2 or body[1] is None:
        return [], None  # No data available (not an error)

    records = _parse_wb_records(body[1])
    return records, None


def _parse_wb_records(raw: list) -> list[dict]:
    """Parse World Bank indicator response into records."""
    records = []
    for entry in raw:
        year = entry.get("date", "")
        value = entry.get("value")
        if value is None:
            continue
        try:
            value = float(value)
        except (ValueError, TypeError):
            continue
        records.append({
            "year": year,
            "value": value,
            "country": entry.get("country", {}).get("id", ""),
            "country_name": entry.get("country", {}).get("value", ""),
            "indicator": entry.get("indicator", {}).get("id", ""),
        })

    # Sort chronologically (WB returns newest-first)
    records.sort(key=lambda r: r["year"])
    return records


# ── Signal computation ──────────────────────────────────────────


def _compute_displacement_signals(records: list[dict]) -> dict:
    """Compute signals from displacement data."""
    if not records:
        return {"status": "NO_DATA", "alert": None}

    total_refugees = sum(r["refugees"] for r in records)
    total_asylum = sum(r["asylum_seekers"] for r in records)
    total_idps = sum(r["idps"] for r in records)
    total_stateless = sum(r["stateless"] for r in records)
    total_displaced = total_refugees + total_asylum + total_idps

    signals: dict[str, Any] = {
        "total_refugees": total_refugees,
        "total_asylum_seekers": total_asylum,
        "total_idps": total_idps,
        "total_stateless": total_stateless,
        "total_displaced": total_displaced,
        "record_count": len(records),
    }

    # Alert thresholds (on aggregate for the query)
    if total_displaced > 10_000_000:
        signals["alert"] = "CRITICAL — >10M displaced persons"
    elif total_displaced > 1_000_000:
        signals["alert"] = "WARNING — >1M displaced persons"
    elif total_displaced > 100_000:
        signals["alert"] = "NOTICE — significant displacement"
    else:
        signals["alert"] = None

    # Composition breakdown
    if total_displaced > 0:
        signals["refugee_pct"] = round(100 * total_refugees / total_displaced, 1)
        signals["idp_pct"] = round(100 * total_idps / total_displaced, 1)
        signals["asylum_pct"] = round(100 * total_asylum / total_displaced, 1)
    else:
        signals["refugee_pct"] = 0
        signals["idp_pct"] = 0
        signals["asylum_pct"] = 0

    return signals


def _compute_asylum_signals(records: list[dict]) -> dict:
    """Compute signals from asylum decision data."""
    if not records:
        return {"status": "NO_DATA", "alert": None}

    total_recognized = sum(r["dec_recognized"] for r in records)
    total_rejected = sum(r["dec_rejected"] for r in records)
    total_closed = sum(r["dec_closed"] for r in records)
    total_other = sum(r["dec_other"] for r in records)
    total_decisions = sum(r["dec_total"] for r in records)

    signals: dict[str, Any] = {
        "total_recognized": total_recognized,
        "total_rejected": total_rejected,
        "total_closed": total_closed,
        "total_other": total_other,
        "total_decisions": total_decisions,
        "record_count": len(records),
    }

    # Acceptance rate
    substantive = total_recognized + total_rejected
    if substantive > 0:
        acceptance_rate = round(100 * total_recognized / substantive, 1)
        signals["acceptance_rate"] = acceptance_rate
        if acceptance_rate < 20:
            signals["alert"] = f"RESTRICTIVE — only {acceptance_rate}% acceptance rate"
        elif acceptance_rate > 80:
            signals["alert"] = f"LIBERAL — {acceptance_rate}% acceptance rate"
        else:
            signals["alert"] = None
    else:
        signals["acceptance_rate"] = None
        signals["alert"] = None

    # Closure rate (cases abandoned/dismissed without decision)
    if total_decisions > 0:
        signals["closure_rate"] = round(100 * total_closed / total_decisions, 1)
    else:
        signals["closure_rate"] = None

    return signals


def _compute_remittance_signals(records: list[dict]) -> dict:
    """Compute signals from remittance data."""
    if not records:
        return {"status": "NO_DATA", "alert": None}

    values = [r["value"] for r in records]
    latest = values[-1]
    avg = sum(values) / len(values)
    peak = max(values)

    signals: dict[str, Any] = {
        "latest_value": latest,
        "latest_value_billions": round(latest / 1e9, 2),
        "period_average": round(avg, 2),
        "period_average_billions": round(avg / 1e9, 2),
        "period_peak": peak,
        "total_years": len(values),
    }

    # YoY change if we have at least 2 data points
    if len(values) >= 2:
        prior = values[-2]
        if prior > 0:
            yoy_pct = round(100 * (latest - prior) / prior, 1)
            signals["yoy_change_pct"] = yoy_pct
            if yoy_pct < -20:
                signals["alert"] = f"CRITICAL — remittances dropped {abs(yoy_pct)}% YoY"
            elif yoy_pct < -10:
                signals["alert"] = f"WARNING — remittances dropped {abs(yoy_pct)}% YoY"
            elif yoy_pct > 30:
                signals["alert"] = f"SURGE — remittances up {yoy_pct}% YoY"
            else:
                signals["alert"] = None
        else:
            signals["yoy_change_pct"] = None
            signals["alert"] = None
    else:
        signals["yoy_change_pct"] = None
        signals["alert"] = None

    # Trend: compare last 3 years average to prior 3 years
    if len(values) >= 6:
        recent = values[-3:]
        prior_3 = values[-6:-3]
        recent_avg = sum(recent) / len(recent)
        prior_avg = sum(prior_3) / len(prior_3)
        if prior_avg > 0:
            trend_ratio = round(recent_avg / prior_avg, 2)
            signals["trend_ratio"] = trend_ratio
            if trend_ratio > 1.2:
                signals["trend"] = "GROWING"
            elif trend_ratio < 0.8:
                signals["trend"] = "DECLINING"
            else:
                signals["trend"] = "STABLE"
        else:
            signals["trend"] = "NO_BASELINE"
            signals["trend_ratio"] = None
    else:
        signals["trend"] = "INSUFFICIENT_DATA"
        signals["trend_ratio"] = None

    return signals


# ── Formatting ──────────────────────────────────────────────────


def _format_displacement_summary(
    records: list[dict], signals: dict, country: str, year: int | None, role: str,
) -> str:
    """Format displacement summary."""
    scope = f"{country} ({role})" if country else "Global"
    yr = f" — {year}" if year else ""
    lines = [f"UNHCR Displacement — {scope}{yr}"]
    lines.append(f"Records: {len(records)}")

    if not records:
        lines.append("No displacement data available.")
        return "\n".join(lines)

    def _fmt(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.0f}K"
        return str(n)

    lines.append(f"Total displaced: {_fmt(signals.get('total_displaced', 0))}")
    lines.append(f"  Refugees: {_fmt(signals.get('total_refugees', 0))} ({signals.get('refugee_pct', 0)}%)")
    lines.append(f"  IDPs: {_fmt(signals.get('total_idps', 0))} ({signals.get('idp_pct', 0)}%)")
    lines.append(f"  Asylum seekers: {_fmt(signals.get('total_asylum_seekers', 0))} ({signals.get('asylum_pct', 0)}%)")
    lines.append(f"  Stateless: {_fmt(signals.get('total_stateless', 0))}")

    alert = signals.get("alert")
    if alert:
        lines.append(f"⚠ {alert}")

    return "\n".join(lines)


def _format_asylum_summary(
    records: list[dict], signals: dict, country: str, year: int | None, role: str,
) -> str:
    """Format asylum decision summary."""
    scope = f"{country} ({role})" if country else "Global"
    yr = f" — {year}" if year else ""
    lines = [f"UNHCR Asylum Decisions — {scope}{yr}"]
    lines.append(f"Records: {len(records)}")

    if not records:
        lines.append("No asylum decision data available.")
        return "\n".join(lines)

    lines.append(f"Total decisions: {signals.get('total_decisions', 0):,}")
    lines.append(f"  Recognized: {signals.get('total_recognized', 0):,}")
    lines.append(f"  Rejected: {signals.get('total_rejected', 0):,}")
    lines.append(f"  Closed: {signals.get('total_closed', 0):,}")
    lines.append(f"  Other: {signals.get('total_other', 0):,}")

    acc = signals.get("acceptance_rate")
    if acc is not None:
        lines.append(f"Acceptance rate: {acc}%")

    closure = signals.get("closure_rate")
    if closure is not None:
        lines.append(f"Closure rate: {closure}%")

    alert = signals.get("alert")
    if alert:
        lines.append(f"⚠ {alert}")

    return "\n".join(lines)


def _format_remittance_summary(
    records: list[dict], signals: dict, country: str, start: int, end: int,
) -> str:
    """Format remittance summary."""
    lines = [f"World Bank Remittances — {country} ({start}-{end})"]
    lines.append(f"Records: {len(records)} years")

    if not records:
        lines.append("No remittance data available.")
        return "\n".join(lines)

    latest_b = signals.get("latest_value_billions", 0)
    avg_b = signals.get("period_average_billions", 0)
    lines.append(f"Latest: ${latest_b}B")
    lines.append(f"Period average: ${avg_b}B")

    yoy = signals.get("yoy_change_pct")
    if yoy is not None:
        lines.append(f"YoY change: {yoy:+.1f}%")

    trend = signals.get("trend", "N/A")
    lines.append(f"Trend: {trend}")

    alert = signals.get("alert")
    if alert:
        lines.append(f"⚠ {alert}")

    # Recent values
    lines.append("")
    lines.append("Recent data:")
    for r in records[-5:]:
        val_b = round(r["value"] / 1e9, 2)
        lines.append(f"  {r['year']}: ${val_b}B")

    return "\n".join(lines)
