"""
Tool: Food Security Monitor — World Bank Agricultural Indicators

Global agricultural production, cereal yields, and food trade dependency
from the World Bank Open Data API.  Free, no authentication, 200+ countries.

Sources:
  World Bank Development Indicators  — production indices (base 2014-2016),
    cereal yield/area, food trade shares.  Updated annually.

Modes:
  production   — Food/crop/livestock production indices by country/year.
                 Index base = 2014-2016.  Declining index = supply stress.
  cereal_yield — Cereal yield (kg/ha) and area harvested (hectares).
                 Physical measurement of agricultural productive capacity.
  food_trade   — Food import/export share of total merchandise trade.
                 Import-dependent nations most vulnerable to price shocks.

Signal theory:
  - Production index declining 2+ consecutive years = structural supply problem
  - Cereal yield drop >10 pct from 5-yr average = drought / pest / conflict
  - Food import share >30 pct + production decline = food crisis vulnerability
  - Livestock diverging from crop production = feed shortage
  - Multi-country production declines in same year = global shortage → price spike

Market relevance:
  Food security → commodity prices (wheat/corn/soy/rice), fertilizer demand,
  agricultural stocks, social instability in import-dependent nations (Egypt,
  Lebanon), migration/political risk, shipping volumes, central bank inflation
  response (food CPI weight 10-70 pct by country).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

_UA = "TirraMind/0.1"
_TIMEOUT = 20
_CACHE_TTL = 43200  # 12 hours — WB data updates infrequently

_WB_BASE = "https://api.worldbank.org/v2"

# ── World Bank indicator codes ──────────────────────────────────

_PRODUCTION_INDICATORS: dict[str, str] = {
    "food": "AG.PRD.FOOD.XD",
    "crop": "AG.PRD.CROP.XD",
    "livestock": "AG.PRD.LVSK.XD",
}

_CEREAL_INDICATORS: dict[str, str] = {
    "yield_kg_per_ha": "AG.YLD.CREL.KG",
    "area_hectares": "AG.LND.CREL.HA",
}

_TRADE_INDICATORS: dict[str, str] = {
    "food_import_pct": "TM.VAL.FOOD.ZS.UN",
    "food_export_pct": "TX.VAL.FOOD.ZS.UN",
}

VALID_MODES = {"production", "cereal_yield", "food_trade"}

# Reference sets for context (not enforced as validation)
MAJOR_PRODUCERS = {"US", "CN", "IN", "BR", "RU", "AR", "FR", "AU", "CA", "DE"}
VULNERABLE_IMPORTERS = {"EG", "LB", "YE", "SD", "SO", "AF", "HT", "BD", "PK", "NG"}


class FoodSecurityTool(Tool):
    """Monitor global food security via World Bank agricultural indicators."""

    def __init__(self, cache: DataCache | None = None) -> None:
        self._cache = cache

    @property
    def name(self) -> str:
        return "food_security"

    @property
    def description(self) -> str:
        return (
            "Monitor global food security — crop/food/livestock production indices, "
            "cereal yields, and food trade dependency from World Bank Open Data. "
            "Detects agricultural supply stress, drought impact, and food crisis "
            "vulnerability across 200+ countries."
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
                        "production: food/crop/livestock production indices. "
                        "cereal_yield: cereal yield (kg/ha) and area harvested. "
                        "food_trade: food import/export as pct of total trade."
                    ),
                },
                "country": {
                    "type": "string",
                    "description": (
                        "ISO 3166-1 alpha-2 country code (e.g. 'US', 'CN', 'IN', "
                        "'BR', 'EG'). Use 'WLD' for world aggregate."
                    ),
                },
                "start_year": {
                    "type": "integer",
                    "description": "Start year for data range (default: 5 years ago).",
                },
                "end_year": {
                    "type": "integer",
                    "description": "End year for data range (default: current year).",
                },
                "indicator": {
                    "type": "string",
                    "description": (
                        "For production: 'food', 'crop', or 'livestock' (default: 'food'). "
                        "For cereal_yield: 'yield_kg_per_ha' or 'area_hectares' "
                        "(default: 'yield_kg_per_ha'). "
                        "For food_trade: 'food_import_pct' or 'food_export_pct' "
                        "(default: 'food_import_pct')."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results per page (default: 50, max: 100).",
                },
            },
            "required": ["mode", "country"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        mode = kwargs.get("mode", "")
        if mode not in VALID_MODES:
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Must be one of: {sorted(VALID_MODES)}",
            )

        country = (kwargs.get("country") or "").strip().upper()
        if not country:
            return ToolResult(
                success=False,
                output="'country' is required. Use ISO alpha-2 code (e.g. 'US') or 'WLD'.",
            )
        if len(country) < 2 or len(country) > 3:
            return ToolResult(
                success=False,
                output=f"Invalid country code '{country}'. Must be 2-3 characters.",
            )

        now = datetime.now(timezone.utc)
        end_year = kwargs.get("end_year") or now.year
        start_year = kwargs.get("start_year") or (end_year - 5)
        limit = min(kwargs.get("limit") or 50, 100)

        if start_year > end_year:
            return ToolResult(
                success=False,
                output=f"start_year ({start_year}) cannot be after end_year ({end_year}).",
            )

        if mode == "production":
            return self._handle_production(
                country, start_year, end_year,
                kwargs.get("indicator", "food"), limit,
            )
        elif mode == "cereal_yield":
            return self._handle_cereal_yield(
                country, start_year, end_year,
                kwargs.get("indicator", "yield_kg_per_ha"), limit,
            )
        else:
            return self._handle_food_trade(
                country, start_year, end_year,
                kwargs.get("indicator", "food_import_pct"), limit,
            )

    # ── Mode handlers ───────────────────────────────────────

    def _handle_production(
        self, country: str, start: int, end: int, indicator: str, limit: int,
    ) -> ToolResult:
        ind_code = _PRODUCTION_INDICATORS.get(indicator)
        if not ind_code:
            return ToolResult(
                success=False,
                output=(
                    f"Invalid production indicator '{indicator}'. "
                    f"Must be one of: {sorted(_PRODUCTION_INDICATORS)}"
                ),
            )
        return self._fetch_indicator(
            country, ind_code, start, end, limit, f"production:{indicator}",
        )

    def _handle_cereal_yield(
        self, country: str, start: int, end: int, indicator: str, limit: int,
    ) -> ToolResult:
        ind_code = _CEREAL_INDICATORS.get(indicator)
        if not ind_code:
            return ToolResult(
                success=False,
                output=(
                    f"Invalid cereal indicator '{indicator}'. "
                    f"Must be one of: {sorted(_CEREAL_INDICATORS)}"
                ),
            )
        return self._fetch_indicator(
            country, ind_code, start, end, limit, f"cereal_yield:{indicator}",
        )

    def _handle_food_trade(
        self, country: str, start: int, end: int, indicator: str, limit: int,
    ) -> ToolResult:
        ind_code = _TRADE_INDICATORS.get(indicator)
        if not ind_code:
            return ToolResult(
                success=False,
                output=(
                    f"Invalid trade indicator '{indicator}'. "
                    f"Must be one of: {sorted(_TRADE_INDICATORS)}"
                ),
            )
        return self._fetch_indicator(
            country, ind_code, start, end, limit, f"food_trade:{indicator}",
        )

    # ── Core fetch ──────────────────────────────────────────

    def _fetch_indicator(
        self,
        country: str,
        indicator_code: str,
        start_year: int,
        end_year: int,
        limit: int,
        cache_label: str,
    ) -> ToolResult:
        cache_key = f"food_security:{cache_label}:{country}:{start_year}-{end_year}"
        if self._cache:
            hit = self._cache.get(cache_key)
            if hit is not None:
                return ToolResult(
                    success=True, output=hit["output"], data=hit["data"],
                )

        url = f"{_WB_BASE}/country/{country}/indicator/{indicator_code}"
        params = {
            "format": "json",
            "date": f"{start_year}:{end_year}",
            "per_page": str(limit),
        }

        try:
            with httpx.Client(
                timeout=_TIMEOUT, headers={"User-Agent": _UA},
            ) as client:
                resp = client.get(url, params=params)
        except httpx.TimeoutException:
            return ToolResult(
                success=False, output="World Bank API request timed out.",
            )
        except httpx.HTTPError as exc:
            return ToolResult(success=False, output=f"HTTP error: {exc}")

        if resp.status_code == 429:
            return ToolResult(
                success=False,
                output="World Bank API rate limit reached. Retry later.",
            )
        if resp.status_code != 200:
            return ToolResult(
                success=False,
                output=f"World Bank API returned HTTP {resp.status_code}",
            )

        try:
            body = resp.json()
        except Exception:
            return ToolResult(
                success=False,
                output="Failed to parse World Bank API response.",
            )

        # World Bank returns [metadata, data_array] or [metadata, null]
        if not isinstance(body, list) or len(body) < 2 or body[1] is None:
            return ToolResult(
                success=True,
                output=(
                    f"No data for {country} / {indicator_code} "
                    f"({start_year}-{end_year})."
                ),
                data={
                    "records": [],
                    "country": country,
                    "indicator": indicator_code,
                },
            )

        records = _parse_wb_records(body[1], country, indicator_code)
        records.sort(key=lambda r: r.get("year", ""))

        valid = [r for r in records if r["value"] is not None]
        signals = _compute_signals(valid, cache_label)
        summary = _format_summary(records, valid, signals, country, cache_label)

        result_data = {
            "records": records,
            "valid_count": len(valid),
            "total_count": len(records),
            "country": country,
            "indicator": indicator_code,
            "signals": signals,
        }

        if self._cache:
            self._cache.set(
                cache_key,
                {"output": summary, "data": result_data},
                ttl=_CACHE_TTL,
            )

        return ToolResult(success=True, output=summary, data=result_data)


# ── Helpers (module-level for testability) ──────────────────────


def _parse_wb_records(
    entries: list[dict], country: str, indicator_code: str,
) -> list[dict]:
    """Parse World Bank API response entries into normalized records."""
    records = []
    for entry in entries:
        records.append({
            "country": entry.get("country", {}).get("id", country),
            "country_name": entry.get("country", {}).get("value", ""),
            "year": entry.get("date", ""),
            "value": entry.get("value"),
            "indicator": entry.get("indicator", {}).get("id", indicator_code),
            "indicator_name": entry.get("indicator", {}).get("value", ""),
        })
    return records


def _compute_signals(valid: list[dict], label: str) -> dict[str, Any]:
    """Derive trend and stress signals from time series."""
    signals: dict[str, Any] = {}
    if len(valid) < 2:
        return signals

    values = [r["value"] for r in valid]
    latest = values[-1]
    previous = values[-2]

    # Year-over-year change
    if previous and previous != 0:
        yoy = ((latest - previous) / abs(previous)) * 100
        signals["yoy_change_pct"] = round(yoy, 2)

    # Period average
    if len(values) >= 3:
        avg = sum(values) / len(values)
        signals["period_average"] = round(avg, 2)
        if avg != 0:
            dev = ((latest - avg) / abs(avg)) * 100
            signals["deviation_from_avg_pct"] = round(dev, 2)

    # Trend — consecutive increases or decreases
    consecutive = 0
    direction = None
    for i in range(len(values) - 1, 0, -1):
        if values[i] > values[i - 1]:
            if direction is None:
                direction = "up"
            if direction == "up":
                consecutive += 1
            else:
                break
        elif values[i] < values[i - 1]:
            if direction is None:
                direction = "down"
            if direction == "down":
                consecutive += 1
            else:
                break
        else:
            break

    if direction:
        signals["trend_direction"] = direction
        signals["consecutive_years"] = consecutive

    # Stress alerts
    if "production" in label:
        if (
            signals.get("trend_direction") == "down"
            and signals.get("consecutive_years", 0) >= 2
        ):
            signals["stress_alert"] = (
                "Production declining 2+ consecutive years"
            )
        elif signals.get("deviation_from_avg_pct", 0) < -10:
            signals["stress_alert"] = (
                f"Production {signals['deviation_from_avg_pct']:.1f}% "
                f"below period average"
            )

    if "food_import" in label:
        if latest is not None and latest > 30:
            signals["vulnerability"] = "high"
            signals["vulnerability_note"] = (
                f"Food imports = {latest:.1f}% of merchandise imports"
            )
        elif latest is not None and latest > 20:
            signals["vulnerability"] = "moderate"

    return signals


def _format_summary(
    records: list[dict],
    valid: list[dict],
    signals: dict,
    country: str,
    label: str,
) -> str:
    """Format human-readable summary."""
    parts = [f"Food Security — {label} — {country}"]
    parts.append(f"Records: {len(valid)} with data / {len(records)} total")

    if valid:
        latest = valid[-1]
        parts.append(f"Latest: {latest['year']} = {latest['value']}")
        if latest.get("indicator_name"):
            parts.append(f"Indicator: {latest['indicator_name']}")

    if signals:
        if "yoy_change_pct" in signals:
            arrow = "↑" if signals["yoy_change_pct"] > 0 else "↓"
            parts.append(f"YoY change: {arrow} {signals['yoy_change_pct']}%")
        if "deviation_from_avg_pct" in signals:
            parts.append(
                f"Deviation from period avg: {signals['deviation_from_avg_pct']}%"
            )
        if "trend_direction" in signals:
            parts.append(
                f"Trend: {signals['trend_direction']} "
                f"for {signals['consecutive_years']} year(s)"
            )
        if "stress_alert" in signals:
            parts.append(f"⚠ STRESS: {signals['stress_alert']}")
        if "vulnerability" in signals:
            parts.append(f"Import vulnerability: {signals['vulnerability']}")
            if "vulnerability_note" in signals:
                parts.append(f"  {signals['vulnerability_note']}")

    if valid:
        parts.append("\nRecent values:")
        for r in valid[-6:]:
            parts.append(f"  {r['year']}: {r['value']}")

    return "\n".join(parts)
