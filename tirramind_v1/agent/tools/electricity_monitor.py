"""
Tool: Electricity Monitor — US-Wide Grid Demand, Generation, Interchange

EIA API v2  https://api.eia.gov/v2/  (free, API key required)

Expands electricity observation beyond NYISO to ALL US balancing authorities
(RTOs/ISOs: PJM, CAISO, ERCOT, MISO, SPP, ISO-NE, etc.).

Modes
-----
demand        Hourly electricity demand by balancing authority.
              Peak/trough/avg MW, demand-forecast deviation.

generation    Generation by fuel type (coal, gas, nuclear, solar, wind, hydro).
              Fuel mix proportions, renewable vs fossil share.

interchange   Inter-regional power flows between balancing authorities.
              Net imports/exports, largest trading relationships.

Signal theory:
  - Demand anomalies by region → economic activity shifts (factory shutdowns,
    data center buildout, extreme weather).
  - Fuel mix shifts → energy cost structure, transition speed.
  - Cross-region interchange patterns → grid stress, congestion, surplus/deficit.
  - Demand-forecast deviation → unplanned load changes (industrial activity proxy).

Data source: EIA API v2 — electricity/rto/ endpoints (free with key).
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

try:
    from agent.pipeline.entity import entity_id_from_key as _entity_id_from_key
except ImportError:
    _entity_id_from_key = None

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_UA = "TirraMind/0.1 (electricity-monitor)"
_TIMEOUT = 25
_EIA_BASE = "https://api.eia.gov/v2"

VALID_MODES = {"demand", "generation", "interchange"}

# Major US balancing authorities (non-exhaustive — EIA has many more)
KNOWN_REGIONS = {
    "BPAT": "Bonneville Power Administration",
    "CISO": "California ISO",
    "ERCO": "Electric Reliability Council of Texas",
    "ISNE": "ISO New England",
    "MISO": "Midcontinent ISO",
    "NYIS": "New York ISO",
    "PJM": "PJM Interconnection",
    "SC": "South Carolina",
    "SCEG": "Dominion Energy South Carolina",
    "SOCO": "Southern Company",
    "SPA": "Southwestern Power Administration",
    "SWPP": "Southwest Power Pool",
    "TVA": "Tennessee Valley Authority",
    "WACM": "Western Area Power - CO/MO",
}

EIA_FUEL_TYPES = {
    "COL": "Coal",
    "NG": "Natural Gas",
    "NUC": "Nuclear",
    "OIL": "Petroleum",
    "OTH": "Other",
    "SUN": "Solar",
    "WAT": "Hydro",
    "WND": "Wind",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _fetch_eia(
    endpoint: str,
    facets: dict[str, list[str]] | None,
    api_key: str,
    length: int = 5000,
    sort_col: str = "period",
    sort_dir: str = "desc",
) -> list[dict] | None:
    """Fetch data from EIA API v2.

    Returns list of data records or None on failure.
    """
    url = f"{_EIA_BASE}/{endpoint}/data/"
    params: dict[str, Any] = {
        "api_key": api_key,
        "length": length,
        "sort[0][column]": sort_col,
        "sort[0][direction]": sort_dir,
    }

    if facets:
        for key, values in facets.items():
            for i, v in enumerate(values):
                params[f"facets[{key}][]"] = v
                if len(values) > 1:
                    # EIA API supports multiple values per facet
                    params[f"facets[{key}][{i}]"] = v

    try:
        resp = httpx.get(
            url,
            params=params,
            headers={"User-Agent": _UA},
            timeout=_TIMEOUT,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            log.warning("EIA HTTP %d for %s", resp.status_code, endpoint)
            return None
        body = resp.json()
        return body.get("response", {}).get("data", [])
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("EIA fetch error: %s", exc)
        return None


def _aggregate_hourly(records: list[dict]) -> dict:
    """Compute peak, trough, avg from hourly demand records."""
    values = [_safe_float(r.get("value")) for r in records]
    values = [v for v in values if v > 0]
    if not values:
        return {"peak_mw": 0, "trough_mw": 0, "avg_mw": 0, "hours": 0}
    return {
        "peak_mw": round(max(values)),
        "trough_mw": round(min(values)),
        "avg_mw": round(sum(values) / len(values)),
        "hours": len(values),
    }


def _fuel_mix_proportions(records: list[dict]) -> dict:
    """Compute generation proportions by fuel type."""
    by_fuel: dict[str, float] = {}
    for r in records:
        fuel = r.get("fueltype", r.get("type-name", "unknown"))
        val = _safe_float(r.get("value"))
        if val > 0:
            by_fuel[fuel] = by_fuel.get(fuel, 0) + val

    total = sum(by_fuel.values())
    if total == 0:
        return {}

    result: dict[str, Any] = {}
    for fuel, mw in sorted(by_fuel.items(), key=lambda x: -x[1]):
        pct = (mw / total) * 100
        label = EIA_FUEL_TYPES.get(fuel, fuel)
        result[label] = {
            "total_mwh": round(mw),
            "share_pct": round(pct, 1),
        }

    # Compute renewable vs fossil
    renewable_fuels = {"SUN", "WND", "WAT"}
    fossil_fuels = {"COL", "NG", "OIL"}
    renewable = sum(by_fuel.get(f, 0) for f in renewable_fuels)
    fossil = sum(by_fuel.get(f, 0) for f in fossil_fuels)
    result["_summary"] = {
        "renewable_pct": round((renewable / total) * 100, 1) if total else 0,
        "fossil_pct": round((fossil / total) * 100, 1) if total else 0,
        "total_mwh": round(total),
    }
    return result


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------


class ElectricityMonitorTool(Tool):
    """US-wide electricity demand, generation mix, and interchange flows."""

    name = "electricity_monitor"
    description = (
        "Monitor US-wide electricity via EIA API. Covers all balancing "
        "authorities (PJM, CAISO, ERCOT, MISO, ISO-NE, etc.). "
        "Modes: 'demand' for hourly load (MW) with peak/trough, "
        "'generation' for fuel mix (coal/gas/nuclear/solar/wind/hydro), "
        "'interchange' for inter-regional power flows. "
        "Requires TIRRA_EIA_API_KEY."
    )
    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": sorted(VALID_MODES),
                "description": "demand|generation|interchange",
            },
            "region": {
                "type": "string",
                "description": (
                    "Balancing authority code: PJM, CISO, ERCO, MISO, NYIS, ISNE, SWPP, SOCO, TVA, BPAT, etc."
                ),
            },
            "days": {
                "type": "integer",
                "description": "Number of days of data (1-7, default 1).",
            },
        },
        "required": ["mode", "region"],
    }

    def __init__(
        self,
        *,
        cache: DataCache | None = None,
        pipeline_store: PipelineStore | None = None,
    ) -> None:
        self._cache = cache
        self._store = pipeline_store
        self._api_key = self._get_api_key()

    @staticmethod
    def _get_api_key() -> str | None:
        key = os.environ.get("TIRRA_EIA_API_KEY", "").strip()
        return key if key else None

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
                output="EIA API key required. Set TIRRA_EIA_API_KEY.",
            )

        region = (kwargs.get("region") or "").strip().upper()
        if not region:
            return ToolResult(
                success=False,
                output="Parameter 'region' required. Use BA code: PJM, CISO, ERCO, etc.",
            )

        days = kwargs.get("days", 1)
        try:
            days = int(days)
        except (TypeError, ValueError):
            days = 1
        days = max(1, min(7, days))

        if mode == "demand":
            result = self._demand(region, days)
        elif mode == "generation":
            result = self._generation(region, days)
        else:
            result = self._interchange(region, days)

        if result.success:
            self._persist_entities(region, mode)
        return result

    # ------------------------------------------------------------------
    # Mode: demand
    # ------------------------------------------------------------------

    def _demand(self, region: str, days: int) -> ToolResult:
        cache_ns = "electricity_demand"
        cache_key = f"{region}_{days}"
        if self._cache:
            cached = self._cache.get(cache_ns, cache_key)
            if cached is not None:
                return ToolResult(success=True, output=cached)

        # EIA uses respondent facet for BA code
        facets = {"respondent": [region]}
        records = _fetch_eia(
            "electricity/rto/region-data",
            facets,
            self._api_key,
            length=days * 24 * 2,  # hourly data, buffer
        )
        if records is None:
            return ToolResult(
                success=False,
                output=f"Failed to fetch demand data for {region}.",
            )

        # Filter to actual demand type
        demand_records = [
            r
            for r in records
            if r.get("type-name", "").lower() in ("demand", "demand forecast", "net generation")
            or r.get("type", "").upper() == "D"
        ]
        if not demand_records:
            demand_records = records  # Use all if no type filter matches

        if not demand_records:
            result = f"No demand data available for {region} (last {days} day(s))."
            if self._cache:
                self._cache.put(cache_ns, cache_key, result)
            return ToolResult(success=True, output=result)

        stats = _aggregate_hourly(demand_records)
        region_name = KNOWN_REGIONS.get(region, region)

        lines = [
            f"⚡ Electricity Demand — {region_name} ({region})",
            f"Period: last {days} day(s), {stats['hours']} hours of data",
            f"Peak: {stats['peak_mw']:,} MW",
            f"Trough: {stats['trough_mw']:,} MW",
            f"Average: {stats['avg_mw']:,} MW",
        ]

        # Show latest few records
        latest = demand_records[:6]
        if latest:
            lines.append("")
            lines.append("Recent readings:")
            for r in latest:
                period = r.get("period", "?")
                val = _safe_float(r.get("value"))
                tname = r.get("type-name", "")
                lines.append(f"  {period}: {val:,.0f} MW ({tname})")

        result = "\n".join(lines)
        if self._cache:
            self._cache.put(cache_ns, cache_key, result)
        return ToolResult(success=True, output=result)

    # ------------------------------------------------------------------
    # Mode: generation
    # ------------------------------------------------------------------

    def _generation(self, region: str, days: int) -> ToolResult:
        cache_ns = "electricity_generation"
        cache_key = f"{region}_{days}"
        if self._cache:
            cached = self._cache.get(cache_ns, cache_key)
            if cached is not None:
                return ToolResult(success=True, output=cached)

        facets = {"respondent": [region]}
        records = _fetch_eia(
            "electricity/rto/fuel-type-data",
            facets,
            self._api_key,
            length=days * 24 * len(EIA_FUEL_TYPES),
        )
        if records is None:
            return ToolResult(
                success=False,
                output=f"Failed to fetch generation data for {region}.",
            )

        if not records:
            result = f"No generation data available for {region} (last {days} day(s))."
            if self._cache:
                self._cache.put(cache_ns, cache_key, result)
            return ToolResult(success=True, output=result)

        mix = _fuel_mix_proportions(records)
        region_name = KNOWN_REGIONS.get(region, region)
        summary = mix.pop("_summary", {})

        lines = [
            f"⚡ Generation Mix — {region_name} ({region})",
            f"Period: last {days} day(s)",
            f"Total generation: {summary.get('total_mwh', 0):,} MWh",
            f"Renewable share: {summary.get('renewable_pct', 0)}%",
            f"Fossil share: {summary.get('fossil_pct', 0)}%",
            "",
            "By fuel type:",
        ]
        for fuel, info in mix.items():
            lines.append(f"  {fuel}: {info['total_mwh']:,} MWh ({info['share_pct']}%)")

        result = "\n".join(lines)
        if self._cache:
            self._cache.put(cache_ns, cache_key, result)
        return ToolResult(success=True, output=result)

    # ------------------------------------------------------------------
    # Mode: interchange
    # ------------------------------------------------------------------

    def _interchange(self, region: str, days: int) -> ToolResult:
        cache_ns = "electricity_interchange"
        cache_key = f"{region}_{days}"
        if self._cache:
            cached = self._cache.get(cache_ns, cache_key)
            if cached is not None:
                return ToolResult(success=True, output=cached)

        # Interchange has fromba and toba — get both directions
        records_from = _fetch_eia(
            "electricity/rto/interchange-data",
            {"fromba": [region]},
            self._api_key,
            length=days * 24 * 20,
        )
        records_to = _fetch_eia(
            "electricity/rto/interchange-data",
            {"toba": [region]},
            self._api_key,
            length=days * 24 * 20,
        )

        if records_from is None and records_to is None:
            return ToolResult(
                success=False,
                output=f"Failed to fetch interchange data for {region}.",
            )

        exports: dict[str, float] = {}
        imports: dict[str, float] = {}

        for r in records_from or []:
            partner = r.get("toba", "?")
            val = _safe_float(r.get("value"))
            exports[partner] = exports.get(partner, 0) + val

        for r in records_to or []:
            partner = r.get("fromba", "?")
            val = _safe_float(r.get("value"))
            imports[partner] = imports.get(partner, 0) + val

        total_export = sum(exports.values())
        total_import = sum(imports.values())
        net = total_import - total_export
        region_name = KNOWN_REGIONS.get(region, region)

        lines = [
            f"⚡ Interchange Flows — {region_name} ({region})",
            f"Period: last {days} day(s)",
            f"Total exports: {total_export:,.0f} MWh",
            f"Total imports: {total_import:,.0f} MWh",
            f"Net: {'import' if net > 0 else 'export'} {abs(net):,.0f} MWh",
        ]

        if exports:
            lines.append("")
            lines.append("Exports to:")
            for partner, mwh in sorted(exports.items(), key=lambda x: -x[1])[:10]:
                lines.append(f"  → {partner}: {mwh:,.0f} MWh")

        if imports:
            lines.append("")
            lines.append("Imports from:")
            for partner, mwh in sorted(imports.items(), key=lambda x: -x[1])[:10]:
                lines.append(f"  ← {partner}: {mwh:,.0f} MWh")

        result = "\n".join(lines)
        if self._cache:
            self._cache.put(cache_ns, cache_key, result)
        return ToolResult(success=True, output=result)

    # ------------------------------------------------------------------
    # L2 entity persistence
    # ------------------------------------------------------------------

    def _persist_entities(self, region: str, mode: str) -> dict[str, int]:
        if self._store is None or _entity_id_from_key is None:
            return {"grid_demand_obs": 0}
        try:
            return self._persist_entities_inner(region, mode)
        except Exception:
            log.exception("Electricity monitor entity persistence failed (non-fatal)")
            return {"grid_demand_obs": 0}

    def _persist_entities_inner(self, region: str, mode: str) -> dict[str, int]:
        assert self._store is not None
        assert _entity_id_from_key is not None

        if not region:
            return {"grid_demand_obs": 0}

        region_name = KNOWN_REGIONS.get(region, region)
        eid = _entity_id_from_key("organization", region)
        self._store.register_entity("organization", region, eid)
        self._store.store_entity_observation(
            entity_id=eid,
            source_tool="electricity_monitor",
            observed_at=time.time(),
            observation_type="grid_demand",
            value={
                "mode": mode,
                "region": region,
                "region_name": region_name,
            },
            depth_level=2,
        )
        return {"grid_demand_obs": 1}
