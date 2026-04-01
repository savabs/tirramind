"""
Tool: Transportation Throughput — US Border Crossings + FRED Freight Indicators

Two complementary data sources, both free and no auth:

  BTS Border Crossings (data.transportation.gov, Socrata API):
    Monthly counts at every land border port of entry (US-Canada, US-Mexico).
    Measures: Trucks, Trains, Rail Containers (loaded/empty), Personal Vehicles,
    Buses, Pedestrians, Passengers. 333K+ records, back to 1996.
    JSON API with SoQL query support.

  FRED Freight Indicators (via existing macro_data tool — referenced here for context):
    RAILFRTCARLOADSD11 = weekly rail carloads
    TRUCKD11 = truck tonnage index
    Can be fetched separately via macro_data tool.

Signal theory:
  - Truck volume at US-Mexico border = nearshoring/trade war barometer
  - Rail containers loaded vs empty = trade balance direction indicator
  - Month-over-month change in truck traffic = GDP leading indicator
  - Canada vs Mexico volume divergence = supply chain realignment
  - Port-level anomalies = localized trade disruption (e.g., strike, policy)

Modes:
  recent   — Latest month's border crossing data aggregated by mode + border.
  trend    — Monthly time series for a specific measure and border.
  port     — Detail for a specific border port or state.
  compare  — Side-by-side: US-Canada vs US-Mexico for a measure.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

_BTS_BASE = "https://data.transportation.gov/resource/keg4-3bc2.json"
_UA = "TirraMind/0.1"
_TIMEOUT = 15

# Valid measure types
VALID_MEASURES = {
    "Trucks",
    "Trains",
    "Personal Vehicles",
    "Pedestrians",
    "Buses",
    "Rail Containers Loaded",
    "Rail Containers Empty",
    "Personal Vehicle Passengers",
    "Bus Passengers",
    "Train Passengers",
}

# Short aliases → full measure names
MEASURE_ALIASES: dict[str, str] = {
    "trucks": "Trucks",
    "trains": "Trains",
    "rail": "Trains",
    "cars": "Personal Vehicles",
    "vehicles": "Personal Vehicles",
    "pedestrians": "Pedestrians",
    "buses": "Buses",
    "containers_loaded": "Rail Containers Loaded",
    "containers_empty": "Rail Containers Empty",
    "rail_loaded": "Rail Containers Loaded",
    "rail_empty": "Rail Containers Empty",
    "passengers": "Personal Vehicle Passengers",
    "bus_passengers": "Bus Passengers",
    "train_passengers": "Train Passengers",
}

VALID_BORDERS = {"US-Canada Border", "US-Mexico Border"}
BORDER_ALIASES: dict[str, str] = {
    "canada": "US-Canada Border",
    "mexico": "US-Mexico Border",
    "ca": "US-Canada Border",
    "mx": "US-Mexico Border",
}

# Key trade measures (most economically significant)
_KEY_MEASURES = [
    "Trucks",
    "Trains",
    "Rail Containers Loaded",
    "Rail Containers Empty",
    "Personal Vehicles",
]


def _resolve_measure(raw: str) -> str | None:
    """Resolve a measure alias to the canonical name."""
    raw_lower = raw.strip().lower()
    if raw_lower in MEASURE_ALIASES:
        return MEASURE_ALIASES[raw_lower]
    # Check if it's already a valid measure (case-insensitive match)
    for m in VALID_MEASURES:
        if raw_lower == m.lower():
            return m
    return None


def _resolve_border(raw: str) -> str | None:
    """Resolve a border alias to canonical name."""
    raw_lower = raw.strip().lower()
    if raw_lower in BORDER_ALIASES:
        return BORDER_ALIASES[raw_lower]
    for b in VALID_BORDERS:
        if raw_lower == b.lower():
            return b
    return None


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


class TransportThroughputTool(Tool):
    name = "transport_throughput"
    description = (
        "Monitor US border crossing traffic volumes (trucks, trains, rail "
        "containers, personal vehicles) at US-Canada and US-Mexico borders. "
        "BTS Socrata API — free, no auth, monthly data since 1996. "
        "Mode 'recent' shows latest aggregates. Mode 'trend' shows time series. "
        "Mode 'port' drills into specific ports/states. Mode 'compare' shows "
        "Canada vs Mexico side-by-side. Trucks = trade proxy, rail containers "
        "loaded/empty = trade balance indicator."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["recent", "trend", "port", "compare"],
                "default": "recent",
                "description": (
                    "recent = latest month aggregate. "
                    "trend = time series for a measure. "
                    "port = detail by port or state. "
                    "compare = Canada vs Mexico side-by-side."
                ),
            },
            "measure": {
                "type": "string",
                "default": "trucks",
                "description": (
                    "Transport measure: trucks, trains, rail_loaded, "
                    "rail_empty, vehicles, pedestrians, buses, passengers. "
                    "Default: trucks."
                ),
            },
            "border": {
                "type": "string",
                "default": "",
                "description": (
                    "Border filter: 'canada' or 'mexico' (or 'ca'/'mx'). "
                    "Empty = both borders."
                ),
            },
            "state": {
                "type": "string",
                "default": "",
                "description": "Filter by US state name (e.g., 'Texas', 'Michigan'). Port mode only.",
            },
            "months_back": {
                "type": "integer",
                "default": 6,
                "description": "Number of months of history for trend mode. Default 6, max 60.",
            },
            "limit": {
                "type": "integer",
                "default": 25,
                "description": "Max results. Default 25, max 200.",
            },
        },
        "required": [],
    }

    def __init__(self, cache: DataCache | None = None) -> None:
        self._cache = cache

    def execute(
        self,
        *,
        mode: str = "recent",
        measure: str = "trucks",
        border: str = "",
        state: str = "",
        months_back: int = 6,
        limit: int = 25,
        **_: Any,
    ) -> ToolResult:
        mode = mode.lower().strip()
        if mode not in ("recent", "trend", "port", "compare"):
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Use 'recent', 'trend', 'port', or 'compare'.",
            )

        months_back = max(1, min(months_back, 60))
        limit = max(1, min(limit, 200))

        # Resolve measure
        resolved_measure = _resolve_measure(measure)
        if not resolved_measure:
            return ToolResult(
                success=False,
                output=f"Unknown measure '{measure}'. Valid: {', '.join(sorted(MEASURE_ALIASES.keys()))}",
            )

        # Resolve border
        resolved_border = None
        if border.strip():
            resolved_border = _resolve_border(border)
            if not resolved_border:
                return ToolResult(
                    success=False,
                    output=f"Unknown border '{border}'. Use 'canada'/'ca' or 'mexico'/'mx'.",
                )

        if mode == "recent":
            return self._execute_recent(
                measure=resolved_measure,
                border=resolved_border,
                limit=limit,
            )
        if mode == "trend":
            return self._execute_trend(
                measure=resolved_measure,
                border=resolved_border,
                months_back=months_back,
            )
        if mode == "port":
            return self._execute_port(
                measure=resolved_measure,
                border=resolved_border,
                state=state.strip(),
                limit=limit,
            )
        # compare
        return self._execute_compare(
            measure=resolved_measure,
            months_back=months_back,
        )

    # ------------------------------------------------------------------
    # recent mode
    # ------------------------------------------------------------------

    def _execute_recent(
        self,
        *,
        measure: str,
        border: str | None,
        limit: int,
    ) -> ToolResult:
        # Get latest month across all key measures
        where_parts = []
        if border:
            where_parts.append(f"border='{border}'")

        # First: find the latest date
        params: dict[str, str] = {
            "$select": "max(date) as max_date",
            "$limit": "1",
        }
        if where_parts:
            params["$where"] = " AND ".join(where_parts)

        data, error = self._fetch_bts(params)
        if error:
            return ToolResult(success=False, output=error)

        if not data or not data[0].get("max_date"):
            return ToolResult(
                success=True,
                output="BTS: No border crossing data found.",
                data={"records": [], "count": 0},
            )

        max_date = data[0]["max_date"]

        # Now fetch all key measures for that date
        where_date = [f"date='{max_date}'"]
        if border:
            where_date.append(f"border='{border}'")

        params2: dict[str, str] = {
            "$select": "border, measure, sum(value) as total",
            "$where": " AND ".join(where_date),
            "$group": "border, measure",
            "$order": "border, measure",
            "$limit": "50",
        }

        data2, error2 = self._fetch_bts(params2)
        if error2:
            return ToolResult(success=False, output=error2)

        # Format output
        period = max_date[:10]
        lines = [f"BTS Border Crossings — {period}:", ""]

        by_border: dict[str, list[dict[str, Any]]] = {}
        for row in data2:
            b = row.get("border", "Unknown")
            by_border.setdefault(b, []).append(row)

        formatted_records = []
        for b_name, rows in sorted(by_border.items()):
            lines.append(f"  {b_name}:")
            for row in sorted(rows, key=lambda r: -_safe_int(r.get("total"))):
                m = row.get("measure", "")
                total = _safe_int(row.get("total"))
                if total == 0:
                    continue
                is_key = m in _KEY_MEASURES
                marker = " *" if is_key else ""
                lines.append(f"    {m:35s} {total:>12,}{marker}")
                formatted_records.append(
                    {
                        "border": b_name,
                        "measure": m,
                        "total": total,
                        "period": period,
                    }
                )
            lines.append("")

        lines.append("  (* = key trade indicator)")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "records": formatted_records,
                "count": len(formatted_records),
                "period": period,
            },
        )

    # ------------------------------------------------------------------
    # trend mode
    # ------------------------------------------------------------------

    def _execute_trend(
        self,
        *,
        measure: str,
        border: str | None,
        months_back: int,
    ) -> ToolResult:
        now = datetime.now(timezone.utc)
        start_date = (now - timedelta(days=months_back * 31)).strftime(
            "%Y-%m-%dT00:00:00"
        )

        where_parts = [f"measure='{measure}'", f"date>='{start_date}'"]
        if border:
            where_parts.append(f"border='{border}'")

        params: dict[str, str] = {
            "$select": "date, border, sum(value) as total",
            "$where": " AND ".join(where_parts),
            "$group": "date, border",
            "$order": "date ASC",
            "$limit": "500",
        }

        data, error = self._fetch_bts(params)
        if error:
            return ToolResult(success=False, output=error)

        if not data:
            return ToolResult(
                success=True,
                output=f"BTS: No {measure} data in last {months_back} months.",
                data={"series": [], "count": 0},
            )

        # Build time series
        series: list[dict[str, Any]] = []
        for row in data:
            total = _safe_int(row.get("total"))
            series.append(
                {
                    "date": row.get("date", "")[:10],
                    "border": row.get("border", ""),
                    "total": total,
                }
            )

        # Compute MoM changes
        border_label = border if border else "Both Borders"
        lines = [f"BTS Trend: {measure} — {border_label} (last {months_back}mo):", ""]

        # Group by border for display
        by_border: dict[str, list[dict[str, Any]]] = {}
        for s in series:
            by_border.setdefault(s["border"], []).append(s)

        for b_name, b_series in sorted(by_border.items()):
            lines.append(f"  {b_name}:")
            prev_val = None
            for s in b_series:
                pct = ""
                if prev_val and prev_val > 0:
                    change = (s["total"] - prev_val) / prev_val * 100
                    pct = f"  ({change:+.1f}%)"
                lines.append(f"    {s['date']}  {s['total']:>12,}{pct}")
                prev_val = s["total"]
            lines.append("")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"series": series, "count": len(series), "measure": measure},
        )

    # ------------------------------------------------------------------
    # port mode
    # ------------------------------------------------------------------

    def _execute_port(
        self,
        *,
        measure: str,
        border: str | None,
        state: str,
        limit: int,
    ) -> ToolResult:
        # Get latest date first
        date_params: dict[str, str] = {
            "$select": "max(date) as max_date",
            "$limit": "1",
        }
        date_data, error = self._fetch_bts(date_params)
        if error:
            return ToolResult(success=False, output=error)
        if not date_data:
            return ToolResult(
                success=True, output="BTS: No data.", data={"ports": [], "count": 0}
            )

        max_date = date_data[0]["max_date"]

        where_parts = [f"date='{max_date}'", f"measure='{measure}'"]
        if border:
            where_parts.append(f"border='{border}'")
        if state:
            where_parts.append(f"upper(state)=upper('{state}')")

        params: dict[str, str] = {
            "$select": "port_name, state, border, value",
            "$where": " AND ".join(where_parts),
            "$order": "value DESC",
            "$limit": str(limit),
        }

        data, error = self._fetch_bts(params)
        if error:
            return ToolResult(success=False, output=error)

        if not data:
            return ToolResult(
                success=True,
                output=f"BTS: No {measure} data for {state or 'any state'} in {max_date[:10]}.",
                data={"ports": [], "count": 0},
            )

        formatted = []
        lines = [
            f"BTS Port Detail: {measure} — {max_date[:10]}"
            + (f" ({state})" if state else "")
            + ":",
            "",
        ]
        for row in data:
            val = _safe_int(row.get("value"))
            port = row.get("port_name", "Unknown")
            st = row.get("state", "")
            b = row.get("border", "")
            if val > 0:
                lines.append(f"  {port:30s} {st:15s} {val:>10,}  ({b})")
                formatted.append(
                    {
                        "port": port,
                        "state": st,
                        "border": b,
                        "value": val,
                    }
                )
        lines.append("")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"ports": formatted, "count": len(formatted), "period": max_date[:10]},
        )

    # ------------------------------------------------------------------
    # compare mode
    # ------------------------------------------------------------------

    def _execute_compare(
        self,
        *,
        measure: str,
        months_back: int,
    ) -> ToolResult:
        now = datetime.now(timezone.utc)
        start_date = (now - timedelta(days=months_back * 31)).strftime(
            "%Y-%m-%dT00:00:00"
        )

        params: dict[str, str] = {
            "$select": "date, border, sum(value) as total",
            "$where": f"measure='{measure}' AND date>='{start_date}'",
            "$group": "date, border",
            "$order": "date ASC",
            "$limit": "500",
        }

        data, error = self._fetch_bts(params)
        if error:
            return ToolResult(success=False, output=error)

        if not data:
            return ToolResult(
                success=True,
                output=f"BTS: No {measure} data for comparison.",
                data={"comparison": [], "count": 0},
            )

        # Pivot: date → {canada: val, mexico: val}
        by_date: dict[str, dict[str, int]] = {}
        for row in data:
            d = row.get("date", "")[:10]
            b = row.get("border", "")
            val = _safe_int(row.get("total"))
            by_date.setdefault(d, {})[b] = val

        lines = [
            f"BTS Compare: {measure} — Canada vs Mexico (last {months_back}mo):",
            "",
        ]
        lines.append(f"  {'Date':12s} {'Canada':>12s} {'Mexico':>12s} {'Ratio':>8s}")
        lines.append(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*8}")

        comparison = []
        for date_str in sorted(by_date.keys()):
            vals = by_date[date_str]
            ca_val = vals.get("US-Canada Border", 0)
            mx_val = vals.get("US-Mexico Border", 0)
            ratio = f"{ca_val / mx_val:.2f}" if mx_val > 0 else "n/a"
            lines.append(f"  {date_str:12s} {ca_val:>12,} {mx_val:>12,} {ratio:>8s}")
            comparison.append(
                {
                    "date": date_str,
                    "canada": ca_val,
                    "mexico": mx_val,
                    "ratio": ca_val / mx_val if mx_val > 0 else None,
                }
            )
        lines.append("")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "comparison": comparison,
                "count": len(comparison),
                "measure": measure,
            },
        )

    # ------------------------------------------------------------------
    # BTS Socrata fetch
    # ------------------------------------------------------------------

    def _fetch_bts(
        self,
        params: dict[str, str],
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Fetch from BTS Socrata API. Returns (rows, error)."""
        cache_key = {k: v for k, v in sorted(params.items())}
        if self._cache:
            cached = self._cache.get("transport_throughput", cache_key)
            if cached is not None:
                return cached, None

        try:
            with httpx.Client(
                timeout=_TIMEOUT,
                headers={"User-Agent": _UA, "Accept": "application/json"},
                follow_redirects=True,
            ) as client:
                resp = client.get(_BTS_BASE, params=params)
                if resp.status_code == 429:
                    return [], "BTS API: Rate limited."
                if resp.status_code >= 400:
                    return [], f"BTS API error: HTTP {resp.status_code}"
                data = resp.json()
        except httpx.TimeoutException:
            return [], "BTS API: Request timed out."
        except Exception as exc:
            log.exception("BTS fetch failed")
            return [], f"BTS fetch error: {exc}"

        if not isinstance(data, list):
            return [], "BTS API: Unexpected response format."

        if self._cache and data:
            self._cache.put("transport_throughput", cache_key, data, ttl=7200)

        return data, None
