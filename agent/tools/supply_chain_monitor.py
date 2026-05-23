"""
Tool: Supply Chain Price Monitor — BLS PPI + Import Prices

Tracks producer and import price indices across key manufacturing supply chains.
All data from BLS (free, no auth, 25 requests/day).

Modes:
  producer_prices  — BLS PPI for semiconductors, computers, construction machinery,
                     iron & steel, petroleum refining, synthetic dyes/pigments.
  import_prices    — BLS Import Price Indices: all imports, fuels, industrial supplies.
  pressure_index   — Composite: latest PPI values + MoM changes → supply pressure score.

Signal theory:
  - Semiconductor PPI rising while computer PPI flat → margin squeeze in tech
  - Iron/steel and machinery rising together → capex inflation → delayed infrastructure
  - Import prices rising faster than domestic PPI → trade flow disruption
  - Multiple PPI sectors spiking simultaneously → broad inflationary pressure
  - MoM acceleration across sectors → cost-push inflation forming
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone; UTC = timezone.utc
from typing import TYPE_CHECKING, Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

try:
    from agent.pipeline.entity import entity_id_from_key
except ImportError:  # pragma: no cover
    entity_id_from_key = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

_UA = "TirraMind/0.1 (research)"
_TIMEOUT = 15
_CACHE_TTL = 21600  # 6 hours — monthly data

_BLS_BASE = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# ── PPI Series ──────────────────────────────────────────────────
# NAICS-based Producer Price Index series
_PPI_SERIES: dict[str, dict[str, str]] = {
    "PCU334413334413": {
        "label": "Semiconductors & Related Devices",
        "sector": "tech",
    },
    "PCU334111334111": {
        "label": "Electronic Computers",
        "sector": "tech",
    },
    "PCU333120333120": {
        "label": "Construction Machinery & Equipment",
        "sector": "industrial",
    },
    "PCU331110331110": {
        "label": "Iron & Steel Mills",
        "sector": "materials",
    },
    "PCU324110324110": {
        "label": "Petroleum Refineries",
        "sector": "energy",
    },
    "PCU325130325130": {
        "label": "Synthetic Dyes & Pigments",
        "sector": "chemicals",
    },
}

# ── Import Price Series ─────────────────────────────────────────
_IMPORT_SERIES: dict[str, str] = {
    "EIUIR": "All Imports",
    "EIUIR1": "Fuels & Lubricants (Imports)",
    "EIUIR2": "Industrial Supplies excl. Fuels (Imports)",
}

VALID_MODES = frozenset({"producer_prices", "import_prices", "pressure_index"})


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s in (".", "NaN", "nan", "null", ""):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


class SupplyChainMonitorTool(Tool):
    """BLS PPI + Import Price tracking across key supply chain sectors."""

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

    def _persist_entities(self, series_data: dict[str, Any]) -> None:
        """Register topic entities for each BLS series."""
        if self._store is None or entity_id_from_key is None:
            return
        if not series_data:
            return
        try:
            self._persist_entities_inner(series_data)
        except Exception:
            log.exception("Entity persistence failed (non-fatal)")

    def _persist_entities_inner(self, series_data: dict[str, Any]) -> None:
        assert self._store is not None  # noqa: S101
        store = self._store
        now_ts = datetime.now(tz=UTC).timestamp()

        for series_id, info in series_data.items():
            # info may be a dict with 'label'/'sector'/'values', or a list
            if isinstance(info, dict):
                label = info.get("label", series_id)
                sector = info.get("sector", "unknown")
                values = info.get("values", [])
            else:
                label = series_id
                sector = "unknown"
                values = info if isinstance(info, list) else []

            eid = entity_id_from_key("topic", series_id)
            store.register_entity(
                entity_type="topic",
                canonical_name=label,
                entity_id=eid,
                metadata={"series_id": series_id, "sector": sector},
            )

            # Latest value as observation
            latest_val = None
            if isinstance(values, list) and values:
                last = values[-1] if isinstance(values[-1], dict) else {}
                latest_val = _safe_float(last.get("value"))

            store.store_entity_observation(
                entity_id=eid,
                source_tool="supply_chain_monitor",
                observed_at=now_ts,
                observation_type="price_movement",
                depth_level=2,
                value={
                    "series_id": series_id,
                    "sector": sector,
                    "latest_value": latest_val,
                    "num_periods": len(values) if isinstance(values, list) else 0,
                },
            )

    @property
    def name(self) -> str:
        return "supply_chain_prices"

    @property
    def description(self) -> str:
        return (
            "Track producer and import prices across key supply chain sectors. "
            "Modes: producer_prices (PPI for semiconductors, computers, steel, "
            "machinery, petroleum, chemicals), import_prices (all imports, fuels, "
            "industrial supplies), pressure_index (composite supply chain pressure score). "
            "Free BLS data, no auth required."
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
                        "producer_prices: PPI for 6 key sectors. "
                        "import_prices: Import price indices. "
                        "pressure_index: Composite supply chain pressure score."
                    ),
                },
                "months": {
                    "type": "integer",
                    "default": 6,
                    "description": "Months of data (1-24). Default: 6.",
                },
                "sectors": {
                    "type": "string",
                    "default": "all",
                    "description": (
                        "Comma-separated sector filter for producer_prices: "
                        "tech, industrial, materials, energy, chemicals, or 'all'. "
                        "Default: all."
                    ),
                },
            },
            "required": ["mode"],
        }

    # ── Public execute ──────────────────────────────────────────

    def execute(self, **kwargs: Any) -> ToolResult:
        mode = kwargs.get("mode", "")
        if mode not in VALID_MODES:
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Choose from: {', '.join(sorted(VALID_MODES))}.",
            )

        months = kwargs.get("months") or 6
        if not isinstance(months, int):
            try:
                months = int(months)
            except (ValueError, TypeError):
                months = 6
        months = max(1, min(months, 24))

        if mode == "producer_prices":
            sectors = kwargs.get("sectors") or "all"
            return self._handle_producer_prices(months, sectors)
        elif mode == "import_prices":
            return self._handle_import_prices(months)
        else:
            sectors = kwargs.get("sectors") or "all"
            return self._handle_pressure_index(months, sectors)

    # ── Producer Prices ─────────────────────────────────────────

    def _handle_producer_prices(self, months: int, sectors_str: str) -> ToolResult:
        series_ids = _filter_ppi_series(sectors_str)
        if not series_ids:
            return ToolResult(
                success=False,
                output=f"No PPI series match sectors '{sectors_str}'. "
                f"Available sectors: tech, industrial, materials, energy, chemicals, all.",
            )

        cache_key = f"supply_chain:ppi:{','.join(sorted(series_ids))}:{months}"
        if self._cache:
            hit = self._cache.get(cache_key)
            if hit is not None:
                return ToolResult(success=True, output=hit["output"], data=hit["data"])

        data, err = _fetch_bls_multi(series_ids, months)
        if err:
            return ToolResult(success=False, output=err)

        signals = _compute_ppi_signals(data)
        summary = _format_ppi_summary(data, signals, months)

        result_data = {
            "mode": "producer_prices",
            "months": months,
            "series": data,
            "signals": signals,
        }

        if self._cache:
            self._cache.set(cache_key, {"output": summary, "data": result_data}, ttl=_CACHE_TTL)

        self._persist_entities(data)

        return ToolResult(success=True, output=summary, data=result_data)

    # ── Import Prices ───────────────────────────────────────────

    def _handle_import_prices(self, months: int) -> ToolResult:
        cache_key = f"supply_chain:imports:{months}"
        if self._cache:
            hit = self._cache.get(cache_key)
            if hit is not None:
                return ToolResult(success=True, output=hit["output"], data=hit["data"])

        series_ids = list(_IMPORT_SERIES.keys())
        data, err = _fetch_bls_multi(series_ids, months)
        if err:
            return ToolResult(success=False, output=err)

        signals = _compute_import_signals(data)
        summary = _format_import_summary(data, signals, months)

        result_data = {
            "mode": "import_prices",
            "months": months,
            "series": data,
            "signals": signals,
        }

        if self._cache:
            self._cache.set(cache_key, {"output": summary, "data": result_data}, ttl=_CACHE_TTL)

        self._persist_entities(data)

        return ToolResult(success=True, output=summary, data=result_data)

    # ── Pressure Index ──────────────────────────────────────────

    def _handle_pressure_index(self, months: int, sectors_str: str) -> ToolResult:
        ppi_ids = _filter_ppi_series(sectors_str)
        import_ids = list(_IMPORT_SERIES.keys())
        all_ids = list(ppi_ids) + import_ids

        cache_key = f"supply_chain:pressure:{','.join(sorted(all_ids))}:{months}"
        if self._cache:
            hit = self._cache.get(cache_key)
            if hit is not None:
                return ToolResult(success=True, output=hit["output"], data=hit["data"])

        data, err = _fetch_bls_multi(all_ids, months)
        if err:
            return ToolResult(success=False, output=err)

        ppi_data = {k: v for k, v in data.items() if k in _PPI_SERIES}
        import_data = {k: v for k, v in data.items() if k in _IMPORT_SERIES}

        ppi_sig = _compute_ppi_signals(ppi_data)
        import_sig = _compute_import_signals(import_data)
        pressure = _compute_pressure_score(ppi_sig, import_sig)

        summary = _format_pressure_summary(ppi_sig, import_sig, pressure, months)

        result_data = {
            "mode": "pressure_index",
            "months": months,
            "ppi_signals": ppi_sig,
            "import_signals": import_sig,
            "pressure": pressure,
        }

        if self._cache:
            self._cache.set(cache_key, {"output": summary, "data": result_data}, ttl=_CACHE_TTL)

        self._persist_entities(data)

        return ToolResult(success=True, output=summary, data=result_data)


# ── Helpers ─────────────────────────────────────────────────────


def _filter_ppi_series(sectors_str: str) -> list[str]:
    """Filter PPI series by sector."""
    if sectors_str.strip().lower() == "all":
        return list(_PPI_SERIES.keys())

    requested = {s.strip().lower() for s in sectors_str.split(",")}
    return [sid for sid, info in _PPI_SERIES.items() if info["sector"] in requested]


# ── BLS fetch ───────────────────────────────────────────────────


def _fetch_bls_multi(
    series_ids: list[str],
    months: int,
) -> tuple[dict[str, list[dict]], str | None]:
    """Fetch multiple BLS series in a single request (up to 50 per request)."""
    now = datetime.now(UTC)
    end_year = now.year
    start_year = max(end_year - 2, end_year - (months // 12 + 1))

    payload = {
        "seriesid": series_ids,
        "startyear": str(start_year),
        "endyear": str(end_year),
    }

    try:
        with httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": _UA}) as client:
            resp = client.post(_BLS_BASE, json=payload)
    except httpx.TimeoutException:
        return {}, "BLS API timed out."
    except httpx.HTTPError as exc:
        return {}, f"BLS HTTP error: {exc}"

    if resp.status_code == 429:
        return {}, "BLS API rate limit reached. Retry later."
    if resp.status_code != 200:
        return {}, f"BLS returned HTTP {resp.status_code}"

    try:
        body = resp.json()
    except Exception:
        return {}, "Failed to parse BLS response."

    if body.get("status") != "REQUEST_SUCCEEDED":
        msg = "; ".join(body.get("message", ["Unknown error"]))
        return {}, f"BLS request failed: {msg}"

    result: dict[str, list[dict]] = {}
    for s in body.get("Results", {}).get("series", []):
        sid = s.get("seriesID", "")
        raw = s.get("data", [])
        records = []
        for entry in raw:
            period = entry.get("period", "")
            if period == "M13":
                continue
            val = _safe_float(entry.get("value"))
            if val is None:
                continue
            records.append(
                {
                    "year": entry.get("year", ""),
                    "period": period,
                    "value": val,
                }
            )
        records.sort(key=lambda r: (r["year"], r["period"]))
        if len(records) > months:
            records = records[-months:]
        result[sid] = records

    return result, None


# ── Signal computation ──────────────────────────────────────────


def _compute_ppi_signals(
    data: dict[str, list[dict]],
) -> dict[str, Any]:
    """Compute per-series PPI signals."""
    if not data:
        return {"status": "NO_DATA", "series": {}}

    signals: dict[str, Any] = {"series": {}}
    mom_changes: list[float] = []

    for sid, records in data.items():
        info = _PPI_SERIES.get(sid, {"label": sid, "sector": "unknown"})
        sig: dict[str, Any] = {
            "label": info["label"],
            "sector": info["sector"],
        }

        if not records:
            sig["latest"] = None
            signals["series"][sid] = sig
            continue

        sig["latest"] = records[-1]["value"]
        sig["latest_period"] = f"{records[-1]['year']}-{records[-1]['period']}"

        if len(records) >= 2:
            prev = records[-2]["value"]
            if prev > 0:
                mom_pct = ((records[-1]["value"] / prev) - 1) * 100
                sig["mom_pct"] = round(mom_pct, 2)
                mom_changes.append(mom_pct)

                if mom_pct > 2.0:
                    sig["alert"] = "RISING — MoM > 2%"
                elif mom_pct < -2.0:
                    sig["alert"] = "FALLING — MoM < -2%"
                else:
                    sig["alert"] = None
            else:
                sig["mom_pct"] = None
                sig["alert"] = None
        else:
            sig["mom_pct"] = None
            sig["alert"] = None

        # 3-month trend
        if len(records) >= 4:
            start_val = records[-4]["value"]
            end_val = records[-1]["value"]
            if start_val > 0:
                three_mo_pct = ((end_val / start_val) - 1) * 100
                sig["three_month_pct"] = round(three_mo_pct, 2)
            else:
                sig["three_month_pct"] = None
        else:
            sig["three_month_pct"] = None

        signals["series"][sid] = sig

    # Cross-sector summary
    if mom_changes:
        signals["avg_mom_pct"] = round(sum(mom_changes) / len(mom_changes), 2)
        signals["sectors_rising"] = sum(1 for m in mom_changes if m > 0.5)
        signals["sectors_falling"] = sum(1 for m in mom_changes if m < -0.5)
        signals["broad_inflation"] = signals["sectors_rising"] >= 4
    else:
        signals["avg_mom_pct"] = None
        signals["broad_inflation"] = False

    return signals


def _compute_import_signals(
    data: dict[str, list[dict]],
) -> dict[str, Any]:
    """Compute import price signals."""
    if not data:
        return {"status": "NO_DATA", "series": {}}

    signals: dict[str, Any] = {"series": {}}

    for sid, records in data.items():
        label = _IMPORT_SERIES.get(sid, sid)
        sig: dict[str, Any] = {"label": label}

        if not records:
            sig["latest"] = None
            signals["series"][sid] = sig
            continue

        sig["latest"] = records[-1]["value"]
        sig["latest_period"] = f"{records[-1]['year']}-{records[-1]['period']}"

        if len(records) >= 2:
            prev = records[-2]["value"]
            if prev > 0:
                mom_pct = ((records[-1]["value"] / prev) - 1) * 100
                sig["mom_pct"] = round(mom_pct, 2)
            else:
                sig["mom_pct"] = None
        else:
            sig["mom_pct"] = None

        signals["series"][sid] = sig

    return signals


def _compute_pressure_score(
    ppi_sig: dict[str, Any],
    import_sig: dict[str, Any],
) -> dict[str, Any]:
    """Compute composite supply chain pressure score (0-100)."""
    score_components: list[float] = []

    # PPI component: rising sectors / total × weight
    sectors_rising = ppi_sig.get("sectors_rising", 0)
    total_series = len(ppi_sig.get("series", {}))
    if total_series > 0:
        ppi_breadth = (sectors_rising / total_series) * 50
        score_components.append(ppi_breadth)

    # PPI average MoM magnitude
    avg_mom = ppi_sig.get("avg_mom_pct")
    if avg_mom is not None:
        # Scale: 0% = 0, ±3% = 25 (capped)
        magnitude = min(abs(avg_mom), 3.0) / 3.0 * 25
        if avg_mom > 0:
            score_components.append(magnitude)
        else:
            score_components.append(-magnitude * 0.5)  # Deflation less alarming than inflation

    # Import price component
    all_imports = import_sig.get("series", {}).get("EIUIR", {})
    import_mom = all_imports.get("mom_pct")
    if import_mom is not None:
        import_pressure = min(abs(import_mom), 3.0) / 3.0 * 25
        if import_mom > 0:
            score_components.append(import_pressure)
        else:
            score_components.append(-import_pressure * 0.3)

    raw_score = sum(score_components) if score_components else 0
    clamped = max(0, min(100, raw_score))

    if clamped >= 70:
        level = "HIGH — broad cost-push pressure"
    elif clamped >= 40:
        level = "MODERATE — selective sector pressure"
    elif clamped >= 15:
        level = "LOW — limited price pressure"
    else:
        level = "MINIMAL — deflationary or flat"

    return {
        "score": round(clamped, 1),
        "level": level,
        "components": score_components,
    }


# ── Formatting ──────────────────────────────────────────────────


def _format_ppi_summary(
    data: dict[str, list[dict]],
    signals: dict,
    months: int,
) -> str:
    lines = [f"Producer Price Index Summary ({months} months, BLS PPI):\n"]

    for sid, sig in signals.get("series", {}).items():
        label = sig.get("label", sid)
        latest = sig.get("latest")
        if latest is None:
            lines.append(f"  {label}: no data")
            continue

        period = sig.get("latest_period", "")
        mom = sig.get("mom_pct")
        mom_str = f" MoM: {'+' if mom and mom > 0 else ''}{mom}%" if mom is not None else ""
        three_mo = sig.get("three_month_pct")
        three_str = f" 3mo: {'+' if three_mo and three_mo > 0 else ''}{three_mo}%" if three_mo is not None else ""
        alert = sig.get("alert")
        alert_str = f" ⚠ {alert}" if alert else ""
        lines.append(f"  {label}: {latest} ({period}){mom_str}{three_str}{alert_str}")

    avg = signals.get("avg_mom_pct")
    if avg is not None:
        rising = signals.get("sectors_rising", 0)
        falling = signals.get("sectors_falling", 0)
        lines.append(f"\n  Cross-sector: avg MoM {'+' if avg > 0 else ''}{avg}% ({rising} rising, {falling} falling)")

    if signals.get("broad_inflation"):
        lines.append("  ⚠ BROAD INFLATION — 4+ sectors rising simultaneously")

    return "\n".join(lines)


def _format_import_summary(
    data: dict[str, list[dict]],
    signals: dict,
    months: int,
) -> str:
    lines = [f"Import Price Summary ({months} months, BLS):\n"]

    for sid, sig in signals.get("series", {}).items():
        label = sig.get("label", sid)
        latest = sig.get("latest")
        if latest is None:
            lines.append(f"  {label}: no data")
            continue

        period = sig.get("latest_period", "")
        mom = sig.get("mom_pct")
        mom_str = f" MoM: {'+' if mom and mom > 0 else ''}{mom}%" if mom is not None else ""
        lines.append(f"  {label}: {latest} ({period}){mom_str}")

    return "\n".join(lines)


def _format_pressure_summary(
    ppi_sig: dict,
    import_sig: dict,
    pressure: dict,
    months: int,
) -> str:
    lines = [f"Supply Chain Pressure Index ({months} months):\n"]

    score = pressure.get("score", 0)
    level = pressure.get("level", "UNKNOWN")
    lines.append(f"  Overall Score: {score}/100 [{level}]\n")

    # PPI highlights
    lines.append("  PPI Highlights:")
    for sid, sig in ppi_sig.get("series", {}).items():
        label = sig.get("label", sid)
        mom = sig.get("mom_pct")
        if mom is not None:
            direction = "↑" if mom > 0 else ("↓" if mom < 0 else "→")
            lines.append(f"    {label}: {direction} {abs(mom)}%")

    avg = ppi_sig.get("avg_mom_pct")
    if avg is not None:
        lines.append(f"    Avg PPI MoM: {'+' if avg > 0 else ''}{avg}%")

    # Import highlights
    lines.append("\n  Import Highlights:")
    for sid, sig in import_sig.get("series", {}).items():
        label = sig.get("label", sid)
        mom = sig.get("mom_pct")
        if mom is not None:
            direction = "↑" if mom > 0 else ("↓" if mom < 0 else "→")
            lines.append(f"    {label}: {direction} {abs(mom)}%")

    if ppi_sig.get("broad_inflation"):
        lines.append("\n  ⚠ BROAD COST-PUSH PRESSURE — multiple sectors and imports rising")

    return "\n".join(lines)
