"""
Tool: Interconnection Queue — US Generator Pipeline

EIA API v2  https://api.eia.gov/v2/  (free, API key required)

Shows what generation capacity is planned or under construction across the US.
This is committed capital — projects in the queue represent billions in
investment decisions that reveal future energy landscape, grid stress, and
hyperscaler expansion.

Modes
-----
queue         Search planned/under-construction generators by state, fuel type,
              technology, and capacity. Individual project detail.

summary       Aggregate MW in pipeline by fuel type, state, and status.
              Technology transition speeds, regional concentration.

datacenter    Detect likely data center power projects by matching entity/plant
              names against hyperscaler patterns (Amazon, Microsoft, Google,
              Meta, Equinix, Digital Realty, QTS, etc.).

Signal theory:
  - Pipeline MW by fuel type → energy transition speed / policy direction
  - Data center queue concentration → hyperscaler expansion geography
  - Queue withdrawal rate → project viability (high withdrawal = bubble)
  - Regional capacity buildout → grid stress, land use, NIMBY risk
  - Battery storage MW → grid reliability investment, renewable intermittency hedge

Data source: EIA electricity/operating-generator-capacity/ (free with key).
"""

from __future__ import annotations

import logging
import os
import re
import time
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_UA = "TirraMind/0.1 (interconnection-queue)"
_TIMEOUT = 25
_EIA_BASE = "https://api.eia.gov/v2"

VALID_MODES = {"queue", "summary", "datacenter"}

# EIA status codes
STATUS_MAP = {
    "planned": "PL",
    "construction": "U",
    "operating": "OP",
    "testing": "TS",
    "standby": "SB",
    "retired": "RE",
}

# Technology / energy source codes
ENERGY_SOURCES = {
    "SUN": "Solar",
    "WND": "Wind",
    "NG": "Natural Gas",
    "NUC": "Nuclear",
    "WAT": "Hydro",
    "MWH": "Battery Storage",
    "COL": "Coal",
    "DFO": "Distillate Fuel Oil",
    "GEO": "Geothermal",
    "WH": "Waste Heat",
    "BIT": "Bituminous Coal",
    "SUB": "Subbituminous Coal",
    "LIG": "Lignite",
    "WDS": "Wood/Wood Waste",
    "OBG": "Other Biomass Gas",
    "LFG": "Landfill Gas",
    "PC": "Petroleum Coke",
    "OTH": "Other",
    "PUR": "Purchased Steam",
    "MSW": "Municipal Solid Waste",
    "AB": "Agricultural Byproduct",
    "BLQ": "Black Liquor",
    "JF": "Jet Fuel",
    "KER": "Kerosene",
    "RFO": "Residual Fuel Oil",
    "SGC": "Coal-Derived Synthesis Gas",
    "SGP": "Syngas from Petroleum Coke",
    "SC": "Coal Syngas",
    "WO": "Waste Oil",
    "OBS": "Other Biomass Solid",
    "TDF": "Tire-Derived Fuel",
    "H2": "Hydrogen",
}

# Hyperscaler / data center operator patterns (case-insensitive)
_DC_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bamazon\b",
        r"\baws\b",
        r"\bmicrosoft\b",
        r"\bazure\b",
        r"\bgoogle\b",
        r"\bmeta\b",
        r"\bfacebook\b",
        r"\bapple\b",
        r"\boracle\b",
        r"\bequinix\b",
        r"\bdigital\s*realty\b",
        r"\bcyrusone\b",
        r"\bqts\b",
        r"\bcoresite\b",
        r"\bvantage\b",
        r"\bstack\s*infrastructure\b",
        r"\bswitch\b",
        r"\bt5\s*data\b",
        r"\bdata\s*center\b",
        r"\bdatacenter\b",
        r"\bcloud\s*computing\b",
        r"\bhyperscale\b",
        r"\bcolocation\b",
        r"\bcolo\b",
    ]
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _is_datacenter(entity_name: str, plant_name: str) -> bool:
    """Check if entity or plant name matches known data center patterns."""
    combined = f"{entity_name} {plant_name}"
    return any(pat.search(combined) for pat in _DC_PATTERNS)


def _status_to_eia(status: str) -> str | None:
    """Convert human-readable status to EIA code."""
    return STATUS_MAP.get(status.lower())


def _fetch_generators(
    facets: dict[str, list[str]],
    api_key: str,
    length: int = 5000,
) -> list[dict] | None:
    """Fetch generator data from EIA API.

    Returns list of generator records or None on failure.
    """
    url = f"{_EIA_BASE}/electricity/operating-generator-capacity/data/"
    params: dict[str, Any] = {
        "api_key": api_key,
        "length": length,
        "sort[0][column]": "nameplate-capacity-mw",
        "sort[0][direction]": "desc",
    }

    for key, values in facets.items():
        for v in values:
            params[f"facets[{key}][]"] = v

    try:
        resp = httpx.get(
            url,
            params=params,
            headers={"User-Agent": _UA},
            timeout=_TIMEOUT,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            log.warning("EIA generators HTTP %d", resp.status_code)
            return None
        body = resp.json()
        return body.get("response", {}).get("data", [])
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("EIA generators fetch error: %s", exc)
        return None


def _summarize_pipeline(records: list[dict]) -> dict:
    """Aggregate generator records into summary stats."""
    by_fuel: dict[str, dict] = {}
    by_state: dict[str, float] = {}
    by_status: dict[str, float] = {}
    total_mw = 0.0
    count = 0

    for r in records:
        mw = _safe_float(r.get("nameplate-capacity-mw", r.get("nameplate_capacity_mw")))
        fuel = r.get("energy-source-code", r.get("energy_source_code", "OTH"))
        state = r.get("stateid", r.get("state", "??"))
        status = r.get("status", "?")

        total_mw += mw
        count += 1

        if fuel not in by_fuel:
            by_fuel[fuel] = {
                "mw": 0.0,
                "count": 0,
                "label": ENERGY_SOURCES.get(fuel, fuel),
            }
        by_fuel[fuel]["mw"] += mw
        by_fuel[fuel]["count"] += 1

        by_state[state] = by_state.get(state, 0) + mw
        by_status[status] = by_status.get(status, 0) + mw

    return {
        "total_mw": round(total_mw, 1),
        "project_count": count,
        "by_fuel": {
            k: {"mw": round(v["mw"], 1), "count": v["count"], "label": v["label"]}
            for k, v in sorted(by_fuel.items(), key=lambda x: -x[1]["mw"])
        },
        "by_state": {k: round(v, 1) for k, v in sorted(by_state.items(), key=lambda x: -x[1])[:15]},
        "by_status": {k: round(v, 1) for k, v in by_status.items()},
    }


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------


class InterconnectionQueueTool(Tool):
    """US interconnection queue — planned & under-construction generators."""

    name = "interconnection_queue"
    description = (
        "Search the US generator interconnection queue via EIA. "
        "Modes: 'queue' for individual planned/under-construction projects, "
        "'summary' for aggregate MW by fuel type/state/status, "
        "'datacenter' for detecting hyperscaler power projects "
        "(Amazon, Microsoft, Google, Meta, Equinix, etc.). "
        "Requires TIRRA_EIA_API_KEY."
    )
    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": sorted(VALID_MODES),
                "description": "datacenter|queue|summary",
            },
            "state": {
                "type": "string",
                "description": "2-letter state code (e.g. TX, CA, VA). Optional filter.",
            },
            "fuel": {
                "type": "string",
                "description": (
                    "Energy source code: SUN, WND, NG, NUC, WAT, MWH (battery), COL, GEO, etc. Optional filter."
                ),
            },
            "status": {
                "type": "string",
                "description": "planned or construction (default: planned).",
            },
            "min_mw": {
                "type": "number",
                "description": "Minimum nameplate capacity MW (default: 0 for queue/summary, 50 for datacenter).",
            },
        },
        "required": ["mode"],
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

    # ------------------------------------------------------------------
    # Entity persistence (L2)
    # ------------------------------------------------------------------

    def _persist_entities(self, records: list[dict[str, Any]]) -> None:
        """Register company entities and store L2 project observations."""
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

        seen_companies: set[str] = set()
        now = time.time()
        for rec in records:
            entity_name = rec.get("entityName", rec.get("entity_name", ""))
            if not entity_name or entity_name in seen_companies:
                continue
            seen_companies.add(entity_name)

            try:
                canon = normalize_company_name(entity_name) if normalize_company_name else entity_name
            except ValueError:
                canon = entity_name

            company_eid = entity_id_from_key("company", canon)
            store.register_entity(
                entity_type="company",
                canonical_name=canon,
                entity_id=company_eid,
            )

            mw = rec.get("nameplate-capacity-mw", rec.get("nameplate_capacity_mw", 0))
            src = rec.get("energy-source-code", rec.get("energy_source_code", ""))
            state = rec.get("stateid", rec.get("state", ""))

            store.store_entity_observation(
                entity_id=company_eid,
                source_tool="interconnection_queue",
                observed_at=now,
                observation_type="project_status",
                depth_level=2,
                value={
                    "plant_name": rec.get("plantName", rec.get("plant_name", "")),
                    "nameplate_capacity_mw": mw,
                    "energy_source_code": src,
                    "state": state,
                    "status": rec.get("status", ""),
                    "technology": rec.get("technology", ""),
                },
            )

            # ── Link company → US country ──
            us_eid = entity_id_from_key("country", "US")
            store.register_entity(
                entity_type="country",
                canonical_name="US",
                entity_id=us_eid,
            )
            store.link_entities(
                entity_id_a=company_eid,
                entity_id_b=us_eid,
                link_type="located_in",
                source="interconnection_queue",
                confidence=1.0,
                metadata={"state": state},
            )

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
        if mode == "queue":
            return self._queue(**kwargs)
        if mode == "summary":
            return self._summary(**kwargs)
        return self._datacenter(**kwargs)

    # ------------------------------------------------------------------
    # Mode: queue
    # ------------------------------------------------------------------

    _QUEUE_STATUSES = {"planned", "construction"}

    def _queue(self, **kwargs: Any) -> ToolResult:
        status_str = (kwargs.get("status") or "planned").strip().lower()
        if status_str not in self._QUEUE_STATUSES:
            return ToolResult(
                success=False,
                output=f"Invalid status '{status_str}'. Use: planned, construction.",
            )
        eia_status = _status_to_eia(status_str)

        facets: dict[str, list[str]] = {"status": [eia_status]}
        state = (kwargs.get("state") or "").strip().upper()
        if state:
            facets["stateid"] = [state]
        fuel = (kwargs.get("fuel") or "").strip().upper()
        if fuel:
            facets["energy_source_code"] = [fuel]

        min_mw = _safe_float(kwargs.get("min_mw"), 0)

        cache_ns = "interconnection_queue"
        cache_key = f"queue_{status_str}_{state}_{fuel}_{min_mw}"
        if self._cache:
            cached = self._cache.get(cache_ns, cache_key)
            if cached is not None:
                return ToolResult(success=True, output=cached)

        records = _fetch_generators(facets, self._api_key)
        if records is None:
            return ToolResult(
                success=False,
                output="Failed to fetch generator queue data from EIA.",
            )

        # Apply min_mw filter
        if min_mw > 0:
            records = [
                r
                for r in records
                if _safe_float(r.get("nameplate-capacity-mw", r.get("nameplate_capacity_mw"))) >= min_mw
            ]

        self._persist_entities(records)

        if not records:
            result = f"No {status_str} generators found"
            if state:
                result += f" in {state}"
            if fuel:
                result += f" ({ENERGY_SOURCES.get(fuel, fuel)})"
            result += "."
            if self._cache:
                self._cache.put(cache_ns, cache_key, result)
            return ToolResult(success=True, output=result)

        lines = [
            f"🔌 Generator Queue — {status_str.title()} (status={eia_status})",
            f"Results: {len(records)} projects",
        ]
        if state:
            lines[0] += f", State={state}"
        if fuel:
            lines[0] += f", Fuel={ENERGY_SOURCES.get(fuel, fuel)}"

        lines.append("")
        for i, r in enumerate(records[:25]):
            name = r.get("plantName", r.get("plant_name", "?"))
            entity = r.get("entityName", r.get("entity_name", "?"))
            mw = _safe_float(r.get("nameplate-capacity-mw", r.get("nameplate_capacity_mw")))
            src = r.get("energy-source-code", r.get("energy_source_code", "?"))
            st = r.get("stateid", r.get("state", "?"))
            tech = r.get("technology", "?")
            lines.append(
                f"  {i + 1}. {name} ({entity}) — {mw:,.1f} MW, {ENERGY_SOURCES.get(src, src)}, {st}, tech={tech}"
            )

        if len(records) > 25:
            lines.append(f"  ... and {len(records) - 25} more projects")

        # Quick summary
        total_mw = sum(_safe_float(r.get("nameplate-capacity-mw", r.get("nameplate_capacity_mw"))) for r in records)
        lines.insert(2, f"Total capacity: {total_mw:,.1f} MW")

        result = "\n".join(lines)
        if self._cache:
            self._cache.put(cache_ns, cache_key, result)
        return ToolResult(success=True, output=result)

    # ------------------------------------------------------------------
    # Mode: summary
    # ------------------------------------------------------------------

    _SUMMARY_STATUSES = {"planned", "construction", "both"}

    def _summary(self, **kwargs: Any) -> ToolResult:
        status_str = (kwargs.get("status") or "both").strip().lower()
        if status_str not in self._SUMMARY_STATUSES:
            return ToolResult(
                success=False,
                output=f"Invalid status '{status_str}'. Use: planned, construction, both.",
            )

        statuses = []
        if status_str == "both":
            statuses = ["PL", "U"]
        else:
            eia_status = _status_to_eia(status_str)
            statuses = [eia_status]

        state = (kwargs.get("state") or "").strip().upper()

        cache_ns = "interconnection_summary"
        cache_key = f"summary_{status_str}_{state}"
        if self._cache:
            cached = self._cache.get(cache_ns, cache_key)
            if cached is not None:
                return ToolResult(success=True, output=cached)

        all_records: list[dict] = []
        for st in statuses:
            facets: dict[str, list[str]] = {"status": [st]}
            if state:
                facets["stateid"] = [state]
            records = _fetch_generators(facets, self._api_key)
            if records:
                all_records.extend(records)

        if not all_records:
            result = "No planned/under-construction generators found"
            if state:
                result += f" in {state}"
            result += "."
            if self._cache:
                self._cache.put(cache_ns, cache_key, result)
            return ToolResult(success=True, output=result)

        summary = _summarize_pipeline(all_records)

        lines = [
            "🔌 Generator Pipeline Summary",
            f"Status: {status_str}",
        ]
        if state:
            lines.append(f"State: {state}")
        lines.extend(
            [
                f"Total capacity: {summary['total_mw']:,.1f} MW ({summary['project_count']} projects)",
                "",
                "By fuel type:",
            ]
        )
        for code, info in summary["by_fuel"].items():
            lines.append(f"  {info['label']} ({code}): {info['mw']:,.1f} MW, {info['count']} projects")

        lines.append("")
        lines.append("Top states:")
        for st, mw in list(summary["by_state"].items())[:10]:
            lines.append(f"  {st}: {mw:,.1f} MW")

        if summary["by_status"]:
            lines.append("")
            lines.append("By status:")
            for st, mw in summary["by_status"].items():
                label = {v: k for k, v in STATUS_MAP.items()}.get(st, st)
                lines.append(f"  {label} ({st}): {mw:,.1f} MW")

        result = "\n".join(lines)
        if self._cache:
            self._cache.put(cache_ns, cache_key, result)
        return ToolResult(success=True, output=result)

    # ------------------------------------------------------------------
    # Mode: datacenter
    # ------------------------------------------------------------------

    def _datacenter(self, **kwargs: Any) -> ToolResult:
        state = (kwargs.get("state") or "").strip().upper()
        min_mw = _safe_float(kwargs.get("min_mw"), 50)

        cache_ns = "interconnection_datacenter"
        cache_key = f"datacenter_{state}_{min_mw}"
        if self._cache:
            cached = self._cache.get(cache_ns, cache_key)
            if cached is not None:
                return ToolResult(success=True, output=cached)

        # Fetch both planned and under-construction
        all_records: list[dict] = []
        for st in ["PL", "U"]:
            facets: dict[str, list[str]] = {"status": [st]}
            if state:
                facets["stateid"] = [state]
            records = _fetch_generators(facets, self._api_key)
            if records:
                all_records.extend(records)

        if not all_records:
            result = "No planned/under-construction generators found."
            if self._cache:
                self._cache.put(cache_ns, cache_key, result)
            return ToolResult(success=True, output=result)

        # Filter for data center matches
        dc_projects = []
        for r in all_records:
            entity = r.get("entityName", r.get("entity_name", ""))
            plant = r.get("plantName", r.get("plant_name", ""))
            mw = _safe_float(r.get("nameplate-capacity-mw", r.get("nameplate_capacity_mw")))
            if mw < min_mw:
                continue
            if _is_datacenter(entity, plant):
                dc_projects.append(r)

        if not dc_projects:
            result = f"No suspected data center projects found (min {min_mw} MW)"
            if state:
                result += f" in {state}"
            result += "."
            if self._cache:
                self._cache.put(cache_ns, cache_key, result)
            return ToolResult(success=True, output=result)

        total_mw = sum(_safe_float(r.get("nameplate-capacity-mw", r.get("nameplate_capacity_mw"))) for r in dc_projects)

        lines = [
            "🏢 Data Center Power Projects",
            f"Suspected hyperscaler/colocation projects: {len(dc_projects)}",
            f"Total capacity: {total_mw:,.1f} MW",
        ]
        if state:
            lines.append(f"State: {state}")
        lines.append("")

        for i, r in enumerate(dc_projects[:20]):
            entity = r.get("entityName", r.get("entity_name", "?"))
            plant = r.get("plantName", r.get("plant_name", "?"))
            mw = _safe_float(r.get("nameplate-capacity-mw", r.get("nameplate_capacity_mw")))
            src = r.get("energy-source-code", r.get("energy_source_code", "?"))
            st = r.get("stateid", r.get("state", "?"))
            status = r.get("status", "?")
            status_label = {v: k for k, v in STATUS_MAP.items()}.get(status, status)
            lines.append(
                f"  {i + 1}. {plant} ({entity}) — {mw:,.1f} MW, {ENERGY_SOURCES.get(src, src)}, {st}, {status_label}"
            )

        if len(dc_projects) > 20:
            lines.append(f"  ... and {len(dc_projects) - 20} more projects")

        # State concentration
        state_mw: dict[str, float] = {}
        for r in dc_projects:
            st = r.get("stateid", r.get("state", "?"))
            mw = _safe_float(r.get("nameplate-capacity-mw", r.get("nameplate_capacity_mw")))
            state_mw[st] = state_mw.get(st, 0) + mw
        if state_mw:
            lines.append("")
            lines.append("State concentration:")
            for st, mw in sorted(state_mw.items(), key=lambda x: -x[1]):
                lines.append(f"  {st}: {mw:,.1f} MW")

        result = "\n".join(lines)
        if self._cache:
            self._cache.put(cache_ns, cache_key, result)
        return ToolResult(success=True, output=result)
