"""
Tool: Patent & Trademark Filings — Global Innovation Pipeline Monitor

USPTO ODP:  https://data.uspto.gov/  (free, registration required for key)

Patent filings across all major patent offices reveal the global innovation
pipeline 12-18 months before products ship.  A surge in AI chip patents by
a single company → they're building something big.  Cross-office filing
(PCT → national phase) = serious commercial intent, not just defensive IP.

Modes
-----
search          Search patent grants/applications by keyword, assignee,
                CPC class, or date range.

trends          Filing volume trends by CPC technology class over time.
                Detects innovation acceleration or deceleration.

assignee        Patent portfolio analysis for a specific company/assignee.
                Filing velocity, technology focus, geographic coverage.

Signal theory:
  - Patent velocity acceleration by company → product launch imminent
  - Sudden CPC class concentration shift → strategic pivot
  - Cross-office filing surge → commercialization intent
  - Defensive patent acquisition sprees → M&A preparation
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
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

# USPTO Open Data Portal (PatentsView replacement)
_USPTO_SEARCH = "https://api.patentsview.org/patents/query"
_USPTO_ASSIGNEE = "https://api.patentsview.org/assignees/query"
_UA = "TirraMind/0.1 (patent-filings-tool)"
_TIMEOUT = 25
_CACHE_TTL = 7200  # 2 hours — patent data is slow-moving
_MAX_RESULTS = 50

VALID_MODES = frozenset({"search", "trends", "assignee"})

# Major CPC (Cooperative Patent Classification) technology classes
CPC_CLASSES: dict[str, str] = {
    "A": "Human Necessities",
    "B": "Operations & Transport",
    "C": "Chemistry & Metallurgy",
    "D": "Textiles & Paper",
    "E": "Fixed Constructions",
    "F": "Mechanical Engineering",
    "G": "Physics (instruments, computing, nuclear)",
    "H": "Electricity (circuits, semiconductors, telecom)",
    "Y": "Emerging Cross-Sectional Technologies",
}

# Key CPC subclasses for signal detection
SIGNAL_CPC: dict[str, str] = {
    "G06N": "Machine learning / AI",
    "G06F": "Digital data processing",
    "H01L": "Semiconductor devices",
    "H04L": "Digital information transmission",
    "H04W": "Wireless communication",
    "G16H": "Healthcare informatics",
    "G16B": "Bioinformatics",
    "H02J": "Power supply / energy storage",
    "B60L": "Electric vehicles",
    "C12N": "Biotech / genetic engineering",
    "G01N": "Testing / analysis / sensors",
    "A61K": "Pharmaceuticals",
    "H10K": "Organic electronics / OLEDs",
    "G06Q": "Business methods / fintech",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_patents(
    query: dict[str, Any],
    fields: list[str],
    sort: list[dict[str, str]] | None = None,
    per_page: int = 25,
) -> dict[str, Any] | None:
    """Query PatentsView API.  Returns JSON response or None."""
    payload: dict[str, Any] = {
        "q": query,
        "f": fields,
        "o": {"per_page": min(per_page, _MAX_RESULTS)},
    }
    if sort:
        payload["s"] = sort

    try:
        with httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": _UA}) as client:
            resp = client.post(_USPTO_SEARCH, json=payload)
            if resp.status_code != 200:
                log.warning("USPTO API returned %d", resp.status_code)
                return None
            return resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("USPTO API error: %s", exc)
        return None


def _fetch_assignees(
    query: dict[str, Any],
    fields: list[str],
    per_page: int = 25,
) -> dict[str, Any] | None:
    """Query PatentsView assignee endpoint."""
    payload: dict[str, Any] = {
        "q": query,
        "f": fields,
        "o": {"per_page": min(per_page, _MAX_RESULTS)},
    }
    try:
        with httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": _UA}) as client:
            resp = client.post(_USPTO_ASSIGNEE, json=payload)
            if resp.status_code != 200:
                log.warning("USPTO assignee API returned %d", resp.status_code)
                return None
            return resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("USPTO assignee API error: %s", exc)
        return None


def _parse_date(date_str: str) -> str:
    """Normalize a date string to YYYY-MM-DD."""
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str


def _year_range(years_back: int = 5) -> tuple[str, str]:
    """Return (start_date, end_date) for a lookback window."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=years_back * 365)
    return start.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class PatentFilingsTool(Tool):
    """Search and analyze patent filings from USPTO."""

    name = "patent_filings"
    description = (
        "Search US patent filings by keyword, assignee, or CPC technology "
        "class.  Tracks innovation pipeline velocity, technology pivots, "
        "and cross-company filing patterns."
    )
    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": sorted(VALID_MODES),
                "description": (
                    "assignee: patent portfolio for a company/assignee. "
                    "search: search patents by keyword, CPC class, date range. "
                    "trends: filing volume trends by CPC class."
                ),
            },
            "query": {
                "type": "string",
                "description": "Search keywords (for search mode).",
            },
            "assignee": {
                "type": "string",
                "description": "Company/assignee name (for assignee or search mode).",
            },
            "cpc_class": {
                "type": "string",
                "description": "CPC class code (e.g., 'G06N' for AI, 'H01L' for semiconductors).",
            },
            "date_from": {
                "type": "string",
                "description": "Start date (YYYY-MM-DD).",
            },
            "date_to": {
                "type": "string",
                "description": "End date (YYYY-MM-DD).",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default 25, max 50).",
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

    # ------------------------------------------------------------------
    # Entity persistence (L2)
    # ------------------------------------------------------------------

    def _persist_entities(self, patents: list[dict[str, Any]]) -> None:
        """Register company entities and store L2 patent observations."""
        if self._store is None or entity_id_from_key is None:
            return
        if not patents:
            return
        try:
            self._persist_entities_inner(patents)
        except Exception:
            log.exception("Entity persistence failed (non-fatal)")

    def _persist_entities_inner(self, patents: list[dict[str, Any]]) -> None:
        assert self._store is not None  # noqa: S101
        store = self._store

        seen_assignees: set[str] = set()
        for patent in patents:
            assignee = patent.get("assignee_organization", "")
            if isinstance(assignee, list):
                assignee = assignee[0] if assignee else ""
            if not assignee or assignee in seen_assignees:
                continue
            seen_assignees.add(assignee)

            try:
                canon = (
                    normalize_company_name(assignee)
                    if normalize_company_name
                    else assignee
                )
            except ValueError:
                canon = assignee

            company_eid = entity_id_from_key("company", canon)
            store.register_entity(
                entity_type="company",
                canonical_name=canon,
                entity_id=company_eid,
            )

            # Parse patent date
            pdate = patent.get("patent_date", "")
            try:
                ts = (
                    datetime.strptime(pdate, "%Y-%m-%d")
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                )
            except (ValueError, AttributeError):
                ts = datetime.now(tz=timezone.utc).timestamp()

            cpc = patent.get("cpc_subgroup_id", "")
            if isinstance(cpc, list):
                cpc = cpc[0] if cpc else ""

            store.store_entity_observation(
                entity_id=company_eid,
                source_tool="patent_filings",
                observed_at=ts,
                observation_type="patent_filing",
                depth_level=2,
                value={
                    "patent_number": patent.get("patent_number", ""),
                    "patent_title": patent.get("patent_title", ""),
                    "cpc_subgroup_id": cpc,
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
                link_type="patents_in",
                source="patent_filings",
                confidence=1.0,
                metadata={"patent_number": patent.get("patent_number", "")},
            )

    def execute(self, **kwargs: Any) -> ToolResult:
        mode = (kwargs.get("mode") or "").strip().lower()
        if mode not in VALID_MODES:
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Use: {', '.join(sorted(VALID_MODES))}",
            )

        if mode == "search":
            return self._search(**kwargs)
        if mode == "trends":
            return self._trends(**kwargs)
        return self._assignee(**kwargs)

    # ── search mode ──────────────────────────────────────────────

    def _search(self, **kwargs: Any) -> ToolResult:
        keyword = (kwargs.get("query") or "").strip()
        assignee = (kwargs.get("assignee") or "").strip()
        cpc = (kwargs.get("cpc_class") or "").strip().upper()
        date_from = (kwargs.get("date_from") or "").strip()
        date_to = (kwargs.get("date_to") or "").strip()
        limit = min(int(kwargs.get("limit") or 25), _MAX_RESULTS)

        if not keyword and not assignee and not cpc:
            return ToolResult(
                success=False,
                output="At least one of 'query', 'assignee', or 'cpc_class' is required for search mode.",
            )

        # Build PatentsView query
        conditions: list[dict[str, Any]] = []
        if keyword:
            conditions.append({"_text_any": {"patent_abstract": keyword}})
        if assignee:
            conditions.append({"_text_any": {"assignee_organization": assignee}})
        if cpc:
            conditions.append({"_begins": {"cpc_subgroup_id": cpc}})
        if date_from:
            conditions.append({"_gte": {"patent_date": _parse_date(date_from)}})
        if date_to:
            conditions.append({"_lte": {"patent_date": _parse_date(date_to)}})

        q = conditions[0] if len(conditions) == 1 else {"_and": conditions}

        fields = [
            "patent_number",
            "patent_title",
            "patent_date",
            "assignee_organization",
            "cpc_subgroup_id",
            "patent_abstract",
        ]

        cache_key = {"mode": "search", "q": str(q), "limit": limit}
        if self._cache:
            cached = self._cache.get("patent_filings", cache_key)
            if cached is not None:
                return self._format_search(cached, from_cache=True)

        data = _fetch_patents(q, fields, [{"patent_date": "desc"}], limit)
        if data is None:
            return ToolResult(success=False, output="USPTO API unavailable.")

        patents = data.get("patents") or []
        total = data.get("total_patent_count", len(patents))

        result_data = {
            "patents": patents,
            "total_count": total,
            "returned": len(patents),
        }

        if self._cache:
            self._cache.put("patent_filings", cache_key, result_data)

        return self._format_search(result_data)

    def _format_search(
        self, data: dict[str, Any], *, from_cache: bool = False
    ) -> ToolResult:
        tag = " (cached)" if from_cache else ""
        patents = data.get("patents", [])
        total = data.get("total_count", 0)
        lines = [
            f"# USPTO Patent Search{tag}",
            f"Total matches: {total:,} | Returned: {len(patents)}\n",
        ]
        for p in patents[:25]:
            title = (p.get("patent_title") or "Untitled")[:80]
            num = p.get("patent_number", "?")
            date = p.get("patent_date", "?")
            assignee = p.get("assignee_organization", "Unknown")
            if isinstance(assignee, list):
                assignee = assignee[0] if assignee else "Unknown"
            cpc = p.get("cpc_subgroup_id", "")
            if isinstance(cpc, list):
                cpc = cpc[0] if cpc else ""
            lines.append(f"**{num}** ({date}) — {title}")
            lines.append(f"  Assignee: {assignee} | CPC: {cpc}")

        self._persist_entities(patents)

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"mode": "search", **data},
        )

    # ── trends mode ──────────────────────────────────────────────

    def _trends(self, **kwargs: Any) -> ToolResult:
        cpc = (kwargs.get("cpc_class") or "").strip().upper()

        if not cpc:
            # Return overview of signal CPC classes
            lines = ["# Key CPC Technology Classes\n"]
            for code, desc in sorted(SIGNAL_CPC.items()):
                lines.append(f"  **{code}**: {desc}")
            lines.append("\nUse 'cpc_class' parameter to see filing trends.")
            return ToolResult(
                success=True,
                output="\n".join(lines),
                data={"mode": "trends", "signal_classes": SIGNAL_CPC},
            )

        # Query patents by CPC class in yearly buckets
        start_date, end_date = _year_range(5)

        cache_key = {"mode": "trends", "cpc": cpc, "start": start_date}
        if self._cache:
            cached = self._cache.get("patent_filings", cache_key)
            if cached is not None:
                return self._format_trends(cached, cpc, from_cache=True)

        q: dict[str, Any] = {
            "_and": [
                {"_begins": {"cpc_subgroup_id": cpc}},
                {"_gte": {"patent_date": start_date}},
            ]
        }
        fields = ["patent_number", "patent_date", "cpc_subgroup_id"]
        data = _fetch_patents(q, fields, [{"patent_date": "desc"}], _MAX_RESULTS)

        if data is None:
            return ToolResult(success=False, output="USPTO API unavailable.")

        patents = data.get("patents") or []
        total = data.get("total_patent_count", len(patents))

        # Aggregate by year
        yearly: dict[str, int] = {}
        for p in patents:
            date = p.get("patent_date", "")
            if date and len(date) >= 4:
                year = date[:4]
                yearly[year] = yearly.get(year, 0) + 1

        result_data = {
            "cpc_class": cpc,
            "yearly_counts": yearly,
            "total_count": total,
            "sample_size": len(patents),
        }

        if self._cache:
            self._cache.put("patent_filings", cache_key, result_data)

        return self._format_trends(result_data, cpc)

    def _format_trends(
        self, data: dict[str, Any], cpc: str, *, from_cache: bool = False
    ) -> ToolResult:
        tag = " (cached)" if from_cache else ""
        cpc_desc = SIGNAL_CPC.get(cpc, CPC_CLASSES.get(cpc[:1], "Unknown"))
        yearly = data.get("yearly_counts", {})
        total = data.get("total_count", 0)

        lines = [
            f"# Patent Filing Trends: {cpc} — {cpc_desc}{tag}",
            f"Total filings: {total:,}\n",
        ]

        if yearly:
            sorted_years = sorted(yearly.items())
            for year, count in sorted_years:
                bar = "█" * min(count, 40)
                lines.append(f"  {year}: {count:>5} {bar}")

            # Trend direction
            if len(sorted_years) >= 2:
                first_count = sorted_years[0][1]
                last_count = sorted_years[-1][1]
                if first_count > 0:
                    trend_pct = ((last_count - first_count) / first_count) * 100
                    direction = (
                        "accelerating"
                        if trend_pct > 10
                        else "decelerating" if trend_pct < -10 else "stable"
                    )
                    lines.append(
                        f"\nTrend: {direction} ({trend_pct:+.1f}% over period)"
                    )
        else:
            lines.append("  No filings found for this CPC class.")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"mode": "trends", **data},
        )

    # ── assignee mode ────────────────────────────────────────────

    def _assignee(self, **kwargs: Any) -> ToolResult:
        assignee_name = (kwargs.get("assignee") or "").strip()
        if not assignee_name:
            return ToolResult(
                success=False,
                output="Parameter 'assignee' required (company name, e.g., 'Apple', 'Google').",
            )

        limit = min(int(kwargs.get("limit") or 25), _MAX_RESULTS)

        cache_key = {"mode": "assignee", "name": assignee_name, "limit": limit}
        if self._cache:
            cached = self._cache.get("patent_filings", cache_key)
            if cached is not None:
                return self._format_assignee(cached, assignee_name, from_cache=True)

        q: dict[str, Any] = {"_text_any": {"assignee_organization": assignee_name}}
        fields = [
            "patent_number",
            "patent_title",
            "patent_date",
            "assignee_organization",
            "cpc_subgroup_id",
        ]
        data = _fetch_patents(q, fields, [{"patent_date": "desc"}], limit)

        if data is None:
            return ToolResult(success=False, output="USPTO API unavailable.")

        patents = data.get("patents") or []
        total = data.get("total_patent_count", len(patents))

        # Analyze CPC distribution
        cpc_counts: dict[str, int] = {}
        for p in patents:
            cpc = p.get("cpc_subgroup_id", "")
            if isinstance(cpc, list):
                for c in cpc:
                    prefix = c[:4] if len(c) >= 4 else c
                    cpc_counts[prefix] = cpc_counts.get(prefix, 0) + 1
            elif cpc:
                prefix = cpc[:4] if len(cpc) >= 4 else cpc
                cpc_counts[prefix] = cpc_counts.get(prefix, 0) + 1

        # Yearly filing velocity
        yearly: dict[str, int] = {}
        for p in patents:
            date = p.get("patent_date", "")
            if date and len(date) >= 4:
                yearly[date[:4]] = yearly.get(date[:4], 0) + 1

        result_data = {
            "assignee": assignee_name,
            "total_patents": total,
            "returned": len(patents),
            "patents": patents,
            "cpc_distribution": cpc_counts,
            "yearly_velocity": yearly,
        }

        if self._cache:
            self._cache.put("patent_filings", cache_key, result_data)

        return self._format_assignee(result_data, assignee_name)

    def _format_assignee(
        self, data: dict[str, Any], assignee: str, *, from_cache: bool = False
    ) -> ToolResult:
        tag = " (cached)" if from_cache else ""
        patents = data.get("patents", [])
        total = data.get("total_patents", 0)
        cpc_dist = data.get("cpc_distribution", {})
        yearly = data.get("yearly_velocity", {})

        lines = [
            f"# Patent Portfolio: {assignee}{tag}",
            f"Total patents: {total:,} | Returned: {len(patents)}\n",
        ]

        # Technology focus
        if cpc_dist:
            lines.append("**Technology Focus (CPC):**")
            top_cpc = sorted(cpc_dist.items(), key=lambda x: x[1], reverse=True)[:10]
            for code, count in top_cpc:
                desc = SIGNAL_CPC.get(code, "")
                label = f" ({desc})" if desc else ""
                lines.append(f"  {code}{label}: {count}")

        # Filing velocity
        if yearly:
            lines.append("\n**Filing Velocity (yearly):**")
            for year, count in sorted(yearly.items()):
                lines.append(f"  {year}: {count}")

        # Recent patents
        if patents:
            lines.append("\n**Recent Patents:**")
            for p in patents[:10]:
                title = (p.get("patent_title") or "Untitled")[:70]
                num = p.get("patent_number", "?")
                date = p.get("patent_date", "?")
                lines.append(f"  {num} ({date}): {title}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"mode": "assignee", **data},
        )
