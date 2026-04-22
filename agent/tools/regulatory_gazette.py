"""
Tool: Regulatory Gazette — US Federal Regulatory Pipeline Monitor

US Federal Register API + UK legislation.gov.uk — free, no auth.

Four modes:
  recent  — Latest rules / proposed rules. Optional agency, keyword, type filter.
             Default: proposed rules from last 7 days. Shows what the government
             is about to do before it does it.
  search  — Full-text keyword search across Federal Register (rules + proposed).
             "What regulation mentions <company/technology/sector>?"
  agency  — All recent rules from a specific agency (SEC, FDA, FERC, etc.).
             Track regulatory posture toward specific sectors.
  upcoming — Open comment period rules (comments_close_on > today).
             The strongest leading indicator: these WILL become law in some form,
             and you can see the exact timeline.

Signal theory:
  - Proposed rules with open comment periods = 30-90 day lead before final rule
  - "Significant" flag = economically significant (>$100M impact estimated)
  - Agency clustering = coordinated regulatory wave (SEC+CFTC+FTC same topic)
  - New keyword surge in CFR titles = emerging policy focus

API: https://www.federalregister.gov/api/v1/
Auth: None. Free. 470+ agencies. JSON. Up to 1000 results per page.
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

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

_FR_BASE = "https://www.federalregister.gov/api/v1"
_UK_BASE = "https://www.legislation.gov.uk"
_UA = "TirraMind/0.1 (research; https://github.com/tirramind)"
_TIMEOUT = 15

# Document types recognized by the Federal Register API
_VALID_TYPES = {"RULE", "PRORULE", "NOTICE", "PRESDOCU"}

# Market-relevant agency slugs — pre-curated for the agent.
# The full list of 470+ is available via agencies endpoint.
MARKET_AGENCIES: dict[str, dict[str, Any]] = {
    "sec": {
        "slug": "securities-and-exchange-commission",
        "id": 466,
        "sector": "finance",
    },
    "fed": {"slug": "federal-reserve-system", "id": 188, "sector": "finance"},
    "cftc": {
        "slug": "commodity-futures-trading-commission",
        "id": 77,
        "sector": "finance",
    },
    "ftc": {"slug": "federal-trade-commission", "id": 192, "sector": "consumer"},
    "epa": {"slug": "environmental-protection-agency", "id": 145, "sector": "energy"},
    "fda": {"slug": "food-and-drug-administration", "id": 199, "sector": "health"},
    "fcc": {
        "slug": "federal-communications-commission",
        "id": 161,
        "sector": "telecom",
    },
    "ferc": {
        "slug": "federal-energy-regulatory-commission",
        "id": 167,
        "sector": "energy",
    },
    "treasury": {"slug": "treasury-department", "id": 497, "sector": "finance"},
    "doj": {"slug": "justice-department", "id": 268, "sector": "legal"},
    "dod": {"slug": "defense-department", "id": 103, "sector": "defense"},
    "commerce": {"slug": "commerce-department", "id": 54, "sector": "trade"},
    "energy": {"slug": "energy-department", "id": 136, "sector": "energy"},
    "dot": {"slug": "transportation-department", "id": 492, "sector": "transport"},
    "usda": {"slug": "agriculture-department", "id": 12, "sector": "agriculture"},
    "cfpb": {
        "slug": "consumer-financial-protection-bureau",
        "id": 573,
        "sector": "finance",
    },
    "nrc": {"slug": "nuclear-regulatory-commission", "id": 383, "sector": "energy"},
    "interior": {"slug": "interior-department", "id": 253, "sector": "resources"},
    "hhs": {
        "slug": "health-and-human-services-department",
        "id": 221,
        "sector": "health",
    },
    "labor": {"slug": "labor-department", "id": 271, "sector": "labor"},
}

# Fields we request from the API — balance between completeness and payload size
_FIELDS = [
    "title",
    "type",
    "abstract",
    "document_number",
    "publication_date",
    "agencies",
    "action",
    "comments_close_on",
    "effective_on",
    "docket_ids",
    "topics",
    "significant",
    "regulation_id_number_info",
    "cfr_references",
    "page_length",
    "html_url",
]


def _resolve_agency(alias: str) -> str:
    """Resolve a short alias (sec, fda, fed) to the Federal Register slug."""
    alias_lower = alias.strip().lower()
    if alias_lower in MARKET_AGENCIES:
        return MARKET_AGENCIES[alias_lower]["slug"]
    # Maybe they passed the full slug already
    return alias_lower


def _parse_date(s: str) -> str | None:
    """Parse a date string to YYYY-MM-DD, or None."""
    s = s.strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _safe_int(v: Any, default: int = 0) -> int:
    """Safely convert to int."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _format_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Federal Register document into a clean dict."""
    agencies = doc.get("agencies") or []
    agency_names = [a.get("name") or a.get("raw_name", "Unknown") for a in agencies]
    return {
        "title": (doc.get("title") or "").strip(),
        "type": doc.get("type", ""),
        "document_number": doc.get("document_number", ""),
        "publication_date": doc.get("publication_date", ""),
        "agencies": agency_names,
        "abstract": (doc.get("abstract") or "")[:500],
        "action": (doc.get("action") or "").strip(),
        "comments_close_on": doc.get("comments_close_on"),
        "effective_on": doc.get("effective_on"),
        "topics": doc.get("topics") or [],
        "significant": doc.get("significant"),
        "docket_ids": doc.get("docket_ids") or [],
        "page_length": _safe_int(doc.get("page_length")),
        "url": doc.get("html_url", ""),
    }


def _days_until(date_str: str | None) -> int | None:
    """Days from now until a date string (YYYY-MM-DD). Negative = past."""
    if not date_str:
        return None
    try:
        target = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (target - now).days
    except (ValueError, TypeError):
        return None


class RegulatoryGazetteTool(Tool):
    name = "regulatory_gazette"
    description = (
        "Monitor the US federal regulatory pipeline via the Federal Register. "
        "Mode 'recent' shows latest rules/proposed rules (default: proposed rules "
        "from last 7 days). Mode 'search' does keyword search across all rules. "
        "Mode 'agency' filters by regulator (sec, fda, fed, epa, etc.). "
        "Mode 'upcoming' shows rules with open comment periods — the clearest "
        "leading indicator of future regulation. "
        "Free, no API key, 470+ agencies, JSON API."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["recent", "search", "agency", "upcoming"],
                "default": "recent",
                "description": (
                    "recent = latest rules/proposed rules. "
                    "search = keyword search. "
                    "agency = filter by regulator. "
                    "upcoming = open comment period rules."
                ),
            },
            "keyword": {
                "type": "string",
                "description": (
                    "Search term for 'search' mode. Also used as optional "
                    "additional filter in 'recent' and 'agency' modes. "
                    "Example: 'semiconductor', 'crypto', 'emissions'"
                ),
                "default": "",
            },
            "agency": {
                "type": "string",
                "description": (
                    "Agency filter. Short alias (sec, fda, fed, cftc, epa, "
                    "fcc, ferc, doj, dod, commerce, energy, dot, usda, cfpb, "
                    "nrc, treasury, interior, hhs, labor) or full slug. "
                    "Used in 'agency' mode, optional in 'recent'."
                ),
                "default": "",
            },
            "doc_type": {
                "type": "string",
                "description": (
                    "Document type filter: RULE, PRORULE (proposed rule), "
                    "NOTICE, PRESDOCU. Comma-separated for multiple. "
                    "Default: RULE,PRORULE"
                ),
                "default": "RULE,PRORULE",
            },
            "days_back": {
                "type": "integer",
                "description": "How many days of history. Default 7. Max 365.",
                "default": 7,
            },
            "significant_only": {
                "type": "boolean",
                "description": (
                    "If true, only return economically significant rules "
                    "(>$100M estimated impact). Default false."
                ),
                "default": False,
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return. Default 25, max 200.",
                "default": 25,
            },
        },
        "required": [],
    }

    def __init__(
        self,
        cache: DataCache | None = None,
        pipeline_store: "PipelineStore | None" = None,
    ) -> None:
        self._cache = cache
        self._store = pipeline_store

    def execute(
        self,
        *,
        mode: str = "recent",
        keyword: str = "",
        agency: str = "",
        doc_type: str = "RULE,PRORULE",
        days_back: int = 7,
        significant_only: bool = False,
        limit: int = 25,
        **_: Any,
    ) -> ToolResult:
        mode = mode.lower().strip()
        if mode not in ("recent", "search", "agency", "upcoming"):
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Use 'recent', 'search', 'agency', or 'upcoming'.",
            )

        keyword = keyword.strip()
        agency = agency.strip()
        days_back = max(1, min(days_back, 365))
        limit = max(1, min(limit, 200))

        # Parse doc_type
        types = _parse_doc_types(doc_type)

        if mode == "search":
            if not keyword:
                return ToolResult(
                    success=False,
                    output="Search mode requires a 'keyword' parameter.",
                )
            result = self._execute_search(
                keyword=keyword,
                types=types,
                days_back=days_back,
                significant_only=significant_only,
                limit=limit,
            )
            if result.success and result.data:
                self._persist_entities(result.data, mode)
            return result

        if mode == "agency":
            if not agency:
                return self._list_agencies()
            result = self._execute_agency(
                agency=agency,
                keyword=keyword,
                types=types,
                days_back=days_back,
                significant_only=significant_only,
                limit=limit,
            )
            if result.success and result.data:
                self._persist_entities(result.data, mode)
            return result

        if mode == "upcoming":
            result = self._execute_upcoming(
                keyword=keyword,
                agency=agency,
                significant_only=significant_only,
                limit=limit,
            )
            if result.success and result.data:
                self._persist_entities(result.data, mode)
            return result

        # mode == "recent"
        result = self._execute_recent(
            keyword=keyword,
            agency=agency,
            types=types,
            days_back=days_back,
            significant_only=significant_only,
            limit=limit,
        )
        if result.success and result.data:
            self._persist_entities(result.data, mode)
        return result

    # ------------------------------------------------------------------
    # recent mode
    # ------------------------------------------------------------------

    def _execute_recent(
        self,
        *,
        keyword: str,
        agency: str,
        types: list[str],
        days_back: int,
        significant_only: bool,
        limit: int,
    ) -> ToolResult:
        now = datetime.now(timezone.utc)
        date_gte = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")

        params = _build_params(
            types=types,
            date_gte=date_gte,
            keyword=keyword,
            agency=agency,
            per_page=min(limit, 100),
        )

        docs, total, error = self._fetch_fr(params)
        if error:
            return ToolResult(success=False, output=error)

        if significant_only:
            docs = [d for d in docs if d.get("significant") is True]

        formatted = [_format_doc(d) for d in docs[:limit]]
        return self._make_result(
            formatted,
            header=f"Federal Register: Recent ({', '.join(types)}, last {days_back}d)",
            total=total,
            limit=limit,
        )

    # ------------------------------------------------------------------
    # search mode
    # ------------------------------------------------------------------

    def _execute_search(
        self,
        *,
        keyword: str,
        types: list[str],
        days_back: int,
        significant_only: bool,
        limit: int,
    ) -> ToolResult:
        now = datetime.now(timezone.utc)
        date_gte = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")

        params = _build_params(
            types=types,
            date_gte=date_gte,
            keyword=keyword,
            per_page=min(limit, 100),
        )

        docs, total, error = self._fetch_fr(params)
        if error:
            return ToolResult(success=False, output=error)

        if significant_only:
            docs = [d for d in docs if d.get("significant") is True]

        formatted = [_format_doc(d) for d in docs[:limit]]
        return self._make_result(
            formatted,
            header=f"Federal Register Search: \"{keyword}\" ({', '.join(types)}, last {days_back}d)",
            total=total,
            limit=limit,
        )

    # ------------------------------------------------------------------
    # agency mode
    # ------------------------------------------------------------------

    def _execute_agency(
        self,
        *,
        agency: str,
        keyword: str,
        types: list[str],
        days_back: int,
        significant_only: bool,
        limit: int,
    ) -> ToolResult:
        slug = _resolve_agency(agency)
        now = datetime.now(timezone.utc)
        date_gte = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")

        params = _build_params(
            types=types,
            date_gte=date_gte,
            keyword=keyword,
            agency=slug,
            per_page=min(limit, 100),
        )

        docs, total, error = self._fetch_fr(params)
        if error:
            return ToolResult(success=False, output=error)

        if significant_only:
            docs = [d for d in docs if d.get("significant") is True]

        formatted = [_format_doc(d) for d in docs[:limit]]
        return self._make_result(
            formatted,
            header=f"Federal Register: {slug} ({', '.join(types)}, last {days_back}d)",
            total=total,
            limit=limit,
        )

    def _list_agencies(self) -> ToolResult:
        """Return the curated list of market-relevant agencies."""
        lines = [
            "Market-Relevant Federal Agencies (use alias in 'agency' parameter):",
            "",
        ]
        for alias, info in sorted(MARKET_AGENCIES.items()):
            lines.append(f"  {alias:12s}  {info['slug']:50s}  sector={info['sector']}")
        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"agencies": MARKET_AGENCIES},
        )

    # ------------------------------------------------------------------
    # upcoming mode
    # ------------------------------------------------------------------

    def _execute_upcoming(
        self,
        *,
        keyword: str,
        agency: str,
        significant_only: bool,
        limit: int,
    ) -> ToolResult:
        """Documents with comment periods still open (comments_close_on >= today)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Fetch proposed rules — they're the ones with comment periods
        params = _build_params(
            types=["PRORULE"],
            keyword=keyword,
            agency=agency if agency else "",
            per_page=min(limit * 2, 100),  # over-fetch since we filter
        )

        docs, total, error = self._fetch_fr(params)
        if error:
            return ToolResult(success=False, output=error)

        # Filter to only those with open comment periods
        upcoming = []
        for doc in docs:
            close_date = doc.get("comments_close_on")
            if close_date and close_date >= today:
                upcoming.append(doc)

        if significant_only:
            upcoming = [d for d in upcoming if d.get("significant") is True]

        formatted = [_format_doc(d) for d in upcoming[:limit]]

        # Annotate with days remaining
        for f in formatted:
            days = _days_until(f.get("comments_close_on"))
            f["days_remaining"] = days

        return self._make_upcoming_result(formatted, total_scanned=total, limit=limit)

    def _make_upcoming_result(
        self,
        docs: list[dict[str, Any]],
        *,
        total_scanned: int,
        limit: int,
    ) -> ToolResult:
        if not docs:
            return ToolResult(
                success=True,
                output="Federal Register: No rules with open comment periods found.",
                data={"documents": [], "count": 0},
            )

        # Sort by closest deadline first
        docs.sort(key=lambda d: d.get("comments_close_on") or "9999-99-99")

        lines = [
            f"Federal Register: Open Comment Periods ({len(docs)} rules, {total_scanned} scanned):",
            "",
        ]
        for doc in docs:
            days = doc.get("days_remaining")
            days_str = f"{days}d left" if days is not None else "?"
            sig = " [SIGNIFICANT]" if doc.get("significant") else ""
            agencies_str = ", ".join(doc.get("agencies", [])[:2])
            lines.append(f"  [{days_str:>8s}] {doc['title'][:80]}")
            lines.append(
                f"           Agency: {agencies_str}  Close: {doc.get('comments_close_on', '?')}{sig}"
            )
            if doc.get("abstract"):
                lines.append(f"           {doc['abstract'][:120]}")
            lines.append("")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"documents": docs, "count": len(docs)},
        )

    # ------------------------------------------------------------------
    # Federal Register API fetch
    # ------------------------------------------------------------------

    def _fetch_fr(
        self,
        params: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], int, str | None]:
        """Fetch from Federal Register API. Returns (docs, total_count, error_msg)."""
        # Build cache key from params
        cache_key = {k: v for k, v in sorted(params.items()) if v}
        if self._cache:
            cached = self._cache.get("regulatory_gazette", cache_key)
            if cached is not None:
                return cached.get("results", []), cached.get("count", 0), None

        # Build URL with proper array parameter encoding
        url = f"{_FR_BASE}/documents.json"
        query_string = _encode_fr_params(params)

        try:
            with httpx.Client(
                timeout=_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": _UA},
            ) as client:
                resp = client.get(f"{url}?{query_string}")
                if resp.status_code == 400:
                    return (
                        [],
                        0,
                        f"Federal Register API: Bad request (check agency slug or parameters)",
                    )
                if resp.status_code == 429:
                    return (
                        [],
                        0,
                        "Federal Register API: Rate limited. Try again shortly.",
                    )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            return [], 0, f"Federal Register API error: HTTP {exc.response.status_code}"
        except httpx.TimeoutException:
            return [], 0, "Federal Register API: Request timed out."
        except Exception as exc:
            log.exception("Federal Register fetch failed")
            return [], 0, f"Federal Register fetch error: {exc}"

        results = data.get("results", [])
        count = data.get("count", len(results))

        if self._cache and results:
            self._cache.put("regulatory_gazette", cache_key, data, ttl=7200)

        return results, count, None

    # ------------------------------------------------------------------
    # Result formatting
    # ------------------------------------------------------------------

    def _make_result(
        self,
        docs: list[dict[str, Any]],
        *,
        header: str,
        total: int,
        limit: int,
    ) -> ToolResult:
        if not docs:
            return ToolResult(
                success=True,
                output=f"{header}: No documents found.",
                data={"documents": [], "count": 0, "total": total},
            )

        lines = [f"{header} — {len(docs)} of {total:,} results:", ""]
        for doc in docs:
            sig = " [SIGNIFICANT]" if doc.get("significant") else ""
            agencies_str = ", ".join(doc.get("agencies", [])[:2])
            comment_info = ""
            if doc.get("comments_close_on"):
                days = _days_until(doc["comments_close_on"])
                if days is not None and days >= 0:
                    comment_info = f"  [comments close in {days}d]"
                elif days is not None:
                    comment_info = f"  [comments closed {abs(days)}d ago]"

            lines.append(
                f"  {doc['publication_date']}  {doc['type']:14s}  "
                f"{doc['title'][:72]}{sig}"
            )
            lines.append(f"    Agency: {agencies_str}{comment_info}")
            if doc.get("abstract"):
                lines.append(f"    {doc['abstract'][:140]}")
            lines.append("")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"documents": docs, "count": len(docs), "total": total},
        )

    # ------------------------------------------------------------------
    # L2 entity persistence
    # ------------------------------------------------------------------

    # Reverse lookup: agency slug → short alias for entity key
    _SLUG_TO_ALIAS: dict[str, str] = {
        info["slug"]: alias for alias, info in MARKET_AGENCIES.items()
    }

    def _persist_entities(self, data: dict[str, Any], mode: str) -> dict[str, int]:
        if self._store is None or _entity_id_from_key is None:
            return {"regulatory_velocity_obs": 0}
        try:
            return self._persist_entities_inner(data, mode)
        except Exception:
            log.exception("Regulatory gazette entity persistence failed (non-fatal)")
            return {"regulatory_velocity_obs": 0}

    def _persist_entities_inner(
        self, data: dict[str, Any], mode: str
    ) -> dict[str, int]:
        assert self._store is not None
        assert _entity_id_from_key is not None

        docs = data.get("documents", [])
        if not docs:
            return {"regulatory_velocity_obs": 0}

        # Aggregate per agency
        agency_stats: dict[str, dict[str, Any]] = {}
        for doc in docs:
            agencies = doc.get("agencies", [])
            for agency_name in agencies:
                if not agency_name or not agency_name.strip():
                    continue
                # Resolve to short alias if known
                slug = agency_name.lower().replace(" ", "-")
                key = self._SLUG_TO_ALIAS.get(slug, "")
                if not key:
                    # Try matching from MARKET_AGENCIES by name substring
                    for alias, info in MARKET_AGENCIES.items():
                        if info["slug"] in slug or slug in info["slug"]:
                            key = alias
                            break
                if not key:
                    key = agency_name.strip().lower().replace(" ", "_")[:40]
                if not key:
                    continue

                if key not in agency_stats:
                    agency_stats[key] = {
                        "name": agency_name.strip(),
                        "doc_count": 0,
                        "significant_count": 0,
                        "types": set(),
                    }
                agency_stats[key]["doc_count"] += 1
                if doc.get("significant"):
                    agency_stats[key]["significant_count"] += 1
                if doc.get("type"):
                    agency_stats[key]["types"].add(doc["type"])

        count = 0
        now = time.time()
        for agency_key, stats in agency_stats.items():
            eid = _entity_id_from_key("organization", agency_key)
            self._store.register_entity("organization", agency_key, eid)
            self._store.store_entity_observation(
                entity_id=eid,
                source_tool="regulatory_gazette",
                observed_at=now,
                observation_type="regulatory_velocity",
                value={
                    "mode": mode,
                    "name": stats["name"],
                    "doc_count": stats["doc_count"],
                    "significant_count": stats["significant_count"],
                    "types": sorted(stats["types"]),
                },
                depth_level=2,
            )
            count += 1

        return {"regulatory_velocity_obs": count}


# ------------------------------------------------------------------
# Parameter building helpers
# ------------------------------------------------------------------


def _parse_doc_types(raw: str) -> list[str]:
    """Parse comma-separated doc types, validating against allowed types."""
    if not raw or not raw.strip():
        return ["RULE", "PRORULE"]
    types = []
    for t in raw.upper().split(","):
        t = t.strip()
        if t in _VALID_TYPES:
            types.append(t)
    return types or ["RULE", "PRORULE"]


def _build_params(
    *,
    types: list[str],
    date_gte: str = "",
    keyword: str = "",
    agency: str = "",
    per_page: int = 25,
) -> dict[str, Any]:
    """Build a flat params dict for the Federal Register API."""
    params: dict[str, Any] = {
        "per_page": per_page,
        "order": "newest",
    }
    params["types"] = types
    if date_gte:
        params["date_gte"] = date_gte
    if keyword:
        params["keyword"] = keyword
    if agency:
        slug = _resolve_agency(agency)
        params["agency"] = slug
    return params


def _encode_fr_params(params: dict[str, Any]) -> str:
    """Encode parameters into Federal Register API query string format.

    The FR API uses bracket notation for arrays: conditions[type][]=RULE
    """
    parts: list[str] = []

    parts.append(f"per_page={params.get('per_page', 25)}")
    parts.append(f"order={params.get('order', 'newest')}")

    # Fields
    for f in _FIELDS:
        parts.append(f"fields[]={f}")

    # Types
    for t in params.get("types", []):
        parts.append(f"conditions[type][]={t}")

    # Date filter
    if params.get("date_gte"):
        parts.append(f"conditions[publication_date][gte]={params['date_gte']}")

    # Keyword
    if params.get("keyword"):
        parts.append(f"conditions[term]={_url_encode_value(params['keyword'])}")

    # Agency
    if params.get("agency"):
        parts.append(f"conditions[agencies][]={params['agency']}")

    return "&".join(parts)


def _url_encode_value(s: str) -> str:
    """Minimal URL encoding for query parameter values."""
    return (
        s.replace(" ", "+").replace("&", "%26").replace("=", "%3D").replace("#", "%23")
    )
