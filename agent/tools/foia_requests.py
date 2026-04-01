"""
Tool: FOIA / FOI Request Logs — Investigation Formation Detector

MuckRock:          https://www.muckrock.com/api_v1/  (free, no auth)
WhatDoTheyKnow:   https://www.whatdotheyknow.com/    (free, Alaveteli API)

Clusters of information requests about the same entity signal investigation
formation.  When multiple journalists or researchers independently file FOIA
requests targeting the same agency or topic, it's a leading indicator of
upcoming disclosures, regulatory action, or investigative journalism — often
weeks before stories break.

Nobody auto-monitors FOIA request patterns as a *predictive signal*.  The
requests themselves are public.  The cluster analysis is the edge.

Modes:
  search           — Search FOIA/FOI requests by keyword/entity/topic.
  agency_activity  — Request volume for a specific agency.  Detects surges.
  entity_cluster   — Find all requests mentioning an entity across agencies
                     and jurisdictions.  Detects investigation convergence.

Data sources:
  MuckRock (US federal + state + local, 250K+ requests)
  WhatDoTheyKnow (UK, every FOI request filed, Alaveteli-based)

Signal theory:
  - Surge in requests to one agency = upcoming enforcement or disclosure
  - Same entity queried across agencies = multi-agency investigation forming
  - Same entity queried across jurisdictions (US + UK) = international probe
  - Journalist clusters on a topic = story about to break
  - Cross-reference: FOIA surge + insider selling + DNS changes = crisis
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MUCKROCK_BASE = "https://www.muckrock.com/api_v1"
_WDTK_BASE = "https://www.whatdotheyknow.com"
_UA = "TirraMind/0.1"
_TIMEOUT = 20
_CACHE_TTL = 1800  # 30 min — FOIA data doesn't change rapidly
_MAX_PAGES = 5  # Max pagination depth per source
_DEFAULT_PAGE_SIZE = 20
_MAX_LIMIT = 100

VALID_MODES = frozenset({"search", "agency_activity", "entity_cluster"})

# MuckRock status codes
_STATUS_MAP: dict[str, str] = {
    "submitted": "Submitted",
    "ack": "Acknowledged",
    "processed": "Processed",
    "appealing": "Appealing",
    "fix": "Fix Required",
    "payment": "Payment Required",
    "rejected": "Rejected",
    "no_docs": "No Responsive Docs",
    "done": "Completed",
    "partial": "Partially Completed",
    "abandoned": "Abandoned",
    "lawsuit": "Lawsuit Filed",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_date(date_str: str | None) -> datetime | None:
    """Parse date string from MuckRock or WDTK. Returns None on failure."""
    if not date_str:
        return None
    # MuckRock uses  "2025-06-15" or "2025-06-15T14:30:00-05:00"
    # WDTK uses ISO 8601 "2025-06-15T14:30:00.000+00:00"
    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
    ):
        try:
            return datetime.strptime(date_str[:32], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    # Fallback: try just the date portion
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _normalize_status(raw: str | None) -> str:
    """Normalize a MuckRock status string."""
    if not raw:
        return "Unknown"
    return _STATUS_MAP.get(raw.lower().strip(), raw.strip().title())


def _normalize_muckrock(rec: dict[str, Any]) -> dict[str, Any]:
    """Normalize a MuckRock FOIA request record."""
    return {
        "title": (rec.get("title") or "Untitled").strip(),
        "agency": (
            (
                rec.get("agency_name")
                or rec.get("agency", {}).get("name", "Unknown agency")
            ).strip()
            if isinstance(rec.get("agency"), dict)
            else (
                rec.get("agency_name") or str(rec.get("agency", "Unknown agency"))
            ).strip()
        ),
        "status": _normalize_status(rec.get("status")),
        "date_filed": (
            rec.get("datetime_submitted") or rec.get("date_submitted") or ""
        )[:10],
        "date_done": (rec.get("datetime_done") or rec.get("date_done") or "")[:10],
        "jurisdiction": (rec.get("jurisdiction_name") or "US").strip(),
        "source": "muckrock",
        "url": rec.get("absolute_url") or rec.get("url", ""),
        "requester": (
            (rec.get("user", {}).get("username", "anonymous"))
            if isinstance(rec.get("user"), dict)
            else str(rec.get("user", "anonymous"))
        ),
    }


def _normalize_wdtk(rec: dict[str, Any]) -> dict[str, Any]:
    """Normalize a WhatDoTheyKnow FOI request record."""
    return {
        "title": (rec.get("title") or "Untitled").strip(),
        "agency": (
            (rec.get("public_body", {}).get("name", "Unknown body")).strip()
            if isinstance(rec.get("public_body"), dict)
            else str(rec.get("public_body", "Unknown body")).strip()
        ),
        "status": (rec.get("described_state") or "unknown").replace("_", " ").title(),
        "date_filed": (rec.get("created_at") or "")[:10],
        "date_done": "",
        "jurisdiction": "UK",
        "source": "wdtk",
        "url": rec.get("url") or "",
        "requester": (
            (rec.get("user", {}).get("name", "anonymous"))
            if isinstance(rec.get("user"), dict)
            else "anonymous"
        ),
    }


def _format_request(req: dict[str, Any], *, index: int = 0) -> str:
    """Format a normalized request record for text output."""
    parts = [f"  {index}. [{req['date_filed'] or '?'}] {req['title']}"]
    parts.append(f"     Agency: {req['agency']} ({req['jurisdiction']})")
    parts.append(f"     Status: {req['status']}  |  Source: {req['source']}")
    if req.get("url"):
        parts.append(f"     URL: {req['url']}")
    return "\n".join(parts)


def _cache_key(mode: str, **kwargs: Any) -> str:
    """Build a deterministic cache key."""
    raw = f"foia:{mode}:" + "&".join(f"{k}={v}" for k, v in sorted(kwargs.items()) if v)
    h = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"foia:{mode}:{h}"


# ---------------------------------------------------------------------------
# API fetchers
# ---------------------------------------------------------------------------


def _fetch_muckrock(
    endpoint: str,
    params: dict[str, Any] | None = None,
    *,
    max_pages: int = _MAX_PAGES,
) -> list[dict[str, Any]]:
    """Fetch paginated results from MuckRock API.

    Returns list of raw records. On error, returns empty list (graceful).
    """
    url = f"{_MUCKROCK_BASE}/{endpoint.lstrip('/')}/"
    all_results: list[dict[str, Any]] = []
    params = dict(params or {})
    params.setdefault("page_size", _DEFAULT_PAGE_SIZE)

    for _page in range(max_pages):
        try:
            resp = httpx.get(
                url,
                params=params,
                headers={"User-Agent": _UA, "Accept": "application/json"},
                timeout=_TIMEOUT,
                follow_redirects=True,
            )
            if resp.status_code == 429:
                log.warning("MuckRock rate limit hit")
                break
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException:
            log.warning("MuckRock timeout: %s", url)
            break
        except httpx.HTTPStatusError as exc:
            log.warning("MuckRock HTTP %s: %s", exc.response.status_code, url)
            break
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("MuckRock error: %s", exc)
            break

        # DRF paginated response: {"count": N, "next": url, "results": [...]}
        if isinstance(data, dict):
            results = data.get("results", [])
            all_results.extend(results)
            next_url = data.get("next")
            if not next_url or not results:
                break
            # Use next URL directly for subsequent pages
            url = next_url
            params = {}  # params are embedded in next URL
        elif isinstance(data, list):
            all_results.extend(data)
            break
        else:
            break

    return all_results


def _fetch_wdtk(query: str, *, max_results: int = 50) -> list[dict[str, Any]]:
    """Fetch FOI requests from WhatDoTheyKnow (Alaveteli API).

    Returns list of raw records. On error, returns empty list (graceful).
    """
    url = f"{_WDTK_BASE}/api/v2/requests.json"
    params: dict[str, Any] = {
        "query": query,
        "per_page": min(max_results, 50),
        "order": "newest",
    }

    try:
        resp = httpx.get(
            url,
            params=params,
            headers={"User-Agent": _UA, "Accept": "application/json"},
            timeout=_TIMEOUT,
            follow_redirects=True,
        )
        if resp.status_code == 429:
            log.warning("WDTK rate limit hit")
            return []
        resp.raise_for_status()
        data = resp.json()
    except httpx.TimeoutException:
        log.warning("WDTK timeout")
        return []
    except httpx.HTTPStatusError as exc:
        log.warning("WDTK HTTP %s", exc.response.status_code)
        return []
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("WDTK error: %s", exc)
        return []

    # Alaveteli API returns list or {"requests": [...]} depending on version
    if isinstance(data, list):
        return data[:max_results]
    if isinstance(data, dict):
        return (data.get("requests") or data.get("results") or [])[:max_results]
    return []


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------


class FoiaRequestsTool(Tool):
    """FOIA/FOI request log monitor — investigation formation detector."""

    name = "foia_requests"
    description = (
        "Monitor FOIA/FOI request activity to detect investigation formation. "
        "Mode 'search' finds requests by keyword/entity. "
        "Mode 'agency_activity' measures request volume to a specific agency "
        "and detects surges. "
        "Mode 'entity_cluster' finds all requests about an entity across "
        "agencies and jurisdictions, flagging investigation convergence. "
        "Sources: MuckRock (US) + WhatDoTheyKnow (UK). Free, no API key."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": sorted(VALID_MODES),
                "description": (
                    "agency_activity: request volume for an agency + surge detection. "
                    "entity_cluster: requests about an entity across agencies/jurisdictions. "
                    "search: keyword search across FOIA/FOI requests."
                ),
            },
            "query": {
                "type": "string",
                "description": (
                    "Search term — entity name, topic, or keyword. "
                    "Required for 'search' and 'entity_cluster' modes. "
                    "Examples: 'Boeing', 'PFAS contamination', 'Goldman Sachs'."
                ),
            },
            "agency": {
                "type": "string",
                "description": (
                    "Agency name or keyword for 'agency_activity' mode. "
                    "Examples: 'FBI', 'SEC', 'EPA', 'Department of Defense'."
                ),
            },
            "days_back": {
                "type": "integer",
                "default": 90,
                "description": "Lookback window in days. Default 90, max 365.",
            },
            "jurisdiction": {
                "type": "string",
                "default": "all",
                "description": (
                    "'us' for MuckRock only, 'uk' for WDTK only, "
                    "'all' (default) for both."
                ),
            },
            "limit": {
                "type": "integer",
                "default": 30,
                "description": "Max results to return. Default 30, max 100.",
            },
        },
        "required": ["mode"],
    }

    def __init__(self, cache: DataCache | None = None) -> None:
        self._cache = cache

    # ── Public entry point ─────────────────────────────────────

    def execute(
        self,
        *,
        mode: str = "",
        query: str = "",
        agency: str = "",
        days_back: int = 90,
        jurisdiction: str = "all",
        limit: int = 30,
        **_: Any,
    ) -> ToolResult:
        mode = (mode or "").strip().lower()
        if mode not in VALID_MODES:
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Use: {', '.join(sorted(VALID_MODES))}.",
            )

        query = (query or "").strip()
        agency = (agency or "").strip()
        days_back = max(1, min(int(days_back), 365))
        jurisdiction = (jurisdiction or "all").strip().lower()
        if jurisdiction not in ("us", "uk", "all"):
            jurisdiction = "all"
        limit = max(1, min(int(limit), _MAX_LIMIT))

        if mode == "search":
            if not query:
                return ToolResult(
                    success=False,
                    output="A 'query' parameter is required for search mode.",
                )
            return self._search(
                query=query,
                days_back=days_back,
                jurisdiction=jurisdiction,
                limit=limit,
            )
        elif mode == "agency_activity":
            if not agency:
                return ToolResult(
                    success=False,
                    output="An 'agency' parameter is required for agency_activity mode.",
                )
            return self._agency_activity(
                agency=agency,
                days_back=days_back,
                limit=limit,
            )
        else:  # entity_cluster
            if not query:
                return ToolResult(
                    success=False,
                    output="A 'query' parameter is required for entity_cluster mode.",
                )
            return self._entity_cluster(
                query=query,
                days_back=days_back,
                jurisdiction=jurisdiction,
                limit=limit,
            )

    # ── Mode: search ───────────────────────────────────────────

    def _search(
        self,
        *,
        query: str,
        days_back: int,
        jurisdiction: str,
        limit: int,
    ) -> ToolResult:
        """Search FOIA/FOI requests by keyword."""
        key = _cache_key("search", query=query, days=str(days_back), jur=jurisdiction)
        cached = self._cache.get(key) if self._cache else None
        if cached is not None:
            return ToolResult(success=True, output=cached)

        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        records: list[dict[str, Any]] = []

        # MuckRock (US)
        if jurisdiction in ("us", "all"):
            raw = _fetch_muckrock("foia", {"q": query, "page_size": min(limit, 50)})
            for rec in raw:
                normed = _normalize_muckrock(rec)
                dt = _parse_date(normed["date_filed"])
                if dt and dt >= cutoff:
                    records.append(normed)

        # WDTK (UK)
        if jurisdiction in ("uk", "all"):
            raw = _fetch_wdtk(query, max_results=min(limit, 50))
            for rec in raw:
                normed = _normalize_wdtk(rec)
                dt = _parse_date(normed["date_filed"])
                if dt and dt >= cutoff:
                    records.append(normed)

        # Sort by date descending
        records.sort(
            key=lambda r: r.get("date_filed", ""),
            reverse=True,
        )
        records = records[:limit]

        output = self._format_search_output(query, records, days_back, jurisdiction)
        if self._cache:
            self._cache.put(key, output, ttl=_CACHE_TTL)
        return ToolResult(success=True, output=output)

    def _format_search_output(
        self,
        query: str,
        records: list[dict[str, Any]],
        days_back: int,
        jurisdiction: str,
    ) -> str:
        """Format search results."""
        lines = [
            f'FOIA/FOI Request Search: "{query}"',
            f"Period: last {days_back} days  |  Jurisdiction: {jurisdiction.upper()}",
            f"Results: {len(records)}",
            "",
        ]
        if not records:
            lines.append("No matching requests found.")
            return "\n".join(lines)

        # Source breakdown
        us_count = sum(1 for r in records if r["source"] == "muckrock")
        uk_count = sum(1 for r in records if r["source"] == "wdtk")
        if us_count or uk_count:
            lines.append(f"Sources: MuckRock(US)={us_count}, WDTK(UK)={uk_count}")
            lines.append("")

        # Status breakdown
        statuses: dict[str, int] = {}
        for r in records:
            s = r["status"]
            statuses[s] = statuses.get(s, 0) + 1
        if statuses:
            status_str = ", ".join(f"{k}: {v}" for k, v in sorted(statuses.items()))
            lines.append(f"Status breakdown: {status_str}")
            lines.append("")

        lines.append("Requests:")
        for i, rec in enumerate(records, 1):
            lines.append(_format_request(rec, index=i))
            lines.append("")

        return "\n".join(lines)

    # ── Mode: agency_activity ──────────────────────────────────

    def _agency_activity(
        self,
        *,
        agency: str,
        days_back: int,
        limit: int,
    ) -> ToolResult:
        """Check FOIA request volume for a specific agency. Detect surges."""
        key = _cache_key("agency", agency=agency, days=str(days_back))
        cached = self._cache.get(key) if self._cache else None
        if cached is not None:
            return ToolResult(success=True, output=cached)

        # Fetch requests mentioning this agency
        raw = _fetch_muckrock(
            "foia",
            {"q": agency, "page_size": 50},
            max_pages=3,
        )

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=days_back)
        baseline_start = cutoff - timedelta(days=days_back)  # 2× window for baseline

        recent: list[dict[str, Any]] = []
        baseline: list[dict[str, Any]] = []

        for rec in raw:
            normed = _normalize_muckrock(rec)
            # Filter by agency name match (case-insensitive)
            if agency.lower() not in normed["agency"].lower():
                continue
            dt = _parse_date(normed["date_filed"])
            if not dt:
                continue
            if dt >= cutoff:
                recent.append(normed)
            elif dt >= baseline_start:
                baseline.append(normed)

        # Surge detection
        recent_count = len(recent)
        baseline_count = len(baseline)
        # Normalize to same time period
        baseline_rate = baseline_count / max(days_back, 1)  # per day
        recent_rate = recent_count / max(days_back, 1)

        surge = False
        surge_ratio = 0.0
        if baseline_rate > 0:
            surge_ratio = recent_rate / baseline_rate
            surge = surge_ratio >= 2.0
        elif recent_count >= 3:
            # No baseline but some recent activity — flag as new activity
            surge = True
            surge_ratio = float("inf")

        output = self._format_agency_output(
            agency=agency,
            recent=recent[:limit],
            recent_count=recent_count,
            baseline_count=baseline_count,
            days_back=days_back,
            surge=surge,
            surge_ratio=surge_ratio,
        )
        if self._cache:
            self._cache.put(key, output, ttl=_CACHE_TTL)
        return ToolResult(success=True, output=output)

    def _format_agency_output(
        self,
        *,
        agency: str,
        recent: list[dict[str, Any]],
        recent_count: int,
        baseline_count: int,
        days_back: int,
        surge: bool,
        surge_ratio: float,
    ) -> str:
        """Format agency activity results."""
        surge_flag = "⚠️ SURGE DETECTED" if surge else "Normal"
        ratio_str = f"{surge_ratio:.1f}×" if surge_ratio != float("inf") else "NEW"

        lines = [
            f'FOIA Agency Activity: "{agency}"',
            f"Period: last {days_back} days",
            "",
            f"Recent requests: {recent_count}",
            f"Baseline requests (prior {days_back}d): {baseline_count}",
            f"Activity ratio: {ratio_str} baseline",
            f"Signal: {surge_flag}",
            "",
        ]

        if not recent:
            lines.append("No recent FOIA requests found for this agency.")
            return "\n".join(lines)

        lines.append("Recent requests:")
        for i, rec in enumerate(recent, 1):
            lines.append(_format_request(rec, index=i))
            lines.append("")

        return "\n".join(lines)

    # ── Mode: entity_cluster ───────────────────────────────────

    def _entity_cluster(
        self,
        *,
        query: str,
        days_back: int,
        jurisdiction: str,
        limit: int,
    ) -> ToolResult:
        """Find all requests about an entity, group by agency/jurisdiction."""
        key = _cache_key("entity", query=query, days=str(days_back), jur=jurisdiction)
        cached = self._cache.get(key) if self._cache else None
        if cached is not None:
            return ToolResult(success=True, output=cached)

        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        records: list[dict[str, Any]] = []

        # MuckRock
        if jurisdiction in ("us", "all"):
            raw = _fetch_muckrock(
                "foia",
                {"q": query, "page_size": 50},
                max_pages=3,
            )
            for rec in raw:
                normed = _normalize_muckrock(rec)
                dt = _parse_date(normed["date_filed"])
                if dt and dt >= cutoff:
                    records.append(normed)

        # WDTK
        if jurisdiction in ("uk", "all"):
            raw = _fetch_wdtk(query, max_results=50)
            for rec in raw:
                normed = _normalize_wdtk(rec)
                dt = _parse_date(normed["date_filed"])
                if dt and dt >= cutoff:
                    records.append(normed)

        # Group by agency
        agency_groups: dict[str, list[dict[str, Any]]] = {}
        for rec in records:
            ag = rec["agency"]
            agency_groups.setdefault(ag, []).append(rec)

        # Group by jurisdiction
        jur_set: set[str] = {r["jurisdiction"] for r in records}

        # Convergence detection
        n_agencies = len(agency_groups)
        n_jurisdictions = len(jur_set)
        convergence = n_agencies >= 3 or n_jurisdictions >= 2

        output = self._format_cluster_output(
            query=query,
            records=records[:limit],
            agency_groups=agency_groups,
            jur_set=jur_set,
            n_agencies=n_agencies,
            n_jurisdictions=n_jurisdictions,
            convergence=convergence,
            days_back=days_back,
            jurisdiction=jurisdiction,
        )
        if self._cache:
            self._cache.put(key, output, ttl=_CACHE_TTL)
        return ToolResult(success=True, output=output)

    def _format_cluster_output(
        self,
        *,
        query: str,
        records: list[dict[str, Any]],
        agency_groups: dict[str, list[dict[str, Any]]],
        jur_set: set[str],
        n_agencies: int,
        n_jurisdictions: int,
        convergence: bool,
        days_back: int,
        jurisdiction: str,
    ) -> str:
        """Format entity cluster results."""
        signal = "⚠️ INVESTIGATION CONVERGENCE" if convergence else "No convergence"

        lines = [
            f'FOIA/FOI Entity Cluster: "{query}"',
            f"Period: last {days_back} days  |  Jurisdiction: {jurisdiction.upper()}",
            "",
            f"Total requests: {len(records)}",
            f"Distinct agencies: {n_agencies}",
            f"Distinct jurisdictions: {n_jurisdictions} ({', '.join(sorted(jur_set)) or 'none'})",
            f"Signal: {signal}",
            "",
        ]

        if not records:
            lines.append("No matching requests found.")
            return "\n".join(lines)

        # Agency breakdown (sorted by count descending)
        lines.append("Agency breakdown:")
        sorted_agencies = sorted(
            agency_groups.items(), key=lambda x: len(x[1]), reverse=True
        )
        for ag, reqs in sorted_agencies:
            jurs = {r["jurisdiction"] for r in reqs}
            lines.append(f"  {ag} ({', '.join(sorted(jurs))}): {len(reqs)} request(s)")
        lines.append("")

        # Timeline — most recent first
        lines.append("Recent requests:")
        sorted_recs = sorted(
            records, key=lambda r: r.get("date_filed", ""), reverse=True
        )
        for i, rec in enumerate(sorted_recs[:30], 1):
            lines.append(_format_request(rec, index=i))
            lines.append("")

        return "\n".join(lines)
