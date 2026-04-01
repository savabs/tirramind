"""
Tool: UCC / Creditor Filings — Creditor Stress Detector

SEC EDGAR EFTS:       https://efts.sec.gov/LATEST/search-index?...  (free, no auth)
UK Companies House:   https://api.company-information.service.gov.uk (free key)

Material credit events (new liens, security interest grants, collateral pledges)
appear in SEC 8-K filings under Items 1.01 (Material Agreement) and 2.03
(Creation of a Direct Financial Obligation).  A surge of these filings for
one entity or across a sector signals financial distress — often weeks before
credit-rating downgrades or defaults.

UK Companies House publishes every registered charge (mortgage, debenture,
floating charge) for every UK company.  A cluster of new unsatisfied charges
or unusual charge patterns is an early distress signal.

Nobody auto-monitors creditor filings as a *predictive signal*.  The filings
are public.  The cluster / surge analysis is the edge.

Modes
-----
search          Search EDGAR 8-K filings for credit-event language by entity.
                Also searches UK Companies House charges if key is set.

uk_charges      List charges for a UK company (by company number or name).
                Requires TIRRA_COMPANIES_HOUSE_KEY env var.

stress_scan     Broad scan for recent credit stress across EDGAR 8-K filings.
                Detects filing clusters (multiple filings for same entity or
                sector in short windows).
"""

from __future__ import annotations

import logging
import os
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

_SEC_EFTS = "https://efts.sec.gov/LATEST/search-index"
_CH_BASE = "https://api.company-information.service.gov.uk"
_UA = "TirraMind/0.1 (creditor-filings-tool)"
_TIMEOUT = 20
_CACHE_TTL = 1800  # 30 min
_MAX_LIMIT = 100
_DEFAULT_LIMIT = 20

VALID_MODES = frozenset({"search", "uk_charges", "stress_scan"})

# Credit-event search terms for EDGAR EFTS
_CREDIT_TERMS = [
    '"security interest"',
    '"pledge agreement"',
    '"credit facility"',
    '"collateral"',
    '"lien"',
    '"loan agreement"',
]

# 8-K items related to credit events
_CREDIT_ITEMS = {"1.01", "2.03", "2.04"}

# UK Companies House charge status classification
_CHARGE_RED_FLAGS = frozenset(
    {
        "outstanding",
        "part-satisfied",
    }
)


# ---------------------------------------------------------------------------
# Helpers — SEC EDGAR EFTS
# ---------------------------------------------------------------------------


def _fetch_json(url: str, client: httpx.Client, **params: Any) -> dict | None:
    """Fetch URL and parse as JSON.  Returns dict or None."""
    try:
        r = client.get(url, params=params)
        if r.status_code != 200:
            log.warning("HTTP %d from %s", r.status_code, url)
            return None
        return r.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("Error fetching %s: %s", url, exc)
        return None


def _parse_efts_hits(data: dict) -> list[dict[str, Any]]:
    """Parse EFTS search-index JSON into structured entries."""
    entries: list[dict[str, Any]] = []
    hits = data.get("hits", {}).get("hits", [])
    for hit in hits:
        src = hit.get("_source", {})
        names = src.get("display_names", [])
        company = names[0] if names else "Unknown"
        ciks = src.get("ciks", [])
        cik = ciks[0] if ciks else ""
        entries.append(
            {
                "company_name": company,
                "cik": cik,
                "file_date": src.get("file_date", ""),
                "form": src.get("form", "8-K"),
                "items": src.get("items", []),
            }
        )
    return entries


def _search_efts(
    client: httpx.Client,
    *,
    query: str,
    days_back: int,
    limit: int,
) -> tuple[list[dict[str, Any]], int]:
    """Search EDGAR EFTS for 8-K filings matching query.

    Returns (entries, total_count).
    """
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end = now.strftime("%Y-%m-%d")

    data = _fetch_json(
        _SEC_EFTS,
        client,
        q=query,
        forms="8-K",
        dateRange="custom",
        startdt=start,
        enddt=end,
        **{"from": "0"},
        size=str(min(limit, 100)),
        _source="form,file_date,display_names,items,ciks",
    )
    if data is None:
        return [], 0

    entries = _parse_efts_hits(data)
    total = data.get("hits", {}).get("total", {}).get("value", 0)
    return entries[:limit], total


# ---------------------------------------------------------------------------
# Helpers — UK Companies House
# ---------------------------------------------------------------------------


def _get_ch_key() -> str | None:
    """Get Companies House API key from environment."""
    key = os.environ.get("TIRRA_COMPANIES_HOUSE_KEY", "").strip()
    return key if key else None


def _ch_client(api_key: str) -> httpx.Client:
    """Create an authenticated Companies House HTTP client."""
    return httpx.Client(
        timeout=_TIMEOUT,
        headers={"User-Agent": _UA},
        auth=(api_key, ""),  # HTTP Basic: key as username, empty password
        follow_redirects=True,
    )


def _search_ch_company(name: str, client: httpx.Client) -> list[dict[str, Any]]:
    """Search Companies House for companies by name."""
    data = _fetch_json(
        f"{_CH_BASE}/search/companies",
        client,
        q=name,
        items_per_page="5",
    )
    if data is None:
        return []
    results = []
    for item in data.get("items", []):
        results.append(
            {
                "company_name": item.get("title", ""),
                "company_number": item.get("company_number", ""),
                "company_status": item.get("company_status", ""),
                "date_of_creation": item.get("date_of_creation", ""),
            }
        )
    return results


def _get_ch_charges(company_number: str, client: httpx.Client) -> list[dict[str, Any]]:
    """Get all charges for a UK company from Companies House."""
    safe_num = quote(company_number.strip(), safe="")
    data = _fetch_json(
        f"{_CH_BASE}/company/{safe_num}/charges",
        client,
        items_per_page="100",
    )
    if data is None:
        return []

    charges: list[dict[str, Any]] = []
    for item in data.get("items", []):
        status = (item.get("status") or "unknown").lower().strip()
        charges.append(
            {
                "charge_number": item.get("charge_number", 0),
                "status": status,
                "created_on": item.get("created_on", ""),
                "delivered_on": item.get("delivered_on", ""),
                "satisfied_on": item.get("satisfied_on", ""),
                "classification": _classify_charge(item),
                "persons_entitled": [
                    p.get("name", "") for p in item.get("persons_entitled", [])
                ],
                "particulars": _extract_particulars(item),
            }
        )
    return charges


def _classify_charge(item: dict[str, Any]) -> str:
    """Classify charge type from Companies House data."""
    desc = (item.get("particulars", {}).get("description") or "").lower()
    if "debenture" in desc:
        return "debenture"
    if "floating" in desc:
        return "floating_charge"
    if "mortgage" in desc:
        return "mortgage"
    if "fixed" in desc:
        return "fixed_charge"
    return "other"


def _extract_particulars(item: dict[str, Any]) -> str:
    """Extract charge particulars description, truncated."""
    p = item.get("particulars", {})
    if isinstance(p, dict):
        desc = p.get("description", "")
    else:
        desc = str(p)
    return desc[:200] if desc else ""


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def _detect_filing_clusters(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect entity-level filing clusters in EDGAR results.

    Returns entities with 2+ filings (potential stress signals).
    """
    entity_filings: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        name = e.get("company_name", "Unknown")
        entity_filings.setdefault(name, []).append(e)

    clusters = []
    for name, filings in entity_filings.items():
        if len(filings) >= 2:
            dates = sorted(f.get("file_date", "") for f in filings)
            clusters.append(
                {
                    "entity": name,
                    "filing_count": len(filings),
                    "date_range": f"{dates[0]} to {dates[-1]}" if dates else "",
                    "cik": filings[0].get("cik", ""),
                }
            )
    clusters.sort(key=lambda c: c["filing_count"], reverse=True)
    return clusters


def _count_red_flag_charges(charges: list[dict[str, Any]]) -> int:
    """Count charges with red-flag status (outstanding, part-satisfied)."""
    return sum(1 for c in charges if c.get("status", "") in _CHARGE_RED_FLAGS)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class CreditorFilingsTool(Tool):
    """Monitor creditor filings for financial distress signals."""

    name = "creditor_filings"
    description = (
        "Search SEC 8-K filings for credit events (security interests, "
        "pledges, liens, credit facility changes) and UK Companies House "
        "charges. Detects creditor filing surges that precede credit "
        "downgrades and defaults."
    )
    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": sorted(VALID_MODES),
                "description": (
                    "search: EDGAR 8-K credit-event search by entity. "
                    "uk_charges: UK Companies House charges for a company. "
                    "stress_scan: broad recent credit-stress scan."
                ),
            },
            "query": {
                "type": "string",
                "description": "Entity name to search (search mode).",
            },
            "company_number": {
                "type": "string",
                "description": "UK company number (uk_charges mode).",
            },
            "days_back": {
                "type": "integer",
                "description": "Look-back window in days (default 30).",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default 20, max 100).",
            },
        },
        "required": ["mode"],
    }

    def __init__(self, *, cache: DataCache | None = None) -> None:
        self._cache = cache

    def execute(self, **kwargs: Any) -> ToolResult:
        mode = (kwargs.get("mode") or "").strip().lower()
        if mode not in VALID_MODES:
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Use: {', '.join(sorted(VALID_MODES))}",
            )

        limit = min(max(int(kwargs.get("limit", _DEFAULT_LIMIT)), 1), _MAX_LIMIT)
        days_back = max(int(kwargs.get("days_back", 30)), 1)

        if mode == "search":
            return self._search(
                query=kwargs.get("query", ""),
                days_back=days_back,
                limit=limit,
            )
        if mode == "uk_charges":
            return self._uk_charges(
                company_number=kwargs.get("company_number", ""),
                query=kwargs.get("query", ""),
                limit=limit,
            )
        return self._stress_scan(days_back=days_back, limit=limit)

    # ── search mode ──────────────────────────────────────────────────────

    def _search(self, *, query: str, days_back: int, limit: int) -> ToolResult:
        if not query or not query.strip():
            return ToolResult(
                success=False,
                output="Parameter 'query' is required for search mode.",
            )
        query = query.strip()

        cache_key = f"cred_search_{query}_{days_back}_{limit}"
        if self._cache:
            cached = self._cache.get("creditor_filings", cache_key)
            if cached is not None:
                return self._format_search_result(
                    cached["entries"],
                    cached.get("total", 0),
                    query,
                    days_back,
                    cached.get("ch_charges"),
                    from_cache=True,
                )

        # SEC EDGAR search: combine entity name with credit terms
        efts_query = f'"{query}" AND ({" OR ".join(_CREDIT_TERMS)})'

        with httpx.Client(
            timeout=_TIMEOUT,
            headers={"User-Agent": _UA},
            follow_redirects=True,
        ) as client:
            entries, total = _search_efts(
                client, query=efts_query, days_back=days_back, limit=limit
            )

        # UK Companies House (optional — requires key)
        ch_charges: list[dict[str, Any]] | None = None
        ch_key = _get_ch_key()
        if ch_key:
            try:
                with _ch_client(ch_key) as client:
                    companies = _search_ch_company(query, client)
                    if companies:
                        # Get charges for first match
                        ch_charges = _get_ch_charges(
                            companies[0]["company_number"], client
                        )
            except Exception as exc:
                log.warning("Companies House error: %s", exc)

        if self._cache:
            self._cache.set(
                "creditor_filings",
                cache_key,
                {"entries": entries, "total": total, "ch_charges": ch_charges},
                ttl=_CACHE_TTL,
            )

        return self._format_search_result(entries, total, query, days_back, ch_charges)

    def _format_search_result(
        self,
        entries: list[dict[str, Any]],
        total: int,
        query: str,
        days_back: int,
        ch_charges: list[dict[str, Any]] | None,
        *,
        from_cache: bool = False,
    ) -> ToolResult:
        tag = " (cached)" if from_cache else ""
        lines = [
            f"Creditor Filing Search: '{query}'{tag}",
            f"SEC EDGAR 8-K credit events: {len(entries)} (total {total}, last {days_back}d)",
            "",
        ]

        for e in entries[:15]:
            items_str = ", ".join(e.get("items", []))
            lines.append(
                f"  [{e.get('file_date', '?')}] {e.get('company_name', '?')} "
                f"(CIK {e.get('cik', '?')}) — items: {items_str}"
            )
        if len(entries) > 15:
            lines.append(f"  ... and {len(entries) - 15} more")

        if ch_charges is not None:
            red_flags = _count_red_flag_charges(ch_charges)
            lines.append("")
            lines.append(
                f"UK Companies House charges: {len(ch_charges)} total, "
                f"{red_flags} red-flag (outstanding/part-satisfied)"
            )
            for c in ch_charges[:10]:
                entitled = ", ".join(c.get("persons_entitled", [])[:2]) or "N/A"
                lines.append(
                    f"  [{c.get('status', '?')}] #{c.get('charge_number', '?')} "
                    f"created {c.get('created_on', '?')} — {c.get('classification', '?')} "
                    f"— creditor: {entitled}"
                )
            if len(ch_charges) > 10:
                lines.append(f"  ... and {len(ch_charges) - 10} more")
        elif _get_ch_key() is None:
            lines.append("")
            lines.append(
                "UK data: set TIRRA_COMPANIES_HOUSE_KEY for UK charge monitoring"
            )

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "mode": "search",
                "query": query,
                "sec_count": len(entries),
                "sec_total": total,
                "sec_entries": entries,
                "ch_charges": ch_charges,
                "ch_red_flags": (
                    _count_red_flag_charges(ch_charges) if ch_charges else 0
                ),
            },
        )

    # ── uk_charges mode ──────────────────────────────────────────────────

    def _uk_charges(self, *, company_number: str, query: str, limit: int) -> ToolResult:
        ch_key = _get_ch_key()
        if not ch_key:
            return ToolResult(
                success=False,
                output=(
                    "UK Companies House API key not configured. "
                    "Set TIRRA_COMPANIES_HOUSE_KEY environment variable "
                    "(free key from https://developer.company-information.service.gov.uk)."
                ),
            )

        company_number = (company_number or "").strip()
        query = (query or "").strip()

        if not company_number and not query:
            return ToolResult(
                success=False,
                output="Provide 'company_number' or 'query' (company name) for uk_charges mode.",
            )

        cache_key = f"ch_charges_{company_number or query}_{limit}"
        if self._cache:
            cached = self._cache.get("creditor_filings", cache_key)
            if cached is not None:
                return self._format_uk_result(
                    cached["charges"],
                    cached["company_info"],
                    limit,
                    from_cache=True,
                )

        try:
            with _ch_client(ch_key) as client:
                company_info: dict[str, Any] = {}

                if not company_number:
                    # Search by name first
                    companies = _search_ch_company(query, client)
                    if not companies:
                        return ToolResult(
                            success=False,
                            output=f"No UK company found for '{query}'.",
                        )
                    company_info = companies[0]
                    company_number = company_info["company_number"]
                else:
                    company_info = {"company_number": company_number}

                charges = _get_ch_charges(company_number, client)
        except Exception as exc:
            log.warning("Companies House error: %s", exc)
            return ToolResult(
                success=False,
                output=f"Companies House API error: {exc}",
            )

        if self._cache:
            self._cache.set(
                "creditor_filings",
                cache_key,
                {"charges": charges, "company_info": company_info},
                ttl=_CACHE_TTL,
            )

        return self._format_uk_result(charges, company_info, limit)

    def _format_uk_result(
        self,
        charges: list[dict[str, Any]],
        company_info: dict[str, Any],
        limit: int,
        *,
        from_cache: bool = False,
    ) -> ToolResult:
        tag = " (cached)" if from_cache else ""
        name = company_info.get("company_name", company_info.get("company_number", "?"))
        red_flags = _count_red_flag_charges(charges)

        lines = [
            f"UK Companies House Charges: {name}{tag}",
            f"Total charges: {len(charges)}, Red-flag: {red_flags} "
            f"(outstanding/part-satisfied)",
            "",
        ]

        displayed = charges[:limit]
        for c in displayed:
            entitled = ", ".join(c.get("persons_entitled", [])[:2]) or "N/A"
            sat_info = (
                f", satisfied {c.get('satisfied_on', '')}"
                if c.get("satisfied_on")
                else ""
            )
            lines.append(
                f"  [{c.get('status', '?')}] #{c.get('charge_number', '?')} "
                f"created {c.get('created_on', '?')}{sat_info} — "
                f"{c.get('classification', '?')} — creditor: {entitled}"
            )
            if c.get("particulars"):
                lines.append(f"    particulars: {c['particulars'][:120]}")
        if len(charges) > limit:
            lines.append(f"  ... and {len(charges) - limit} more")

        # Stress assessment
        if red_flags >= 3:
            lines.append("")
            lines.append(
                f"⚠ HIGH STRESS: {red_flags} unsatisfied/part-satisfied charges"
            )
        elif red_flags >= 1:
            lines.append("")
            lines.append(f"⚡ MODERATE: {red_flags} outstanding charge(s)")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "mode": "uk_charges",
                "company_info": company_info,
                "charge_count": len(charges),
                "red_flag_count": red_flags,
                "charges": displayed,
            },
        )

    # ── stress_scan mode ─────────────────────────────────────────────────

    def _stress_scan(self, *, days_back: int, limit: int) -> ToolResult:
        cache_key = f"cred_stress_{days_back}_{limit}"
        if self._cache:
            cached = self._cache.get("creditor_filings", cache_key)
            if cached is not None:
                return self._format_stress_result(
                    cached["entries"],
                    cached["total"],
                    cached["clusters"],
                    days_back,
                    from_cache=True,
                )

        # Search EDGAR for credit-event language across ALL 8-K filings
        combined_query = " OR ".join(_CREDIT_TERMS)

        with httpx.Client(
            timeout=_TIMEOUT,
            headers={"User-Agent": _UA},
            follow_redirects=True,
        ) as client:
            entries, total = _search_efts(
                client, query=combined_query, days_back=days_back, limit=limit
            )

        clusters = _detect_filing_clusters(entries)

        if self._cache:
            self._cache.set(
                "creditor_filings",
                cache_key,
                {"entries": entries, "total": total, "clusters": clusters},
                ttl=_CACHE_TTL,
            )

        return self._format_stress_result(entries, total, clusters, days_back)

    def _format_stress_result(
        self,
        entries: list[dict[str, Any]],
        total: int,
        clusters: list[dict[str, Any]],
        days_back: int,
        *,
        from_cache: bool = False,
    ) -> ToolResult:
        tag = " (cached)" if from_cache else ""
        lines = [
            f"Credit Stress Scan{tag} (last {days_back}d)",
            f"Total 8-K credit-event filings: {total} ({len(entries)} returned)",
            "",
        ]

        if clusters:
            lines.append(f"Entity clusters (2+ filings): {len(clusters)}")
            for cl in clusters[:10]:
                lines.append(
                    f"  ⚠ {cl['entity']} — {cl['filing_count']} filings "
                    f"({cl['date_range']}) CIK {cl['cik']}"
                )
            if len(clusters) > 10:
                lines.append(f"  ... and {len(clusters) - 10} more clusters")
            lines.append("")

        lines.append("Recent filings:")
        for e in entries[:15]:
            items_str = ", ".join(e.get("items", []))
            lines.append(
                f"  [{e.get('file_date', '?')}] {e.get('company_name', '?')} "
                f"(CIK {e.get('cik', '?')}) — items: {items_str}"
            )
        if len(entries) > 15:
            lines.append(f"  ... and {len(entries) - 15} more")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "mode": "stress_scan",
                "days_back": days_back,
                "sec_count": len(entries),
                "sec_total": total,
                "clusters": clusters,
                "cluster_count": len(clusters),
                "entries": entries,
            },
        )
