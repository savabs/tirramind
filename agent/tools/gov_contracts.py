"""
Tool: Government Contract Awards — Multi-Region Public Procurement

Fetches public procurement / contract award data from government transparency APIs.

Supported regions:
  us  — USASpending.gov (POST, no auth). US federal contract awards.
  uk  — UK Contracts Finder (GET, no auth, OCDS standard). All UK public sector procurement.

Endpoints used:
  US:  POST https://api.usaspending.gov/api/v2/search/spending_by_award/
  UK:  GET  https://www.contractsfinder.service.gov.uk/Published/Notices/OCDS/Search

Signal theory:
  - Defense spending surges (DoD, DHS, MoD) = geopolitical escalation, defense sector tailwind
  - Contractor concentration = monopoly rent → stock signal for winners
  - New agency spending patterns = policy direction (e.g., DOE clean energy ramp)
  - Award acceleration before fiscal year-end = budget flush (predictable timing)
  - Large-dollar awards to small firms = acquisition target (gov contractor M&A)
  - Sudden stop in awards = continuing resolution risk, government shutdown signal
  - Cross-border comparison: divergent procurement patterns between US/UK signal policy divergence

Modes:
  recent   — Most recent contract awards (sorted by date, filterable).
  top      — Largest awards by dollar/pound amount in a time period.
  agency   — Awards filtered by awarding agency/buyer (e.g., 'Department of Defense', 'Ministry of Defence').
  search   — Free-text search across recipient names and award descriptions.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

_AWARDS_URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
_AUTOCOMPLETE_URL = "https://api.usaspending.gov/api/v2/autocomplete/awarding_agency/"
_UK_OCDS_URL = (
    "https://www.contractsfinder.service.gov.uk/Published/Notices/OCDS/Search"
)
_UA = "TirraMind/0.1"
_TIMEOUT = 20

VALID_MODES = {"recent", "top", "agency", "search"}
VALID_REGIONS = {"us", "uk"}

# Contract type codes (A-D = contracts, other = grants/loans/etc.)
_CONTRACT_CODES = ["A", "B", "C", "D"]

# Standard fields to request
_FIELDS = [
    "Award ID",
    "Recipient Name",
    "Award Amount",
    "Total Outlays",
    "Awarding Agency",
    "Awarding Sub Agency",
    "Award Type",
    "Start Date",
    "End Date",
    "Description",
]


class GovContractsTool(Tool):
    """Query government contract awards from US (USASpending) and UK (Contracts Finder)."""

    def __init__(self, cache: DataCache | None = None) -> None:
        self._cache = cache

    @property
    def name(self) -> str:
        return "gov_contracts"

    @property
    def description(self) -> str:
        return (
            "Query government contract awards. "
            "US: USASpending.gov (federal contracts). "
            "UK: Contracts Finder (all UK public sector, OCDS standard). "
            "Search by date, agency/buyer, recipient, or dollar/pound amount."
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
                        "Query mode: recent (latest awards), top (largest by $), "
                        "agency (filter by agency), search (keyword search)."
                    ),
                },
                "region": {
                    "type": "string",
                    "enum": sorted(VALID_REGIONS),
                    "description": (
                        "Region: us (USASpending.gov, default), uk (UK Contracts Finder)."
                    ),
                },
                "agency": {
                    "type": "string",
                    "description": "Agency/buyer name filter. For agency mode.",
                },
                "query": {
                    "type": "string",
                    "description": "Search keyword for recipient or description. For search mode.",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date (YYYY-MM-DD). Default: 90 days ago.",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date (YYYY-MM-DD). Default: today.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 20, max 50).",
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

        region = (kwargs.get("region") or "us").strip().lower()
        if region not in VALID_REGIONS:
            return ToolResult(
                success=False,
                output=f"Invalid region '{region}'. Must be one of: {sorted(VALID_REGIONS)}",
            )

        limit = min(max(int(kwargs.get("limit", 20)), 1), 50)

        # Date range
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        end_date = (kwargs.get("end_date") or "").strip() or now.strftime("%Y-%m-%d")
        start_date = (kwargs.get("start_date") or "").strip() or (
            now - timedelta(days=90)
        ).strftime("%Y-%m-%d")

        try:
            if region == "uk":
                return self._execute_uk(mode, start_date, end_date, limit, kwargs)
            return self._execute_us(mode, start_date, end_date, limit, kwargs)
        except httpx.TimeoutException:
            source = "Contracts Finder" if region == "uk" else "USASpending"
            return ToolResult(success=False, output=f"{source} API timed out.")
        except httpx.HTTPError as exc:
            source = "Contracts Finder" if region == "uk" else "USASpending"
            return ToolResult(success=False, output=f"{source} API error: {exc}")
        except Exception as exc:
            log.exception("GovContractsTool error")
            return ToolResult(success=False, output=f"Unexpected error: {exc}")

    def _execute_us(
        self,
        mode: str,
        start_date: str,
        end_date: str,
        limit: int,
        kwargs: dict[str, Any],
    ) -> ToolResult:
        """US dispatch — USASpending.gov."""
        if mode == "recent":
            return self._query_awards(
                start_date,
                end_date,
                limit,
                sort_field="Start Date",
                sort_order="desc",
            )
        elif mode == "top":
            return self._query_awards(
                start_date,
                end_date,
                limit,
                sort_field="Award Amount",
                sort_order="desc",
            )
        elif mode == "agency":
            agency = (kwargs.get("agency") or "").strip()
            if not agency:
                return ToolResult(
                    success=False,
                    output="Agency mode requires an 'agency' parameter.",
                )
            return self._query_awards(
                start_date,
                end_date,
                limit,
                sort_field="Award Amount",
                sort_order="desc",
                agency_name=agency,
            )
        elif mode == "search":
            query = (kwargs.get("query") or "").strip()
            if not query:
                return ToolResult(
                    success=False,
                    output="Search mode requires a 'query' parameter.",
                )
            return self._query_awards(
                start_date,
                end_date,
                limit,
                sort_field="Award Amount",
                sort_order="desc",
                keyword=query,
            )
        return ToolResult(success=False, output=f"Unhandled US mode: {mode}")

    def _query_awards(
        self,
        start_date: str,
        end_date: str,
        limit: int,
        sort_field: str,
        sort_order: str,
        agency_name: str | None = None,
        keyword: str | None = None,
    ) -> ToolResult:
        filters: dict[str, Any] = {
            "time_period": [{"start_date": start_date, "end_date": end_date}],
            "award_type_codes": _CONTRACT_CODES,
        }

        if agency_name:
            filters["agencies"] = [
                {
                    "type": "awarding",
                    "tier": "toptier",
                    "name": agency_name,
                }
            ]

        if keyword:
            filters["keywords"] = [keyword]

        payload = {
            "filters": filters,
            "fields": _FIELDS,
            "limit": limit,
            "page": 1,
            "sort": sort_field,
            "order": sort_order,
        }

        data = self._post_json(_AWARDS_URL, payload)
        if data is None:
            return ToolResult(success=False, output="Failed to fetch award data.")

        raw_results = data.get("results", [])
        results = []
        for r in raw_results:
            results.append(
                {
                    "award_id": r.get("Award ID"),
                    "recipient": r.get("Recipient Name"),
                    "amount_usd": r.get("Award Amount"),
                    "agency": r.get("Awarding Agency"),
                    "sub_agency": r.get("Awarding Sub Agency"),
                    "award_type": r.get("Award Type"),
                    "start_date": r.get("Start Date"),
                    "end_date": r.get("End Date"),
                    "description": (r.get("Description") or "")[:200],
                }
            )

        page_meta = data.get("page_metadata", {})
        total = page_meta.get("total", len(results))

        summary = (
            f"Found {total} federal contract awards ({start_date} to {end_date})"
            + (f" from {agency_name}" if agency_name else "")
            + (f' matching "{keyword}"' if keyword else "")
            + f". Showing top {len(results)}."
        )
        return ToolResult(
            success=True,
            output=summary,
            data={"awards": results, "total": total, "count": len(results)},
        )

    def _post_json(self, url: str, payload: dict) -> Any:
        """POST JSON to USASpending API."""
        with httpx.Client(
            timeout=_TIMEOUT,
            headers={"User-Agent": _UA, "Content-Type": "application/json"},
        ) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()

    # ── UK Contracts Finder (OCDS) ──────────────────────────────

    def _execute_uk(
        self,
        mode: str,
        start_date: str,
        end_date: str,
        limit: int,
        kwargs: dict[str, Any],
    ) -> ToolResult:
        """UK dispatch — Contracts Finder OCDS."""
        query = (kwargs.get("query") or "").strip() if mode == "search" else None
        buyer = (kwargs.get("agency") or "").strip() if mode == "agency" else None

        if mode == "search" and not query:
            return ToolResult(
                success=False,
                output="Search mode requires a 'query' parameter.",
            )
        if mode == "agency" and not buyer:
            return ToolResult(
                success=False,
                output="Agency mode requires an 'agency' parameter.",
            )

        releases = self._fetch_uk_contracts(start_date, end_date)
        if releases is None:
            return ToolResult(
                success=False, output="Failed to fetch UK Contracts Finder data."
            )

        # Parse OCDS releases into normalized award records
        awards = self._parse_uk_releases(releases, query=query, buyer=buyer)

        # Sort based on mode
        if mode == "top":
            awards.sort(key=lambda a: a.get("amount") or 0, reverse=True)
        else:  # recent (default), agency, search — sort by date descending
            awards.sort(key=lambda a: a.get("start_date") or "", reverse=True)

        total = len(awards)
        awards = awards[:limit]

        summary = (
            f"Found {total} UK contract awards ({start_date} to {end_date})"
            + (f" from buyer '{buyer}'" if buyer else "")
            + (f' matching "{query}"' if query else "")
            + f". Showing top {len(awards)}."
        )
        return ToolResult(
            success=True,
            output=summary,
            data={
                "awards": awards,
                "total": total,
                "count": len(awards),
                "region": "uk",
            },
        )

    def _fetch_uk_contracts(self, start_date: str, end_date: str) -> list[dict] | None:
        """Fetch OCDS releases from UK Contracts Finder."""
        params = {
            "publishedFrom": f"{start_date}T00:00:00Z",
            "publishedTo": f"{end_date}T23:59:59Z",
        }
        with httpx.Client(
            timeout=_TIMEOUT,
            headers={"User-Agent": _UA},
        ) as client:
            resp = client.get(_UK_OCDS_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        return data.get("releases", [])

    def _parse_uk_releases(
        self,
        releases: list[dict],
        query: str | None = None,
        buyer: str | None = None,
    ) -> list[dict]:
        """Parse OCDS releases into normalized award dicts. Filter by query/buyer."""
        awards = []
        for release in releases:
            tender = release.get("tender", {})
            buyer_obj = release.get("buyer", {})
            buyer_name = buyer_obj.get("name", "")

            # Apply buyer filter
            if buyer and buyer.lower() not in buyer_name.lower():
                continue

            title = tender.get("title", "")
            description = tender.get("description", "")

            # Apply search filter
            if query:
                combined = f"{title} {description} {buyer_name}".lower()
                if query.lower() not in combined:
                    continue

            # Extract award amount from awards array
            raw_awards = release.get("awards", [])
            amount = None
            currency = "GBP"
            supplier = None
            for aw in raw_awards:
                val = aw.get("value", {})
                if val.get("amount") is not None:
                    amount = val["amount"]
                    currency = val.get("currency", "GBP")
                suppliers = aw.get("suppliers", [])
                if suppliers:
                    supplier = suppliers[0].get("name")

            # Extract period
            period = tender.get("contractPeriod") or tender.get("tenderPeriod", {})
            start = period.get("startDate", "")[:10] if period else ""
            end = period.get("endDate", "")[:10] if period else ""

            awards.append(
                {
                    "award_id": release.get("ocid", release.get("id", "")),
                    "recipient": supplier or "",
                    "amount": amount,
                    "currency": currency,
                    "agency": buyer_name,
                    "award_type": tender.get("procurementMethod", ""),
                    "start_date": start,
                    "end_date": end,
                    "description": (title or description or "")[:200],
                    "region": "uk",
                }
            )

        return awards
