"""
Tool: Lobbying Expenditure — Political Spending Intelligence

LDA.gov  https://lda.gov/api/v1/  (free, 120 req/min with key)

Companies spend money lobbying BEFORE policy changes happen.  Track which
companies/industries suddenly ramp up lobbying spend → regulatory change
is coming.  New-issue lobbying (first-time topic) is especially predictive
of upcoming legislation.

The Senate Lobbying Disclosure Act (LDA) database contains all registered
lobbying activity in the US — quarterly LD-2 reports with income/expenses,
specific issues lobbied, and government entities contacted.

Modes
-----
search          Search lobbying filings by registrant, client, or issue.

spending        Aggregate spending by industry/registrant over time.
                Detect abnormal spend spikes.

issues          Track lobbying activity on specific policy issues.
                New-issue lobbying = legislation coming.

Signal theory:
  - Abnormal spend increase by industry → regulation imminent
  - New-issue lobbying (first-time topic appearance) → bill drafting
  - Coordinated multi-firm lobbying on same issue → coalition formation
  - Lobbying spend correlates with regulatory outcome 12-18 months later
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
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

_LDA_BASE = "https://lda.gov/api/v1"
_UA = "TirraMind/0.1 (lobbying-tool)"
_TIMEOUT = 20
_CACHE_TTL = 3600  # 1 hr — filings posted within 2 business days
_PAGE_SIZE = 25  # LDA max per page

VALID_MODES = frozenset({"search", "spending", "issues"})

# Common lobbying issue area codes (LDA general issue areas)
ISSUE_AREAS: dict[str, str] = {
    "BNK": "Banking",
    "DEF": "Defense",
    "ENE": "Energy/Nuclear",
    "ENV": "Environment/Superfund",
    "FIN": "Financial Institutions/Investments/Securities",
    "HCR": "Health Issues",
    "IMM": "Immigration",
    "TAX": "Taxation/Internal Revenue Code",
    "TEC": "Science/Technology",
    "TRD": "Trade (Domestic & Foreign)",
    "TRA": "Transportation",
    "COM": "Communications/Broadcasting/Telephone",
    "CPT": "Computer Industry",
    "AGR": "Agriculture",
    "EDU": "Education",
    "MIA": "Media (Information/Publishing)",
    "LBR": "Labor Issues/Antitrust/Workplace",
    "GOV": "Government Issues",
    "AER": "Aerospace",
    "PHM": "Pharmacy",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_lda(
    endpoint: str,
    params: dict[str, str],
    api_key: str | None = None,
) -> dict[str, Any] | None:
    """Fetch from LDA.gov API. Returns JSON or None."""
    url = f"{_LDA_BASE}/{endpoint}/"
    headers: dict[str, str] = {"User-Agent": _UA}
    if api_key:
        headers["Authorization"] = f"Token {api_key}"
    try:
        with httpx.Client(timeout=_TIMEOUT, headers=headers) as client:
            resp = client.get(url, params=params)
            if resp.status_code != 200:
                log.warning("LDA API returned %d for %s", resp.status_code, endpoint)
                return None
            return resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("LDA API error: %s", exc)
        return None


def _parse_filing(filing: dict[str, Any]) -> dict[str, Any]:
    """Parse a raw LDA filing into a clean record."""
    registrant = filing.get("registrant") or {}
    client = filing.get("client") or {}
    activities = filing.get("lobbying_activities") or []

    # Extract issue codes from activities
    issue_codes = []
    descriptions = []
    for act in activities:
        code = act.get("general_issue_code", "")
        if code:
            issue_codes.append(code)
        desc = act.get("description", "")
        if desc:
            descriptions.append(desc[:200])

    income = filing.get("income")
    expenses = filing.get("expenses")
    amount = 0.0
    if income and income != "0.00":
        try:
            amount = float(income)
        except (ValueError, TypeError):
            pass
    if not amount and expenses and expenses != "0.00":
        try:
            amount = float(expenses)
        except (ValueError, TypeError):
            pass

    return {
        "filing_uuid": filing.get("filing_uuid", ""),
        "filing_type": filing.get("filing_type_display", filing.get("filing_type", "")),
        "filing_year": filing.get("filing_year"),
        "filing_period": filing.get("filing_period", ""),
        "dt_posted": filing.get("dt_posted", ""),
        "registrant_name": registrant.get("name", "Unknown"),
        "registrant_id": registrant.get("id"),
        "client_name": client.get("name", "Unknown"),
        "client_id": client.get("id"),
        "amount": amount,
        "issue_codes": issue_codes,
        "issue_descriptions": descriptions[:5],
    }


def _current_year() -> int:
    return datetime.now(timezone.utc).year


def _detect_spend_anomaly(
    amounts: list[float],
    threshold_multiplier: float = 2.0,
) -> dict[str, Any]:
    """Detect if recent spending is anomalously high vs historical average."""
    if len(amounts) < 2:
        return {"anomaly": False, "ratio": None}
    avg = sum(amounts[:-1]) / len(amounts[:-1])
    if avg <= 0:
        return {"anomaly": False, "ratio": None}
    ratio = amounts[-1] / avg
    return {
        "anomaly": ratio >= threshold_multiplier,
        "ratio": round(ratio, 2),
        "latest": amounts[-1],
        "historical_avg": round(avg, 2),
    }


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class LobbyingTool(Tool):
    """Track US lobbying expenditures via Senate LDA database."""

    name = "lobbying"
    description = (
        "Search and analyze US lobbying filings — track spending by company, "
        "industry, or issue.  Detects abnormal spend spikes, new-issue "
        "lobbying, and coordinated industry campaigns that precede regulation."
    )
    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": sorted(VALID_MODES),
                "description": (
                    "issues: lobbying activity on specific policy issues. "
                    "search: search filings by registrant, client, or year. "
                    "spending: aggregate spending trends by registrant."
                ),
            },
            "registrant": {
                "type": "string",
                "description": "Lobbying firm name.",
            },
            "client": {
                "type": "string",
                "description": "Client company being represented.",
            },
            "issue_code": {
                "type": "string",
                "description": "Issue area code (e.g., 'HCR' for health, 'TAX' for taxation).",
            },
            "year": {
                "type": "integer",
                "description": "Filing year (2008-present, default: current year).",
            },
            "quarter": {
                "type": "string",
                "enum": ["Q1", "Q2", "Q3", "Q4"],
                "description": "Filing quarter.",
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
        import os

        key = os.environ.get("TIRRA_LDA_API_KEY", "").strip()
        return key if key else None

    # ------------------------------------------------------------------
    # Entity persistence (L2)
    # ------------------------------------------------------------------

    def _persist_entities(self, filings: list[dict[str, Any]]) -> None:
        """Register company entities and store L2 lobbying observations."""
        if self._store is None or entity_id_from_key is None:
            return
        if not filings:
            return
        try:
            self._persist_entities_inner(filings)
        except Exception:
            log.exception("Entity persistence failed (non-fatal)")

    def _persist_entities_inner(self, filings: list[dict[str, Any]]) -> None:
        assert self._store is not None  # noqa: S101
        store = self._store

        seen_companies: set[str] = set()
        for filing in filings:
            registrant = filing.get("registrant_name", "")
            if not registrant:
                continue

            try:
                canon = (
                    normalize_company_name(registrant)
                    if normalize_company_name
                    else registrant
                )
            except ValueError:
                canon = registrant

            company_eid = entity_id_from_key("company", canon)

            if registrant not in seen_companies:
                seen_companies.add(registrant)
                store.register_entity(
                    entity_type="company",
                    canonical_name=canon,
                    entity_id=company_eid,
                )
                if filing.get("registrant_id"):
                    store.add_entity_alias(
                        company_eid, "lda_registrant_id", str(filing["registrant_id"])
                    )

            # Parse posted date
            dt_posted = filing.get("dt_posted", "")
            try:
                ts = datetime.fromisoformat(
                    dt_posted.replace("Z", "+00:00")
                ).timestamp()
            except (ValueError, AttributeError):
                ts = datetime.now(tz=timezone.utc).timestamp()

            store.store_entity_observation(
                entity_id=company_eid,
                source_tool="lobbying",
                observed_at=ts,
                observation_type="lobbying_spend",
                depth_level=2,
                value={
                    "client_name": filing.get("client_name", ""),
                    "amount": filing.get("amount", 0),
                    "filing_year": filing.get("filing_year"),
                    "filing_period": filing.get("filing_period", ""),
                    "issue_codes": filing.get("issue_codes", []),
                },
            )

            # ── Link registrant → client company ──
            client_name = (filing.get("client_name") or "").strip()
            if (
                client_name
                and client_name != registrant
                and client_name.lower() != "self"
            ):
                try:
                    client_canon = (
                        normalize_company_name(client_name)
                        if normalize_company_name
                        else client_name
                    )
                except (ValueError, TypeError):
                    client_canon = client_name
                client_eid = entity_id_from_key("company", client_canon)
                store.register_entity(
                    entity_type="company",
                    canonical_name=client_canon,
                    entity_id=client_eid,
                )
                store.link_entities(
                    entity_id_a=company_eid,
                    entity_id_b=client_eid,
                    link_type="lobbies_for",
                    source="lobbying",
                    confidence=0.9,
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
        if mode == "spending":
            return self._spending(**kwargs)
        return self._issues(**kwargs)

    # ── search mode ──────────────────────────────────────────────

    def _search(self, **kwargs: Any) -> ToolResult:
        registrant = (kwargs.get("registrant") or "").strip()
        client = (kwargs.get("client") or "").strip()
        year = kwargs.get("year")
        quarter = (kwargs.get("quarter") or "").strip().upper()

        if not registrant and not client and not year:
            return ToolResult(
                success=False,
                output="At least one of 'registrant', 'client', or 'year' required.",
            )

        params: dict[str, str] = {}
        if registrant:
            params["registrant_name"] = registrant
        if client:
            params["client_name"] = client
        if year:
            yr = int(year)
            if yr < 2008 or yr > _current_year():
                return ToolResult(
                    success=False,
                    output=f"Year must be 2008-{_current_year()}.",
                )
            params["filing_year"] = str(yr)
        if quarter:
            quarter_map = {
                "Q1": "first_quarter",
                "Q2": "second_quarter",
                "Q3": "third_quarter",
                "Q4": "fourth_quarter",
            }
            if quarter in quarter_map:
                params["filing_period"] = quarter_map[quarter]

        cache_key = {"mode": "search", **params}
        if self._cache:
            cached = self._cache.get("lobbying", cache_key)
            if cached is not None:
                return self._format_search(cached, from_cache=True)

        data = _fetch_lda("filings", params, self._api_key)
        if data is None:
            return ToolResult(success=False, output="LDA API unavailable.")

        filings = [_parse_filing(f) for f in data.get("results", [])]
        total = data.get("count", len(filings))

        result_data = {
            "filings": filings,
            "total_count": total,
            "returned": len(filings),
        }

        if self._cache:
            self._cache.put("lobbying", cache_key, result_data)

        return self._format_search(result_data)

    def _format_search(
        self, data: dict[str, Any], *, from_cache: bool = False
    ) -> ToolResult:
        tag = " (cached)" if from_cache else ""
        filings = data.get("filings", [])
        total = data.get("total_count", 0)

        lines = [
            f"# Lobbying Filing Search{tag}",
            f"Total: {total:,} | Returned: {len(filings)}\n",
        ]

        for f in filings[:_PAGE_SIZE]:
            amount_str = f"${f['amount']:,.0f}" if f["amount"] else "N/A"
            period = f.get("filing_period", "").replace("_", " ").title()
            lines.append(
                f"**{f['registrant_name']}** → {f['client_name']} "
                f"({f.get('filing_year', '?')} {period})"
            )
            lines.append(f"  Amount: {amount_str} | Type: {f['filing_type']}")
            if f["issue_codes"]:
                issue_labels = [
                    f"{c} ({ISSUE_AREAS.get(c, '?')})" for c in f["issue_codes"][:5]
                ]
                lines.append(f"  Issues: {', '.join(issue_labels)}")

        self._persist_entities(filings)

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"mode": "search", **data},
        )

    # ── spending mode ────────────────────────────────────────────

    def _spending(self, **kwargs: Any) -> ToolResult:
        registrant = (kwargs.get("registrant") or "").strip()
        client = (kwargs.get("client") or "").strip()

        if not registrant and not client:
            return ToolResult(
                success=False,
                output="'registrant' or 'client' required for spending mode.",
            )

        # Fetch filings across recent years
        current_yr = _current_year()
        yearly_totals: dict[int, float] = {}
        all_filings: list[dict[str, Any]] = []
        errors: list[str] = []

        for yr in range(current_yr - 4, current_yr + 1):
            params: dict[str, str] = {"filing_year": str(yr)}
            if registrant:
                params["registrant_name"] = registrant
            if client:
                params["client_name"] = client

            cache_key = {"mode": "spending", **params}
            if self._cache:
                cached = self._cache.get("lobbying", cache_key)
                if cached is not None:
                    filings = cached.get("filings", [])
                    total_spend = sum(f["amount"] for f in filings)
                    yearly_totals[yr] = total_spend
                    all_filings.extend(filings)
                    continue

            data = _fetch_lda("filings", params, self._api_key)
            if data is None:
                errors.append(f"{yr}: API unavailable")
                continue

            filings = [_parse_filing(f) for f in data.get("results", [])]
            total_spend = sum(f["amount"] for f in filings)
            yearly_totals[yr] = total_spend
            all_filings.extend(filings)

            if self._cache:
                self._cache.put("lobbying", cache_key, {"filings": filings})

        if not yearly_totals:
            return ToolResult(
                success=False,
                output="No spending data found."
                + (f" Errors: {', '.join(errors)}" if errors else ""),
            )

        # Anomaly detection
        amounts = [yearly_totals.get(yr, 0) for yr in sorted(yearly_totals)]
        anomaly = _detect_spend_anomaly(amounts)

        target_name = registrant or client
        lines = [f"# Lobbying Spend: {target_name}\n"]

        for yr in sorted(yearly_totals):
            total = yearly_totals[yr]
            bar = "█" * min(int(total / 100000), 40)  # Scale bar
            lines.append(f"  {yr}: ${total:>12,.0f}  {bar}")

        if anomaly["anomaly"]:
            lines.append(
                f"\n⚠ **ANOMALOUS SPENDING**: {anomaly['ratio']:.1f}x historical average"
            )
        elif anomaly["ratio"] is not None:
            lines.append(f"\nSpend ratio vs avg: {anomaly['ratio']:.1f}x")

        if errors:
            lines.append("\n**Notes:**")
            for e in errors:
                lines.append(f"  - {e}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "mode": "spending",
                "target": target_name,
                "yearly_totals": yearly_totals,
                "anomaly": anomaly,
                "total_filings": len(all_filings),
                "errors": errors,
            },
        )

    # ── issues mode ──────────────────────────────────────────────

    def _issues(self, **kwargs: Any) -> ToolResult:
        issue_code = (kwargs.get("issue_code") or "").strip().upper()
        year = kwargs.get("year") or _current_year()

        if not issue_code:
            # Return list of issue area codes
            lines = ["# Lobbying Issue Area Codes\n"]
            for code, desc in sorted(ISSUE_AREAS.items()):
                lines.append(f"  **{code}**: {desc}")
            lines.append("\nUse 'issue_code' to see filings for a specific issue.")
            return ToolResult(
                success=True,
                output="\n".join(lines),
                data={"mode": "issues", "issue_areas": ISSUE_AREAS},
            )

        yr = int(year)
        if yr < 2008 or yr > _current_year():
            yr = _current_year()

        # LDA doesn't have a direct issue filter, so we search filings for the year
        # and filter by issue activity. We use a broader search.
        params: dict[str, str] = {"filing_year": str(yr)}

        cache_key = {"mode": "issues", "issue_code": issue_code, "year": yr}
        if self._cache:
            cached = self._cache.get("lobbying", cache_key)
            if cached is not None:
                return self._format_issues(cached, issue_code, yr, from_cache=True)

        data = _fetch_lda("filings", params, self._api_key)
        if data is None:
            return ToolResult(success=False, output="LDA API unavailable.")

        # Parse and filter for filings that include our issue code
        all_filings = [_parse_filing(f) for f in data.get("results", [])]
        matching = [f for f in all_filings if issue_code in f.get("issue_codes", [])]

        # Aggregate by registrant
        registrant_spend: dict[str, float] = {}
        for f in matching:
            name = f["registrant_name"]
            registrant_spend[name] = registrant_spend.get(name, 0) + f["amount"]

        result_data = {
            "issue_code": issue_code,
            "issue_name": ISSUE_AREAS.get(issue_code, "Unknown"),
            "year": yr,
            "total_filings": len(matching),
            "total_searched": len(all_filings),
            "filings": matching,
            "registrant_spend": registrant_spend,
        }

        if self._cache:
            self._cache.put("lobbying", cache_key, result_data)

        return self._format_issues(result_data, issue_code, yr)

    def _format_issues(
        self,
        data: dict[str, Any],
        issue_code: str,
        year: int,
        *,
        from_cache: bool = False,
    ) -> ToolResult:
        tag = " (cached)" if from_cache else ""
        issue_name = data.get("issue_name", "Unknown")
        filings = data.get("filings", [])
        registrant_spend = data.get("registrant_spend", {})

        lines = [
            f"# Lobbying on {issue_code} ({issue_name}) — {year}{tag}",
            f"Matching filings: {len(filings)}\n",
        ]

        # Top spenders on this issue
        if registrant_spend:
            lines.append("**Top Spenders:**")
            top = sorted(registrant_spend.items(), key=lambda x: x[1], reverse=True)[
                :15
            ]
            for name, amount in top:
                lines.append(f"  {name}: ${amount:,.0f}")

        # Recent filings
        if filings:
            lines.append("\n**Recent Filings:**")
            for f in filings[:10]:
                amount_str = f"${f['amount']:,.0f}" if f["amount"] else "N/A"
                lines.append(
                    f"  {f['registrant_name']} → {f['client_name']} — {amount_str}"
                )
                if f.get("issue_descriptions"):
                    lines.append(f"    {f['issue_descriptions'][0][:120]}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"mode": "issues", **data},
        )
