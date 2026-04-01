"""
Tool: Political Risk Monitor — FEC Campaign Finance API

US Federal Election Commission data: candidates, campaign filings,
and independent expenditures (Super PAC spending for/against candidates).

Source: https://api.open.fec.gov/v1/ (free, DEMO_KEY or registered key)

Modes:
  candidates    — Search US federal candidates: name, party, office,
                  fundraising status, election cycles.
  filings       — Campaign finance filings: form type, cash on hand,
                  receipts, disbursements, coverage period.
  expenditures  — Independent expenditures (Schedule E): Super PAC
                  spending for/against candidates, amounts, payees.

Signal theory:
  - Independent expenditure surge = election uncertainty rising
  - Cash-on-hand dropoff mid-cycle = candidate weakness signal
  - Oppose expenditures outpacing support = negative sentiment accelerating
  - New candidate filings in off-cycle = political disruption signal
  - Concentration of spending on few races = capital-relevant policy battles

Market relevance:
  Election outcomes → regulatory regime (energy, health, tech, finance),
  tax policy, trade policy, defense spending.  Super PAC spending patterns
  reveal which policy fights capital considers most important.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

_UA = "TirraMind/0.1"
_TIMEOUT = 15
_CACHE_TTL = 7200  # 2 hours — FEC data updates daily

_FEC_BASE = "https://api.open.fec.gov/v1"

VALID_MODES = {"candidates", "filings", "expenditures"}
VALID_OFFICES = {"P", "S", "H"}  # President, Senate, House


def _get_api_key() -> str:
    """Get FEC API key from env or fall back to DEMO_KEY."""
    return os.environ.get("TIRRA_FEC_API_KEY", "DEMO_KEY")


class PoliticalRiskTool(Tool):
    """Monitor US political risk via FEC campaign finance data."""

    def __init__(self, cache: DataCache | None = None) -> None:
        self._cache = cache

    @property
    def name(self) -> str:
        return "political_risk"

    @property
    def description(self) -> str:
        return (
            "Monitor US political risk via FEC campaign finance data — "
            "candidate filings, campaign cash on hand, and independent "
            "expenditures (Super PAC spending for/against candidates). "
            "Detects election spending surges and political uncertainty."
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
                        "candidates: search federal candidates. "
                        "filings: campaign finance filings. "
                        "expenditures: independent expenditures (Super PAC)."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": (
                        "For candidates: name search. "
                        "For filings: committee_id (e.g. 'C00703975'). "
                        "For expenditures: candidate_id filter (optional)."
                    ),
                },
                "office": {
                    "type": "string",
                    "enum": ["P", "S", "H"],
                    "description": (
                        "Filter by office: P=President, S=Senate, "
                        "H=House (candidates mode)."
                    ),
                },
                "cycle": {
                    "type": "integer",
                    "description": "Election cycle year (e.g. 2024). Must be even.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default: 20, max: 100).",
                },
                "sort_order": {
                    "type": "string",
                    "enum": ["asc", "desc"],
                    "description": "Sort order (default: desc for recency).",
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

        limit = min(kwargs.get("limit") or 20, 100)
        cycle = kwargs.get("cycle")
        if cycle is not None and cycle % 2 != 0:
            return ToolResult(
                success=False,
                output=f"Election cycle must be an even year, got {cycle}.",
            )

        if mode == "candidates":
            return self._handle_candidates(kwargs, limit)
        elif mode == "filings":
            return self._handle_filings(kwargs, limit)
        else:
            return self._handle_expenditures(kwargs, limit)

    # ── Mode handlers ───────────────────────────────────────

    def _handle_candidates(self, kwargs: dict, limit: int) -> ToolResult:
        query = (kwargs.get("query") or "").strip()
        office = (kwargs.get("office") or "").strip().upper()
        cycle = kwargs.get("cycle")

        if office and office not in VALID_OFFICES:
            return ToolResult(
                success=False,
                output=f"Invalid office '{office}'. Must be one of: P, S, H.",
            )

        params: dict[str, str] = {
            "api_key": _get_api_key(),
            "per_page": str(limit),
            "sort": "-election_year",
        }
        if query:
            params["q"] = query
        if office:
            params["office"] = office
        if cycle:
            params["election_year"] = str(cycle)

        cache_key = f"fec:candidates:{query}:{office}:{cycle}:{limit}"
        return self._fetch_fec(
            f"{_FEC_BASE}/candidates/search/", params, cache_key, "candidates",
        )

    def _handle_filings(self, kwargs: dict, limit: int) -> ToolResult:
        query = (kwargs.get("query") or "").strip()
        cycle = kwargs.get("cycle")
        sort_order = (kwargs.get("sort_order") or "desc").lower()

        params: dict[str, str] = {
            "api_key": _get_api_key(),
            "per_page": str(limit),
            "sort": (
                "-receipt_date" if sort_order == "desc" else "receipt_date"
            ),
        }
        if query:
            params["committee_id"] = query
        if cycle:
            params["cycle"] = str(cycle)

        cache_key = f"fec:filings:{query}:{cycle}:{limit}"
        return self._fetch_fec(
            f"{_FEC_BASE}/filings/", params, cache_key, "filings",
        )

    def _handle_expenditures(self, kwargs: dict, limit: int) -> ToolResult:
        query = (kwargs.get("query") or "").strip()
        cycle = kwargs.get("cycle")
        sort_order = (kwargs.get("sort_order") or "desc").lower()

        params: dict[str, str] = {
            "api_key": _get_api_key(),
            "per_page": str(limit),
            "sort": (
                "-expenditure_date" if sort_order == "desc"
                else "expenditure_date"
            ),
        }
        if query:
            params["candidate_id"] = query
        if cycle:
            params["cycle"] = str(cycle)

        cache_key = f"fec:expenditures:{query}:{cycle}:{limit}"
        return self._fetch_fec(
            f"{_FEC_BASE}/schedules/schedule_e/",
            params, cache_key, "expenditures",
        )

    # ── Core fetch ──────────────────────────────────────────

    def _fetch_fec(
        self, url: str, params: dict, cache_key: str, result_type: str,
    ) -> ToolResult:
        if self._cache:
            hit = self._cache.get(cache_key)
            if hit is not None:
                return ToolResult(
                    success=True, output=hit["output"], data=hit["data"],
                )

        try:
            with httpx.Client(
                timeout=_TIMEOUT, headers={"User-Agent": _UA},
            ) as client:
                resp = client.get(url, params=params)
        except httpx.TimeoutException:
            return ToolResult(
                success=False, output="FEC API request timed out.",
            )
        except httpx.HTTPError as exc:
            return ToolResult(success=False, output=f"HTTP error: {exc}")

        if resp.status_code == 429:
            return ToolResult(
                success=False,
                output=(
                    "FEC API rate limit reached. "
                    "Set TIRRA_FEC_API_KEY for higher limits."
                ),
            )
        if resp.status_code == 422:
            return ToolResult(
                success=False,
                output="FEC API validation error — check parameters.",
            )
        if resp.status_code != 200:
            return ToolResult(
                success=False,
                output=f"FEC API returned HTTP {resp.status_code}",
            )

        try:
            body = resp.json()
        except Exception:
            return ToolResult(
                success=False, output="Failed to parse FEC API response.",
            )

        results = body.get("results", [])
        pagination = body.get("pagination", {})
        total = pagination.get("count", len(results))

        if result_type == "candidates":
            records = _parse_candidates(results)
        elif result_type == "filings":
            records = _parse_filings(results)
        else:
            records = _parse_expenditures(results)

        signals = _compute_signals(records, result_type)
        summary = _format_summary(records, signals, result_type, total)

        result_data = {
            "records": records,
            "count": len(records),
            "total": total,
            "result_type": result_type,
            "signals": signals,
        }

        if self._cache:
            self._cache.set(
                cache_key,
                {"output": summary, "data": result_data},
                ttl=_CACHE_TTL,
            )

        return ToolResult(success=True, output=summary, data=result_data)


# ── Parsers (module-level for testability) ──────────────────────


def _parse_candidates(results: list) -> list[dict]:
    records = []
    for r in results:
        records.append({
            "candidate_id": r.get("candidate_id", ""),
            "name": r.get("name", ""),
            "party": r.get("party", ""),
            "office": r.get("office", ""),
            "office_full": r.get("office_full", ""),
            "state": r.get("state", ""),
            "district": r.get("district", ""),
            "incumbent_challenge": r.get("incumbent_challenge", ""),
            "cycles": r.get("cycles", []),
            "has_raised_funds": r.get("has_raised_funds", False),
            "candidate_status": r.get("candidate_status", ""),
        })
    return records


def _parse_filings(results: list) -> list[dict]:
    records = []
    for r in results:
        records.append({
            "committee_id": r.get("committee_id", ""),
            "committee_name": r.get("committee_name", ""),
            "form_type": r.get("form_type", ""),
            "receipt_date": r.get("receipt_date", ""),
            "coverage_start_date": r.get("coverage_start_date", ""),
            "coverage_end_date": r.get("coverage_end_date", ""),
            "total_receipts": r.get("total_receipts"),
            "total_disbursements": r.get("total_disbursements"),
            "cash_on_hand_end": r.get("cash_on_hand_end_period"),
            "debts_owed_by": r.get("debts_owed_by_committee"),
            "document_description": r.get("document_description", ""),
        })
    return records


def _parse_expenditures(results: list) -> list[dict]:
    records = []
    for r in results:
        # committee can be a nested dict or just committee_name string
        if isinstance(r.get("committee"), dict):
            committee_name = r["committee"].get("name", "")
        else:
            committee_name = r.get("committee_name", "")

        records.append({
            "committee_id": r.get("committee_id", ""),
            "committee_name": committee_name,
            "candidate_id": r.get("candidate_id", ""),
            "candidate_name": r.get("candidate_name", ""),
            "support_oppose": r.get("support_oppose_indicator", ""),
            "expenditure_amount": r.get("expenditure_amount"),
            "expenditure_date": r.get("expenditure_date", ""),
            "payee_name": r.get("payee_name", ""),
            "expenditure_description": r.get("expenditure_description", ""),
            "office": r.get("candidate_office", ""),
            "state": r.get("candidate_office_state", ""),
        })
    return records


# ── Signal computation ──────────────────────────────────────────


def _compute_signals(records: list[dict], result_type: str) -> dict[str, Any]:
    signals: dict[str, Any] = {}
    if not records:
        return signals

    if result_type == "candidates":
        parties: dict[str, int] = {}
        offices: dict[str, int] = {}
        active_fundraisers = 0
        for r in records:
            p = r.get("party") or "unknown"
            parties[p] = parties.get(p, 0) + 1
            o = r.get("office_full") or r.get("office") or "unknown"
            offices[o] = offices.get(o, 0) + 1
            if r.get("has_raised_funds"):
                active_fundraisers += 1
        signals["party_breakdown"] = parties
        signals["office_breakdown"] = offices
        signals["active_fundraisers"] = active_fundraisers

    elif result_type == "filings":
        total_cash = 0
        total_receipts_sum = 0
        filing_count = 0
        for r in records:
            if r.get("cash_on_hand_end") is not None:
                total_cash += r["cash_on_hand_end"]
                filing_count += 1
            if r.get("total_receipts") is not None:
                total_receipts_sum += r["total_receipts"]
        if filing_count:
            signals["avg_cash_on_hand"] = round(total_cash / filing_count, 2)
        signals["total_receipts_sum"] = round(total_receipts_sum, 2)

    elif result_type == "expenditures":
        support_total = 0.0
        oppose_total = 0.0
        support_count = 0
        oppose_count = 0
        by_candidate: dict[str, float] = {}
        for r in records:
            amt = r.get("expenditure_amount") or 0
            so = r.get("support_oppose", "")
            if so == "S":
                support_total += amt
                support_count += 1
            elif so == "O":
                oppose_total += amt
                oppose_count += 1
            cand = r.get("candidate_name") or "Unknown"
            by_candidate[cand] = by_candidate.get(cand, 0) + amt

        signals["support_total"] = round(support_total, 2)
        signals["oppose_total"] = round(oppose_total, 2)
        signals["support_count"] = support_count
        signals["oppose_count"] = oppose_count
        combined = support_total + oppose_total
        if combined > 0:
            signals["oppose_ratio"] = round(oppose_total / combined, 3)

        top_targets = sorted(
            by_candidate.items(), key=lambda x: x[1], reverse=True,
        )[:5]
        signals["top_targets"] = [
            {"candidate": c, "total_spent": round(s, 2)}
            for c, s in top_targets
        ]

    return signals


# ── Output formatting ───────────────────────────────────────────


def _format_summary(
    records: list[dict],
    signals: dict,
    result_type: str,
    total: int,
) -> str:
    parts = [f"FEC Political Risk — {result_type}"]
    parts.append(f"Results: {len(records)} returned / {total} total")

    if result_type == "candidates" and records:
        parts.append("\nCandidates:")
        for r in records[:10]:
            fund = "fundraising" if r.get("has_raised_funds") else "no funds"
            parts.append(
                f"  {r['name']} ({r['party']}) — "
                f"{r.get('office_full') or r.get('office', '?')} "
                f"{r.get('state', '')}{r.get('district', '')} [{fund}]"
            )
        if signals.get("party_breakdown"):
            parts.append(f"\nParty breakdown: {signals['party_breakdown']}")
        if signals.get("active_fundraisers") is not None:
            parts.append(
                f"Active fundraisers: "
                f"{signals['active_fundraisers']}/{len(records)}"
            )

    elif result_type == "filings" and records:
        parts.append("\nRecent filings:")
        for r in records[:8]:
            cash = (
                f"${r['cash_on_hand_end']:,.0f}"
                if r.get("cash_on_hand_end") is not None
                else "N/A"
            )
            parts.append(
                f"  {r['committee_name'][:50]} — {r.get('form_type', '?')} "
                f"({r.get('receipt_date', '?')}) — Cash: {cash}"
            )
        if "avg_cash_on_hand" in signals:
            parts.append(
                f"\nAvg cash on hand: ${signals['avg_cash_on_hand']:,.0f}"
            )

    elif result_type == "expenditures" and records:
        parts.append("\nIndependent expenditures:")
        for r in records[:8]:
            so = (
                "SUPPORT" if r.get("support_oppose") == "S"
                else "OPPOSE" if r.get("support_oppose") == "O"
                else "?"
            )
            amt = (
                f"${r['expenditure_amount']:,.0f}"
                if r.get("expenditure_amount")
                else "N/A"
            )
            parts.append(
                f"  {so} {r.get('candidate_name', '?')}: {amt} "
                f"by {r.get('committee_name', '?')[:40]} "
                f"({r.get('expenditure_date', '?')})"
            )
        if "support_total" in signals:
            parts.append(
                f"\nSupport: ${signals['support_total']:,.0f} "
                f"({signals.get('support_count', 0)} items)"
            )
            parts.append(
                f"Oppose: ${signals['oppose_total']:,.0f} "
                f"({signals.get('oppose_count', 0)} items)"
            )
            if "oppose_ratio" in signals:
                parts.append(
                    f"Oppose ratio: {signals['oppose_ratio']:.1%}"
                )
        if signals.get("top_targets"):
            parts.append("Top targeted candidates:")
            for t in signals["top_targets"]:
                parts.append(
                    f"  {t['candidate']}: ${t['total_spent']:,.0f}"
                )

    return "\n".join(parts)
