"""
Tool: Treasury Receipts — US Daily Treasury Statement (DTS)

Fetch and parse the Daily Treasury Statement via the Treasury Fiscal Data API.
Provides operating cash balance (TGA), tax receipt breakdowns (deposits/withdrawals
by category), and public debt transactions.

Data source: https://api.fiscaldata.treasury.gov/ (zero cost, public domain, no auth).
Schedule: Daily (M-F), next-business-day release.
History: 10/03/2005 to present.

Signal theory:
  - Withheld income taxes = T+0 employment × wages proxy (faster than ADP/BLS)
  - Corporate tax deposits = real-time earnings proxy
  - Customs duties = real-time import volume proxy
  - TGA balance movement = overnight repo rate / money market liquidity driver
  - Public debt issuance pace = Treasury supply signal for bond markets
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

_BASE_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
_ENDPOINTS = {
    "cash_balance": "/v1/accounting/dts/operating_cash_balance",
    "deposits_withdrawals": "/v1/accounting/dts/deposits_withdrawals_operating_cash",
    "public_debt": "/v1/accounting/dts/public_debt_transactions",
}
_UA = "TirraMind/0.1 (research)"
_TIMEOUT = 20
_PAGE_SIZE = 500

VALID_MODES = frozenset(_ENDPOINTS)

# Date format used by the API
_DATE_FMT = "%Y-%m-%d"


def _parse_amount(val: str | None) -> float | None:
    """Parse a numeric amount string, returning None for missing/invalid."""
    if val is None:
        return None
    v = val.strip().replace(",", "")
    if not v or v in ("", "null", "-"):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _validate_date(date_str: str) -> bool:
    """Validate YYYY-MM-DD date string."""
    try:
        datetime.strptime(date_str, _DATE_FMT)
        return True
    except ValueError:
        return False


class TreasuryReceiptsTool(Tool):
    name = "treasury_receipts"
    description = (
        "Fetch US Daily Treasury Statement (DTS) data. "
        "Modes: cash_balance (TGA operating balance), "
        "deposits_withdrawals (tax receipts/outlays by category), "
        "public_debt (issuance/redemption transactions). "
        "Contains real-time fiscal signals: withheld income tax = employment proxy, "
        "corporate tax = earnings proxy, customs duties = import proxy, "
        "TGA balance = money market liquidity signal. Daily, public domain, no auth."
    )
    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": sorted(VALID_MODES),
                "description": (
                    "cash_balance = TGA operating balance; "
                    "deposits_withdrawals = tax receipts and outlays by category; "
                    "public_debt = debt issuance and redemption transactions"
                ),
            },
            "start_date": {
                "type": "string",
                "default": "",
                "description": "Start date (YYYY-MM-DD). Default: 30 days ago.",
            },
            "end_date": {
                "type": "string",
                "default": "",
                "description": "End date (YYYY-MM-DD). Default: today.",
            },
            "category_filter": {
                "type": "string",
                "default": "",
                "description": (
                    "For deposits_withdrawals mode: case-insensitive substring filter "
                    "on transaction description (e.g. 'income tax', 'customs', "
                    "'corporate'). Empty = all categories."
                ),
            },
            "top_n": {
                "type": "integer",
                "default": 25,
                "description": "Max records to return, sorted by most recent date.",
            },
        },
        "required": ["mode"],
    }

    def __init__(self, cache: DataCache | None = None) -> None:
        self._cache = cache

    # ── Public execute ───────────────────────────────────────────────

    def execute(
        self,
        *,
        mode: str = "cash_balance",
        start_date: str = "",
        end_date: str = "",
        category_filter: str = "",
        top_n: int = 25,
        **_: Any,
    ) -> ToolResult:
        if mode not in VALID_MODES:
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Choose from: {', '.join(sorted(VALID_MODES))}.",
            )

        # Validate dates if provided
        if start_date and not _validate_date(start_date):
            return ToolResult(
                success=False,
                output=f"Invalid start_date '{start_date}'. Use YYYY-MM-DD format.",
            )
        if end_date and not _validate_date(end_date):
            return ToolResult(
                success=False,
                output=f"Invalid end_date '{end_date}'. Use YYYY-MM-DD format.",
            )
        if start_date and end_date and start_date > end_date:
            return ToolResult(
                success=False,
                output=f"start_date ({start_date}) is after end_date ({end_date}).",
            )

        top_n = max(1, min(top_n, 1000))

        try:
            records = self._fetch(mode, start_date, end_date)
        except Exception as exc:
            log.exception("Treasury DTS fetch failed for mode=%s", mode)
            return ToolResult(success=False, output=f"Treasury API error: {exc}")

        if not records:
            return ToolResult(
                success=True,
                output=f"No data returned for mode='{mode}' "
                f"(dates: {start_date or 'default'} to {end_date or 'default'}).",
                data={"records": [], "mode": mode},
            )

        # Apply category filter for deposits_withdrawals
        if mode == "deposits_withdrawals" and category_filter:
            cf_lower = category_filter.lower()
            records = [
                r
                for r in records
                if cf_lower in r.get("transaction_catg", "").lower()
                or cf_lower in r.get("transaction_catg_desc", "").lower()
            ]

        # Sort by record_date descending, truncate
        records.sort(key=lambda r: r.get("record_date", ""), reverse=True)
        records = records[:top_n]

        # Build output
        if mode == "cash_balance":
            output, data = self._format_cash_balance(records)
        elif mode == "deposits_withdrawals":
            output, data = self._format_deposits_withdrawals(records)
        else:
            output, data = self._format_public_debt(records)

        return ToolResult(success=True, output=output, data=data)

    # ── Fetch ────────────────────────────────────────────────────────

    def _fetch(self, mode: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
        cache_key = {
            "source": f"treasury_dts_{mode}",
            "start": start_date,
            "end": end_date,
        }
        if self._cache:
            cached = self._cache.get("treasury_receipts", cache_key)
            if cached is not None:
                log.debug("Treasury DTS %s: cache hit", mode)
                return cached

        endpoint = _ENDPOINTS[mode]
        url = f"{_BASE_URL}{endpoint}"

        # Build filter string
        filters = []
        if start_date:
            filters.append(f"record_date:gte:{start_date}")
        if end_date:
            filters.append(f"record_date:lte:{end_date}")

        params: dict[str, str] = {
            "sort": "-record_date",
            "page[size]": str(_PAGE_SIZE),
            "format": "json",
        }
        if filters:
            params["filter"] = ",".join(filters)

        with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
            resp = client.get(url, params=params, headers={"User-Agent": _UA})
            resp.raise_for_status()

        payload = resp.json()
        records = payload.get("data", [])

        if self._cache and records:
            self._cache.put("treasury_receipts", cache_key, records)

        return records

    # ── Formatters ───────────────────────────────────────────────────

    def _format_cash_balance(self, records: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        lines = ["## US Treasury Operating Cash Balance (TGA)\n"]
        parsed = []

        for r in records:
            date = r.get("record_date", "?")
            acct = r.get("account_type", "?")
            # NOTE: the DTS operating_cash_balance endpoint always returns the
            # literal string "null" for close_today_bal — every row's real
            # balance figure (for both "Opening Balance" and "Closing
            # Balance" account_type rows) lives in open_today_bal instead.
            # Reading close_today_bal made every row print "N/A".
            today_bal = _parse_amount(r.get("open_today_bal"))
            month_bal = _parse_amount(r.get("open_month_bal"))
            fy_bal = _parse_amount(r.get("open_fiscal_year_bal"))

            today_str = f"${today_bal:,.0f}M" if today_bal is not None else "N/A"
            lines.append(f"  {date} | {acct}: {today_str}")

            parsed.append(
                {
                    "date": date,
                    "account_type": acct,
                    "close_today_bal": today_bal,
                    "open_month_bal": month_bal,
                    "open_fiscal_year_bal": fy_bal,
                }
            )

        # Compute TGA delta signal if we have multiple TGA entries
        tga_entries = [
            p
            for p in parsed
            if "federal" in (p.get("account_type") or "").lower()
            or "operating" in (p.get("account_type") or "").lower()
        ]
        signals: dict[str, Any] = {}
        if len(tga_entries) >= 2:
            latest = tga_entries[0].get("close_today_bal")
            prev = tga_entries[1].get("close_today_bal")
            if latest is not None and prev is not None and prev != 0:
                delta_pct = (latest - prev) / abs(prev) * 100
                signals["tga_daily_change_pct"] = round(delta_pct, 2)
                signals["tga_daily_change_abs"] = round(latest - prev, 0)
                lines.append(f"\n  Signal: TGA daily Δ = {signals['tga_daily_change_abs']:+,.0f}M ({delta_pct:+.2f}%)")

        lines.append(f"\n  Records: {len(records)}")
        return "\n".join(lines), {
            "mode": "cash_balance",
            "records": parsed,
            "signals": signals,
        }

    def _format_deposits_withdrawals(self, records: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        lines = ["## US Treasury Deposits & Withdrawals\n"]
        parsed = []

        for r in records:
            date = r.get("record_date", "?")
            txn_type = r.get("transaction_type", "?")
            # NOTE: the DTS deposits_withdrawals_operating_cash endpoint
            # always sends the literal string "null" for
            # transaction_catg_desc (a present, truthy value — `.get()`'s
            # default never kicks in), while the real category label lives
            # in transaction_catg. Preferring transaction_catg_desc made
            # every row print "null" as its category.
            catg = r.get("transaction_catg") or r.get("transaction_catg_desc") or "?"
            today_amt = _parse_amount(r.get("transaction_today_amt"))
            mtd_amt = _parse_amount(r.get("transaction_mtd_amt"))
            fytd_amt = _parse_amount(r.get("transaction_fytd_amt"))

            today_str = f"${today_amt:,.0f}M" if today_amt is not None else "N/A"
            lines.append(f"  {date} | {txn_type:10s} | {catg}: {today_str}")

            parsed.append(
                {
                    "date": date,
                    "type": txn_type,
                    "category": catg,
                    "today_amt": today_amt,
                    "mtd_amt": mtd_amt,
                    "fytd_amt": fytd_amt,
                }
            )

        # Summarize by category for the latest date
        signals: dict[str, Any] = {}
        if parsed:
            latest_date = parsed[0]["date"]
            latest_records = [p for p in parsed if p["date"] == latest_date]
            deposits = [p for p in latest_records if "deposit" in (p.get("type") or "").lower()]
            withdrawals = [p for p in latest_records if "withdraw" in (p.get("type") or "").lower()]

            total_deposits = sum(p["today_amt"] for p in deposits if p["today_amt"] is not None)
            total_withdrawals = sum(p["today_amt"] for p in withdrawals if p["today_amt"] is not None)
            signals["latest_date"] = latest_date
            signals["total_deposits_today"] = round(total_deposits, 0)
            signals["total_withdrawals_today"] = round(total_withdrawals, 0)
            signals["net_flow_today"] = round(total_deposits - total_withdrawals, 0)

            lines.append(
                f"\n  Latest ({latest_date}): Deposits=${total_deposits:,.0f}M, "
                f"Withdrawals=${total_withdrawals:,.0f}M, "
                f"Net=${total_deposits - total_withdrawals:+,.0f}M"
            )

        lines.append(f"\n  Records: {len(records)}")
        return "\n".join(lines), {
            "mode": "deposits_withdrawals",
            "records": parsed,
            "signals": signals,
        }

    def _format_public_debt(self, records: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        lines = ["## US Treasury Public Debt Transactions\n"]
        parsed = []

        for r in records:
            date = r.get("record_date", "?")
            txn_type = r.get("transaction_type", "?")
            today_amt = _parse_amount(r.get("transaction_today_amt"))
            mtd_amt = _parse_amount(r.get("transaction_mtd_amt"))
            fytd_amt = _parse_amount(r.get("transaction_fytd_amt"))

            today_str = f"${today_amt:,.0f}M" if today_amt is not None else "N/A"
            lines.append(f"  {date} | {txn_type}: {today_str}")

            parsed.append(
                {
                    "date": date,
                    "type": txn_type,
                    "today_amt": today_amt,
                    "mtd_amt": mtd_amt,
                    "fytd_amt": fytd_amt,
                }
            )

        lines.append(f"\n  Records: {len(records)}")
        return "\n".join(lines), {"mode": "public_debt", "records": parsed}
