"""
Tool: Insider Filings — SEC Form 4 Cluster Detection

Fetches recent Form 4 filings from SEC EDGAR and detects insider buying
clusters. A cluster = 3+ distinct insiders at the same company making
open-market purchases within a 14-day window.

Single insider buys are noise. Clusters are one of the strongest free
equity signals in the literature (Lakonishok & Lee 2001, Jeng et al 2003:
clusters predict 30-day abnormal returns of 3-8%).

All data is free via EDGAR. Rate limit: 10 req/sec. User-Agent required.
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

_EFTS_BASE = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
_USER_AGENT = "TirraMind/1.0 (research@tirramind.com)"
_SEC_REQUEST_DELAY = 0.15  # seconds between requests (< 10 req/sec)


class InsiderFilingsTool(Tool):

    name = "insider_filings"

    description = (
        "Scan recent SEC Form 4 filings for insider buying clusters. "
        "A cluster is 3+ distinct corporate insiders (officers/directors) making "
        "open-market purchases at the same company within 14 days. "
        "This is one of the strongest free equity signals known — clusters of "
        "insider buying predict positive abnormal returns over the next 30 days. "
        "Data from SEC EDGAR, zero cost."
    )

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "days_back": {
                "type": "integer",
                "description": "How many days back to scan. Default: 30. Max: 90.",
                "default": 30,
            },
            "ticker": {
                "type": "string",
                "description": (
                    "Optional: filter to a specific company ticker. "
                    "If omitted, scans all companies."
                ),
                "default": "",
            },
            "min_cluster_size": {
                "type": "integer",
                "description": "Minimum distinct insiders for a cluster. Default: 3.",
                "default": 3,
            },
        },
        "required": [],
    }

    def __init__(self, cache: DataCache | None = None) -> None:
        self._cache = cache

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def execute(
        self,
        *,
        days_back: int = 30,
        ticker: str = "",
        min_cluster_size: int = 3,
        **_: Any,
    ) -> ToolResult:
        days_back = max(1, min(days_back, 90))
        min_cluster_size = max(2, min_cluster_size)
        ticker = ticker.strip().upper()

        end_dt = date.today()
        start_dt = end_dt - timedelta(days=days_back)

        try:
            raw_filings = self._fetch_recent_filings(start_dt, end_dt)
        except Exception as exc:
            log.exception("EDGAR fetch failed")
            return ToolResult(success=False, output=f"SEC EDGAR error: {exc}")

        if not raw_filings:
            return ToolResult(
                success=True,
                output=f"No Form 4 filings found for {start_dt} to {end_dt}.",
                data={"clusters": [], "total_filings": 0},
            )

        # Parse each filing to extract transactions
        transactions = self._parse_filings(raw_filings)

        if not transactions:
            return ToolResult(
                success=True,
                output="Found filings but no open-market purchases detected.",
                data={"clusters": [], "total_filings": len(raw_filings)},
            )

        # Filter by ticker if specified
        if ticker:
            transactions = [t for t in transactions if t["ticker"] == ticker]
            if not transactions:
                return ToolResult(
                    success=True,
                    output=f"No open-market insider purchases found for {ticker} in the last {days_back} days.",
                    data={"clusters": [], "total_filings": len(raw_filings)},
                )

        # Detect clusters
        clusters = self._detect_clusters(transactions, min_cluster_size)

        if not clusters:
            return ToolResult(
                success=True,
                output=f"Scanned {len(raw_filings)} filings, found {len(transactions)} purchases. No clusters of {min_cluster_size}+ insiders detected.",
                data={"clusters": [], "total_filings": len(raw_filings), "total_purchases": len(transactions)},
            )

        # Format output
        lines = [f"Insider Buying Clusters — {len(clusters)} found (last {days_back} days):\n"]
        for i, c in enumerate(clusters, 1):
            lines.append(
                f"  {i}. {c['ticker']} ({c['company']}) — {c['insider_count']} insiders, "
                f"${c['total_value']:,.0f} total\n"
                f"     Window: {c['cluster_start']} → {c['cluster_end']} | "
                f"Conviction: {c['conviction']}"
            )
            for ins in c["insiders"][:5]:  # show top 5 insiders
                lines.append(
                    f"       • {ins['name']}"
                    + (f" ({ins['role']})" if ins["role"] else "")
                    + f" — {ins['shares']:,.0f} shares @ ${ins['price']:.2f} on {ins['date']}"
                )

        output = "\n".join(lines)
        data = {
            "clusters": clusters,
            "total_filings": len(raw_filings),
            "total_purchases": len(transactions),
            "scan_range": {"start": str(start_dt), "end": str(end_dt)},
        }
        return ToolResult(success=True, output=output, data=data)

    # ------------------------------------------------------------------
    # Fetching from EDGAR
    # ------------------------------------------------------------------

    def _fetch_recent_filings(
        self, start_dt: date, end_dt: date
    ) -> list[dict[str, Any]]:
        """Fetch Form 4 filing metadata from EDGAR full-text search."""
        cache_params = {"start": str(start_dt), "end": str(end_dt)}
        if self._cache:
            cached = self._cache.get("insider_filings_search", cache_params)
            if cached is not None:
                log.debug("Cache hit for insider filings search")
                return cached

        all_hits: list[dict[str, Any]] = []
        page_from = 0
        page_size = 100

        with httpx.Client(timeout=20, headers={"User-Agent": _USER_AGENT}) as client:
            while True:
                time.sleep(_SEC_REQUEST_DELAY)

                params = {
                    "forms": "4",
                    "dateRange": "custom",
                    "startdt": str(start_dt),
                    "enddt": str(end_dt),
                    "from": str(page_from),
                    "size": str(page_size),
                }

                try:
                    resp = client.get(_EFTS_BASE, params=params)
                    resp.raise_for_status()
                    result = resp.json()
                except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                    if hasattr(exc, "response") and exc.response.status_code == 429:
                        log.warning("SEC rate limit hit, backing off 2s")
                        time.sleep(2)
                        continue
                    raise

                hits = result.get("hits", {}).get("hits", [])
                if not hits:
                    break

                all_hits.extend(hits)
                total = result.get("hits", {}).get("total", {}).get("value", 0)

                page_from += page_size
                if page_from >= total or page_from >= 500:
                    # Cap at 500 filings per scan to stay within reasonable bounds
                    break

        if self._cache and all_hits:
            self._cache.put("insider_filings_search", cache_params, all_hits)

        return all_hits

    def _fetch_filing_xml(self, cik: str, accession: str, primary_doc: str) -> str | None:
        """Fetch a single Form 4 XML from EDGAR archives."""
        # Normalize accession: remove dashes for URL path
        accession_clean = accession.replace("-", "")
        url = f"{_EDGAR_ARCHIVES}/{cik}/{accession_clean}/{primary_doc}"

        # Check permanent cache (filings never change)
        cache_params = {"url": url}
        if self._cache:
            cached = self._cache.get("insider_filing_xml", cache_params)
            if cached is not None:
                return cached

        time.sleep(_SEC_REQUEST_DELAY)

        try:
            with httpx.Client(timeout=15, headers={"User-Agent": _USER_AGENT}, follow_redirects=True) as client:
                resp = client.get(url)
                resp.raise_for_status()
                xml_text = resp.text
        except Exception as exc:
            log.debug("Failed to fetch filing XML %s: %s", url, exc)
            return None

        if self._cache and xml_text:
            self._cache.put("insider_filing_xml", cache_params, xml_text)

        return xml_text

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_filings(self, raw_filings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Parse EFTS hits into structured transaction records.

        For efficiency, first extracts metadata from EFTS results.
        Only fetches the full XML for filings that look like purchases
        (we can't tell from metadata alone, so we fetch all and filter).

        In practice, we parse metadata to get: CIK, company name, ticker,
        file date, accession number, primary document path. Then fetch
        the XML for transaction details.
        """
        transactions: list[dict[str, Any]] = []

        for hit in raw_filings:
            source = hit.get("_source", {})

            # Extract metadata
            ciks = source.get("ciks", [])
            names = source.get("display_names", [])
            file_date = source.get("file_date", "")
            accession = source.get("adsh", "")  # accession number

            if len(ciks) < 2 or len(names) < 2:
                continue  # Form 4 always has reporter CIK + issuer CIK

            # In EFTS results, first CIK/name is the reporting person,
            # second is the issuer (company)
            reporter_name = _clean_name(names[0])
            company_display = names[1] if len(names) > 1 else ""

            # EFTS display names use "COMPANY (CIK 000xxx)" format — no ticker.
            # Ticker comes from the XML (issuerTradingSymbol). Pass empty here.
            company = _extract_company_name(company_display)
            issuer_cik = ciks[1] if len(ciks) > 1 else ciks[0]

            if not accession:
                continue

            # Determine primary document name from the hit
            primary_doc = hit.get("_id", "").split(":", 1)[-1] if ":" in hit.get("_id", "") else ""
            if not primary_doc:
                primary_doc = "form4.xml"  # fallback

            # Try to get transaction details from XML
            xml_text = self._fetch_filing_xml(issuer_cik, accession, primary_doc)
            if xml_text:
                txns = _parse_form4_xml(xml_text, reporter_name, "", company, file_date)
                transactions.extend(txns)

        return transactions

    def _detect_clusters(
        self, transactions: list[dict[str, Any]], min_size: int = 3
    ) -> list[dict[str, Any]]:
        """Detect insider buying clusters: 3+ distinct insiders within 14 days."""

        # Group by ticker
        by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for txn in transactions:
            if txn["type"] == "P":  # open-market purchase only
                by_ticker[txn["ticker"]].append(txn)

        clusters: list[dict[str, Any]] = []

        for ticker, buys in by_ticker.items():
            if len(buys) < min_size:
                continue

            # Sort by date
            buys.sort(key=lambda t: t["date"])

            # Sliding window: find maximal clusters of distinct insiders within 14 days
            best_cluster = self._find_best_cluster(buys, window_days=14, min_size=min_size)
            if best_cluster:
                clusters.append(best_cluster)

        # Sort by conviction score (insider_count * total_value)
        clusters.sort(key=lambda c: c["insider_count"] * c["total_value"], reverse=True)
        return clusters

    @staticmethod
    def _find_best_cluster(
        buys: list[dict[str, Any]], window_days: int = 14, min_size: int = 3
    ) -> dict[str, Any] | None:
        """Find the densest cluster of distinct-insider buys within window_days."""
        if not buys:
            return None

        best: dict[str, Any] | None = None
        best_score = 0

        for i, anchor in enumerate(buys):
            anchor_date = _parse_date(anchor["date"])
            if anchor_date is None:
                continue

            window_end = anchor_date + timedelta(days=window_days)
            window_buys: list[dict[str, Any]] = []
            seen_names: set[str] = set()

            for j in range(i, len(buys)):
                buy_date = _parse_date(buys[j]["date"])
                if buy_date is None:
                    continue
                if buy_date > window_end:
                    break

                name_key = buys[j]["name"].upper().strip()
                if name_key not in seen_names:
                    seen_names.add(name_key)
                    window_buys.append(buys[j])

            if len(seen_names) >= min_size:
                total_value = sum(b["shares"] * b["price"] for b in window_buys if b["price"] > 0)
                score = len(seen_names) * total_value

                if score > best_score:
                    best_score = score
                    # Classify conviction
                    roles = [b.get("role", "").upper() for b in window_buys]
                    has_csuite = any(
                        r for r in roles
                        if any(title in r for title in ("CEO", "CFO", "COO", "PRESIDENT", "CHIEF"))
                    )

                    best = {
                        "ticker": buys[0]["ticker"],
                        "company": buys[0]["company"],
                        "insiders": sorted(window_buys, key=lambda b: b["date"]),
                        "cluster_start": window_buys[0]["date"],
                        "cluster_end": window_buys[-1]["date"],
                        "total_value": total_value,
                        "insider_count": len(seen_names),
                        "conviction": (
                            "high" if (has_csuite and len(seen_names) >= 4) else
                            "medium-high" if has_csuite else
                            "medium" if len(seen_names) >= 4 else
                            "moderate"
                        ),
                    }

        return best


# ------------------------------------------------------------------
# Form 4 XML parsing
# ------------------------------------------------------------------

def _parse_form4_xml(
    xml_text: str,
    fallback_name: str,
    fallback_ticker: str,
    fallback_company: str,
    fallback_date: str,
) -> list[dict[str, Any]]:
    """Parse a Form 4 XML document. Returns list of transaction dicts.

    Only extracts open-market purchases (transactionCode == "P") from
    nonDerivativeTransactions. Grants (A), sales (S), dispositions (D),
    options exercises (M) are excluded.

    Handles both X0407 and X0508 schema versions.
    """
    transactions: list[dict[str, Any]] = []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        log.debug("XML parse error for filing")
        return transactions

    # Strip namespace if present — EDGAR XMLs sometimes have xmlns
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    # Issuer info
    issuer = root.find(f"{ns}issuer")
    ticker = fallback_ticker
    company = fallback_company
    if issuer is not None:
        t = issuer.find(f"{ns}issuerTradingSymbol")
        if t is not None and t.text:
            ticker = t.text.strip().upper()
        c = issuer.find(f"{ns}issuerName")
        if c is not None and c.text:
            company = c.text.strip()

    # Reporter info
    reporter_name = fallback_name
    role = ""
    for owner in root.findall(f"{ns}reportingOwner"):
        rel = owner.find(f"{ns}reportingOwnerRelationship")
        id_elem = owner.find(f"{ns}reportingOwnerId")
        if id_elem is not None:
            name_elem = id_elem.find(f"{ns}rptOwnerName")
            if name_elem is not None and name_elem.text:
                reporter_name = _clean_name_raw(name_elem.text)
        if rel is not None:
            title_elem = rel.find(f"{ns}officerTitle")
            if title_elem is not None and title_elem.text:
                role = title_elem.text.strip()

    # Non-derivative transactions
    for txn_group in [
        root.find(f"{ns}nonDerivativeTable"),
    ]:
        if txn_group is None:
            continue
        for txn in txn_group.findall(f"{ns}nonDerivativeTransaction"):
            code_elem = txn.find(f".//{ns}transactionCode")
            code = code_elem.text.strip() if code_elem is not None and code_elem.text else ""

            if code != "P":
                continue  # only open-market purchases

            # Shares
            shares_elem = txn.find(f".//{ns}transactionShares/{ns}value")
            shares = _safe_float_or_zero(shares_elem.text if shares_elem is not None else None)

            # Price
            price_elem = txn.find(f".//{ns}transactionPricePerShare/{ns}value")
            price = _safe_float_or_zero(price_elem.text if price_elem is not None else None)

            # Date
            date_elem = txn.find(f".//{ns}transactionDate/{ns}value")
            txn_date = date_elem.text.strip() if date_elem is not None and date_elem.text else fallback_date

            if shares > 0:
                transactions.append({
                    "ticker": ticker,
                    "company": company,
                    "name": reporter_name,
                    "role": role,
                    "type": "P",
                    "shares": shares,
                    "price": price,
                    "date": txn_date,
                })

    return transactions


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_ticker(display_name: str) -> str:
    """Extract ticker from EDGAR display name like 'Apple Inc.  (AAPL)  (CIK ...)'."""
    import re
    # Match first parenthesized token that looks like a ticker (1-5 uppercase + optional suffix)
    m = re.search(r"\(([A-Z][A-Z0-9\-\.]{0,9})(?:,\s*[A-Z0-9\-\.]+)*\)", display_name)
    return m.group(1) if m else ""


def _extract_company_name(display_name: str) -> str:
    """Extract company name from EDGAR display name."""
    # Everything before the first '('
    idx = display_name.find("(")
    if idx > 0:
        return display_name[:idx].strip()
    return display_name.strip()


def _clean_name(display_name: str) -> str:
    """Clean EDGAR display name: 'COOK TIMOTHY D  (CIK 001234)' → 'COOK TIMOTHY D'."""
    idx = display_name.find("(CIK")
    if idx > 0:
        return display_name[:idx].strip()
    return display_name.strip()


def _clean_name_raw(name: str) -> str:
    """Normalize a raw insider name from XML."""
    return " ".join(name.strip().split())  # collapse whitespace


def _safe_float_or_zero(val: str | None) -> float:
    """Convert to float, default 0."""
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _parse_date(date_str: str) -> date | None:
    """Parse YYYY-MM-DD date string."""
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None
