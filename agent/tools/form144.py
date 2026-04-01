"""
Tool: Form 144 — Insider Sell Intent Detection

Fetches SEC Form 144 filings and detects insider sell-intent clusters.
Form 144 is filed BEFORE or concurrently with a sell order (T+0), giving
2+ days of lead time over Form 4 (filed T+2 post-execution).

Most 144s are routine RSU/PSU vesting sells (noise). The tool classifies
acquisition type and weights accordingly: open-market-acquired shares
being sold = strong conviction signal. RSU tax withholding = low signal.

Cluster detection: 2+ distinct insiders filing Form 144 at the same
company within 14 days → sell-intent cluster, ranked by dollar value
and acquisition signal quality.

All data free via EDGAR EFTS. Rate limit: 10 req/sec. User-Agent required.
"""

from __future__ import annotations

import logging
import re
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

# Acquisition type → signal weight for cluster scoring
_ACQ_WEIGHTS: dict[str, float] = {
    "open_market": 3.0,
    "private_placement": 2.0,
    "other": 1.0,
    "vesting": 0.5,
    "gift": 0.0,
}


class Form144Tool(Tool):

    name = "form144"

    description = (
        "Scan SEC Form 144 filings for insider sell-intent clusters. "
        "Form 144 is filed BEFORE the insider sells — earlier signal than Form 4. "
        "Detects 2+ insiders at the same company planning to sell within 14 days. "
        "Classifies acquisition type (open market = high signal, RSU/PSU vesting = low signal). "
        "Data from SEC EDGAR, zero cost."
    )

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "days_back": {
                "type": "integer",
                "description": "How many days back to scan. Default: 14. Max: 60.",
                "default": 14,
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
                "description": "Minimum distinct insiders for a sell-intent cluster. Default: 2.",
                "default": 2,
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
        days_back: int = 14,
        ticker: str = "",
        min_cluster_size: int = 2,
        **_: Any,
    ) -> ToolResult:
        days_back = max(1, min(days_back, 60))
        min_cluster_size = max(2, min_cluster_size)
        ticker = ticker.strip().upper()

        end_dt = date.today()
        start_dt = end_dt - timedelta(days=days_back)

        try:
            raw_hits = self._fetch_recent_144s(start_dt, end_dt)
        except Exception as exc:
            log.exception("EDGAR fetch failed")
            return ToolResult(success=False, output=f"SEC EDGAR error: {exc}")

        if not raw_hits:
            return ToolResult(
                success=True,
                output=f"No Form 144 filings found for {start_dt} to {end_dt}.",
                data={"clusters": [], "total_filings": 0},
            )

        # Parse each filing
        filings = self._parse_filings(raw_hits)

        if not filings:
            return ToolResult(
                success=True,
                output="Found filings but could not parse any sell-intent records.",
                data={"clusters": [], "total_filings": len(raw_hits)},
            )

        # Filter by ticker if specified
        if ticker:
            filings = [f for f in filings if f["ticker"] == ticker]
            if not filings:
                return ToolResult(
                    success=True,
                    output=f"No Form 144 filings found for {ticker} in the last {days_back} days.",
                    data={"clusters": [], "total_filings": len(raw_hits)},
                )

        # Detect clusters
        clusters = self._detect_sell_clusters(filings, min_cluster_size)

        if not clusters:
            return ToolResult(
                success=True,
                output=(
                    f"Scanned {len(raw_hits)} Form 144 filings, parsed {len(filings)} sell intents. "
                    f"No clusters of {min_cluster_size}+ insiders detected."
                ),
                data={
                    "clusters": [],
                    "total_filings": len(raw_hits),
                    "total_parsed": len(filings),
                },
            )

        # Format output
        lines = [f"Insider Sell-Intent Clusters — {len(clusters)} found (last {days_back} days):\n"]
        for i, c in enumerate(clusters, 1):
            pct = f"{c['pct_of_outstanding']:.3f}%" if c["pct_of_outstanding"] > 0 else "N/A"
            lines.append(
                f"  {i}. {c['ticker']} ({c['company']}) — {c['insider_count']} insiders, "
                f"${c['total_value']:,.0f} total ({pct} of outstanding)\n"
                f"     Window: {c['cluster_start']} → {c['cluster_end']} | "
                f"Urgency: {c['urgency']} | Conviction: {c['conviction']}"
            )
            for fil in c["filings"][:5]:
                acq_tag = f" [{fil['acquisition_type']}]" if fil["acquisition_type"] != "other" else ""
                lines.append(
                    f"       ⊖ {fil['insider_name']}"
                    + (f" ({fil['relationship']})" if fil["relationship"] else "")
                    + f" — {fil['shares_to_sell']:,.0f} shares, ${fil['dollar_value']:,.0f}"
                    + f" on {fil['filing_date']}{acq_tag}"
                )

        output = "\n".join(lines)
        data = {
            "clusters": clusters,
            "total_filings": len(raw_hits),
            "total_parsed": len(filings),
            "scan_range": {"start": str(start_dt), "end": str(end_dt)},
        }
        return ToolResult(success=True, output=output, data=data)

    # ------------------------------------------------------------------
    # EFTS fetching
    # ------------------------------------------------------------------

    def _fetch_recent_144s(
        self, start_dt: date, end_dt: date
    ) -> list[dict[str, Any]]:
        """Fetch Form 144 filing metadata from EDGAR full-text search."""
        cache_params = {"form": "144", "start": str(start_dt), "end": str(end_dt)}
        if self._cache:
            cached = self._cache.get("form144_search", cache_params)
            if cached is not None:
                log.debug("Cache hit for Form 144 search")
                return cached

        all_hits: list[dict[str, Any]] = []
        page_from = 0
        page_size = 100

        with httpx.Client(timeout=20, headers={"User-Agent": _USER_AGENT}) as client:
            while True:
                time.sleep(_SEC_REQUEST_DELAY)

                params = {
                    "forms": "144",
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
                    if hasattr(exc, "response") and exc.response.status_code >= 500:
                        log.warning("SEC server error at offset %d, returning %d hits collected", page_from, len(all_hits))
                        break  # Return what we have instead of failing
                    raise

                hits = result.get("hits", {}).get("hits", [])
                if not hits:
                    break

                all_hits.extend(hits)
                total = result.get("hits", {}).get("total", {}).get("value", 0)

                page_from += page_size
                if page_from >= total or page_from >= 500:
                    break

        if self._cache and all_hits:
            self._cache.put("form144_search", cache_params, all_hits)

        return all_hits

    def _fetch_filing_xml(self, cik: str, accession: str, primary_doc: str) -> str | None:
        """Fetch a single Form 144 XML from EDGAR archives."""
        accession_clean = accession.replace("-", "")
        url = f"{_EDGAR_ARCHIVES}/{cik}/{accession_clean}/{primary_doc}"

        cache_params = {"url": url}
        if self._cache:
            cached = self._cache.get("form144_xml", cache_params)
            if cached is not None:
                return cached

        time.sleep(_SEC_REQUEST_DELAY)

        try:
            with httpx.Client(
                timeout=15, headers={"User-Agent": _USER_AGENT}, follow_redirects=True
            ) as client:
                resp = client.get(url)
                resp.raise_for_status()
                xml_text = resp.text
        except Exception as exc:
            log.debug("Failed to fetch Form 144 XML %s: %s", url, exc)
            return None

        if self._cache and xml_text:
            self._cache.put("form144_xml", cache_params, xml_text)

        return xml_text

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_filings(self, raw_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Parse EFTS hits into structured sell-intent records.

        Two-phase approach for performance:
        1. Extract metadata from EFTS (fast, no network) → group by ticker
        2. Only fetch XMLs for companies with 2+ filings (cluster candidates)
        This avoids fetching 300+ individual XMLs when most are singletons.
        """
        # Phase 1: Extract metadata from EFTS results
        metadata_by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for hit in raw_hits:
            source = hit.get("_source", {})
            ciks = source.get("ciks", [])
            names = source.get("display_names", [])
            file_date = source.get("file_date", "")
            accession = source.get("adsh", "")

            if len(ciks) < 2 or len(names) < 2:
                continue

            # Find the display name with a ticker (issuer)
            issuer_display = names[0]
            issuer_cik = ciks[0]
            filer_name = _clean_name(names[1].split("(CIK")[0]) if len(names) > 1 else ""
            ticker = _extract_ticker(issuer_display)

            if not ticker and len(names) > 1:
                ticker = _extract_ticker(names[1])
                if ticker:
                    issuer_display = names[1]
                    issuer_cik = ciks[1] if len(ciks) > 1 else ciks[0]
                    filer_name = _clean_name(names[0].split("(CIK")[0])

            if not ticker or not accession:
                continue

            company = _extract_company_name(issuer_display)

            primary_doc = (
                hit.get("_id", "").split(":", 1)[-1]
                if ":" in hit.get("_id", "")
                else ""
            )
            if not primary_doc:
                primary_doc = "primary_doc.xml"

            metadata_by_ticker[ticker].append({
                "ticker": ticker,
                "company": company,
                "filer_name": filer_name,
                "issuer_cik": issuer_cik,
                "accession": accession,
                "primary_doc": primary_doc,
                "file_date": file_date,
            })

        # Phase 2: Only fetch XMLs for tickers with 2+ filings (cluster candidates)
        filings: list[dict[str, Any]] = []

        for ticker, metas in metadata_by_ticker.items():
            # Count distinct filers
            distinct_filers = {_normalize_name(m["filer_name"]) for m in metas}
            if len(distinct_filers) < 2:
                # Still include a metadata-only record for single filers
                # (useful for ticker-specific queries)
                for m in metas:
                    filings.append({
                        "ticker": ticker,
                        "company": m["company"],
                        "insider_name": m["filer_name"],
                        "relationship": "",
                        "shares_to_sell": 0,
                        "dollar_value": 0.0,
                        "shares_outstanding": 0,
                        "approx_sale_date": "",
                        "filing_date": m["file_date"],
                        "exchange": "",
                        "broker": "",
                        "acquisition_type": "other",
                        "acquisition_details": [],
                        "is_gift": False,
                        "has_10b5_1_plan": False,
                        "urgency": "unknown",
                        "_metadata_only": True,
                    })
                continue

            # Cluster candidate — fetch full XML for rich data
            # Cap XML fetches per ticker to limit network time
            fetched = 0
            for m in metas:
                if fetched >= 15:  # max 15 XMLs per ticker
                    break
                xml_text = self._fetch_filing_xml(
                    m["issuer_cik"], m["accession"], m["primary_doc"]
                )
                fetched += 1
                if not xml_text:
                    # Fallback to metadata-only record
                    filings.append({
                        "ticker": ticker,
                        "company": m["company"],
                        "insider_name": m["filer_name"],
                        "relationship": "",
                        "shares_to_sell": 0,
                        "dollar_value": 0.0,
                        "shares_outstanding": 0,
                        "approx_sale_date": "",
                        "filing_date": m["file_date"],
                        "exchange": "",
                        "broker": "",
                        "acquisition_type": "other",
                        "acquisition_details": [],
                        "is_gift": False,
                        "has_10b5_1_plan": False,
                        "urgency": "unknown",
                        "_metadata_only": True,
                    })
                    continue

                parsed = _parse_form144_xml(
                    xml_text, ticker, m["company"], m["file_date"]
                )
                if parsed and not parsed.get("is_gift"):
                    parsed["ticker"] = ticker
                    filings.append(parsed)

        return filings

    def _detect_sell_clusters(
        self, filings: list[dict[str, Any]], min_size: int = 2
    ) -> list[dict[str, Any]]:
        """Detect sell-intent clusters: 2+ distinct insiders within 14 days."""
        by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for f in filings:
            by_ticker[f["ticker"]].append(f)

        clusters: list[dict[str, Any]] = []

        for ticker, ticker_filings in by_ticker.items():
            if len(ticker_filings) < min_size:
                continue

            ticker_filings.sort(key=lambda f: f["filing_date"])
            best = self._find_best_sell_cluster(ticker_filings, window_days=14, min_size=min_size)
            if best:
                clusters.append(best)

        clusters.sort(key=lambda c: c["score"], reverse=True)
        return clusters

    @staticmethod
    def _find_best_sell_cluster(
        filings: list[dict[str, Any]], window_days: int = 14, min_size: int = 2
    ) -> dict[str, Any] | None:
        """Find the densest cluster of distinct-insider sell intents."""
        if not filings:
            return None

        best: dict[str, Any] | None = None
        best_score = 0.0

        for i, anchor in enumerate(filings):
            anchor_date = _parse_date_iso(anchor["filing_date"])
            if anchor_date is None:
                continue

            window_end = anchor_date + timedelta(days=window_days)
            window_filings: list[dict[str, Any]] = []
            seen_names: set[str] = set()

            for j in range(i, len(filings)):
                f_date = _parse_date_iso(filings[j]["filing_date"])
                if f_date is None:
                    continue
                if f_date > window_end:
                    break

                name_key = _normalize_name(filings[j]["insider_name"])
                if name_key not in seen_names:
                    seen_names.add(name_key)
                    window_filings.append(filings[j])

            if len(seen_names) >= min_size:
                total_value = sum(f["dollar_value"] for f in window_filings)
                max_acq_weight = max(
                    _ACQ_WEIGHTS.get(f["acquisition_type"], 1.0)
                    for f in window_filings
                )
                score = len(seen_names) * total_value * max_acq_weight

                if score > best_score:
                    best_score = score

                    # Urgency from filing date vs approx sale date
                    urgencies = [f.get("urgency", "unknown") for f in window_filings]
                    has_immediate = any(u == "immediate" for u in urgencies)
                    has_near = any(u == "near_term" for u in urgencies)

                    # Pct of outstanding
                    total_shares = sum(f["shares_to_sell"] for f in window_filings)
                    outstanding = max(
                        (f.get("shares_outstanding", 0) for f in window_filings), default=0
                    )
                    pct = (total_shares / outstanding * 100) if outstanding > 0 else 0.0

                    # Conviction
                    has_voluntary = any(
                        f["acquisition_type"] in ("open_market", "private_placement")
                        for f in window_filings
                    )
                    has_officer = any(
                        any(r in f.get("relationship", "").upper()
                            for r in ("OFFICER", "DIRECTOR", "CEO", "CFO", "COO", "PRESIDENT", "CHIEF"))
                        for f in window_filings
                    )

                    conviction = (
                        "high" if (has_voluntary and len(seen_names) >= 3) else
                        "high" if (has_officer and has_voluntary) else
                        "medium-high" if has_voluntary else
                        "medium-high" if (has_officer and len(seen_names) >= 3) else
                        "medium" if has_officer else
                        "moderate"
                    )

                    best = {
                        "ticker": filings[0]["ticker"],
                        "company": filings[0]["company"],
                        "insider_count": len(seen_names),
                        "total_value": total_value,
                        "total_shares": total_shares,
                        "pct_of_outstanding": round(pct, 4),
                        "cluster_start": window_filings[0]["filing_date"],
                        "cluster_end": window_filings[-1]["filing_date"],
                        "urgency": (
                            "immediate" if has_immediate else
                            "near_term" if has_near else
                            "planned"
                        ),
                        "conviction": conviction,
                        "has_voluntary_sells": has_voluntary,
                        "score": score,
                        "filings": sorted(window_filings, key=lambda f: f["filing_date"]),
                    }

        return best


# ------------------------------------------------------------------
# Form 144 XML parsing
# ------------------------------------------------------------------

def _parse_form144_xml(
    xml_text: str,
    fallback_ticker: str,
    fallback_company: str,
    fallback_date: str,
) -> dict[str, Any] | None:
    """Parse a Form 144 XML document into a sell-intent record.

    Returns None if parsing fails or the filing is a gift transaction.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        log.debug("XML parse error for Form 144")
        return None

    # Strip namespace
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    form_data = root.find(f"{ns}formData")
    if form_data is None:
        return None

    # --- Issuer info ---
    issuer = form_data.find(f"{ns}issuerInfo")
    company = fallback_company
    insider_name = ""
    relationship = ""

    if issuer is not None:
        name_el = issuer.find(f"{ns}issuerName")
        if name_el is not None and name_el.text:
            company = name_el.text.strip()

        # Long element name for the person selling
        person_el = issuer.find(
            f"{ns}nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold"
        )
        if person_el is not None and person_el.text:
            insider_name = _clean_name(person_el.text)

        rels = issuer.find(f"{ns}relationshipsToIssuer")
        if rels is not None:
            rel_el = rels.find(f"{ns}relationshipToIssuer")
            if rel_el is not None and rel_el.text:
                relationship = rel_el.text.strip()

    if not insider_name:
        return None  # Can't identify who's selling

    # --- Securities information (the sell intent) ---
    sec_info = form_data.find(f"{ns}securitiesInformation")
    shares_to_sell = 0
    dollar_value = 0.0
    shares_outstanding = 0
    approx_sale_date = ""
    exchange = ""
    broker = ""

    if sec_info is not None:
        shares_el = sec_info.find(f"{ns}noOfUnitsSold")
        shares_to_sell = _safe_int(shares_el.text if shares_el is not None else None)

        value_el = sec_info.find(f"{ns}aggregateMarketValue")
        dollar_value = _safe_float(value_el.text if value_el is not None else None)

        outstanding_el = sec_info.find(f"{ns}noOfUnitsOutstanding")
        shares_outstanding = _safe_int(outstanding_el.text if outstanding_el is not None else None)

        date_el = sec_info.find(f"{ns}approxSaleDate")
        if date_el is not None and date_el.text:
            approx_sale_date = _parse_mmddyyyy(date_el.text.strip())

        exchange_el = sec_info.find(f"{ns}securitiesExchangeName")
        if exchange_el is not None and exchange_el.text:
            exchange = exchange_el.text.strip()

        broker_el = sec_info.find(f"{ns}brokerOrMarketmakerDetails")
        if broker_el is not None:
            bname = broker_el.find(f"{ns}name")
            if bname is not None and bname.text:
                broker = bname.text.strip()

    if shares_to_sell <= 0 and dollar_value <= 0:
        return None  # Nothing to sell

    # --- Acquisition details (securitiesToBeSold[]) ---
    acquisition_details: list[dict[str, Any]] = []
    is_gift = False
    acquisition_types: list[str] = []

    for stbs in form_data.findall(f"{ns}securitiesToBeSold"):
        nature_el = stbs.find(f"{ns}natureOfAcquisitionTransaction")
        nature_text = nature_el.text.strip() if nature_el is not None and nature_el.text else ""

        gift_el = stbs.find(f"{ns}isGiftTransaction")
        gift_flag = gift_el.text.strip().upper() if gift_el is not None and gift_el.text else "N"
        if gift_flag == "Y":
            is_gift = True

        acq_type = _classify_acquisition(nature_text, gift_flag == "Y")
        acquisition_types.append(acq_type)

        acquired_date_el = stbs.find(f"{ns}acquiredDate")
        acquired_date = ""
        if acquired_date_el is not None and acquired_date_el.text:
            acquired_date = _parse_mmddyyyy(acquired_date_el.text.strip())

        amount_el = stbs.find(f"{ns}amountOfSecuritiesAcquired")
        amount = _safe_int(amount_el.text if amount_el is not None else None)

        acquisition_details.append({
            "nature": nature_text,
            "type": acq_type,
            "acquired_date": acquired_date,
            "amount": amount,
            "is_gift": gift_flag == "Y",
        })

    # If ALL acquisition lots are gifts, skip
    if is_gift and all(a["is_gift"] for a in acquisition_details):
        return None

    # Best (highest signal) acquisition type
    best_acq_type = "other"
    if acquisition_types:
        best_acq_type = max(acquisition_types, key=lambda t: _ACQ_WEIGHTS.get(t, 1.0))

    # --- Notice date ---
    notice_sig = form_data.find(f"{ns}noticeSignature")
    notice_date = fallback_date
    has_10b5_1_plan = False
    if notice_sig is not None:
        nd_el = notice_sig.find(f"{ns}noticeDate")
        if nd_el is not None and nd_el.text:
            notice_date = _parse_mmddyyyy(nd_el.text.strip())
        plan_el = notice_sig.find(f"{ns}planAdoptionDates")
        if plan_el is not None and plan_el.text and plan_el.text.strip():
            has_10b5_1_plan = True

    # Use noticeDate as filing_date if available, else fallback
    filing_date = notice_date if notice_date else fallback_date

    # --- Urgency ---
    urgency = _classify_urgency(filing_date, approx_sale_date)

    return {
        "ticker": fallback_ticker,
        "company": company,
        "insider_name": insider_name,
        "relationship": relationship,
        "shares_to_sell": shares_to_sell,
        "dollar_value": dollar_value,
        "shares_outstanding": shares_outstanding,
        "approx_sale_date": approx_sale_date,
        "filing_date": filing_date,
        "exchange": exchange,
        "broker": broker,
        "acquisition_type": best_acq_type,
        "acquisition_details": acquisition_details,
        "is_gift": is_gift and all(a["is_gift"] for a in acquisition_details),
        "has_10b5_1_plan": has_10b5_1_plan,
        "urgency": urgency,
    }


# ------------------------------------------------------------------
# Classification helpers
# ------------------------------------------------------------------

def _classify_acquisition(nature_text: str, is_gift: bool) -> str:
    """Classify acquisition type from natureOfAcquisitionTransaction text."""
    if is_gift:
        return "gift"
    lower = nature_text.lower()
    if "open market" in lower or "market purchase" in lower:
        return "open_market"
    if "private" in lower or "placement" in lower:
        return "private_placement"
    if any(kw in lower for kw in (
        "stock unit", "rsu", "psu", "restricted", "performance",
        "option", "incentive", "vest", "award", "bonus", "compensation",
    )):
        return "vesting"
    return "other"


def _classify_urgency(filing_date: str, approx_sale_date: str) -> str:
    """Classify urgency from gap between filing and planned sale."""
    f_date = _parse_date_iso(filing_date)
    s_date = _parse_date_iso(approx_sale_date)
    if f_date is None or s_date is None:
        return "unknown"
    gap = (s_date - f_date).days
    if gap <= 1:
        return "immediate"
    if gap <= 7:
        return "near_term"
    return "planned"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_ticker(display_name: str) -> str:
    """Extract ticker from EFTS display name: 'Apple Inc.  (AAPL)  (CIK ...)'."""
    m = re.search(r"\(([A-Z][A-Z0-9\-\.]{0,9})(?:,\s*[A-Z0-9\-\.]+)*\)", display_name)
    return m.group(1) if m else ""


def _extract_company_name(display_name: str) -> str:
    """Extract company name from EFTS display name — everything before first '('."""
    idx = display_name.find("(")
    if idx > 0:
        return display_name[:idx].strip()
    return display_name.strip()


def _clean_name(text: str) -> str:
    """Normalize a name string: collapse whitespace."""
    return " ".join(text.strip().split())


def _normalize_name(name: str) -> str:
    """Normalize name for dedup: uppercase, strip suffixes like L.P., Inc., LLC."""
    n = name.upper().strip()
    for suffix in (", L.P.", ", LP", ", INC.", ", INC", ", LLC", ", LTD.", ", LTD"):
        n = n.removesuffix(suffix)
    return n.strip()


def _parse_mmddyyyy(text: str) -> str:
    """Convert MM/DD/YYYY to YYYY-MM-DD. Returns '' on failure."""
    try:
        dt = datetime.strptime(text.strip(), "%m/%d/%Y")
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        # Maybe it's already YYYY-MM-DD
        try:
            datetime.strptime(text.strip()[:10], "%Y-%m-%d")
            return text.strip()[:10]
        except (ValueError, TypeError):
            return ""


def _parse_date_iso(date_str: str) -> date | None:
    """Parse YYYY-MM-DD date string."""
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _safe_float(val: str | None) -> float:
    """Convert to float, default 0.0."""
    if val is None:
        return 0.0
    try:
        return float(val.replace(",", ""))
    except (ValueError, TypeError):
        return 0.0


def _safe_int(val: str | None) -> int:
    """Convert to int, default 0."""
    if val is None:
        return 0
    try:
        return int(float(val.replace(",", "")))
    except (ValueError, TypeError):
        return 0
