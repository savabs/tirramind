"""
Tool: Bankruptcy Court — Global Legal Filings & Enforcement Monitor

Aggregates real-time bankruptcy filings, SEC enforcement actions, and UK insolvency
notices from free public feeds.  No auth required.

Four modes:
  us_bankruptcy    — PACER RSS from 6 major US bankruptcy courts (SDNY, Delaware,
                     S.D. Texas, C.D. California, N.D. Illinois, D. New Jersey).
                     Covers ~90% of large corporate Chapter 11 filings.
  sec_enforcement  — SEC Administrative Proceedings + Litigation Releases RSS.
                     Direct enforcement actions against companies/individuals.
  sec_bankruptcy   — SEC EFTS 8-K Item 1.03 filings. Companies self-reporting
                     entry into bankruptcy or receivership.
  uk_insolvency    — UK Gazette insolvency notices (Atom feed) + UK Serious Fraud
                     Office investigations (GOV.UK API).

Signal theory:
  - Chapter 11 filings in PACER appear hours before wire services digest
  - SEC enforcement = direct regulatory action with immediate market impact
  - 8-K Item 1.03 = company self-reporting insolvency (terminal disclosure)
  - Cross-jurisdiction clustering in same SIC code = systemic stress signal
  - Chapter 7 (liquidation) vs 11 (restructuring) = very different severity
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import defusedxml.ElementTree as ET
import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

try:
    from agent.pipeline.entity import entity_id_from_key as _entity_id_from_key
except ImportError:  # pragma: no cover — optional dependency
    _entity_id_from_key = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

# Browser-like UA — SEC returns 403 for custom User-Agents
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)"
_TIMEOUT = 12

# ── PACER Courts ──────────────────────────────────────────────────────────────
PACER_COURTS: dict[str, tuple[str, str]] = {
    "sdny": ("S.D. New York", "ecf.nysb.uscourts.gov"),
    "del": ("Delaware", "ecf.deb.uscourts.gov"),
    "sdtx": ("S.D. Texas", "ecf.txsb.uscourts.gov"),
    "cdca": ("C.D. California", "ecf.cacb.uscourts.gov"),
    "ndil": ("N.D. Illinois", "ecf.ilnb.uscourts.gov"),
    "nj": ("D. New Jersey", "ecf.njb.uscourts.gov"),
}

# ── SEC URLs ──────────────────────────────────────────────────────────────────
_SEC_ADMIN_RSS = "https://www.sec.gov/rss/litigation/admin.xml"
_SEC_LIT_RSS = "https://www.sec.gov/rss/litigation/litreleases.xml"
_SEC_EFTS = "https://efts.sec.gov/LATEST/search-index"

# ── UK URLs ───────────────────────────────────────────────────────────────────
_UK_GAZETTE = "https://www.thegazette.co.uk/insolvency/data.feed"
_GOV_UK_SEARCH = "https://www.gov.uk/api/search.json"

# Chapter extraction regex
_CHAPTER_RE = re.compile(r"\bch(?:apter)?[\s.]*(\d{1,2})\b", re.IGNORECASE)
# PACER title: case_number followed by debtor name
_TITLE_RE = re.compile(r"^(\S+)\s+(.+)$")


# ── Helper functions ──────────────────────────────────────────────────────────


def _parse_chapter(text: str) -> str | None:
    """Extract bankruptcy chapter from description text."""
    m = _CHAPTER_RE.search(text)
    if m:
        ch = m.group(1)
        if ch in ("7", "11", "13", "15"):
            return ch
    return None


def _parse_pub_date(text: str) -> str:
    """Parse RSS pubDate to ISO format.  Returns original on failure."""
    text = text.strip()
    if not text:
        return ""
    # RFC 2822:  "Thu, 26 Mar 2026 16:14:40 -0400"
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            continue
    return text


def _keyword_match(text: str, keyword: str) -> bool:
    """Case-insensitive keyword filter."""
    if not keyword:
        return True
    return keyword.lower() in text.lower()


def _fetch_xml(url: str, client: httpx.Client) -> ET.Element | None:
    """Fetch URL and parse as XML.  Returns root element or None."""
    try:
        r = client.get(url)
        if r.status_code != 200:
            log.warning("HTTP %d from %s", r.status_code, url)
            return None
        return ET.fromstring(r.content)
    except (httpx.HTTPError, ET.ParseError) as exc:
        log.warning("Error fetching %s: %s", url, exc)
        return None


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


# ── PACER parsing ─────────────────────────────────────────────────────────────


def _parse_pacer_feed(root: ET.Element, court_code: str, court_name: str) -> list[dict[str, Any]]:
    """Parse a PACER RSS feed into structured entries."""
    entries: list[dict[str, Any]] = []
    channel = root.find("channel")
    if channel is None:
        return entries

    for item in channel.findall("item"):
        title_raw = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        desc = (item.findtext("description") or "").strip()
        pub_date = _parse_pub_date(item.findtext("pubDate") or "")

        # Parse case number and debtor name from title
        m = _TITLE_RE.match(title_raw)
        if m:
            case_number = m.group(1).strip()
            debtor_name = m.group(2).strip()
        else:
            case_number = ""
            debtor_name = title_raw

        # Detect chapter type from description or title
        chapter = _parse_chapter(desc) or _parse_chapter(title_raw)

        entries.append(
            {
                "case_number": case_number,
                "debtor_name": debtor_name,
                "chapter": chapter,
                "court": court_code,
                "court_name": court_name,
                "link": link,
                "pub_date": pub_date,
                "description": desc[:300],
            }
        )
    return entries


def _fetch_pacer_court(court_code: str, client: httpx.Client) -> list[dict[str, Any]]:
    """Fetch and parse a single PACER court's RSS feed."""
    court_name, domain = PACER_COURTS[court_code]
    url = f"https://{domain}/cgi-bin/rss_outside.pl"
    root = _fetch_xml(url, client)
    if root is None:
        return []
    return _parse_pacer_feed(root, court_code, court_name)


# ── SEC enforcement parsing ──────────────────────────────────────────────────


def _parse_sec_rss(root: ET.Element, enforce_type: str) -> list[dict[str, Any]]:
    """Parse an SEC RSS feed (admin or litigation)."""
    entries: list[dict[str, Any]] = []
    channel = root.find("channel")
    if channel is None:
        return entries

    for item in channel.findall("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        desc = (item.findtext("description") or "").strip()
        pub_date = _parse_pub_date(item.findtext("pubDate") or "")

        entries.append(
            {
                "title": title,
                "type": enforce_type,
                "link": link,
                "pub_date": pub_date,
                "description": desc[:300],
            }
        )
    return entries


# ── SEC EFTS parsing ─────────────────────────────────────────────────────────


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


# ── UK Gazette Atom parsing ──────────────────────────────────────────────────

_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _parse_gazette_atom(root: ET.Element) -> list[dict[str, Any]]:
    """Parse UK Gazette Atom feed."""
    entries: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", _ATOM_NS):
        title = (entry.findtext("atom:title", "", _ATOM_NS) or "").strip()
        link_el = entry.find("atom:link", _ATOM_NS)
        link = link_el.get("href", "") if link_el is not None else ""
        updated = (entry.findtext("atom:updated", "", _ATOM_NS) or "").strip()
        summary = (entry.findtext("atom:summary", "", _ATOM_NS) or "").strip()

        pub_date = _parse_pub_date(updated) if updated else ""

        entries.append(
            {
                "title": title,
                "source": "gazette",
                "link": link,
                "pub_date": pub_date,
                "description": summary[:300],
            }
        )
    return entries


# ── GOV.UK SFO parsing ──────────────────────────────────────────────────────


def _parse_govuk_results(data: dict) -> list[dict[str, Any]]:
    """Parse GOV.UK search JSON results."""
    entries: list[dict[str, Any]] = []
    for r in data.get("results", []):
        pub_ts = r.get("public_timestamp", "")
        pub_date = pub_ts[:16].replace("T", " ") if pub_ts else ""

        entries.append(
            {
                "title": (r.get("title") or "").strip(),
                "source": "sfo",
                "link": f"https://www.gov.uk{r.get('link', '')}",
                "pub_date": pub_date,
                "description": (r.get("description") or "").strip()[:300],
            }
        )
    return entries


# ── Tool class ───────────────────────────────────────────────────────────────


class BankruptcyCourtTool(Tool):
    name = "bankruptcy_court"
    description = (
        "Monitor bankruptcy filings, SEC enforcement, and UK insolvency notices. "
        "Mode 'us_bankruptcy' fetches PACER RSS from 6 major US courts. "
        "Mode 'sec_enforcement' fetches SEC admin proceedings + litigation releases. "
        "Mode 'sec_bankruptcy' searches SEC EFTS for 8-K Item 1.03 (bankruptcy entry). "
        "Mode 'uk_insolvency' fetches UK Gazette insolvency + SFO investigations. "
        "All free, no API key. PACER covers ~90% of large corporate Chapter 11."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": [
                    "us_bankruptcy",
                    "sec_enforcement",
                    "sec_bankruptcy",
                    "uk_insolvency",
                ],
                "description": (
                    "us_bankruptcy = PACER court filings. "
                    "sec_enforcement = SEC admin proceedings + litigation releases. "
                    "sec_bankruptcy = SEC 8-K Item 1.03 filings. "
                    "uk_insolvency = UK Gazette + SFO investigations."
                ),
            },
            "court": {
                "type": "string",
                "description": (
                    "PACER court filter for us_bankruptcy mode. "
                    "Options: sdny, del, sdtx, cdca, ndil, nj, all. "
                    "Default: all (fetches all 6 courts in parallel)."
                ),
                "default": "all",
            },
            "keyword": {
                "type": "string",
                "description": (
                    "Text filter. For sec_enforcement: filter titles. "
                    "For uk_insolvency: filter titles. Case-insensitive."
                ),
                "default": "",
            },
            "days_back": {
                "type": "integer",
                "description": ("Lookback window in days for sec_bankruptcy mode. Default 7, max 90."),
                "default": 7,
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return. Default 25, max 100.",
                "default": 25,
            },
        },
        "required": ["mode"],
    }

    def __init__(
        self,
        cache: DataCache | None = None,
        pipeline_store: PipelineStore | None = None,
    ) -> None:
        self._cache = cache
        self._store = pipeline_store

    def execute(
        self,
        *,
        mode: str = "us_bankruptcy",
        court: str = "all",
        keyword: str = "",
        days_back: int = 7,
        limit: int = 25,
        **_: Any,
    ) -> ToolResult:
        mode = mode.lower().strip()
        valid_modes = (
            "us_bankruptcy",
            "sec_enforcement",
            "sec_bankruptcy",
            "uk_insolvency",
        )
        if mode not in valid_modes:
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Use one of: {', '.join(valid_modes)}.",
            )

        keyword = keyword.strip()
        days_back = max(
            1,
            min(
                (
                    int(days_back)
                    if isinstance(days_back, (int, float, str)) and str(days_back).lstrip("-").isdigit()
                    else 7
                ),
                90,
            ),
        )
        limit = max(
            1,
            min(
                (int(limit) if isinstance(limit, (int, float, str)) and str(limit).lstrip("-").isdigit() else 25),
                100,
            ),
        )

        if mode == "us_bankruptcy":
            result = self._us_bankruptcy(court=court.strip().lower(), limit=limit)
        elif mode == "sec_enforcement":
            result = self._sec_enforcement(keyword=keyword, limit=limit)
        elif mode == "sec_bankruptcy":
            result = self._sec_bankruptcy(days_back=days_back, limit=limit)
        else:
            result = self._uk_insolvency(keyword=keyword, limit=limit)

        # L2: persist bankruptcy/enforcement observations on company entities
        if result.success and result.data:
            self._persist_entities(result.data, mode)

        return result

    # ── us_bankruptcy ─────────────────────────────────────────────────────

    def _us_bankruptcy(self, *, court: str, limit: int) -> ToolResult:
        """Fetch PACER RSS from selected courts."""
        if court == "all":
            courts_to_fetch = list(PACER_COURTS.keys())
        elif court in PACER_COURTS:
            courts_to_fetch = [court]
        else:
            valid = ", ".join(sorted(PACER_COURTS.keys()) + ["all"])
            return ToolResult(
                success=False,
                output=f"Invalid court '{court}'. Use one of: {valid}.",
            )

        # Check cache first
        cache_key = f"pacer_rss_{'_'.join(sorted(courts_to_fetch))}"
        if self._cache:
            cached = self._cache.get("bankruptcy_court", cache_key)
            if cached is not None:
                entries = cached[:limit]
                return self._format_pacer_result(entries, limit, from_cache=True)

        all_entries: list[dict[str, Any]] = []

        with httpx.Client(
            timeout=_TIMEOUT,
            headers={"User-Agent": _UA},
            follow_redirects=True,
        ) as client:
            if len(courts_to_fetch) == 1:
                all_entries = _fetch_pacer_court(courts_to_fetch[0], client)
            else:
                # Parallel fetch
                with ThreadPoolExecutor(max_workers=6) as pool:
                    futures = {pool.submit(_fetch_pacer_court, cc, client): cc for cc in courts_to_fetch}
                    for fut in as_completed(futures):
                        try:
                            all_entries.extend(fut.result())
                        except Exception as exc:
                            cc = futures[fut]
                            log.warning("PACER %s failed: %s", cc, exc)

        if self._cache:
            self._cache.set("bankruptcy_court", cache_key, all_entries, ttl=600)

        entries = all_entries[:limit]
        return self._format_pacer_result(entries, limit)

    def _format_pacer_result(self, entries: list[dict], limit: int, *, from_cache: bool = False) -> ToolResult:
        # Chapter breakdown
        ch_counts: dict[str, int] = {}
        court_counts: dict[str, int] = {}
        for e in entries:
            ch = e.get("chapter") or "unknown"
            ch_counts[ch] = ch_counts.get(ch, 0) + 1
            cc = e.get("court_name") or e.get("court", "?")
            court_counts[cc] = court_counts.get(cc, 0) + 1

        ch_str = ", ".join(f"Ch.{k}: {v}" for k, v in sorted(ch_counts.items()))
        court_str = ", ".join(f"{k}: {v}" for k, v in sorted(court_counts.items()))
        cache_tag = " (cached)" if from_cache else ""

        lines = [
            f"US Bankruptcy Court Filings{cache_tag}: {len(entries)} entries (limit {limit})",
            f"  Chapters: {ch_str or 'none detected'}",
            f"  Courts: {court_str or 'none'}",
            "",
        ]
        for e in entries[:15]:  # show max 15 in text
            ch_label = f" [Ch.{e['chapter']}]" if e.get("chapter") else ""
            lines.append(
                f"  [{e.get('court', '?')}] {e.get('case_number', '?')} "
                f"{e.get('debtor_name', '?')}{ch_label} — {e.get('pub_date', '')}"
            )

        if len(entries) > 15:
            lines.append(f"  ... and {len(entries) - 15} more")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "mode": "us_bankruptcy",
                "count": len(entries),
                "chapter_breakdown": ch_counts,
                "court_breakdown": court_counts,
                "entries": entries,
            },
        )

    # ── sec_enforcement ──────────────────────────────────────────────────

    def _sec_enforcement(self, *, keyword: str, limit: int) -> ToolResult:
        cache_key = f"sec_enforcement_{keyword}"
        if self._cache:
            cached = self._cache.get("bankruptcy_court", cache_key)
            if cached is not None:
                entries = cached[:limit]
                return self._format_sec_enforce_result(entries, limit, from_cache=True)

        all_entries: list[dict[str, Any]] = []
        with httpx.Client(
            timeout=_TIMEOUT,
            headers={"User-Agent": _UA},
            follow_redirects=True,
        ) as client:
            # Admin proceedings
            root = _fetch_xml(_SEC_ADMIN_RSS, client)
            if root is not None:
                all_entries.extend(_parse_sec_rss(root, "admin"))

            # Litigation releases
            root = _fetch_xml(_SEC_LIT_RSS, client)
            if root is not None:
                all_entries.extend(_parse_sec_rss(root, "litigation"))

        # Keyword filter
        if keyword:
            all_entries = [
                e for e in all_entries if _keyword_match(e.get("title", "") + " " + e.get("description", ""), keyword)
            ]

        # Sort by date descending (best effort — pubDate may vary in format)
        all_entries.sort(key=lambda e: e.get("pub_date", ""), reverse=True)

        if self._cache:
            self._cache.set("bankruptcy_court", cache_key, all_entries, ttl=1800)

        entries = all_entries[:limit]
        return self._format_sec_enforce_result(entries, limit)

    def _format_sec_enforce_result(self, entries: list[dict], limit: int, *, from_cache: bool = False) -> ToolResult:
        type_counts: dict[str, int] = {}
        for e in entries:
            t = e.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        type_str = ", ".join(f"{k}: {v}" for k, v in sorted(type_counts.items()))
        cache_tag = " (cached)" if from_cache else ""

        lines = [
            f"SEC Enforcement Actions{cache_tag}: {len(entries)} entries (limit {limit})",
            f"  Types: {type_str or 'none'}",
            "",
        ]
        for e in entries[:15]:
            lines.append(f"  [{e.get('type', '?')}] {e.get('title', '?')} — {e.get('pub_date', '')}")
        if len(entries) > 15:
            lines.append(f"  ... and {len(entries) - 15} more")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "mode": "sec_enforcement",
                "count": len(entries),
                "type_breakdown": type_counts,
                "entries": entries,
            },
        )

    # ── sec_bankruptcy ───────────────────────────────────────────────────

    def _sec_bankruptcy(self, *, days_back: int, limit: int) -> ToolResult:
        cache_key = f"sec_efts_bankruptcy_{days_back}"
        if self._cache:
            cached = self._cache.get("bankruptcy_court", cache_key)
            if cached is not None:
                entries = cached[:limit]
                return self._format_sec_bk_result(entries, limit, days_back, from_cache=True)

        now = datetime.now(UTC)
        start = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")

        with httpx.Client(
            timeout=_TIMEOUT,
            headers={"User-Agent": _UA},
            follow_redirects=True,
        ) as client:
            data = _fetch_json(
                _SEC_EFTS,
                client,
                q='"1.03"',
                forms="8-K",
                dateRange="custom",
                startdt=start,
                enddt=end,
                **{"from": "0"},
                size=str(min(limit, 100)),
                _source="form,file_date,display_names,items,ciks",
            )

        if data is None:
            return ToolResult(
                success=False,
                output="SEC EFTS unavailable (HTTP error or timeout).",
            )

        all_entries = _parse_efts_hits(data)
        total = data.get("hits", {}).get("total", {}).get("value", 0)

        if self._cache:
            self._cache.set("bankruptcy_court", cache_key, all_entries, ttl=3600)

        entries = all_entries[:limit]
        return self._format_sec_bk_result(entries, limit, days_back, total=total)

    def _format_sec_bk_result(
        self,
        entries: list[dict],
        limit: int,
        days_back: int,
        *,
        total: int = 0,
        from_cache: bool = False,
    ) -> ToolResult:
        cache_tag = " (cached)" if from_cache else ""
        lines = [
            f"SEC 8-K Bankruptcy Filings (Item 1.03){cache_tag}: {len(entries)} entries "
            f"(total {total}, last {days_back}d, limit {limit})",
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

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "mode": "sec_bankruptcy",
                "count": len(entries),
                "total": total,
                "days_back": days_back,
                "entries": entries,
            },
        )

    # ── uk_insolvency ────────────────────────────────────────────────────

    def _uk_insolvency(self, *, keyword: str, limit: int) -> ToolResult:
        cache_key = f"uk_insolvency_{keyword}"
        if self._cache:
            cached = self._cache.get("bankruptcy_court", cache_key)
            if cached is not None:
                entries = cached[:limit]
                return self._format_uk_result(entries, limit, from_cache=True)

        all_entries: list[dict[str, Any]] = []
        with httpx.Client(
            timeout=_TIMEOUT,
            headers={"User-Agent": _UA},
            follow_redirects=True,
        ) as client:
            # UK Gazette insolvency Atom feed (page 1 only — most recent)
            root = _fetch_xml(_UK_GAZETTE, client)
            if root is not None:
                all_entries.extend(_parse_gazette_atom(root))

            # GOV.UK SFO investigations
            data = _fetch_json(
                _GOV_UK_SEARCH,
                client,
                filter_organisations="serious-fraud-office",
                count=str(min(limit, 50)),
                order="-public_timestamp",
            )
            if data is not None:
                all_entries.extend(_parse_govuk_results(data))

        # Keyword filter
        if keyword:
            all_entries = [
                e for e in all_entries if _keyword_match(e.get("title", "") + " " + e.get("description", ""), keyword)
            ]

        # Sort by date descending
        all_entries.sort(key=lambda e: e.get("pub_date", ""), reverse=True)

        if self._cache:
            self._cache.set("bankruptcy_court", cache_key, all_entries, ttl=3600)

        entries = all_entries[:limit]
        return self._format_uk_result(entries, limit)

    def _format_uk_result(self, entries: list[dict], limit: int, *, from_cache: bool = False) -> ToolResult:
        source_counts: dict[str, int] = {}
        for e in entries:
            s = e.get("source", "unknown")
            source_counts[s] = source_counts.get(s, 0) + 1

        src_str = ", ".join(f"{k}: {v}" for k, v in sorted(source_counts.items()))
        cache_tag = " (cached)" if from_cache else ""

        lines = [
            f"UK Insolvency & Enforcement{cache_tag}: {len(entries)} entries (limit {limit})",
            f"  Sources: {src_str or 'none'}",
            "",
        ]
        for e in entries[:15]:
            lines.append(f"  [{e.get('source', '?')}] {e.get('title', '?')} — {e.get('pub_date', '')}")
        if len(entries) > 15:
            lines.append(f"  ... and {len(entries) - 15} more")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "mode": "uk_insolvency",
                "count": len(entries),
                "source_breakdown": source_counts,
                "entries": entries,
            },
        )

    # ── L2 entity persistence ────────────────────────────────────────────

    def _persist_entities(
        self,
        data: dict[str, Any],
        mode: str,
    ) -> dict[str, int]:
        """Persist bankruptcy/enforcement observations onto company entity nodes.

        Skips silently if no PipelineStore or entity module is available.
        """
        if self._store is None or _entity_id_from_key is None:
            return {"bankruptcy_status_obs": 0}
        try:
            return self._persist_entities_inner(data, mode)
        except Exception:
            log.exception("Bankruptcy court entity persistence failed (non-fatal)")
            return {"bankruptcy_status_obs": 0}

    def _persist_entities_inner(
        self,
        data: dict[str, Any],
        mode: str,
    ) -> dict[str, int]:
        """Inner persistence logic separated for testability."""
        assert self._store is not None  # noqa: S101 — guarded
        assert _entity_id_from_key is not None  # noqa: S101

        store = self._store
        counts = {"bankruptcy_status_obs": 0}
        now_ts = time.time()
        entries = data.get("entries", [])

        if mode == "us_bankruptcy":
            for entry in entries:
                name = (entry.get("debtor_name") or "").strip()
                if not name:
                    continue
                # Normalize: strip common legal prefixes
                clean = re.sub(r"^(In\s+re[:\s]*)", "", name, flags=re.IGNORECASE).strip()
                if not clean:
                    continue
                eid = _entity_id_from_key("company", clean)
                store.register_entity(
                    entity_type="company",
                    canonical_name=clean,
                    entity_id=eid,
                )
                store.store_entity_observation(
                    entity_id=eid,
                    source_tool="bankruptcy_court",
                    observed_at=now_ts,
                    observation_type="bankruptcy_status",
                    value={
                        "source": "pacer",
                        "chapter": entry.get("chapter"),
                        "court": entry.get("court"),
                        "case_number": entry.get("case_number"),
                        "pub_date": entry.get("pub_date"),
                    },
                    depth_level=2,
                )
                counts["bankruptcy_status_obs"] += 1

        elif mode == "sec_enforcement":
            for entry in entries:
                title = (entry.get("title") or "").strip()
                if not title:
                    continue
                eid = _entity_id_from_key("company", title)
                store.register_entity(
                    entity_type="company",
                    canonical_name=title,
                    entity_id=eid,
                )
                store.store_entity_observation(
                    entity_id=eid,
                    source_tool="bankruptcy_court",
                    observed_at=now_ts,
                    observation_type="bankruptcy_status",
                    value={
                        "source": "sec_enforcement",
                        "type": entry.get("type"),
                        "pub_date": entry.get("pub_date"),
                    },
                    depth_level=2,
                )
                counts["bankruptcy_status_obs"] += 1

        elif mode == "sec_bankruptcy":
            for entry in entries:
                name = (entry.get("company_name") or "").strip()
                if not name or name == "Unknown":
                    continue
                eid = _entity_id_from_key("company", name)
                store.register_entity(
                    entity_type="company",
                    canonical_name=name,
                    entity_id=eid,
                )
                store.store_entity_observation(
                    entity_id=eid,
                    source_tool="bankruptcy_court",
                    observed_at=now_ts,
                    observation_type="bankruptcy_status",
                    value={
                        "source": "sec_8k_103",
                        "cik": entry.get("cik"),
                        "file_date": entry.get("file_date"),
                        "items": entry.get("items", []),
                    },
                    depth_level=2,
                )
                counts["bankruptcy_status_obs"] += 1

        elif mode == "uk_insolvency":
            for entry in entries:
                title = (entry.get("title") or "").strip()
                if not title:
                    continue
                eid = _entity_id_from_key("company", title)
                store.register_entity(
                    entity_type="company",
                    canonical_name=title,
                    entity_id=eid,
                )
                store.store_entity_observation(
                    entity_id=eid,
                    source_tool="bankruptcy_court",
                    observed_at=now_ts,
                    observation_type="bankruptcy_status",
                    value={
                        "source": entry.get("source", "uk"),
                        "pub_date": entry.get("pub_date"),
                    },
                    depth_level=2,
                )
                counts["bankruptcy_status_obs"] += 1

        if counts["bankruptcy_status_obs"]:
            log.info(
                "Bankruptcy court L2: %d bankruptcy_status obs persisted",
                counts["bankruptcy_status_obs"],
            )
        return counts
