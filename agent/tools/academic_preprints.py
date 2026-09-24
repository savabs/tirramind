"""
Tool: Academic Preprints — arXiv Papers + ClinicalTrials.gov Studies

Two free, no-auth APIs for tracking the frontier of scientific discovery:

  arXiv (export.arxiv.org):
    Preprint papers across physics, CS, math, biology, finance.
    Atom XML API, search by category/keyword/author.
    Leading indicator: new papers signal paradigm shifts before commercialization.

  ClinicalTrials.gov (clinicaltrials.gov/api/v2):
    Active, completed, and recruiting clinical trials for drugs/devices/therapies.
    JSON API, search by condition, intervention, sponsor.
    Leading indicator: Phase III completions → FDA filing → pharma stock moves.

Signal theory:
  - Surge in papers on a topic (e.g., "quantum error correction") = paradigm shift
  - New clinical trial registrations by a company = pipeline expansion
  - Phase III completion + "Completed" status = FDA submission imminent
  - Unusual arXiv activity from corporate labs (Google, Meta) = product launch signal
  - Drug + condition matching = competitive landscape for pharma valuations
  - Paper retraction or trial termination = negative signal for related companies

Modes:
  papers   — Search arXiv preprints by keyword/category.
  trials   — Search ClinicalTrials.gov by condition/intervention/sponsor.
  trending — Recent arXiv papers in market-relevant categories (cs.AI, q-fin, cs.CR).
"""

from __future__ import annotations

import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
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

_ARXIV_URL = "https://export.arxiv.org/api/query"
_CT_URL = "https://clinicaltrials.gov/api/v2/studies"
_UA = "TirraMind/0.1"
_TIMEOUT = 20

VALID_MODES = {"papers", "trials", "trending"}

# arXiv throttles hard (HTTP 429 after ~10 requests in a few minutes) and its
# edge occasionally 5xxs.  Both are transient, and a zero-row day on a
# time-gated source is unrecoverable, so retry before giving up.  arXiv's own
# terms of use ask for >=3s between requests; these delays honour that.
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_RETRY_DELAYS = (5.0, 15.0, 45.0)
_RETRY_AFTER_CAP = 60.0

_ATOM_FEED_TAG = "{http://www.w3.org/2005/Atom}feed"

# arXiv categories with market relevance
_MARKET_CATEGORIES = [
    "q-fin",  # quantitative finance
    "cs.AI",  # artificial intelligence
    "cs.LG",  # machine learning
    "cs.CR",  # cryptography & security
    "cs.CL",  # computation & language (NLP/LLM)
    "econ",  # economics
    "stat.ML",  # machine learning (stats)
    "physics.soc-ph",  # social physics / complex systems
]

# Atom XML namespaces
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}


class ArxivFeedError(RuntimeError):
    """arXiv answered, but the body is not a feed we can trust.

    Raised instead of returning an empty result set, so that a 200 carrying an
    empty body, an HTML error page, or fewer entries than arXiv itself says it
    published can never be reported as a successful collection (LESSONS.md
    F-16: "green is not evidence, row counts are").
    """


def _epoch_from_iso(value: Any) -> float | None:
    """Parse an ISO-8601 date or timestamp into epoch seconds, or None.

    Accepts arXiv's ``2024-01-15T12:00:00Z`` and bare ``2024-01-15``.  A naive
    value is read as UTC.  Returns ``None`` — never "now" — when the input is
    missing or unparseable: a row we cannot timestamp is a row we must not
    invent a timestamp for (see :meth:`_persist_entities_inner`).
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def _utc_day_floor(ts: float) -> float:
    """Floor an epoch timestamp to 00:00:00 UTC of the same day."""
    return datetime.fromtimestamp(ts, tz=UTC).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


class AcademicPreprintsTool(Tool):
    """Search arXiv preprints and ClinicalTrials.gov studies."""

    def __init__(
        self,
        cache: DataCache | None = None,
        pipeline_store: PipelineStore | None = None,
    ) -> None:
        self._cache = cache
        self._store = pipeline_store

    @property
    def name(self) -> str:
        return "academic_preprints"

    @property
    def description(self) -> str:
        return (
            "Search academic preprints (arXiv) and clinical trials (ClinicalTrials.gov). "
            "Find cutting-edge research papers and drug trial status as leading indicators "
            "for technology and pharma sectors."
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
                        "Query mode: papers (arXiv keyword search), trials (ClinicalTrials.gov), "
                        "trending (recent arXiv in market-relevant categories)."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": "Search keyword(s). Required for papers and trials modes.",
                },
                "category": {
                    "type": "string",
                    "description": "arXiv category (e.g. 'cs.AI', 'q-fin'). For papers/trending.",
                },
                "sponsor": {
                    "type": "string",
                    "description": "Trial sponsor name (e.g. 'Pfizer'). For trials mode.",
                },
                "status": {
                    "type": "string",
                    "description": "Trial status filter: RECRUITING, COMPLETED, ACTIVE_NOT_RECRUITING, etc.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 15, max 50).",
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

        limit = min(max(int(kwargs.get("limit", 15)), 1), 50)

        try:
            if mode == "papers":
                query = (kwargs.get("query") or "").strip()
                if not query:
                    return ToolResult(
                        success=False,
                        output="Papers mode requires a 'query' parameter.",
                    )
                category = (kwargs.get("category") or "").strip()
                result = self._arxiv_search(query, category, limit)
            elif mode == "trending":
                category = (kwargs.get("category") or "").strip()
                result = self._arxiv_trending(category, limit)
            elif mode == "trials":
                result = self._clinical_trials(kwargs, limit)
            else:
                return ToolResult(success=False, output=f"Unhandled mode: {mode}")
        except ArxivFeedError as exc:
            # A 200 that carried no usable feed. Loud, never success.
            return ToolResult(success=False, output=str(exc))
        except httpx.TimeoutException:
            return ToolResult(success=False, output="API timed out.")
        except httpx.HTTPError as exc:
            return ToolResult(success=False, output=f"API error: {exc}")
        except Exception as exc:
            log.exception("AcademicPreprintsTool error")
            return ToolResult(success=False, output=f"Unexpected error: {exc}")

        # L2: persist research_velocity observations on entity nodes.
        # The row count is consumed, not discarded: a store failure partway
        # through used to leave rows written, reset the count to 0, and still
        # return success=True — a partial write reported green.
        if result.success and result.data:
            counts = self._persist_entities(result.data, mode)
            written = counts.get("research_velocity_obs", 0)
            failed = counts.get("failed", 0)
            skipped = counts.get("skipped", 0)
            attempted = counts.get("attempted", 0)
            if isinstance(result.data, dict):
                result.data["rows_written"] = written
                result.data["rows_failed"] = failed
                result.data["rows_skipped"] = skipped
            if failed:
                return ToolResult(
                    success=False,
                    output=(
                        f"{result.output} PARTIAL WRITE: {written} observation(s) stored, {failed} failed to persist."
                    ),
                    data=result.data,
                )
            # Every item skipped is a zero-row collection wearing a 200.  A
            # skip is individually legitimate (an arXiv entry with no
            # category, a trial with no lead sponsor), but *all* of them
            # skipped means the field we key on stopped arriving — a parse or
            # schema break — and the run stored nothing.  F-16: the source
            # handed us `attempted` items and we wrote none; that is never
            # green.
            if attempted and written == 0:
                return ToolResult(
                    success=False,
                    output=(
                        f"{result.output} ZERO ROWS: {attempted} item(s) returned, "
                        f"0 observations stored ({skipped} skipped for a missing "
                        f"{'sponsor' if mode == 'trials' else 'category'}). "
                        "Source published rows; none were persisted."
                    ),
                    data=result.data,
                )

        return result

    def _arxiv_search(self, query: str, category: str, limit: int) -> ToolResult:
        search_query = f"all:{query}"
        if category:
            # A literal "+" here is percent-encoded to %2B by the URL encoder,
            # so arXiv saw a plus sign instead of a boolean AND. Space + AND.
            search_query = f"cat:{category} AND all:{query}"

        params = {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": str(limit),
        }

        # A keyword search may legitimately match nothing, so an honestly
        # empty feed (total=0, 0 entries) is allowed here — but a feed that
        # claims results and ships none is not.
        papers, total = self._fetch_arxiv_feed(params, limit, require_results=False)

        summary = (
            f"Found {total} arXiv papers matching '{query}'"
            f"{f' in {category}' if category else ''}"
            f". Showing {len(papers)}."
        )
        return ToolResult(
            success=True,
            output=summary,
            data={
                "papers": papers,
                "total": total,
                "count": len(papers),
                "source": "arxiv",
            },
        )

    def _arxiv_trending(self, category: str, limit: int) -> ToolResult:
        # If no category, search across all market-relevant categories
        if category:
            search_query = f"cat:{category}"
        else:
            cat_terms = " OR ".join(f"cat:{c}" for c in _MARKET_CATEGORIES)
            search_query = cat_terms

        params = {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": str(limit),
        }

        # A standing category query over live arXiv categories is never
        # empty, so "0 results" here means the fetch failed, not that research
        # stopped.
        papers, total = self._fetch_arxiv_feed(params, limit, require_results=True)

        cats = category or ", ".join(_MARKET_CATEGORIES[:4]) + "..."
        summary = f"Trending arXiv preprints in [{cats}]. {total} total, showing {len(papers)}."
        return ToolResult(
            success=True,
            output=summary,
            data={
                "papers": papers,
                "total": total,
                "count": len(papers),
                "source": "arxiv",
            },
        )

    def _clinical_trials(self, kwargs: dict, limit: int) -> ToolResult:
        query = (kwargs.get("query") or "").strip()
        sponsor = (kwargs.get("sponsor") or "").strip()
        status = (kwargs.get("status") or "").strip()

        if not query and not sponsor:
            return ToolResult(
                success=False,
                output="Trials mode requires either 'query' (condition/intervention) or 'sponsor'.",
            )

        params: dict[str, str] = {"pageSize": str(limit)}
        if query:
            params["query.cond"] = query
        if sponsor:
            params["query.spons"] = sponsor
        if status:
            params["filter.overallStatus"] = status

        data = self._fetch_json(_CT_URL, params)
        if data is None:
            return ToolResult(success=False, output="Failed to fetch clinical trials data.")

        studies = data.get("studies", [])
        results = []
        for s in studies:
            proto = s.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            cond_mod = proto.get("conditionsModule", {})
            arms_mod = proto.get("armsInterventionsModule", {})
            sponsor_mod = proto.get("sponsorCollaboratorsModule", {})

            interventions = arms_mod.get("interventions", [])
            lead_sponsor = sponsor_mod.get("leadSponsor", {})

            results.append(
                {
                    "nct_id": ident.get("nctId"),
                    "title": ident.get("briefTitle"),
                    "status": status_mod.get("overallStatus"),
                    "start_date": (status_mod.get("startDateStruct") or {}).get("date"),
                    "completion_date": (status_mod.get("completionDateStruct") or {}).get("date"),
                    "conditions": cond_mod.get("conditions", []),
                    "interventions": [i.get("name") for i in interventions[:5]],
                    "sponsor": lead_sponsor.get("name"),
                    "sponsor_class": lead_sponsor.get("class"),
                }
            )

        total = data.get("totalCount", len(results))
        summary = (
            f"Found {total} clinical trials"
            + (f' for "{query}"' if query else "")
            + (f" sponsored by {sponsor}" if sponsor else "")
            + (f" with status {status}" if status else "")
            + f". Showing {len(results)}."
        )
        return ToolResult(
            success=True,
            output=summary,
            data={
                "trials": results,
                "total": total,
                "count": len(results),
                "source": "clinicaltrials",
            },
        )

    def _fetch_arxiv_feed(
        self,
        params: dict,
        limit: int,
        *,
        require_results: bool,
    ) -> tuple[list[dict], int]:
        """Fetch an arXiv feed and refuse to return a short one quietly.

        F-16 guard.  arXiv publishes its own row count in
        ``opensearch:totalResults``; this compares that number against how many
        ``<entry>`` elements actually arrived and raises
        :class:`ArxivFeedError` on any shortfall.  Four shapes that all used to
        return ``success=True`` with zero rows are now failures:

        * an empty or non-XML body (``ET.fromstring`` fails);
        * a well-formed non-feed document — an HTML error page parses fine as
          XML, so the root tag is checked explicitly;
        * ``totalResults > 0`` with zero entries ("535658 total, showing 0");
        * fewer entries than ``min(max_results, totalResults)``, i.e. a
          truncated page.

        ``require_results`` additionally rejects an honestly-empty feed, for
        queries (trending) whose result set is never legitimately empty.
        """
        xml_text = self._fetch_text(_ARXIV_URL, params)
        if xml_text is None:
            raise ArxivFeedError("Failed to fetch arXiv data.")

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise ArxivFeedError(
                f"arXiv returned {len(xml_text)} bytes that do not parse as XML: {xml_text[:200]!r}"
            ) from exc

        if root.tag != _ATOM_FEED_TAG:
            raise ArxivFeedError(f"arXiv returned a <{root.tag}> document, not an Atom feed: {xml_text[:200]!r}")

        papers = self._parse_arxiv_xml(xml_text)
        total = self._parse_arxiv_total(xml_text)

        if total > 0 and not papers:
            raise ArxivFeedError(
                f"arXiv reported {total} matching results and returned 0 entries. "
                "Source says it published rows; we parsed none."
            )

        expected = min(limit, total)
        if len(papers) < expected:
            raise ArxivFeedError(
                f"arXiv returned {len(papers)} entries for max_results={limit} "
                f"while reporting {total} total; expected {expected}. Truncated feed."
            )

        if require_results and not papers:
            raise ArxivFeedError("arXiv reported 0 results for a standing category query that is never empty.")

        return papers, total

    def _parse_arxiv_xml(self, xml_text: str) -> list[dict]:
        """Parse arXiv Atom XML into a list of paper dicts."""
        papers = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return papers

        for entry in root.findall("atom:entry", _NS):
            title_el = entry.find("atom:title", _NS)
            summary_el = entry.find("atom:summary", _NS)
            published_el = entry.find("atom:published", _NS)
            updated_el = entry.find("atom:updated", _NS)
            id_el = entry.find("atom:id", _NS)

            # Authors
            authors = []
            for author_el in entry.findall("atom:author", _NS):
                name_el = author_el.find("atom:name", _NS)
                if name_el is not None and name_el.text:
                    authors.append(name_el.text.strip())

            # Categories
            categories = []
            for cat_el in entry.findall("atom:category", _NS):
                term = cat_el.get("term")
                if term:
                    categories.append(term)

            # PDF link
            pdf_link = None
            for link_el in entry.findall("atom:link", _NS):
                if link_el.get("title") == "pdf":
                    pdf_link = link_el.get("href")

            papers.append(
                {
                    "id": (id_el.text.strip() if id_el is not None and id_el.text else None),
                    "title": (" ".join((title_el.text or "").split()) if title_el is not None else None),
                    "summary": (" ".join((summary_el.text or "").split())[:300] if summary_el is not None else None),
                    "authors": authors[:5],  # Cap at 5
                    "categories": categories,
                    "published": (
                        published_el.text.strip() if published_el is not None and published_el.text else None
                    ),
                    "updated": (updated_el.text.strip() if updated_el is not None and updated_el.text else None),
                    "pdf_url": pdf_link,
                }
            )

        return papers

    def _parse_arxiv_total(self, xml_text: str) -> int:
        """Extract total results count from arXiv response."""
        try:
            root = ET.fromstring(xml_text)
            total_el = root.find("opensearch:totalResults", _NS)
            if total_el is not None and total_el.text:
                return int(total_el.text)
        except (ET.ParseError, ValueError):
            pass
        return 0

    def _fetch_text(self, url: str, params: dict) -> str | None:
        """Fetch text response (for arXiv XML) over the stdlib HTTP transport.

        **Do not move this back to httpx.**  arXiv sits behind a Fastly edge
        that answers ``406`` with an empty body to *any* HTTPS request made by
        httpx/httpcore, and only to httpx.  Verified on 2026-09-23 with
        interleaved, cache-busted trials against the identical URL and headers:
        httpx 0/6 succeeded, curl 6/6, urllib 5/6, requests 1/1.  The
        discriminator is httpx's HTTPS transport itself — not the query, not
        the User-Agent, not Accept/Accept-Encoding/Connection, not header
        casing, not percent-encoding, not rate limiting, not the HTTP version,
        not the address family.  Every one of those was ruled out by
        experiment.  This block is why collection silently produced nothing
        from 2026-08-27 to 2026-09-23; arXiv is time-gated, so those 27 days
        are unrecoverable.

        Errors are re-raised as httpx exception types so that the funnel in
        :meth:`execute` keeps classifying them exactly as before (timeout vs.
        HTTP error) instead of dropping through to the generic handler.

        Transient answers (429 and 5xx) are retried with backoff before that
        happens.  arXiv throttles a burst of ~10 requests, and a throttled run
        on a time-gated source is a permanently lost day, not a retry-tomorrow
        inconvenience.  Exhausting the retries still fails — it never returns
        a partial or empty body as if it were data.
        """
        full_url = f"{url}?{urllib.parse.urlencode(params)}"
        last_status: int | None = None
        last_reason: str | None = None

        for attempt in range(len(_RETRY_DELAYS) + 1):
            req = urllib.request.Request(full_url, headers={"User-Agent": _UA})  # noqa: S310 — https literal
            try:
                with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310 — https literal
                    return resp.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as exc:
                last_status, last_reason = exc.code, f"HTTP {exc.code}"
                if exc.code in _RETRY_STATUSES and attempt < len(_RETRY_DELAYS):
                    delay = self._retry_delay(exc, attempt)
                    log.warning("arXiv %s, retrying in %.0fs (attempt %d)", exc.code, delay, attempt + 1)
                    time.sleep(delay)
                    continue
                httpx_req = httpx.Request("GET", full_url)
                raise httpx.HTTPStatusError(
                    f"arXiv HTTP {exc.code}",
                    request=httpx_req,
                    response=httpx.Response(exc.code, request=httpx_req),
                ) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_reason = str(exc)
                if attempt < len(_RETRY_DELAYS):
                    delay = _RETRY_DELAYS[attempt]
                    log.warning("arXiv unreachable (%s), retrying in %.0fs", exc, delay)
                    time.sleep(delay)
                    continue
                raise httpx.TimeoutException(f"arXiv unreachable: {exc}") from exc

        # Unreachable: the final attempt either returns or raises above.
        raise httpx.TimeoutException(f"arXiv unreachable after retries: {last_reason} ({last_status})")

    @staticmethod
    def _retry_delay(exc: urllib.error.HTTPError, attempt: int) -> float:
        """Backoff for a retryable status, honouring ``Retry-After`` when sane."""
        raw = exc.headers.get("Retry-After") if exc.headers else None
        if raw:
            try:
                return max(0.0, min(float(raw), _RETRY_AFTER_CAP))
            except (TypeError, ValueError):
                pass
        return _RETRY_DELAYS[attempt]

    def _fetch_json(self, url: str, params: dict) -> Any:
        """Fetch JSON response (for ClinicalTrials.gov)."""
        if self._cache:
            import json

            cache_key = f"{url}?{json.dumps(params, sort_keys=True)}"
            cached = self._cache.get("academic_preprints", {"key": cache_key})
            if cached is not None:
                return cached

        with httpx.Client(
            timeout=_TIMEOUT,
            headers={"User-Agent": _UA},
        ) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        if self._cache and data is not None:
            self._cache.put("academic_preprints", {"key": cache_key}, data)
        return data

    # ------------------------------------------------------------------
    # L2 entity persistence
    # ------------------------------------------------------------------

    def _persist_entities(
        self,
        data: dict[str, Any],
        mode: str,
    ) -> dict[str, int]:
        """Persist research_velocity observations onto entity nodes.

        trials → company entities (by sponsor).
        papers / trending → topic entities (by arXiv category).

        Returns ``{"research_velocity_obs": written, "failed": n,
        "skipped": n, "attempted": n}``.  A failure is *counted*, never
        collapsed into a zero: :meth:`execute` turns any non-zero ``failed``
        into ``success=False``, because a run that wrote 7 of 15 rows is a
        partial write, not a success.  ``attempted`` is how many items the
        fetch actually handed us, so ``attempted > 0 and written == 0`` — every
        item skipped — is likewise reported as a failure rather than as a
        green run that stored nothing.

        Returns all-zero (no failures) when there is no PipelineStore or
        entity module — that is "persistence not configured", not data loss.
        """
        if self._store is None or _entity_id_from_key is None:
            # Persistence not configured. `attempted` stays 0 so this reads as
            # "nothing to write", not as a zero-row collection.
            return {"research_velocity_obs": 0, "failed": 0, "skipped": 0, "attempted": 0}
        try:
            return self._persist_entities_inner(data, mode)
        except Exception:
            log.exception("Academic preprints entity persistence failed")
            items = data.get("trials" if mode == "trials" else "papers") or []
            # Whatever blew up, it was not "nothing to do": report at least one
            # failure so the caller cannot read this as a clean empty run.
            return {
                "research_velocity_obs": 0,
                "failed": max(len(items), 1),
                "skipped": 0,
                "attempted": len(items),
            }

    def _persist_entities_inner(
        self,
        data: dict[str, Any],
        mode: str,
    ) -> dict[str, int]:
        """Inner persistence logic separated for testability."""
        assert self._store is not None  # noqa: S101 — guarded
        assert _entity_id_from_key is not None  # noqa: S101

        store = self._store
        counts: dict[str, int] = {"research_velocity_obs": 0, "failed": 0, "skipped": 0, "attempted": 0}

        if mode == "trials":
            # ClinicalTrials.gov reports a trial's *current* registry status, so
            # the observation is "on this UTC day, the registry said X".
            # Flooring to the UTC day keeps OBSERVATION_UNIQUE_KEY stable across
            # re-runs within a day, so a repeated collection collapses instead
            # of duplicating every row (which is what time.time() did).  The
            # trial's own start_date is deliberately not used: registries carry
            # future start dates, which the store's [1990-01-01, now+1d] range
            # guard rejects outright.
            observed_at = _utc_day_floor(time.time())
            for trial in data.get("trials", []):
                counts["attempted"] += 1
                sponsor = (trial.get("sponsor") or "").strip()
                if not sponsor:
                    counts["skipped"] += 1
                    continue
                try:
                    eid = _entity_id_from_key("company", sponsor)
                    store.register_entity(
                        entity_type="company",
                        canonical_name=sponsor,
                        entity_id=eid,
                    )
                    store.store_entity_observation(
                        entity_id=eid,
                        source_tool="academic_preprints",
                        observed_at=observed_at,
                        observation_type="research_velocity",
                        value={
                            "source": "clinicaltrials",
                            "nct_id": trial.get("nct_id"),
                            "title": trial.get("title"),
                            "status": trial.get("status"),
                            "conditions": trial.get("conditions", []),
                        },
                        depth_level=2,
                    )
                except Exception:
                    log.exception("Failed to persist trial %s", trial.get("nct_id"))
                    counts["failed"] += 1
                else:
                    counts["research_velocity_obs"] += 1

        elif mode in ("papers", "trending"):
            for paper in data.get("papers", []):
                counts["attempted"] += 1
                categories = paper.get("categories") or []
                if not categories:
                    counts["skipped"] += 1
                    continue
                # observed_at is the paper's own publication time, not the
                # fetch wall-clock.  time.time() varied per run, and since
                # observed_at is part of OBSERVATION_UNIQUE_KEY every re-run
                # duplicated every row (15 papers -> 30 rows on the second
                # run).  A paper we cannot timestamp is reported as a failure,
                # not defaulted to "now".
                observed_at = _epoch_from_iso(paper.get("published"))
                if observed_at is None:
                    log.error(
                        "arXiv entry %s has no parseable 'published' timestamp; not storing",
                        paper.get("id"),
                    )
                    counts["failed"] += 1
                    continue
                cat = categories[0]
                try:
                    eid = _entity_id_from_key("topic", cat)
                    store.register_entity(
                        entity_type="topic",
                        canonical_name=cat,
                        entity_id=eid,
                    )
                    store.store_entity_observation(
                        entity_id=eid,
                        source_tool="academic_preprints",
                        observed_at=observed_at,
                        observation_type="research_velocity",
                        value={
                            "source": "arxiv",
                            "paper_id": paper.get("id"),
                            "title": paper.get("title"),
                            "published": paper.get("published"),
                        },
                        depth_level=2,
                    )
                except Exception:
                    log.exception("Failed to persist arXiv entry %s", paper.get("id"))
                    counts["failed"] += 1
                else:
                    counts["research_velocity_obs"] += 1

        return counts
