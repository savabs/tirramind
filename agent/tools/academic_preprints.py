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
        except httpx.TimeoutException:
            return ToolResult(success=False, output="API timed out.")
        except httpx.HTTPError as exc:
            return ToolResult(success=False, output=f"API error: {exc}")
        except Exception as exc:
            log.exception("AcademicPreprintsTool error")
            return ToolResult(success=False, output=f"Unexpected error: {exc}")

        # L2: persist research_velocity observations on entity nodes
        if result.success and result.data:
            self._persist_entities(result.data, mode)

        return result

    def _arxiv_search(self, query: str, category: str, limit: int) -> ToolResult:
        search_query = f"all:{query}"
        if category:
            search_query = f"cat:{category}+AND+all:{query}"

        params = {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": str(limit),
        }

        xml_text = self._fetch_text(_ARXIV_URL, params)
        if xml_text is None:
            return ToolResult(success=False, output="Failed to fetch arXiv data.")

        papers = self._parse_arxiv_xml(xml_text)
        total = self._parse_arxiv_total(xml_text)

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

        xml_text = self._fetch_text(_ARXIV_URL, params)
        if xml_text is None:
            return ToolResult(success=False, output="Failed to fetch arXiv data.")

        papers = self._parse_arxiv_xml(xml_text)
        total = self._parse_arxiv_total(xml_text)

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
        """Fetch text response (for arXiv XML)."""
        with httpx.Client(
            timeout=_TIMEOUT,
            headers={"User-Agent": _UA},
            follow_redirects=True,
        ) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            return resp.text

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

        Skips silently if no PipelineStore or entity module is available.
        """
        if self._store is None or _entity_id_from_key is None:
            return {"research_velocity_obs": 0}
        try:
            return self._persist_entities_inner(data, mode)
        except Exception:
            log.exception("Academic preprints entity persistence failed (non-fatal)")
            return {"research_velocity_obs": 0}

    def _persist_entities_inner(
        self,
        data: dict[str, Any],
        mode: str,
    ) -> dict[str, int]:
        """Inner persistence logic separated for testability."""
        assert self._store is not None  # noqa: S101 — guarded
        assert _entity_id_from_key is not None  # noqa: S101

        store = self._store
        counts: dict[str, int] = {"research_velocity_obs": 0}
        now_ts = time.time()

        if mode == "trials":
            for trial in data.get("trials", []):
                sponsor = (trial.get("sponsor") or "").strip()
                if not sponsor:
                    continue
                eid = _entity_id_from_key("company", sponsor)
                store.register_entity(
                    entity_type="company",
                    canonical_name=sponsor,
                    entity_id=eid,
                )
                store.store_entity_observation(
                    entity_id=eid,
                    source_tool="academic_preprints",
                    observed_at=now_ts,
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
                counts["research_velocity_obs"] += 1

        elif mode in ("papers", "trending"):
            for paper in data.get("papers", []):
                categories = paper.get("categories") or []
                if not categories:
                    continue
                cat = categories[0]
                eid = _entity_id_from_key("topic", cat)
                store.register_entity(
                    entity_type="topic",
                    canonical_name=cat,
                    entity_id=eid,
                )
                store.store_entity_observation(
                    entity_id=eid,
                    source_tool="academic_preprints",
                    observed_at=now_ts,
                    observation_type="research_velocity",
                    value={
                        "source": "arxiv",
                        "paper_id": paper.get("id"),
                        "title": paper.get("title"),
                        "published": paper.get("published"),
                    },
                    depth_level=2,
                )
                counts["research_velocity_obs"] += 1

        return counts
