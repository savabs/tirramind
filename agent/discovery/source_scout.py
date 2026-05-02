"""TirraMind — Data Source Scout (Change 15)

Discovers candidate data sources from API catalogs and probes them for
parseable content.  Pure discovery layer — no signal evaluation or tool
creation here; those are handled by SignalEvaluator and ToolFactory.

No LLM in the hot path.  Discovery uses keyword search + structured catalog
APIs, which are free and deterministic.

Reference: Spec steps 15.2 + 15.7 in [[tier8_autonomous_discovery_spec]].
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.learning.tool_router import ToolRoutingBandit
    from agent.pipeline.store import PipelineStore
    from agent.tools.base import ToolRegistry

log = logging.getLogger(__name__)

# Maximum response size for probes (1 MB)
_MAX_PROBE_BYTES = 1_048_576

# Default catalog endpoints
DEFAULT_CATALOG_URLS: list[str] = [
    "https://catalog.data.gov/api/3/action/package_search",
]


@dataclass
class DataSourceCandidate:
    """A candidate data source discovered from a catalog."""

    source_id: str  # SHA-256[:16] of url
    name: str
    url: str
    description: str
    format: str  # 'json_api' | 'csv_feed'
    update_frequency: str
    topic_tags: list[str] = field(default_factory=list)
    probe_sample: dict | list | None = None
    relevance_score: float = 0.0


def _make_source_id(url: str) -> str:
    """Deterministic source ID from URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _tfidf_relevance(text: str, vocabulary: set[str]) -> float:
    """Simple TF-IDF-like relevance score: fraction of vocabulary terms present."""
    if not vocabulary or not text:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for term in vocabulary if term in text_lower)
    return hits / len(vocabulary)


class SourceScout:
    """Discovers candidate data sources from API catalogs.

    Parameters
    ----------
    catalog_urls : list[str]
        API catalog endpoints to search (CKAN-compatible).
    existing_source_urls : set[str]
        URLs already known (to skip duplicates).
    topic_vocabulary : set[str]
        Terms used for relevance scoring (entity types, feature name prefixes).
    """

    def __init__(
        self,
        catalog_urls: list[str] | None = None,
        existing_source_urls: set[str] | None = None,
        topic_vocabulary: set[str] | None = None,
    ) -> None:
        self._catalog_urls = catalog_urls or DEFAULT_CATALOG_URLS
        self._known_urls = existing_source_urls or set()
        self._vocabulary = topic_vocabulary or set()

    def search(
        self,
        query_terms: list[str],
        *,
        max_results: int = 20,
    ) -> list[DataSourceCandidate]:
        """Search catalogs for datasets matching *query_terms*.

        Returns candidates sorted by relevance (highest first).
        """
        candidates: list[DataSourceCandidate] = []
        query = " ".join(query_terms)

        for catalog_url in self._catalog_urls:
            try:
                results = self._search_ckan(catalog_url, query, max_results)
                candidates.extend(results)
            except Exception:
                log.warning("Catalog search failed for %s", catalog_url, exc_info=True)

        # Deduplicate by URL
        seen: set[str] = set()
        unique: list[DataSourceCandidate] = []
        for c in candidates:
            if c.url in seen or c.url in self._known_urls:
                continue
            seen.add(c.url)
            unique.append(c)

        # Score relevance
        for c in unique:
            c.relevance_score = _tfidf_relevance(
                f"{c.name} {c.description} {' '.join(c.topic_tags)}",
                self._vocabulary,
            )

        unique.sort(key=lambda c: c.relevance_score, reverse=True)
        return unique[:max_results]

    def _search_ckan(
        self,
        catalog_url: str,
        query: str,
        max_results: int,
    ) -> list[DataSourceCandidate]:
        """Search a CKAN-compatible catalog API."""
        import urllib.parse

        params = urllib.parse.urlencode({"q": query, "rows": max_results})
        url = f"{catalog_url}?{params}"

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "TirraMind/1.0 (research@tirramind.com)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                data = json.loads(resp.read(_MAX_PROBE_BYTES).decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
            log.warning("CKAN search failed: %s", url)
            return []

        if not data.get("success"):
            return []

        candidates: list[DataSourceCandidate] = []
        for pkg in data.get("result", {}).get("results", []):
            resources = pkg.get("resources", [])
            # Pick the best resource (prefer JSON/CSV APIs)
            best = self._pick_best_resource(resources)
            if best is None:
                continue

            fmt = self._classify_format(best.get("format", ""), best.get("url", ""))
            if fmt is None:
                continue

            resource_url = best.get("url", "")
            if not resource_url:
                continue

            candidates.append(
                DataSourceCandidate(
                    source_id=_make_source_id(resource_url),
                    name=pkg.get("title", pkg.get("name", "unknown")),
                    url=resource_url,
                    description=pkg.get("notes", "")[:500],
                    format=fmt,
                    update_frequency=self._extract_frequency(pkg),
                    topic_tags=[t.get("name", "") for t in pkg.get("tags", []) if t.get("name")],
                )
            )
        return candidates

    @staticmethod
    def _pick_best_resource(resources: list[dict]) -> dict | None:
        """Pick the most usable resource from a CKAN package."""
        priority = {"api": 0, "json": 1, "csv": 2, "tsv": 3}
        scored = []
        for r in resources:
            fmt = (r.get("format") or "").lower()
            score = priority.get(fmt, 99)
            scored.append((score, r))
        scored.sort(key=lambda x: x[0])
        return scored[0][1] if scored else None

    @staticmethod
    def _classify_format(fmt: str, url: str) -> str | None:
        """Classify resource format into our supported types."""
        fmt = fmt.lower()
        if fmt in ("json", "api", "geojson"):
            return "json_api"
        if fmt in ("csv", "tsv", "txt"):
            return "csv_feed"
        # Heuristic from URL
        if url.endswith(".json"):
            return "json_api"
        if url.endswith((".csv", ".tsv")):
            return "csv_feed"
        return None

    @staticmethod
    def _extract_frequency(pkg: dict) -> str:
        """Extract update frequency from CKAN metadata."""
        freq = pkg.get("update_frequency", "") or pkg.get("frequency", "")
        if freq:
            return freq.lower()
        # Check extras
        for extra in pkg.get("extras", []):
            if extra.get("key", "").lower() in ("frequency", "update_frequency"):
                return str(extra.get("value", "unknown")).lower()
        return "unknown"

    def probe(self, candidate: DataSourceCandidate) -> DataSourceCandidate:
        """Fetch a sample from the candidate's URL and parse it.

        Mutates ``candidate.probe_sample`` in place and returns it.
        On failure, ``probe_sample`` is set to *None*.
        """
        try:
            req = urllib.request.Request(
                candidate.url,
                headers={"User-Agent": "TirraMind/1.0 (research@tirramind.com)"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                raw = resp.read(_MAX_PROBE_BYTES)
        except Exception:
            log.warning("Probe failed for %s: %s", candidate.name, candidate.url)
            candidate.probe_sample = None
            return candidate

        try:
            if candidate.format == "json_api":
                candidate.probe_sample = json.loads(raw.decode("utf-8", errors="replace"))
            elif candidate.format == "csv_feed":
                import csv
                import io

                text = raw.decode("utf-8", errors="replace")
                reader = csv.DictReader(io.StringIO(text))
                candidate.probe_sample = [row for _, row in zip(range(100), reader)]
            else:
                candidate.probe_sample = None
        except Exception:
            log.warning("Parse failed for %s", candidate.name, exc_info=True)
            candidate.probe_sample = None

        return candidate

    def search_and_probe(
        self,
        query_terms: list[str],
        *,
        max_results: int = 10,
        min_relevance: float = 0.1,
    ) -> list[DataSourceCandidate]:
        """Search, filter by relevance, and probe the top candidates."""
        candidates = self.search(query_terms, max_results=max_results * 2)
        filtered = [c for c in candidates if c.relevance_score >= min_relevance]
        probed = []
        for c in filtered[:max_results]:
            self.probe(c)
            if c.probe_sample is not None:
                probed.append(c)
        return probed


# ── Discovery orchestration ────────────────────────────────────


def run_source_discovery(
    store: PipelineStore,
    registry: ToolRegistry,
    bandit: ToolRoutingBandit,
    *,
    query_terms: list[str] | None = None,
    max_new_tools: int = 1,
) -> list[str]:
    """Run one discovery cycle.  Returns list of newly created tool names.

    Pipeline:
    1. Build query terms from existing entity types + feature name prefixes
    2. Build existing_source_urls from store
    3. SourceScout.search_and_probe()
    4. SignalEvaluator.evaluate() each candidate
    5. ToolFactory.create_tool() for candidates passing MI threshold
    6. Register + quarantine

    Reference: Spec step 15.7 in [[tier8_autonomous_discovery_spec]].
    """
    from agent.discovery.signal_evaluator import SignalEvaluator
    from agent.discovery.tool_factory import ToolFactory

    # Build topic vocabulary from existing entity types and feature prefixes
    vocabulary: set[str] = set()
    try:
        types_rows = store.query_entity_types(active_only=True)
        for r in types_rows:
            vocabulary.add(r["type_name"])
    except Exception:
        pass
    # Add feature name prefixes
    try:
        conn = store._get_conn()  # noqa: SLF001
        rows = conn.execute("SELECT DISTINCT feature_name FROM features").fetchall()
        for r in rows:
            prefix = r[0].split(".")[0] if "." in r[0] else r[0]
            vocabulary.add(prefix)
    except Exception:
        pass

    if query_terms is None:
        query_terms = list(vocabulary) if vocabulary else ["economic", "financial", "market"]

    # Known URLs
    existing_urls: set[str] = set()
    for src in store.query_discovered_sources():
        existing_urls.add(src["url"])

    scout = SourceScout(
        existing_source_urls=existing_urls,
        topic_vocabulary=vocabulary,
    )
    evaluator = SignalEvaluator(store=store)
    factory = ToolFactory()

    candidates = scout.search_and_probe(query_terms)
    created: list[str] = []

    for candidate in candidates:
        if len(created) >= max_new_tools:
            break

        # Skip if URL already known
        if candidate.url in existing_urls:
            continue

        # Evaluate signal
        report = evaluator.evaluate(candidate)

        # Store as discovered regardless
        store.store_discovered_source(
            source_id=candidate.source_id,
            name=candidate.name,
            url=candidate.url,
            fmt=candidate.format,
            description=candidate.description,
            update_frequency=candidate.update_frequency,
            topic_tags=candidate.topic_tags,
            probe_result=candidate.probe_sample,
            mi_score=report.max_mi,
            status="quarantine" if report.passes_threshold else "discovered",
        )

        if not report.passes_threshold:
            log.info(
                "Source %s MI=%.4f below threshold, skipped",
                candidate.name,
                report.max_mi,
            )
            continue

        # Create tool
        tool = factory.create_tool(candidate, report)
        if tool is None:
            continue

        # Save config and register
        config = tool.to_config()
        factory.save_config(tool)
        store.store_discovered_source(
            source_id=candidate.source_id,
            name=candidate.name,
            url=candidate.url,
            fmt=candidate.format,
            mi_score=report.max_mi,
            status="quarantine",
            tool_config=config,
        )

        registry.register(tool)
        bandit.add_arm(tool.name)
        created.append(tool.name)
        log.info(
            "Created discovered tool %s from %s (MI=%.4f)",
            tool.name,
            candidate.name,
            report.max_mi,
        )

    return created
