"""
Tool: Drug Regulatory — FDA Drug Approvals, Adverse Events & Labels

Fetch and parse FDA drug regulatory data via the OpenFDA API.
Provides drug approvals (Drugs@FDA), adverse event queries (FAERS),
and drug label data.

Data source: https://api.fda.gov/ (free, no auth required for basic tier).
Rate limits: 240 req/min, 1000 req/day per IP (no key). 120K req/day (with free key).
Approvals: 1939-present, daily updates (M-F).
Adverse events: 2004-present, quarterly updates.

Signal theory:
  - FDA approvals (NDA/BLA) = binary pharma stock events (5-30% moves)
  - Supplemental approvals = label expansion → revenue growth signals
  - Adverse event spikes = safety recall leading indicator → stock crash catalyst
  - Seriousness ratio changes = escalating safety signal
  - Priority review = accelerated timeline, competitive landscape shift
  - Label warning additions = prescribing impact → revenue headwind
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

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

_ENDPOINTS = {
    "approvals": "https://api.fda.gov/drug/drugsfda.json",
    "adverse_events": "https://api.fda.gov/drug/event.json",
    "labels": "https://api.fda.gov/drug/label.json",
}
_UA = "TirraMind/0.1 (research)"
_TIMEOUT = 20

VALID_MODES = frozenset(_ENDPOINTS)


class DrugRegulatoryTool(Tool):
    name = "drug_regulatory"
    description = (
        "Fetch FDA drug regulatory data. "
        "Modes: approvals (Drugs@FDA — recent/historical NDA/BLA approvals), "
        "adverse_events (FAERS — adverse event reports by drug/reaction), "
        "labels (drug label data — warnings, indications, boxed warnings). "
        "Supports Elasticsearch query syntax, date ranges, and faceted counts. "
        "Free, no auth required."
    )
    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": sorted(VALID_MODES),
                "description": (
                    "approvals = Drugs@FDA (NDA/BLA submissions); "
                    "adverse_events = FAERS adverse event reports; "
                    "labels = drug labeling (warnings, indications)"
                ),
            },
            "drug_name": {
                "type": "string",
                "default": "",
                "description": (
                    "Convenience filter: drug brand or generic name. "
                    "Adds appropriate search term for the selected mode."
                ),
            },
            "search": {
                "type": "string",
                "default": "",
                "description": (
                    "Raw OpenFDA Elasticsearch search query. "
                    "Overrides drug_name if both provided. "
                    "Example: 'patient.drug.medicinalproduct:aspirin+AND+serious:1'"
                ),
            },
            "date_start": {
                "type": "string",
                "default": "",
                "description": "Start date (YYYYMMDD) for date-range filtering.",
            },
            "date_end": {
                "type": "string",
                "default": "",
                "description": "End date (YYYYMMDD) for date-range filtering.",
            },
            "count_field": {
                "type": "string",
                "default": "",
                "description": (
                    "Return faceted counts for this field instead of records. "
                    "E.g. 'patient.reaction.reactionmeddrapt' for top reactions."
                ),
            },
            "limit": {
                "type": "integer",
                "default": 25,
                "description": "Max results to return (max 100 for records, 1000 for counts).",
            },
        },
        "required": ["mode"],
    }

    def __init__(
        self,
        cache: DataCache | None = None,
        *,
        pipeline_store: PipelineStore | None = None,
    ) -> None:
        self._cache = cache
        self._store = pipeline_store

    # ------------------------------------------------------------------
    # Entity persistence (L2)
    # ------------------------------------------------------------------

    def _persist_entities(
        self,
        mode: str,
        parsed: list[dict[str, Any]],
        signals: dict[str, Any] | None = None,
    ) -> None:
        """Register company entities and store drug_approval observations."""
        if self._store is None or entity_id_from_key is None:
            return
        if not parsed:
            return
        try:
            self._persist_entities_inner(mode, parsed, signals)
        except Exception:
            log.exception("Entity persistence failed (non-fatal)")

    def _persist_entities_inner(
        self,
        mode: str,
        parsed: list[dict[str, Any]],
        signals: dict[str, Any] | None = None,
    ) -> None:
        assert self._store is not None  # noqa: S101
        store = self._store

        # Register a US country entity once for FDA market-authorization links
        us_eid = entity_id_from_key("country", "US")
        us_registered = False

        seen: set[str] = set()

        if mode == "approvals":
            for rec in parsed:
                sponsor = (rec.get("sponsor") or "").strip()
                if not sponsor:
                    continue
                canon = normalize_company_name(sponsor) if normalize_company_name else sponsor
                eid = entity_id_from_key("company", canon)

                if eid not in seen:
                    seen.add(eid)
                    store.register_entity(
                        entity_type="company",
                        canonical_name=canon,
                        entity_id=eid,
                        metadata={"source": "fda", "sponsor_name": sponsor},
                    )

                # Timestamp from submission date
                date_str = rec.get("latest_submission_date") or ""
                try:
                    ts = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=UTC).timestamp()
                except (ValueError, TypeError):
                    ts = datetime.now(tz=UTC).timestamp()

                store.store_entity_observation(
                    entity_id=eid,
                    observation_type="drug_approval",
                    depth_level=2,
                    value={
                        "application_number": rec.get("application_number"),
                        "brand_names": rec.get("brands", []),
                        "submission_type": rec.get("latest_submission_type"),
                        "submission_date": date_str,
                        "review_priority": rec.get("review_priority"),
                    },
                    observed_at=ts,
                    source_tool="drug_regulatory",
                )

                # Market authorization link: company → US country
                if not us_registered:
                    store.register_entity(
                        entity_type="country",
                        canonical_name="US",
                        entity_id=us_eid,
                        metadata={"source": "fda"},
                    )
                    us_registered = True

                store.link_entities(
                    entity_id_a=eid,
                    entity_id_b=us_eid,
                    link_type="market_authorized_in",
                    source="drug_regulatory",
                    confidence=1.0,
                )

        elif mode == "adverse_events":
            for rec in parsed:
                drugs = rec.get("drugs") or []
                for drug_name in drugs:
                    drug_name = drug_name.strip()
                    if not drug_name or drug_name == "?":
                        continue
                    canon = normalize_company_name(drug_name) if normalize_company_name else drug_name
                    eid = entity_id_from_key("company", canon)

                    if eid not in seen:
                        seen.add(eid)
                        store.register_entity(
                            entity_type="company",
                            canonical_name=canon,
                            entity_id=eid,
                            metadata={"source": "fda_faers", "drug_name": drug_name},
                        )

                    date_str = rec.get("date") or ""
                    try:
                        ts = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=UTC).timestamp()
                    except (ValueError, TypeError):
                        ts = datetime.now(tz=UTC).timestamp()

                    store.store_entity_observation(
                        entity_id=eid,
                        observation_type="drug_approval",
                        depth_level=2,
                        value={
                            "drug_name": drug_name,
                            "reactions": rec.get("reactions", []),
                            "serious": rec.get("serious", False),
                            "receive_date": date_str,
                            "seriousness_ratio": (signals or {}).get("seriousness_ratio"),
                        },
                        observed_at=ts,
                        source_tool="drug_regulatory",
                    )

        # labels mode: informational only — no entity persistence

    # ── Public execute ───────────────────────────────────────────────

    def execute(
        self,
        *,
        mode: str = "approvals",
        drug_name: str = "",
        search: str = "",
        date_start: str = "",
        date_end: str = "",
        count_field: str = "",
        limit: int = 25,
        **_: Any,
    ) -> ToolResult:
        if mode not in VALID_MODES:
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Choose from: {', '.join(sorted(VALID_MODES))}.",
            )

        # Validate date formats if provided
        for label, val in [("date_start", date_start), ("date_end", date_end)]:
            if val:
                if not (len(val) == 8 and val.isdigit()):
                    return ToolResult(
                        success=False,
                        output=f"Invalid {label} '{val}'. Use YYYYMMDD format (e.g. 20260101).",
                    )

        # Clamp limits
        if count_field:
            limit = max(1, min(limit, 1000))
        else:
            limit = max(1, min(limit, 100))

        # Build search string
        search_query = self._build_search(mode, drug_name, search, date_start, date_end)

        try:
            payload = self._fetch(mode, search_query, count_field, limit)
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code == 404:
                return ToolResult(
                    success=True,
                    output=f"No results found for mode='{mode}' with search='{search_query}'.",
                    data={"mode": mode, "results": [], "total": 0},
                )
            if code == 429:
                return ToolResult(
                    success=False,
                    output="OpenFDA rate limit exceeded. Try again later.",
                )
            log.exception("OpenFDA HTTP error %d for mode=%s", code, mode)
            return ToolResult(success=False, output=f"OpenFDA HTTP error {code}.")
        except Exception as exc:
            log.exception("OpenFDA fetch failed for mode=%s", mode)
            return ToolResult(success=False, output=f"OpenFDA error: {exc}")

        # Parse response
        meta = payload.get("meta", {})
        results = payload.get("results", [])
        total = meta.get("results", {}).get("total", len(results))

        if not results:
            return ToolResult(
                success=True,
                output=f"No results for mode='{mode}', search='{search_query}'.",
                data={"mode": mode, "results": [], "total": 0},
            )

        # Format based on mode and whether it's a count query
        if count_field:
            output, data = self._format_counts(mode, results, count_field, total)
        elif mode == "approvals":
            output, data = self._format_approvals(results, total)
        elif mode == "adverse_events":
            output, data = self._format_adverse_events(results, total)
        else:
            output, data = self._format_labels(results, total)

        # L2 entity persistence
        parsed = data.get("results", [])
        signals = data.get("signals")
        self._persist_entities(mode, parsed, signals)

        return ToolResult(success=True, output=output, data=data)

    # ── Search builder ───────────────────────────────────────────────

    def _build_search(
        self,
        mode: str,
        drug_name: str,
        raw_search: str,
        date_start: str,
        date_end: str,
    ) -> str:
        """Build the Elasticsearch search query string."""
        if raw_search:
            return raw_search

        parts: list[str] = []

        # Drug name convenience filter
        if drug_name:
            name = quote(drug_name, safe="")
            if mode == "approvals":
                parts.append(f"products.brand_name:{name}")
            elif mode == "adverse_events":
                parts.append(f"patient.drug.medicinalproduct:{name}")
            else:  # labels
                parts.append(f"openfda.brand_name:{name}")

        # Date range
        if date_start or date_end:
            ds = date_start or "19000101"
            de = date_end or "29991231"
            if mode == "approvals":
                parts.append(f"submissions.submission_status_date:[{ds}+TO+{de}]")
            elif mode == "adverse_events":
                parts.append(f"receivedate:[{ds}+TO+{de}]")
            else:  # labels
                parts.append(f"effective_time:[{ds}+TO+{de}]")

        return "+AND+".join(parts)

    # ── Fetch ────────────────────────────────────────────────────────

    def _fetch(self, mode: str, search: str, count_field: str, limit: int) -> dict[str, Any]:
        cache_key = {
            "source": f"openfda_{mode}",
            "search": search,
            "count": count_field,
            "limit": limit,
        }
        if self._cache:
            cached = self._cache.get("drug_regulatory", cache_key)
            if cached is not None:
                log.debug("OpenFDA %s: cache hit", mode)
                return cached

        url = _ENDPOINTS[mode]
        params: dict[str, str | int] = {"limit": limit}
        if search:
            params["search"] = search
        if count_field:
            params["count"] = count_field

        with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
            resp = client.get(url, params=params, headers={"User-Agent": _UA})
            resp.raise_for_status()

        payload = resp.json()

        if self._cache:
            self._cache.put("drug_regulatory", cache_key, payload)

        return payload

    # ── Formatters ───────────────────────────────────────────────────

    def _format_approvals(self, results: list[dict], total: int) -> tuple[str, dict[str, Any]]:
        lines = [f"## FDA Drug Approvals (Drugs@FDA) — {total} total\n"]
        parsed = []

        for r in results:
            app_num = r.get("application_number", "?")
            sponsor = r.get("sponsor_name", "?")
            products = r.get("products", [])
            submissions = r.get("submissions", [])

            brand_names = list({p.get("brand_name", "?") for p in products if p.get("brand_name")})
            brand_str = ", ".join(brand_names[:3]) or "?"

            # Most recent submission
            latest_sub = {}
            if submissions:
                sorted_subs = sorted(
                    submissions,
                    key=lambda s: s.get("submission_status_date", ""),
                    reverse=True,
                )
                latest_sub = sorted_subs[0]

            sub_type = latest_sub.get("submission_type", "?")
            sub_date = latest_sub.get("submission_status_date", "?")
            priority = latest_sub.get("review_priority", "STANDARD")

            lines.append(f"  {app_num} | {brand_str} | {sponsor} | {sub_type} ({sub_date}) | {priority}")
            parsed.append(
                {
                    "application_number": app_num,
                    "sponsor": sponsor,
                    "brands": brand_names,
                    "latest_submission_type": sub_type,
                    "latest_submission_date": sub_date,
                    "review_priority": priority,
                }
            )

        lines.append(f"\n  Showing: {len(results)} / {total}")
        return "\n".join(lines), {
            "mode": "approvals",
            "results": parsed,
            "total": total,
        }

    def _format_adverse_events(self, results: list[dict], total: int) -> tuple[str, dict[str, Any]]:
        lines = [f"## FDA Adverse Events (FAERS) — {total} total\n"]
        parsed = []
        serious_count = 0

        for r in results:
            receive_date = r.get("receivedate", "?")
            serious = r.get("serious", 0)
            if serious:
                serious_count += 1

            drugs = r.get("patient", {}).get("drug", [])
            reactions = r.get("patient", {}).get("reaction", [])

            drug_names = [d.get("medicinalproduct", "?") for d in drugs[:3]]
            reaction_names = [rx.get("reactionmeddrapt", "?") for rx in reactions[:3]]

            drug_str = ", ".join(drug_names) or "?"
            rx_str = ", ".join(reaction_names) or "?"
            serious_str = "SERIOUS" if serious else "non-serious"

            lines.append(f"  {receive_date} | {serious_str} | {drug_str} → {rx_str}")
            parsed.append(
                {
                    "date": receive_date,
                    "serious": bool(serious),
                    "drugs": drug_names,
                    "reactions": reaction_names,
                }
            )

        # Seriousness ratio signal
        signals: dict[str, Any] = {}
        if parsed:
            ratio = serious_count / len(parsed)
            signals["seriousness_ratio"] = round(ratio, 3)
            signals["serious_count"] = serious_count
            signals["total_in_page"] = len(parsed)
            lines.append(f"\n  Seriousness ratio: {ratio:.1%} ({serious_count}/{len(parsed)})")

        lines.append(f"\n  Showing: {len(results)} / {total}")
        return "\n".join(lines), {
            "mode": "adverse_events",
            "results": parsed,
            "total": total,
            "signals": signals,
        }

    def _format_labels(self, results: list[dict], total: int) -> tuple[str, dict[str, Any]]:
        lines = [f"## FDA Drug Labels — {total} total\n"]
        parsed = []

        for r in results:
            ofda = r.get("openfda", {})
            brand = ofda.get("brand_name", ["?"])[0] if ofda.get("brand_name") else "?"
            generic = ofda.get("generic_name", ["?"])[0] if ofda.get("generic_name") else "?"
            has_boxed = bool(r.get("boxed_warning"))
            warnings_text = (r.get("warnings") or [""])[0][:200] if r.get("warnings") else ""

            boxed_str = " [BOXED WARNING]" if has_boxed else ""
            lines.append(f"  {brand} ({generic}){boxed_str}")
            if warnings_text:
                lines.append(f"    Warning excerpt: {warnings_text}...")

            parsed.append(
                {
                    "brand_name": brand,
                    "generic_name": generic,
                    "has_boxed_warning": has_boxed,
                    "warning_excerpt": warnings_text,
                }
            )

        lines.append(f"\n  Showing: {len(results)} / {total}")
        return "\n".join(lines), {"mode": "labels", "results": parsed, "total": total}

    def _format_counts(
        self, mode: str, results: list[dict], count_field: str, total: int
    ) -> tuple[str, dict[str, Any]]:
        lines = [f"## FDA Counts: {count_field} (mode={mode})\n"]
        parsed = []

        for r in results:
            term = r.get("term", "?")
            count = r.get("count", 0)
            lines.append(f"  {term}: {count:,}")
            parsed.append({"term": term, "count": count})

        lines.append(f"\n  Unique terms: {len(results)} / ~{total} total events")
        return "\n".join(lines), {
            "mode": mode,
            "count_field": count_field,
            "results": parsed,
            "total": total,
        }
