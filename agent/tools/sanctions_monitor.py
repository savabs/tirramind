"""
Tool: Sanctions Monitor — OFAC SDN + UN Security Council Consolidated List

OFAC SDN: https://www.treasury.gov/ofac/downloads/sdn.csv
UN SC:    https://scsanctions.un.org/resources/xml/en/consolidated.xml

Everyone checks if a counterparty is on the SDN list for compliance.
Nobody auto-monitors for *new* additions as a geopolitical escalation signal.

Modes:
  search   — Search sanctioned entities by name across OFAC + UN.
  recent   — Recently listed/updated entities (UN has per-entry dates).
  programs — Overview of active sanctions programs with entity counts.

Signal theory:
  - New entity additions = policy escalation (hawkish, sector-specific)
  - New sanctions program = geopolitical regime change → broad market impact
  - Program entity count growth = conflict intensification
  - OFAC + UN combination = comprehensive global coverage
  - SDN changes precede market reaction by hours (T0 structured data)
"""

from __future__ import annotations

import csv
import io
import logging
import re
from datetime import UTC, datetime, timedelta

UTC = UTC
from typing import TYPE_CHECKING, Any

import defusedxml.ElementTree as ET
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

_OFAC_SDN_URL = "https://www.treasury.gov/ofac/downloads/sdn.csv"
_UN_XML_URL = "https://scsanctions.un.org/resources/xml/en/consolidated.xml"
_UA = "TirraMind/0.1"
_TIMEOUT = 30  # larger timeout for multi-MB downloads
_CACHE_TTL = 21600  # 6 hours — lists change at most weekly

# OFAC CSV field indices (no header row)
_F_UID = 0
_F_NAME = 1
_F_TYPE = 2
_F_PROGRAM = 3
_F_TITLE = 4
# 5-10: vessel fields (call_sign, vess_type, tonnage, grt, vess_flag, vess_owner)
_F_REMARKS = 11

VALID_MODES = {"search", "recent", "programs"}
VALID_SOURCES = {"ofac", "un", "all"}
VALID_ENTITY_TYPES = {"individual", "entity", "vessel", "aircraft", "all"}

# ── Program → Country mapping (ISO 3166-1 alpha-2) ─────────────
# Only programs that map to a single country get links.
_PROGRAM_COUNTRY: dict[str, str | None] = {
    "IRAN": "IR",
    "IRAN-TRA": "IR",
    "IRAN-HR": "IR",
    "IFSR": "IR",
    "IRGC": "IR",
    "IRAN-EO13846": "IR",
    "IRAN-EO13871": "IR",
    "IRAN-EO13902": "IR",
    "CUBA": "CU",
    "UKRAINE-EO13660": "UA",
    "UKRAINE-EO13661": "UA",
    "UKRAINE-EO13662": "UA",
    "UKRAINE-EO13685": "UA",
    "RUSSIA": "RU",
    "RUSSIA-EO14024": "RU",
    "RUSSIA-EO14071": "RU",
    "SYRIA": "SY",
    "DPRK": "KP",
    "DPRK2": "KP",
    "DPRK3": "KP",
    "DPRK4": "KP",
    "CHINA": "CN",
    "CMIC": "CN",
    "HK-EO13936": "HK",
    "VENEZUELA": "VE",
    "VENEZUELA-EO13692": "VE",
    "MYANMAR": "MM",
    "MYANMAR-EO14014": "MM",
    "MALI": "ML",
    "CAR": "CF",
    "DRC": "CD",
    "SOL": "SO",
    "SOMALIA": "SO",
    "YEM": "YE",
    "YEMEN": "YE",
    "LBY": "LY",
    "LIBYA": "LY",
    "HTI": "HT",
    "HAITI": "HT",
    "NICARAGUA": "NI",
    "ETHIOPIA": "ET",
    "LEBANON": "LB",
    "IRAQ": "IQ",
    "BURUNDI": "BI",
    "SUDAN": "SD",
    "ZIMBABWE": "ZW",
    "BELARUS": "BY",
    "BALKANS": None,  # multi-country
    "SDGT": None,  # global terrorism
    "SDNTK": None,  # transnational narcotics
    "FTO": None,  # foreign terrorist organization
    "ISIL": None,  # transnational
    "TCO": None,  # transnational criminal orgs
    "GLOMAG": None,  # global Magnitsky — multi-country
    "CYBER2": None,  # transnational
}


def _clean(val: str) -> str | None:
    """Strip whitespace from a field. Return None if empty or '-0-'."""
    v = val.strip()
    if not v or v == "-0-":
        return None
    return v


def _parse_ofac_programs(raw: str) -> list[str]:
    """Parse OFAC program field. Multiple programs delimited by '] ['."""
    cleaned = _clean(raw)
    if not cleaned:
        return []
    # Programs are separated by '] [' pattern
    # e.g. "SDGT] [IFSR" → ["SDGT", "IFSR"]
    parts = re.split(r"\]\s*\[", cleaned)
    return [p.strip().strip("[]") for p in parts if p.strip().strip("[]")]


def _normalize_type(raw: str) -> str:
    """Normalize SDN type to standard values."""
    cleaned = _clean(raw)
    if not cleaned:
        return "entity"
    t = cleaned.lower()
    if t in ("individual", "vessel", "aircraft"):
        return t
    return "entity"


def _parse_ofac_csv(text: str) -> list[dict[str, Any]]:
    """Parse OFAC SDN CSV into normalized records."""
    records: list[dict[str, Any]] = []
    reader = csv.reader(io.StringIO(text))
    for row in reader:
        if len(row) < 12:
            continue
        uid = _clean(row[_F_UID])
        if not uid:
            continue
        # Skip non-numeric UIDs (EOF marker, etc.)
        try:
            int(uid)
        except ValueError:
            continue

        name = _clean(row[_F_NAME]) or ""
        sdn_type = _normalize_type(row[_F_TYPE])
        programs = _parse_ofac_programs(row[_F_PROGRAM])
        remarks = _clean(row[_F_REMARKS]) or ""

        # Extract aliases from remarks if present
        aliases: list[str] = []
        aka_matches = re.findall(r"a\.k\.a\.\s*['\"]?([^'\";\)]+)['\"]?", remarks, re.IGNORECASE)
        aliases.extend(a.strip() for a in aka_matches if a.strip())

        # Extract nationality from remarks
        nationality: str | None = None
        nat_match = re.search(r"nationality\s+(\w[\w\s]*?)(?:;|$)", remarks, re.IGNORECASE)
        if nat_match:
            nationality = nat_match.group(1).strip()

        records.append(
            {
                "source": "ofac",
                "entity_id": uid,
                "name": name,
                "type": sdn_type,
                "programs": programs,
                "listed_date": None,  # OFAC CSV has no per-entry dates
                "last_updated": None,
                "nationality": nationality,
                "aliases": aliases,
                "remarks": remarks,
            }
        )
    return records


def _parse_un_xml(text: str) -> list[dict[str, Any]]:
    """Parse UN Security Council consolidated XML into normalized records."""
    records: list[dict[str, Any]] = []
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        log.warning("Failed to parse UN XML: %s", exc)
        return records

    # Process INDIVIDUALS
    for indiv in root.iter("INDIVIDUAL"):
        dataid = _text(indiv, "DATAID")
        if not dataid:
            continue

        first = _text(indiv, "FIRST_NAME") or ""
        second = _text(indiv, "SECOND_NAME") or ""
        third = _text(indiv, "THIRD_NAME") or ""
        name_parts = [p for p in (first, second, third) if p]
        name = " ".join(name_parts)

        un_list_type = _text(indiv, "UN_LIST_TYPE") or ""
        listed_on = _text(indiv, "LISTED_ON")
        comments = _text(indiv, "COMMENTS1") or ""

        # Last updated
        last_updated: str | None = None
        for ldu in indiv.iter("LAST_DAY_UPDATED"):
            val = _text(ldu, "VALUE")
            if val:
                last_updated = val
                break

        # Nationality
        nationality: str | None = None
        for nat in indiv.iter("NATIONALITY"):
            val = _text(nat, "VALUE")
            if val:
                nationality = val
                break

        # Aliases
        aliases: list[str] = []
        for alias in indiv.iter("INDIVIDUAL_ALIAS"):
            alias_name = _text(alias, "ALIAS_NAME")
            if alias_name:
                aliases.append(alias_name)

        records.append(
            {
                "source": "un",
                "entity_id": dataid,
                "name": name,
                "type": "individual",
                "programs": [un_list_type] if un_list_type else [],
                "listed_date": listed_on,
                "last_updated": last_updated,
                "nationality": nationality,
                "aliases": aliases,
                "remarks": comments,
            }
        )

    # Process ENTITIES
    for entity in root.iter("ENTITY"):
        dataid = _text(entity, "DATAID")
        if not dataid:
            continue

        first = _text(entity, "FIRST_NAME") or ""
        name = first or _text(entity, "NAME_ORIGINAL_SCRIPT") or ""

        un_list_type = _text(entity, "UN_LIST_TYPE") or ""
        listed_on = _text(entity, "LISTED_ON")
        comments = _text(entity, "COMMENTS1") or ""

        last_updated = None
        for ldu in entity.iter("LAST_DAY_UPDATED"):
            val = _text(ldu, "VALUE")
            if val:
                last_updated = val
                break

        aliases: list[str] = []
        for alias in entity.iter("ENTITY_ALIAS"):
            alias_name = _text(alias, "ALIAS_NAME")
            if alias_name:
                aliases.append(alias_name)

        records.append(
            {
                "source": "un",
                "entity_id": dataid,
                "name": name,
                "type": "entity",
                "programs": [un_list_type] if un_list_type else [],
                "listed_date": listed_on,
                "last_updated": last_updated,
                "nationality": None,
                "aliases": aliases,
                "remarks": comments,
            }
        )

    return records


def _text(elem: ET.Element, tag: str) -> str | None:
    """Extract text from a child element, or None."""
    child = elem.find(tag)
    if child is not None and child.text:
        return child.text.strip()
    return None


def _matches_query(record: dict[str, Any], query: str) -> bool:
    """Check if a record matches a search query (case-insensitive substring)."""
    q = query.lower()
    if q in record["name"].lower():
        return True
    for alias in record.get("aliases", []):
        if q in alias.lower():
            return True
    return False


def _format_record(rec: dict[str, Any], *, brief: bool = False) -> str:
    """Format a single record for text output."""
    src = rec["source"].upper()
    name = rec["name"]
    etype = rec["type"]
    progs = ", ".join(rec["programs"]) if rec["programs"] else "—"
    parts = [f"  [{src}] {name} ({etype}) — Programs: {progs}"]

    if not brief:
        if rec.get("listed_date"):
            parts.append(f"    Listed: {rec['listed_date']}")
        if rec.get("last_updated"):
            parts.append(f"    Updated: {rec['last_updated']}")
        if rec.get("nationality"):
            parts.append(f"    Nationality: {rec['nationality']}")
        if rec.get("aliases"):
            parts.append(f"    AKA: {', '.join(rec['aliases'][:5])}")
        if rec.get("remarks") and len(rec["remarks"]) > 0:
            remarks_trunc = rec["remarks"][:200]
            if len(rec["remarks"]) > 200:
                remarks_trunc += "…"
            parts.append(f"    Remarks: {remarks_trunc}")
    return "\n".join(parts)


class SanctionsMonitorTool(Tool):
    name = "sanctions_monitor"
    description = (
        "Search and monitor global sanctions lists (OFAC SDN + UN Security Council). "
        "Mode 'search' finds entities by name across both lists. "
        "Mode 'recent' shows entities recently added/updated (UN has per-entry dates). "
        "Mode 'programs' shows active sanctions programs with entity counts. "
        "Free, no API key required. OFAC ~18,700 entries, UN ~900 entries."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["search", "recent", "programs"],
                "default": "search",
                "description": (
                    "search = find entities by name. "
                    "recent = recently listed/updated (UN only has dates). "
                    "programs = overview of sanctions programs."
                ),
            },
            "query": {
                "type": "string",
                "default": "",
                "description": (
                    "Name to search for (search mode). Case-insensitive substring match. Also matches aliases."
                ),
            },
            "source": {
                "type": "string",
                "enum": ["ofac", "un", "all"],
                "default": "all",
                "description": "Data source: ofac, un, or all.",
            },
            "entity_type": {
                "type": "string",
                "enum": ["individual", "entity", "vessel", "aircraft", "all"],
                "default": "all",
                "description": "Filter by entity type.",
            },
            "program": {
                "type": "string",
                "default": "",
                "description": (
                    "Filter by sanctions program code (e.g., SDGT, IRAN, CUBA, DRC). Case-insensitive substring match."
                ),
            },
            "days_back": {
                "type": "integer",
                "default": 90,
                "description": "For 'recent' mode: how many days back. Default 90, max 365.",
            },
            "limit": {
                "type": "integer",
                "default": 25,
                "description": "Max results. Default 25, max 100.",
            },
        },
        "required": [],
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

    def _persist_entities(self, results: list[dict[str, Any]]) -> None:
        """Register sanctioned entities and create country links."""
        if self._store is None or entity_id_from_key is None:
            return
        if not results:
            return
        try:
            self._persist_entities_inner(results)
        except Exception:
            log.exception("Entity persistence failed (non-fatal)")

    def _persist_entities_inner(self, results: list[dict[str, Any]]) -> None:
        assert self._store is not None  # noqa: S101
        store = self._store

        _SDN_TYPE_MAP = {
            "individual": "person",
            "entity": "organization",
            "vessel": "vessel",
            "aircraft": "organization",  # no aircraft entity type; treat as org
        }

        seen: set[str] = set()
        for rec in results:
            name = (rec.get("name") or "").strip()
            if not name:
                continue

            sdn_type = (rec.get("type") or "entity").lower()
            ent_type = _SDN_TYPE_MAP.get(sdn_type, "organization")

            # Normalize name: use normalize_company_name for orgs, simpler for persons/vessels
            if ent_type == "organization" and normalize_company_name:
                try:
                    canon = normalize_company_name(name)
                except (ValueError, TypeError):
                    canon = name.strip().lower()
            else:
                canon = name.strip().lower()

            eid = entity_id_from_key(ent_type, canon)

            if eid not in seen:
                seen.add(eid)
                store.register_entity(
                    entity_type=ent_type,
                    canonical_name=canon,
                    entity_id=eid,
                    metadata={
                        "source": rec.get("source", "unknown"),
                        "original_name": name,
                        "sdn_type": sdn_type,
                    },
                )
                # Add source-specific alias
                source_id = rec.get("entity_id", "")
                if source_id:
                    store.add_entity_alias(
                        eid,
                        f"sanctions_{rec.get('source', 'unknown')}",
                        str(source_id),
                    )

            # Observation
            listed_date = rec.get("listed_date") or rec.get("last_updated")
            try:
                ts = (
                    datetime.fromisoformat(listed_date.replace("Z", "+00:00")).timestamp()
                    if listed_date
                    else datetime.now(tz=UTC).timestamp()
                )
            except (ValueError, AttributeError):
                ts = datetime.now(tz=UTC).timestamp()

            store.store_entity_observation(
                entity_id=eid,
                source_tool="sanctions_monitor",
                observed_at=ts,
                observation_type="sanctions_listing",
                depth_level=2,
                value={
                    "source": rec.get("source", "unknown"),
                    "programs": rec.get("programs", []),
                    "nationality": rec.get("nationality"),
                    "aliases": rec.get("aliases", [])[:5],
                },
            )

            # ── Program → country links ──
            for prog in rec.get("programs", []):
                # Try exact match first, then uppercase
                country_code = _PROGRAM_COUNTRY.get(prog) or _PROGRAM_COUNTRY.get(prog.upper())
                if not country_code:
                    continue

                country_eid = entity_id_from_key("country", country_code.lower())
                store.register_entity(
                    entity_type="country",
                    canonical_name=country_code,
                    entity_id=country_eid,
                )
                # Avoid self-link (shouldn't happen, but guard)
                if eid != country_eid:
                    store.link_entities(
                        entity_id_a=eid,
                        entity_id_b=country_eid,
                        link_type="sanctioned_under",
                        source="sanctions_monitor",
                        confidence=0.95,
                        metadata={"program": prog, "data_source": rec.get("source")},
                    )

    def execute(
        self,
        *,
        mode: str = "search",
        query: str = "",
        source: str = "all",
        entity_type: str = "all",
        program: str = "",
        days_back: int = 90,
        limit: int = 25,
        _backfill: bool = False,
        **_: Any,
    ) -> ToolResult:
        mode = mode.lower().strip()
        if mode not in VALID_MODES:
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Use: {', '.join(sorted(VALID_MODES))}.",
            )

        source = source.lower().strip()
        if source not in VALID_SOURCES:
            return ToolResult(
                success=False,
                output=f"Invalid source '{source}'. Use: {', '.join(sorted(VALID_SOURCES))}.",
            )

        entity_type = entity_type.lower().strip()
        if entity_type not in VALID_ENTITY_TYPES:
            return ToolResult(
                success=False,
                output=f"Invalid entity_type '{entity_type}'. Use: {', '.join(sorted(VALID_ENTITY_TYPES))}.",
            )

        if not _backfill:
            days_back = max(1, min(days_back, 365))
        limit = max(1, min(limit, 100))

        if mode == "search":
            if not query.strip():
                return ToolResult(
                    success=False,
                    output="Search mode requires a 'query' parameter (entity name to search for).",
                )
            return self._execute_search(
                query=query.strip(),
                source=source,
                entity_type=entity_type,
                program=program.strip(),
                limit=limit,
            )

        if mode == "recent":
            return self._execute_recent(
                source=source,
                entity_type=entity_type,
                program=program.strip(),
                days_back=days_back,
                limit=limit,
            )

        # programs
        return self._execute_programs(source=source)

    # ------------------------------------------------------------------
    # search mode
    # ------------------------------------------------------------------

    def _execute_search(
        self,
        *,
        query: str,
        source: str,
        entity_type: str,
        program: str,
        limit: int,
    ) -> ToolResult:
        records, error = self._get_records(source)
        if error:
            return ToolResult(success=False, output=error)

        # Filter
        matched = [r for r in records if _matches_query(r, query)]

        if entity_type != "all":
            matched = [r for r in matched if r["type"] == entity_type]

        if program:
            prog_lower = program.lower()
            matched = [r for r in matched if any(prog_lower in p.lower() for p in r["programs"])]

        matched = matched[:limit]

        if not matched:
            return ToolResult(
                success=True,
                output=f"Sanctions search: no results for '{query}'"
                + (f" in {source.upper()}" if source != "all" else "")
                + ".",
                data={"query": query, "results": [], "count": 0},
            )

        lines = [
            f"Sanctions search: {len(matched)} result(s) for '{query}':",
            "",
        ]
        for rec in matched:
            lines.append(_format_record(rec))
            lines.append("")

        self._persist_entities(matched)

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"query": query, "results": matched, "count": len(matched)},
        )

    # ------------------------------------------------------------------
    # recent mode
    # ------------------------------------------------------------------

    def _execute_recent(
        self,
        *,
        source: str,
        entity_type: str,
        program: str,
        days_back: int,
        limit: int,
    ) -> ToolResult:
        # For recent mode, prefer UN (has dates). Include OFAC if requested,
        # but note OFAC has no per-entry dates.
        records, error = self._get_records(source)
        if error:
            return ToolResult(success=False, output=error)

        cutoff = (datetime.now(UTC) - timedelta(days=days_back)).strftime("%Y-%m-%d")

        recent: list[dict[str, Any]] = []
        for rec in records:
            # Use listed_date or last_updated, whichever is more recent
            date_str = rec.get("last_updated") or rec.get("listed_date")
            if not date_str:
                continue
            if date_str >= cutoff:
                rec_copy = dict(rec)
                rec_copy["sort_date"] = date_str
                recent.append(rec_copy)

        if entity_type != "all":
            recent = [r for r in recent if r["type"] == entity_type]

        if program:
            prog_lower = program.lower()
            recent = [r for r in recent if any(prog_lower in p.lower() for p in r["programs"])]

        recent.sort(key=lambda r: r.get("sort_date", ""), reverse=True)
        recent = recent[:limit]

        if not recent:
            note = ""
            if source in ("ofac", "all"):
                note = " Note: OFAC SDN has no per-entry listing dates."
            return ToolResult(
                success=True,
                output=f"Sanctions recent: no entities listed/updated in last {days_back}d." + note,
                data={"results": [], "count": 0, "days_back": days_back},
            )

        lines = [
            f"Sanctions: {len(recent)} entities listed/updated in last {days_back}d:",
            "",
        ]
        if source in ("ofac", "all"):
            lines.append("  (Note: OFAC entries excluded — no per-entry dates)")
            lines.append("")

        for rec in recent:
            lines.append(_format_record(rec))
            lines.append("")

        self._persist_entities(recent)

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"results": recent, "count": len(recent), "days_back": days_back},
        )

    # ------------------------------------------------------------------
    # programs mode
    # ------------------------------------------------------------------

    def _execute_programs(self, *, source: str) -> ToolResult:
        records, error = self._get_records(source)
        if error:
            return ToolResult(success=False, output=error)

        # Aggregate by program
        prog_counts: dict[str, dict[str, Any]] = {}
        for rec in records:
            for prog in rec["programs"]:
                if prog not in prog_counts:
                    prog_counts[prog] = {
                        "program": prog,
                        "count": 0,
                        "sources": set(),
                        "types": {},
                        "examples": [],
                    }
                entry = prog_counts[prog]
                entry["count"] += 1
                entry["sources"].add(rec["source"])
                entry["types"][rec["type"]] = entry["types"].get(rec["type"], 0) + 1
                if len(entry["examples"]) < 3:
                    entry["examples"].append(rec["name"])

        if not prog_counts:
            return ToolResult(
                success=True,
                output="No sanctions programs found.",
                data={"programs": [], "count": 0},
            )

        # Sort by count descending
        sorted_progs = sorted(prog_counts.values(), key=lambda p: -p["count"])

        # Serialize sources from set to list
        for p in sorted_progs:
            p["sources"] = sorted(p["sources"])

        lines = [
            f"Sanctions Programs: {len(sorted_progs)} active programs "
            f"({sum(p['count'] for p in sorted_progs)} total entries):",
            "",
        ]
        for p in sorted_progs:
            src_str = "/".join(s.upper() for s in p["sources"])
            type_str = ", ".join(f"{v} {k}" for k, v in sorted(p["types"].items(), key=lambda x: -x[1]))
            examples = "; ".join(p["examples"][:3])
            lines.append(f"  {p['program']:30s}  {p['count']:>5d} entries  [{src_str}]  ({type_str})")
            lines.append(f"    e.g.: {examples}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"programs": sorted_progs, "count": len(sorted_progs)},
        )

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _get_records(self, source: str) -> tuple[list[dict[str, Any]], str | None]:
        """Fetch and merge records from requested sources."""
        records: list[dict[str, Any]] = []
        errors: list[str] = []

        if source in ("ofac", "all"):
            ofac, err = self._fetch_ofac()
            if err:
                errors.append(f"OFAC: {err}")
            else:
                records.extend(ofac)

        if source in ("un", "all"):
            un, err = self._fetch_un()
            if err:
                errors.append(f"UN: {err}")
            else:
                records.extend(un)

        # If ALL sources failed, return error
        if not records and errors:
            return [], "; ".join(errors)

        return records, None

    def _fetch_ofac(self) -> tuple[list[dict[str, Any]], str | None]:
        """Download and parse OFAC SDN CSV."""
        cache_key = {"source": "ofac_sdn"}
        if self._cache:
            cached = self._cache.get("sanctions_monitor", cache_key)
            if cached is not None:
                return cached, None

        try:
            resp = httpx.get(
                _OFAC_SDN_URL,
                timeout=_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": _UA},
            )
            resp.raise_for_status()
        except httpx.TimeoutException:
            return [], "OFAC SDN download timed out (file is ~5MB)."
        except httpx.HTTPStatusError as exc:
            return [], f"OFAC SDN HTTP {exc.response.status_code}."
        except httpx.ConnectError:
            return [], "OFAC SDN connection failed."

        records = _parse_ofac_csv(resp.text)
        if not records:
            return [], "OFAC SDN: parsed 0 records (unexpected)."

        if self._cache:
            self._cache.put("sanctions_monitor", cache_key, records)

        return records, None

    def _fetch_un(self) -> tuple[list[dict[str, Any]], str | None]:
        """Download and parse UN Security Council consolidated XML."""
        cache_key = {"source": "un_consolidated"}
        if self._cache:
            cached = self._cache.get("sanctions_monitor", cache_key)
            if cached is not None:
                return cached, None

        try:
            resp = httpx.get(
                _UN_XML_URL,
                timeout=_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": _UA},
            )
            resp.raise_for_status()
        except httpx.TimeoutException:
            return [], "UN SC XML download timed out."
        except httpx.HTTPStatusError as exc:
            return [], f"UN SC HTTP {exc.response.status_code}."
        except httpx.ConnectError:
            return [], "UN SC connection failed."

        records = _parse_un_xml(resp.text)
        if not records:
            return [], "UN SC: parsed 0 records (unexpected)."

        if self._cache:
            self._cache.put("sanctions_monitor", cache_key, records)

        return records, None
