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
  snapshot — Persist the full merged roster, first-seen gated. The collection mode.

Why ``snapshot`` exists (2026-09-23)
------------------------------------
Both feeds are *current-state snapshots*, not event streams.  The OFAC SDN CSV
carries no per-entry dates at all (``_parse_ofac_csv`` hard-codes
``listed_date=None``), so a date-windowed mode structurally discards it:
measured on a live download, OFAC parses to 19,393 records, every one of them
dateless, and ``recent`` mode could only ever persist the handful of dated UN
rows inside its window.  The live DB confirmed the damage — 11 distinct
entities for this tool over its whole life, zero of them OFAC.

The only way to know *when* an SDN entry appeared or disappeared is to hold a
prior snapshot and diff it, so every un-snapshotted day is a delta that cannot
be reconstructed at any price.  ``snapshot`` mode therefore stores the roster
itself: one ``sanctions_listing`` observation the first time a source uid is
seen (steady-state row count == new designations since the last run), plus one
unconditional ``sanctions_roster_snapshot`` row per run carrying per-source
record counts and a content hash of each fetched body.

That roster row is load-bearing twice over.  It is stamped with *collection*
time, so it can never dedup away — the node can never again report green while
writing nothing.  And consecutive content hashes make add/remove diffs
computable downstream.  Layer 1 stores the snapshot fact and nothing more: the
diff math belongs in Layer 2, not here.

Two warnings for whoever consumes these rows in Layer 2
-------------------------------------------------------
1. **Day one is not an event.**  The OFAC SDN CSV carries no per-entry dates
   (measured: 1,011 of 20,402 fetched records have any date at all, all UN),
   so the backfill writes ~19k ``sanctions_listing`` rows stamped with
   *collection* time — all within the same second.  Read as designations that
   would be the largest sanctions event in history.  Every row records which
   it is in ``metadata_json.observed_at_source``
   (``source_listed_date`` | ``collection_time`` | ``unparsed_source_date`` |
   ``out_of_range_source_date``); a windowed feature must exclude or weight
   everything that is not ``source_listed_date``.  ``first_seen_backfill``
   marks a row as written by the first-seen-gated ``snapshot`` path; it is
   true for day one *and* for every genuinely new designation found later, so
   it is not by itself a day-one marker — the roster row's ``new_listings``
   is (day one is the run where it equals ``total_records``).
2. **Honour ``complete``.**  A roster row with ``complete: false`` is a
   fragment — at least one requested source did not answer — and diffing it
   against a full one reads as thousands of delistings.  ``rejected > 0``
   means ``new_listings`` is a floor, not a count.

Signal theory:
  - New entity additions = policy escalation (hawkish, sector-specific)
  - New sanctions program = geopolitical regime change → broad market impact
  - Program entity count growth = conflict intensification
  - OFAC + UN combination = comprehensive global coverage
  - SDN changes precede market reaction by hours (T0 structured data)
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import re
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, NamedTuple

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

try:  # pragma: no cover - exercised implicitly by every persist test
    from agent.pipeline.store import (
        OBSERVED_AT_MAX_AHEAD_SECONDS,
        OBSERVED_AT_MIN_EPOCH,
        ObservationRejected,
    )
except ImportError:  # pragma: no cover - tool must stay importable standalone
    OBSERVED_AT_MIN_EPOCH = 631_152_000.0  # 1990-01-01T00:00:00Z
    OBSERVED_AT_MAX_AHEAD_SECONDS = 86_400.0

    class ObservationRejected(ValueError):  # type: ignore[no-redef]
        """Fallback stand-in when the pipeline store is unavailable."""


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

VALID_MODES = {"search", "recent", "programs", "snapshot"}
VALID_SOURCES = {"ofac", "un", "all"}
VALID_ENTITY_TYPES = {"individual", "entity", "vessel", "aircraft", "all"}

#: Singleton entity the per-run roster observation is attached to. ``topic`` is
#: a seed entity type, so this needs no ontology extension.
_ROSTER_ENTITY_TYPE = "topic"
_ROSTER_ENTITY_NAME = "global_sanctions_roster"

#: Observation type for the unconditional per-run roster row.
_ROSTER_OBSERVATION_TYPE = "sanctions_roster_snapshot"


class PersistOutcome(NamedTuple):
    """Outcome of one persist pass.

    ``written`` counts distinct ``sanctions_listing`` rows in the store.
    ``rejected`` counts candidate records the store refused or that blew up
    mid-write.  Both travel together everywhere: reporting ``written`` alone
    is how "0 new listings" gets printed on a run that lost rows.
    """

    written: int
    rejected: int


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


def _parse_source_date(raw: Any) -> float | None:
    """Parse an ISO-ish source date to epoch seconds (UTC), or None.

    Returns None for anything unusable rather than falling back to "now" —
    the caller decides what an absent date means, because that decision
    differs per mode (see ``_observation_timestamp``).

    A *naive* value is forced to UTC.  ``UN LISTED_ON`` is a bare
    ``YYYY-MM-DD`` with no zone, and ``datetime.timestamp()`` on a naive
    datetime interprets it in the **host machine's** local timezone: on a
    +05:30 box ``"2025-06-16"`` came out as ``2025-06-15T18:30:00Z``, so every
    dated row landed on the previous calendar day and the same feed produced
    different ``observed_at`` values on different machines.  The live DB holds
    one of these (entity 5cfe20e6…, UN LISTED_ON 2025-06-16, stored at
    2025-06-15 18:30:00).  ``.replace("Z", "+00:00")`` only rescues strings
    that carry a zone to begin with.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


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
        "Mode 'snapshot' persists the full merged roster (first-seen gated) — the collection mode. "
        "Free, no API key required. OFAC ~19,400 entries, UN ~1,000 entries."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["search", "recent", "programs", "snapshot"],
                "default": "search",
                "description": (
                    "search = find entities by name. "
                    "recent = recently listed/updated (UN only has dates). "
                    "programs = overview of sanctions programs. "
                    "snapshot = persist the full roster, first-seen gated "
                    "(ignores days_back/limit; OFAC has no dates so any date "
                    "window silently discards all 19k of its records)."
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

    def _persist_entities(
        self,
        results: list[dict[str, Any]],
        *,
        snapshot: bool = False,
    ) -> PersistOutcome:
        """Register sanctioned entities and create country links.

        Returns :class:`PersistOutcome` — how many ``sanctions_listing``
        observations were written, and how many records the store refused.
        Both numbers are reported by the caller; a rejected record is never
        silently folded into "nothing new happened", which is the failure
        mode this whole module exists to stop.

        With *snapshot* true the write is **first-seen gated**: a record whose
        source uid already resolves to an entity is skipped outright.  That is
        what keeps a full-roster mode from writing 19k rows every single day —
        after the one-time backfill the row count equals "new designations
        since the last run", which is the actual signal.
        """
        if self._store is None or entity_id_from_key is None:
            return PersistOutcome(0, 0)
        if not results:
            return PersistOutcome(0, 0)
        try:
            return self._persist_entities_inner(results, snapshot=snapshot)
        except Exception:
            # Per-record failures are handled inside the loop; reaching here
            # means the batch itself could not run (dead connection, bad
            # store).  Report every candidate as unwritten rather than 0/0.
            log.exception("Entity persistence failed for the whole batch")
            return PersistOutcome(0, len(results))

    @staticmethod
    def _observation_timestamp(
        rec: dict[str, Any],
        *,
        now_ts: float,
    ) -> tuple[float, str]:
        """Resolve ``observed_at`` for a listing observation.

        Returns ``(timestamp, provenance)``.  *provenance* is stored in the
        row's metadata so a downstream consumer can tell a real designation
        date from a collection-time stand-in — day one of the backfill writes
        ~19k OFAC rows stamped with collection time, and without this marker
        they are indistinguishable from 19k designations in the same second.

        The source's own listing date is the honest timestamp **when it is
        usable**; otherwise we fall back to collection time.  "Usable" means
        inside the store's accepted range ``[OBSERVED_AT_MIN_EPOCH, now]`` —
        the guard is symmetric and mode-independent on purpose:

        * The old version guarded only the *future* side, and only in
          snapshot mode.  A single source record carrying a placeholder date
          like ``1899-01-01`` therefore reached the store, which raised
          ``ObservationRejected`` and aborted the rest of a 20k-record batch.
        * This is a fallback, not a clamp of a *valid* date: the store's
          guard is untouched and still rejects anything out of range that
          reaches it.  What changes is that one malformed source date can no
          longer zero a collection run, and the substitution is recorded in
          the row instead of being invisible.

        Preference order is ``listed_date or last_updated`` — the same order
        ``_execute_recent``'s filter uses, so a record can never be stored
        under a timestamp older than the one that admitted it.
        """
        raw = rec.get("listed_date") or rec.get("last_updated")
        if not (isinstance(raw, str) and raw.strip()):
            return now_ts, "collection_time"
        parsed = _parse_source_date(raw)
        if parsed is None:
            return now_ts, "unparsed_source_date"
        if parsed < OBSERVED_AT_MIN_EPOCH or parsed > now_ts:
            # Implausibly old, or dated in the future — bad source data, not
            # a fact. Falling back keeps one bad row from aborting the batch.
            return now_ts, "out_of_range_source_date"
        return parsed, "source_listed_date"

    def _persist_one(
        self,
        rec: dict[str, Any],
        *,
        store: PipelineStore,
        now_ts: float,
        snapshot: bool,
        seen: set[str],
        provenance_counts: dict[str, int],
    ) -> Any | None:
        """Persist a single record. Returns its observation row id, or None if skipped.

        Raises whatever the store raises — the caller counts the failure and
        moves on to the next record.
        """
        _SDN_TYPE_MAP = {
            "individual": "person",
            "entity": "organization",
            "vessel": "vessel",
            "aircraft": "organization",  # no aircraft entity type; treat as org
        }

        name = (rec.get("name") or "").strip()
        if not name:
            return None

        alias_source = f"sanctions_{rec.get('source', 'unknown')}"
        source_id = str(rec.get("entity_id") or "").strip()

        if snapshot:
            # No source uid → no stable key → it would be "new" on every
            # run forever. Skip rather than write an unbounded duplicate.
            if not source_id:
                return None
            if store.resolve_entity(alias_source, source_id) is not None:
                return None

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

        # ── Program → country links ──
        # Before the observation, because the alias below is the first-seen
        # gate's key: nothing that can still fail may run after it.
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

        # Observation
        ts, provenance = self._observation_timestamp(rec, now_ts=now_ts)
        provenance_counts[provenance] = provenance_counts.get(provenance, 0) + 1

        row_id = store.store_entity_observation(
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
            # metadata_json is NOT part of OBSERVATION_UNIQUE_KEY, so this
            # costs nothing in dedup terms. Without it, a backfill row
            # stamped with collection time is indistinguishable from a
            # designation that genuinely happened today.
            metadata={
                "observed_at_source": provenance,
                "source_date_raw": rec.get("listed_date") or rec.get("last_updated"),
                "first_seen_backfill": bool(snapshot),
            },
        )

        # Source-specific alias — written for EVERY record, not only the
        # first one that maps to this canonical entity. Two OFAC uids can
        # normalise to the same name; if only the first got an alias the
        # second would never resolve, so the first-seen gate above would
        # re-admit it on every single run.
        #
        # Written *after* the observation, deliberately. The alias is what the
        # first-seen gate reads, so writing it first buries a record whose
        # observation then fails: it resolves forever after, is skipped on
        # every future run, and never gets its row.
        if source_id:
            store.add_entity_alias(eid, alias_source, source_id)

        return row_id

    def _persist_entities_inner(
        self,
        results: list[dict[str, Any]],
        *,
        snapshot: bool = False,
    ) -> PersistOutcome:
        assert self._store is not None  # noqa: S101
        store = self._store
        # time.time(), not datetime.now(UTC).timestamp(): the latter truncates
        # to microseconds, so a caller that sampled time.time() immediately
        # before this call can observe a timestamp fractionally *behind* it.
        now_ts = time.time()
        # Distinct store row ids, not call count. Two source uids can normalise
        # to the same entity with an identical payload, and the store collapses
        # those onto one row — counting calls would report rows that do not
        # exist, which is the exact kind of "green but empty" claim this whole
        # change is here to stop.
        written_rows: set[Any] = set()
        rejected = 0
        provenance_counts: dict[str, int] = {}

        seen: set[str] = set()
        for rec in results:
            try:
                row_id = self._persist_one(
                    rec,
                    store=store,
                    now_ts=now_ts,
                    snapshot=snapshot,
                    seen=seen,
                    provenance_counts=provenance_counts,
                )
            except ObservationRejected:
                # One refused record must not abort the other 20,000. The
                # refusal is counted and surfaced by the caller, never
                # swallowed into a green "0 new listings".
                rejected += 1
                log.warning(
                    "sanctions_monitor: store refused record %s/%s",
                    rec.get("source"),
                    rec.get("entity_id"),
                    exc_info=True,
                )
                continue
            except Exception:
                rejected += 1
                log.exception(
                    "sanctions_monitor: failed to persist record %s/%s",
                    rec.get("source"),
                    rec.get("entity_id"),
                )
                continue
            if row_id is not None:
                written_rows.add(row_id)

        substituted = sum(v for k, v in provenance_counts.items() if k != "source_listed_date")
        if substituted:
            log.info(
                "sanctions_monitor: %d/%d listing rows stamped with collection time rather than a source date (%s)",
                substituted,
                sum(provenance_counts.values()),
                provenance_counts,
            )
        if rejected:
            log.error(
                "sanctions_monitor: %d of %d candidate records were not persisted",
                rejected,
                len(results),
            )

        return PersistOutcome(len(written_rows), rejected)

    # ------------------------------------------------------------------
    # Roster snapshot observation (the row that can never dedup away)
    # ------------------------------------------------------------------

    def _persist_roster_snapshot(
        self,
        *,
        source_stats: dict[str, dict[str, Any]],
        new_listings: int,
        rejected: int,
        complete: bool,
        observed_at: float,
    ) -> tuple[bool, str | None]:
        """Write one roster-level observation for this run.

        Returns ``(written, error)``.  The error text travels back to the
        caller instead of dying in a log line: ``_execute_snapshot`` puts it
        in the ``ToolResult``, which is the string ``ToolOperator.execute``
        raises, so whoever reads the DAG failure learns *why* the run was
        not diffable.

        Unconditional and stamped with *collection* time, so no two runs can
        produce the same idempotency key
        (``entity_id, source_tool, observation_type, observed_at, value_json``
        — see ``agent/pipeline/store.py``).  That is the whole point: the node
        physically cannot report "completed" with zero rows written again.

        That guarantee only holds because *failing* to write this row fails
        the run — see ``_execute_snapshot``.  While the failure was merely
        logged, the single path that writes nothing at all (no store
        injected) was also the single path excluded from the warning, and a
        ``database is locked`` here — live-reachable, since ``DAGExecutor``
        runs nodes in worker threads against a database a backfill may be
        writing — produced a green run with no diffable row.

        The payload carries each source's record count and a content hash of
        the fetched body.  Consecutive hashes are what make add/remove diffs
        computable — but the diffing itself is Layer 2's job, not this tool's.

        It also carries ``complete`` and ``rejected``.  A diff that ignores
        ``complete`` will read a single failed download as thousands of
        delistings, and ``rejected > 0`` means ``new_listings`` undercounts.
        """
        if self._store is None:
            return False, "no pipeline store configured"
        if entity_id_from_key is None:
            return False, "agent.pipeline.entity unavailable"
        try:
            store = self._store
            roster_eid = entity_id_from_key(_ROSTER_ENTITY_TYPE, _ROSTER_ENTITY_NAME)
            store.register_entity(
                entity_type=_ROSTER_ENTITY_TYPE,
                canonical_name=_ROSTER_ENTITY_NAME,
                entity_id=roster_eid,
                metadata={"source": "sanctions_monitor", "kind": "roster_singleton"},
            )
            store.store_entity_observation(
                entity_id=roster_eid,
                source_tool="sanctions_monitor",
                observed_at=observed_at,
                observation_type=_ROSTER_OBSERVATION_TYPE,
                depth_level=1,
                value={
                    "sources": source_stats,
                    "total_records": sum(int(s.get("records", 0)) for s in source_stats.values()),
                    "new_listings": new_listings,
                    # ``complete`` is the flag a downstream diff must honour.
                    # False means at least one requested source did not
                    # answer, so this row is a fragment of the roster, not the
                    # roster. ``rejected`` means rows were lost on the write
                    # side, so ``new_listings`` is a floor, not a count.
                    "complete": complete,
                    "rejected": rejected,
                },
            )
        except Exception as exc:
            log.exception("Roster snapshot persistence failed — run is not diffable")
            return False, f"{type(exc).__name__}: {exc}"
        return True, None

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

        if mode == "snapshot":
            return self._execute_snapshot(
                source=source,
                entity_type=entity_type,
                program=program.strip(),
            )

        # programs
        return self._execute_programs(source=source)

    # ------------------------------------------------------------------
    # snapshot mode — the collection mode
    # ------------------------------------------------------------------

    def _execute_snapshot(
        self,
        *,
        source: str,
        entity_type: str,
        program: str,
    ) -> ToolResult:
        """Persist the full merged roster, first-seen gated, plus a roster row.

        Deliberately takes no ``days_back`` and no ``limit``: both of those
        are what made the DAG node discard 95% of what it fetched.
        """
        records, stats, error = self._get_records(source)
        if error:
            return ToolResult(success=False, output=error)

        # A source that was requested and failed makes this run a *partial*
        # snapshot, never a roster. Diffing a partial against the previous
        # full one reads as a mass delisting followed by a mass
        # re-designation — noise that is indistinguishable from the exact
        # signal this tool exists to produce.
        failed_sources = {name: st["error"] for name, st in stats.items() if st.get("error")}
        complete = not failed_sources

        candidates = records
        if entity_type != "all":
            candidates = [r for r in candidates if r["type"] == entity_type]
        if program:
            prog_lower = program.lower()
            candidates = [r for r in candidates if any(prog_lower in p.lower() for p in r["programs"])]

        outcome = self._persist_entities(candidates, snapshot=True)
        collected_at = time.time()
        roster_written, roster_error = self._persist_roster_snapshot(
            source_stats=stats,
            new_listings=outcome.written,
            rejected=outcome.rejected,
            complete=complete,
            observed_at=collected_at,
        )

        src_summary = ", ".join(
            f"{name.upper()} {s['records']} records (sha256 {str(s.get('body_sha256') or '?')[:12]})"
            + (f" [FAILED: {s['error']}]" if s.get("error") else "")
            for name, s in sorted(stats.items())
        )
        lines = [
            f"Sanctions snapshot: {len(records)} roster records fetched — {src_summary}.",
            f"  New listings persisted this run: {outcome.written}",
            f"  Records the store refused: {outcome.rejected}",
            f"  Roster observation written: {'yes' if roster_written else 'no'}",
        ]
        if not roster_written:
            # No `and self._store is not None` clause. That clause excluded the
            # one path that writes *nothing at all* — snapshot mode with no
            # store — from the only warning about writing nothing.
            lines.append(
                f"  FAILED: roster observation not written ({roster_error}) — "
                "this run is not diffable and does not count as collected."
            )
        if failed_sources:
            lines.append(
                "  FAILED: partial snapshot — "
                + "; ".join(f"{n.upper()}: {e}" for n, e in sorted(failed_sources.items()))
                + ". Marked incomplete; do not diff against a full roster."
            )
        if outcome.rejected:
            lines.append(
                f"  FAILED: {outcome.rejected} candidate record(s) were not persisted — "
                "the new-listing count above understates this run."
            )

        # NOTE: the record list is deliberately NOT returned in ``data``.
        # 19k records would land in the pipeline_data envelope on every run;
        # the roster observation is where this run's facts live.
        return ToolResult(
            # Reporting success on a partial fetch or a lossy write is the
            # failure mode itself: the DAG node has retries=2 and they can
            # only fire if the tool admits the run did not complete.
            #
            # ``roster_written`` is in here because the module docstring
            # claims that row is what makes the node unable to be green with
            # nothing written — a claim that is only true if not writing it
            # is a failure. Caveat for whoever reads the retry: the listing
            # rows of a run that died on the roster write are already in the
            # store, so the retry's roster row reports new_listings=0 for
            # them (the first-seen gate skips them). Their own
            # ``first_seen_backfill`` metadata, not that count, is the record
            # that they were written.
            success=complete and outcome.rejected == 0 and roster_written,
            output="\n".join(lines),
            data={
                "mode": "snapshot",
                "sources": stats,
                "complete": complete,
                "failed_sources": sorted(failed_sources),
                "total_records": len(records),
                "candidates": len(candidates),
                "new_listings": outcome.written,
                "rejected": outcome.rejected,
                "roster_observation_written": roster_written,
                "roster_error": roster_error,
                "collected_at": collected_at,
            },
        )

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
        records, _stats, error = self._get_records(source)
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

        outcome = self._persist_entities(matched)
        if outcome.rejected:
            lines.append(f"  WARNING: {outcome.rejected} record(s) could not be persisted.")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "query": query,
                "results": matched,
                "count": len(matched),
                "persisted": outcome.written,
                "rejected": outcome.rejected,
            },
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
        records, _stats, error = self._get_records(source)
        if error:
            return ToolResult(success=False, output=error)

        cutoff = (datetime.now(UTC) - timedelta(days=days_back)).strftime("%Y-%m-%d")

        recent: list[dict[str, Any]] = []
        for rec in records:
            # Preference order MUST match the persist path's
            # (``_observation_timestamp``: listed_date or last_updated).
            # It used to be the other way round, so a record updated recently
            # but listed years ago was admitted by ``last_updated`` and then
            # stored under the older ``listed_date`` — the live DB holds one:
            # entity 5cfe20e6…, listed 2025-06-16 / updated 2026-08-14,
            # stored at observed_at 2025-06-15.
            date_str = rec.get("listed_date") or rec.get("last_updated")
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

        outcome = self._persist_entities(recent)
        if outcome.rejected:
            lines.append(f"  WARNING: {outcome.rejected} record(s) could not be persisted.")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "results": recent,
                "count": len(recent),
                "days_back": days_back,
                "persisted": outcome.written,
                "rejected": outcome.rejected,
            },
        )

    # ------------------------------------------------------------------
    # programs mode
    # ------------------------------------------------------------------

    def _execute_programs(self, *, source: str) -> ToolResult:
        records, _stats, error = self._get_records(source)
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

    def _get_records(
        self,
        source: str,
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], str | None]:
        """Fetch and merge records from requested sources.

        Returns ``(records, per_source_stats, error)``.  ``per_source_stats``
        maps a source name to ``{"records": n, "body_sha256": hex}`` and is
        what the roster observation is built from.

        A source that was **requested but failed** gets an entry too:
        ``{"records": 0, "body_sha256": None, "error": "..."}``.  It used to
        get no entry at all whenever the other source returned something, so
        a 503 on the OFAC download produced a roster row that looked like a
        complete snapshot of a world where 19,391 designations had just been
        lifted — and ``success=True``, so the node's ``retries=2`` never
        fired.  ``error`` is only returned when *every* requested source
        failed; the caller decides what a partial fetch means for its mode.
        """
        records: list[dict[str, Any]] = []
        stats: dict[str, dict[str, Any]] = {}
        errors: list[str] = []

        if source in ("ofac", "all"):
            ofac, ofac_hash, err = self._fetch_ofac()
            if err:
                errors.append(f"OFAC: {err}")
                stats["ofac"] = {"records": 0, "body_sha256": None, "error": err}
            else:
                records.extend(ofac)
                stats["ofac"] = {"records": len(ofac), "body_sha256": ofac_hash}

        if source in ("un", "all"):
            un, un_hash, err = self._fetch_un()
            if err:
                errors.append(f"UN: {err}")
                stats["un"] = {"records": 0, "body_sha256": None, "error": err}
            else:
                records.extend(un)
                stats["un"] = {"records": len(un), "body_sha256": un_hash}

        # If ALL sources failed, return error
        if not records and errors:
            return [], stats, "; ".join(errors)

        return records, stats, None

    @staticmethod
    def _unpack_cached(cached: Any) -> tuple[list[dict[str, Any]], str | None] | None:
        """Read a cache entry in either the current or the legacy shape.

        Entries written before the roster snapshot existed are a bare list of
        records with no body hash; returning ``None`` for the hash there is
        correct — we genuinely do not know it — and is preferable to evicting
        a cache the DAG shares with other modes.
        """
        if isinstance(cached, dict) and "records" in cached:
            recs = cached.get("records")
            if isinstance(recs, list):
                return recs, cached.get("body_sha256")
            return None
        if isinstance(cached, list):
            return cached, None
        return None

    def _fetch_ofac(self) -> tuple[list[dict[str, Any]], str | None, str | None]:
        """Download and parse OFAC SDN CSV. Returns (records, body_sha256, error)."""
        cache_key = {"source": "ofac_sdn"}
        if self._cache:
            cached = self._cache.get("sanctions_monitor", cache_key)
            unpacked = self._unpack_cached(cached)
            if unpacked is not None:
                return unpacked[0], unpacked[1], None

        try:
            resp = httpx.get(
                _OFAC_SDN_URL,
                timeout=_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": _UA},
            )
            resp.raise_for_status()
        except httpx.TimeoutException:
            return [], None, "OFAC SDN download timed out (file is ~5MB)."
        except httpx.HTTPStatusError as exc:
            return [], None, f"OFAC SDN HTTP {exc.response.status_code}."
        except httpx.ConnectError:
            return [], None, "OFAC SDN connection failed."

        body_hash = hashlib.sha256(resp.content).hexdigest()
        records = _parse_ofac_csv(resp.text)
        if not records:
            return [], body_hash, "OFAC SDN: parsed 0 records (unexpected)."

        if self._cache:
            self._cache.put(
                "sanctions_monitor",
                cache_key,
                {"records": records, "body_sha256": body_hash},
            )

        return records, body_hash, None

    def _fetch_un(self) -> tuple[list[dict[str, Any]], str | None, str | None]:
        """Download and parse UN SC consolidated XML. Returns (records, body_sha256, error)."""
        cache_key = {"source": "un_consolidated"}
        if self._cache:
            cached = self._cache.get("sanctions_monitor", cache_key)
            unpacked = self._unpack_cached(cached)
            if unpacked is not None:
                return unpacked[0], unpacked[1], None

        try:
            resp = httpx.get(
                _UN_XML_URL,
                timeout=_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": _UA},
            )
            resp.raise_for_status()
        except httpx.TimeoutException:
            return [], None, "UN SC XML download timed out."
        except httpx.HTTPStatusError as exc:
            return [], None, f"UN SC HTTP {exc.response.status_code}."
        except httpx.ConnectError:
            return [], None, "UN SC connection failed."

        body_hash = hashlib.sha256(resp.content).hexdigest()
        records = _parse_un_xml(resp.text)
        if not records:
            return [], body_hash, "UN SC: parsed 0 records (unexpected)."

        if self._cache:
            self._cache.put(
                "sanctions_monitor",
                cache_key,
                {"records": records, "body_sha256": body_hash},
            )

        return records, body_hash, None
