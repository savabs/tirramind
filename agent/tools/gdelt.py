"""
Tool: GDELT — Global Geopolitical Event Surveillance

Fetches structured geopolitical events from GDELT's raw event files.
Every 15 minutes, GDELT publishes ~1,200 events covering conflicts,
sanctions, protests, diplomatic actions, and military posture worldwide,
coded with CAMEO taxonomy, geocoordinates, and intensity scores.

Secondary mode: keyword/theme article search via DOC API.

This is the "poor man's Palantir" — the same structured event
intelligence governments pay millions for, available as open data files.
Zero cost, no authentication required.

Primary files: http://data.gdeltproject.org/gdeltv2/
DOC API: https://api.gdeltproject.org/api/v2/doc/doc
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

try:
    from agent.pipeline.entity import entity_id_from_key
except ImportError:  # pragma: no cover
    entity_id_from_key = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

_GDELT_BASE = "http://data.gdeltproject.org/gdeltv2"
_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
_BATCH_INTERVAL = 15  # minutes between GDELT batches
_BATCH_DELAY = 0.2  # seconds between batch downloads (politeness)

# ── CAMEO root event codes → human-readable labels ──────────────────
CAMEO_ROOT_CODES: dict[str, str] = {
    "01": "Make Public Statement",
    "02": "Appeal",
    "03": "Express Intent to Cooperate",
    "04": "Consult",
    "05": "Engage in Diplomatic Cooperation",
    "06": "Engage in Material Cooperation",
    "07": "Provide Aid",
    "08": "Yield",
    "09": "Investigate",
    "10": "Demand",
    "11": "Disapprove",
    "12": "Reject",
    "13": "Threaten",
    "14": "Protest",
    "15": "Exhibit Military Posture",
    "16": "Reduce Relations",
    "17": "Coerce",
    "18": "Assault",
    "19": "Fight",
    "20": "Unconventional Mass Violence",
}

QUAD_LABELS: dict[int, str] = {
    1: "Verbal Cooperation",
    2: "Material Cooperation",
    3: "Verbal Conflict",
    4: "Material Conflict",
}

# ── Column indices in the 61-column GDELT export CSV ────────────────
# Tab-separated, no header. Full schema: docs/research/gdelt.md
_COL = {
    "global_event_id": 0,
    "day": 1,
    "fraction_date": 4,
    "actor1_code": 5,
    "actor1_name": 6,
    "actor1_country": 7,
    "actor1_type": 12,
    "actor2_code": 15,
    "actor2_name": 16,
    "actor2_country": 17,
    "actor2_type": 22,
    "is_root_event": 25,
    "event_code": 26,
    "event_base_code": 27,
    "event_root_code": 28,
    "quad_class": 29,
    "goldstein": 30,
    "num_mentions": 31,
    "num_sources": 32,
    "num_articles": 33,
    "avg_tone": 34,
    "action_geo_type": 51,
    "action_geo_fullname": 52,
    "action_geo_country": 53,
    "action_geo_lat": 56,
    "action_geo_long": 57,
    "date_added": 59,
    "source_url": 60,
}


class GDELTTool(Tool):
    name = "gdelt"

    description = (
        "Fetch structured geopolitical events from GDELT — the world's largest "
        "open event database. Monitors conflicts, sanctions, protests, military "
        "actions, and diplomatic events across 300+ CAMEO event types, updated "
        "every 15 minutes. Use 'events' mode for structured surveillance or "
        "'articles' mode for keyword-based article search. Zero cost, no auth."
    )

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "description": (
                    "Data mode: 'events' for structured geopolitical events from "
                    "raw GDELT files, 'articles' for keyword search via DOC API."
                ),
                "enum": ["events", "articles"],
                "default": "events",
            },
            "hours_back": {
                "type": "integer",
                "description": (
                    "Events mode only. How many hours of data to fetch (1-24). "
                    "Each hour = 4 batches of ~1,200 events. Default: 1."
                ),
                "default": 1,
            },
            "country_filter": {
                "type": "string",
                "description": (
                    "Events mode only. FIPS 2-letter country code to filter by "
                    "action location (e.g., 'IR' for Iran, 'UP' for Ukraine, "
                    "'CH' for China, 'RS' for Russia)."
                ),
                "default": "",
            },
            "min_goldstein": {
                "type": "number",
                "description": (
                    "Events mode only. Return only events with Goldstein score "
                    "≤ this value. E.g., -5.0 for significant conflict events. "
                    "Range: -10 (war) to +10 (cooperation)."
                ),
            },
            "quad_class": {
                "type": "string",
                "description": (
                    "Events mode only. Filter by event type: 'conflict' "
                    "(verbal+material conflict), 'cooperation', or 'all'. "
                    "Default: 'conflict'."
                ),
                "enum": ["conflict", "cooperation", "all"],
                "default": "conflict",
            },
            "event_codes": {
                "type": "string",
                "description": (
                    "Events mode only. Comma-separated CAMEO root codes to include "
                    "(e.g., '18,19,20' for assault/fight/mass violence)."
                ),
                "default": "",
            },
            "query": {
                "type": "string",
                "description": (
                    "Articles mode only (required). Search keywords for GDELT "
                    "DOC API. Supports: 'theme:TERROR', 'sourcecountry:US', "
                    "'tone<-5', 'near20:\"trump putin\"'."
                ),
                "default": "",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum events or articles to return. Default: 50.",
                "default": 50,
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
    # Core execution
    # ------------------------------------------------------------------

    def execute(
        self,
        *,
        mode: str = "events",
        hours_back: int = 1,
        country_filter: str = "",
        min_goldstein: float | None = None,
        quad_class: str = "conflict",
        event_codes: str = "",
        query: str = "",
        limit: int = 50,
        _backfill: bool = False,
        days_back: int = 730,
        sample_every_days: int = 7,
        **_: Any,
    ) -> ToolResult:
        if _backfill:
            return self._execute_backfill(
                days_back=max(1, days_back),
                sample_every_days=max(1, sample_every_days),
            )

        mode = mode.lower().strip()
        if mode not in ("events", "articles"):
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Use 'events' or 'articles'.",
            )

        if mode == "articles":
            return self._execute_articles(query=query, limit=limit)

        return self._execute_events(
            hours_back=hours_back,
            country_filter=country_filter.strip().upper(),
            min_goldstein=min_goldstein,
            quad_class=quad_class.lower().strip(),
            event_codes=event_codes.strip(),
            limit=max(1, min(limit, 500)),
        )

    # ------------------------------------------------------------------
    # Historical backfill mode
    # ------------------------------------------------------------------

    def _execute_backfill(self, days_back: int, sample_every_days: int) -> ToolResult:
        """Sample one 15-min GDELT batch per sample_every_days going back days_back days.

        Downloads evenly-spaced historical event batches and persists country
        entities with the correct historical observed_at timestamp from the
        event's own date field.  Cached batches are not re-fetched.
        """
        timestamps = self._compute_historical_sample_timestamps(days_back, sample_every_days)
        if not timestamps:
            return ToolResult(
                success=True,
                output="GDELT backfill: no timestamps to sample",
                data={"batches": 0, "total_events": 0},
            )

        total_events = 0
        batches_fetched = 0

        with httpx.Client(timeout=20, follow_redirects=True) as client:
            for ts in timestamps:
                cache_key = {"batch": ts}
                raw: str | None = None
                if self._cache:
                    raw = self._cache.get("gdelt_events", cache_key)

                if raw is None:
                    url = f"{_GDELT_BASE}/{ts}.export.CSV.zip"
                    try:
                        resp = client.get(url)
                        if resp.status_code == 404:
                            log.debug("GDELT historical batch %s not found — skipping", ts)
                            continue
                        resp.raise_for_status()
                        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                            names = zf.namelist()
                            if not names:
                                continue
                            raw = zf.read(names[0]).decode("utf-8", errors="replace")
                    except (
                        httpx.HTTPError,
                        zipfile.BadZipFile,
                        KeyError,
                        UnicodeDecodeError,
                    ) as exc:
                        log.warning("GDELT historical batch %s failed: %s", ts, exc)
                        continue

                    if self._cache and raw:
                        self._cache.put("gdelt_events", cache_key, raw)
                    time.sleep(_BATCH_DELAY)

                events = self._parse_events([raw])
                total_events += len(events)
                batches_fetched += 1
                try:
                    self._persist_entities(events)
                except Exception:
                    log.exception("GDELT backfill persist failed for ts %s (non-fatal)", ts)

        num_samples = len(timestamps)
        return ToolResult(
            success=True,
            output=(
                f"GDELT backfill: {batches_fetched}/{num_samples} batches fetched, "
                f"{total_events} events parsed, {days_back}d history "
                f"sampled every {sample_every_days}d"
            ),
            data={"batches": batches_fetched, "total_events": total_events},
        )

    @staticmethod
    def _compute_historical_sample_timestamps(days_back: int, sample_every_days: int) -> list[str]:
        """Return GDELT batch timestamps sampled every sample_every_days going back days_back.

        Uses noon UTC per sample so batches are almost certainly already archived.
        """
        now = datetime.now(UTC)
        samples: list[str] = []
        offset = days_back
        while offset > 0:
            target = now - timedelta(days=offset)
            target = target.replace(hour=12, minute=0, second=0, microsecond=0)
            # Round down to nearest 15-min boundary
            minute = (target.minute // 15) * 15
            target = target.replace(minute=minute)
            samples.append(target.strftime("%Y%m%d%H%M%S"))
            offset -= sample_every_days
        return samples

    # ------------------------------------------------------------------
    # Events mode
    # ------------------------------------------------------------------

    def _execute_events(
        self,
        *,
        hours_back: int,
        country_filter: str,
        min_goldstein: float | None,
        quad_class: str,
        event_codes: str,
        limit: int,
    ) -> ToolResult:
        hours_back = max(1, min(hours_back, 24))

        try:
            raw_batches = self._fetch_event_batches(hours_back)
        except Exception as exc:
            log.exception("GDELT event fetch failed")
            return ToolResult(success=False, output=f"GDELT fetch error: {exc}")

        if not raw_batches:
            return ToolResult(
                success=True,
                output=("No GDELT event batches available. The most recent batches may not be published yet."),
                data={"events": [], "summary": {}},
            )

        events = self._parse_events(raw_batches)
        total_raw = len(events)

        # ── Apply filters ────────────────────────────────────────────
        if quad_class == "conflict":
            events = [e for e in events if e["quad_class"] in (3, 4)]
        elif quad_class == "cooperation":
            events = [e for e in events if e["quad_class"] in (1, 2)]

        if country_filter:
            events = [e for e in events if e["location"]["country"].upper() == country_filter]

        if min_goldstein is not None:
            events = [e for e in events if e["goldstein"] is not None and e["goldstein"] <= min_goldstein]

        if event_codes:
            codes_set = {c.strip() for c in event_codes.split(",")}
            events = [e for e in events if e["event_root"] in codes_set]

        # ── Sort: most impactful first ───────────────────────────────
        events.sort(
            key=lambda e: (
                abs(e["goldstein"]) if e["goldstein"] is not None else 0,
                e["num_mentions"],
            ),
            reverse=True,
        )

        events = events[:limit]

        # ── Summary stats ────────────────────────────────────────────
        country_counts = Counter(e["location"]["country"] for e in events if e["location"]["country"])
        top_countries = country_counts.most_common(10)

        summary = {
            "total_events_fetched": total_raw,
            "filtered_count": len(events),
            "batches": len(raw_batches),
            "hours_back": hours_back,
            "top_countries": top_countries,
            "filters": {
                "quad_class": quad_class,
                "country": country_filter or "all",
                "min_goldstein": min_goldstein,
                "event_codes": event_codes or "all",
            },
        }

        if not events:
            return ToolResult(
                success=True,
                output=(
                    f"GDELT: {total_raw} events fetched from {len(raw_batches)} batches, but none matched filters."
                ),
                data={"events": [], "summary": summary},
            )

        # ── Format human-readable output ─────────────────────────────
        lines = [
            (
                f"GDELT — {len(events)} events "
                f"(from {total_raw} total, {len(raw_batches)} batches, "
                f"{hours_back}h lookback):"
            ),
            (f"Top countries: {', '.join(f'{c} ({n})' for c, n in top_countries[:5])}"),
            "",
        ]
        for i, e in enumerate(events[:30], 1):
            actor1 = e["actor1"]["name"] or "UNKNOWN"
            actor2 = e["actor2"]["name"] or "UNKNOWN"
            gs = f"G={e['goldstein']:+.1f}" if e["goldstein"] is not None else "G=?"
            loc = e["location"]["name"] or e["location"]["country"] or "?"
            lines.append(
                f"  {i}. {actor1} → {actor2} | "
                f"{e['event_description']} | {gs} | "
                f"{e['quad_label']} | {loc} | "
                f"mentions={e['num_mentions']}"
            )

        if len(events) > 30:
            lines.append(f"\n  ... and {len(events) - 30} more events (see data for full list)")

        # L2: entity persistence
        try:
            self._persist_entities(events)
        except Exception:
            log.exception("Entity persistence failed in events mode (non-fatal)")

        # L2: entity_ids in actor sub-dicts
        if entity_id_from_key is not None:
            for e in events:
                a1c = e["actor1"]["country"]
                a2c = e["actor2"]["country"]
                if a1c:
                    e["actor1"]["entity_id"] = entity_id_from_key("country", a1c)
                if a2c:
                    e["actor2"]["entity_id"] = entity_id_from_key("country", a2c)

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"events": events, "summary": summary},
        )

    # ------------------------------------------------------------------
    # Fetching event batches
    # ------------------------------------------------------------------

    def _fetch_event_batches(self, hours_back: int) -> list[str]:
        """Download GDELT 15-min event batches. Returns raw CSV text per batch."""
        num_batches = hours_back * 4
        timestamps = self._compute_batch_timestamps(num_batches)
        batches: list[str] = []

        with httpx.Client(timeout=15, follow_redirects=True) as client:
            for ts in timestamps:
                # Cache check — batches are immutable, long TTL is fine
                cache_key = {"batch": ts}
                if self._cache:
                    cached = self._cache.get("gdelt_events", cache_key)
                    if cached is not None:
                        batches.append(cached)
                        continue

                url = f"{_GDELT_BASE}/{ts}.export.CSV.zip"
                try:
                    resp = client.get(url)
                    if resp.status_code == 404:
                        log.debug("GDELT batch %s not yet published — skipping", ts)
                        continue
                    resp.raise_for_status()
                except httpx.HTTPError as exc:
                    log.warning("GDELT batch %s failed: %s", ts, exc)
                    continue

                # Extract CSV from zip
                try:
                    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                        names = zf.namelist()
                        if not names:
                            continue
                        csv_text = zf.read(names[0]).decode("utf-8", errors="replace")
                except (zipfile.BadZipFile, KeyError, UnicodeDecodeError) as exc:
                    log.warning("GDELT batch %s bad zip: %s", ts, exc)
                    continue

                if self._cache and csv_text:
                    self._cache.put("gdelt_events", cache_key, csv_text)

                batches.append(csv_text)
                time.sleep(_BATCH_DELAY)

        return batches

    @staticmethod
    def _compute_batch_timestamps(num_batches: int) -> list[str]:
        """Compute GDELT batch timestamps going backwards from now.

        Returns list of 'YYYYMMDDHHMMSS' strings. Starts one batch
        behind current time to avoid fetching a batch not yet published.
        """
        now = datetime.now(UTC)
        # Round down to nearest 15-min boundary
        minute = (now.minute // 15) * 15
        current = now.replace(minute=minute, second=0, microsecond=0)
        # Start one batch behind (most recent may not be published yet)
        current -= timedelta(minutes=_BATCH_INTERVAL)

        timestamps = []
        for i in range(num_batches):
            ts = current - timedelta(minutes=_BATCH_INTERVAL * i)
            timestamps.append(ts.strftime("%Y%m%d%H%M%S"))
        return timestamps

    # ------------------------------------------------------------------
    # Parsing events
    # ------------------------------------------------------------------

    def _parse_events(self, raw_batches: list[str]) -> list[dict[str, Any]]:
        """Parse raw CSV batches into structured event dicts."""
        events: list[dict[str, Any]] = []

        for batch in raw_batches:
            for line in batch.strip().split("\n"):
                if not line:
                    continue
                cols = line.split("\t")
                if len(cols) < 61:
                    continue

                quad = _safe_int(cols[_COL["quad_class"]])
                goldstein = _safe_float(cols[_COL["goldstein"]])
                event_root = cols[_COL["event_root_code"]].strip()

                events.append(
                    {
                        "id": cols[_COL["global_event_id"]].strip(),
                        "date": cols[_COL["day"]].strip(),
                        "actor1": {
                            "name": cols[_COL["actor1_name"]].strip() or None,
                            "country": cols[_COL["actor1_country"]].strip(),
                            "type": cols[_COL["actor1_type"]].strip(),
                        },
                        "actor2": {
                            "name": cols[_COL["actor2_name"]].strip() or None,
                            "country": cols[_COL["actor2_country"]].strip(),
                            "type": cols[_COL["actor2_type"]].strip(),
                        },
                        "event_code": cols[_COL["event_code"]].strip(),
                        "event_root": event_root,
                        "event_description": CAMEO_ROOT_CODES.get(event_root, f"Code {event_root}"),
                        "quad_class": quad,
                        "quad_label": QUAD_LABELS.get(quad, "Unknown"),
                        "goldstein": goldstein,
                        "num_mentions": _safe_int(cols[_COL["num_mentions"]]) or 0,
                        "num_sources": _safe_int(cols[_COL["num_sources"]]) or 0,
                        "avg_tone": _safe_float(cols[_COL["avg_tone"]]),
                        "location": {
                            "name": cols[_COL["action_geo_fullname"]].strip(),
                            "country": cols[_COL["action_geo_country"]].strip(),
                            "lat": _safe_float(cols[_COL["action_geo_lat"]]),
                            "lon": _safe_float(cols[_COL["action_geo_long"]]),
                        },
                        "source_url": cols[_COL["source_url"]].strip(),
                    }
                )

        return events

    # ------------------------------------------------------------------
    # Articles mode
    # ------------------------------------------------------------------

    def _execute_articles(self, *, query: str, limit: int) -> ToolResult:
        """Handle articles mode — keyword search via GDELT DOC API."""
        query = query.strip()
        if not query:
            return ToolResult(
                success=False,
                output="Articles mode requires a 'query' parameter.",
            )

        limit = max(1, min(limit, 250))

        try:
            articles = self._fetch_articles(query, limit)
        except Exception as exc:
            log.exception("GDELT DOC API failed")
            return ToolResult(
                success=False,
                output=f"GDELT article search error: {exc}",
            )

        if not articles:
            return ToolResult(
                success=True,
                output=f"No articles found for query: {query}",
                data={"articles": []},
            )

        lines = [f"GDELT Articles — {len(articles)} results for '{query}':\n"]
        for i, art in enumerate(articles[:30], 1):
            tone = art.get("tone")
            tone_str = f"tone={tone:+.1f}" if tone is not None else ""
            lines.append(f"  {i}. {art['title']}\n     {art['domain']} | {art['seendate'][:10]} | {tone_str}")

        if len(articles) > 30:
            lines.append(f"\n  ... and {len(articles) - 30} more articles")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"articles": articles},
        )

    def _fetch_articles(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Search GDELT DOC API. Returns list of article dicts."""
        cache_key = {"query": query, "limit": limit}
        if self._cache:
            cached = self._cache.get("gdelt_articles", cache_key)
            if cached is not None:
                log.debug("Cache hit for GDELT articles query=%s", query)
                return cached

        with httpx.Client(timeout=15, follow_redirects=True) as client:
            resp = client.get(
                _DOC_API,
                params={
                    "query": query,
                    "mode": "artlist",
                    "format": "json",
                    "maxrecords": str(min(limit, 250)),
                    "timespan": "1w",
                    "sort": "datedesc",
                },
            )
            if resp.status_code == 429:
                raise RuntimeError("GDELT rate limited (429). Try again in a minute.")
            resp.raise_for_status()
            data = resp.json()

        articles = []
        for art in data.get("articles", []):
            articles.append(
                {
                    "title": art.get("title", ""),
                    "url": art.get("url", ""),
                    "seendate": art.get("seendate", ""),
                    "domain": art.get("domain", ""),
                    "language": art.get("language", ""),
                    "sourcecountry": art.get("sourcecountry", ""),
                    "tone": _safe_float(art.get("tone")),
                }
            )

        if self._cache and articles:
            self._cache.put("gdelt_articles", cache_key, articles)

        return articles

    # ------------------------------------------------------------------
    # L2 Entity Persistence
    # ------------------------------------------------------------------

    def _persist_entities(self, events: list[dict[str, Any]]) -> None:
        """Persist country entities from GDELT event actor pairs."""
        if self._store is None or entity_id_from_key is None:
            return
        self._persist_entities_inner(events)

    def _persist_entities_inner(self, events: list[dict[str, Any]]) -> None:
        seen: set[str] = set()
        now_ts = datetime.now(UTC).timestamp()
        for ev in events:
            event_id = ev.get("id", "")
            raw_date = ev.get("date", "")
            # Convert YYYYMMDD string to Unix timestamp (noon UTC).
            # observed_at must be a float — storing a date string was a latent bug.
            observed_ts: float
            if raw_date and len(raw_date) == 8:
                try:
                    dt = datetime(
                        int(raw_date[:4]),
                        int(raw_date[4:6]),
                        int(raw_date[6:8]),
                        12,
                        0,
                        0,
                        tzinfo=UTC,
                    )
                    observed_ts = dt.timestamp()
                except ValueError:
                    observed_ts = now_ts
            else:
                observed_ts = now_ts

            for role, actor_key, counterpart_key in [
                ("initiator", "actor1", "actor2"),
                ("target", "actor2", "actor1"),
            ]:
                actor = ev.get(actor_key, {})
                counterpart = ev.get(counterpart_key, {})
                country = actor.get("country", "").strip()
                if not country:
                    continue

                eid = entity_id_from_key("country", country)

                if eid not in seen:
                    seen.add(eid)
                    name = actor.get("name") or country
                    self._store.register_entity(
                        entity_type="country",
                        canonical_name=name,
                        entity_id=eid,
                        metadata={
                            "fips_code": country,
                            "actor_type": actor.get("type", ""),
                        },
                    )
                    self._store.add_entity_alias(eid, "fips", country)

                self._store.store_entity_observation(
                    entity_id=eid,
                    source_tool="gdelt",
                    observed_at=observed_ts,
                    observation_type="geopolitical_event",
                    value={
                        "event_id": event_id,
                        "counterpart_country": counterpart.get("country", ""),
                        "event_root": ev.get("event_root", ""),
                        "event_description": ev.get("event_description", ""),
                        "goldstein": ev.get("goldstein"),
                        "quad_class": ev.get("quad_class"),
                        "role": role,
                        "num_mentions": ev.get("num_mentions", 0),
                        "location": ev.get("location", {}).get("country", ""),
                    },
                    depth_level=2,
                )

            # ── Link actor1 ↔ actor2 countries ──
            c1 = ev.get("actor1", {}).get("country", "").strip()
            c2 = ev.get("actor2", {}).get("country", "").strip()
            if c1 and c2 and c1 != c2:
                self._store.link_entities(
                    entity_id_a=entity_id_from_key("country", c1),
                    entity_id_b=entity_id_from_key("country", c2),
                    link_type="event_involves",
                    source="gdelt",
                    confidence=0.9,
                    metadata={
                        "event_id": event_id,
                        "event_root": ev.get("event_root", ""),
                    },
                )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _safe_float(val: Any) -> float | None:
    """Convert value to float, returning None on failure."""
    if val is None or val == "":
        return None
    try:
        f = float(val)
        return f if f == f else None  # NaN check
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> int | None:
    """Convert value to int, returning None on failure."""
    if val is None or val == "":
        return None
    try:
        return int(float(val))  # handle "3.0" style strings
    except (ValueError, TypeError):
        return None
