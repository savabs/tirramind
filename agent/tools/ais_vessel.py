"""
Tool: AIS Vessel Tracking — Baltic/Northern Europe Maritime Intelligence

Finnish Digitraffic AIS API — zero cost, no auth, real-time positions for 18K+
vessels in the Baltic Sea and Northern European waters.

Four modes:
  area             — Vessels in a bounding box or named area (danish_straits,
                     gulf_of_finland, st_petersburg, gotland, skagerrak, kiel,
                     full_baltic). Filter by ship type. Returns vessel count +
                     details. Feed count into changepoint detection for anomaly.
  vessel           — Position + metadata + destination for a specific MMSI.
  port_calls       — Finnish port arrivals/departures. Track port activity for
                     Nordic economic activity signal.
  destination_flow — Aggregate: how many vessels are heading to each destination,
                     optionally filtered by ship type. THE killer feature: when
                     ships heading to PORT SAID drops 40%, Suez is disrupted.
                     When RU LED drops, sanctions are biting.

Coverage: Lat 54.9–65.2°N, Lon 11.5–37.5°E (Baltic + approaches).
  Captures: Finland, Sweden, Estonia, Latvia, Lithuania, Poland, Denmark,
  Norway, NW Russia (St. Petersburg) — plus destination intent for global routes.

Ship types visible: ~5K tankers, ~9K cargo, ~660 passenger, ~850 fishing, ~1K tugs.

API docs: https://www.digitraffic.fi/en/marine/
Rate limits: None detected. Sub-second responses.

What this tool measures vs. what it writes (2026-09-23)
-------------------------------------------------------
``mode="area_daily_snapshot"`` is the DAG's node, and it used to be able to
report success while writing a fraction of what the feed published — the F-16
shape.  Three separate leaks, all now closed and all now *counted*:

  * the per-vessel positions were fetched and thrown away, then capped at 500
    of ~1,185 (58% dropped, silently, on a feed that only ever serves "now");
  * a single store failure aborted the rest of the batch while the tool still
    returned ``success=True`` (9 rows of 500, green);
  * ``observed_at`` was the fetch clock, not the AIS report time, so a third of
    rows were mis-dated by up to 24h and every retry appended duplicates
    instead of collapsing on ``OBSERVATION_UNIQUE_KEY``.

The rule the fixes share: the returned ``data`` and the persisted row carry
``vessels_persisted`` beside ``vessels_available``, and a shortfall returns
``success=False``.  A count of what was seen upstream is not evidence that
anything landed.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx

from agent.data.cache import DataCache
from agent.pipeline.store import PipelineStore
from agent.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    pass

try:
    from agent.pipeline.entity import entity_id_from_key
except ImportError:  # pragma: no cover — entity module always available
    entity_id_from_key = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

# Destination string → ISO-2 country code for common Baltic/global ports.
# AIS destination is free-text entered by crew; this maps the most frequent
# values seen in Digitraffic data to countries for the port_call_to edge.
_DEST_COUNTRY: dict[str, str] = {
    # Nordic / Baltic
    "HELSINKI": "FI",
    "TURKU": "FI",
    "KOTKA": "FI",
    "HAMINA": "FI",
    "RAUMA": "FI",
    "PORI": "FI",
    "OULU": "FI",
    "KOKKOLA": "FI",
    "STOCKHOLM": "SE",
    "GOTHENBURG": "SE",
    "GOTEBORG": "SE",
    "MALMO": "SE",
    "LULEA": "SE",
    "TALLINN": "EE",
    "MUUGA": "EE",
    "RIGA": "LV",
    "VENTSPILS": "LV",
    "KLAIPEDA": "LT",
    "GDANSK": "PL",
    "GDYNIA": "PL",
    "SZCZECIN": "PL",
    "COPENHAGEN": "DK",
    "AARHUS": "DK",
    "FREDERICIA": "DK",
    "OSLO": "NO",
    "BERGEN": "NO",
    "STAVANGER": "NO",
    # Russia
    "ST PETERSBURG": "RU",
    "SAINT PETERSBURG": "RU",
    "SPB": "RU",
    "PRIMORSK": "RU",
    "UST LUGA": "RU",
    "KALININGRAD": "RU",
    "MURMANSK": "RU",
    "NOVOROSSIYSK": "RU",
    # Western Europe
    "ROTTERDAM": "NL",
    "AMSTERDAM": "NL",
    "ANTWERP": "BE",
    "ZEEBRUGGE": "BE",
    "HAMBURG": "DE",
    "BREMERHAVEN": "DE",
    "WILHELMSHAVEN": "DE",
    "KIEL": "DE",
    "LE HAVRE": "FR",
    "MARSEILLE": "FR",
    "DUNKIRK": "FR",
    "FELIXSTOWE": "GB",
    "LONDON": "GB",
    "SOUTHAMPTON": "GB",
    "IMMINGHAM": "GB",
    "TILBURY": "GB",
    # Mediterranean
    "ALGECIRAS": "ES",
    "BARCELONA": "ES",
    "VALENCIA": "ES",
    "GENOA": "IT",
    "TRIESTE": "IT",
    "GIOIA TAURO": "IT",
    "PIRAEUS": "GR",
    "PORT SAID": "EG",
    "SUEZ": "EG",
    "ISTANBUL": "TR",
    "MERSIN": "TR",
    # Middle East / Asia
    "JEDDAH": "SA",
    "RAS TANURA": "SA",
    "FUJAIRAH": "AE",
    "JEBEL ALI": "AE",
    "SINGAPORE": "SG",
    "SHANGHAI": "CN",
    "NINGBO": "CN",
    "SHENZHEN": "CN",
    "QINGDAO": "CN",
    "BUSAN": "KR",
    "TOKYO": "JP",
    "YOKOHAMA": "JP",
    # Americas
    "HOUSTON": "US",
    "NEW YORK": "US",
    "LOS ANGELES": "US",
    "LONG BEACH": "US",
    "SAVANNAH": "US",
    "NORFOLK": "US",
    "SANTOS": "BR",
    "COLON": "PA",
}

_BASE = "https://meri.digitraffic.fi/api"
_UA = "TirraMind/0.1 (research; https://github.com/tirramind)"
_TIMEOUT = 30  # bulk fetches are ~7MB

# MP-1 ghost chains: one daily row per area (not per-vessel position spam).
_AREA_DAILY_OBS = "area_daily_activity"
_AIS_PROXY_OBS = "baltic_activity_proxy"
_DEFAULT_MP1_AREA = "full_baltic"
# Runaway guard on per-vessel position rows written per snapshot run.  This is
# NOT a sampling policy.  The whole Digitraffic locations feed carries ~1,200
# live vessels (1,185 of them inside the full_baltic bbox on 2026-09-23), so at
# one snapshot per day the real volume is ~440k rows/year and this ceiling never
# bites; it exists only so an upstream shape change cannot write unbounded rows
# in a single run.  The previous value (500) was borrowed from the *display*
# limit of the old ``mode="area"`` wiring — whose own default was 50 — and it
# silently dropped 58% of every run.  AIS is time-gated: those positions were
# gone for good, which is F-16's shape (200 OK, green, partial rows).  If the
# ceiling ever does bite, ``_rank_vessels_for_persist`` logs it and the snapshot
# records ``vessels_available`` beside ``vessels_persisted``.
_MAX_VESSEL_OBS_PER_RUN = 5000

# The locations feed publishes its own ``dataUpdatedTime``.  Older than this and
# the feed has stopped moving: the counts are then a replay of an earlier
# snapshot rather than a measurement of today, and a zero from a frozen feed
# must not enter the MP-1 series looking like a quiet Baltic.
_MAX_FEED_AGE_SECONDS = 6 * 3600
_MIN_LIVE_DAYS_FOR_PRIMARY_SERIES = 30

# Cache TTLs (seconds)
_LOC_TTL = 300  # 5 min — positions change constantly
_META_TTL = 21600  # 6 hr — ship names/types rarely change
_PORT_TTL = 3600  # 1 hr — port call data updates periodically

# AIS ship type code ranges
_SHIP_TYPE_RANGES = {
    "tanker": (80, 89),
    "cargo": (70, 79),
    "passenger": (60, 69),
    "fishing": (30, 39),
    "tug": (50, 59),
}

# AIS navigation status codes
_NAV_STATUS = {
    0: "under_way_engine",
    1: "at_anchor",
    2: "not_under_command",
    3: "restricted_manoeuvrability",
    4: "constrained_by_draught",
    5: "moored",
    6: "aground",
    7: "fishing",
    8: "under_way_sailing",
    9: "reserved_hsc",
    10: "reserved_wig",
    11: "power_driven_towing",
    12: "pushing_ahead_alongside",
    14: "ais_sart",
    15: "undefined",
}

# Pre-defined named areas — strategically important maritime zones
_NAMED_AREAS: dict[str, tuple[float, float, float, float]] = {
    # (lat_min, lat_max, lon_min, lon_max)
    "danish_straits": (
        54.5,
        58.0,
        9.5,
        13.5,
    ),  # The Sound + Great Belt — Baltic gateway
    "gulf_of_finland": (59.0, 61.0, 23.0, 30.5),  # St. Petersburg approach
    "st_petersburg": (59.5, 60.5, 28.0, 31.0),  # St. Petersburg port area
    "gotland": (56.5, 59.5, 17.0, 21.0),  # Central Baltic — everyone crosses here
    "skagerrak": (57.0, 59.5, 7.5, 12.5),  # North Sea → Baltic approach
    "kiel": (54.0, 55.0, 9.5, 11.0),  # Kiel Canal approaches
    "gulf_of_bothnia": (60.0, 66.0, 17.0, 26.0),  # Finland–Sweden strait
    "riga_gulf": (56.5, 58.5, 22.0, 25.0),  # Gulf of Riga
    "full_baltic": (54.0, 66.0, 9.0, 31.0),  # Entire Baltic Sea
}


def _in_bbox(
    lat: float,
    lon: float,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> bool:
    """Check if a point is within a bounding box."""
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def _ship_type_label(code: int) -> str:
    """Convert AIS ship type code to human label."""
    for label, (lo, hi) in _SHIP_TYPE_RANGES.items():
        if lo <= code <= hi:
            return label
    return "other"


def _normalize_mmsi(mmsi: int | str | None) -> int | None:
    if mmsi is None:
        return None
    try:
        return int(mmsi)
    except (TypeError, ValueError):
        return None


def _vessel_meta(meta: dict[Any, dict], mmsi: int | str | None) -> dict:
    """Lookup metadata whether the index keys are int or str MMSI."""
    key = _normalize_mmsi(mmsi)
    if key is None:
        return {}
    if key in meta:
        return meta[key]
    s = str(key)
    if s in meta:
        return meta[s]
    return {}


def _ship_type_matches(code: int, filter_type: str) -> bool:
    """Check if a ship type code matches the requested filter."""
    if filter_type == "all":
        return True
    rng = _SHIP_TYPE_RANGES.get(filter_type)
    if rng is None:
        return True
    return rng[0] <= code <= rng[1]


def _report_timestamp(props: dict[str, Any]) -> float | None:
    """The vessel's own AIS report time in epoch **seconds**, or None.

    ``timestampExternal`` is epoch milliseconds — when the position was actually
    reported.  ``timestamp`` in the same properties dict is the AIS
    second-of-minute field (observed values 11, 49, 55…), *not* an epoch;
    reading it as one would date every row to 1970 and the store's
    ``[1990, now+1d]`` guard would reject the lot.  Returning None (rather than
    a fabricated value) lets the caller fall back to fetch time explicitly.
    """
    raw = props.get("timestampExternal")
    if not isinstance(raw, (int, float)) or isinstance(raw, bool) or raw <= 0:
        return None
    return float(raw) / 1000.0


class AISVesselTool(Tool):
    name = "ais_vessel_tracking"
    description = (
        "Track real-time vessel positions, port activity, and trade flows in "
        "Baltic/Northern European waters via Finnish Digitraffic AIS. L0 physics "
        "signal — ships can't fake 300m tanker positions. Modes: 'area' = vessels "
        "in a named zone or bounding box. 'vessel' = track specific MMSI. "
        "'port_calls' = Finnish port arrivals/departures. 'destination_flow' = "
        "aggregate destination distribution (detect Suez/Russia trade shifts). "
        "18K+ vessels, zero cost, no API key. Coverage: Baltic Sea + Northern Europe."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["area", "area_daily_snapshot", "vessel", "port_calls", "destination_flow"],
                "description": (
                    "area = vessels in bounding box or named area. "
                    "vessel = specific MMSI position + metadata. "
                    "port_calls = Finnish port activity. "
                    "destination_flow = destination distribution by ship type."
                ),
            },
            "area_name": {
                "type": "string",
                "enum": list(_NAMED_AREAS.keys()),
                "description": (
                    "Named area for 'area' mode. Overrides lat/lon if set. "
                    "Options: danish_straits, gulf_of_finland, st_petersburg, "
                    "gotland, skagerrak, kiel, gulf_of_bothnia, riga_gulf, "
                    "full_baltic."
                ),
                "default": "",
            },
            "lat_min": {
                "type": "number",
                "description": "Bounding box south latitude (area mode, custom box).",
            },
            "lat_max": {
                "type": "number",
                "description": "Bounding box north latitude (area mode, custom box).",
            },
            "lon_min": {
                "type": "number",
                "description": "Bounding box west longitude (area mode, custom box).",
            },
            "lon_max": {
                "type": "number",
                "description": "Bounding box east longitude (area mode, custom box).",
            },
            "ship_type": {
                "type": "string",
                "enum": ["tanker", "cargo", "passenger", "fishing", "tug", "all"],
                "description": "Filter by ship type. Default: all.",
                "default": "all",
            },
            "mmsi": {
                "type": "integer",
                "description": "Maritime Mobile Service Identity (vessel mode).",
            },
            "from_date": {
                "type": "string",
                "description": ("Start date for port_calls mode (YYYY-MM-DD). Default: yesterday."),
                "default": "",
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return. Default: 50.",
                "default": 50,
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
        # ``dataUpdatedTime`` of the most recent locations payload this instance
        # saw (cache hit included — a stale cache is a stale feed).  Read right
        # after the fetch that sets it; see _count_area_vessels.
        self._last_feed_updated_at: str | None = None

    # ------------------------------------------------------------------
    # Entity ID helper
    # ------------------------------------------------------------------

    @staticmethod
    def _vessel_entity_id(mmsi: int, imo: int | None = None) -> str | None:
        """Compute entity_id: IMO-first (stable hull ID), MMSI-fallback."""
        if entity_id_from_key is None:
            return None
        if imo:
            return entity_id_from_key("vessel", str(imo))
        return entity_id_from_key("vessel", f"mmsi:{mmsi}")

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, url: str, params: dict | None = None, timeout: int = _TIMEOUT) -> httpx.Response:
        """HTTP GET with standard headers, gzip, and 429 retry."""
        headers = {
            "User-Agent": _UA,
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
        }
        last_exc: httpx.HTTPError | None = None
        for attempt in range(4):
            try:
                resp = httpx.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                    follow_redirects=True,
                )
                if resp.status_code == 429 and attempt < 3:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                resp.raise_for_status()
                return resp
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt < 3:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("unreachable")

    # ------------------------------------------------------------------
    # Bulk data fetchers (with caching)
    # ------------------------------------------------------------------

    def _fetch_locations(self) -> list[dict]:
        """Fetch all vessel locations. Cached for 5 min.

        Side effect by design: records the feed's own ``dataUpdatedTime`` on the
        instance, so a caller can tell "the Baltic is quiet" from "the feed
        stopped publishing".  The whole payload is cached (not just
        ``features``) precisely so a cache hit carries that timestamp too.
        """
        payload = self._fetch_locations_payload()
        self._last_feed_updated_at = payload.get("dataUpdatedTime")
        features = payload.get("features") or []
        return features if isinstance(features, list) else []

    def _fetch_locations_payload(self) -> dict:
        """Raw locations response (``type``/``dataUpdatedTime``/``features``)."""
        cache_key = "ais_locations_bulk"
        if self._cache:
            cached = self._cache.get("ais_vessel", {"key": cache_key})
            if isinstance(cached, dict):
                return cached
            if isinstance(cached, list):
                # Entry written before the payload envelope was cached: usable
                # as features, but it carries no dataUpdatedTime.
                return {"features": cached}

        resp = self._get(f"{_BASE}/ais/v1/locations")
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            data = {"features": data or []}

        if self._cache and data.get("features"):
            self._cache.put("ais_vessel", {"key": cache_key}, data)

        return data

    def _fetch_metadata(self) -> dict[int, dict]:
        """Fetch all vessel metadata, indexed by MMSI. Cached for 6hr."""
        cache_key = "ais_metadata_bulk_v2"
        if self._cache:
            cached = self._cache.get("ais_vessel_meta", {"key": cache_key})
            if cached is not None:
                return cached

        resp = self._get(f"{_BASE}/ais/v1/vessels")
        resp.raise_for_status()
        vessels = resp.json()

        # Index by MMSI for O(1) lookups
        indexed: dict[int, dict] = {}
        if isinstance(vessels, list):
            for v in vessels:
                mmsi = _normalize_mmsi(v.get("mmsi"))
                if mmsi is not None:
                    indexed[mmsi] = v

        if self._cache and indexed:
            self._cache.put("ais_vessel_meta", {"key": cache_key}, indexed)

        return indexed

    def _fetch_vessel_metadata_single(self, mmsi: int) -> dict | None:
        """Fetch metadata for a single vessel by MMSI."""
        cache_key = f"ais_meta_{mmsi}"
        if self._cache:
            cached = self._cache.get("ais_vessel_meta_single", {"key": cache_key})
            if cached is not None:
                return cached

        resp = self._get(f"{_BASE}/ais/v1/vessels/{mmsi}", timeout=10)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()

        if self._cache and data:
            self._cache.put("ais_vessel_meta_single", {"key": cache_key}, data)

        return data

    def _fetch_port_calls(self, from_date: str, to_date: str | None = None) -> list[dict]:
        """Fetch Finnish port call data for a date range (inclusive from, optional to)."""
        cache_key = f"ais_port_calls_{from_date}_{to_date or 'open'}"
        if self._cache:
            cached = self._cache.get("ais_port_calls", {"key": cache_key})
            if cached is not None:
                return cached

        params: dict[str, str] = {"from": f"{from_date}T00:00:00Z"}
        if to_date:
            params["to"] = f"{to_date}T23:59:59Z"
        resp = self._get(
            f"{_BASE}/port-call/v1/port-calls",
            params=params,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        calls = []
        if isinstance(data, dict):
            calls = data.get("portCalls", [])
        elif isinstance(data, list):
            calls = data

        if self._cache and calls:
            self._cache.put("ais_port_calls", {"key": cache_key}, calls)

        return calls

    # ------------------------------------------------------------------
    # Mode implementations
    # ------------------------------------------------------------------

    def _count_area_vessels(
        self,
        area_name: str,
        ship_type: str = "all",
    ) -> dict[str, Any]:
        """Count vessels in a named area (full scan, no display limit)."""
        if area_name not in _NAMED_AREAS:
            raise ValueError(f"Unknown area_name: {area_name}")
        lat_min, lat_max, lon_min, lon_max = _NAMED_AREAS[area_name]
        features = self._fetch_locations()
        meta = self._fetch_metadata()
        matched: list[dict[str, Any]] = []
        for f in features:
            geom = f.get("geometry")
            if not geom or geom.get("type") != "Point":
                continue
            coords = geom.get("coordinates", [])
            if len(coords) < 2:
                continue
            lon, lat = coords[0], coords[1]
            if not _in_bbox(lat, lon, lat_min, lat_max, lon_min, lon_max):
                continue
            mmsi = f.get("mmsi") or f.get("properties", {}).get("mmsi")
            if mmsi is None:
                continue
            vm = _vessel_meta(meta, mmsi)
            st_code = vm.get("shipType", 0)
            label = _ship_type_label(st_code)
            if ship_type != "all" and not _ship_type_matches(st_code, ship_type):
                continue
            props = f.get("properties", {})
            matched.append(
                {
                    "mmsi": mmsi,
                    "lat": lat,
                    "lon": lon,
                    # The AIS report time, not the fetch time: a third of the
                    # feed's positions are over an hour old (p90 ~10h, max 24h),
                    # so stamping them "now" mis-dates them by up to a day and
                    # defeats OBSERVATION_UNIQUE_KEY, which includes observed_at
                    # — a retry then appends duplicates instead of collapsing.
                    "timestamp": _report_timestamp(props),
                    "sog": props.get("sog"),
                    "cog": props.get("cog"),
                    "heading": props.get("heading"),
                    "nav_status": _NAV_STATUS.get(props.get("navStat", 15), "unknown"),
                    "name": vm.get("name", ""),
                    "imo": vm.get("imo"),
                    "destination": vm.get("destination", ""),
                    "ship_type": label,
                    "ship_type_code": st_code,
                }
            )
        type_counts: dict[str, int] = {}
        for m in matched:
            st = m.get("ship_type", "unknown")
            type_counts[st] = type_counts.get(st, 0) + 1
        tanker_count = type_counts.get("tanker", 0)
        return {
            "area": area_name,
            "ship_type_filter": ship_type,
            "vessel_count": len(matched),
            "tanker_count": tanker_count,
            "type_counts": type_counts,
            # Evidence about the fetch itself, so a zero in this row can be read
            # back as "no ships in the bbox" (feed_feature_count > 0) or "the
            # fetch broke" (feed_feature_count == 0).
            "feed_feature_count": len(features),
            "feed_updated_at": self._last_feed_updated_at,
            # Full per-vessel records, kept out of the persisted area row by
            # the caller.  Discarding these is what starved the time-gated
            # ``vessel_position`` series between 2026-06-09 and 2026-09-23.
            "vessels": matched,
        }

    def _persist_area_observation(
        self,
        area_name: str,
        observed_at: float,
        value: dict[str, Any],
        observation_type: str,
    ) -> None:
        if self._store is None or entity_id_from_key is None:
            return
        eid = entity_id_from_key("maritime_area", area_name)
        self._store.register_entity(
            entity_type="maritime_area",
            canonical_name=area_name,
            entity_id=eid,
            metadata={"region": "baltic", "bbox": _NAMED_AREAS.get(area_name)},
        )
        self._store.store_entity_observation(
            entity_id=eid,
            source_tool="ais_vessel",
            observed_at=observed_at,
            observation_type=observation_type,
            value=value,
            depth_level=1,
        )

    def _mode_area(self, **kw: Any) -> ToolResult:
        """Vessels in a bounding box or named area."""
        area_name = (kw.get("area_name") or kw.get("area") or "").strip().lower()
        ship_type = (kw.get("ship_type") or "all").strip().lower()
        limit = min(max(int(kw.get("limit", 50)), 1), 500)

        # Resolve bounding box
        if area_name and area_name in _NAMED_AREAS:
            lat_min, lat_max, lon_min, lon_max = _NAMED_AREAS[area_name]
        elif all(kw.get(k) is not None for k in ("lat_min", "lat_max", "lon_min", "lon_max")):
            lat_min = float(kw["lat_min"])
            lat_max = float(kw["lat_max"])
            lon_min = float(kw["lon_min"])
            lon_max = float(kw["lon_max"])
        else:
            return ToolResult(
                success=False,
                output="area mode requires 'area_name' or all of lat_min/lat_max/lon_min/lon_max.",
            )

        # Validate bbox
        if lat_min >= lat_max or lon_min >= lon_max:
            return ToolResult(
                success=False,
                output=f"Invalid bounding box: lat [{lat_min}, {lat_max}], lon [{lon_min}, {lon_max}]. Min must be < max.",
            )

        try:
            features = self._fetch_locations()
        except httpx.HTTPError as exc:
            return ToolResult(success=False, output=f"AIS locations fetch failed: {exc}")

        # Need metadata if filtering by ship type
        meta = {}
        if ship_type != "all":
            try:
                meta = self._fetch_metadata()
            except httpx.HTTPError as exc:
                return ToolResult(success=False, output=f"AIS metadata fetch failed: {exc}")

        # Filter vessels
        matched = []
        for f in features:
            geom = f.get("geometry")
            if not geom or geom.get("type") != "Point":
                continue
            coords = geom.get("coordinates", [])
            if len(coords) < 2:
                continue
            lon, lat = coords[0], coords[1]
            if not _in_bbox(lat, lon, lat_min, lat_max, lon_min, lon_max):
                continue

            mmsi = f.get("mmsi") or f.get("properties", {}).get("mmsi")
            if mmsi is None:
                continue

            # Ship type filter
            if ship_type != "all":
                vessel_meta = _vessel_meta(meta, mmsi)
                st_code = vessel_meta.get("shipType", 0)
                if not _ship_type_matches(st_code, ship_type):
                    continue

            props = f.get("properties", {})
            entry = {
                "mmsi": mmsi,
                "lat": lat,
                "lon": lon,
                "timestamp": _report_timestamp(props),  # AIS report time, not fetch time
                "sog": props.get("sog"),  # speed over ground (knots)
                "cog": props.get("cog"),  # course over ground (degrees)
                "heading": props.get("heading"),
                "nav_status": _NAV_STATUS.get(props.get("navStat", 15), "unknown"),
            }

            # Enrich with metadata if available
            if meta:
                vm = _vessel_meta(meta, mmsi)
                entry["name"] = vm.get("name", "")
                entry["destination"] = vm.get("destination", "")
                entry["ship_type"] = _ship_type_label(vm.get("shipType", 0))
                entry["imo"] = vm.get("imo")
            elif ship_type == "all":
                entry["ship_type"] = "unknown"

            matched.append(entry)

        # Count by nav status and ship type
        status_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        for m in matched:
            ns = m.get("nav_status", "unknown")
            status_counts[ns] = status_counts.get(ns, 0) + 1
            st = m.get("ship_type", "unknown")
            type_counts[st] = type_counts.get(st, 0) + 1

        area_label = area_name if area_name else f"[{lat_min:.1f}-{lat_max:.1f}°N, {lon_min:.1f}-{lon_max:.1f}°E]"

        lines = [
            f"AIS Area: {area_label} | {len(matched)} vessels"
            + (f" (type: {ship_type})" if ship_type != "all" else ""),
            "",
        ]

        if type_counts and ship_type == "all":
            lines.append(
                "By type: " + ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items(), key=lambda x: -x[1]))
            )
        if status_counts:
            lines.append(
                "By status: " + ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items(), key=lambda x: -x[1]))
            )
        lines.append("")

        # Top vessels by speed (most active)
        moving = sorted(
            [m for m in matched if (m.get("sog") or 0) > 0.5],
            key=lambda x: -(x.get("sog") or 0),
        )
        for v in moving[:limit]:
            name = v.get("name", "").strip() or f"MMSI:{v['mmsi']}"
            dest = v.get("destination", "").strip()
            dest_str = f" → {dest}" if dest else ""
            lines.append(
                f"  {name} | {v.get('ship_type', '?')} | "
                f"{v['lat']:.4f}°N {v['lon']:.4f}°E | "
                f"SOG:{v.get('sog', 0):.1f}kn COG:{v.get('cog', 0):.0f}° | "
                f"{v.get('nav_status', '?')}{dest_str}"
            )

        # L2: entity_ids + persistence
        if entity_id_from_key is not None:
            for v in matched:
                v["entity_id"] = self._vessel_entity_id(v["mmsi"], v.get("imo"))

        try:
            self._persist_entities(matched[:limit])
        except Exception:
            log.exception("Entity persistence failed in area mode (non-fatal)")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "area": area_label,
                "total_vessels": len(matched),
                "type_counts": type_counts,
                "status_counts": status_counts,
                "vessels": matched[:limit],
            },
        )

    def _mode_vessel(self, **kw: Any) -> ToolResult:
        """Track a specific vessel by MMSI."""
        mmsi = kw.get("mmsi")
        if mmsi is None:
            return ToolResult(success=False, output="vessel mode requires 'mmsi' parameter.")

        mmsi = int(mmsi)

        # Fetch metadata (single vessel — faster than bulk)
        try:
            meta = self._fetch_vessel_metadata_single(mmsi)
        except httpx.HTTPError as exc:
            return ToolResult(success=False, output=f"AIS metadata fetch failed: {exc}")

        # Find position in bulk locations
        position = None
        try:
            features = self._fetch_locations()
            for f in features:
                f_mmsi = f.get("mmsi") or f.get("properties", {}).get("mmsi")
                if f_mmsi == mmsi:
                    geom = f.get("geometry", {})
                    coords = geom.get("coordinates", [])
                    if len(coords) >= 2:
                        props = f.get("properties", {})
                        position = {
                            "lat": coords[1],
                            "lon": coords[0],
                            "sog": props.get("sog"),
                            "cog": props.get("cog"),
                            "heading": props.get("heading"),
                            "nav_status": _NAV_STATUS.get(props.get("navStat", 15), "unknown"),
                        }
                    break
        except httpx.HTTPError as exc:
            log.warning("Failed to fetch locations for MMSI %d: %s", mmsi, exc)

        if meta is None and position is None:
            return ToolResult(success=False, output=f"MMSI {mmsi} not found in AIS data.")

        result: dict[str, Any] = {"mmsi": mmsi}

        if meta:
            result["name"] = meta.get("name", "")
            result["imo"] = meta.get("imo")
            result["call_sign"] = meta.get("callSign", "")
            result["destination"] = meta.get("destination", "")
            result["ship_type"] = _ship_type_label(meta.get("shipType", 0))
            result["ship_type_code"] = meta.get("shipType")
            result["draught"] = meta.get("draught")

        if position:
            result.update(position)

        # Format output
        name = result.get("name", "").strip() or f"MMSI:{mmsi}"
        lines = [f"Vessel: {name} (MMSI: {mmsi})"]

        if meta:
            lines.append(f"  IMO: {result.get('imo', '?')} | Call: {result.get('call_sign', '?')}")
            lines.append(f"  Type: {result.get('ship_type', '?')} (code: {result.get('ship_type_code', '?')})")
            dest = result.get("destination", "").strip()
            if dest:
                lines.append(f"  Destination: {dest}")
            dr = result.get("draught")
            if dr is not None:
                lines.append(f"  Draught: {dr / 10:.1f}m")

        if position:
            lines.append(f"  Position: {position['lat']:.5f}°N {position['lon']:.5f}°E")
            lines.append(
                f"  SOG: {position.get('sog', 0):.1f}kn | "
                f"COG: {position.get('cog', 0):.0f}° | "
                f"Heading: {position.get('heading', '?')}° | "
                f"Status: {position.get('nav_status', '?')}"
            )
        else:
            lines.append("  Position: not in current AIS coverage")

        # L2: entity_id + persistence
        if entity_id_from_key is not None:
            result["entity_id"] = self._vessel_entity_id(result["mmsi"], result.get("imo"))

        try:
            self._persist_entities([result])
        except Exception:
            log.exception("Entity persistence failed in vessel mode (non-fatal)")

        return ToolResult(success=True, output="\n".join(lines), data=result)

    def _mode_port_calls(self, **kw: Any) -> ToolResult:
        """Finnish port call activity."""
        from_date = (kw.get("from_date") or "").strip()
        limit = min(max(int(kw.get("limit", 50)), 1), 500)

        if not from_date:
            yesterday = datetime.now(UTC) - timedelta(days=1)
            from_date = yesterday.strftime("%Y-%m-%d")

        try:
            calls = self._fetch_port_calls(from_date)
        except httpx.HTTPError as exc:
            return ToolResult(success=False, output=f"Port call fetch failed: {exc}")

        # Aggregate by port
        port_counts: dict[str, int] = {}
        for c in calls:
            port = c.get("portToVisit", "UNKNOWN")
            port_counts[port] = port_counts.get(port, 0) + 1

        lines = [
            f"Finnish Port Calls since {from_date}: {len(calls)} total across {len(port_counts)} ports",
            "",
        ]

        if port_counts:
            lines.append(
                "By port: " + ", ".join(f"{k}={v}" for k, v in sorted(port_counts.items(), key=lambda x: -x[1])[:20])
            )
            lines.append("")

        # Show individual calls
        for c in calls[:limit]:
            name = c.get("vesselName", "?").strip()
            port = c.get("portToVisit", "?")
            prev_port = c.get("prevPort", "?")
            next_port = c.get("nextPort", "?")
            cargo = "cargo" if c.get("arrivalWithCargo") else "ballast"
            lines.append(f"  {name} | {prev_port} → {port} → {next_port} | {cargo}")

        # L2: entity persistence for port calls
        try:
            self._persist_port_call_entities(calls[:limit])
        except Exception:
            log.exception("Entity persistence failed in port_calls mode (non-fatal)")

        # L2: entity_ids
        if entity_id_from_key is not None:
            for c in calls[:limit]:
                mmsi = c.get("mmsi")
                imo = c.get("imoLloyds")
                if mmsi or imo:
                    c["entity_id"] = self._vessel_entity_id(mmsi, imo)

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "from_date": from_date,
                "total_calls": len(calls),
                "port_counts": port_counts,
                "calls": calls[:limit],
            },
        )

    @staticmethod
    def _rank_vessels_for_persist(
        vessels: list[dict[str, Any]],
        ship_type: str,
    ) -> list[dict[str, Any]]:
        """Order the requested ship type first, then cap at the per-run limit.

        The cap is a runaway guard, not a sampling policy (see
        ``_MAX_VESSEL_OBS_PER_RUN``); if it ever bites it is data loss on a
        time-gated feed, so it is logged here and counted by the caller.  The
        ordering keeps the signal ships (tankers for MP-1) ahead of the cut.
        """
        if ship_type and ship_type != "all":
            preferred = [v for v in vessels if v.get("ship_type") == ship_type]
            rest = [v for v in vessels if v.get("ship_type") != ship_type]
            vessels = preferred + rest
        if len(vessels) > _MAX_VESSEL_OBS_PER_RUN:
            log.warning(
                "AIS vessel persistence truncated: %d of %d vessels dropped by "
                "_MAX_VESSEL_OBS_PER_RUN=%d (ship_type=%s). AIS is time-gated — "
                "dropped positions cannot be refetched.",
                len(vessels) - _MAX_VESSEL_OBS_PER_RUN,
                len(vessels),
                _MAX_VESSEL_OBS_PER_RUN,
                ship_type,
            )
        return vessels[:_MAX_VESSEL_OBS_PER_RUN]

    @staticmethod
    def _feed_age_seconds(updated_at: Any) -> float | None:
        """Seconds since the locations feed last published, None if unknown."""
        if not updated_at:
            return None
        try:
            ts = datetime.fromisoformat(str(updated_at).replace("Z", "+00:00"))
        except ValueError:
            log.warning("AIS locations feed: unparseable dataUpdatedTime %r", updated_at)
            return None
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return datetime.now(UTC).timestamp() - ts.timestamp()

    def _mode_area_daily_snapshot(self, **kw: Any) -> ToolResult:
        """Store one daily Baltic area activity row for ghost-chain z-scores."""
        area_name = (kw.get("area_name") or kw.get("area") or _DEFAULT_MP1_AREA).strip().lower()
        ship_type = (kw.get("ship_type") or "tanker").strip().lower()
        as_of = kw.get("as_of")
        if as_of:
            day = str(as_of)[:10]
        else:
            day = datetime.now(UTC).strftime("%Y-%m-%d")
        try:
            counts = self._count_area_vessels(area_name, ship_type="all")
        except httpx.HTTPError as exc:
            return ToolResult(success=False, output=f"AIS area snapshot failed: {exc}")
        except ValueError as exc:
            return ToolResult(success=False, output=str(exc))

        # Per-vessel records travel alongside the counts but must never land in
        # the area row's JSON value — ``series_count`` and the counts the MP-1
        # consumers read keep their pre-fix meaning.
        vessels = counts.pop("vessels", [])
        feed_features = counts.get("feed_feature_count", len(vessels))

        # A zero that means "the fetch broke" must never be written as a zero
        # that means "no ships": it would enter the ghost-chain z-score series
        # as a real measurement.  An empty feature list from this endpoint is
        # not a quiet Baltic — the feed always carries ~1,200 vessels.
        if not feed_features:
            return ToolResult(
                success=False,
                output=(
                    f"AIS daily snapshot ABORTED: locations feed returned 0 features "
                    f"({area_name}, {day}) — refusing to write a fabricated zero."
                ),
            )

        feed_age = self._feed_age_seconds(counts.get("feed_updated_at"))
        if counts.get("feed_updated_at") is None:
            log.warning("AIS locations feed carried no dataUpdatedTime — staleness unchecked")

        ranked = self._rank_vessels_for_persist(vessels, ship_type)

        # The moat: AIS positions are time-gated — the API serves only "now", so
        # a day not written here is a day gone for good.  Persist FIRST, then
        # write the area row carrying the receipt, so the stored row records how
        # many positions actually landed and not merely how many were seen.
        receipt = {
            "received": len(ranked),
            "unique": 0,
            "positions": 0,
            "failed": len(ranked),
            "skipped": 0,
            "fetch_clock": 0,
        }
        persist_error: str | None = None
        try:
            receipt = self._persist_entities(ranked)
        except Exception as exc:  # a store-level failure, not a per-vessel one
            persist_error = f"{type(exc).__name__}: {exc}"
            log.exception("Vessel entity persistence failed in daily snapshot")

        metric_count = counts["tanker_count"] if ship_type == "tanker" else counts["vessel_count"]
        value = {
            **counts,
            "metric": "baltic_area_live",
            "series_count": metric_count,
            "observed_day": day,
            # Row-count evidence: collected-vs-available, in the row itself.
            "vessels_available": len(vessels),
            "vessels_persisted": receipt["positions"],
            "vessels_truncated": len(vessels) - len(ranked),
            "vessels_failed": receipt["failed"],
            "vessels_without_report_time": receipt["fetch_clock"],
        }
        observed_at = datetime.fromisoformat(day + "T12:00:00+00:00").replace(tzinfo=UTC).timestamp()
        area_error: str | None = None
        try:
            self._persist_area_observation(area_name, observed_at, value, _AREA_DAILY_OBS)
        except Exception as exc:
            area_error = f"{type(exc).__name__}: {exc}"
            log.exception("Area daily snapshot persistence failed")

        lines = [
            f"AIS daily snapshot: {area_name} | {counts['tanker_count']} tankers | "
            f"{counts['vessel_count']} total vessels ({day})",
            f"vessel positions: {receipt['positions']} written / {len(vessels)} available"
            + (f", {len(vessels) - len(ranked)} truncated" if len(vessels) > len(ranked) else "")
            + (f", {receipt['skipped']} skipped" if receipt["skipped"] else ""),
        ]

        # Anything below this line is the difference between a green node and an
        # honest one.  Partial writes report success=False; the rows already
        # written stay written, and the DAG's retry re-writes them into the same
        # idempotency key (observed_at now comes from the AIS report time).
        problems: list[str] = []
        if area_error:
            problems.append(f"area_daily_activity row NOT written: {area_error}")
        if persist_error:
            problems.append(f"vessel persistence aborted after {receipt['positions']}/{len(ranked)}: {persist_error}")
        elif receipt["failed"]:
            problems.append(f"{receipt['failed']} of {receipt['unique']} vessel writes failed")
        if len(vessels) > len(ranked):
            # The cap is a runaway guard set at ~4x live volume; if it fires,
            # the feed changed shape and positions AIS can never serve again
            # were dropped.  A warning in a log file is not a signal — the node
            # itself has to go red.
            problems.append(
                f"{len(vessels) - len(ranked)} of {len(vessels)} positions dropped by "
                f"_MAX_VESSEL_OBS_PER_RUN={_MAX_VESSEL_OBS_PER_RUN} — AIS is time-gated, "
                "they cannot be refetched"
            )
        if feed_age is not None and feed_age > _MAX_FEED_AGE_SECONDS:
            problems.append(
                f"locations feed is stale: dataUpdatedTime={counts.get('feed_updated_at')} "
                f"({feed_age / 3600:.1f}h old, limit {_MAX_FEED_AGE_SECONDS / 3600:.0f}h) — "
                "counts are a replay, not today's measurement"
            )

        if problems:
            return ToolResult(
                success=False,
                output="\n".join(lines + ["AIS daily snapshot DEGRADED:"] + [f"  - {p}" for p in problems]),
                data=value,
            )

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data=value,
        )

    def _mode_destination_flow(self, **kw: Any) -> ToolResult:
        """Aggregate destination distribution — the killer feature."""
        ship_type = (kw.get("ship_type") or "all").strip().lower()
        limit = min(max(int(kw.get("limit", 50)), 1), 500)

        try:
            meta_index = self._fetch_metadata()
        except httpx.HTTPError as exc:
            return ToolResult(success=False, output=f"AIS metadata fetch failed: {exc}")

        # Filter by ship type and aggregate destinations
        dest_counts: dict[str, int] = {}
        total = 0
        for mmsi, v in meta_index.items():
            if ship_type != "all":
                st_code = v.get("shipType", 0)
                if not _ship_type_matches(st_code, ship_type):
                    continue

            dest = (v.get("destination") or "").strip().upper()
            if not dest:
                continue

            total += 1
            dest_counts[dest] = dest_counts.get(dest, 0) + 1

        # Sort by count descending
        sorted_dests = sorted(dest_counts.items(), key=lambda x: -x[1])

        type_label = ship_type if ship_type != "all" else "all types"
        lines = [
            f"AIS Destination Flow ({type_label}): {total} vessels with destinations, "
            f"{len(dest_counts)} unique destinations",
            "",
        ]

        # Highlight strategic destinations
        strategic = {
            "suez": ["PORT SAID", "SUEZ", "EGPSD", "EG PSD", "EG SUZ"],
            "russia": [
                "RU LED",
                "RULED",
                "SPB",
                "ST PETERSBURG",
                "RU VYS",
                "RUVYB",
                "RU PRI",
                "RUPRI",
            ],
            "rotterdam": ["ROTTERDAM", "NLRTM", "NL RTM"],
            "antwerp": ["ANTWERP", "BEANR", "BE ANR"],
        }

        for group_name, keywords in strategic.items():
            count = sum(dest_counts.get(k, 0) for k in keywords)
            if count > 0:
                lines.append(f"  [{group_name.upper()}]: {count} vessels heading there")

        lines.append("")
        lines.append("Top destinations:")

        for dest, count in sorted_dests[:limit]:
            pct = count / total * 100 if total > 0 else 0
            lines.append(f"  {dest}: {count} ({pct:.1f}%)")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "ship_type": ship_type,
                "total_with_destination": total,
                "unique_destinations": len(dest_counts),
                "destinations": dict(sorted_dests[:limit]),
                "strategic": {
                    group: sum(dest_counts.get(k, 0) for k in keywords) for group, keywords in strategic.items()
                },
            },
        )

    # ------------------------------------------------------------------
    # L2 Entity Persistence
    # ------------------------------------------------------------------

    def _persist_entities(self, vessels: list[dict[str, Any]]) -> dict[str, int]:
        """Persist vessel entities from area/vessel mode (position observations).

        Returns a write receipt — ``received`` / ``unique`` / ``positions`` /
        ``failed`` / ``skipped``.  Callers must compare ``positions`` against
        what the source published: a count of what was *seen upstream* is not
        evidence that anything landed (LESSONS F-16).
        """
        if self._store is None or entity_id_from_key is None:
            return {
                "received": len(vessels),
                "unique": 0,
                "positions": 0,
                "failed": 0,
                "skipped": len(vessels),
                "fetch_clock": 0,
            }
        return self._persist_entities_inner(vessels)

    @staticmethod
    def _observed_at_ts(raw: Any) -> float:
        """Normalize observed_at to unix seconds for PipelineStore."""
        if raw is None:
            return datetime.now(UTC).timestamp()
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str):
            try:
                return float(raw)
            except ValueError:
                s = raw.replace("Z", "+00:00")
                return datetime.fromisoformat(s).timestamp()
        return datetime.now(UTC).timestamp()

    def _persist_entities_inner(self, vessels: list[dict[str, Any]]) -> dict[str, int]:
        """Write one vessel at a time, isolated: one bad vessel costs one vessel.

        Before isolation, a single store failure aborted the rest of the batch
        while the caller still returned success — 9 rows of 500 on a feed that
        cannot be refetched.  Failures are counted and returned, never
        swallowed: the caller decides whether the shortfall is fatal.
        """
        seen: set[str] = set()
        now_ts = datetime.now(UTC).timestamp()
        receipt = {
            "received": len(vessels),
            "unique": 0,
            "positions": 0,
            "failed": 0,
            "skipped": 0,
            # Positions stamped with the fetch clock because the feed carried no
            # ``timestampExternal``.  Those rows are dated wrong by however long
            # the report sat upstream, and — since observed_at is part of
            # OBSERVATION_UNIQUE_KEY — a retry appends a copy instead of
            # collapsing.  Counted so the degradation cannot arrive silently.
            "fetch_clock": 0,
        }
        for v in vessels:
            mmsi = v.get("mmsi")
            imo = v.get("imo")
            if not mmsi and not imo:
                receipt["skipped"] += 1
                continue
            eid = self._vessel_entity_id(mmsi, imo)
            if eid in seen:
                receipt["skipped"] += 1
                continue
            seen.add(eid)
            receipt["unique"] += 1

            try:
                self._persist_one_vessel(v, eid, mmsi, imo, now_ts, receipt)
            except Exception:
                receipt["failed"] += 1
                log.exception(
                    "AIS vessel persistence failed for mmsi=%s imo=%s — continuing with the batch",
                    mmsi,
                    imo,
                )

        if receipt["failed"]:
            log.warning(
                "AIS vessel persistence: %d of %d vessels failed to write (%d positions stored)",
                receipt["failed"],
                receipt["unique"],
                receipt["positions"],
            )
        if receipt["fetch_clock"]:
            log.warning(
                "AIS vessel persistence: %d of %d positions carried no timestampExternal and "
                "were stamped with the fetch clock — those rows are mis-dated and will not "
                "collapse on retry.",
                receipt["fetch_clock"],
                receipt["positions"],
            )
        return receipt

    def _persist_one_vessel(
        self,
        v: dict[str, Any],
        eid: str,
        mmsi: Any,
        imo: Any,
        now_ts: float,
        receipt: dict[str, int],
    ) -> None:
        """Register one vessel, store its position, link its destination."""
        name = v.get("name") or v.get("vessel_name") or str(mmsi or imo)
        self._store.register_entity(
            entity_type="vessel",
            canonical_name=name,
            entity_id=eid,
            metadata={
                k: v.get(k) for k in ("ship_type", "ship_type_code", "destination", "draught") if v.get(k) is not None
            },
        )

        if mmsi:
            self._store.add_entity_alias(eid, "mmsi", str(mmsi))
        if imo:
            self._store.add_entity_alias(eid, "imo", str(imo))

        # Position observation
        lat = v.get("lat")
        lon = v.get("lon")
        if lat is not None and lon is not None:
            report_ts = v.get("timestamp")
            if not report_ts:
                receipt["fetch_clock"] += 1
            self._store.store_entity_observation(
                entity_id=eid,
                source_tool="ais_vessel",
                observed_at=self._observed_at_ts(report_ts or now_ts),
                observation_type="vessel_position",
                value={
                    "lat": lat,
                    "lon": lon,
                    "sog": v.get("sog"),
                    "cog": v.get("cog"),
                    "heading": v.get("heading"),
                    "nav_status": v.get("nav_status"),
                },
                depth_level=2,
            )
            receipt["positions"] += 1

        # ── Link vessel → destination country ──
        dest = (v.get("destination") or "").strip().upper()
        country_code = _DEST_COUNTRY.get(dest)
        if country_code and eid:
            country_eid = entity_id_from_key("country", country_code)
            self._store.register_entity(
                entity_type="country",
                canonical_name=country_code,
                entity_id=country_eid,
            )
            self._store.link_entities(
                entity_id_a=eid,
                entity_id_b=country_eid,
                link_type="port_call_to",
                source="ais_vessel",
                confidence=0.8,
                metadata={"destination_raw": dest},
            )

    def _persist_port_call_entities(self, calls: list[dict[str, Any]]) -> None:
        """Persist vessel entities from port_calls mode."""
        if self._store is None or entity_id_from_key is None:
            return
        self._persist_port_call_entities_inner(calls)

    def _persist_port_call_entities_inner(self, calls: list[dict[str, Any]]) -> None:
        seen: set[str] = set()
        now_ts = datetime.now(UTC).timestamp()
        for c in calls:
            mmsi = c.get("mmsi")
            imo = c.get("imoLloyds")
            if not mmsi and not imo:
                continue
            eid = self._vessel_entity_id(mmsi, imo)

            name = c.get("vesselName") or str(mmsi or imo)
            if eid not in seen:
                seen.add(eid)
                self._store.register_entity(
                    entity_type="vessel",
                    canonical_name=name.strip(),
                    entity_id=eid,
                    metadata={k: c.get(k) for k in ("vesselTypeCode", "nationality") if c.get(k) is not None},
                )
                if mmsi:
                    self._store.add_entity_alias(eid, "mmsi", str(mmsi))
                if imo:
                    self._store.add_entity_alias(eid, "imo", str(imo))

            self._store.store_entity_observation(
                entity_id=eid,
                source_tool="ais_vessel",
                observed_at=self._observed_at_ts(c.get("eta") or now_ts),
                observation_type="port_call",
                value={
                    "port": c.get("portToVisit"),
                    "prev_port": c.get("prevPort"),
                    "next_port": c.get("nextPort"),
                    "arrival_with_cargo": c.get("arrivalWithCargo"),
                },
                depth_level=2,
            )

            # ── Link vessel → port country ──
            port_name = (c.get("portToVisit") or "").strip().upper()
            country_code = _DEST_COUNTRY.get(port_name)
            if country_code and eid:
                country_eid = entity_id_from_key("country", country_code)
                self._store.register_entity(
                    entity_type="country",
                    canonical_name=country_code,
                    entity_id=country_eid,
                )
                self._store.link_entities(
                    entity_id_a=eid,
                    entity_id_b=country_eid,
                    link_type="port_call_to",
                    source="ais_vessel",
                    confidence=0.8,
                    metadata={"destination_raw": port_name},
                )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def execute(self, **kwargs: Any) -> ToolResult:
        mode = (kwargs.get("mode") or "").strip().lower()

        dispatch = {
            "area": self._mode_area,
            "area_daily_snapshot": self._mode_area_daily_snapshot,
            "vessel": self._mode_vessel,
            "port_calls": self._mode_port_calls,
            "destination_flow": self._mode_destination_flow,
        }

        handler = dispatch.get(mode)
        if handler is None:
            return ToolResult(
                success=False,
                output=f"Unknown mode '{mode}'. Use: {', '.join(dispatch.keys())}",
            )

        try:
            return handler(**kwargs)
        except Exception as exc:
            log.exception("AIS tool error in mode '%s'", mode)
            return ToolResult(success=False, output=f"AIS tool error: {exc}")


# ── MP-1 ghost pattern ingest / backfill ───────────────────────────────


def _port_call_day(call: dict[str, Any]) -> str | None:
    raw = call.get("portCallTimestamp") or call.get("eta")
    if not raw:
        return None
    return str(raw)[:10]


def _is_tanker_port_call(call: dict[str, Any]) -> bool:
    code = call.get("vesselTypeCode")
    if code is None:
        return False
    return _SHIP_TYPE_RANGES["tanker"][0] <= int(code) <= _SHIP_TYPE_RANGES["tanker"][1]


def backfill_ais_port_call_proxy(
    store: PipelineStore,
    *,
    lookback_days: int = 90,
    area_name: str = _DEFAULT_MP1_AREA,
    cache: DataCache | None = None,
) -> dict[str, Any]:
    """Backfill daily Finnish port-call counts as Baltic activity proxy.

    Digitraffic has no historical AIS area snapshots; port-call tanker counts
    are a consistent pre-live series until ``area_daily_activity`` accumulates.
    """
    if entity_id_from_key is None:
        return {"days_stored": 0, "error": "entity_id_from_key unavailable"}

    tool = AISVesselTool(cache=cache, pipeline_store=store)
    end = datetime.now(UTC).date()
    start = end - timedelta(days=max(lookback_days, 1))
    from_date = start.isoformat()
    to_date = end.isoformat()

    day_all: dict[str, int] = {}
    day_tankers: dict[str, int] = {}
    total_calls = 0
    failed_days = 0

    # Digitraffic rejects long multi-day ranges; single-day queries work for history.
    day = start
    while day <= end:
        day_s = day.isoformat()
        try:
            calls = tool._fetch_port_calls(day_s, day_s)
        except httpx.HTTPError as exc:
            log.warning("AIS port-call fetch failed for %s: %s", day_s, exc)
            failed_days += 1
            day += timedelta(days=1)
            time.sleep(0.5)
            continue
        total_calls += len(calls)
        day_all[day_s] = len(calls)
        day_tankers[day_s] = sum(1 for c in calls if _is_tanker_port_call(c))
        day += timedelta(days=1)
        time.sleep(0.35)

    stored = 0
    for day_s in sorted(day_all):
        observed_at = datetime.fromisoformat(day_s + "T12:00:00+00:00").replace(tzinfo=UTC).timestamp()
        tankers = day_tankers.get(day_s, 0)
        value = {
            "area": area_name,
            "metric": "finnish_port_calls",
            "vessel_count": day_all[day_s],
            "tanker_count": tankers,
            "series_count": tankers,
            "observed_day": day_s,
            "proxy": True,
        }
        tool._persist_area_observation(area_name, observed_at, value, _AIS_PROXY_OBS)
        stored += 1

    return {
        "days_stored": stored,
        "from_date": from_date,
        "to_date": to_date,
        "port_calls_fetched": total_calls,
        "failed_days": failed_days,
    }


def ingest_area_daily_snapshot(
    store: PipelineStore,
    *,
    area_name: str = _DEFAULT_MP1_AREA,
    ship_type: str = "tanker",
    cache: DataCache | None = None,
    as_of: date | None = None,
) -> dict[str, Any]:
    """Fetch live Baltic bbox tanker count and store one row for ``as_of`` day."""
    tool = AISVesselTool(cache=cache, pipeline_store=store)
    result = tool.execute(
        mode="area_daily_snapshot",
        area_name=area_name,
        ship_type=ship_type,
        as_of=(as_of or datetime.now(UTC).date()).isoformat(),
    )
    if not result.success:
        # Carry the receipt through the failure too: a degraded run may still
        # have written most of its rows, and the caller should see how many.
        return {"stored": False, "error": result.output, **(result.data or {})}
    return {"stored": True, **(result.data or {})}


def backfill_ais_area_daily(
    store: PipelineStore,
    *,
    lookback_days: int = 90,
    area_name: str = _DEFAULT_MP1_AREA,
    cache: DataCache | None = None,
) -> dict[str, Any]:
    """Port-call proxy history + today's live Baltic area snapshot."""
    proxy = backfill_ais_port_call_proxy(store, lookback_days=lookback_days, area_name=area_name, cache=cache)
    live = ingest_area_daily_snapshot(store, area_name=area_name, cache=cache)
    return {"proxy": proxy, "live": live}
