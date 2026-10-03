"""
Tool: Satellite-Derived Physical Activity

Three free APIs for raw physical-world observation:

  fire         NASA FIRMS thermal hotspots (industrial activity, wildfires)
  vegetation   MODIS NDVI crop health (agricultural commodity driver)
  events       NASA EONET natural disaster tracker (supply chain disruption)

Signal theory:
  - Fire Radiative Power (FRP) near refineries/factories → operational intensity
  - Wildfire clusters near infrastructure → supply chain disruption risk
  - NDVI decline in agricultural zones → crop stress → commodity price pressure
  - EONET events (volcanoes, storms) → physical disruption to shipping, production

Data sources:
  - NASA FIRMS  https://firms.modaps.eosdis.nasa.gov/api/  (free, MAP key)
  - ORNL MODIS  https://modis.ornl.gov/rst/api/v1/        (free, no auth)
  - NASA EONET  https://eonet.gsfc.nasa.gov/api/v3/        (free, no auth)
"""

from __future__ import annotations

import csv
import io
import logging
import math
import os
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_UA = "TirraMind/0.1 (satellite-activity-tool)"
_TIMEOUT = 25

_FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api"
_MODIS_BASE = "https://modis.ornl.gov/rst/api/v1"
_EONET_BASE = "https://eonet.gsfc.nasa.gov/api/v3"

VALID_MODES = {"fire", "vegetation", "events"}

FIRMS_SOURCES = {
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
    "VIIRS_SNPP_NRT",
    "MODIS_NRT",
}

# FIRMS /api/area/csv answers HTTP 400 "Invalid day range. Expects [1..5]."
# for anything above 5. The old clamp here was min(10, days), so a caller
# asking for the documented "1-10" got a hard fetch failure. Probed
# 2026-09-23: days=5 -> HTTP 200, days=10 -> HTTP 400.
FIRMS_MAX_DAYS = 5

# Columns every FIRMS /api/area/csv response carries (probed 2026-09-23:
# latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,...).
# A 200 whose body does not have at least these is not a FIRMS CSV — it is an
# error page or a truncated proxy response, and csv.DictReader turns it into
# an empty list that looks exactly like "no fires today". Guarded in
# :func:`_fetch_firms` so that case fails loudly instead of writing a green
# zero row (LESSONS F-16 shape).
FIRMS_REQUIRED_COLUMNS = ("latitude", "longitude")

# /api/data_availability/csv/<key>/ALL — free, same key, does not spend the
# 5000-transaction budget. Returns "data_id,min_date,max_date" per sensor.
# Probed 2026-09-23: VIIRS_NOAA20_NRT,2026-07-01,2026-09-23.
#
# This matters because a `date` outside the NRT rolling archive is NOT an
# error to FIRMS: it answers HTTP 200 with a header-only CSV, which parses to
# [] and used to be reported as "no thermal hotspots detected" — a fabricated
# zero that the convergence extractor happily turns into four evidence items.
# Probed 2026-09-23: date=2026-06-01 -> HTTP 200, 0 rows; date=2026-09-01 ->
# HTTP 200, 1501 rows.
FIRMS_AVAILABILITY_TTL = 6 * 3600
_availability_memo: dict[str, Any] = {"fetched_at": 0.0, "windows": None}

# The /api/area/ endpoint takes ONLY a bounding box "west,south,east,north".
# Country codes belong to a different endpoint (/api/country/csv/...) which
# answers "Invalid API call." for our key, so the tool resolves the handful
# of country codes callers already pass into a bbox before fetching. Values
# are W,S,E,N in degrees; "usa" is CONUS (Alaska/Hawaii excluded on purpose,
# they add ~40% empty ocean to every scan).
COUNTRY_BBOX = {
    "world": "-180,-90,180,90",
    "usa": "-125,24,-66,50",
    "can": "-141,41,-52,70",
    "mex": "-118,14,-86,33",
    "bra": "-74,-34,-34,6",
    "arg": "-74,-56,-53,-21",
    "chn": "73,18,135,54",
    "ind": "68,6,98,36",
    "idn": "95,-11,141,6",
    "aus": "112,-44,154,-10",
    "rus": "19,41,180,82",
    "zaf": "16,-35,33,-22",
    "nga": "2,4,15,14",
    "ukr": "22,44,41,53",
}

EONET_CATEGORIES = {
    "wildfires",
    "volcanoes",
    "severeStorms",
    "seaLakeIce",
    "earthquakes",
    "floods",
    "landslides",
    "drought",
    "dustHaze",
    "manmadeLake",
    "snow",
    "tempExtremes",
    "waterColor",
}


# ---------------------------------------------------------------------------
# Helpers — NDVI health classification
# ---------------------------------------------------------------------------


def _ndvi_health(value: float) -> str:
    """Classify NDVI value into human-readable health category.

    NDVI scale: -0.2 to 1.0 (after dividing by 10000).
    """
    if value < 0.0:
        return "water_or_barren"
    if value < 0.15:
        return "bare_soil"
    if value < 0.3:
        return "sparse"
    if value < 0.5:
        return "moderate"
    if value < 0.7:
        return "healthy"
    return "dense"


# ---------------------------------------------------------------------------
# Helpers — Grid-based hotspot clustering
# ---------------------------------------------------------------------------


def _cluster_hotspots(
    points: list[dict],
    cell_size_deg: float = 0.1,
) -> list[dict]:
    """Group fire hotspots into grid cells and return cluster summaries.

    Uses a simple grid approach: floor(lat/cell) and floor(lon/cell) form
    cell keys. Each cell becomes a cluster with count, avg FRP, centroid, etc.
    cell_size_deg ≈ 11 km at equator.
    """
    if not points:
        return []

    cells: dict[tuple[int, int], list[dict]] = {}
    for p in points:
        try:
            lat = float(p.get("latitude", 0))
            lon = float(p.get("longitude", 0))
        except (TypeError, ValueError):
            continue
        key = (
            int(math.floor(lat / cell_size_deg)),
            int(math.floor(lon / cell_size_deg)),
        )
        cells.setdefault(key, []).append(p)

    clusters: list[dict] = []
    for (lat_cell, lon_cell), pts in cells.items():
        frps = []
        lats = []
        lons = []
        for p in pts:
            try:
                frps.append(float(p.get("frp", 0)))
                lats.append(float(p["latitude"]))
                lons.append(float(p["longitude"]))
            except (TypeError, ValueError, KeyError):
                continue
        if not lats:
            continue
        clusters.append(
            {
                "centroid_lat": round(sum(lats) / len(lats), 4),
                "centroid_lon": round(sum(lons) / len(lons), 4),
                "count": len(pts),
                "avg_frp": round(sum(frps) / len(frps), 2) if frps else 0.0,
                "max_frp": round(max(frps), 2) if frps else 0.0,
                "total_frp": round(sum(frps), 2),
            }
        )

    clusters.sort(key=lambda c: c["total_frp"], reverse=True)
    return clusters


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------


def _parse_bbox(area: str) -> str | None:
    """Return *area* normalised as a FIRMS bbox, or None if it isn't one.

    FIRMS expects exactly four comma-separated numbers, west,south,east,north.
    Anything else (a country code, a place name, a 3-value typo) is rejected
    here so the caller can fail with a useful message instead of eating an
    HTTP 400 body.
    """
    parts = [p.strip() for p in area.split(",")]
    if len(parts) != 4:
        return None
    try:
        west, south, east, north = (float(p) for p in parts)
    except ValueError:
        return None
    if not (-180 <= west <= 180 and -180 <= east <= 180):
        return None
    if not (-90 <= south <= 90 and -90 <= north <= 90):
        return None
    if west >= east or south >= north:
        return None
    return f"{west:g},{south:g},{east:g},{north:g}"


def _resolve_firms_area(area: str) -> str | None:
    """Resolve a caller-supplied area to a FIRMS bbox, or None if unresolvable.

    Accepts either a literal bbox or one of the :data:`COUNTRY_BBOX` codes.
    """
    bbox = _parse_bbox(area)
    if bbox is not None:
        return bbox
    return COUNTRY_BBOX.get(area.strip().lower())


def _fetch_firms(
    area: str,
    source: str,
    days: int,
    api_key: str,
    date: str | None = None,
) -> list[dict] | None:
    """Fetch fire/thermal data from NASA FIRMS.

    *area* must already be a bbox "west,south,east,north" — see
    :func:`_resolve_firms_area`. *days* must be 1..5.

    *date* (YYYY-MM-DD) is the optional trailing path segment the FIRMS area
    endpoint accepts. It is the **start** of the *days*-long window, not the
    end: probed 2026-09-23 against the live API, ``days=3 date=2026-09-15``
    returned acq_dates {2026-09-15, 2026-09-16, 2026-09-17}, i.e.
    ``date .. date+days-1``. With no *date*, FIRMS instead returns a trailing
    window **ending** today (``days=5`` -> 2026-09-19..2026-09-23). The two
    forms have opposite anchors; callers backfilling a gap must pass the first
    missing day, not the last.

    Returns a list of hotspot dicts, or None on failure. An empty list means
    FIRMS served a well-formed CSV with no detections. A 200 whose body is not
    a FIRMS CSV at all (empty body, HTML error page, truncated proxy response)
    is a *failure*, not an empty day — FIRMS always sends the header row, so a
    body without :data:`FIRMS_REQUIRED_COLUMNS` never means "no fires".
    """
    url = f"{_FIRMS_BASE}/area/csv/{api_key}/{source}/{area}/{days}"
    if date:
        url = f"{url}/{date}"
    try:
        resp = httpx.get(
            url,
            headers={"User-Agent": _UA},
            timeout=_TIMEOUT,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            log.warning("FIRMS HTTP %d for %s", resp.status_code, area)
            return None

        text = resp.text.strip()
        if not text:
            # A 200 with no body is not "no fires" — FIRMS always emits the
            # CSV header. Treating this as [] is how a broken fetch became a
            # green zero row.
            log.warning("FIRMS returned HTTP 200 with an empty body for %s — treating as failure", area)
            return None

        reader = csv.DictReader(io.StringIO(text))
        fields = {(f or "").strip().lower() for f in (reader.fieldnames or [])}
        missing = [c for c in FIRMS_REQUIRED_COLUMNS if c not in fields]
        if missing:
            log.warning(
                "FIRMS HTTP 200 body is not a FIRMS CSV for %s (missing %s, header=%r) — treating as failure",
                area,
                missing,
                (reader.fieldnames or [])[:6],
            )
            return None

        return list(reader)

    except httpx.HTTPError as exc:
        log.warning("FIRMS fetch error: %s", exc)
        return None


def _fetch_firms_availability(api_key: str) -> dict[str, tuple[str, str]] | None:
    """Return ``{source: (min_date, max_date)}`` from FIRMS, or None if unknown.

    Same host, same key, free, and it does not spend the transaction budget.
    Memoised for :data:`FIRMS_AVAILABILITY_TTL` so a backfill loop asks once.

    Returns None — never a guess — when the endpoint cannot be read, so the
    caller can say "unverified" instead of inventing an archive window.
    """
    now = time.time()
    cached = _availability_memo.get("windows")
    if cached is not None and (now - _availability_memo.get("fetched_at", 0.0)) < FIRMS_AVAILABILITY_TTL:
        return cached

    url = f"{_FIRMS_BASE}/data_availability/csv/{api_key}/ALL"
    try:
        resp = httpx.get(
            url,
            headers={"User-Agent": _UA},
            timeout=_TIMEOUT,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            log.warning("FIRMS data_availability HTTP %d", resp.status_code)
            return None
        reader = csv.DictReader(io.StringIO(resp.text.strip()))
        windows: dict[str, tuple[str, str]] = {}
        for row in reader:
            data_id = (row.get("data_id") or "").strip()
            lo = (row.get("min_date") or "").strip()
            hi = (row.get("max_date") or "").strip()
            if data_id and _is_iso_date(lo) and _is_iso_date(hi):
                windows[data_id] = (lo, hi)
        if not windows:
            log.warning("FIRMS data_availability returned no parseable rows")
            return None
    except (httpx.HTTPError, csv.Error) as exc:
        log.warning("FIRMS data_availability fetch error: %s", exc)
        return None

    _availability_memo["windows"] = windows
    _availability_memo["fetched_at"] = now
    return windows


def _fetch_ndvi(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    km_radius: int = 0,
) -> dict | None:
    """Fetch NDVI time series from MODIS Web Service.

    Dates use MODIS format: A{YYYYDDD} where DDD is day of year.
    Returns parsed JSON or None on failure.
    """
    start_modis = _date_to_modis(start_date)
    end_modis = _date_to_modis(end_date)
    if start_modis is None or end_modis is None:
        return None

    url = f"{_MODIS_BASE}/MOD13Q1/subset"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "band": "250m_16_days_NDVI",
        "startDate": start_modis,
        "endDate": end_modis,
        "kmAboveBelow": km_radius,
        "kmLeftRight": km_radius,
    }
    try:
        resp = httpx.get(
            url,
            params=params,
            headers={"User-Agent": _UA},
            timeout=_TIMEOUT,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            log.warning("MODIS HTTP %d", resp.status_code)
            return None
        return resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("MODIS fetch error: %s", exc)
        return None


def _fetch_eonet(
    category: str | None = None,
    days: int = 30,
    status: str = "open",
    bbox: str | None = None,
) -> list[dict] | None:
    """Fetch natural events from NASA EONET v3.

    Returns list of event dicts or None on failure.
    """
    url = f"{_EONET_BASE}/events"
    params: dict[str, Any] = {
        "status": status,
        "days": days,
        "limit": 500,
    }
    if category:
        params["category"] = category
    if bbox:
        params["bbox"] = bbox

    try:
        resp = httpx.get(
            url,
            params=params,
            headers={"User-Agent": _UA},
            timeout=_TIMEOUT,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            log.warning("EONET HTTP %d", resp.status_code)
            return None
        data = resp.json()
        return data.get("events", [])
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("EONET fetch error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Date conversion helpers
# ---------------------------------------------------------------------------


def _date_to_modis(date_str: str) -> str | None:
    """Convert YYYY-MM-DD to MODIS format A{YYYYDDD}."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        doy = dt.timetuple().tm_yday
        return f"A{dt.year}{doy:03d}"
    except (ValueError, TypeError):
        return None


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert to float."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _is_iso_date(value: Any) -> bool:
    """True when *value* is a YYYY-MM-DD calendar date."""
    if not isinstance(value, str):
        return False
    try:
        datetime.strptime(value.strip(), "%Y-%m-%d")
    except ValueError:
        return False
    return True


def _window_bounds(start: str, days: int) -> tuple[str, str]:
    """Return (window_start, window_end) for a FIRMS *days*-long window.

    FIRMS anchors a dated request at the window START (probed 2026-09-23), so
    the window it serves is ``start .. start + days - 1`` inclusive.
    """
    dt = datetime.strptime(start, "%Y-%m-%d")
    return start, (dt + timedelta(days=days - 1)).strftime("%Y-%m-%d")


def _days_until(start: str, end: str) -> int:
    """Inclusive day count from *start* to *end*; 0 when *end* precedes *start*."""
    a = datetime.strptime(start, "%Y-%m-%d")
    b = datetime.strptime(end, "%Y-%m-%d")
    return max(0, (b - a).days + 1)


def _window_covers(window_start: str, window_end: str, lo: str | None, hi: str | None) -> bool:
    """True when the observed span *lo..hi* sits inside the claimed window.

    A False here means the tool's model of the FIRMS anchor is wrong — the
    row would describe a window the API did not serve. Probed 2026-09-23 the
    two anchors are: dated -> ``date..date+days-1``, undated -> trailing
    window ending today (days=3 -> 09-21..09-23). Both keep the observed
    acq_dates inside the claimed window, so a False is always a defect.
    """
    if lo is None or hi is None:
        return True
    return window_start <= lo and hi <= window_end


# ---------------------------------------------------------------------------
# Cache entry format
# ---------------------------------------------------------------------------
#
# The cache used to hold only the rendered human-readable string. On a hit the
# tool returned ToolResult(success=True, output=<str>, data=None), and
# ToolOperator's `result.data if result.data is not None else result.output`
# (agent/pipeline/operators.py) then persisted that prose blob as the node's
# row. The run was green, a row existed, and the convergence extractor —
# which returns [] for anything that is not a dict — yielded zero evidence
# from it. Caching the structured payload alongside the text closes that.
#
# The version tag makes a legacy string-only entry a cache MISS rather than a
# degraded hit; the entry then gets rewritten in the new shape on refetch.
_CACHE_ENTRY_VERSION = 2


def _cache_entry(output: str, data: dict) -> dict:
    """Wrap a rendered summary plus its structured payload for the cache."""
    return {"v": _CACHE_ENTRY_VERSION, "output": output, "data": data}


def _result_from_cache_entry(entry: Any) -> ToolResult | None:
    """Rebuild a ToolResult from a cache entry, or None if it is unusable.

    Unusable means: a legacy string-only entry, a wrong/absent version tag, or
    a missing structured payload. All of those are treated as a miss so the
    caller refetches — never as a success carrying no data.
    """
    if not isinstance(entry, dict):
        return None
    if entry.get("v") != _CACHE_ENTRY_VERSION:
        return None
    output = entry.get("output")
    data = entry.get("data")
    if not isinstance(output, str) or not isinstance(data, dict):
        return None
    return ToolResult(success=True, output=output, data=data)


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------


class SatelliteActivityTool(Tool):
    """Satellite-derived physical activity — fire hotspots, crop health, natural events."""

    name = "satellite_activity"
    description = (
        "Observe physical-world activity from space. "
        "Modes: 'fire' for NASA FIRMS thermal hotspots (industrial activity, "
        "wildfires — requires TIRRA_NASA_FIRMS_KEY), 'vegetation' for MODIS "
        "NDVI crop health assessment (no auth), 'events' for NASA EONET natural "
        "disaster tracking (no auth)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": sorted(VALID_MODES),
                "description": "events|fire|vegetation",
            },
            "area": {
                "type": "string",
                "description": (
                    "For fire mode: bounding box 'west,south,east,north' "
                    "(e.g. '-125,24,-66,50' for CONUS). A shorthand code from "
                    f"{sorted(COUNTRY_BBOX)} is resolved to a bbox; FIRMS itself "
                    "rejects bare country codes. For events mode: optional bbox "
                    "'west,south,east,north'."
                ),
            },
            "source": {
                "type": "string",
                "description": "Fire satellite source: VIIRS_NOAA20_NRT (default), VIIRS_SNPP_NRT, MODIS_NRT",
            },
            "days": {
                "type": "integer",
                "description": f"Fire: 1-{FIRMS_MAX_DAYS} (default 1). Events: 1-365 (default 30).",
            },
            "date": {
                "type": "string",
                "description": (
                    "Fire mode: optional START date YYYY-MM-DD for backfilling a "
                    "gap — FIRMS returns 'days' days BEGINNING on this date "
                    "(probed 2026-09-23). Omit it for a trailing window ending "
                    "today. The WHOLE window (date..date+days-1) must fall inside "
                    "the sensor's rolling NRT archive; a window that starts or "
                    "ends outside it is rejected rather than reported as a zero "
                    "or as a short window silently marked filled."
                ),
            },
            "latitude": {
                "type": "number",
                "description": "Vegetation mode: latitude (-90 to 90).",
            },
            "longitude": {
                "type": "number",
                "description": "Vegetation mode: longitude (-180 to 180).",
            },
            "start_date": {
                "type": "string",
                "description": "Vegetation mode: start date YYYY-MM-DD.",
            },
            "end_date": {
                "type": "string",
                "description": "Vegetation mode: end date YYYY-MM-DD.",
            },
            "km_radius": {
                "type": "integer",
                "description": "Vegetation mode: area radius in km (0-100, default 0).",
            },
            "category": {
                "type": "string",
                "description": "Events mode: category filter (wildfires, volcanoes, severeStorms, etc.).",
            },
            "status": {
                "type": "string",
                "description": "Events mode: event status (open/closed, default open).",
            },
        },
        "required": ["mode"],
    }

    def __init__(self, *, cache: DataCache | None = None) -> None:
        self._cache = cache
        self._firms_key = self._get_firms_key()

    @staticmethod
    def _get_firms_key() -> str | None:
        key = os.environ.get("TIRRA_NASA_FIRMS_KEY", "").strip()
        return key if key else None

    def execute(self, **kwargs: Any) -> ToolResult:
        from agent.preflight import FeaturePreflight  # noqa: PLC0415

        mode = (kwargs.get("mode") or "").strip().lower()
        if mode not in VALID_MODES:
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Use: {', '.join(sorted(VALID_MODES))}",
            )

        # Fire mode requires NASA FIRMS key — check before dispatch
        if mode == "fire":
            ok, pf = FeaturePreflight.for_api_key(
                key_value=self._firms_key,
                env_var="TIRRA_NASA_FIRMS_KEY",
                tool_name="SatelliteActivityTool (fire mode)",
                signup_url="https://firms.modaps.eosdis.nasa.gov/api/map_key/",
            )
            if not ok:
                return ToolResult(success=False, output=pf.user_message)
            return self._fire(**kwargs)

        if mode == "vegetation":
            return self._vegetation(**kwargs)
        return self._events(**kwargs)

    # ------------------------------------------------------------------
    # Mode: fire
    # ------------------------------------------------------------------

    def _fire(self, **kwargs: Any) -> ToolResult:

        area = (kwargs.get("area") or "").strip()
        if not area:
            return ToolResult(
                success=False,
                output=(
                    "Parameter 'area' required for fire mode. Use a bounding box "
                    "'west,south,east,north' (e.g. '-125,24,-66,50' for CONUS) or one "
                    f"of {sorted(COUNTRY_BBOX)}. FIRMS rejects bare country codes."
                ),
            )

        bbox = _resolve_firms_area(area)
        if bbox is None:
            return ToolResult(
                success=False,
                output=(
                    f"Invalid area '{area}'. Use a bounding box 'west,south,east,north' "
                    f"with west<east and south<north, or one of {sorted(COUNTRY_BBOX)}."
                ),
            )

        source = (kwargs.get("source") or "VIIRS_NOAA20_NRT").strip()
        if source not in FIRMS_SOURCES:
            return ToolResult(
                success=False,
                output=f"Invalid source '{source}'. Use: {', '.join(sorted(FIRMS_SOURCES))}",
            )

        days = kwargs.get("days", 1)
        try:
            days = int(days)
        except (TypeError, ValueError):
            days = 1
        days = max(1, min(FIRMS_MAX_DAYS, days))

        date = (kwargs.get("date") or "").strip() or None
        if date is not None and not _is_iso_date(date):
            return ToolResult(
                success=False,
                output=f"Invalid date '{date}'. Use YYYY-MM-DD.",
            )

        # The window FIRMS will actually serve. A dated request is anchored at
        # the window START (probed 2026-09-23: days=3 date=2026-09-15 returned
        # 09-15..09-17); an undated one is a trailing window ENDING today
        # (probed 2026-09-23: days=3 -> 09-21..09-23). Computed before the
        # archive guard because the guard has to check both ends of it.
        if date is not None:
            window_start, window_end = _window_bounds(date, days)
        else:
            today = datetime.now(tz=UTC)
            window_end = today.strftime("%Y-%m-%d")
            window_start = (today - timedelta(days=days - 1)).strftime("%Y-%m-%d")

        # Archive guard. A `date` outside the sensor's rolling NRT archive is
        # not an error to FIRMS — it answers HTTP 200 with a header-only CSV,
        # which parses to [] and used to be reported as "no thermal hotspots
        # detected". A gap-fill that fetched nothing then looked exactly like
        # a genuinely quiet day, and the convergence extractor turned the
        # fabricated zero into four real evidence items. Probed 2026-09-23:
        # date=2026-06-01 -> 200 / 0 rows, date=2026-09-01 -> 200 / 1501 rows,
        # while every genuine error (bad key, bad source, days=10) is a loud
        # HTTP 400. So this 200-with-nothing is the one case the HTTP check
        # cannot see, and it is exactly the case `date` exists to exercise.
        archive_min: str | None = None
        archive_max: str | None = None
        archive_checked = False
        if date is not None:
            windows = _fetch_firms_availability(self._firms_key)
            window = windows.get(source) if windows else None
            if window is not None:
                archive_min, archive_max = window
                archive_checked = True
                if date < archive_min or date > archive_max:
                    return ToolResult(
                        success=False,
                        output=(
                            f"date {date} is outside the {source} archive "
                            f"[{archive_min}..{archive_max}]. FIRMS answers such a window "
                            "with HTTP 200 and an empty CSV, so reporting it would "
                            "fabricate a zero. Pick a date inside the archive, or a "
                            "non-NRT source, and treat this window as unfilled."
                        ),
                    )
                # Both ends, not just the start. A window that BEGINS inside
                # the archive can still run off the end of it, and FIRMS
                # answers that with HTTP 200 carrying only the days it has —
                # probed live 2026-09-23: date=2026-09-23 days=5 (archive_max
                # 2026-09-23) -> HTTP 200, 76 rows, acq_dates {2026-09-23}.
                # The row then claimed window 09-23..09-27 with
                # empty_result=False and archive_checked=True, so a backfill
                # loop would mark four never-fetched days as filled. Checking
                # only `date` let that through.
                if window_end > archive_max:
                    servable = _days_until(date, archive_max)
                    return ToolResult(
                        success=False,
                        output=(
                            f"days={days} from {date} asks for {window_start}..{window_end}, "
                            f"but the {source} archive ends {archive_max}. FIRMS answers such a "
                            "window with HTTP 200 carrying only the days it holds, so reporting "
                            f"it would mark {days - servable} unfetched day(s) as filled. "
                            f"Use days<={servable} for this start date, and treat the remaining "
                            "day(s) as unfilled."
                        ),
                    )
            else:
                log.warning(
                    "FIRMS archive window for %s unknown — proceeding with date=%s unverified",
                    source,
                    date,
                )

        # Cache check. Keyed on the *resolved* bbox so "usa" and
        # "-125,24,-66,50" share one entry instead of double-spending the
        # key's 5000-transaction budget, and on date so a backfill never
        # serves today's answer for an older window.
        cache_ns = "satellite_fire"
        cache_key = f"{bbox}_{source}_{days}_{date or 'today'}"
        if self._cache:
            cached = self._cache.get(cache_ns, cache_key)
            hit = _result_from_cache_entry(cached)
            if hit is not None:
                return hit
            if cached is not None:
                log.debug("Ignoring legacy/unusable satellite_fire cache entry for %s", cache_key)

        hotspots = _fetch_firms(bbox, source, days, self._firms_key, date)
        if hotspots is None:
            return ToolResult(
                success=False,
                output=f"Failed to fetch FIRMS data for area={area} (bbox={bbox}).",
            )

        # Provenance every fire payload carries, so a stored row describes the
        # window it observed rather than only the wall clock at which it was
        # fetched (pipeline_data.fetched_at is the only other time on the row).
        provenance = {
            "mode": "fire",
            "area": area,
            "area_bbox": bbox,
            "source": source,
            "days": days,
            "date": date,
            "window_start": window_start,
            "window_end": window_end,
            "archive_min": archive_min,
            "archive_max": archive_max,
            "archive_checked": archive_checked,
        }

        if not hotspots:
            # A zero is not a collection. `status: "skipped"` is the literal
            # word agent/pipeline/operators.py:classify_payload_status reads,
            # so the node is recorded as skipped-with-a-reason instead of
            # "completed" with one row — which is what it used to be, and
            # which is indistinguishable from a healthy run in every
            # observable way (the executor stores only on "completed", so no
            # row lands, and agent/convergence/extractors.py can no longer
            # turn hotspot_count=0 into four evidence items).
            reason = (
                f"FIRMS served a well-formed CSV with zero hotspots for {bbox} "
                f"({window_start}..{window_end}, source={source}) — a continental "
                "window with no detections is a fetch to distrust, not a reading."
            )
            result = f"No thermal hotspots detected in {area} ({window_start}..{window_end}, source={source})."
            payload = {
                **provenance,
                "status": "skipped",
                "reason": reason,
                "hotspot_count": 0,
                "empty_result": True,
                "acq_date_min": None,
                "acq_date_max": None,
                "window_matches_observed": True,
                "frp_avg": 0.0,
                "frp_max": 0.0,
                "frp_total": 0.0,
                "confidence_counts": {},
                "daynight_counts": {},
                "cluster_count": 0,
                "clusters": [],
            }
            # Deliberately NOT cached. Caching a zero pins it for the cache's
            # whole TTL (6h in production, agent/data/cache.py:25), so one
            # transient bad window would be re-served to every run inside it.
            log.warning("satellite fire: %s", reason)
            return ToolResult(success=True, output=result, data=payload)

        # Compute stats
        frps = [_safe_float(h.get("frp")) for h in hotspots]
        frps_valid = [f for f in frps if f > 0]
        confs = {}
        for h in hotspots:
            c = str(h.get("confidence", "unknown")).lower()
            confs[c] = confs.get(c, 0) + 1
        daynight = {}
        for h in hotspots:
            dn = str(h.get("daynight", "?"))
            daynight[dn] = daynight.get(dn, 0) + 1

        # Observation timestamps. FIRMS carries acq_date on every row; dropping
        # it left fetched_at (wall clock) as the only time on the stored row,
        # so a backfilled August window landed stamped "today".
        acq_dates = sorted({d for h in hotspots if _is_iso_date(d := str(h.get("acq_date", "")).strip())})
        acq_date_min = acq_dates[0] if acq_dates else None
        acq_date_max = acq_dates[-1] if acq_dates else None

        # Cross-check the window the tool CLAIMS against the dates FIRMS
        # actually served. Nothing else can catch the anchor being wrong: the
        # `date` parameter was shipped documented as the window END when the
        # live API treats it as the START, and every assertion on the URL
        # string passed anyway. If this ever goes False the row is describing
        # a window it did not observe.
        window_matches_observed = _window_covers(window_start, window_end, acq_date_min, acq_date_max)
        if not window_matches_observed:
            log.warning(
                "satellite fire: observed acq_dates %s..%s fall outside the claimed window "
                "%s..%s (date=%s, days=%d) — the FIRMS window anchor is not what this tool assumes",
                acq_date_min,
                acq_date_max,
                window_start,
                window_end,
                date,
                days,
            )

        clusters = _cluster_hotspots(hotspots)

        lines = [
            f"🔥 FIRMS Thermal Hotspots — {area} ({days} day(s), {source})",
            f"Window: {window_start}..{window_end}"
            + (f" (observed {acq_date_min}..{acq_date_max})" if acq_date_min else ""),
            f"Total hotspots: {len(hotspots)}",
        ]
        if frps_valid:
            lines.append(
                f"FRP (MW): avg={sum(frps_valid) / len(frps_valid):.1f}, "
                f"max={max(frps_valid):.1f}, total={sum(frps_valid):.1f}"
            )
        lines.append(f"Confidence: {confs}")
        lines.append(f"Day/Night: {daynight}")

        if clusters:
            lines.append(f"\nTop clusters ({len(clusters)} total):")
            for i, cl in enumerate(clusters[:10]):
                lines.append(
                    f"  {i + 1}. ({cl['centroid_lat']}, {cl['centroid_lon']}) — "
                    f"{cl['count']} pts, FRP avg={cl['avg_frp']} max={cl['max_frp']} total={cl['total_frp']}"
                )

        result = "\n".join(lines)
        payload = {
            **provenance,
            "hotspot_count": len(hotspots),
            "empty_result": False,
            "acq_date_min": acq_date_min,
            "acq_date_max": acq_date_max,
            "window_matches_observed": window_matches_observed,
            "frp_avg": (round(sum(frps_valid) / len(frps_valid), 2) if frps_valid else 0.0),
            "frp_max": round(max(frps_valid), 2) if frps_valid else 0.0,
            "frp_total": round(sum(frps_valid), 2) if frps_valid else 0.0,
            "confidence_counts": confs,
            "daynight_counts": daynight,
            "cluster_count": len(clusters),
            "clusters": clusters[:10],
        }
        if self._cache:
            self._cache.put(cache_ns, cache_key, _cache_entry(result, payload))
        return ToolResult(success=True, output=result, data=payload)

    # ------------------------------------------------------------------
    # Mode: vegetation
    # ------------------------------------------------------------------

    def _vegetation(self, **kwargs: Any) -> ToolResult:
        lat = kwargs.get("latitude")
        lon = kwargs.get("longitude")
        start = kwargs.get("start_date")
        end = kwargs.get("end_date")

        if lat is None or lon is None:
            return ToolResult(
                success=False,
                output="Parameters 'latitude' and 'longitude' required for vegetation mode.",
            )
        try:
            lat = float(lat)
            lon = float(lon)
        except (TypeError, ValueError):
            return ToolResult(
                success=False,
                output="latitude/longitude must be numeric.",
            )
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return ToolResult(
                success=False,
                output="latitude must be -90..90, longitude must be -180..180.",
            )

        if not start or not end:
            return ToolResult(
                success=False,
                output="Parameters 'start_date' and 'end_date' required (YYYY-MM-DD).",
            )

        km = kwargs.get("km_radius", 0)
        try:
            km = int(km)
        except (TypeError, ValueError):
            km = 0
        km = max(0, min(100, km))

        # Cache check
        cache_ns = "satellite_vegetation"
        cache_key = f"{lat}_{lon}_{start}_{end}_{km}"
        if self._cache:
            cached = self._cache.get(cache_ns, cache_key)
            hit = _result_from_cache_entry(cached)
            if hit is not None:
                return hit
            if cached is not None:
                log.debug("Ignoring legacy/unusable satellite_vegetation cache entry for %s", cache_key)

        data = _fetch_ndvi(lat, lon, start, end, km)
        if data is None:
            return ToolResult(
                success=False,
                output="Failed to fetch NDVI data from MODIS.",
            )

        # Parse subset
        subset = data.get("subset", [])
        if not subset:
            result = f"No NDVI data available for ({lat}, {lon}) from {start} to {end}."
            payload = {
                "mode": "vegetation",
                "latitude": lat,
                "longitude": lon,
                "start_date": start,
                "end_date": end,
                "observation_count": 0,
                "latest_ndvi": 0.0,
                "latest_date": "",
                "latest_health": "bare_soil",
                "avg_ndvi": 0.0,
                "min_ndvi": 0.0,
                "max_ndvi": 0.0,
                "anomaly_pct": 0.0,
                "series": [],
            }
            if self._cache:
                self._cache.put(cache_ns, cache_key, _cache_entry(result, payload))
            return ToolResult(success=True, output=result, data=payload)

        # Extract time series
        series: list[dict] = []
        for entry in subset:
            cal_date = entry.get("calendar_date", "")
            raw_values = entry.get("data", [])
            if not raw_values:
                continue
            # Each data point is scaled by 0.0001
            scale = _safe_float(entry.get("scale", 0.0001), 0.0001)
            values = [_safe_float(v) * scale for v in raw_values]
            mean_ndvi = sum(values) / len(values) if values else 0.0
            series.append(
                {
                    "date": cal_date,
                    "ndvi": round(mean_ndvi, 4),
                    "health": _ndvi_health(mean_ndvi),
                    "pixels": len(values),
                }
            )

        if not series:
            result = f"No valid NDVI observations for ({lat}, {lon}) in date range."
            payload = {
                "mode": "vegetation",
                "latitude": lat,
                "longitude": lon,
                "start_date": start,
                "end_date": end,
                "observation_count": 0,
                "latest_ndvi": 0.0,
                "latest_date": "",
                "latest_health": "bare_soil",
                "avg_ndvi": 0.0,
                "min_ndvi": 0.0,
                "max_ndvi": 0.0,
                "anomaly_pct": 0.0,
                "series": [],
            }
            if self._cache:
                self._cache.put(cache_ns, cache_key, _cache_entry(result, payload))
            return ToolResult(success=True, output=result, data=payload)

        ndvi_values = [s["ndvi"] for s in series]
        avg_ndvi = sum(ndvi_values) / len(ndvi_values)
        latest = series[-1]

        # Anomaly: compare latest to historical mean
        anomaly_pct = 0.0
        if avg_ndvi != 0:
            anomaly_pct = ((latest["ndvi"] - avg_ndvi) / abs(avg_ndvi)) * 100

        lines = [
            f"🌱 NDVI Crop Health — ({lat}, {lon})",
            f"Period: {start} to {end} ({len(series)} observations)",
            f"Latest: NDVI={latest['ndvi']:.4f} ({latest['health']}), date={latest['date']}",
            f"Historical avg: {avg_ndvi:.4f} ({_ndvi_health(avg_ndvi)})",
            f"Anomaly vs mean: {anomaly_pct:+.1f}%",
            f"Min: {min(ndvi_values):.4f}, Max: {max(ndvi_values):.4f}",
            "",
            "Time series:",
        ]
        for s in series:
            lines.append(f"  {s['date']}: NDVI={s['ndvi']:.4f} ({s['health']})")

        result = "\n".join(lines)
        payload = {
            "mode": "vegetation",
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "observation_count": len(series),
            "latest_ndvi": round(latest["ndvi"], 4),
            "latest_date": latest["date"],
            "latest_health": latest["health"],
            "avg_ndvi": round(avg_ndvi, 4),
            "min_ndvi": round(min(ndvi_values), 4),
            "max_ndvi": round(max(ndvi_values), 4),
            "anomaly_pct": round(anomaly_pct, 2),
            "series": series,
        }
        if self._cache:
            self._cache.put(cache_ns, cache_key, _cache_entry(result, payload))
        return ToolResult(success=True, output=result, data=payload)

    # ------------------------------------------------------------------
    # Mode: events
    # ------------------------------------------------------------------

    def _events(self, **kwargs: Any) -> ToolResult:
        category = (kwargs.get("category") or "").strip().lower() or None
        if category and category not in EONET_CATEGORIES:
            return ToolResult(
                success=False,
                output=f"Invalid category '{category}'. Use: {', '.join(sorted(EONET_CATEGORIES))}",
            )

        days = kwargs.get("days", 30)
        try:
            days = int(days)
        except (TypeError, ValueError):
            days = 30
        days = max(1, min(365, days))

        status = (kwargs.get("status") or "open").strip().lower()
        if status not in ("open", "closed"):
            return ToolResult(
                success=False,
                output="Status must be 'open' or 'closed'.",
            )

        bbox = (kwargs.get("bbox") or kwargs.get("area") or "").strip() or None

        # Cache check
        cache_ns = "satellite_events"
        cache_key = f"{category}_{days}_{status}_{bbox}"
        if self._cache:
            cached = self._cache.get(cache_ns, cache_key)
            hit = _result_from_cache_entry(cached)
            if hit is not None:
                return hit
            if cached is not None:
                log.debug("Ignoring legacy/unusable satellite_events cache entry for %s", cache_key)

        events = _fetch_eonet(category, days, status, bbox)
        if events is None:
            return ToolResult(
                success=False,
                output="Failed to fetch EONET events.",
            )

        if not events:
            result = f"No {status} natural events in last {days} day(s)"
            if category:
                result += f" (category={category})"
            result += "."
            payload = {
                "mode": "events",
                "days": days,
                "status": status,
                "category_filter": category,
                "event_count": 0,
                "category_counts": {},
                "events": [],
            }
            if self._cache:
                self._cache.put(cache_ns, cache_key, _cache_entry(result, payload))
            return ToolResult(success=True, output=result, data=payload)

        # Categorize
        cat_counts: dict[str, int] = {}
        for ev in events:
            cats = ev.get("categories", [])
            for c in cats:
                cid = c.get("id", "unknown")
                cat_counts[cid] = cat_counts.get(cid, 0) + 1

        lines = [
            f"🌍 EONET Natural Events — {status} (last {days} day(s))",
            f"Total events: {len(events)}",
            f"Categories: {cat_counts}",
            "",
        ]

        for i, ev in enumerate(events[:20]):
            title = ev.get("title", "Unknown")
            cats = ", ".join(c.get("id", "?") for c in ev.get("categories", []))
            geom = ev.get("geometry", [])
            loc = ""
            if geom:
                last_geom = geom[-1]
                coords = last_geom.get("coordinates", [])
                if coords and isinstance(coords, list) and len(coords) >= 2:
                    loc = f"({coords[1]}, {coords[0]})"
                date = last_geom.get("date", "")
                if date:
                    loc += f" @ {date[:10]}"
            lines.append(f"  {i + 1}. [{cats}] {title} {loc}")

        if len(events) > 20:
            lines.append(f"  ... and {len(events) - 20} more events")

        # Build structured event list for data= dict
        structured_events = []
        for ev in events:
            title = ev.get("title", "Unknown")
            cats = [c.get("id", "unknown") for c in ev.get("categories", [])]
            ev_lat, ev_lon, ev_date = None, None, None
            geom = ev.get("geometry", [])
            if geom:
                last_geom = geom[-1]
                coords = last_geom.get("coordinates", [])
                if isinstance(coords, list) and len(coords) >= 2:
                    ev_lon = coords[0]
                    ev_lat = coords[1]
                ev_date = (last_geom.get("date") or "")[:10] or None
            structured_events.append(
                {
                    "title": title,
                    "categories": cats,
                    "lat": ev_lat,
                    "lon": ev_lon,
                    "date": ev_date,
                }
            )

        result = "\n".join(lines)
        payload = {
            "mode": "events",
            "days": days,
            "status": status,
            "category_filter": category,
            "event_count": len(events),
            "category_counts": cat_counts,
            "events": structured_events,
        }
        if self._cache:
            self._cache.put(cache_ns, cache_key, _cache_entry(result, payload))
        return ToolResult(success=True, output=result, data=payload)
