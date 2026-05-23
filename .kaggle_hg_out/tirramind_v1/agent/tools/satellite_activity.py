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
from datetime import datetime
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


def _fetch_firms(
    area: str,
    source: str,
    days: int,
    api_key: str,
) -> list[dict] | None:
    """Fetch fire/thermal data from NASA FIRMS.

    Returns list of hotspot dicts or None on failure.
    """
    url = f"{_FIRMS_BASE}/area/csv/{api_key}/{source}/{area}/{days}"
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
            return []

        reader = csv.DictReader(io.StringIO(text))
        return list(reader)

    except httpx.HTTPError as exc:
        log.warning("FIRMS fetch error: %s", exc)
        return None


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
                    "For fire mode: country code (USA, BRA, CHN) or "
                    "bounding box 'west,south,east,north'. For events mode: "
                    "optional bbox 'west,south,east,north'."
                ),
            },
            "source": {
                "type": "string",
                "description": "Fire satellite source: VIIRS_NOAA20_NRT (default), VIIRS_SNPP_NRT, MODIS_NRT",
            },
            "days": {
                "type": "integer",
                "description": "Fire: 1-10 (default 1). Events: 1-365 (default 30).",
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
        mode = (kwargs.get("mode") or "").strip().lower()
        if mode not in VALID_MODES:
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Use: {', '.join(sorted(VALID_MODES))}",
            )
        if mode == "fire":
            return self._fire(**kwargs)
        if mode == "vegetation":
            return self._vegetation(**kwargs)
        return self._events(**kwargs)

    # ------------------------------------------------------------------
    # Mode: fire
    # ------------------------------------------------------------------

    def _fire(self, **kwargs: Any) -> ToolResult:
        if not self._firms_key:
            return ToolResult(
                success=False,
                output="NASA FIRMS API key required. Set TIRRA_NASA_FIRMS_KEY.",
            )

        area = (kwargs.get("area") or "").strip()
        if not area:
            return ToolResult(
                success=False,
                output="Parameter 'area' required for fire mode. Use country code (USA) or bbox (W,S,E,N).",
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
        days = max(1, min(10, days))

        # Cache check
        cache_ns = "satellite_fire"
        cache_key = f"{area}_{source}_{days}"
        if self._cache:
            cached = self._cache.get(cache_ns, cache_key)
            if cached is not None:
                return ToolResult(success=True, output=cached)

        hotspots = _fetch_firms(area, source, days, self._firms_key)
        if hotspots is None:
            return ToolResult(
                success=False,
                output=f"Failed to fetch FIRMS data for area={area}.",
            )

        if not hotspots:
            result = f"No thermal hotspots detected in {area} (last {days} day(s), source={source})."
            if self._cache:
                self._cache.put(cache_ns, cache_key, result)
            return ToolResult(
                success=True,
                output=result,
                data={
                    "mode": "fire",
                    "area": area,
                    "source": source,
                    "days": days,
                    "hotspot_count": 0,
                    "frp_avg": 0.0,
                    "frp_max": 0.0,
                    "frp_total": 0.0,
                    "confidence_counts": {},
                    "daynight_counts": {},
                    "cluster_count": 0,
                    "clusters": [],
                },
            )

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

        clusters = _cluster_hotspots(hotspots)

        lines = [
            f"🔥 FIRMS Thermal Hotspots — {area} ({days} day(s), {source})",
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
        if self._cache:
            self._cache.put(cache_ns, cache_key, result)
        return ToolResult(
            success=True,
            output=result,
            data={
                "mode": "fire",
                "area": area,
                "source": source,
                "days": days,
                "hotspot_count": len(hotspots),
                "frp_avg": (round(sum(frps_valid) / len(frps_valid), 2) if frps_valid else 0.0),
                "frp_max": round(max(frps_valid), 2) if frps_valid else 0.0,
                "frp_total": round(sum(frps_valid), 2) if frps_valid else 0.0,
                "confidence_counts": confs,
                "daynight_counts": daynight,
                "cluster_count": len(clusters),
                "clusters": clusters[:10],
            },
        )

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
            if cached is not None:
                return ToolResult(success=True, output=cached)

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
            if self._cache:
                self._cache.put(cache_ns, cache_key, result)
            return ToolResult(
                success=True,
                output=result,
                data={
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
                },
            )

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
            if self._cache:
                self._cache.put(cache_ns, cache_key, result)
            return ToolResult(
                success=True,
                output=result,
                data={
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
                },
            )

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
        if self._cache:
            self._cache.put(cache_ns, cache_key, result)
        return ToolResult(
            success=True,
            output=result,
            data={
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
            },
        )

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
            if cached is not None:
                return ToolResult(success=True, output=cached)

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
            if self._cache:
                self._cache.put(cache_ns, cache_key, result)
            return ToolResult(
                success=True,
                output=result,
                data={
                    "mode": "events",
                    "days": days,
                    "status": status,
                    "category_filter": category,
                    "event_count": 0,
                    "category_counts": {},
                    "events": [],
                },
            )

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
        if self._cache:
            self._cache.put(cache_ns, cache_key, result)
        return ToolResult(
            success=True,
            output=result,
            data={
                "mode": "events",
                "days": days,
                "status": status,
                "category_filter": category,
                "event_count": len(events),
                "category_counts": cat_counts,
                "events": structured_events,
            },
        )
