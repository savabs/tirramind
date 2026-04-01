"""
Tool: Earthquake Proximity — USGS Earthquake + Critical Infrastructure Overlay

USGS Earthquake Hazards API: free, global, real-time, GeoJSON.
https://earthquake.usgs.gov/fdsnws/event/1/

Everyone sees the earthquake headline. Nobody auto-cross-references it with
the specific factory, pipeline, mine, or port in the blast radius.

Modes:
  recent   — Recent significant earthquakes worldwide. Filter by magnitude,
             time range. Annotates each quake with nearby infrastructure.
  monitor  — Check specific infrastructure zones for recent seismic activity.
             "Has anything shaken near TSMC fabs in the last 7 days?"
  infrastructure — List all monitored infrastructure zones.

Signal theory:
  - M6+ near TSMC Hsinchu → semiconductor supply disruption
  - M5+ near Chilean copper mines → copper price spike
  - M4+ near Japan nuclear plants → energy policy / safety shutdowns
  - Induced seismicity in Permian Basin → fracking regulation risk
  - M5+ near major ports → shipping delays
  - Swarm activity → potential larger event warning
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

_USGS_BASE = "https://earthquake.usgs.gov/fdsnws/event/1"
_UA = "TirraMind/0.1"
_TIMEOUT = 15

# Critical infrastructure positions: (name, lat, lon, radius_km, sector, detail)
CRITICAL_INFRA: list[dict[str, Any]] = [
    # Semiconductors
    {
        "name": "TSMC Hsinchu",
        "lat": 24.78,
        "lon": 120.99,
        "radius_km": 100,
        "sector": "semiconductor",
        "detail": "TSMC fabs 12/14/15",
    },
    {
        "name": "TSMC Tainan",
        "lat": 23.05,
        "lon": 120.31,
        "radius_km": 80,
        "sector": "semiconductor",
        "detail": "TSMC Fab 18 (3nm/5nm)",
    },
    {
        "name": "Samsung Pyeongtaek",
        "lat": 36.99,
        "lon": 127.11,
        "radius_km": 80,
        "sector": "semiconductor",
        "detail": "Samsung foundry mega-campus",
    },
    # Mining
    {
        "name": "Escondida Copper Mine",
        "lat": -24.27,
        "lon": -69.07,
        "radius_km": 100,
        "sector": "mining",
        "detail": "World's largest copper mine (Chile)",
    },
    {
        "name": "Chuquicamata Copper",
        "lat": -22.31,
        "lon": -68.90,
        "radius_km": 80,
        "sector": "mining",
        "detail": "Chile copper belt",
    },
    {
        "name": "Grasberg Gold/Copper",
        "lat": -4.05,
        "lon": 137.11,
        "radius_km": 80,
        "sector": "mining",
        "detail": "Indonesia (Freeport)",
    },
    {
        "name": "Indonesia Nickel Belt",
        "lat": -2.5,
        "lon": 121.5,
        "radius_km": 200,
        "sector": "mining",
        "detail": "Sulawesi nickel smelters",
    },
    # Nuclear
    {
        "name": "Fukushima Daiichi",
        "lat": 37.42,
        "lon": 141.03,
        "radius_km": 80,
        "sector": "nuclear",
        "detail": "Japan nuclear (decommissioning)",
    },
    {
        "name": "Kashiwazaki-Kariwa",
        "lat": 37.43,
        "lon": 138.60,
        "radius_km": 80,
        "sector": "nuclear",
        "detail": "World's largest nuclear plant",
    },
    {
        "name": "Turkey Point Nuclear",
        "lat": 25.43,
        "lon": -80.33,
        "radius_km": 80,
        "sector": "nuclear",
        "detail": "Florida nuclear plant",
    },
    # Energy pipelines
    {
        "name": "BTC Pipeline (Turkey)",
        "lat": 39.9,
        "lon": 43.0,
        "radius_km": 150,
        "sector": "energy",
        "detail": "Baku-Tbilisi-Ceyhan oil pipeline",
    },
    {
        "name": "Permian Basin",
        "lat": 31.9,
        "lon": -102.1,
        "radius_km": 200,
        "sector": "energy",
        "detail": "US shale oil/gas (induced seismicity)",
    },
    {
        "name": "Oklahoma Injection",
        "lat": 35.5,
        "lon": -97.5,
        "radius_km": 150,
        "sector": "energy",
        "detail": "Wastewater injection / induced quakes",
    },
    # Ports & logistics
    {
        "name": "Port of Los Angeles",
        "lat": 33.73,
        "lon": -118.27,
        "radius_km": 60,
        "sector": "logistics",
        "detail": "Largest US container port",
    },
    {
        "name": "Port of Shanghai",
        "lat": 30.63,
        "lon": 122.07,
        "radius_km": 80,
        "sector": "logistics",
        "detail": "World's busiest port",
    },
    {
        "name": "Strait of Hormuz",
        "lat": 26.5,
        "lon": 56.3,
        "radius_km": 150,
        "sector": "energy",
        "detail": "20% of world oil transit",
    },
    # Agriculture
    {
        "name": "NZ Dairy Waikato",
        "lat": -37.8,
        "lon": 175.3,
        "radius_km": 100,
        "sector": "agriculture",
        "detail": "New Zealand dairy heartland",
    },
    {
        "name": "Japan Pacific Coast",
        "lat": 35.5,
        "lon": 140.5,
        "radius_km": 200,
        "sector": "industrial",
        "detail": "Tokyo-Yokohama industrial corridor",
    },
    # Data centers
    {
        "name": "Northern Virginia DCs",
        "lat": 39.04,
        "lon": -77.49,
        "radius_km": 60,
        "sector": "tech",
        "detail": "Ashburn — world's largest DC cluster",
    },
]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate distance in km using equirectangular approximation.

    Good enough for earthquake proximity (within ~1% error at these scales).
    """
    import math

    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1) * math.cos((lat1_r + lat2_r) / 2)
    return math.sqrt(dlat**2 + dlon**2) * 6371


def _mag_label(mag: float) -> str:
    """Human-readable magnitude category."""
    if mag >= 8.0:
        return "GREAT"
    if mag >= 7.0:
        return "MAJOR"
    if mag >= 6.0:
        return "STRONG"
    if mag >= 5.0:
        return "MODERATE"
    if mag >= 4.0:
        return "LIGHT"
    return "MINOR"


def _find_nearby_infra(lat: float, lon: float) -> list[dict[str, Any]]:
    """Find all infrastructure zones within range of a point."""
    nearby = []
    for infra in CRITICAL_INFRA:
        dist = _haversine_km(lat, lon, infra["lat"], infra["lon"])
        if dist <= infra["radius_km"]:
            nearby.append({**infra, "distance_km": round(dist, 1)})
    nearby.sort(key=lambda x: x["distance_km"])
    return nearby


def _format_quake(feat: dict[str, Any]) -> dict[str, Any]:
    """Normalize a USGS earthquake feature into a clean dict."""
    props = feat.get("properties", {})
    coords = feat.get("geometry", {}).get("coordinates", [0, 0, 0])
    lon, lat, depth = coords[0], coords[1], coords[2] if len(coords) > 2 else 0

    mag = props.get("mag") or 0
    place = props.get("place", "")
    time_ms = props.get("time")
    time_str = ""
    if time_ms:
        try:
            time_str = datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M UTC"
            )
        except (ValueError, OSError):
            pass

    nearby = _find_nearby_infra(lat, lon)

    return {
        "magnitude": round(mag, 1),
        "mag_label": _mag_label(mag),
        "place": place,
        "lat": round(lat, 3),
        "lon": round(lon, 3),
        "depth_km": round(depth, 1),
        "time": time_str,
        "tsunami": bool(props.get("tsunami")),
        "alert": props.get("alert"),  # green/yellow/orange/red
        "significance": props.get("sig", 0),
        "url": props.get("url", ""),
        "nearby_infrastructure": nearby,
    }


class EarthquakeProximityTool(Tool):
    name = "earthquake_proximity"
    description = (
        "Monitor global earthquakes and cross-reference with critical infrastructure. "
        "USGS API — free, real-time, global. Mode 'recent' shows significant quakes "
        "worldwide with nearby infrastructure overlay (TSMC fabs, copper mines, "
        "nuclear plants, pipelines, ports). Mode 'monitor' checks specific zones "
        "for seismic activity. Mode 'infrastructure' lists all monitored zones."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["recent", "monitor", "infrastructure"],
                "default": "recent",
                "description": (
                    "recent = significant quakes worldwide. "
                    "monitor = check specific infrastructure zone. "
                    "infrastructure = list all monitored zones."
                ),
            },
            "min_magnitude": {
                "type": "number",
                "default": 4.0,
                "description": "Minimum magnitude. Default 4.0.",
            },
            "days_back": {
                "type": "integer",
                "default": 7,
                "description": "How many days of history. Default 7, max 30.",
            },
            "zone": {
                "type": "string",
                "default": "",
                "description": (
                    "Infrastructure zone name for 'monitor' mode. "
                    "E.g., 'TSMC Hsinchu', 'Permian Basin', 'Escondida'. "
                    "Case-insensitive partial match."
                ),
            },
            "infra_only": {
                "type": "boolean",
                "default": False,
                "description": "If true, only show quakes near monitored infrastructure.",
            },
            "limit": {
                "type": "integer",
                "default": 25,
                "description": "Max results. Default 25, max 200.",
            },
        },
        "required": [],
    }

    def __init__(self, cache: DataCache | None = None) -> None:
        self._cache = cache

    def execute(
        self,
        *,
        mode: str = "recent",
        min_magnitude: float = 4.0,
        days_back: int = 7,
        zone: str = "",
        infra_only: bool = False,
        limit: int = 25,
        **_: Any,
    ) -> ToolResult:
        mode = mode.lower().strip()
        if mode not in ("recent", "monitor", "infrastructure"):
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Use 'recent', 'monitor', or 'infrastructure'.",
            )

        days_back = max(1, min(days_back, 30))
        limit = max(1, min(limit, 200))
        min_magnitude = max(0.0, min(min_magnitude, 10.0))

        if mode == "infrastructure":
            return self._list_infrastructure()

        if mode == "monitor":
            if not zone.strip():
                return ToolResult(
                    success=False,
                    output="Monitor mode requires a 'zone' parameter. Use mode='infrastructure' to list zones.",
                )
            return self._execute_monitor(
                zone=zone,
                min_magnitude=min_magnitude,
                days_back=days_back,
                limit=limit,
            )

        # recent
        return self._execute_recent(
            min_magnitude=min_magnitude,
            days_back=days_back,
            infra_only=infra_only,
            limit=limit,
        )

    # ------------------------------------------------------------------
    # recent mode
    # ------------------------------------------------------------------

    def _execute_recent(
        self,
        *,
        min_magnitude: float,
        days_back: int,
        infra_only: bool,
        limit: int,
    ) -> ToolResult:
        quakes, error = self._fetch_usgs(
            min_magnitude=min_magnitude,
            days_back=days_back,
        )
        if error:
            return ToolResult(success=False, output=error)

        formatted = [_format_quake(f) for f in quakes]

        if infra_only:
            formatted = [q for q in formatted if q["nearby_infrastructure"]]

        formatted.sort(key=lambda q: -q["magnitude"])
        formatted = formatted[:limit]

        if not formatted:
            return ToolResult(
                success=True,
                output=f"USGS: No earthquakes M{min_magnitude}+ in last {days_back}d"
                + (" near infrastructure" if infra_only else "")
                + ".",
                data={"quakes": [], "count": 0},
            )

        lines = [
            f"USGS Earthquakes: {len(formatted)} events "
            f"(M{min_magnitude}+, last {days_back}d):",
            "",
        ]
        for q in formatted:
            alert_str = f" ALERT={q['alert'].upper()}" if q["alert"] else ""
            tsunami_str = " TSUNAMI" if q["tsunami"] else ""
            lines.append(
                f"  M{q['magnitude']} [{q['mag_label']:8s}] {q['place']}"
                f"{alert_str}{tsunami_str}"
            )
            lines.append(
                f"    {q['time']}  depth={q['depth_km']}km  "
                f"({q['lat']}, {q['lon']})"
            )
            if q["nearby_infrastructure"]:
                for infra in q["nearby_infrastructure"][:3]:
                    lines.append(
                        f"    ⚠ {infra['distance_km']}km from {infra['name']} "
                        f"({infra['sector']}) — {infra['detail']}"
                    )
            lines.append("")

        # Count infrastructure proximity hits
        infra_hits = sum(1 for q in formatted if q["nearby_infrastructure"])

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "quakes": formatted,
                "count": len(formatted),
                "near_infrastructure": infra_hits,
            },
        )

    # ------------------------------------------------------------------
    # monitor mode
    # ------------------------------------------------------------------

    def _execute_monitor(
        self,
        *,
        zone: str,
        min_magnitude: float,
        days_back: int,
        limit: int,
    ) -> ToolResult:
        # Find matching zone
        zone_lower = zone.strip().lower()
        matched = [z for z in CRITICAL_INFRA if zone_lower in z["name"].lower()]
        if not matched:
            return ToolResult(
                success=False,
                output=f"No infrastructure zone matching '{zone}'. "
                "Use mode='infrastructure' to see all zones.",
            )

        target = matched[0]

        # Fetch quakes near the zone with a lower magnitude threshold
        quakes, error = self._fetch_usgs(
            min_magnitude=max(min_magnitude - 1.0, 1.0),
            days_back=days_back,
        )
        if error:
            return ToolResult(success=False, output=error)

        # Filter to quakes near the target zone
        nearby_quakes = []
        for feat in quakes:
            coords = feat.get("geometry", {}).get("coordinates", [0, 0, 0])
            lon, lat = coords[0], coords[1]
            dist = _haversine_km(lat, lon, target["lat"], target["lon"])
            if dist <= target["radius_km"]:
                q = _format_quake(feat)
                q["distance_to_zone_km"] = round(dist, 1)
                nearby_quakes.append(q)

        nearby_quakes.sort(key=lambda q: -q["magnitude"])
        nearby_quakes = nearby_quakes[:limit]

        if not nearby_quakes:
            return ToolResult(
                success=True,
                output=(
                    f"USGS Monitor: No earthquakes M{max(min_magnitude-1,1)}+ "
                    f"within {target['radius_km']}km of {target['name']} "
                    f"in last {days_back}d. Zone is quiet."
                ),
                data={"zone": target, "quakes": [], "count": 0},
            )

        lines = [
            f"USGS Monitor: {target['name']} ({target['sector']})",
            f"  {target['detail']}",
            f"  Radius: {target['radius_km']}km, {len(nearby_quakes)} quakes "
            f"in last {days_back}d:",
            "",
        ]
        for q in nearby_quakes:
            lines.append(
                f"  M{q['magnitude']} [{q['mag_label']:8s}] "
                f"{q.get('distance_to_zone_km', '?')}km away — {q['place']}"
            )
            lines.append(f"    {q['time']}  depth={q['depth_km']}km")
        lines.append("")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "zone": target,
                "quakes": nearby_quakes,
                "count": len(nearby_quakes),
            },
        )

    # ------------------------------------------------------------------
    # infrastructure listing
    # ------------------------------------------------------------------

    def _list_infrastructure(self) -> ToolResult:
        lines = [
            f"Monitored Infrastructure Zones ({len(CRITICAL_INFRA)}):",
            "",
        ]
        by_sector: dict[str, list[dict[str, Any]]] = {}
        for infra in CRITICAL_INFRA:
            by_sector.setdefault(infra["sector"], []).append(infra)

        for sector, zones in sorted(by_sector.items()):
            lines.append(f"  {sector.upper()}:")
            for z in zones:
                lines.append(
                    f"    {z['name']:30s}  ({z['lat']:.1f}, {z['lon']:.1f})  "
                    f"r={z['radius_km']}km — {z['detail']}"
                )
            lines.append("")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"zones": CRITICAL_INFRA, "count": len(CRITICAL_INFRA)},
        )

    # ------------------------------------------------------------------
    # USGS fetch
    # ------------------------------------------------------------------

    def _fetch_usgs(
        self,
        *,
        min_magnitude: float,
        days_back: int,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Fetch from USGS earthquake API. Returns (features, error)."""
        now = datetime.now(timezone.utc)
        start = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")

        cache_key = {"min_mag": min_magnitude, "start": start, "end": end}
        if self._cache:
            cached = self._cache.get("earthquake_proximity", cache_key)
            if cached is not None:
                return cached.get("features", []), None

        params = {
            "format": "geojson",
            "starttime": start,
            "endtime": end,
            "minmagnitude": str(min_magnitude),
            "orderby": "magnitude",
            "limit": "500",
        }
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{_USGS_BASE}/query?{qs}"

        try:
            with httpx.Client(
                timeout=_TIMEOUT,
                headers={"User-Agent": _UA},
            ) as client:
                resp = client.get(url)
                if resp.status_code == 429:
                    return [], "USGS API: Rate limited."
                if resp.status_code >= 400:
                    return [], f"USGS API error: HTTP {resp.status_code}"
                data = resp.json()
        except httpx.TimeoutException:
            return [], "USGS API: Request timed out."
        except Exception as exc:
            log.exception("USGS fetch failed")
            return [], f"USGS fetch error: {exc}"

        features = data.get("features", [])

        if self._cache and features:
            self._cache.put(
                "earthquake_proximity", cache_key, {"features": features}, ttl=1800
            )

        return features, None
