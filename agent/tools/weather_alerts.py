"""
Tool: Weather Alerts — NOAA NWS Active Weather Alerts + NASA FIRMS Fire Detection

Two complementary free APIs, no auth required:

  NOAA NWS (api.weather.gov):
    Active weather alerts for the US.  GeoJSON FeatureCollection.
    Filters: severity, urgency, certainty, event type, area/state.
    ~100-500 active alerts at any time.

  NASA FIRMS (firms.modaps.eosdis.nasa.gov):
    Global active fire detections from MODIS satellite.  CSV, 24h rolling.
    ~15-30K fire points worldwide per day.  Lat/lon + brightness + confidence.
    We filter to high-confidence fires near critical infrastructure.

Signal theory:
  - Severe weather → supply chain disruption, utility liability, crop damage
  - Wildfire near refinery/pipeline/substation → energy supply risk
  - Hurricane/tornado warning in industrial zone → production halt
  - Freeze/heat warnings → energy demand spike
  - Drought conditions → water-intensive industry + agriculture impact

Modes:
  alerts   — Active NWS weather alerts (US). Filter by severity/state/event.
  fires    — NASA FIRMS active fire detections near infrastructure zones.
  summary  — Combined: severe alerts count + fire hotspots by region.
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

_NWS_BASE = "https://api.weather.gov"
_FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/c6.1/csv/MODIS_C6_1_Global_24h.csv"
_UA = "(TirraMind/0.1, tirramind@example.com)"
_TIMEOUT = 20

# Severity levels recognized by NWS
_SEVERITIES = {"Extreme", "Severe", "Moderate", "Minor", "Unknown"}

# Market-relevant event types (weather events that move markets)
_MARKET_EVENTS = {
    "Hurricane Warning",
    "Hurricane Watch",
    "Tropical Storm Warning",
    "Tornado Warning",
    "Tornado Watch",
    "Blizzard Warning",
    "Ice Storm Warning",
    "Winter Storm Warning",
    "Extreme Cold Warning",
    "Excessive Heat Warning",
    "Heat Advisory",
    "Red Flag Warning",
    "Fire Weather Watch",
    "Flash Flood Warning",
    "Flood Warning",
    "Tsunami Warning",
    "Storm Surge Warning",
    "Freeze Warning",
    "Hard Freeze Warning",
    "Frost Advisory",
    "Drought Information Statement",
}

# US states
_US_STATES = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
    "PR",
    "VI",
    "GU",
    "AS",
    "MP",
}

# Critical infrastructure zones for fire proximity detection
# (lat, lon, radius_deg, name, sector)
INFRA_ZONES: list[dict[str, Any]] = [
    {
        "name": "Permian Basin",
        "lat": 31.9,
        "lon": -102.1,
        "radius": 2.0,
        "sector": "oil",
    },
    {
        "name": "Gulf Coast Refineries",
        "lat": 29.8,
        "lon": -93.9,
        "radius": 1.5,
        "sector": "oil",
    },
    {
        "name": "California Refineries",
        "lat": 33.9,
        "lon": -118.2,
        "radius": 0.8,
        "sector": "oil",
    },
    {
        "name": "Powder River Basin Coal",
        "lat": 44.8,
        "lon": -106.3,
        "radius": 1.5,
        "sector": "coal",
    },
    {
        "name": "Appalachian Gas Fields",
        "lat": 39.5,
        "lon": -80.5,
        "radius": 1.5,
        "sector": "gas",
    },
    {
        "name": "Bakken Oil Field",
        "lat": 48.1,
        "lon": -103.5,
        "radius": 1.5,
        "sector": "oil",
    },
    {
        "name": "ERCOT Texas Grid Core",
        "lat": 32.0,
        "lon": -97.0,
        "radius": 2.0,
        "sector": "power",
    },
    {
        "name": "PJM East Grid",
        "lat": 40.0,
        "lon": -75.5,
        "radius": 1.0,
        "sector": "power",
    },
    {
        "name": "Corn Belt Central",
        "lat": 41.5,
        "lon": -89.5,
        "radius": 3.0,
        "sector": "agriculture",
    },
    {
        "name": "California Central Valley",
        "lat": 36.7,
        "lon": -119.8,
        "radius": 1.5,
        "sector": "agriculture",
    },
    {
        "name": "Pacific NW Timber",
        "lat": 44.5,
        "lon": -122.0,
        "radius": 2.0,
        "sector": "timber",
    },
    {
        "name": "Colorado River Basin",
        "lat": 36.0,
        "lon": -111.8,
        "radius": 2.0,
        "sector": "water",
    },
]


def _severity_rank(severity: str) -> int:
    """Numeric rank for severity (lower = more severe)."""
    return {"Extreme": 0, "Severe": 1, "Moderate": 2, "Minor": 3}.get(severity, 4)


def _format_alert(props: dict[str, Any]) -> dict[str, Any]:
    """Normalize an NWS alert feature's properties into a clean dict."""
    return {
        "event": props.get("event", ""),
        "severity": props.get("severity", "Unknown"),
        "urgency": props.get("urgency", ""),
        "certainty": props.get("certainty", ""),
        "headline": (props.get("headline") or "")[:200],
        "area": (props.get("areaDesc") or "")[:200],
        "onset": props.get("onset", ""),
        "expires": props.get("expires", ""),
        "sender": props.get("senderName", ""),
        "category": props.get("category", ""),
        "market_relevant": props.get("event", "") in _MARKET_EVENTS,
    }


def _point_in_zone(lat: float, lon: float, zone: dict[str, Any]) -> bool:
    """Simple bounding-box proximity check (not geodesic, good enough for alerting)."""
    r = zone["radius"]
    return abs(lat - zone["lat"]) <= r and abs(lon - zone["lon"]) <= r


def _parse_fires_csv(text: str) -> list[dict[str, Any]]:
    """Parse NASA FIRMS MODIS CSV into list of dicts."""
    reader = csv.DictReader(io.StringIO(text))
    fires = []
    for row in reader:
        try:
            lat = float(row.get("latitude", ""))
            lon = float(row.get("longitude", ""))
            brightness = float(row.get("brightness", "0"))
            confidence = int(row.get("confidence", "0"))
            frp = float(row.get("frp", "0"))
        except (ValueError, TypeError):
            continue
        fires.append(
            {
                "lat": lat,
                "lon": lon,
                "brightness": brightness,
                "confidence": confidence,
                "frp": frp,
                "acq_date": row.get("acq_date", ""),
                "acq_time": row.get("acq_time", ""),
                "daynight": row.get("daynight", ""),
                "satellite": row.get("satellite", ""),
            }
        )
    return fires


class WeatherAlertsTool(Tool):
    name = "weather_alerts"
    description = (
        "Monitor active weather alerts (NOAA NWS, US) and global wildfire "
        "detections (NASA FIRMS). Mode 'alerts' shows active severe weather. "
        "Mode 'fires' shows active fires near critical infrastructure (refineries, "
        "pipelines, grid, agriculture). Mode 'summary' gives combined overview. "
        "Free, no API key."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["alerts", "fires", "summary"],
                "default": "alerts",
                "description": (
                    "alerts = active NWS warnings/watches. "
                    "fires = NASA FIRMS fire detections near infrastructure. "
                    "summary = combined overview."
                ),
            },
            "severity": {
                "type": "string",
                "default": "Severe",
                "description": (
                    "Minimum severity for alerts: Extreme, Severe, Moderate, Minor. "
                    "Default: Severe (includes Extreme)."
                ),
            },
            "state": {
                "type": "string",
                "default": "",
                "description": "Filter alerts by US state code (e.g., TX, CA, FL). Empty = all.",
            },
            "market_only": {
                "type": "boolean",
                "default": False,
                "description": (
                    "If true, only return market-relevant events (hurricanes, "
                    "tornadoes, freezes, fires, floods, heat, drought)."
                ),
            },
            "min_confidence": {
                "type": "integer",
                "default": 70,
                "description": "Minimum fire detection confidence (0-100). Default 70.",
            },
            "limit": {
                "type": "integer",
                "default": 30,
                "description": "Max results. Default 30, max 200.",
            },
        },
        "required": [],
    }

    def __init__(self, cache: DataCache | None = None) -> None:
        self._cache = cache

    def execute(
        self,
        *,
        mode: str = "alerts",
        severity: str = "Severe",
        state: str = "",
        market_only: bool = False,
        min_confidence: int = 70,
        limit: int = 30,
        **_: Any,
    ) -> ToolResult:
        mode = mode.lower().strip()
        if mode not in ("alerts", "fires", "summary"):
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Use 'alerts', 'fires', or 'summary'.",
            )

        state = state.strip().upper()
        if state and state not in _US_STATES:
            return ToolResult(
                success=False,
                output=f"Unknown state code '{state}'. Use 2-letter code (e.g., TX, CA).",
            )

        limit = max(1, min(limit, 200))
        min_confidence = max(0, min(min_confidence, 100))

        if severity not in _SEVERITIES:
            severity = "Severe"

        if mode == "alerts":
            return self._execute_alerts(
                severity=severity,
                state=state,
                market_only=market_only,
                limit=limit,
            )
        if mode == "fires":
            return self._execute_fires(
                min_confidence=min_confidence,
                limit=limit,
            )
        # summary
        return self._execute_summary(
            severity=severity,
            state=state,
            market_only=market_only,
            min_confidence=min_confidence,
            limit=limit,
        )

    # ------------------------------------------------------------------
    # alerts mode
    # ------------------------------------------------------------------

    def _execute_alerts(
        self,
        *,
        severity: str,
        state: str,
        market_only: bool,
        limit: int,
    ) -> ToolResult:
        alerts, error = self._fetch_nws(severity=severity, state=state)
        if error:
            return ToolResult(success=False, output=error)

        formatted = [_format_alert(f["properties"]) for f in alerts]
        if market_only:
            formatted = [a for a in formatted if a["market_relevant"]]

        formatted.sort(key=lambda a: _severity_rank(a["severity"]))
        formatted = formatted[:limit]

        if not formatted:
            return ToolResult(
                success=True,
                output=f"NWS: No active alerts (severity >= {severity}"
                + (f", state={state}" if state else "")
                + ").",
                data={"alerts": [], "count": 0},
            )

        lines = [
            f"NWS Active Weather Alerts ({len(formatted)} shown"
            + (f", state={state}" if state else "")
            + f", severity >= {severity}):",
            "",
        ]
        for a in formatted:
            mkt = " [MARKET]" if a["market_relevant"] else ""
            lines.append(f"  [{a['severity']:8s}] {a['event']}{mkt}")
            lines.append(f"    Area: {a['area'][:100]}")
            if a["headline"]:
                lines.append(f"    {a['headline'][:140]}")
            lines.append("")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"alerts": formatted, "count": len(formatted)},
        )

    # ------------------------------------------------------------------
    # fires mode
    # ------------------------------------------------------------------

    def _execute_fires(
        self,
        *,
        min_confidence: int,
        limit: int,
    ) -> ToolResult:
        fires, error = self._fetch_firms()
        if error:
            return ToolResult(success=False, output=error)

        # Filter to high-confidence fires near infrastructure
        near_infra: list[dict[str, Any]] = []
        for fire in fires:
            if fire["confidence"] < min_confidence:
                continue
            for zone in INFRA_ZONES:
                if _point_in_zone(fire["lat"], fire["lon"], zone):
                    entry = {**fire, "zone": zone["name"], "sector": zone["sector"]}
                    near_infra.append(entry)
                    break  # one zone match is enough

        near_infra.sort(key=lambda f: -f["brightness"])
        near_infra = near_infra[:limit]

        if not near_infra:
            return ToolResult(
                success=True,
                output=f"NASA FIRMS: No high-confidence fires near critical infrastructure "
                f"(confidence >= {min_confidence}, {len(fires)} total fire points globally).",
                data={"fires": [], "count": 0, "total_global": len(fires)},
            )

        # Group by zone
        by_zone: dict[str, list[dict[str, Any]]] = {}
        for f in near_infra:
            by_zone.setdefault(f["zone"], []).append(f)

        lines = [
            f"NASA FIRMS: {len(near_infra)} fires near infrastructure "
            f"({len(fires)} global, confidence >= {min_confidence}):",
            "",
        ]
        for zone_name, zone_fires in sorted(by_zone.items(), key=lambda x: -len(x[1])):
            sector = zone_fires[0]["sector"]
            max_bright = max(f["brightness"] for f in zone_fires)
            lines.append(
                f"  {zone_name} ({sector}) — {len(zone_fires)} detections, "
                f"max brightness {max_bright:.0f}K"
            )
            for f in zone_fires[:3]:
                lines.append(
                    f"    ({f['lat']:.2f}, {f['lon']:.2f}) "
                    f"bright={f['brightness']:.0f} conf={f['confidence']} "
                    f"frp={f['frp']:.1f} {f['acq_date']} {f['acq_time']}"
                )
            if len(zone_fires) > 3:
                lines.append(f"    ... and {len(zone_fires) - 3} more")
            lines.append("")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "fires": near_infra,
                "count": len(near_infra),
                "total_global": len(fires),
                "zones_affected": list(by_zone.keys()),
            },
        )

    # ------------------------------------------------------------------
    # summary mode
    # ------------------------------------------------------------------

    def _execute_summary(
        self,
        *,
        severity: str,
        state: str,
        market_only: bool,
        min_confidence: int,
        limit: int,
    ) -> ToolResult:
        alerts, alerts_err = self._fetch_nws(severity=severity, state=state)
        fires, fires_err = self._fetch_firms()

        lines = ["Weather & Fire Summary:", ""]

        # Alerts
        if alerts_err:
            lines.append(f"  NWS Alerts: ERROR — {alerts_err}")
        else:
            formatted = [_format_alert(f["properties"]) for f in alerts]
            if market_only:
                formatted = [a for a in formatted if a["market_relevant"]]
            formatted.sort(key=lambda a: _severity_rank(a["severity"]))

            # Count by severity
            by_sev: dict[str, int] = {}
            for a in formatted:
                by_sev[a["severity"]] = by_sev.get(a["severity"], 0) + 1

            sev_str = ", ".join(
                f"{s}: {c}"
                for s, c in sorted(by_sev.items(), key=lambda x: _severity_rank(x[0]))
            )
            lines.append(f"  NWS Alerts: {len(formatted)} active ({sev_str})")

            # Count by event type
            by_event: dict[str, int] = {}
            for a in formatted:
                by_event[a["event"]] = by_event.get(a["event"], 0) + 1
            for event, count in sorted(by_event.items(), key=lambda x: -x[1])[:8]:
                mkt = " *" if event in _MARKET_EVENTS else ""
                lines.append(f"    {event}: {count}{mkt}")

        lines.append("")

        # Fires
        if fires_err:
            lines.append(f"  NASA FIRMS: ERROR — {fires_err}")
        else:
            near_infra = []
            for fire in fires:
                if fire["confidence"] < min_confidence:
                    continue
                for zone in INFRA_ZONES:
                    if _point_in_zone(fire["lat"], fire["lon"], zone):
                        near_infra.append(
                            {**fire, "zone": zone["name"], "sector": zone["sector"]}
                        )
                        break

            lines.append(
                f"  NASA FIRMS: {len(fires)} global fire points, "
                f"{len(near_infra)} near critical infrastructure"
            )
            if near_infra:
                by_zone: dict[str, int] = {}
                for f in near_infra:
                    by_zone[f["zone"]] = by_zone.get(f["zone"], 0) + 1
                for zone_name, count in sorted(by_zone.items(), key=lambda x: -x[1]):
                    lines.append(f"    {zone_name}: {count} fires")

        data = {
            "alert_count": len(alerts) if not alerts_err else 0,
            "fire_count_global": len(fires) if not fires_err else 0,
            "fire_count_infra": len(near_infra) if not fires_err else 0,
        }

        return ToolResult(success=True, output="\n".join(lines), data=data)

    # ------------------------------------------------------------------
    # NWS fetch
    # ------------------------------------------------------------------

    def _fetch_nws(
        self,
        *,
        severity: str,
        state: str,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Fetch active alerts from NWS. Returns (features, error)."""
        cache_key = {"source": "nws", "severity": severity, "state": state}
        if self._cache:
            cached = self._cache.get("weather_alerts", cache_key)
            if cached is not None:
                return cached.get("features", []), None

        params: dict[str, str] = {"status": "actual"}
        # Build severity filter: include everything >= requested level
        sev_rank = _severity_rank(severity)
        included = [
            s for s in _SEVERITIES if _severity_rank(s) <= sev_rank and s != "Unknown"
        ]
        if included:
            params["severity"] = ",".join(included)
        if state:
            params["area"] = state

        url = f"{_NWS_BASE}/alerts/active"
        qs = "&".join(f"{k}={v}" for k, v in params.items())

        try:
            with httpx.Client(
                timeout=_TIMEOUT,
                headers={"User-Agent": _UA, "Accept": "application/geo+json"},
            ) as client:
                resp = client.get(f"{url}?{qs}")
                if resp.status_code == 429:
                    return [], "NWS API: Rate limited. Try again shortly."
                if resp.status_code >= 400:
                    return [], f"NWS API error: HTTP {resp.status_code}"
                data = resp.json()
        except httpx.TimeoutException:
            return [], "NWS API: Request timed out."
        except Exception as exc:
            log.exception("NWS fetch failed")
            return [], f"NWS fetch error: {exc}"

        features = data.get("features", [])

        if self._cache and features:
            self._cache.put("weather_alerts", cache_key, {"features": features})

        return features, None

    # ------------------------------------------------------------------
    # FIRMS fetch
    # ------------------------------------------------------------------

    def _fetch_firms(self) -> tuple[list[dict[str, Any]], str | None]:
        """Fetch 24h MODIS fire data from NASA FIRMS. Returns (fires, error)."""
        cache_key = {"source": "firms", "period": "24h"}
        if self._cache:
            cached = self._cache.get("weather_alerts_firms", cache_key)
            if cached is not None:
                return cached, None

        try:
            with httpx.Client(
                timeout=30,
                headers={"User-Agent": _UA},
            ) as client:
                resp = client.get(_FIRMS_URL)
                if resp.status_code >= 400:
                    return [], f"NASA FIRMS error: HTTP {resp.status_code}"
                fires = _parse_fires_csv(resp.text)
        except httpx.TimeoutException:
            return [], "NASA FIRMS: Request timed out (large dataset)."
        except Exception as exc:
            log.exception("FIRMS fetch failed")
            return [], f"FIRMS fetch error: {exc}"

        if self._cache and fires:
            self._cache.put("weather_alerts_firms", cache_key, fires)

        return fires, None
