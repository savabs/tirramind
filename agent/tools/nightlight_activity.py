"""
Tool: Nightlight & NDVI Economic Zone Activity Signals (Idea 14)

Two satellite observation channels unavailable from financial data feeds:

  nightlight   VIIRS thermal emission index for 50 economic zones.
               Uses NASA FIRMS fire radiative power (FRP) aggregated
               within zone bounding boxes as an industrial activity
               proxy — the same physical signal underlying nightlight
               radiance (thermal emissions from industrial operations,
               flaring, transportation).  Week-over-week delta is the
               alpha signal.

  ndvi         MODIS MOD13Q1 250m 16-day NDVI for agricultural zones.
               Crop health proxy for soft commodity prediction (wheat,
               corn, soy, palm oil, sugar).

Signal theory
-------------
- Industrial zones: sustained FRP near refineries/smelters/factories →
  operational throughput signal.  Lead time: 1-2 weeks vs equity data.
- Port zones: FRP near container terminals (vessel engine emissions,
  yard equipment) → shipping volume proxy.
- Agricultural zones: NDVI decline in primary crop region → crop stress
  → commodity price pressure.  Lead time: 2-4 weeks vs spot prices.
- Nightlight delta YoY is a GDP nowcast for emerging markets (no lag).

Data sources (both free, no auth required for FIRMS public API key)
--------------------------------------------------------------------
- NASA FIRMS (VIIRS SNPP/NOAA-20 NRT, 375m): already proven in
  agent/tools/satellite_activity.py.  FRP unit: MW (megawatts).
  API: https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/...
- MODIS ORNL NDVI (MOD13Q1, 250m, 16-day): already proven.
  API: https://modis.ornl.gov/rst/api/v1/MOD13Q1/subset

References
----------
Henderson, J.V. et al. (2012). "Measuring Economic Growth from Outer
  Space." American Economic Review, 102(2): 994–1028.
  Established nightlight radiance → GDP per capita relationship.

Elvidge, C.D. et al. (2017). "VIIRS Night-Time Lights." ISPRS JPRSE.
  VIIRS DNB 500m monthly composites outperform DMSP for economic
  monitoring.

Chen, X. & Nordhaus, W. (2011). "Using luminosity data as a proxy for
  economic statistics." PNAS.

NASA FIRMS documentation: https://firms.modaps.eosdis.nasa.gov/
MODIS ORNL documentation: https://modis.ornl.gov/documentation.html
"""

from __future__ import annotations

import csv
import io
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

_UA = "TirraMind/0.1 (nightlight-activity-tool)"
_TIMEOUT = 30
_FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api"
_MODIS_BASE = "https://modis.ornl.gov/rst/api/v1"

VALID_MODES = {"nightlight", "ndvi", "both"}

# ═══════════════════════════════════════════════════════════════
# Economic Zone Manifest — 50 globally significant zones
# ═══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class EconomicZone:
    """A named geographic zone to monitor via satellite.

    Attributes
    ----------
    zone_id   : Unique snake_case identifier used in signal names.
    name      : Human-readable name.
    lat       : Centre latitude (WGS84).
    lon       : Centre longitude (WGS84).
    bbox_deg  : Half-width of the bounding box in degrees (square).
    category  : "industrial" | "agricultural" | "port" | "urban"
    commodities: Related commodity identifiers (for signal routing).
    """

    zone_id: str
    name: str
    lat: float
    lon: float
    bbox_deg: float
    category: str
    commodities: tuple[str, ...] = ()


# 50 zones spanning key industrial, agricultural, port, and urban regions
ECONOMIC_ZONES: list[EconomicZone] = [
    # ── Oil & Gas / Industrial ─────────────────────────────────────────────
    EconomicZone(
        "permian_basin",
        "Permian Basin, TX",
        31.8,
        -102.0,
        1.5,
        "industrial",
        ("crude_oil", "nat_gas"),
    ),
    EconomicZone(
        "cushing_ok", "Cushing, OK Hub", 36.0, -96.8, 0.5, "industrial", ("crude_oil",)
    ),
    EconomicZone(
        "rotterdam_port",
        "Rotterdam Refinery Cluster",
        51.9,
        4.4,
        0.4,
        "industrial",
        ("crude_oil", "gasoline"),
    ),
    EconomicZone(
        "jubail_ksa",
        "Jubail Industrial City, KSA",
        27.0,
        49.7,
        0.5,
        "industrial",
        ("crude_oil", "petrochemicals"),
    ),
    EconomicZone(
        "shandong_cn",
        "Shandong Refinery Belt, China",
        37.4,
        118.0,
        1.0,
        "industrial",
        ("crude_oil",),
    ),
    EconomicZone(
        "alberta_oilsands",
        "Alberta Oil Sands, Canada",
        57.0,
        -111.5,
        1.5,
        "industrial",
        ("crude_oil",),
    ),
    EconomicZone(
        "vaca_muerta_ar",
        "Vaca Muerta, Argentina",
        -38.5,
        -69.5,
        1.0,
        "industrial",
        ("crude_oil", "nat_gas"),
    ),
    EconomicZone(
        "eagle_ford_tx",
        "Eagle Ford Shale, TX",
        28.5,
        -99.0,
        1.5,
        "industrial",
        ("crude_oil", "nat_gas"),
    ),
    EconomicZone(
        "siberia_yamal",
        "Yamal LNG, Siberia",
        71.0,
        72.0,
        1.5,
        "industrial",
        ("nat_gas", "lng"),
    ),
    EconomicZone(
        "gulf_coast_us",
        "US Gulf Coast Refinery Row",
        29.8,
        -95.0,
        0.8,
        "industrial",
        ("crude_oil", "gasoline"),
    ),
    # ── Metals / Mining ────────────────────────────────────────────────────
    EconomicZone(
        "atacama_chile",
        "Atacama Copper Belt, Chile",
        -22.5,
        -68.5,
        1.5,
        "industrial",
        ("copper",),
    ),
    EconomicZone(
        "pilbara_au",
        "Pilbara Iron Ore, Australia",
        -22.5,
        117.5,
        1.5,
        "industrial",
        ("iron_ore",),
    ),
    EconomicZone(
        "carajas_br",
        "Carajás Iron Mine, Brazil",
        -6.0,
        -50.2,
        0.8,
        "industrial",
        ("iron_ore",),
    ),
    EconomicZone(
        "norilsk_ru",
        "Norilsk Nickel, Russia",
        69.3,
        88.2,
        0.8,
        "industrial",
        ("nickel", "palladium"),
    ),
    EconomicZone(
        "witwatersrand_za",
        "Witwatersrand Gold, SA",
        -26.2,
        27.5,
        1.0,
        "industrial",
        ("gold",),
    ),
    EconomicZone(
        "katanga_cd",
        "Katanga Copper-Cobalt, DRC",
        -9.5,
        25.5,
        1.5,
        "industrial",
        ("copper", "cobalt"),
    ),
    EconomicZone(
        "ural_steel_ru",
        "Ural Steel Belt, Russia",
        56.8,
        60.6,
        1.0,
        "industrial",
        ("steel",),
    ),
    # ── Port / Shipping ────────────────────────────────────────────────────
    EconomicZone(
        "shanghai_port",
        "Port of Shanghai",
        31.4,
        121.8,
        0.5,
        "port",
        ("container_shipping",),
    ),
    EconomicZone(
        "singapore_port",
        "Port of Singapore",
        1.26,
        103.8,
        0.3,
        "port",
        ("container_shipping", "lng"),
    ),
    EconomicZone(
        "shenzhen_port",
        "Shenzhen/Yantian Port",
        22.5,
        114.3,
        0.4,
        "port",
        ("container_shipping",),
    ),
    EconomicZone(
        "busan_port",
        "Port of Busan, South Korea",
        35.1,
        129.0,
        0.3,
        "port",
        ("container_shipping",),
    ),
    EconomicZone(
        "antwerp_port",
        "Port of Antwerp",
        51.3,
        4.3,
        0.3,
        "port",
        ("container_shipping", "chemicals"),
    ),
    EconomicZone(
        "los_angeles_port",
        "Port of LA/Long Beach",
        33.7,
        -118.2,
        0.4,
        "port",
        ("container_shipping",),
    ),
    EconomicZone(
        "houston_port",
        "Port of Houston",
        29.6,
        -95.0,
        0.4,
        "port",
        ("crude_oil", "lng", "chemicals"),
    ),
    EconomicZone(
        "suez_canal",
        "Suez Canal Transit Zone",
        30.7,
        32.3,
        0.5,
        "port",
        ("container_shipping", "crude_oil"),
    ),
    # ── Agricultural — Grains ──────────────────────────────────────────────
    EconomicZone(
        "iowa_cornbelt",
        "Iowa Corn Belt, US",
        42.0,
        -93.5,
        2.0,
        "agricultural",
        ("corn", "soybean"),
    ),
    EconomicZone(
        "kansas_wheat",
        "Kansas Winter Wheat",
        38.5,
        -99.0,
        2.0,
        "agricultural",
        ("wheat",),
    ),
    EconomicZone(
        "mato_grosso_br",
        "Mato Grosso Soy, Brazil",
        -13.0,
        -56.0,
        2.5,
        "agricultural",
        ("soybean",),
    ),
    EconomicZone(
        "ukraine_wheat",
        "Ukraine Wheat Belt",
        49.5,
        32.0,
        3.0,
        "agricultural",
        ("wheat", "corn"),
    ),
    EconomicZone(
        "punjab_wheat_in",
        "Punjab Wheat, India",
        30.5,
        75.5,
        2.0,
        "agricultural",
        ("wheat",),
    ),
    EconomicZone(
        "pampas_ar",
        "Argentine Pampas (Soy/Corn)",
        -34.0,
        -62.0,
        3.0,
        "agricultural",
        ("soybean", "corn"),
    ),
    EconomicZone(
        "paris_basin_fr",
        "Paris Basin Wheat, France",
        48.5,
        2.5,
        2.0,
        "agricultural",
        ("wheat",),
    ),
    EconomicZone(
        "huang_he_cn",
        "Yellow River Plain, China",
        35.0,
        115.0,
        2.5,
        "agricultural",
        ("wheat", "corn"),
    ),
    # ── Agricultural — Soft Commodities ───────────────────────────────────
    EconomicZone(
        "minas_gerais_br",
        "Minas Gerais Coffee, Brazil",
        -19.5,
        -44.0,
        2.0,
        "agricultural",
        ("coffee",),
    ),
    EconomicZone(
        "ivory_coast_cacao",
        "Ivory Coast Cacao Belt",
        6.8,
        -5.6,
        2.5,
        "agricultural",
        ("cacao",),
    ),
    EconomicZone(
        "borneo_palm_oil",
        "Borneo Palm Oil Belt",
        1.5,
        115.0,
        3.0,
        "agricultural",
        ("palm_oil",),
    ),
    EconomicZone(
        "thailand_sugar",
        "Thai Sugar Cane East",
        14.5,
        101.5,
        2.0,
        "agricultural",
        ("sugar",),
    ),
    EconomicZone(
        "queensland_sugar",
        "Queensland Sugar, Australia",
        -20.5,
        147.0,
        2.0,
        "agricultural",
        ("sugar",),
    ),
    # ── Urban Economic Hubs ────────────────────────────────────────────────
    EconomicZone(
        "pearl_river_delta",
        "Pearl River Delta Mfg., China",
        22.8,
        113.5,
        1.5,
        "urban",
        ("electronics",),
    ),
    EconomicZone(
        "yangtze_delta_cn",
        "Yangtze River Delta, China",
        31.8,
        121.5,
        2.0,
        "urban",
        ("manufacturing",),
    ),
    EconomicZone(
        "ruhr_valley_de",
        "Ruhr Valley Industry, Germany",
        51.5,
        7.0,
        0.8,
        "urban",
        ("steel", "chemicals"),
    ),
    EconomicZone(
        "tokyo_bay_jp",
        "Tokyo Bay Industrial Zone",
        35.5,
        140.0,
        0.8,
        "urban",
        ("electronics", "manufacturing"),
    ),
    EconomicZone(
        "chennai_in",
        "Chennai Auto Cluster, India",
        13.0,
        80.3,
        0.5,
        "urban",
        ("autos",),
    ),
    EconomicZone(
        "istanbul_tr",
        "Istanbul Industrial Belt",
        41.0,
        28.9,
        0.8,
        "urban",
        ("manufacturing",),
    ),
    # ── Energy Transition ─────────────────────────────────────────────────
    EconomicZone(
        "inner_mongolia_cn",
        "Inner Mongolia Solar/Wind Belt",
        41.0,
        112.0,
        3.0,
        "industrial",
        ("solar", "wind"),
    ),
    EconomicZone(
        "texas_wind",
        "West Texas Wind Corridor",
        32.5,
        -101.5,
        2.0,
        "industrial",
        ("wind",),
    ),
    EconomicZone(
        "north_sea_wind",
        "North Sea Offshore Wind",
        55.0,
        5.0,
        2.0,
        "industrial",
        ("wind",),
    ),
    EconomicZone(
        "xinjiang_coal_cn",
        "Xinjiang Coal Basin, China",
        42.0,
        86.5,
        3.0,
        "industrial",
        ("coal",),
    ),
    EconomicZone(
        "jharkhand_coal_in",
        "Jharkhand Coal Belt, India",
        23.5,
        85.5,
        2.0,
        "industrial",
        ("coal",),
    ),
    EconomicZone(
        "great_plains_corn",
        "Great Plains Corn Belt, US",
        41.5,
        -96.0,
        2.5,
        "agricultural",
        ("corn", "soybean"),
    ),
]

# Fast lookup by zone_id
ZONE_BY_ID: dict[str, EconomicZone] = {z.zone_id: z for z in ECONOMIC_ZONES}


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════


def _zone_area_str(zone: EconomicZone) -> str:
    """FIRMS area string: 'lon_min,lat_min,lon_max,lat_max'."""
    d = zone.bbox_deg
    return f"{zone.lon - d},{zone.lat - d},{zone.lon + d},{zone.lat + d}"


def _date_to_modis(date_str: str) -> str | None:
    """Convert YYYY-MM-DD to MODIS date format A{YYYYDDD}."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        doy = dt.timetuple().tm_yday
        return f"A{dt.year}{doy:03d}"
    except ValueError:
        log.warning("Invalid date: %s", date_str)
        return None


def _aggregate_frp(hotspots: list[dict]) -> dict[str, float]:
    """Aggregate VIIRS fire radiative power within a zone.

    Returns dict with keys:
        frp_total_mw      — sum of fire radiative power (MW)
        hotspot_count     — number of thermal anomalies detected
        mean_brightness_k — mean brightness temperature (K)
        industrial_ratio  — fraction of detections classified industrial
                            (confidence='high' + frp > 50 MW)
    """
    if not hotspots:
        return {
            "frp_total_mw": 0.0,
            "hotspot_count": 0,
            "mean_brightness_k": 0.0,
            "industrial_ratio": 0.0,
        }

    frp_vals = []
    bright_vals = []
    industrial = 0

    for row in hotspots:
        try:
            frp = float(row.get("frp", 0) or 0)
            bright = float(row.get("bright_ti4", row.get("brightness", 0)) or 0)
            conf = str(row.get("confidence", "")).lower()
            frp_vals.append(frp)
            if bright > 0:
                bright_vals.append(bright)
            if conf in ("high", "h") and frp > 50.0:
                industrial += 1
        except (ValueError, TypeError):
            continue

    n = len(frp_vals) or 1
    return {
        "frp_total_mw": sum(frp_vals),
        "hotspot_count": len(frp_vals),
        "mean_brightness_k": sum(bright_vals) / max(len(bright_vals), 1),
        "industrial_ratio": industrial / n,
    }


def _extract_ndvi(modis_json: dict) -> float | None:
    """Extract the latest valid NDVI scalar from MODIS ORNL response."""
    try:
        subset = modis_json.get("subset", [])
        ndvi_vals = []
        for entry in subset:
            if "250m_16_days_NDVI" in str(entry.get("band", "")):
                raw = entry.get("data", [])
                for v in raw:
                    try:
                        fv = float(v)
                        if -2000 <= fv <= 10000:  # MODIS scale factor range
                            ndvi_vals.append(fv / 10000.0)
                    except (ValueError, TypeError):
                        continue
        if ndvi_vals:
            return sum(ndvi_vals) / len(ndvi_vals)
        return None
    except Exception:
        return None


def _fetch_zone_frp(
    zone: EconomicZone,
    firms_key: str,
    days: int = 7,
    source: str = "VIIRS_SNPP_NRT",
) -> dict[str, float]:
    """Fetch VIIRS FRP data for a zone from NASA FIRMS."""
    area = _zone_area_str(zone)
    url = f"{_FIRMS_BASE}/area/csv/{firms_key}/{source}/{area}/{days}"
    try:
        resp = httpx.get(
            url,
            headers={"User-Agent": _UA},
            timeout=_TIMEOUT,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            log.warning("FIRMS HTTP %d for zone %s", resp.status_code, zone.zone_id)
            return {}
        text = resp.text.strip()
        if not text:
            return _aggregate_frp([])
        reader = csv.DictReader(io.StringIO(text))
        return _aggregate_frp(list(reader))
    except httpx.HTTPError as exc:
        log.warning("FIRMS fetch error for %s: %s", zone.zone_id, exc)
        return {}


def _fetch_zone_ndvi(
    zone: EconomicZone,
    days_back: int = 32,
) -> float | None:
    """Fetch recent MODIS NDVI for an agricultural zone centre."""
    end_dt = datetime.now(tz=timezone.utc)
    start_dt = end_dt - timedelta(days=days_back)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")
    start_modis = _date_to_modis(start_str)
    end_modis = _date_to_modis(end_str)
    if start_modis is None or end_modis is None:
        return None
    url = f"{_MODIS_BASE}/MOD13Q1/subset"
    params = {
        "latitude": zone.lat,
        "longitude": zone.lon,
        "band": "250m_16_days_NDVI",
        "startDate": start_modis,
        "endDate": end_modis,
        "kmAboveBelow": 5,
        "kmLeftRight": 5,
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
            log.warning("MODIS HTTP %d for zone %s", resp.status_code, zone.zone_id)
            return None
        return _extract_ndvi(resp.json())
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("MODIS fetch error for %s: %s", zone.zone_id, exc)
        return None


# ═══════════════════════════════════════════════════════════════
# NightlightActivityTool
# ═══════════════════════════════════════════════════════════════


class NightlightActivityTool(Tool):
    """Satellite-derived economic activity index for 50 global zones.

    Two modes of operation:

    nightlight
        VIIRS thermal emission index (FRP) for industrial and port zones.
        Uses NASA FIRMS VIIRS SNPP/NOAA-20 NRT API (same as
        SatelliteActivityTool).  FRP aggregated per zone → activity index.
        Stores signals:
            nightlight.{zone_id}.frp_total_mw
            nightlight.{zone_id}.hotspot_count
            nightlight.{zone_id}.industrial_ratio

    ndvi
        MODIS MOD13Q1 NDVI for agricultural zones.
        Stores signals:
            ndvi.{zone_id}.value   (normalised 0–1)
            ndvi.{zone_id}.health  (0=poor, 0.5=average, 1=excellent)

    both
        Run both modes in sequence.

    Parameters
    ----------
    firms_api_key : str  NASA FIRMS MAP API key (env: FIRMS_API_KEY).
    mode          : "nightlight" | "ndvi" | "both"
    zone_ids      : optional list of zone_id strings to restrict scan.
    days          : lookback days for FIRMS data (default 7).
    store         : PipelineStore to persist signals (optional).
    """

    @property
    def name(self) -> str:
        return "nightlight_activity"

    @property
    def description(self) -> str:
        return (
            "Compute satellite-derived economic activity index for 50 global "
            "economic zones using VIIRS thermal emissions (nightlight proxy) "
            "and MODIS NDVI crop health.  Returns activity index per zone."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": list(VALID_MODES),
                    "description": "Which signals to compute: nightlight, ndvi, or both.",
                    "default": "both",
                },
                "zone_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of zone IDs to restrict scan. "
                    "Omit to scan all zones.",
                },
                "days": {
                    "type": "integer",
                    "description": "FIRMS lookback window in days (1-10).",
                    "default": 7,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["mode"],
        }

    def __init__(
        self,
        firms_api_key: str | None = None,
        store: Any = None,
        cache: DataCache | None = None,
    ) -> None:
        self._firms_key = firms_api_key or os.getenv("FIRMS_API_KEY", "")
        self._store = store
        self._cache = cache

    def execute(self, **kwargs: Any) -> ToolResult:
        from agent.preflight import FeaturePreflight  # noqa: PLC0415

        mode = kwargs.get("mode", "both")
        if mode not in VALID_MODES:
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Choose from: {', '.join(VALID_MODES)}",
            )

        ok, pf = FeaturePreflight.for_nightlight(
            firms_api_key=self._firms_key,
            store=self._store,
            mode=mode,
        )
        if not ok and pf.reason.value == "MISSING_CONFIG":
            # Missing API key is a hard stop for nightlight mode
            if mode == "nightlight":
                return ToolResult(success=False, output=pf.user_message)
            # For ndvi/both modes, log warning but continue to NDVI
            log.warning("Preflight warning: %s", pf.user_message)

        requested_ids: list[str] | None = kwargs.get("zone_ids")
        days = int(kwargs.get("days", 7))

        zones = ECONOMIC_ZONES
        if requested_ids:
            zones = [z for z in ECONOMIC_ZONES if z.zone_id in requested_ids]
            if not zones:
                return ToolResult(
                    success=False,
                    output=f"No zones matched: {requested_ids}",
                )

        results: dict[str, dict] = {}
        errors: list[str] = []
        now = time.time()

        # ── Nightlight (VIIRS FRP) pass ────────────────────────────────────
        if mode in ("nightlight", "both"):
            if not self._firms_key:
                errors.append("FIRMS_API_KEY not set — nightlight mode skipped.")
            else:
                industrial_zones = [
                    z for z in zones if z.category in ("industrial", "port")
                ]
                for zone in industrial_zones:
                    frp_data = _fetch_zone_frp(zone, self._firms_key, days=days)
                    if frp_data:
                        results.setdefault(zone.zone_id, {}).update(
                            {
                                "frp_total_mw": frp_data.get("frp_total_mw", 0.0),
                                "hotspot_count": frp_data.get("hotspot_count", 0),
                                "industrial_ratio": frp_data.get(
                                    "industrial_ratio", 0.0
                                ),
                                "zone_name": zone.name,
                                "category": zone.category,
                                "commodities": list(zone.commodities),
                            }
                        )
                        if self._store:
                            self._persist_nightlight(zone, frp_data, now)

        # ── NDVI (MODIS) pass ──────────────────────────────────────────────
        if mode in ("ndvi", "both"):
            ag_zones = [z for z in zones if z.category == "agricultural"]
            for zone in ag_zones:
                ndvi = _fetch_zone_ndvi(zone)
                if ndvi is not None:
                    health = self._ndvi_to_health(ndvi)
                    results.setdefault(zone.zone_id, {}).update(
                        {
                            "ndvi": ndvi,
                            "ndvi_health": health,
                            "zone_name": zone.name,
                            "category": zone.category,
                            "commodities": list(zone.commodities),
                        }
                    )
                    if self._store:
                        self._persist_ndvi(zone, ndvi, health, now)

        if not results and errors:
            return ToolResult(
                success=False,
                output="; ".join(errors),
                data={},
            )

        n_zones = len(results)
        summary_parts = []
        if errors:
            summary_parts.append(f"Warnings: {'; '.join(errors)}")
        summary_parts.append(
            f"Retrieved activity data for {n_zones} zones "
            f"(mode={mode}, days={days})."
        )
        if results:
            top = sorted(
                [(zid, d.get("frp_total_mw", 0.0)) for zid, d in results.items()],
                key=lambda x: -x[1],
            )[:3]
            if any(v > 0 for _, v in top):
                summary_parts.append(
                    "Top FRP zones: " + ", ".join(f"{z}={v:.1f}MW" for z, v in top)
                )

        return ToolResult(
            success=True,
            output="\n".join(summary_parts),
            data=results,
        )

    # ── Persistence helpers ────────────────────────────────────────────────

    def _persist_nightlight(
        self, zone: EconomicZone, frp_data: dict, ts: float
    ) -> None:
        if self._store is None:
            return
        for sig_suffix, val in [
            ("frp_total_mw", frp_data.get("frp_total_mw", 0.0)),
            ("hotspot_count", float(frp_data.get("hotspot_count", 0))),
            ("industrial_ratio", frp_data.get("industrial_ratio", 0.0)),
        ]:
            try:
                self._store.store_signal(
                    signal_name=f"nightlight.{zone.zone_id}.{sig_suffix}",
                    value=val,
                    observed_at=ts,
                    source_tool="nightlight_activity",
                )
            except Exception:
                log.warning(
                    "Failed to store nightlight signal for %s.%s",
                    zone.zone_id,
                    sig_suffix,
                    exc_info=True,
                )

    def _persist_ndvi(
        self, zone: EconomicZone, ndvi: float, health: float, ts: float
    ) -> None:
        if self._store is None:
            return
        for sig_suffix, val in [("value", ndvi), ("health", health)]:
            try:
                self._store.store_signal(
                    signal_name=f"ndvi.{zone.zone_id}.{sig_suffix}",
                    value=val,
                    observed_at=ts,
                    source_tool="nightlight_activity",
                )
            except Exception:
                log.warning(
                    "Failed to store NDVI signal for %s.%s",
                    zone.zone_id,
                    sig_suffix,
                    exc_info=True,
                )

    # ── Static helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _ndvi_to_health(ndvi: float) -> float:
        """Map NDVI [-1, 1] to a crop health index [0, 1].

        Thresholds based on common agricultural NDVI interpretation:
          < 0.2  → stressed / bare soil  → 0.0 - 0.2
          0.2–0.4 → sparse/moderate      → 0.2 - 0.5
          0.4–0.6 → moderate crop        → 0.5 - 0.75
          > 0.6  → dense healthy crop    → 0.75 - 1.0
        """
        ndvi = max(-1.0, min(1.0, ndvi))
        if ndvi < 0.2:
            return max(0.0, ndvi / 0.2 * 0.2)
        elif ndvi < 0.4:
            return 0.2 + (ndvi - 0.2) / 0.2 * 0.3
        elif ndvi < 0.6:
            return 0.5 + (ndvi - 0.4) / 0.2 * 0.25
        else:
            return min(1.0, 0.75 + (ndvi - 0.6) / 0.4 * 0.25)
