"""Per-tool evidence extraction functions.

Each tool's raw ToolResult.data gets a thin extractor that produces
a list of Evidence objects. Extractors are defensive: unexpected schemas
produce an empty list and a logged warning, never an exception.

Extractor registration is explicit — one function per tool name.
Unknown tools yield an empty list.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from agent.convergence.evidence import Evidence

log = logging.getLogger(__name__)

# ── Type alias + registry ──────────────────────────────────────

ExtractorFn = Callable[[str, Any], list[Evidence]]

_REGISTRY: dict[str, ExtractorFn] = {}


def register_extractor(tool_name: str, fn: ExtractorFn) -> None:
    """Register an extractor function for a tool name."""
    if tool_name in _REGISTRY:
        raise ValueError(f"Extractor already registered for {tool_name!r}")
    _REGISTRY[tool_name] = fn


def extract_evidence(tool_name: str, tool_data: Any) -> list[Evidence]:
    """Extract evidence from a tool's data dict.

    Returns an empty list if the tool has no extractor or if
    extraction fails. Never raises.
    """
    if tool_data is None:
        return []

    fn = _REGISTRY.get(tool_name)
    if fn is None:
        log.debug("No extractor registered for tool %r", tool_name)
        return []

    try:
        result = fn(tool_name, tool_data)
        if not isinstance(result, list):
            log.warning(
                "Extractor for %r returned %s, not list",
                tool_name,
                type(result).__name__,
            )
            return []
        return result
    except Exception:
        log.warning("Extractor for %r failed", tool_name, exc_info=True)
        return []


def registered_tools() -> list[str]:
    """Return sorted list of tool names with registered extractors."""
    return sorted(_REGISTRY.keys())


# ── Helpers ────────────────────────────────────────────────────


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Coerce to float, returning default on failure."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    """Coerce to int, returning default on failure."""
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _now_ts() -> float:
    """Current unix timestamp."""
    return time.time()


# ══════════════════════════════════════════════════════════════
#  EXTRACTOR IMPLEMENTATIONS (10 tools)
# ══════════════════════════════════════════════════════════════


# ── 1. CFTC ───────────────────────────────────────────────────


def _extract_cftc(tool_name: str, data: Any) -> list[Evidence]:
    """Extract managed-money net positioning per contract from CFTC data.

    Key signal: _mm_net_pct_oi (managed money net as % of open interest).
    A rising mm_net_pct_oi = increasing speculative positioning.
    """
    contracts = data.get("contracts") if isinstance(data, dict) else None
    if not contracts or not isinstance(contracts, list):
        return []

    results: list[Evidence] = []
    for c in contracts:
        if not isinstance(c, dict):
            continue
        market = c.get("Market_and_Exchange_Names", "")
        mm_net_pct = c.get("_mm_net_pct_oi")
        if mm_net_pct is None:
            continue

        # Derive a clean commodity slug from the market name
        slug = (
            market.split("-")[0].strip().lower().replace(" ", "_")[:30]
            if market
            else "unknown"
        )
        val = _safe_float(mm_net_pct)

        # Direction: positive net = speculative longs dominate (+1 stress/risk-on)
        direction = 1 if val > 0 else (-1 if val < 0 else 0)

        results.append(
            Evidence(
                source=tool_name,
                signal_id=f"cftc.{slug}.mm_net_pct_oi",
                timestamp=_now_ts(),
                value=val,
                direction=direction,
                confidence=0.85,
                category="positioning",
                tags=("commodity", slug),
                ttl=604_800,  # 7 days — weekly data
            )
        )

    return results


register_extractor("cftc", _extract_cftc)


# ── 2. Weather Alerts ─────────────────────────────────────────

_SEVERITY_SCORE = {
    "Extreme": 1.0,
    "Severe": 0.8,
    "Moderate": 0.5,
    "Minor": 0.2,
    "Unknown": 0.1,
}


def _extract_weather_alerts(tool_name: str, data: Any) -> list[Evidence]:
    """Extract alert count and severity from weather_alerts data.

    Mode 'summary' gives us alert_count + fire_count_infra.
    Mode 'alerts' gives individual alert records.
    We handle both gracefully.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    # Summary mode
    alert_count = data.get("alert_count")
    if alert_count is not None:
        results.append(
            Evidence(
                source=tool_name,
                signal_id="weather.us.alert_count",
                timestamp=ts,
                value=_safe_float(alert_count),
                direction=1 if _safe_int(alert_count) > 0 else 0,
                confidence=0.7,
                category="physical_disruption",
                tags=("weather", "us"),
                ttl=43_200,  # 12 hours
            )
        )

    fire_count = data.get("fire_count_infra") or data.get("count")
    fires = data.get("fires")
    if fire_count is not None and fires is None:
        results.append(
            Evidence(
                source=tool_name,
                signal_id="weather.global.fire_count_infra",
                timestamp=ts,
                value=_safe_float(fire_count),
                direction=1 if _safe_int(fire_count) > 0 else 0,
                confidence=0.6,
                category="physical_disruption",
                tags=("fire", "infrastructure"),
                ttl=43_200,
            )
        )

    # Alerts mode — aggregate by severity
    alerts = data.get("alerts")
    if alerts and isinstance(alerts, list):
        market_relevant_count = sum(
            1 for a in alerts if isinstance(a, dict) and a.get("market_relevant")
        )
        severe_count = sum(
            1
            for a in alerts
            if isinstance(a, dict) and a.get("severity") in ("Extreme", "Severe")
        )
        results.append(
            Evidence(
                source=tool_name,
                signal_id="weather.us.severe_alert_count",
                timestamp=ts,
                value=float(severe_count),
                direction=1 if severe_count > 0 else 0,
                confidence=0.75,
                category="physical_disruption",
                tags=("weather", "us", "severe"),
                ttl=43_200,
            )
        )
        if market_relevant_count > 0:
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="weather.us.market_relevant_alerts",
                    timestamp=ts,
                    value=float(market_relevant_count),
                    direction=1,
                    confidence=0.8,
                    category="physical_disruption",
                    tags=("weather", "us", "market_relevant"),
                    ttl=43_200,
                )
            )

    # Fires mode — zone-level
    if fires and isinstance(fires, list):
        zones_affected = data.get("zones_affected", [])
        results.append(
            Evidence(
                source=tool_name,
                signal_id="weather.global.infra_fire_zones",
                timestamp=ts,
                value=float(
                    len(zones_affected) if isinstance(zones_affected, list) else 0
                ),
                direction=1 if zones_affected else 0,
                confidence=0.65,
                category="physical_disruption",
                tags=("fire", "infrastructure"),
                ttl=43_200,
            )
        )

    return results


register_extractor("weather_alerts", _extract_weather_alerts)


# ── 3. Sanctions Monitor ─────────────────────────────────────


def _extract_sanctions_monitor(tool_name: str, data: Any) -> list[Evidence]:
    """Extract sanctions additions from sanctions_monitor data.

    Key signal: count of recently listed entities (recent mode).
    Programs mode: number of active programs.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    # Recent mode
    recent_results = data.get("results")
    count = data.get("count", 0)
    days_back = data.get("days_back")

    if days_back is not None and isinstance(recent_results, list):
        results.append(
            Evidence(
                source=tool_name,
                signal_id="sanctions.global.recent_additions",
                timestamp=ts,
                value=_safe_float(count),
                direction=1 if count > 0 else 0,
                confidence=0.9,
                category="regulatory_action",
                tags=("sanctions", "ofac", "un"),
                ttl=86_400,
            )
        )

    # Programs mode
    programs = data.get("programs")
    if programs and isinstance(programs, list) and days_back is None:
        total_entries = sum(
            _safe_int(p.get("count")) for p in programs if isinstance(p, dict)
        )
        results.append(
            Evidence(
                source=tool_name,
                signal_id="sanctions.global.program_count",
                timestamp=ts,
                value=float(len(programs)),
                direction=0,  # Neutral — count alone is not directional
                confidence=0.85,
                category="regulatory_action",
                tags=("sanctions", "programs"),
                ttl=86_400,
            )
        )
        results.append(
            Evidence(
                source=tool_name,
                signal_id="sanctions.global.total_entries",
                timestamp=ts,
                value=float(total_entries),
                direction=0,
                confidence=0.85,
                category="regulatory_action",
                tags=("sanctions",),
                ttl=86_400,
            )
        )

    # Search mode — not useful for convergence (single-entity lookup)

    return results


register_extractor("sanctions_monitor", _extract_sanctions_monitor)


# ── 4. AIS Vessel Tracking ───────────────────────────────────


def _extract_ais_vessel(tool_name: str, data: Any) -> list[Evidence]:
    """Extract vessel counts and type distribution from AIS area data.

    Key signal: total_vessels and tanker_ratio in strategic areas.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    area = data.get("area", "unknown")
    total_vessels = data.get("total_vessels")

    if total_vessels is not None:
        slug = str(area).lower().replace(" ", "_")[:30]
        results.append(
            Evidence(
                source=tool_name,
                signal_id=f"ais.{slug}.vessel_count",
                timestamp=ts,
                value=_safe_float(total_vessels),
                direction=0,  # Count alone isn't directional
                confidence=0.7,
                category="physical_flow",
                tags=("shipping", slug),
                ttl=86_400,
            )
        )

        type_counts = data.get("type_counts")
        if isinstance(type_counts, dict):
            tankers = _safe_float(type_counts.get("tanker"))
            total = _safe_float(total_vessels)
            if total > 0:
                results.append(
                    Evidence(
                        source=tool_name,
                        signal_id=f"ais.{slug}.tanker_ratio",
                        timestamp=ts,
                        value=tankers / total,
                        direction=0,
                        confidence=0.65,
                        category="physical_flow",
                        tags=("shipping", slug, "tanker"),
                        ttl=86_400,
                    )
                )

    # Destination flow mode — strategic chokepoint traffic
    strategic = data.get("strategic")
    if isinstance(strategic, dict):
        for chokepoint, count in strategic.items():
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id=f"ais.destination.{chokepoint}",
                    timestamp=ts,
                    value=_safe_float(count),
                    direction=0,
                    confidence=0.6,
                    category="physical_flow",
                    tags=("shipping", "destination", str(chokepoint)),
                    ttl=86_400,
                )
            )

    return results


register_extractor("ais_vessel_tracking", _extract_ais_vessel)


# ── 5. FINRA Short Volume ────────────────────────────────────


def _extract_finra_short_volume(tool_name: str, data: Any) -> list[Evidence]:
    """Extract short volume ratio and anomaly flags.

    Key signal: latest short_ratio and whether it's anomalous.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    signals = data.get("signals")
    ticker = data.get("ticker", "unknown")

    if isinstance(signals, dict) and ticker != "unknown":
        slug = ticker.lower()
        ratio = signals.get("latest_ratio")
        if ratio is not None:
            zscore = _safe_float(signals.get("zscore"))
            is_anomaly = bool(signals.get("is_anomaly"))
            # High short ratio = bearish positioning = +1 stress
            direction = 1 if _safe_float(ratio) > 0.5 else -1

            results.append(
                Evidence(
                    source=tool_name,
                    signal_id=f"finra.{slug}.short_ratio",
                    timestamp=ts,
                    value=_safe_float(ratio),
                    direction=direction,
                    confidence=0.8 if is_anomaly else 0.6,
                    category="positioning",
                    tags=("short_volume", slug),
                    ttl=86_400,
                )
            )

    # Scan mode — aggregate across tickers
    scan_results = data.get("results")
    if isinstance(scan_results, list) and "ticker" not in data:
        total_tickers = _safe_int(data.get("total_tickers"))
        if total_tickers > 0:
            avg_ratio = sum(
                _safe_float(r.get("short_ratio"))
                for r in scan_results
                if isinstance(r, dict)
            ) / max(len(scan_results), 1)
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="finra.market.avg_short_ratio",
                    timestamp=ts,
                    value=avg_ratio,
                    direction=1 if avg_ratio > 0.5 else -1,
                    confidence=0.7,
                    category="positioning",
                    tags=("short_volume", "market_wide"),
                    ttl=86_400,
                )
            )

    # Short interest mode
    si_signals = data.get("signals")
    if isinstance(si_signals, dict) and "squeeze_risk" in si_signals:
        slug = data.get("ticker", "unknown").lower()
        dtc = _safe_float(si_signals.get("days_to_cover"))
        results.append(
            Evidence(
                source=tool_name,
                signal_id=f"finra.{slug}.days_to_cover",
                timestamp=ts,
                value=dtc,
                direction=1 if dtc > 5.0 else 0,
                confidence=0.75,
                category="positioning",
                tags=("short_interest", slug),
                ttl=86_400 * 14,  # Bi-monthly data
            )
        )

    return results


register_extractor("finra_short_volume", _extract_finra_short_volume)


# ── 6. Disease Surveillance ──────────────────────────────────


def _extract_disease_surveillance(tool_name: str, data: Any) -> list[Evidence]:
    """Extract pathogen detection rates and outbreak counts.

    Key signals: wastewater detection_rate, WHO outbreak count.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    # Wastewater mode
    summaries = data.get("summaries")
    pathogen = data.get("pathogen", "unknown")
    if isinstance(summaries, list) and pathogen:
        slug = pathogen.lower().replace("-", "_").replace(" ", "_")[:30]
        total_samples = _safe_int(data.get("total_samples"))
        hot_states = _safe_int(data.get("hot_states"))

        if total_samples > 0:
            # Aggregate national detection rate
            total_detect = sum(
                _safe_int(s.get("detections")) for s in summaries if isinstance(s, dict)
            )
            rate = total_detect / total_samples if total_samples > 0 else 0

            results.append(
                Evidence(
                    source=tool_name,
                    signal_id=f"disease.{slug}.detection_rate",
                    timestamp=ts,
                    value=rate,
                    direction=1 if rate > 0.3 else 0,
                    confidence=0.8,
                    category="biological",
                    tags=("wastewater", slug),
                    ttl=604_800,  # Weekly
                )
            )

        if hot_states is not None:
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id=f"disease.{slug}.hot_states",
                    timestamp=ts,
                    value=float(hot_states),
                    direction=1 if hot_states > 5 else 0,
                    confidence=0.75,
                    category="biological",
                    tags=("wastewater", slug, "geographic_spread"),
                    ttl=604_800,
                )
            )

    # Outbreak mode (WHO DON)
    entries = data.get("entries")
    if isinstance(entries, list):
        results.append(
            Evidence(
                source=tool_name,
                signal_id="disease.who.outbreak_count",
                timestamp=ts,
                value=float(len(entries)),
                direction=1 if len(entries) > 3 else 0,
                confidence=0.7,
                category="biological",
                tags=("who", "outbreaks"),
                ttl=604_800,
            )
        )

    return results


register_extractor("disease_surveillance", _extract_disease_surveillance)


# ── 7. Earthquake Proximity ──────────────────────────────────


def _extract_earthquake_proximity(tool_name: str, data: Any) -> list[Evidence]:
    """Extract earthquake count, max magnitude, and infra-proximity hits.

    Key signals: count near infrastructure, maximum magnitude.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    quakes = data.get("quakes")
    if not isinstance(quakes, list):
        return []

    count = len(quakes)
    if count == 0:
        return []

    # Max magnitude
    max_mag = max(
        (_safe_float(q.get("magnitude")) for q in quakes if isinstance(q, dict)),
        default=0.0,
    )
    near_infra = _safe_int(data.get("near_infrastructure"))

    # Zone-specific (monitor mode)
    zone = data.get("zone")
    if isinstance(zone, dict):
        zone_name = zone.get("name", "unknown").lower().replace(" ", "_")[:30]
        results.append(
            Evidence(
                source=tool_name,
                signal_id=f"earthquake.{zone_name}.count",
                timestamp=ts,
                value=float(count),
                direction=1 if max_mag >= 4.0 else 0,
                confidence=0.9,
                category="physical_disruption",
                tags=("earthquake", zone_name, zone.get("sector", "")),
                ttl=86_400,
            )
        )
        results.append(
            Evidence(
                source=tool_name,
                signal_id=f"earthquake.{zone_name}.max_magnitude",
                timestamp=ts,
                value=max_mag,
                direction=1 if max_mag >= 5.0 else 0,
                confidence=0.95,
                category="physical_disruption",
                tags=("earthquake", zone_name),
                ttl=86_400,
            )
        )
        return results

    # Global (recent mode)
    results.append(
        Evidence(
            source=tool_name,
            signal_id="earthquake.global.count",
            timestamp=ts,
            value=float(count),
            direction=1 if count > 5 else 0,
            confidence=0.9,
            category="physical_disruption",
            tags=("earthquake", "global"),
            ttl=86_400,
        )
    )
    results.append(
        Evidence(
            source=tool_name,
            signal_id="earthquake.global.max_magnitude",
            timestamp=ts,
            value=max_mag,
            direction=1 if max_mag >= 6.0 else 0,
            confidence=0.95,
            category="physical_disruption",
            tags=("earthquake",),
            ttl=86_400,
        )
    )
    if near_infra > 0:
        results.append(
            Evidence(
                source=tool_name,
                signal_id="earthquake.global.near_infrastructure",
                timestamp=ts,
                value=float(near_infra),
                direction=1,
                confidence=0.9,
                category="physical_disruption",
                tags=("earthquake", "infrastructure"),
                ttl=86_400,
            )
        )

    return results


register_extractor("earthquake_proximity", _extract_earthquake_proximity)


# ── 8. Global PMI ────────────────────────────────────────────


def _extract_global_pmi(tool_name: str, data: Any) -> list[Evidence]:
    """Extract per-country CLI/BCI/CCI signals from OECD data.

    Key signal: per-country regime and month-over-month change.
    CLI > 100 = expansion, < 100 = contraction.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()
    mode = data.get("mode", "cli")

    signals = data.get("signals")
    if not isinstance(signals, dict):
        return []

    for country, sig in signals.items():
        if country.startswith("_") or not isinstance(sig, dict):
            continue

        slug = country.lower()
        latest_val = sig.get("latest_value")
        if latest_val is None:
            continue

        val = _safe_float(latest_val)
        mom = _safe_float(sig.get("mom_change"))
        regime = sig.get("regime", "")

        # CLI/BCI: > 100 = expanding, < 100 = contracting
        direction = 1 if val > 100 else (-1 if val < 100 else 0)

        results.append(
            Evidence(
                source=tool_name,
                signal_id=f"pmi.{slug}.{mode}",
                timestamp=ts,
                value=val,
                direction=direction,
                confidence=0.8,
                category="macro_momentum",
                tags=(mode, slug, regime) if regime else (mode, slug),
                ttl=2_592_000,  # 30 days — monthly data
            )
        )

        # Month-over-month momentum as a separate signal
        if mom != 0:
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id=f"pmi.{slug}.{mode}.mom",
                    timestamp=ts,
                    value=mom,
                    direction=1 if mom > 0 else -1,
                    confidence=0.75,
                    category="macro_momentum",
                    tags=(mode, slug, "momentum"),
                    ttl=2_592_000,
                )
            )

    return results


register_extractor("global_pmi", _extract_global_pmi)


# ── 9. Treasury Receipts ─────────────────────────────────────


def _extract_treasury_receipts(tool_name: str, data: Any) -> list[Evidence]:
    """Extract Treasury General Account balance and flows.

    Key signals: TGA daily change, net deposit/withdrawal flow.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    signals = data.get("signals")
    mode = data.get("mode", "")

    if isinstance(signals, dict):
        # Cash balance mode — TGA change
        tga_change = signals.get("tga_daily_change_pct")
        if tga_change is not None:
            val = _safe_float(tga_change)
            # TGA drop = Treasury spending = liquidity injection (+1)
            # TGA rise = Treasury rebuilding cash = liquidity drain (-1)
            direction = -1 if val > 0 else (1 if val < 0 else 0)
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="treasury.tga.daily_change_pct",
                    timestamp=ts,
                    value=val,
                    direction=direction,
                    confidence=0.85,
                    category="macro_momentum",
                    tags=("treasury", "tga", "liquidity"),
                    ttl=86_400,
                )
            )

        # Deposits/withdrawals mode — net flow
        net_flow = signals.get("net_flow_today")
        if net_flow is not None:
            val = _safe_float(net_flow)
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="treasury.us.net_flow_today",
                    timestamp=ts,
                    value=val,
                    direction=1 if val > 0 else (-1 if val < 0 else 0),
                    confidence=0.8,
                    category="macro_momentum",
                    tags=("treasury", "flow"),
                    ttl=86_400,
                )
            )

        total_deposits = signals.get("total_deposits_today")
        if total_deposits is not None:
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="treasury.us.deposits_today",
                    timestamp=ts,
                    value=_safe_float(total_deposits),
                    direction=0,
                    confidence=0.8,
                    category="macro_momentum",
                    tags=("treasury", "deposits"),
                    ttl=86_400,
                )
            )

    return results


register_extractor("treasury_receipts", _extract_treasury_receipts)


# ── 10. Job Postings (JOLTS) ─────────────────────────────────


def _extract_job_postings(tool_name: str, data: Any) -> list[Evidence]:
    """Extract labor market signals from JOLTS/BLS data.

    Key signals: job openings, quits, layoffs, unemployment rate, initial claims.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    summary = data.get("summary")
    if not isinstance(summary, dict):
        return []

    # Map FRED/BLS series IDs to our signal taxonomy
    _SERIES_MAP: dict[str, tuple[str, bool, float]] = {
        # series_id: (signal_suffix, flip_sign, confidence)
        # flip_sign=True means higher raw = less economic stress
        "JTSJOL": ("openings", True, 0.75),  # More openings = less stress
        "JTSQUL": ("quits", True, 0.7),  # More quits = confidence = less stress
        "JTSHIL": ("hires", True, 0.7),  # More hires = less stress
        "JTSLDR": ("layoffs", False, 0.75),  # More layoffs = more stress
        "UNRATE": ("unemployment_rate", False, 0.8),  # Higher = more stress
        "ICSA": ("initial_claims", False, 0.8),  # Higher = more stress
        "PAYEMS": ("payrolls", True, 0.75),  # More payrolls = less stress
    }

    for series_id, (suffix, flip_sign, conf) in _SERIES_MAP.items():
        entry = summary.get(series_id)
        if not isinstance(entry, dict):
            continue

        latest_val = entry.get("latest_value")
        if latest_val is None:
            continue

        val = _safe_float(latest_val)
        trend = entry.get("trend", "")

        # Direction after conceptual flip: positive = stress/contraction
        if flip_sign:
            direction = -1 if trend == "rising" else (1 if trend == "falling" else 0)
        else:
            direction = 1 if trend == "rising" else (-1 if trend == "falling" else 0)

        results.append(
            Evidence(
                source=tool_name,
                signal_id=f"jobs.us.{suffix}",
                timestamp=ts,
                value=val,
                direction=direction,
                confidence=conf,
                category=(
                    "behavioral_intent"
                    if suffix in ("openings", "quits", "hires")
                    else "macro_momentum"
                ),
                tags=("labor", "us", suffix),
                ttl=2_592_000,  # Monthly data
            )
        )

    return results


register_extractor("job_postings", _extract_job_postings)


# ══════════════════════════════════════════════════════════════
#  EXTRACTOR IMPLEMENTATIONS — Batch 2 (remaining tools)
# ══════════════════════════════════════════════════════════════


# ── 11. Transport Throughput ──────────────────────────────────


def _extract_transport_throughput(tool_name: str, data: Any) -> list[Evidence]:
    """Extract border-crossing trade volume from BTS data.

    Key signal: total trade volume across borders. Declining totals
    = trade contraction = physical flow disruption.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    # Recent / trend mode — aggregate records
    records = data.get("records")
    if isinstance(records, list) and records:
        total = sum(_safe_float(r.get("total")) for r in records if isinstance(r, dict))
        count = len(records)
        results.append(
            Evidence(
                source=tool_name,
                signal_id="transport.us.border_total",
                timestamp=ts,
                value=total,
                direction=0,  # Level signal — anomaly detection handles direction
                confidence=0.7,
                category="physical_flow",
                tags=("transport", "border", "us"),
                ttl=604_800,  # Weekly
            )
        )

    # Comparison mode — Canada/Mexico ratio
    comparison = data.get("comparison")
    if isinstance(comparison, list) and comparison:
        latest = comparison[-1] if isinstance(comparison[-1], dict) else {}
        ratio = _safe_float(latest.get("ratio"))
        if ratio > 0:
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="transport.us.canada_mexico_ratio",
                    timestamp=ts,
                    value=ratio,
                    direction=0,
                    confidence=0.6,
                    category="physical_flow",
                    tags=("transport", "border", "ratio"),
                    ttl=604_800,
                )
            )

    # Series mode (trend)
    series = data.get("series")
    if isinstance(series, list) and len(series) >= 2:
        vals = [_safe_float(s.get("total")) for s in series if isinstance(s, dict)]
        if len(vals) >= 2 and vals[-2] > 0:
            pct_change = (vals[-1] - vals[-2]) / vals[-2]
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="transport.us.volume_change",
                    timestamp=ts,
                    value=pct_change,
                    direction=(
                        -1 if pct_change < -0.05 else (1 if pct_change > 0.05 else 0)
                    ),
                    confidence=0.65,
                    category="physical_flow",
                    tags=("transport", "momentum"),
                    ttl=604_800,
                )
            )

    return results


register_extractor("transport_throughput", _extract_transport_throughput)


# ── 12. Capital Flows ────────────────────────────────────────


def _extract_capital_flows(tool_name: str, data: Any) -> list[Evidence]:
    """Extract sovereign holdings, flow reversals, and reserve stress.

    Key signals: coordinated selling/buying, reserve drawdowns, flow reversals.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()
    mode = data.get("mode", "")

    # Holdings mode — coordination signals
    coordination = data.get("coordination")
    if isinstance(coordination, dict):
        sellers = coordination.get("sellers")
        buyers = coordination.get("buyers")
        if coordination.get("coordinated_selling"):
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="capital_flows.ust.coordinated_selling",
                    timestamp=ts,
                    value=float(len(sellers) if isinstance(sellers, list) else 0),
                    direction=1,  # Stress — multiple sovereigns dumping UST
                    confidence=0.85,
                    category="monetary_policy",
                    tags=("treasury", "holdings", "selling"),
                    ttl=2_592_000,  # Monthly
                )
            )
        if coordination.get("coordinated_buying"):
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="capital_flows.ust.coordinated_buying",
                    timestamp=ts,
                    value=float(len(buyers) if isinstance(buyers, list) else 0),
                    direction=-1,  # Relief — sovereigns adding UST
                    confidence=0.85,
                    category="monetary_policy",
                    tags=("treasury", "holdings", "buying"),
                    ttl=2_592_000,
                )
            )

    # Holdings entries — per-country MoM change
    holdings = data.get("holdings")
    if isinstance(holdings, list):
        for h in holdings:
            if not isinstance(h, dict):
                continue
            country = h.get("country", "")
            mom = h.get("mom_change_pct")
            if mom is None or not country:
                continue
            slug = country.lower().replace(" ", "_")[:20]
            val = _safe_float(mom)
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id=f"capital_flows.{slug}.holdings_mom_pct",
                    timestamp=ts,
                    value=val,
                    direction=1 if val < -3.0 else (-1 if val > 3.0 else 0),
                    confidence=0.7,
                    category="monetary_policy",
                    tags=("treasury", "holdings", slug),
                    ttl=2_592_000,
                )
            )

    # Flows mode — flow reversals
    flows = data.get("flows")
    if isinstance(flows, list):
        for f in flows:
            if not isinstance(f, dict):
                continue
            if f.get("flow_reversal"):
                key = f.get("key", "unknown")
                results.append(
                    Evidence(
                        source=tool_name,
                        signal_id=f"capital_flows.{key}.reversal",
                        timestamp=ts,
                        value=_safe_float(f.get("latest_value")),
                        direction=1,  # Flow reversal = stress
                        confidence=0.8,
                        category="monetary_policy",
                        tags=("capital_flows", "reversal"),
                        ttl=2_592_000,
                    )
                )

    # Reserves mode — stress alerts
    stress_alerts = data.get("stress_alerts")
    if isinstance(stress_alerts, list) and stress_alerts:
        results.append(
            Evidence(
                source=tool_name,
                signal_id="capital_flows.reserves.stress_count",
                timestamp=ts,
                value=float(len(stress_alerts)),
                direction=1,
                confidence=0.85,
                category="monetary_policy",
                tags=("reserves", "stress"),
                ttl=2_592_000,
            )
        )

    return results


register_extractor("capital_flows", _extract_capital_flows)


# ── 13. Sovereign Debt ───────────────────────────────────────


def _extract_sovereign_debt(tool_name: str, data: Any) -> list[Evidence]:
    """Extract yield curve shape and spread signals.

    Key signals: 2s10s inversion, peripheral spreads widening.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    # US yields — curve shape
    records = data.get("records")
    if isinstance(records, list) and records:
        latest = records[-1] if isinstance(records[-1], dict) else {}
        curve_2s10s = latest.get("curve_2s10s")
        curve_3m10y = latest.get("curve_3m10y")

        if curve_2s10s is not None:
            val = _safe_float(curve_2s10s)
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="sovereign.us.curve_2s10s",
                    timestamp=ts,
                    value=val,
                    direction=1 if val < 0 else -1,  # Inversion = stress
                    confidence=0.9,
                    category="financial_stress",
                    tags=("yields", "us", "curve"),
                    ttl=86_400,
                )
            )

        if curve_3m10y is not None:
            val = _safe_float(curve_3m10y)
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="sovereign.us.curve_3m10y",
                    timestamp=ts,
                    value=val,
                    direction=1 if val < 0 else -1,
                    confidence=0.9,
                    category="financial_stress",
                    tags=("yields", "us", "curve"),
                    ttl=86_400,
                )
            )

    # Spreads mode — peripheral vs. Germany
    spreads = data.get("spreads")
    if isinstance(spreads, list):
        for s in spreads:
            if not isinstance(s, dict):
                continue
            country = s.get("country", "")
            spread_val = s.get("spread_vs_de")
            if spread_val is None or not country:
                continue
            slug = country.lower().replace(" ", "_")[:20]
            val = _safe_float(spread_val)
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id=f"sovereign.{slug}.spread_vs_de",
                    timestamp=ts,
                    value=val,
                    direction=1 if val > 200 else 0,  # >200bps = stress
                    confidence=0.8,
                    category="financial_stress",
                    tags=("spreads", slug, "eu"),
                    ttl=86_400,
                )
            )

    # US curve from spreads mode
    us_curve = data.get("us_curve")
    if isinstance(us_curve, dict):
        for key in ("curve_2s10s", "curve_3m10y"):
            val_raw = us_curve.get(key)
            if val_raw is not None:
                val = _safe_float(val_raw)
                results.append(
                    Evidence(
                        source=tool_name,
                        signal_id=f"sovereign.us.{key}",
                        timestamp=ts,
                        value=val,
                        direction=1 if val < 0 else -1,
                        confidence=0.9,
                        category="financial_stress",
                        tags=("yields", "us", "curve"),
                        ttl=86_400,
                    )
                )

    return results


register_extractor("sovereign_debt", _extract_sovereign_debt)


# ── 14. Creditor Filings ─────────────────────────────────────


def _extract_creditor_filings(tool_name: str, data: Any) -> list[Evidence]:
    """Extract filing clusters and red flags from SEC/UK filings.

    Key signal: filing cluster count and red flag count.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    # SEC filing count
    sec_count = data.get("sec_count")
    if sec_count is not None:
        val = _safe_float(sec_count)
        results.append(
            Evidence(
                source=tool_name,
                signal_id="creditor.sec.filing_count",
                timestamp=ts,
                value=val,
                direction=1 if val > 5 else 0,
                confidence=0.7,
                category="financial_stress",
                tags=("creditor", "sec"),
                ttl=86_400,
            )
        )

    # Red flags (UK charges)
    red_flags = data.get("red_flags") or data.get("ch_red_flags")
    if red_flags is not None:
        val = _safe_float(red_flags)
        results.append(
            Evidence(
                source=tool_name,
                signal_id="creditor.uk.red_flags",
                timestamp=ts,
                value=val,
                direction=1 if val > 0 else 0,
                confidence=0.75,
                category="financial_stress",
                tags=("creditor", "uk", "red_flag"),
                ttl=86_400,
            )
        )

    # Stress scan — cluster count
    clusters = data.get("clusters")
    if isinstance(clusters, list) and clusters:
        results.append(
            Evidence(
                source=tool_name,
                signal_id="creditor.sec.cluster_count",
                timestamp=ts,
                value=float(len(clusters)),
                direction=1 if len(clusters) > 3 else 0,
                confidence=0.8,
                category="financial_stress",
                tags=("creditor", "stress_scan"),
                ttl=86_400,
            )
        )

    return results


register_extractor("creditor_filings", _extract_creditor_filings)


# ── 15. Bankruptcy Court ──────────────────────────────────────


def _extract_bankruptcy_court(tool_name: str, data: Any) -> list[Evidence]:
    """Extract bankruptcy filing counts and chapter breakdown.

    Key signal: total filings, chapter 11 (corporate reorganization) count.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()
    mode = data.get("mode", "")

    count = data.get("count")
    if count is not None:
        val = _safe_float(count)
        # More filings = more financial stress
        results.append(
            Evidence(
                source=tool_name,
                signal_id=f"bankruptcy.{mode or 'us'}.filing_count",
                timestamp=ts,
                value=val,
                direction=1 if val > 10 else 0,
                confidence=0.75,
                category="financial_stress",
                tags=("bankruptcy", mode or "us"),
                ttl=86_400,
            )
        )

    # Chapter breakdown — highlight Chapter 11
    chapter_breakdown = data.get("chapter_breakdown")
    if isinstance(chapter_breakdown, dict):
        ch11 = _safe_int(
            chapter_breakdown.get("11") or chapter_breakdown.get("Chapter 11")
        )
        if ch11 > 0:
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="bankruptcy.us.chapter_11",
                    timestamp=ts,
                    value=float(ch11),
                    direction=1,  # Corporate reorganization = stress
                    confidence=0.8,
                    category="financial_stress",
                    tags=("bankruptcy", "chapter_11", "corporate"),
                    ttl=86_400,
                )
            )

    # SEC enforcement count
    type_breakdown = data.get("type_breakdown")
    if isinstance(type_breakdown, dict) and mode == "sec_enforcement":
        results.append(
            Evidence(
                source=tool_name,
                signal_id="bankruptcy.sec.enforcement_count",
                timestamp=ts,
                value=_safe_float(count),
                direction=1 if _safe_int(count) > 3 else 0,
                confidence=0.8,
                category="financial_stress",
                tags=("sec", "enforcement"),
                ttl=86_400,
            )
        )

    return results


register_extractor("bankruptcy_court", _extract_bankruptcy_court)


# ── 16. Liquidity Regime ──────────────────────────────────────


def _extract_liquidity_regime(tool_name: str, data: Any) -> list[Evidence]:
    """Extract HMM regime state and composite z-score.

    Key signal: current regime, composite z-score, changepoint detection.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    regime = data.get("current_regime")
    zscore = data.get("composite_zscore")

    if regime is not None:
        # Map regime to direction: contraction=+1 (stress), expansion=-1 (relief)
        direction_map = {"contraction": 1, "neutral": 0, "expansion": -1}
        direction = direction_map.get(regime, 0)
        state = _safe_int(data.get("current_state"))

        results.append(
            Evidence(
                source=tool_name,
                signal_id="liquidity.us.regime",
                timestamp=ts,
                value=float(state),
                direction=direction,
                confidence=0.85,
                category="financial_stress",
                tags=("liquidity", "regime", str(regime)),
                ttl=604_800,  # Weekly
            )
        )

    if zscore is not None:
        val = _safe_float(zscore)
        results.append(
            Evidence(
                source=tool_name,
                signal_id="liquidity.us.composite_zscore",
                timestamp=ts,
                value=val,
                direction=1 if val < -1.5 else (-1 if val > 1.5 else 0),
                confidence=0.85,
                category="financial_stress",
                tags=("liquidity", "zscore"),
                ttl=604_800,
            )
        )

    n_cp = data.get("n_changepoints")
    if n_cp is not None:
        results.append(
            Evidence(
                source=tool_name,
                signal_id="liquidity.us.changepoint_count",
                timestamp=ts,
                value=_safe_float(n_cp),
                direction=0,
                confidence=0.7,
                category="financial_stress",
                tags=("liquidity", "changepoint"),
                ttl=604_800,
            )
        )

    return results


register_extractor("liquidity_regime", _extract_liquidity_regime)


# ── 17. Central Bank Balance ──────────────────────────────────


def _extract_central_bank_balance(tool_name: str, data: Any) -> list[Evidence]:
    """Extract balance sheet changes, net liquidity, and policy divergence.

    Key signals: WoW balance sheet change, net liquidity index,
    synchronized policy direction.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    # Balance sheets mode — per-bank WoW change
    banks = data.get("banks")
    if isinstance(banks, list):
        for b in banks:
            if not isinstance(b, dict):
                continue
            code = b.get("code", "unknown").lower()
            wow = b.get("wow_pct")
            if wow is None:
                continue
            val = _safe_float(wow)
            # Balance sheet expansion = easing = -1 (relief)
            # Balance sheet contraction = tightening = +1 (stress)
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id=f"cb.{code}.balance_wow_pct",
                    timestamp=ts,
                    value=val,
                    direction=1 if val < -0.5 else (-1 if val > 0.5 else 0),
                    confidence=0.8,
                    category="monetary_policy",
                    tags=("central_bank", code, "balance_sheet"),
                    ttl=604_800,
                )
            )

    # Liquidity index mode
    net_usd = data.get("net_usd")
    if net_usd is not None:
        val = _safe_float(net_usd)
        results.append(
            Evidence(
                source=tool_name,
                signal_id="cb.us.net_liquidity_usd",
                timestamp=ts,
                value=val,
                direction=0,  # Level signal — let anomaly detection handle
                confidence=0.85,
                category="monetary_policy",
                tags=("liquidity", "net", "usd"),
                ttl=604_800,
            )
        )

    # Policy divergence mode
    synchronized = data.get("synchronized")
    if synchronized is not None:
        results.append(
            Evidence(
                source=tool_name,
                signal_id="cb.global.policy_synchronized",
                timestamp=ts,
                value=1.0 if synchronized else 0.0,
                direction=0,
                confidence=0.75,
                category="monetary_policy",
                tags=("policy", "divergence"),
                ttl=2_592_000,
            )
        )

    divergences = data.get("divergences")
    if isinstance(divergences, list) and divergences:
        results.append(
            Evidence(
                source=tool_name,
                signal_id="cb.global.divergence_count",
                timestamp=ts,
                value=float(len(divergences)),
                direction=1 if len(divergences) >= 3 else 0,
                confidence=0.7,
                category="monetary_policy",
                tags=("policy", "divergence"),
                ttl=2_592_000,
            )
        )

    return results


register_extractor("central_bank_balance", _extract_central_bank_balance)


# ── 18. Drug Regulatory ───────────────────────────────────────


def _extract_drug_regulatory(tool_name: str, data: Any) -> list[Evidence]:
    """Extract FDA approval and adverse event signals.

    Key signal: adverse event seriousness ratio, approval count.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()
    mode = data.get("mode", "")

    # Approvals — count
    if mode == "approvals":
        total = data.get("total")
        if total is not None:
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="fda.approvals.count",
                    timestamp=ts,
                    value=_safe_float(total),
                    direction=0,
                    confidence=0.7,
                    category="regulatory_action",
                    tags=("fda", "approvals"),
                    ttl=86_400,
                )
            )

    # Adverse events — seriousness ratio
    if mode == "adverse_events":
        signals = data.get("signals")
        if isinstance(signals, dict):
            ratio = signals.get("seriousness_ratio")
            if ratio is not None:
                val = _safe_float(ratio)
                results.append(
                    Evidence(
                        source=tool_name,
                        signal_id="fda.adverse_events.seriousness_ratio",
                        timestamp=ts,
                        value=val,
                        direction=1 if val > 0.5 else 0,
                        confidence=0.75,
                        category="regulatory_action",
                        tags=("fda", "adverse_events"),
                        ttl=86_400,
                    )
                )
            serious = signals.get("serious_count")
            if serious is not None:
                results.append(
                    Evidence(
                        source=tool_name,
                        signal_id="fda.adverse_events.serious_count",
                        timestamp=ts,
                        value=_safe_float(serious),
                        direction=1 if _safe_int(serious) > 10 else 0,
                        confidence=0.7,
                        category="regulatory_action",
                        tags=("fda", "adverse_events", "serious"),
                        ttl=86_400,
                    )
                )

    # Labels — boxed warning presence (batch over results)
    results_list = data.get("results")
    if mode == "labels" and isinstance(results_list, list):
        boxed = sum(
            1
            for r in results_list
            if isinstance(r, dict) and r.get("has_boxed_warning")
        )
        if boxed > 0:
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="fda.labels.boxed_warning_count",
                    timestamp=ts,
                    value=float(boxed),
                    direction=1,
                    confidence=0.8,
                    category="regulatory_action",
                    tags=("fda", "labels", "boxed_warning"),
                    ttl=604_800,
                )
            )

    return results


register_extractor("drug_regulatory", _extract_drug_regulatory)


# ── 19. Regulatory Gazette ────────────────────────────────────


def _extract_regulatory_gazette(tool_name: str, data: Any) -> list[Evidence]:
    """Extract Federal Register document counts, significant rules.

    Key signal: significant rule count, total regulatory volume.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    documents = data.get("documents")
    if not isinstance(documents, list):
        return []

    count = data.get("count", len(documents))
    significant_count = sum(
        1 for d in documents if isinstance(d, dict) and d.get("significant")
    )

    results.append(
        Evidence(
            source=tool_name,
            signal_id="regulatory.us.document_count",
            timestamp=ts,
            value=_safe_float(count),
            direction=0,  # Volume — anomaly detection handles direction
            confidence=0.65,
            category="regulatory_action",
            tags=("federal_register", "us"),
            ttl=86_400,
        )
    )

    if significant_count > 0:
        results.append(
            Evidence(
                source=tool_name,
                signal_id="regulatory.us.significant_count",
                timestamp=ts,
                value=float(significant_count),
                direction=1,  # Significant rules = regulatory change
                confidence=0.8,
                category="regulatory_action",
                tags=("federal_register", "us", "significant"),
                ttl=86_400,
            )
        )

    return results


register_extractor("regulatory_gazette", _extract_regulatory_gazette)


# ── 20. Building Permits ─────────────────────────────────────


def _extract_building_permits(tool_name: str, data: Any) -> list[Evidence]:
    """Extract housing permit trends and momentum.

    Key signal: MoM change, consecutive declines, YoY change.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    summary = data.get("summary")
    if not isinstance(summary, dict):
        return []

    for series_id, entry in summary.items():
        if not isinstance(entry, dict):
            continue
        mom = entry.get("mom_pct")
        yoy = entry.get("yoy_pct")
        declines = entry.get("consecutive_declines", 0)
        label = entry.get("label", series_id)
        slug = series_id.lower().replace(" ", "_")[:30]

        if mom is not None:
            val = _safe_float(mom)
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id=f"permits.us.{slug}.mom_pct",
                    timestamp=ts,
                    value=val,
                    direction=1 if val < -5 else (-1 if val > 5 else 0),
                    confidence=0.7,
                    category="macro_momentum",
                    tags=("housing", "permits", slug),
                    ttl=2_592_000,  # Monthly
                )
            )

        if _safe_int(declines) >= 3:
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id=f"permits.us.{slug}.consecutive_declines",
                    timestamp=ts,
                    value=float(declines),
                    direction=1,  # Sustained decline = macro weakness
                    confidence=0.8,
                    category="macro_momentum",
                    tags=("housing", "permits", slug, "declining"),
                    ttl=2_592_000,
                )
            )

    return results


register_extractor("building_permits", _extract_building_permits)


# ── 21. Patent Filings ───────────────────────────────────────


def _extract_patent_filings(tool_name: str, data: Any) -> list[Evidence]:
    """Extract patent filing trends for signal technology classes.

    Key signal: total count, trend-class signals.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()
    mode = data.get("mode", "")

    # Trends mode — per-CPC class yearly counts
    yearly_counts = data.get("yearly_counts")
    if isinstance(yearly_counts, dict) and yearly_counts:
        cpc = data.get("cpc_class", "unknown")
        years = sorted(yearly_counts.keys())
        total = data.get("total_count", 0)
        results.append(
            Evidence(
                source=tool_name,
                signal_id=f"patent.{cpc}.total_count",
                timestamp=ts,
                value=_safe_float(total),
                direction=0,
                confidence=0.6,
                category="behavioral_intent",
                tags=("patent", cpc),
                ttl=2_592_000,
            )
        )
        # YoY growth if enough years
        if len(years) >= 2:
            prev = _safe_float(yearly_counts.get(years[-2]))
            curr = _safe_float(yearly_counts.get(years[-1]))
            if prev > 0:
                growth = (curr - prev) / prev
                results.append(
                    Evidence(
                        source=tool_name,
                        signal_id=f"patent.{cpc}.yoy_growth",
                        timestamp=ts,
                        value=growth,
                        direction=1 if growth > 0.2 else (-1 if growth < -0.2 else 0),
                        confidence=0.6,
                        category="behavioral_intent",
                        tags=("patent", cpc, "growth"),
                        ttl=2_592_000,
                    )
                )

    # Signal classes (strategic CPC categories)
    signal_classes = data.get("signal_classes")
    if isinstance(signal_classes, dict):
        results.append(
            Evidence(
                source=tool_name,
                signal_id="patent.signal_classes.count",
                timestamp=ts,
                value=float(len(signal_classes)),
                direction=0,
                confidence=0.5,
                category="behavioral_intent",
                tags=("patent", "signal_classes"),
                ttl=2_592_000,
            )
        )

    # Search mode — total count
    if mode == "search":
        total = data.get("total_count")
        if total is not None:
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="patent.search.total_count",
                    timestamp=ts,
                    value=_safe_float(total),
                    direction=0,
                    confidence=0.5,
                    category="behavioral_intent",
                    tags=("patent", "search"),
                    ttl=604_800,
                )
            )

    return results


register_extractor("patent_filings", _extract_patent_filings)


# ── 22. Lobbying ──────────────────────────────────────────────


def _extract_lobbying(tool_name: str, data: Any) -> list[Evidence]:
    """Extract lobbying spend signals and anomalies.

    Key signal: spending anomaly detection, filing volume.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()
    mode = data.get("mode", "")

    # Spending mode — anomaly flag
    anomaly = data.get("anomaly")
    if isinstance(anomaly, dict):
        is_anomaly = anomaly.get("anomaly", False)
        ratio = _safe_float(anomaly.get("ratio"))
        target = data.get("target", "unknown")
        slug = target.lower().replace(" ", "_")[:30]

        results.append(
            Evidence(
                source=tool_name,
                signal_id=f"lobbying.{slug}.spend_anomaly",
                timestamp=ts,
                value=ratio,
                direction=1 if is_anomaly else 0,
                confidence=0.7 if is_anomaly else 0.4,
                category="behavioral_intent",
                tags=("lobbying", slug),
                ttl=2_592_000,
            )
        )

    # Search mode — filing volume
    total_count = data.get("total_count")
    if total_count is not None and mode == "search":
        results.append(
            Evidence(
                source=tool_name,
                signal_id="lobbying.us.filing_count",
                timestamp=ts,
                value=_safe_float(total_count),
                direction=0,
                confidence=0.5,
                category="behavioral_intent",
                tags=("lobbying", "volume"),
                ttl=2_592_000,
            )
        )

    return results


register_extractor("lobbying", _extract_lobbying)


# ── 23. Wikipedia Pageviews ───────────────────────────────────


def _extract_wikipedia_pageviews(tool_name: str, data: Any) -> list[Evidence]:
    """Extract Wikipedia spike signals — attention anomalies.

    Key signals: z-score spikes on watchlist articles.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    # Spike mode — the primary signal
    spikes = data.get("spikes")
    if isinstance(spikes, list):
        for s in spikes:
            if not isinstance(s, dict):
                continue
            article = s.get("article", "")
            z = _safe_float(s.get("z_score"))
            if not article or z < 2.0:
                continue
            slug = article.lower().replace(" ", "_")[:40]
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id=f"wiki.{slug}.spike_zscore",
                    timestamp=ts,
                    value=z,
                    direction=1,  # Attention spike = something happening
                    confidence=min(0.3 + z * 0.1, 0.9),  # Scale confidence with z
                    category="behavioral_intent",
                    tags=("wikipedia", slug, "attention"),
                    ttl=86_400,
                )
            )

    # Series mode — time series stats
    stats = data.get("stats")
    if isinstance(stats, dict):
        article = data.get("article", "unknown")
        slug = article.lower().replace(" ", "_")[:40]
        mean_views = _safe_float(stats.get("mean"))
        std_views = _safe_float(stats.get("std"))
        if mean_views > 0 and std_views > 0:
            latest_views = _safe_float(stats.get("max"))  # Approximate
            z = (latest_views - mean_views) / std_views if std_views > 1e-10 else 0
            if abs(z) > 2.0:
                results.append(
                    Evidence(
                        source=tool_name,
                        signal_id=f"wiki.{slug}.views_zscore",
                        timestamp=ts,
                        value=z,
                        direction=1 if z > 0 else -1,
                        confidence=0.5,
                        category="behavioral_intent",
                        tags=("wikipedia", slug),
                        ttl=86_400,
                    )
                )

    return results


register_extractor("wikipedia_pageviews", _extract_wikipedia_pageviews)


# ── 24. Cert Transparency ────────────────────────────────────


def _extract_cert_transparency(tool_name: str, data: Any) -> list[Evidence]:
    """Extract certificate issuance volume and subdomain activity.

    Key signal: unusual cert volume, new subdomain count.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()
    domain = data.get("domain", "unknown")
    slug = domain.replace(".", "_")[:30]

    # Search/recent mode — cert count
    cert_count = data.get("count")
    if cert_count is not None:
        results.append(
            Evidence(
                source=tool_name,
                signal_id=f"cert.{slug}.count",
                timestamp=ts,
                value=_safe_float(cert_count),
                direction=0,  # Volume — anomaly detection handles
                confidence=0.5,
                category="behavioral_intent",
                tags=("cert_transparency", slug),
                ttl=86_400,
            )
        )

    # Active/expired ratio
    active = data.get("active")
    expired = data.get("expired")
    if active is not None and expired is not None:
        total = _safe_int(active) + _safe_int(expired)
        if total > 0:
            active_ratio = _safe_int(active) / total
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id=f"cert.{slug}.active_ratio",
                    timestamp=ts,
                    value=active_ratio,
                    direction=0,
                    confidence=0.4,
                    category="behavioral_intent",
                    tags=("cert_transparency", slug, "active"),
                    ttl=86_400,
                )
            )

    # Subdomains mode
    subdomain_count = data.get("count")
    subdomains = data.get("subdomains")
    if isinstance(subdomains, list):
        results.append(
            Evidence(
                source=tool_name,
                signal_id=f"cert.{slug}.subdomain_count",
                timestamp=ts,
                value=float(len(subdomains)),
                direction=0,
                confidence=0.4,
                category="behavioral_intent",
                tags=("cert_transparency", slug, "subdomains"),
                ttl=86_400,
            )
        )

    return results


register_extractor("cert_transparency", _extract_cert_transparency)


# ── 25. DNS Monitor ──────────────────────────────────────────


def _extract_dns_monitor(tool_name: str, data: Any) -> list[Evidence]:
    """Extract DNS change signals — infrastructure changes.

    Key signal: change detection count, record changes.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    # Diff mode — change detection
    changes = data.get("changes")
    domain = data.get("domain", "unknown")
    slug = domain.replace(".", "_")[:30]

    if isinstance(changes, list) and changes:
        results.append(
            Evidence(
                source=tool_name,
                signal_id=f"dns.{slug}.change_count",
                timestamp=ts,
                value=float(len(changes)),
                direction=1,  # DNS changes = infrastructure shift
                confidence=0.6,
                category="behavioral_intent",
                tags=("dns", slug, "changes"),
                ttl=86_400,
            )
        )

    # Bulk resolve — total record count as infrastructure footprint
    total_records = data.get("total_records")
    domain_count = data.get("domain_count")
    if total_records is not None and domain_count is not None:
        results.append(
            Evidence(
                source=tool_name,
                signal_id="dns.bulk.total_records",
                timestamp=ts,
                value=_safe_float(total_records),
                direction=0,
                confidence=0.4,
                category="behavioral_intent",
                tags=("dns", "bulk"),
                ttl=86_400,
            )
        )

    return results


register_extractor("dns_monitor", _extract_dns_monitor)


# ── 26. Polymarket ────────────────────────────────────────────


def _extract_polymarket(tool_name: str, data: Any) -> list[Evidence]:
    """Extract prediction market signals — crowd probability estimates.

    Key signal: high-volume markets with extreme probabilities or large moves.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    markets = data.get("markets")
    if not isinstance(markets, list):
        return []

    for m in markets:
        if not isinstance(m, dict):
            continue
        slug_raw = m.get("slug") or m.get("question", "")
        yes_price = _safe_float(m.get("yes_price"))
        volume = _safe_float(m.get("volume_24h"))
        change_24h = _safe_float(m.get("price_change_24h"))
        if not slug_raw or yes_price <= 0:
            continue

        slug = slug_raw[:40].lower().replace(" ", "_").replace("-", "_")

        # Only emit for markets with meaningful volume
        if volume < 1000:
            continue

        # Extreme probability (>0.85 or <0.15) = high conviction
        if yes_price > 0.85 or yes_price < 0.15:
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id=f"polymarket.{slug}.probability",
                    timestamp=ts,
                    value=yes_price,
                    direction=1 if yes_price > 0.85 else -1,
                    confidence=min(0.5 + volume / 100_000, 0.9),
                    category="positioning",
                    tags=("prediction_market", slug),
                    ttl=43_200,  # 12h
                )
            )

        # Large 24h move
        if abs(change_24h) > 0.10:
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id=f"polymarket.{slug}.price_change_24h",
                    timestamp=ts,
                    value=change_24h,
                    direction=1 if change_24h > 0 else -1,
                    confidence=min(0.4 + volume / 100_000, 0.85),
                    category="positioning",
                    tags=("prediction_market", slug, "momentum"),
                    ttl=43_200,
                )
            )

    return results


register_extractor("polymarket", _extract_polymarket)


# ── 27. Polymarket Whales ────────────────────────────────────


def _extract_polymarket_whales(tool_name: str, data: Any) -> list[Evidence]:
    """Extract whale wallet signals — smart money positioning.

    Key signal: top wallet composite scores, market whale concentration.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    # Top wallets mode
    wallets = data.get("wallets")
    if isinstance(wallets, list) and wallets:
        # Average composite score of top whales
        composites = [
            _safe_float(w.get("composite")) for w in wallets if isinstance(w, dict)
        ]
        composites = [c for c in composites if c > 0]
        if composites:
            avg_composite = sum(composites) / len(composites)
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="polymarket_whales.avg_composite",
                    timestamp=ts,
                    value=avg_composite,
                    direction=0,
                    confidence=0.6,
                    category="positioning",
                    tags=("prediction_market", "whales"),
                    ttl=86_400,
                )
            )

    # Market whales mode — concentration
    whales = data.get("whales")
    if isinstance(whales, list) and whales:
        total_usdc = sum(
            _safe_float(w.get("total_usdc")) for w in whales if isinstance(w, dict)
        )
        results.append(
            Evidence(
                source=tool_name,
                signal_id="polymarket_whales.market_concentration",
                timestamp=ts,
                value=total_usdc,
                direction=0,
                confidence=0.55,
                category="positioning",
                tags=("prediction_market", "whales", "concentration"),
                ttl=86_400,
            )
        )

    return results


register_extractor("polymarket_whales", _extract_polymarket_whales)


# ── 28. Insider Filings ──────────────────────────────────────


def _extract_insider_filings(tool_name: str, data: Any) -> list[Evidence]:
    """Extract insider buying clusters — conviction signals.

    Key signal: cluster count, total value, insider count per cluster.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    clusters = data.get("clusters")
    if not isinstance(clusters, list):
        return []

    total_purchases = _safe_int(data.get("total_purchases"))
    if total_purchases > 0:
        results.append(
            Evidence(
                source=tool_name,
                signal_id="insider.market.purchase_count",
                timestamp=ts,
                value=float(total_purchases),
                direction=-1,  # Insider buying = bullish = relief
                confidence=0.8,
                category="positioning",
                tags=("insider", "purchases"),
                ttl=604_800,
            )
        )

    for c in clusters:
        if not isinstance(c, dict):
            continue
        ticker = c.get("ticker", "")
        if not ticker:
            continue
        slug = ticker.lower()
        insider_count = _safe_int(c.get("insider_count"))
        total_val = _safe_float(c.get("total_value"))
        conviction = _safe_float(c.get("conviction"))

        if insider_count >= 2:  # Cluster = 2+ insiders
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id=f"insider.{slug}.cluster",
                    timestamp=ts,
                    value=total_val,
                    direction=-1,  # Insider cluster buying = bullish conviction
                    confidence=(
                        min(0.5 + conviction * 0.4, 0.95) if conviction > 0 else 0.7
                    ),
                    category="positioning",
                    tags=("insider", slug, "cluster"),
                    ttl=604_800,
                )
            )

    return results


register_extractor("insider_filings", _extract_insider_filings)


# ── 29. Form 144 ─────────────────────────────────────────────


def _extract_form144(tool_name: str, data: Any) -> list[Evidence]:
    """Extract Form 144 selling intent clusters.

    Key signal: cluster count, urgency level, total value planned for sale.
    Form 144 = intent to sell restricted stock. Cluster = multiple insiders filing.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    clusters = data.get("clusters")
    if not isinstance(clusters, list):
        return []

    total_filings = _safe_int(data.get("total_filings"))
    if total_filings > 0:
        results.append(
            Evidence(
                source=tool_name,
                signal_id="form144.market.filing_count",
                timestamp=ts,
                value=float(total_filings),
                direction=1,  # Filing to sell = bearish intent
                confidence=0.7,
                category="positioning",
                tags=("form144", "sell_intent"),
                ttl=604_800,
            )
        )

    for c in clusters:
        if not isinstance(c, dict):
            continue
        ticker = c.get("ticker", "")
        if not ticker:
            continue
        slug = ticker.lower()
        urgency = _safe_float(c.get("urgency"))
        total_val = _safe_float(c.get("total_value"))
        pct_outstanding = _safe_float(c.get("pct_of_outstanding"))

        results.append(
            Evidence(
                source=tool_name,
                signal_id=f"form144.{slug}.sell_cluster",
                timestamp=ts,
                value=total_val,
                direction=1,  # Selling intent = bearish
                confidence=min(0.5 + urgency * 0.3, 0.9) if urgency > 0 else 0.6,
                category="positioning",
                tags=("form144", slug, "sell_intent"),
                ttl=604_800,
            )
        )

        if pct_outstanding > 1.0:  # >1% of float = significant
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id=f"form144.{slug}.pct_outstanding",
                    timestamp=ts,
                    value=pct_outstanding,
                    direction=1,
                    confidence=0.8,
                    category="positioning",
                    tags=("form144", slug, "significant"),
                    ttl=604_800,
                )
            )

    return results


register_extractor("form144", _extract_form144)


# ── 30. GDELT ─────────────────────────────────────────────────


def _extract_gdelt(tool_name: str, data: Any) -> list[Evidence]:
    """Extract geopolitical event signals from GDELT.

    Key signals: Goldstein scale, event volume, quad class distribution.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    events = data.get("events")
    if not isinstance(events, list) or not events:
        return []

    # Aggregate Goldstein scores — negative = conflict, positive = cooperation
    goldsteins = [
        _safe_float(e.get("goldstein"))
        for e in events
        if isinstance(e, dict) and e.get("goldstein") is not None
    ]
    if goldsteins:
        avg_goldstein = sum(goldsteins) / len(goldsteins)
        results.append(
            Evidence(
                source=tool_name,
                signal_id="gdelt.global.avg_goldstein",
                timestamp=ts,
                value=avg_goldstein,
                direction=1 if avg_goldstein < -3 else (-1 if avg_goldstein > 3 else 0),
                confidence=0.6,
                category="geopolitical",
                tags=("gdelt", "goldstein"),
                ttl=86_400,
            )
        )

    # Event volume
    results.append(
        Evidence(
            source=tool_name,
            signal_id="gdelt.global.event_count",
            timestamp=ts,
            value=float(len(events)),
            direction=0,
            confidence=0.5,
            category="geopolitical",
            tags=("gdelt", "volume"),
            ttl=86_400,
        )
    )

    # Quad class distribution — material conflict ratio
    quad_counts: dict[str, int] = {}
    for e in events:
        if isinstance(e, dict):
            ql = e.get("quad_label", "")
            if ql:
                quad_counts[ql] = quad_counts.get(ql, 0) + 1

    material_conflict = quad_counts.get("Material Conflict", 0)
    total_events = len(events)
    if total_events > 0:
        conflict_ratio = material_conflict / total_events
        results.append(
            Evidence(
                source=tool_name,
                signal_id="gdelt.global.material_conflict_ratio",
                timestamp=ts,
                value=conflict_ratio,
                direction=1 if conflict_ratio > 0.3 else 0,
                confidence=0.65,
                category="geopolitical",
                tags=("gdelt", "conflict"),
                ttl=86_400,
            )
        )

    # Summary stats if available
    summary = data.get("summary")
    if isinstance(summary, dict):
        mentions = _safe_float(
            summary.get("total_mentions") or summary.get("num_mentions")
        )
        if mentions > 0:
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="gdelt.global.total_mentions",
                    timestamp=ts,
                    value=mentions,
                    direction=0,
                    confidence=0.5,
                    category="geopolitical",
                    tags=("gdelt", "mentions"),
                    ttl=86_400,
                )
            )

    return results


register_extractor("gdelt", _extract_gdelt)


# ── 31. Whale Alert (Crypto) ─────────────────────────────────


def _extract_whale_alert(tool_name: str, data: Any) -> list[Evidence]:
    """Extract large crypto transaction signals.

    Key signal: transaction count and total BTC volume.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    txns = data.get("transactions")
    if not isinstance(txns, list):
        return []

    count = len(txns)
    if count == 0:
        return []

    results.append(
        Evidence(
            source=tool_name,
            signal_id="crypto.btc.large_tx_count",
            timestamp=ts,
            value=float(count),
            direction=0,  # Volume signal
            confidence=0.6,
            category="financial_stress",
            tags=("crypto", "btc", "whale"),
            ttl=43_200,  # 12h
        )
    )

    summary = data.get("summary")
    if isinstance(summary, dict):
        total_btc = _safe_float(summary.get("total_btc") or summary.get("total_value"))
        if total_btc > 0:
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="crypto.btc.whale_volume",
                    timestamp=ts,
                    value=total_btc,
                    direction=0,
                    confidence=0.55,
                    category="financial_stress",
                    tags=("crypto", "btc", "volume"),
                    ttl=43_200,
                )
            )

    return results


register_extractor("whale_alert", _extract_whale_alert)


# ── 32. Comtrade ──────────────────────────────────────────────


def _extract_comtrade(tool_name: str, data: Any) -> list[Evidence]:
    """Extract international trade flow signals from UN Comtrade.

    Key signals: trade value, bilateral flow shifts.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    records = data.get("records")
    if not isinstance(records, list) or not records:
        return []

    # Aggregate trade value
    total_value = sum(
        _safe_float(r.get("trade_value_usd")) for r in records if isinstance(r, dict)
    )
    record_count = data.get("record_count", len(records))
    flow = data.get("flow", "trade")

    reporter = data.get("reporter", "")
    partner = data.get("partner", "")
    slug = (
        f"{reporter}_{partner}".lower().replace(" ", "_")[:30] if reporter else "global"
    )

    results.append(
        Evidence(
            source=tool_name,
            signal_id=f"comtrade.{slug}.{flow}_value_usd",
            timestamp=ts,
            value=total_value,
            direction=0,
            confidence=0.6,
            category="supply_chain",
            tags=("trade", slug, flow),
            ttl=2_592_000,  # Monthly
        )
    )

    # Commodity-specific signal
    commodity_name = data.get("commodity_name")
    commodity_code = data.get("commodity_code")
    if commodity_code:
        results.append(
            Evidence(
                source=tool_name,
                signal_id=f"comtrade.{commodity_code}.trade_volume",
                timestamp=ts,
                value=total_value,
                direction=0,
                confidence=0.6,
                category="supply_chain",
                tags=("trade", "commodity", str(commodity_code)),
                ttl=2_592_000,
            )
        )

    return results


register_extractor("comtrade", _extract_comtrade)


# ── 33. Energy Supply ────────────────────────────────────────


def _extract_energy_supply(tool_name: str, data: Any) -> list[Evidence]:
    """Extract EIA petroleum stocks, supply, and rig count signals.

    Key signals: inventory levels, week-over-week changes, rig count trend.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    # Petroleum stocks / supply mode — signals dict
    signals = data.get("signals")
    label = data.get("label", "")

    if isinstance(signals, dict):
        for name, sig in signals.items():
            if not isinstance(sig, dict):
                continue
            slug = name.lower().replace(" ", "_")[:30]

            # Extract week-over-week change if available
            wow = sig.get("wow_change") or sig.get("change")
            latest = sig.get("latest") or sig.get("latest_value")

            if latest is not None:
                val = _safe_float(latest)
                results.append(
                    Evidence(
                        source=tool_name,
                        signal_id=f"energy.{slug}.level",
                        timestamp=ts,
                        value=val,
                        direction=0,
                        confidence=0.75,
                        category="physical_flow",
                        tags=("energy", slug, label or "stocks"),
                        ttl=604_800,  # Weekly
                    )
                )

            if wow is not None:
                wow_val = _safe_float(wow)
                # Stock decline = supply tightening = +1
                results.append(
                    Evidence(
                        source=tool_name,
                        signal_id=f"energy.{slug}.wow_change",
                        timestamp=ts,
                        value=wow_val,
                        direction=1 if wow_val < 0 else (-1 if wow_val > 0 else 0),
                        confidence=0.75,
                        category="physical_flow",
                        tags=("energy", slug, "change"),
                        ttl=604_800,
                    )
                )

    # Rig count mode
    rig_records = data.get("records")
    rig_signals = data.get("signals") if data.get("count") is not None else None
    rig_count = data.get("count")
    if rig_count is not None and isinstance(rig_records, list):
        results.append(
            Evidence(
                source=tool_name,
                signal_id="energy.rig_count.total",
                timestamp=ts,
                value=_safe_float(rig_count),
                direction=0,
                confidence=0.7,
                category="physical_flow",
                tags=("energy", "rig_count"),
                ttl=2_592_000,  # Monthly
            )
        )

    return results


register_extractor("energy_supply", _extract_energy_supply)


# ── 34. Supply Chain Prices ──────────────────────────────────


def _extract_supply_chain_prices(tool_name: str, data: Any) -> list[Evidence]:
    """Extract BLS PPI and import price signals.

    Key signals: MoM price changes, cross-sector pressure.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()
    mode = data.get("mode", "")

    # Producer prices / import prices — signals dict
    signals = data.get("signals")
    if isinstance(signals, dict):
        for name, sig in signals.items():
            if not isinstance(sig, dict):
                continue
            slug = name.lower().replace(" ", "_")[:30]
            mom = sig.get("mom_change") or sig.get("mom_pct")
            latest = sig.get("latest") or sig.get("latest_value")

            if mom is not None:
                val = _safe_float(mom)
                # Rising prices = inflationary pressure = +1 stress
                results.append(
                    Evidence(
                        source=tool_name,
                        signal_id=f"supply_chain.{slug}.mom_pct",
                        timestamp=ts,
                        value=val,
                        direction=1 if val > 0.5 else (-1 if val < -0.5 else 0),
                        confidence=0.7,
                        category="supply_chain",
                        tags=("ppi" if "producer" in mode else "import_prices", slug),
                        ttl=2_592_000,  # Monthly
                    )
                )

    # Pressure index mode — composite signal
    pressure = data.get("pressure")
    if isinstance(pressure, dict):
        score = pressure.get("score") or pressure.get("pressure_score")
        if score is not None:
            val = _safe_float(score)
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="supply_chain.pressure_index",
                    timestamp=ts,
                    value=val,
                    direction=1 if val > 0.6 else (-1 if val < 0.3 else 0),
                    confidence=0.75,
                    category="supply_chain",
                    tags=("supply_chain", "pressure"),
                    ttl=2_592_000,
                )
            )

    return results


register_extractor("supply_chain_prices", _extract_supply_chain_prices)


# ── 35. Macro Data (FRED/ECB/World Bank) ─────────────────────


def _extract_macro_data(tool_name: str, data: Any) -> list[Evidence]:
    """Extract generic macro series signals.

    Data is dict[series_id → list[{date, value}]]. We extract the latest
    observation per series. Direction and category must be inferred from
    series ID — we use a conservative neutral direction.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    for series_id, observations in data.items():
        if not isinstance(observations, list) or not observations:
            continue
        # Latest observation is last in list
        latest = observations[-1] if isinstance(observations[-1], dict) else {}
        val_raw = latest.get("value")
        if val_raw is None:
            continue

        val = _safe_float(val_raw)
        slug = series_id.lower().replace(":", "_").replace("/", "_")[:30]

        results.append(
            Evidence(
                source=tool_name,
                signal_id=f"macro.{slug}.latest",
                timestamp=ts,
                value=val,
                direction=0,  # Generic — no way to know direction without context
                confidence=0.6,
                category="macro_momentum",
                tags=("macro", slug),
                ttl=2_592_000,
            )
        )

    return results


register_extractor("macro_data", _extract_macro_data)


# ── 36. Market Data (Yahoo Finance) ──────────────────────────


def _extract_market_data(tool_name: str, data: Any) -> list[Evidence]:
    """Extract price/volume signals from Yahoo Finance OHLCV data.

    Data is dict[ticker → list[{Date/Datetime, Open, High, Low, Close, Volume}]].
    We extract latest close and volume per ticker.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    for ticker, records in data.items():
        if not isinstance(records, list) or not records:
            continue
        latest = records[-1] if isinstance(records[-1], dict) else {}
        close = latest.get("Close")
        volume = latest.get("Volume")
        slug = ticker.lower().replace("^", "").replace("=", "")[:20]

        if close is not None:
            val = _safe_float(close)
            # Compute return if we have 2+ records
            prev_close = None
            if len(records) >= 2 and isinstance(records[-2], dict):
                prev_close = _safe_float(records[-2].get("Close"))

            if prev_close and prev_close > 0:
                ret = (val - prev_close) / prev_close
                results.append(
                    Evidence(
                        source=tool_name,
                        signal_id=f"market.{slug}.return",
                        timestamp=ts,
                        value=ret,
                        direction=1 if ret < -0.02 else (-1 if ret > 0.02 else 0),
                        confidence=0.7,
                        category="positioning",
                        tags=("market", slug),
                        ttl=86_400,
                    )
                )

        if volume is not None:
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id=f"market.{slug}.volume",
                    timestamp=ts,
                    value=_safe_float(volume),
                    direction=0,
                    confidence=0.5,
                    category="positioning",
                    tags=("market", slug, "volume"),
                    ttl=86_400,
                )
            )

    return results


register_extractor("market_data", _extract_market_data)


# ══════════════════════════════════════════════════════════════
#  OUTPUT-ONLY TOOLS — Stubs returning []
#
#  These tools return formatted text in ToolResult.output without
#  a structured data dict. Extractors registered for forward
#  compatibility; they become real when tools gain data= dicts.
# ══════════════════════════════════════════════════════════════


def _stub_extractor(tool_name: str, data: Any) -> list[Evidence]:
    """Placeholder for tools that don't yet return structured data."""
    return []


register_extractor("foia_requests", _stub_extractor)
register_extractor("interconnection_queue", _stub_extractor)
register_extractor("electricity_monitor", _stub_extractor)


# ══════════════════════════════════════════════════════════════
#  Satellite Activity (NASA FIRMS fire, MODIS NDVI, EONET events)
# ══════════════════════════════════════════════════════════════

# NDVI health classification → ordinal for Evidence.value
_NDVI_HEALTH_ORDINAL: dict[str, float] = {
    "water_or_barren": 0.0,
    "bare_soil": 1.0,
    "sparse": 2.0,
    "moderate": 3.0,
    "healthy": 4.0,
    "dense": 5.0,
}


def _extract_satellite_activity(tool_name: str, data: Any) -> list[Evidence]:
    """Extract fire, vegetation, and natural-event signals from satellite data.

    Modes handled:
      - fire: hotspot count, aggregate FRP, cluster count
            → category physical_disruption
      - vegetation: latest NDVI, anomaly %, health ordinal
            → category supply_chain (crop stress proxy)
      - events: active event count, per-category counts
            → category physical_disruption

    Data source: NASA FIRMS (fire), ORNL MODIS (vegetation), NASA EONET (events).
    """
    if not isinstance(data, dict):
        return []

    mode = data.get("mode", "")
    if mode == "fire":
        return _extract_satellite_fire(tool_name, data)
    if mode == "vegetation":
        return _extract_satellite_vegetation(tool_name, data)
    if mode == "events":
        return _extract_satellite_events(tool_name, data)
    return []


def _extract_satellite_fire(tool_name: str, data: dict) -> list[Evidence]:
    """Fire mode: hotspot count, aggregate FRP, peak FRP, cluster count.

    All signals are physical_disruption — fire activity near infrastructure
    or crop regions is a direct physical disruption indicator.

    Thresholds:
      - hotspot_count > 100  → direction=1 (elevated fire activity)
      - frp_total > 1000 MW  → direction=1 (significant fire energy)
      - frp_max > 100 MW     → direction=1 (intense single point)
      - cluster_count > 5    → direction=1 (widespread spatial distribution)

    TTL = 21600s (6 hours) — fire data is near-real-time, stales quickly.
    """
    results: list[Evidence] = []
    ts = _now_ts()
    ttl = 21_600  # 6 hours — NRT fire data

    hotspot_count = data.get("hotspot_count")
    if hotspot_count is None:
        return []
    hotspot_count = _safe_float(hotspot_count)

    results.append(
        Evidence(
            source=tool_name,
            signal_id="satellite.fire.hotspot_count",
            timestamp=ts,
            value=hotspot_count,
            direction=1 if hotspot_count > 100 else 0,
            confidence=0.8,
            category="physical_disruption",
            tags=("satellite", "fire", "hotspot", "count"),
            ttl=ttl,
        )
    )

    frp_total = _safe_float(data.get("frp_total"))
    results.append(
        Evidence(
            source=tool_name,
            signal_id="satellite.fire.frp_total",
            timestamp=ts,
            value=frp_total,
            direction=1 if frp_total > 1000.0 else 0,
            confidence=0.75,
            category="physical_disruption",
            tags=("satellite", "fire", "frp", "total"),
            ttl=ttl,
        )
    )

    frp_max = _safe_float(data.get("frp_max"))
    results.append(
        Evidence(
            source=tool_name,
            signal_id="satellite.fire.frp_max",
            timestamp=ts,
            value=frp_max,
            direction=1 if frp_max > 100.0 else 0,
            confidence=0.7,
            category="physical_disruption",
            tags=("satellite", "fire", "frp", "peak"),
            ttl=ttl,
        )
    )

    cluster_count = _safe_float(data.get("cluster_count"))
    results.append(
        Evidence(
            source=tool_name,
            signal_id="satellite.fire.cluster_count",
            timestamp=ts,
            value=cluster_count,
            direction=1 if cluster_count > 5 else 0,
            confidence=0.75,
            category="physical_disruption",
            tags=("satellite", "fire", "cluster", "spatial"),
            ttl=ttl,
        )
    )

    return results


def _extract_satellite_vegetation(tool_name: str, data: dict) -> list[Evidence]:
    """Vegetation mode: latest NDVI, anomaly %, health ordinal.

    All signals are supply_chain — vegetation health is a leading indicator
    for crop stress → food/commodity prices → supply chain pressure.

    NDVI ranges from approximately -0.2 (water/barren) to 1.0 (dense vegetation).
    The anomaly percentage measures latest vs historical mean:
      - Negative anomaly = vegetation decline = crop stress
      - Positive anomaly = above-average health

    Direction semantics:
      - ndvi_latest: direction=0 (neutral — absolute value, not directional)
      - anomaly_pct: direction=-1 if < -10% (stress), +1 if > +10% (boom)
      - health_class_ordinal: direction=0

    TTL = 604800s (7 days) — MODIS NDVI updates every 16 days;
    7-day TTL gives overlap without excessive staleness.
    """
    results: list[Evidence] = []
    ts = _now_ts()
    ttl = 604_800  # 7 days — MODIS 16-day cycle

    observation_count = data.get("observation_count", 0)
    if not isinstance(observation_count, (int, float)) or observation_count == 0:
        # No observations → can still emit if latest_ndvi is present
        # (e.g. single-observation case), but if truly empty, bail
        if data.get("latest_ndvi") is None and data.get("avg_ndvi") is None:
            return []

    latest_ndvi = _safe_float(data.get("latest_ndvi"))
    results.append(
        Evidence(
            source=tool_name,
            signal_id="satellite.vegetation.ndvi_latest",
            timestamp=ts,
            value=latest_ndvi,
            direction=0,
            confidence=0.85,
            category="supply_chain",
            tags=("satellite", "vegetation", "ndvi", "latest"),
            ttl=ttl,
        )
    )

    # Anomaly: safe against division by zero (tool already handles this —
    # if avg_ndvi==0, anomaly_pct is set to 0.0 in the tool)
    anomaly_pct = _safe_float(data.get("anomaly_pct"))
    anom_direction = 0
    if anomaly_pct < -10.0:
        anom_direction = -1  # crop stress
    elif anomaly_pct > 10.0:
        anom_direction = 1  # above-average growth

    results.append(
        Evidence(
            source=tool_name,
            signal_id="satellite.vegetation.anomaly_pct",
            timestamp=ts,
            value=anomaly_pct,
            direction=anom_direction,
            confidence=0.8,
            category="supply_chain",
            tags=("satellite", "vegetation", "ndvi", "anomaly"),
            ttl=ttl,
        )
    )

    # Health classification as ordinal (0=water/barren .. 5=dense)
    health_str = data.get("latest_health", "")
    health_ordinal = _NDVI_HEALTH_ORDINAL.get(
        str(health_str).lower(), 1.0  # default to bare_soil if unknown
    )
    results.append(
        Evidence(
            source=tool_name,
            signal_id="satellite.vegetation.health_class_ordinal",
            timestamp=ts,
            value=health_ordinal,
            direction=0,
            confidence=0.85,
            category="supply_chain",
            tags=("satellite", "vegetation", "ndvi", "health"),
            ttl=ttl,
        )
    )

    return results


def _extract_satellite_events(tool_name: str, data: dict) -> list[Evidence]:
    """Events mode: total active natural events, per-category counts.

    All signals are physical_disruption — natural events (wildfires, storms,
    volcanoes, earthquakes) are direct physical disruptions.

    Special per-category counts for highest-impact event types:
      - wildfires: supply chain and infrastructure damage
      - severeStorms: shipping/logistics disruption

    Direction: 1 if count exceeds threshold (elevated activity).
      - active_count > 20 → direction=1
      - wildfire_count > 5 → direction=1
      - severe_storm_count > 3 → direction=1

    TTL = 43200s (12 hours) — EONET updates roughly daily.
    """
    results: list[Evidence] = []
    ts = _now_ts()
    ttl = 43_200  # 12 hours

    event_count = data.get("event_count")
    if event_count is None:
        return []
    event_count = _safe_float(event_count)

    results.append(
        Evidence(
            source=tool_name,
            signal_id="satellite.events.active_count",
            timestamp=ts,
            value=event_count,
            direction=1 if event_count > 20 else 0,
            confidence=0.8,
            category="physical_disruption",
            tags=("satellite", "events", "eonet", "count"),
            ttl=ttl,
        )
    )

    cat_counts = data.get("category_counts", {})
    if not isinstance(cat_counts, dict):
        cat_counts = {}

    wildfire_count = _safe_float(cat_counts.get("wildfires"))
    results.append(
        Evidence(
            source=tool_name,
            signal_id="satellite.events.wildfire_count",
            timestamp=ts,
            value=wildfire_count,
            direction=1 if wildfire_count > 5 else 0,
            confidence=0.8,
            category="physical_disruption",
            tags=("satellite", "events", "wildfire", "count"),
            ttl=ttl,
        )
    )

    storm_count = _safe_float(cat_counts.get("severeStorms"))
    results.append(
        Evidence(
            source=tool_name,
            signal_id="satellite.events.severe_storm_count",
            timestamp=ts,
            value=storm_count,
            direction=1 if storm_count > 3 else 0,
            confidence=0.8,
            category="physical_disruption",
            tags=("satellite", "events", "storm", "count"),
            ttl=ttl,
        )
    )

    return results


register_extractor("satellite_activity", _extract_satellite_activity)


# ══════════════════════════════════════════════════════════════
#  Internet Infrastructure (IODA outages + OONI censorship)
# ══════════════════════════════════════════════════════════════


def _extract_internet_infrastructure(tool_name: str, data: Any) -> list[Evidence]:
    """Extract outage, censorship, and connectivity signals.

    Modes handled:
      - outages: country-level alert severity and event breadth
      - censorship: anomaly rate and trend from OONI measurements
      - signals: real-time gtr-norm connectivity level
      - incidents: ongoing censorship incident count and breadth
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()
    mode = data.get("mode", "")

    if mode == "outages":
        alerts = data.get("alerts")
        events = data.get("events")

        if isinstance(alerts, list):
            critical = [
                a
                for a in alerts
                if isinstance(a, dict) and a.get("level") == "critical"
            ]
            critical_countries = {a.get("country", "??") for a in critical}
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="internet.outage.critical_count",
                    timestamp=ts,
                    value=float(len(critical_countries)),
                    direction=1 if critical_countries else 0,
                    confidence=0.85,
                    category="physical_disruption",
                    tags=("internet", "outage", "critical"),
                    ttl=3600,
                )
            )

        if isinstance(events, list):
            scores = [
                _safe_float(e.get("score")) for e in events if isinstance(e, dict)
            ]
            event_countries = {
                e.get("country", "??") for e in events if isinstance(e, dict)
            }
            if scores:
                max_score = max(scores)
                results.append(
                    Evidence(
                        source=tool_name,
                        signal_id="internet.outage.event_max_score",
                        timestamp=ts,
                        value=max_score,
                        direction=1 if max_score > 50 else 0,
                        confidence=0.8,
                        category="physical_disruption",
                        tags=("internet", "outage", "severity"),
                        ttl=3600,
                    )
                )
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="internet.outage.event_breadth",
                    timestamp=ts,
                    value=float(len(event_countries)),
                    direction=1 if len(event_countries) > 2 else 0,
                    confidence=0.8,
                    category="physical_disruption",
                    tags=("internet", "outage", "breadth"),
                    ttl=3600,
                )
            )

    elif mode == "censorship":
        avg_rate = _safe_float(data.get("avg_rate"))
        max_rate = _safe_float(data.get("max_rate"))
        trend = data.get("trend", "stable")
        rows = data.get("rows")

        results.append(
            Evidence(
                source=tool_name,
                signal_id="internet.censorship.anomaly_rate",
                timestamp=ts,
                value=avg_rate,
                direction=1 if avg_rate > 0.1 else 0,
                confidence=0.75,
                category="geopolitical",
                tags=("internet", "censorship", "anomaly_rate"),
                ttl=86_400,
            )
        )
        results.append(
            Evidence(
                source=tool_name,
                signal_id="internet.censorship.trend_rising",
                timestamp=ts,
                value=1.0 if trend == "rising" else 0.0,
                direction=1 if trend == "rising" else (-1 if trend == "falling" else 0),
                confidence=0.7,
                category="geopolitical",
                tags=("internet", "censorship", "trend"),
                ttl=86_400,
            )
        )
        # Total confirmed blocks across all rows
        if isinstance(rows, list):
            confirmed_total = sum(
                r.get("confirmed", 0) for r in rows if isinstance(r, dict)
            )
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="internet.censorship.confirmed_total",
                    timestamp=ts,
                    value=float(confirmed_total),
                    direction=1 if confirmed_total > 0 else 0,
                    confidence=0.8,
                    category="geopolitical",
                    tags=("internet", "censorship", "confirmed"),
                    ttl=86_400,
                )
            )

    elif mode == "signals":
        current = _safe_float(data.get("current"), 1.0)
        severity = data.get("severity", "normal")
        drops = data.get("drops")

        # Connectivity level: lower = worse (inverted signal)
        results.append(
            Evidence(
                source=tool_name,
                signal_id="internet.signals.connectivity_level",
                timestamp=ts,
                value=current,
                direction=(
                    -1
                    if severity == "critical"
                    else (-1 if severity == "warning" else 0)
                ),
                confidence=0.85,
                category="physical_disruption",
                tags=("internet", "connectivity", severity),
                ttl=1800,
            )
        )
        if isinstance(drops, list):
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="internet.signals.drop_count",
                    timestamp=ts,
                    value=float(len(drops)),
                    direction=1 if len(drops) > 3 else 0,
                    confidence=0.8,
                    category="physical_disruption",
                    tags=("internet", "connectivity", "drops"),
                    ttl=1800,
                )
            )

    elif mode == "incidents":
        incidents = data.get("incidents")
        country_freq = data.get("country_frequency")

        if isinstance(incidents, list):
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="internet.incidents.active_count",
                    timestamp=ts,
                    value=float(len(incidents)),
                    direction=1 if len(incidents) > 3 else 0,
                    confidence=0.75,
                    category="geopolitical",
                    tags=("internet", "censorship", "incidents"),
                    ttl=3600,
                )
            )
            # Country breadth from incidents
            all_countries: set[str] = set()
            for inc in incidents:
                if isinstance(inc, dict):
                    ccs = inc.get("countries", [])
                    if isinstance(ccs, list):
                        all_countries.update(ccs)
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="internet.incidents.country_breadth",
                    timestamp=ts,
                    value=float(len(all_countries)),
                    direction=1 if len(all_countries) > 5 else 0,
                    confidence=0.75,
                    category="geopolitical",
                    tags=("internet", "censorship", "breadth"),
                    ttl=3600,
                )
            )

    return results


register_extractor("internet_infrastructure", _extract_internet_infrastructure)


# ══════════════════════════════════════════════════════════════
#  39. Consumer Sentiment (Eurostat + FRED UMichigan + BLS CPI)
# ══════════════════════════════════════════════════════════════


def _extract_consumer_sentiment(tool_name: str, data: Any) -> list[Evidence]:
    """Extract consumer confidence signals from consumer_sentiment tool.

    Handles three modes:
    - eu_confidence: per-country balance % → direction from MoM trend
    - us_sentiment: UMichigan headline + inflation expectations
    - inflation_reality: CPI actual + expectations gap
    """
    if not isinstance(data, dict):
        return []

    mode = data.get("mode", "")
    results: list[Evidence] = []

    if mode == "eu_confidence":
        signals = data.get("signals") or {}
        countries = signals.get("countries") or {}
        for geo, csig in countries.items():
            if not isinstance(csig, dict):
                continue
            latest = csig.get("latest")
            if latest is None:
                continue
            mom = csig.get("mom_change")
            if mom is not None:
                direction = 1 if mom > 0 else (-1 if mom < 0 else 0)
            else:
                direction = 0
            slug = str(geo).lower()
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id=f"consumer_sentiment.eu.{slug}",
                    timestamp=_now_ts(),
                    value=_safe_float(latest),
                    direction=direction,
                    confidence=0.8,
                    category="macro_momentum",
                    tags=("consumer_confidence", "eu", slug),
                    ttl=2_592_000,  # 30 days — monthly data
                )
            )
        # Synchronized decline → high-signal event
        if signals.get("synchronized_decline"):
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="consumer_sentiment.eu.synchronized_decline",
                    timestamp=_now_ts(),
                    value=1.0,
                    direction=-1,
                    confidence=0.9,
                    category="macro_momentum",
                    tags=("consumer_confidence", "eu", "synchronized"),
                    ttl=2_592_000,
                )
            )

    elif mode == "us_sentiment":
        signals = data.get("signals") or {}
        sent = signals.get("sentiment_latest")
        if sent is not None:
            # UMich sentiment: higher = more optimistic
            # Direction based on MoM if available, else from level
            mom = signals.get("sentiment_mom_change")
            if mom is not None:
                direction = 1 if mom > 0 else (-1 if mom < 0 else 0)
            else:
                direction = 1 if sent >= 70 else (-1 if sent <= 50 else 0)
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="consumer_sentiment.us.headline",
                    timestamp=_now_ts(),
                    value=_safe_float(sent),
                    direction=direction,
                    confidence=0.85,
                    category="macro_momentum",
                    tags=("consumer_confidence", "us", "umichigan"),
                    ttl=2_592_000,
                )
            )
        inf_exp = signals.get("inflation_exp_latest")
        if inf_exp is not None:
            # Inflation expectations: >4% = unanchored = stress
            direction = 1 if _safe_float(inf_exp) > 3.0 else -1
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="consumer_sentiment.us.inflation_expectations",
                    timestamp=_now_ts(),
                    value=_safe_float(inf_exp),
                    direction=direction,
                    confidence=0.8,
                    category="macro_momentum",
                    tags=("inflation", "expectations", "us"),
                    ttl=2_592_000,
                )
            )

    elif mode == "inflation_reality":
        signals = data.get("signals") or {}
        cpi_mom = signals.get("cpi_mom_change")
        if cpi_mom is not None:
            direction = (
                1
                if _safe_float(cpi_mom) > 0
                else (-1 if _safe_float(cpi_mom) < 0 else 0)
            )
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="consumer_sentiment.cpi.mom",
                    timestamp=_now_ts(),
                    value=_safe_float(cpi_mom),
                    direction=direction,
                    confidence=0.9,
                    category="macro_momentum",
                    tags=("cpi", "inflation", "us"),
                    ttl=2_592_000,
                )
            )
        gap = signals.get("expectations_gap")
        if gap is not None:
            # Positive gap = expectations > reality = inflation fears
            direction = 1 if _safe_float(gap) > 0 else -1
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="consumer_sentiment.expectations_gap",
                    timestamp=_now_ts(),
                    value=_safe_float(gap),
                    direction=direction,
                    confidence=0.75,
                    category="macro_momentum",
                    tags=("inflation", "expectations_gap", "us"),
                    ttl=2_592_000,
                )
            )

    return results


register_extractor("consumer_sentiment", _extract_consumer_sentiment)


# ══════════════════════════════════════════════════════════════
#  40. Food Security (World Bank Agricultural Indicators)
# ══════════════════════════════════════════════════════════════


def _extract_food_security(tool_name: str, data: Any) -> list[Evidence]:
    """Extract food security signals from World Bank agricultural data.

    Records have: country, year, value, indicator, indicator_name.
    Signals have: yoy_change_pct, trend_direction, consecutive_years,
    stress_alert, deviation_from_avg_pct.
    """
    if not isinstance(data, dict):
        return []

    records = data.get("records")
    if not records or not isinstance(records, list):
        return []

    signals = data.get("signals") or {}
    country = str(data.get("country", "unknown")).lower()
    indicator = str(data.get("indicator", "")).lower()

    results: list[Evidence] = []

    # Use the most recent valid record value
    valid_records = [r for r in records if r.get("value") is not None]
    if not valid_records:
        return []

    latest = valid_records[-1]
    val = _safe_float(latest.get("value"))

    # Derive signal_id from indicator code
    if "prd.food" in indicator:
        sig_suffix = "production.food"
    elif "prd.crop" in indicator:
        sig_suffix = "production.crop"
    elif "prd.lvsk" in indicator:
        sig_suffix = "production.livestock"
    elif "yld.crel" in indicator:
        sig_suffix = "cereal_yield"
    elif "lnd.crel" in indicator:
        sig_suffix = "cereal_area"
    elif "tm.val.food" in indicator:
        sig_suffix = "food_import_pct"
    elif "tx.val.food" in indicator:
        sig_suffix = "food_export_pct"
    else:
        sig_suffix = indicator.replace(".", "_")[:30]

    # Direction from trend signals
    trend = signals.get("trend_direction", "")
    if trend == "down":
        direction = -1
    elif trend == "up":
        direction = 1
    else:
        direction = 0

    # Bump confidence if there's a stress alert
    confidence = 0.8
    has_stress = bool(signals.get("stress_alert"))
    if has_stress:
        confidence = 0.9

    results.append(
        Evidence(
            source=tool_name,
            signal_id=f"food_security.{country}.{sig_suffix}",
            timestamp=_now_ts(),
            value=val,
            direction=direction,
            confidence=confidence,
            category="biological",
            tags=("food_security", country, sig_suffix),
            ttl=2_592_000,  # 30 days — annual data
        )
    )

    # Emit a stress event if flagged
    if has_stress:
        results.append(
            Evidence(
                source=tool_name,
                signal_id=f"food_security.{country}.stress",
                timestamp=_now_ts(),
                value=1.0,
                direction=-1,
                confidence=0.9,
                category="biological",
                tags=("food_security", country, "stress_alert"),
                ttl=2_592_000,
            )
        )

    return results


register_extractor("food_security", _extract_food_security)


# ══════════════════════════════════════════════════════════════
#  41. Political Risk (FEC Campaign Finance)
# ══════════════════════════════════════════════════════════════


def _extract_political_risk(tool_name: str, data: Any) -> list[Evidence]:
    """Extract political risk signals from FEC campaign finance data.

    Handles three result_types: candidates, filings, expenditures.
    """
    if not isinstance(data, dict):
        return []

    result_type = data.get("result_type", "")
    records = data.get("records")
    signals = data.get("signals") or {}
    if not records or not isinstance(records, list):
        return []

    results: list[Evidence] = []

    if result_type == "expenditures":
        # Independent expenditures are the highest-signal political risk indicator
        support = _safe_float(signals.get("support_total", 0))
        oppose = _safe_float(signals.get("oppose_total", 0))
        total_spend = support + oppose

        if total_spend > 0:
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="political_risk.ie_total_spend",
                    timestamp=_now_ts(),
                    value=total_spend,
                    direction=1,  # more spending = more uncertainty
                    confidence=0.8,
                    category="geopolitical",
                    tags=("political_risk", "expenditures", "fec"),
                    ttl=604_800,  # 7 days
                )
            )

        oppose_ratio = _safe_float(signals.get("oppose_ratio", 0))
        if oppose_ratio > 0:
            # High oppose ratio → negative sentiment accelerating
            direction = 1 if oppose_ratio > 0.5 else -1
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="political_risk.oppose_ratio",
                    timestamp=_now_ts(),
                    value=oppose_ratio,
                    direction=direction,
                    confidence=0.75,
                    category="geopolitical",
                    tags=("political_risk", "oppose_ratio", "fec"),
                    ttl=604_800,
                )
            )

        # Per-candidate spending concentration
        top_targets = signals.get("top_targets") or []
        for target in top_targets[:3]:
            if not isinstance(target, dict):
                continue
            cand = str(target.get("candidate", "unknown"))
            spent = _safe_float(target.get("total_spent", 0))
            if spent <= 0:
                continue
            slug = cand.lower().replace(" ", "_").replace(",", "")[:30]
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id=f"political_risk.target.{slug}",
                    timestamp=_now_ts(),
                    value=spent,
                    direction=1,
                    confidence=0.7,
                    category="geopolitical",
                    tags=("political_risk", "target_spending", slug),
                    ttl=604_800,
                )
            )

    elif result_type == "filings":
        avg_cash = _safe_float(signals.get("avg_cash_on_hand", 0))
        if avg_cash > 0:
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="political_risk.avg_cash_on_hand",
                    timestamp=_now_ts(),
                    value=avg_cash,
                    direction=0,
                    confidence=0.7,
                    category="geopolitical",
                    tags=("political_risk", "filings", "cash"),
                    ttl=604_800,
                )
            )

    elif result_type == "candidates":
        fundraisers = _safe_int(signals.get("active_fundraisers", 0))
        if fundraisers > 0:
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="political_risk.active_fundraisers",
                    timestamp=_now_ts(),
                    value=float(fundraisers),
                    direction=1 if fundraisers > 5 else 0,
                    confidence=0.65,
                    category="geopolitical",
                    tags=("political_risk", "candidates", "fundraising"),
                    ttl=604_800,
                )
            )

    return results


register_extractor("political_risk", _extract_political_risk)


# ══════════════════════════════════════════════════════════════
#  Power Grid (NYISO demand, fuel mix, pricing, forecast)
# ══════════════════════════════════════════════════════════════

_RENEWABLE_FUELS = frozenset({"wind", "solar", "hydro", "other renewables"})


def _extract_power_grid(tool_name: str, data: Any) -> list[Evidence]:
    """Extract grid stress, fuel mix, and pricing spread signals.

    Modes handled:
      - demand: total peak MW and reporting zone count
      - fuel_mix: gas share and renewable share as % of total
      - pricing: stressed zone count, max DA-RT spread, avg DA price
      - forecast: persistent deviation zone count, max significant deviations
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()
    mode = data.get("mode", "")
    # PowerGridTool doesn't include a "mode" key in its data dict.
    # Infer mode from the keys present.
    if not mode:
        if "total_peak_mw" in data:
            mode = "demand"
        elif "fuels" in data:
            mode = "fuel_mix"
        elif "stressed_zones" in data:
            mode = "pricing"
        elif "persistent_deviation_zones" in data:
            mode = "forecast"

    if mode == "demand":
        total_peak = _safe_float(data.get("total_peak_mw"))
        zones = data.get("zones")
        zone_count = len(zones) if isinstance(zones, list) else 0

        results.append(
            Evidence(
                source=tool_name,
                signal_id="power_grid.demand.total_peak_mw",
                timestamp=ts,
                value=total_peak,
                direction=0,
                confidence=0.8,
                category="physical_flow",
                tags=("power_grid", "demand", "peak"),
                ttl=86_400,
            )
        )
        results.append(
            Evidence(
                source=tool_name,
                signal_id="power_grid.demand.zone_count",
                timestamp=ts,
                value=float(zone_count),
                direction=0,
                confidence=0.9,
                category="physical_flow",
                tags=("power_grid", "demand", "zones"),
                ttl=86_400,
            )
        )

    elif mode == "fuel_mix":
        fuels = data.get("fuels")
        total_mw = _safe_float(data.get("total_mw"))
        if isinstance(fuels, list) and total_mw > 0:
            gas_mw = sum(
                _safe_float(f.get("mw"))
                for f in fuels
                if isinstance(f, dict) and "gas" in str(f.get("fuel_type", "")).lower()
            )
            renewable_mw = sum(
                _safe_float(f.get("mw"))
                for f in fuels
                if isinstance(f, dict)
                and str(f.get("fuel_type", "")).lower() in _RENEWABLE_FUELS
            )
            gas_pct = (gas_mw / total_mw) * 100
            renewable_pct = (renewable_mw / total_mw) * 100

            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="power_grid.fuel.gas_share_pct",
                    timestamp=ts,
                    value=round(gas_pct, 1),
                    direction=1 if gas_pct > 50 else 0,
                    confidence=0.85,
                    category="physical_flow",
                    tags=("power_grid", "fuel_mix", "gas"),
                    ttl=86_400,
                )
            )
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="power_grid.fuel.renewable_share_pct",
                    timestamp=ts,
                    value=round(renewable_pct, 1),
                    direction=0,
                    confidence=0.85,
                    category="physical_flow",
                    tags=("power_grid", "fuel_mix", "renewable"),
                    ttl=86_400,
                )
            )

    elif mode == "pricing":
        zones = data.get("zones")
        stressed = data.get("stressed_zones")
        stressed_count = len(stressed) if isinstance(stressed, list) else 0

        results.append(
            Evidence(
                source=tool_name,
                signal_id="power_grid.pricing.stressed_zone_count",
                timestamp=ts,
                value=float(stressed_count),
                direction=1 if stressed_count > 0 else 0,
                confidence=0.8,
                category="financial_stress",
                tags=("power_grid", "pricing", "stress"),
                ttl=86_400,
            )
        )

        if isinstance(zones, list):
            spreads = [
                abs(_safe_float(z.get("spread")))
                for z in zones
                if isinstance(z, dict) and z.get("spread") is not None
            ]
            da_prices = [
                _safe_float(z.get("da_lbmp"))
                for z in zones
                if isinstance(z, dict) and z.get("da_lbmp") is not None
            ]
            if spreads:
                max_spread = max(spreads)
                results.append(
                    Evidence(
                        source=tool_name,
                        signal_id="power_grid.pricing.max_spread",
                        timestamp=ts,
                        value=max_spread,
                        direction=1 if max_spread > 5.0 else 0,
                        confidence=0.8,
                        category="financial_stress",
                        tags=("power_grid", "pricing", "spread"),
                        ttl=86_400,
                    )
                )
            if da_prices:
                avg_da = sum(da_prices) / len(da_prices)
                results.append(
                    Evidence(
                        source=tool_name,
                        signal_id="power_grid.pricing.avg_da_lbmp",
                        timestamp=ts,
                        value=round(avg_da, 2),
                        direction=1 if avg_da > 100 else 0,
                        confidence=0.8,
                        category="financial_stress",
                        tags=("power_grid", "pricing", "day_ahead"),
                        ttl=86_400,
                    )
                )

    elif mode == "forecast":
        zones = data.get("zones")
        persistent = data.get("persistent_deviation_zones")
        persistent_count = len(persistent) if isinstance(persistent, list) else 0

        results.append(
            Evidence(
                source=tool_name,
                signal_id="power_grid.forecast.persistent_deviation_count",
                timestamp=ts,
                value=float(persistent_count),
                direction=1 if persistent_count > 0 else 0,
                confidence=0.75,
                category="physical_disruption",
                tags=("power_grid", "forecast", "deviation"),
                ttl=86_400,
            )
        )

        if isinstance(zones, list):
            sig_devs = [
                z.get("significant_deviations", 0) for z in zones if isinstance(z, dict)
            ]
            if sig_devs:
                max_sig = max(sig_devs)
                results.append(
                    Evidence(
                        source=tool_name,
                        signal_id="power_grid.forecast.max_significant_deviations",
                        timestamp=ts,
                        value=float(max_sig),
                        direction=1 if max_sig > 5 else 0,
                        confidence=0.7,
                        category="physical_disruption",
                        tags=("power_grid", "forecast", "significant"),
                        ttl=86_400,
                    )
                )

    return results


register_extractor("power_grid", _extract_power_grid)


# ══════════════════════════════════════════════════════════════
#  DeFi Flows (DefiLlama TVL, stablecoins, DEX volume, chains)
# ══════════════════════════════════════════════════════════════


def _extract_defi_flows(tool_name: str, data: Any) -> list[Evidence]:
    """Extract DeFi liquidity, stablecoin supply, and DEX stress signals.

    Modes handled:
      - tvl: total TVL, drawdown breadth, top concentration
      - stablecoins: total supply, top stablecoin share
      - dex_volume: total 24h volume, panic breadth
      - chain: total chain TVL, top chain concentration
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()
    mode = data.get("mode", "")
    # DefiFlowsTool doesn't include "mode" either. Infer.
    if not mode:
        if "protocols" in data and "total_tvl" in data:
            mode = "tvl"
        elif "stablecoins" in data:
            mode = "stablecoins"
        elif "dexes" in data:
            mode = "dex_volume"
        elif "chains" in data and "grand_total_tvl" in data:
            mode = "chain"

    if mode == "tvl":
        total_tvl = _safe_float(data.get("total_tvl"))
        protocols = data.get("protocols")

        results.append(
            Evidence(
                source=tool_name,
                signal_id="defi.tvl.total_usd",
                timestamp=ts,
                value=total_tvl,
                direction=0,
                confidence=0.8,
                category="financial_stress",
                tags=("defi", "tvl", "total"),
                ttl=3600,
            )
        )

        if isinstance(protocols, list):
            # Drawdown breadth: protocols with 1d change < -5%
            drawdown_count = sum(
                1
                for p in protocols
                if isinstance(p, dict)
                and p.get("change_1d_pct") is not None
                and _safe_float(p.get("change_1d_pct")) < -5.0
            )
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="defi.tvl.drawdown_breadth",
                    timestamp=ts,
                    value=float(drawdown_count),
                    direction=1 if drawdown_count > 3 else 0,
                    confidence=0.75,
                    category="financial_stress",
                    tags=("defi", "tvl", "drawdown"),
                    ttl=3600,
                )
            )
            # Top concentration
            if protocols and total_tvl > 0:
                top_tvl = _safe_float(
                    protocols[0].get("tvl_usd") if isinstance(protocols[0], dict) else 0
                )
                top_pct = (top_tvl / total_tvl) * 100
                results.append(
                    Evidence(
                        source=tool_name,
                        signal_id="defi.tvl.top_concentration_pct",
                        timestamp=ts,
                        value=round(top_pct, 2),
                        direction=1 if top_pct > 30 else 0,
                        confidence=0.7,
                        category="positioning",
                        tags=("defi", "tvl", "concentration"),
                        ttl=3600,
                    )
                )

    elif mode == "stablecoins":
        total_supply = _safe_float(data.get("total_supply"))
        stablecoins = data.get("stablecoins")

        results.append(
            Evidence(
                source=tool_name,
                signal_id="defi.stablecoin.total_supply",
                timestamp=ts,
                value=total_supply,
                direction=0,
                confidence=0.85,
                category="financial_stress",
                tags=("defi", "stablecoin", "supply"),
                ttl=3600,
            )
        )

        if isinstance(stablecoins, list) and stablecoins and total_supply > 0:
            top = stablecoins[0]
            if isinstance(top, dict):
                top_circ = _safe_float(top.get("circulating_usd"))
                top_pct = (top_circ / total_supply) * 100
                results.append(
                    Evidence(
                        source=tool_name,
                        signal_id="defi.stablecoin.top_share_pct",
                        timestamp=ts,
                        value=round(top_pct, 2),
                        direction=1 if top_pct > 60 else 0,
                        confidence=0.8,
                        category="positioning",
                        tags=("defi", "stablecoin", "concentration"),
                        ttl=3600,
                    )
                )

    elif mode == "dex_volume":
        total_vol = _safe_float(data.get("total_volume_24h"))
        dexes = data.get("dexes")

        results.append(
            Evidence(
                source=tool_name,
                signal_id="defi.dex.total_volume_24h",
                timestamp=ts,
                value=total_vol,
                direction=0,
                confidence=0.75,
                category="positioning",
                tags=("defi", "dex", "volume"),
                ttl=3600,
            )
        )

        if isinstance(dexes, list):
            # Panic breadth: DEXes with 1d volume spike > +50%
            panic_count = sum(
                1
                for d in dexes
                if isinstance(d, dict)
                and d.get("change_1d_pct") is not None
                and _safe_float(d.get("change_1d_pct")) > 50.0
            )
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="defi.dex.panic_breadth",
                    timestamp=ts,
                    value=float(panic_count),
                    direction=1 if panic_count > 2 else 0,
                    confidence=0.7,
                    category="financial_stress",
                    tags=("defi", "dex", "panic"),
                    ttl=3600,
                )
            )

    elif mode == "chain":
        grand_total = _safe_float(data.get("grand_total_tvl"))
        chains = data.get("chains")

        results.append(
            Evidence(
                source=tool_name,
                signal_id="defi.chain.total_tvl",
                timestamp=ts,
                value=grand_total,
                direction=0,
                confidence=0.8,
                category="financial_stress",
                tags=("defi", "chain", "tvl"),
                ttl=3600,
            )
        )

        if isinstance(chains, list) and chains and grand_total > 0:
            top = chains[0]
            if isinstance(top, dict):
                top_tvl = _safe_float(top.get("tvl_usd"))
                top_pct = (top_tvl / grand_total) * 100
                results.append(
                    Evidence(
                        source=tool_name,
                        signal_id="defi.chain.top_concentration_pct",
                        timestamp=ts,
                        value=round(top_pct, 2),
                        direction=1 if top_pct > 50 else 0,
                        confidence=0.75,
                        category="positioning",
                        tags=("defi", "chain", "concentration"),
                        ttl=3600,
                    )
                )

    return results


register_extractor("defi_flows", _extract_defi_flows)


# ══════════════════════════════════════════════════════════════
#  EXTRACTOR IMPLEMENTATIONS — Batch 3 (uncovered tools)
# ══════════════════════════════════════════════════════════════


# ── 47. Labor Disruptions ─────────────────────────────────────


def _extract_labor_disruptions(tool_name: str, data: Any) -> list[Evidence]:
    """Extract strike/work-stoppage signals from BLS labor disruptions data.

    Handles two modes:
      - Overview: data["signals"] has nested "workers" and "idle_days" sub-dicts
        plus "intensity_ratio" and "consecutive_active_months".
      - Single-series: data["label"] == "workers" or "idle_days", with flat
        data["signals"] containing trend/latest_value directly.
    """
    if not isinstance(data, dict):
        return []

    signals = data.get("signals")
    if not isinstance(signals, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    def _trend_direction(trend: str | None) -> int:
        if trend in ("ESCALATING", "RISING", "NEW_ACTIVITY"):
            return 1
        if trend == "DECLINING":
            return -1
        return 0

    label = data.get("label")

    if label is not None:
        # ── Single-series mode ──
        raw_val = signals.get("latest_value")
        if raw_val is None:
            return results
        val = _safe_float(raw_val)
        trend = signals.get("trend")
        direction = _trend_direction(trend)

        if label == "workers":
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="strike.us.workers_involved",
                    timestamp=ts,
                    value=val,
                    direction=direction,
                    confidence=0.75,
                    category="behavioral_intent",
                    tags=("labor", "strike", "workers"),
                    ttl=2_592_000,
                )
            )
        elif label == "idle_days":
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="strike.us.idle_days",
                    timestamp=ts,
                    value=val,
                    direction=direction,
                    confidence=0.70,
                    category="macro_momentum",
                    tags=("labor", "strike", "idle_days"),
                    ttl=2_592_000,
                )
            )
    else:
        # ── Overview mode (nested sub-dicts) ──
        w_sub = signals.get("workers")
        if isinstance(w_sub, dict) and w_sub.get("latest_value") is not None:
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="strike.us.workers_involved",
                    timestamp=ts,
                    value=_safe_float(w_sub["latest_value"]),
                    direction=_trend_direction(w_sub.get("trend")),
                    confidence=0.75,
                    category="behavioral_intent",
                    tags=("labor", "strike", "workers"),
                    ttl=2_592_000,
                )
            )

        i_sub = signals.get("idle_days")
        if isinstance(i_sub, dict) and i_sub.get("latest_value") is not None:
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="strike.us.idle_days",
                    timestamp=ts,
                    value=_safe_float(i_sub["latest_value"]),
                    direction=_trend_direction(i_sub.get("trend")),
                    confidence=0.70,
                    category="macro_momentum",
                    tags=("labor", "strike", "idle_days"),
                    ttl=2_592_000,
                )
            )

        # Intensity ratio (overview only)
        raw_intensity = signals.get("intensity_ratio")
        if raw_intensity is not None:
            intensity = _safe_float(raw_intensity)
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="strike.us.intensity",
                    timestamp=ts,
                    value=intensity,
                    direction=1 if intensity > 1.5 else (-1 if intensity < 0.5 else 0),
                    confidence=0.65,
                    category="macro_momentum",
                    tags=("labor", "strike", "intensity"),
                    ttl=2_592_000,
                )
            )

        # Consecutive active months (overview only)
        consec = signals.get("consecutive_active_months")
        if isinstance(consec, (int, float)) and consec > 0:
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="strike.us.consecutive_months",
                    timestamp=ts,
                    value=float(consec),
                    direction=1 if consec >= 3 else 0,
                    confidence=0.70,
                    category="behavioral_intent",
                    tags=("labor", "strike", "persistence"),
                    ttl=2_592_000,
                )
            )

    return results


register_extractor("labor_disruptions", _extract_labor_disruptions)


# ── 48. Government Contracts ──────────────────────────────────


_GOV_DEFENSE_KEYWORDS = frozenset((
    "defense", "defence", "dod", "mod", "military", "army", "navy",
    "air force", "pentagon", "armed forces",
))


def _extract_gov_contracts(tool_name: str, data: Any) -> list[Evidence]:
    """Extract fiscal-intent signals from USASpending / UK Contracts Finder data.

    Signals: award count, total value, defense spending share.
    Region is 'us' (default) or 'uk' (from data["region"]).
    """
    if not isinstance(data, dict):
        return []

    awards = data.get("awards")
    if not isinstance(awards, list) or not awards:
        return []

    results: list[Evidence] = []
    ts = _now_ts()
    region = str(data.get("region", "us")).lower()

    # ── Award count ──
    count = data.get("count")
    if count is None:
        count = len(awards)
    count_val = _safe_float(count)
    if count_val is not None and count_val > 0:
        results.append(
            Evidence(
                source=tool_name,
                signal_id=f"gov_contract.{region}.award_count",
                timestamp=ts,
                value=count_val,
                direction=1,
                confidence=0.65,
                category="regulatory_action",
                tags=("gov", "contracts", region, "count"),
                ttl=21_600,
            )
        )

    # ── Total value ──
    # US awards use "amount_usd"; UK awards use "amount"
    total = 0.0
    for a in awards:
        if not isinstance(a, dict):
            continue
        amt = a.get("amount_usd") or a.get("amount")
        v = _safe_float(amt)
        if v is not None:
            total += v

    if total > 0:
        results.append(
            Evidence(
                source=tool_name,
                signal_id=f"gov_contract.{region}.total_value",
                timestamp=ts,
                value=round(total, 2),
                direction=1,
                confidence=0.70,
                category="regulatory_action",
                tags=("gov", "contracts", region, "value"),
                ttl=21_600,
            )
        )

    # ── Defense share ──
    defense_count = 0
    for a in awards:
        if not isinstance(a, dict):
            continue
        agency = str(a.get("agency") or "").lower()
        desc = str(a.get("description") or "").lower()
        combined = f"{agency} {desc}"
        if any(kw in combined for kw in _GOV_DEFENSE_KEYWORDS):
            defense_count += 1

    if len(awards) > 0:
        defense_share = defense_count / len(awards)
        results.append(
            Evidence(
                source=tool_name,
                signal_id=f"gov_contract.{region}.defense_share",
                timestamp=ts,
                value=round(defense_share, 4),
                direction=1 if defense_share > 0.3 else 0,
                confidence=0.75,
                category="geopolitical",
                tags=("gov", "contracts", region, "defense"),
                ttl=21_600,
            )
        )

    return results


register_extractor("gov_contracts", _extract_gov_contracts)


# ── 49. Academic Preprints ────────────────────────────────────

_TRIAL_ACTIVE_STATUSES = frozenset((
    "Recruiting",
    "Active, not recruiting",
    "Not yet recruiting",
    "Enrolling by invitation",
))


def _extract_academic_preprints(tool_name: str, data: Any) -> list[Evidence]:
    """Extract research frontier signals from arXiv / ClinicalTrials.gov data.

    Detect mode from keys: "trials" → clinical trials, "papers" → arXiv.
    """
    if not isinstance(data, dict):
        return []

    results: list[Evidence] = []
    ts = _now_ts()

    trials = data.get("trials")
    papers = data.get("papers")

    if isinstance(trials, list):
        # ── Clinical trials mode ──
        active_count = 0
        completed_count = 0
        industry_count = 0
        valid_sponsor = 0

        for s in trials:
            if not isinstance(s, dict):
                continue
            status = s.get("status") or ""
            if status in _TRIAL_ACTIVE_STATUSES:
                active_count += 1
            elif status == "Completed":
                completed_count += 1

            sc = s.get("sponsor_class")
            if sc is not None:
                valid_sponsor += 1
                if sc == "INDUSTRY":
                    industry_count += 1

        if active_count > 0:
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="trials.active_count",
                    timestamp=ts,
                    value=float(active_count),
                    direction=1,
                    confidence=0.60,
                    category="biological",
                    tags=("pharma", "trials", "active"),
                    ttl=86_400,
                )
            )

        if completed_count > 0:
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="trials.completed_count",
                    timestamp=ts,
                    value=float(completed_count),
                    direction=1,
                    confidence=0.75,
                    category="regulatory_action",
                    tags=("pharma", "trials", "completed"),
                    ttl=86_400,
                )
            )

        if valid_sponsor > 0:
            ratio = industry_count / valid_sponsor
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="trials.industry_ratio",
                    timestamp=ts,
                    value=round(ratio, 4),
                    direction=1 if ratio > 0.5 else 0,
                    confidence=0.60,
                    category="behavioral_intent",
                    tags=("pharma", "trials", "industry"),
                    ttl=86_400,
                )
            )

    elif isinstance(papers, list):
        # ── arXiv mode ──
        volume = data.get("count") or data.get("total") or len(papers)
        vol_val = _safe_float(volume)
        if vol_val is not None and vol_val > 0:
            results.append(
                Evidence(
                    source=tool_name,
                    signal_id="arxiv.volume",
                    timestamp=ts,
                    value=vol_val,
                    direction=1,
                    confidence=0.50,
                    category="behavioral_intent",
                    tags=("research", "arxiv", "volume"),
                    ttl=86_400,
                )
            )

    return results


register_extractor("academic_preprints", _extract_academic_preprints)
register_extractor("internet_outages", _stub_extractor)
register_extractor("migration_flows", _stub_extractor)
