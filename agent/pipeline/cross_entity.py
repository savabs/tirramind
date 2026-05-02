"""
TirraMind — Cross-Entity Pattern Detection (Phase 11b/11c/11d)

Discovers L3 signals by linking entities across data domains and
detecting temporal co-occurrences that reveal hidden structure.

Architecture:
  CrossEntityDetector takes a PipelineStore, uses entity_links to
  find cross-domain pairs, queries co-occurrences within configurable
  time windows, and stores significant patterns as depth_level=3
  observations.

Patterns:
  1. Insider Selling × GDELT Conflict (Phase 11b)
  2. Vessel Rerouting × Sanctions Escalation (Phase 11c)
  3. Whale Crypto Transfers × Geopolitical Events (Phase 11d)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from agent.pipeline.entity import entity_id_from_key

log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────

# Default co-occurrence window: 72 hours (accommodates SEC T+2 filing lag)
DEFAULT_WINDOW_SECONDS = 72 * 3600

# Goldstein score threshold for "negative" GDELT events.
# Ranges from -10 (most conflictual) to +10 (most cooperative).
# We treat anything <= -2 as meaningfully negative.
GOLDSTEIN_THRESHOLD = -2.0

# Vessel × Sanctions: tighter window (AIS is T0, no filing lag)
VESSEL_WINDOW_SECONDS = 48 * 3600

# CAMEO root codes indicating sanctions / coercion / trade restriction
SANCTIONS_ROOT_CODES: frozenset[str] = frozenset({"16", "17"})

# Whale × Geopolitical: tightest window (crypto is near-instant)
WHALE_WINDOW_SECONDS = 24 * 3600

# Stricter Goldstein threshold for whale patterns — only high-impact events
# trigger capital flight. Standard threshold (-2.0) is too permissive here.
WHALE_GOLDSTEIN_THRESHOLD = -5.0

# BTC value normalization for scoring. Transfers ≥ this amount get full
# value_weight (1.0). Smaller transfers are proportionally discounted.
WHALE_VALUE_SCALE = 100.0

# Known exchange BTC addresses → (exchange_name, country_fips).
# Empty by default; populate via seed_whale_country_links(exchange_wallets=...)
# or load from an external config (e.g. data/exchange_wallets.json).
# Production deployments should maintain this from blockchain analytics sources.
KNOWN_EXCHANGE_WALLETS: dict[str, tuple[str, str]] = {}

# ── ISO 3166 alpha-2 → FIPS 10-4 conversion ─────────────────
# Only countries where the codes differ (most are identical).
# Source: FIPS 10-4 standard (used by GDELT for actor country codes).

ISO_TO_FIPS: dict[str, str] = {
    "AT": "AU",  # Austria
    "AU": "AS",  # Australia
    "BD": "BG",  # Bangladesh
    "BJ": "BN",  # Benin
    "BN": "BX",  # Brunei
    "CH": "SZ",  # Switzerland
    "CZ": "EZ",  # Czechia
    "DE": "GM",  # Germany
    "DK": "DA",  # Denmark
    "EE": "EN",  # Estonia
    "GB": "UK",  # United Kingdom
    "HR": "HR",  # Croatia (same)
    "LT": "LH",  # Lithuania
    "LV": "LG",  # Latvia
    "MM": "BM",  # Myanmar
    "RO": "RO",  # Romania (same)
    "RU": "RS",  # Russia
    "SE": "SW",  # Sweden
    "TL": "TT",  # Timor-Leste
}

# ── Baltic/Northern European port names → FIPS country code ──
# Covers major ports in the Digitraffic AIS coverage area.
# Keys are UPPERCASE substrings matched against port names.

BALTIC_PORT_TO_FIPS: dict[str, str] = {
    # Finland (FI)
    "HELSINKI": "FI",
    "KOTKA": "FI",
    "HAMINA": "FI",
    "TURKU": "FI",
    "RAUMA": "FI",
    "PORI": "FI",
    "OULU": "FI",
    "VAASA": "FI",
    "NAANTALI": "FI",
    "KOKKOLA": "FI",
    "HANKO": "FI",
    "KEMI": "FI",
    # Sweden (SW in FIPS)
    "STOCKHOLM": "SW",
    "GOTHENBURG": "SW",
    "GOTEBORG": "SW",
    "LULEA": "SW",
    "MALMO": "SW",
    "NORRKOPING": "SW",
    "GAVLE": "SW",
    "SUNDSVALL": "SW",
    "KARLSHAMN": "SW",
    "OXELOSUND": "SW",
    # Estonia (EN in FIPS)
    "TALLINN": "EN",
    "MUUGA": "EN",
    "PALDISKI": "EN",
    "SILLAMAE": "EN",
    # Latvia (LG in FIPS)
    "RIGA": "LG",
    "VENTSPILS": "LG",
    "LIEPAJA": "LG",
    # Lithuania (LH in FIPS)
    "KLAIPEDA": "LH",
    # Poland (PL)
    "GDANSK": "PL",
    "GDYNIA": "PL",
    "SZCZECIN": "PL",
    "SWINOUJSCIE": "PL",
    # Denmark (DA in FIPS)
    "COPENHAGEN": "DA",
    "AARHUS": "DA",
    "FREDERICIA": "DA",
    # Germany (GM in FIPS)
    "HAMBURG": "GM",
    "ROSTOCK": "GM",
    "LUBECK": "GM",
    "BREMERHAVEN": "GM",
    "WISMAR": "GM",
    "KIEL": "GM",
    # Norway (NO)
    "OSLO": "NO",
    "BERGEN": "NO",
    "STAVANGER": "NO",
    # Russia (RS in FIPS)
    "ST PETERSBURG": "RS",
    "UST-LUGA": "RS",
    "UST LUGA": "RS",
    "PRIMORSK": "RS",
    "VYSOTSK": "RS",
    "KALININGRAD": "RS",
    "MURMANSK": "RS",
    "ARKHANGELSK": "RS",
}


def resolve_port_country(port_name: str | None) -> str | None:
    """Resolve a port name or UN LOCODE to a FIPS country code.

    Resolution order:
    1. Baltic port name lookup (case-insensitive substring) — curated, reliable.
    2. UN LOCODE prefix (first 2 chars if uppercase alpha) → ISO → FIPS — broad fallback.
    3. None if unresolvable.
    """
    if not port_name or not port_name.strip():
        return None

    name = port_name.strip().upper()

    # Strategy 1: Known Baltic port name lookup (most reliable)
    for port_key, fips in BALTIC_PORT_TO_FIPS.items():
        if port_key in name:
            return fips

    # Strategy 2: UN LOCODE-style prefix (e.g., "RU LED", "RULED", "NLRTM")
    # Must look like: 2 alpha letters + space or 3+ alpha-city code.
    # Only fire when prefix is a known ISO code (in ISO_TO_FIPS) or when
    # the format is clearly LOCODE (2 letters + space + 3 letters).
    if len(name) >= 5:
        prefix = name[:2]
        if prefix.isalpha():
            # Strong signal: "XX YYY" format (space-separated LOCODE)
            if name[2] == " " and len(name) >= 5 and name[3:6].replace(" ", "").isalpha():
                fips = ISO_TO_FIPS.get(prefix, prefix)
                if len(fips) == 2 and fips.isalpha():
                    return fips
            # Moderate signal: known ISO prefix in ISO_TO_FIPS table
            elif prefix in ISO_TO_FIPS:
                return ISO_TO_FIPS[prefix]

    return None


# ── link seeders ──────────────────────────────────────────────


def seed_company_country_links(
    store: Any,
    tickers_json_path: str | None = None,
) -> int:
    """Create ``headquartered_in`` links from company entities to country entities.

    Uses SEC company_tickers.json (all US-registered filers → country "US").
    The country entity is registered if it doesn't already exist.

    Args:
        store: PipelineStore instance.
        tickers_json_path: Path to company_tickers.json. If None, uses SEC EDGAR.

    Returns:
        Number of links created (new only; existing links are skipped).
    """
    from agent.pipeline.entity import _load_tickers_data

    data = _load_tickers_data(tickers_json_path)

    # US country entity (FIPS code "US")
    us_entity_id = entity_id_from_key("country", "US")
    store.register_entity("country", "United States", us_entity_id, metadata={"fips": "US"})
    store.add_entity_alias(us_entity_id, "fips_country", "US")

    created = 0
    for entry in data.values():
        cik = str(entry.get("cik_str", ""))
        title = str(entry.get("title", ""))
        if not cik or not title:
            continue

        company_eid = entity_id_from_key("company", cik)
        # Only link if the company entity is already registered
        existing = store.get_entity(company_eid)
        if existing is None:
            continue

        link_id = store.link_entities(
            entity_id_a=company_eid,
            entity_id_b=us_entity_id,
            link_type="headquartered_in",
            source="sec_tickers",
            confidence=1.0,
            metadata={"cik": cik},
        )
        if link_id is not None:
            created += 1

    log.info("Seeded %d company→country links (headquartered_in US)", created)
    return created


def seed_vessel_country_links(store: Any) -> int:
    """Create ``port_call_to`` links from vessel entities to country entities.

    Scans existing ``port_call`` observations for all vessel entities and
    resolves port names (port, prev_port, next_port) to FIPS country codes.

    Args:
        store: PipelineStore instance.

    Returns:
        Number of new links created.
    """
    conn = store._get_conn()
    # Find all vessel entities that have port_call observations
    rows = conn.execute(
        "SELECT DISTINCT e.entity_id "
        "FROM entities e "
        "JOIN entity_observations o ON e.entity_id = o.entity_id "
        "WHERE e.entity_type='vessel' AND o.observation_type='port_call'"
    ).fetchall()

    created = 0
    for row in rows:
        vessel_eid = row["entity_id"]
        obs_list = store.query_entity_observations(vessel_eid, source_tool="ais_vessel", limit=500)

        # Collect unique country FIPS codes from port names
        countries_seen: set[str] = set()
        for obs in obs_list:
            val = obs.get("value", {})
            for field in ("port", "prev_port", "next_port"):
                port_name = val.get(field)
                fips = resolve_port_country(port_name)
                if fips and fips not in countries_seen:
                    countries_seen.add(fips)

        # Create links for each resolved country
        for fips in countries_seen:
            country_eid = entity_id_from_key("country", fips)
            # Register country entity if it doesn't exist
            existing = store.get_entity(country_eid)
            if existing is None:
                store.register_entity(
                    "country",
                    fips,
                    country_eid,
                    metadata={"fips": fips},
                )
                store.add_entity_alias(country_eid, "fips_country", fips)

            # Avoid self-link (would be a bug, but be safe)
            if vessel_eid == country_eid:
                continue

            link_id = store.link_entities(
                entity_id_a=vessel_eid,
                entity_id_b=country_eid,
                link_type="port_call_to",
                source="ais_vessel_obs",
                confidence=1.0,
            )
            if link_id is not None:
                created += 1

    log.info("Seeded %d vessel→country links (port_call_to)", created)
    return created


def resolve_wallet_exchange(
    address: str | None,
    exchange_wallets: dict[str, tuple[str, str]] | None = None,
) -> tuple[str, str] | None:
    """Look up a BTC address in the known-exchange wallet dict.

    Args:
        address: BTC address string.
        exchange_wallets: Override dict; falls back to ``KNOWN_EXCHANGE_WALLETS``.

    Returns:
        ``(exchange_name, country_fips)`` if found, else ``None``.
    """
    if not address or not address.strip():
        return None
    wallets = exchange_wallets if exchange_wallets is not None else KNOWN_EXCHANGE_WALLETS
    return wallets.get(address.strip())


def seed_whale_country_links(
    store: Any,
    exchange_wallets: dict[str, tuple[str, str]] | None = None,
) -> int:
    """Create ``exchange_based_in`` links from wallet entities to country entities.

    Scans all wallet entities, checks their ``btc_address`` alias against a
    known-exchange dict, and creates links for matches.

    Args:
        store: PipelineStore instance.
        exchange_wallets: Dict mapping BTC address → (exchange, fips).
            If None, uses module-level ``KNOWN_EXCHANGE_WALLETS``.

    Returns:
        Number of new links created.
    """
    wallets = exchange_wallets if exchange_wallets is not None else KNOWN_EXCHANGE_WALLETS
    if not wallets:
        return 0

    conn = store._get_conn()
    rows = conn.execute("SELECT DISTINCT entity_id FROM entities WHERE entity_type='wallet'").fetchall()

    created = 0
    for row in rows:
        wallet_eid = row["entity_id"]
        # Get the btc_address alias to look up in exchange dict
        aliases = conn.execute(
            "SELECT external_id FROM entity_aliases WHERE entity_id=? AND source='btc_address'",
            (wallet_eid,),
        ).fetchall()

        for alias_row in aliases:
            addr = alias_row["external_id"]
            result = resolve_wallet_exchange(addr, wallets)
            if result is None:
                continue

            exchange_name, fips = result
            country_eid = entity_id_from_key("country", fips)

            # Register country entity if it doesn't exist
            existing = store.get_entity(country_eid)
            if existing is None:
                store.register_entity(
                    "country",
                    fips,
                    country_eid,
                    metadata={"fips": fips},
                )
                store.add_entity_alias(country_eid, "fips_country", fips)

            link_id = store.link_entities(
                entity_id_a=wallet_eid,
                entity_id_b=country_eid,
                link_type="exchange_based_in",
                source="exchange_wallet_match",
                confidence=1.0,
                metadata={"exchange": exchange_name},
            )
            if link_id is not None:
                created += 1

    log.info("Seeded %d wallet→country links (exchange_based_in)", created)
    return created


# ── CrossEntityDetector ─────────────────────────────────────


class CrossEntityDetector:
    """Discovers cross-domain L3 patterns from linked entity pairs.

    Usage::

        detector = CrossEntityDetector(store)
        patterns = detector.detect_insider_gdelt(
            company_entity_id="abc123...",
            window_seconds=72 * 3600,
        )
        detector.store_l3_observations(patterns)
    """

    def __init__(self, store: Any) -> None:
        self._store = store

    # ── Insider × GDELT ─────────────────────────────────────

    def detect_insider_gdelt(
        self,
        company_entity_id: str,
        *,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        goldstein_threshold: float = GOLDSTEIN_THRESHOLD,
        since: float | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Detect co-occurrences between insider trades and GDELT conflict events.

        Steps:
        1. Find the company's ``headquartered_in`` link → country entity.
        2. Query co-occurrences between insider_trade obs (company)
           and geopolitical_event obs (country).
        3. Filter to negative-Goldstein events (conflict).
        4. Score each co-occurrence and return.

        Returns list of pattern dicts ready for ``store_l3_observations()``.
        """
        # Step 1: Find linked country
        links = self._store.query_entity_links(
            company_entity_id,
            link_type="headquartered_in",
            direction="outgoing",
        )
        if not links:
            log.debug("No headquartered_in link for entity %s", company_entity_id)
            return []

        patterns: list[dict[str, Any]] = []
        for link in links:
            country_entity_id = link["entity_id_b"]

            # Step 2: Query temporal co-occurrences
            cooccs = self._store.query_co_occurrences(
                company_entity_id,
                country_entity_id,
                window_seconds=window_seconds,
                source_tool_a="insider_filings",
                source_tool_b="gdelt",
                since=since,
                limit=limit,
            )

            if not cooccs:
                continue

            # Step 3: Filter to negative Goldstein
            for coocc in cooccs:
                gdelt_value = coocc["obs_b"].get("value", {})
                goldstein = gdelt_value.get("goldstein")
                if goldstein is None:
                    continue
                try:
                    goldstein = float(goldstein)
                except (TypeError, ValueError):
                    continue
                if goldstein > goldstein_threshold:
                    continue

                # Step 4: Score — simple absolute-Goldstein × time-proximity
                time_delta_h = coocc["time_delta_seconds"] / 3600.0
                proximity = max(0.0, 1.0 - abs(time_delta_h) / (window_seconds / 3600.0))
                score = abs(goldstein) / 10.0 * proximity

                patterns.append(
                    {
                        "pattern_type": "insider_x_gdelt",
                        "entity_a": company_entity_id,
                        "entity_b": country_entity_id,
                        "insider_event": coocc["obs_a"].get("value", {}),
                        "gdelt_event": coocc["obs_b"].get("value", {}),
                        "time_delta_hours": time_delta_h,
                        "goldstein": goldstein,
                        "score": round(score, 6),
                        # Carry observation IDs for traceability
                        "obs_a_id": coocc["obs_a"]["id"],
                        "obs_b_id": coocc["obs_b"]["id"],
                    }
                )

        return patterns

    # ── Vessel × Sanctions ──────────────────────────────────

    def detect_vessel_sanctions(
        self,
        vessel_entity_id: str,
        *,
        window_seconds: float = VESSEL_WINDOW_SECONDS,
        sanctions_root_codes: frozenset[str] = SANCTIONS_ROOT_CODES,
        goldstein_threshold: float = GOLDSTEIN_THRESHOLD,
        since: float | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Detect co-occurrences between vessel movements and sanctions events.

        Steps:
        1. Find the vessel's ``port_call_to`` links → country entities.
        2. Query co-occurrences between vessel observations (port_call or
           vessel_position) and the country's geopolitical_event observations.
        3. Filter to sanctions-relevant CAMEO root codes and negative Goldstein.
        4. Score and return.

        Returns list of pattern dicts ready for ``store_l3_observations()``.
        """
        links = self._store.query_entity_links(
            vessel_entity_id,
            link_type="port_call_to",
            direction="outgoing",
        )
        if not links:
            log.debug("No port_call_to links for vessel %s", vessel_entity_id)
            return []

        patterns: list[dict[str, Any]] = []
        for link in links:
            country_entity_id = link["entity_id_b"]

            cooccs = self._store.query_co_occurrences(
                vessel_entity_id,
                country_entity_id,
                window_seconds=window_seconds,
                source_tool_a="ais_vessel",
                source_tool_b="gdelt",
                since=since,
                limit=limit,
            )

            if not cooccs:
                continue

            for coocc in cooccs:
                gdelt_value = coocc["obs_b"].get("value", {})

                # Filter: sanctions-relevant CAMEO root codes
                root_code = str(gdelt_value.get("event_root_code", ""))
                quad_class = gdelt_value.get("quad_class")
                is_sanctions = root_code in sanctions_root_codes
                is_material_conflict = quad_class == 4

                if not (is_sanctions or is_material_conflict):
                    continue

                # Filter: negative Goldstein
                goldstein = gdelt_value.get("goldstein")
                if goldstein is None:
                    continue
                try:
                    goldstein = float(goldstein)
                except (TypeError, ValueError):
                    continue
                if goldstein > goldstein_threshold:
                    continue

                # Score: severity × proximity (same formula as insider×gdelt)
                time_delta_h = coocc["time_delta_seconds"] / 3600.0
                window_h = window_seconds / 3600.0
                proximity = max(0.0, 1.0 - abs(time_delta_h) / window_h)
                score = abs(goldstein) / 10.0 * proximity

                patterns.append(
                    {
                        "pattern_type": "vessel_x_sanctions",
                        "entity_a": vessel_entity_id,
                        "entity_b": country_entity_id,
                        "vessel_event": coocc["obs_a"].get("value", {}),
                        "gdelt_event": coocc["obs_b"].get("value", {}),
                        "vessel_obs_type": coocc["obs_a"].get("observation_type", ""),
                        "time_delta_hours": time_delta_h,
                        "goldstein": goldstein,
                        "event_root_code": root_code,
                        "score": round(score, 6),
                        "obs_a_id": coocc["obs_a"]["id"],
                        "obs_b_id": coocc["obs_b"]["id"],
                    }
                )

        return patterns

    # ── Whale × Geopolitical ────────────────────────────────

    def detect_whale_geopolitical(
        self,
        wallet_entity_id: str,
        *,
        window_seconds: float = WHALE_WINDOW_SECONDS,
        goldstein_threshold: float = WHALE_GOLDSTEIN_THRESHOLD,
        value_scale: float = WHALE_VALUE_SCALE,
        since: float | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Detect co-occurrences between whale BTC transfers and GDELT events.

        Steps:
        1. Find the wallet's ``exchange_based_in`` links → country entities.
        2. Query co-occurrences between btc_transfer obs (wallet)
           and geopolitical_event obs (country).
        3. Filter to high-impact negative Goldstein (default ≤ -5.0).
        4. Score with value_btc weighting and return.

        Returns list of pattern dicts ready for ``store_l3_observations()``.
        """
        links = self._store.query_entity_links(
            wallet_entity_id,
            link_type="exchange_based_in",
            direction="outgoing",
        )
        if not links:
            log.debug("No exchange_based_in links for wallet %s", wallet_entity_id)
            return []

        patterns: list[dict[str, Any]] = []
        for link in links:
            country_entity_id = link["entity_id_b"]

            cooccs = self._store.query_co_occurrences(
                wallet_entity_id,
                country_entity_id,
                window_seconds=window_seconds,
                source_tool_a="whale_alert",
                source_tool_b="gdelt",
                since=since,
                limit=limit,
            )

            if not cooccs:
                continue

            for coocc in cooccs:
                gdelt_value = coocc["obs_b"].get("value", {})

                # Filter: high-impact negative Goldstein only
                goldstein = gdelt_value.get("goldstein")
                if goldstein is None:
                    continue
                try:
                    goldstein = float(goldstein)
                except (TypeError, ValueError):
                    continue
                if goldstein > goldstein_threshold:
                    continue

                # Extract transfer value for scoring
                whale_value = coocc["obs_a"].get("value", {})
                value_btc = float(whale_value.get("value_btc", 0))

                # Score: value_weight × severity × proximity
                value_weight = min(value_btc / value_scale, 1.0) if value_scale > 0 else 0.0
                time_delta_h = coocc["time_delta_seconds"] / 3600.0
                window_h = window_seconds / 3600.0
                proximity = max(0.0, 1.0 - abs(time_delta_h) / window_h)
                score = value_weight * (abs(goldstein) / 10.0) * proximity

                patterns.append(
                    {
                        "pattern_type": "whale_x_geopolitical",
                        "entity_a": wallet_entity_id,
                        "entity_b": country_entity_id,
                        "whale_event": whale_value,
                        "gdelt_event": coocc["obs_b"].get("value", {}),
                        "value_btc": value_btc,
                        "direction": whale_value.get("direction", ""),
                        "time_delta_hours": time_delta_h,
                        "goldstein": goldstein,
                        "score": round(score, 6),
                        "obs_a_id": coocc["obs_a"]["id"],
                        "obs_b_id": coocc["obs_b"]["id"],
                    }
                )

        return patterns

    # ── L3 observation storage ──────────────────────────────

    def store_l3_observations(
        self,
        patterns: list[dict[str, Any]],
        *,
        min_score: float = 0.0,
    ) -> int:
        """Store scored patterns as depth_level=3 entity observations.

        Each pattern is stored under the company entity (entity_a)
        with observation_type ``cross_entity_pattern``.

        Args:
            patterns: List of pattern dicts from a detect_* method.
            min_score: Only store patterns with score >= this threshold.

        Returns:
            Number of observations stored.
        """
        stored = 0
        for p in patterns:
            if p.get("score", 0.0) < min_score:
                continue
            self._store.store_entity_observation(
                entity_id=p["entity_a"],
                source_tool="cross_entity",
                observed_at=time.time(),
                observation_type="cross_entity_pattern",
                value=p,
                depth_level=3,
                metadata={"pattern_type": p["pattern_type"]},
            )
            stored += 1

        if stored:
            log.info("Stored %d L3 cross-entity observations", stored)
        return stored
