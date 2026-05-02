"""TirraMind — Instrument Universe

Defines the global tradeable instrument universe and daily price ingest.

Design:
    - InstrumentDef: frozen dataclass describing one instrument.
    - INSTRUMENTS: canonical tuple of ~90 instruments (all yfinance-verified).
    - ingest_daily_prices(): batch-fetch prices, register entities, store observations.

Instruments are first-class GNN entity nodes (type="instrument").
The GNN's cross-type attention discovers which entity patterns predict
instrument behavior. No hand-coded entity→instrument mapping.

Verified 2026-04-13: 89/90 tickers returned data from yfinance (period='5d').
LBS=F (lumber) was delisted — replaced with OJ=F (orange juice).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)


# ── Instrument definition ──────────────────────────────────────


@dataclass(frozen=True)
class InstrumentDef:
    """A single tradeable instrument."""

    ticker: str
    name: str
    asset_class: str  # commodity_future | fx | equity_index | equity_etf | sector_etf | fixed_income | vol | crypto
    region: str  # US | Europe | Asia | LatAm | Pacific | Global | EM
    is_tradeable: bool = True
    # Cross-domain linking metadata (Phase 25).
    # Only explicit, verifiable relationships — leave None when unknown.
    issuer: str | None = None  # canonical company/organization name (for ETFs/stocks)
    country: str | None = None  # ISO 3166-1 alpha-2 code for primary market/country
    cftc_code: str | None = None  # CFTC contract market code for futures
    # FX pair country metadata (Phase 27).
    # Deterministic two-country structure so both sides are graph-visible.
    base_country: str | None = None  # ISO code for base currency country
    quote_country: str | None = None  # ISO code for quote currency country
    # Crypto protocol metadata (Phase 30).
    protocol: str | None = None  # lowercased protocol name (e.g., "bitcoin", "ethereum")
    # Exchange country metadata (Phase 34).
    # ISO 3166-1 alpha-2 code for the country where the exchange is domiciled.
    # Distinct from `country` (underlying domicile) — enables GNN to learn
    # exchange-venue links separately from domicile links.
    primary_exchange_country: str | None = None


# ── Instrument universe ────────────────────────────────────────

INSTRUMENTS: tuple[InstrumentDef, ...] = (
    # ── Commodity Futures (20) ─────────────────────────────
    # country=None for commodities: they trade globally, no single issuer country.
    # primary_exchange_country="US": all trade on US exchanges (CME/NYMEX/CBOT/ICE US).
    # cftc_code: verified against CFTC disaggregated report (2026-04-15).
    InstrumentDef(
        "CL=F",
        "WTI Crude Oil",
        "commodity_future",
        "Global",
        cftc_code="06765A",
        primary_exchange_country="US",
    ),
    InstrumentDef(
        "BZ=F",
        "Brent Crude Oil",
        "commodity_future",
        "Global",
        cftc_code="06765T",
        primary_exchange_country="US",
    ),
    InstrumentDef(
        "NG=F",
        "Natural Gas",
        "commodity_future",
        "US",
        cftc_code="023651",
        primary_exchange_country="US",
    ),
    InstrumentDef(
        "RB=F",
        "RBOB Gasoline",
        "commodity_future",
        "US",
        cftc_code="111659",
        primary_exchange_country="US",
    ),
    InstrumentDef(
        "GC=F",
        "Gold",
        "commodity_future",
        "Global",
        cftc_code="088691",
        primary_exchange_country="US",
    ),
    InstrumentDef(
        "SI=F",
        "Silver",
        "commodity_future",
        "Global",
        cftc_code="084691",
        primary_exchange_country="US",
    ),
    InstrumentDef(
        "PL=F",
        "Platinum",
        "commodity_future",
        "Global",
        cftc_code="076651",
        primary_exchange_country="US",
    ),
    InstrumentDef(
        "PA=F",
        "Palladium",
        "commodity_future",
        "Global",
        cftc_code="075651",
        primary_exchange_country="US",
    ),
    InstrumentDef(
        "HG=F",
        "Copper",
        "commodity_future",
        "Global",
        cftc_code="085692",
        primary_exchange_country="US",
    ),
    InstrumentDef(
        "ZW=F",
        "Wheat",
        "commodity_future",
        "US",
        cftc_code="001602",
        primary_exchange_country="US",
    ),
    InstrumentDef(
        "ZC=F",
        "Corn",
        "commodity_future",
        "US",
        cftc_code="002602",
        primary_exchange_country="US",
    ),
    InstrumentDef(
        "ZS=F",
        "Soybeans",
        "commodity_future",
        "US",
        cftc_code="005602",
        primary_exchange_country="US",
    ),
    InstrumentDef(
        "KC=F",
        "Coffee",
        "commodity_future",
        "Global",
        cftc_code="083731",
        primary_exchange_country="US",
    ),
    InstrumentDef(
        "CC=F",
        "Cocoa",
        "commodity_future",
        "Global",
        cftc_code="073732",
        primary_exchange_country="US",
    ),
    InstrumentDef(
        "CT=F",
        "Cotton",
        "commodity_future",
        "US",
        cftc_code="033661",
        primary_exchange_country="US",
    ),
    InstrumentDef(
        "SB=F",
        "Sugar",
        "commodity_future",
        "Global",
        cftc_code="080732",
        primary_exchange_country="US",
    ),
    InstrumentDef(
        "ZO=F",
        "Oats",
        "commodity_future",
        "US",
        primary_exchange_country="US",
    ),  # no CFTC disagg code
    InstrumentDef(
        "OJ=F",
        "Orange Juice",
        "commodity_future",
        "US",
        cftc_code="040701",
        primary_exchange_country="US",
    ),
    InstrumentDef(
        "LE=F",
        "Live Cattle",
        "commodity_future",
        "US",
        cftc_code="057642",
        primary_exchange_country="US",
    ),
    InstrumentDef(
        "HE=F",
        "Lean Hogs",
        "commodity_future",
        "US",
        cftc_code="054642",
        primary_exchange_country="US",
    ),
    # ── FX Pairs (15) ─────────────────────────────────────
    # FX: base_country/quote_country capture both sides explicitly (Phase 27).
    # country retained for backward compat (non-USD side for USD pairs, base for crosses).
    # No CFTC code: disaggregated report covers physical commodities only.
    InstrumentDef(
        "EURUSD=X",
        "EUR/USD",
        "fx",
        "Global",
        country="EU",
        base_country="EU",
        quote_country="US",
    ),
    InstrumentDef(
        "USDJPY=X",
        "USD/JPY",
        "fx",
        "Global",
        country="JP",
        base_country="US",
        quote_country="JP",
    ),
    InstrumentDef(
        "GBPUSD=X",
        "GBP/USD",
        "fx",
        "Global",
        country="GB",
        base_country="GB",
        quote_country="US",
    ),
    InstrumentDef(
        "USDCHF=X",
        "USD/CHF",
        "fx",
        "Global",
        country="CH",
        base_country="US",
        quote_country="CH",
    ),
    InstrumentDef(
        "AUDUSD=X",
        "AUD/USD",
        "fx",
        "Global",
        country="AU",
        base_country="AU",
        quote_country="US",
    ),
    InstrumentDef(
        "USDCAD=X",
        "USD/CAD",
        "fx",
        "Global",
        country="CA",
        base_country="US",
        quote_country="CA",
    ),
    InstrumentDef(
        "NZDUSD=X",
        "NZD/USD",
        "fx",
        "Global",
        country="NZ",
        base_country="NZ",
        quote_country="US",
    ),
    InstrumentDef(
        "EURGBP=X",
        "EUR/GBP",
        "fx",
        "Global",
        country="EU",
        base_country="EU",
        quote_country="GB",
    ),
    InstrumentDef(
        "EURJPY=X",
        "EUR/JPY",
        "fx",
        "Global",
        country="EU",
        base_country="EU",
        quote_country="JP",
    ),
    InstrumentDef(
        "GBPJPY=X",
        "GBP/JPY",
        "fx",
        "Global",
        country="GB",
        base_country="GB",
        quote_country="JP",
    ),
    InstrumentDef(
        "USDMXN=X",
        "USD/MXN",
        "fx",
        "EM",
        country="MX",
        base_country="US",
        quote_country="MX",
    ),
    InstrumentDef(
        "USDBRL=X",
        "USD/BRL",
        "fx",
        "EM",
        country="BR",
        base_country="US",
        quote_country="BR",
    ),
    InstrumentDef(
        "USDINR=X",
        "USD/INR",
        "fx",
        "EM",
        country="IN",
        base_country="US",
        quote_country="IN",
    ),
    InstrumentDef(
        "USDCNY=X",
        "USD/CNY",
        "fx",
        "EM",
        country="CN",
        base_country="US",
        quote_country="CN",
    ),
    InstrumentDef(
        "USDZAR=X",
        "USD/ZAR",
        "fx",
        "EM",
        country="ZA",
        base_country="US",
        quote_country="ZA",
    ),
    # ── Equity Index Futures (4) ───────────────────────────
    # No CFTC code: handled by TFF report, not disaggregated.
    InstrumentDef("ES=F", "S&P 500 Futures", "equity_index", "US", country="US"),
    InstrumentDef("NQ=F", "Nasdaq 100 Futures", "equity_index", "US", country="US"),
    InstrumentDef("YM=F", "Dow Futures", "equity_index", "US", country="US"),
    InstrumentDef("RTY=F", "Russell 2000 Futures", "equity_index", "US", country="US"),
    # ── Equity ETFs (21) ──────────────────────────────────
    # issuer = ETF sponsor. country = primary country the ETF tracks.
    InstrumentDef(
        "SPY",
        "S&P 500 ETF",
        "equity_etf",
        "US",
        issuer="State Street Global Advisors",
        country="US",
    ),
    InstrumentDef("QQQ", "Nasdaq 100 ETF", "equity_etf", "US", issuer="Invesco", country="US"),
    InstrumentDef("IWM", "Russell 2000 ETF", "equity_etf", "US", issuer="BlackRock", country="US"),
    InstrumentDef(
        "DIA",
        "Dow Jones ETF",
        "equity_etf",
        "US",
        issuer="State Street Global Advisors",
        country="US",
    ),
    InstrumentDef("EWZ", "Brazil ETF", "equity_etf", "LatAm", issuer="BlackRock", country="BR"),
    InstrumentDef("EWG", "Germany ETF", "equity_etf", "Europe", issuer="BlackRock", country="DE"),
    InstrumentDef("FXI", "China ETF", "equity_etf", "Asia", issuer="BlackRock", country="CN"),
    InstrumentDef("EWJ", "Japan ETF", "equity_etf", "Asia", issuer="BlackRock", country="JP"),
    InstrumentDef("EWY", "South Korea ETF", "equity_etf", "Asia", issuer="BlackRock", country="KR"),
    InstrumentDef(
        "EWA",
        "Australia ETF",
        "equity_etf",
        "Pacific",
        issuer="BlackRock",
        country="AU",
    ),
    InstrumentDef("EWC", "Canada ETF", "equity_etf", "US", issuer="BlackRock", country="CA"),
    InstrumentDef(
        "EWU",
        "United Kingdom ETF",
        "equity_etf",
        "Europe",
        issuer="BlackRock",
        country="GB",
    ),
    InstrumentDef("EWQ", "France ETF", "equity_etf", "Europe", issuer="BlackRock", country="FR"),
    InstrumentDef("EWP", "Spain ETF", "equity_etf", "Europe", issuer="BlackRock", country="ES"),
    InstrumentDef("EWI", "Italy ETF", "equity_etf", "Europe", issuer="BlackRock", country="IT"),
    InstrumentDef("INDA", "India ETF", "equity_etf", "Asia", issuer="BlackRock", country="IN"),
    InstrumentDef("EWT", "Taiwan ETF", "equity_etf", "Asia", issuer="BlackRock", country="TW"),
    InstrumentDef("EWH", "Hong Kong ETF", "equity_etf", "Asia", issuer="BlackRock", country="HK"),
    InstrumentDef("THD", "Thailand ETF", "equity_etf", "Asia", issuer="BlackRock", country="TH"),
    InstrumentDef("EWW", "Mexico ETF", "equity_etf", "LatAm", issuer="BlackRock", country="MX"),
    InstrumentDef(
        "VGK",
        "FTSE Europe ETF",
        "equity_etf",
        "Europe",
        issuer="Vanguard",
        country="EU",
    ),
    # ── Sector ETFs (15) ──────────────────────────────────
    # Sector ETFs: issuer = sponsor, country = US (all US sector ETFs).
    InstrumentDef(
        "XLE",
        "Energy Select",
        "sector_etf",
        "US",
        issuer="State Street Global Advisors",
        country="US",
    ),
    InstrumentDef(
        "XLF",
        "Financials Select",
        "sector_etf",
        "US",
        issuer="State Street Global Advisors",
        country="US",
    ),
    InstrumentDef(
        "XLK",
        "Technology Select",
        "sector_etf",
        "US",
        issuer="State Street Global Advisors",
        country="US",
    ),
    InstrumentDef(
        "XLV",
        "Healthcare Select",
        "sector_etf",
        "US",
        issuer="State Street Global Advisors",
        country="US",
    ),
    InstrumentDef(
        "XLI",
        "Industrials Select",
        "sector_etf",
        "US",
        issuer="State Street Global Advisors",
        country="US",
    ),
    InstrumentDef(
        "XLP",
        "Consumer Staples Select",
        "sector_etf",
        "US",
        issuer="State Street Global Advisors",
        country="US",
    ),
    InstrumentDef(
        "XLY",
        "Consumer Discretionary Select",
        "sector_etf",
        "US",
        issuer="State Street Global Advisors",
        country="US",
    ),
    InstrumentDef(
        "XLB",
        "Materials Select",
        "sector_etf",
        "US",
        issuer="State Street Global Advisors",
        country="US",
    ),
    InstrumentDef(
        "XLU",
        "Utilities Select",
        "sector_etf",
        "US",
        issuer="State Street Global Advisors",
        country="US",
    ),
    InstrumentDef(
        "XLRE",
        "Real Estate Select",
        "sector_etf",
        "US",
        issuer="State Street Global Advisors",
        country="US",
    ),
    InstrumentDef(
        "XLC",
        "Communication Services Select",
        "sector_etf",
        "US",
        issuer="State Street Global Advisors",
        country="US",
    ),
    InstrumentDef("GDX", "Gold Miners ETF", "sector_etf", "Global", issuer="VanEck", country="US"),
    InstrumentDef("SLV", "Silver ETF", "sector_etf", "Global", issuer="BlackRock", country="US"),
    InstrumentDef("USO", "US Oil Fund", "sector_etf", "US", issuer="USCF", country="US"),
    InstrumentDef("UNG", "US Natural Gas Fund", "sector_etf", "US", issuer="USCF", country="US"),
    # ── Fixed Income (10) ─────────────────────────────────
    InstrumentDef("ZN=F", "10-Year T-Note Futures", "fixed_income", "US", country="US"),
    InstrumentDef("ZB=F", "30-Year T-Bond Futures", "fixed_income", "US", country="US"),
    InstrumentDef("ZF=F", "5-Year T-Note Futures", "fixed_income", "US", country="US"),
    InstrumentDef(
        "TLT",
        "20+ Year Treasury ETF",
        "fixed_income",
        "US",
        issuer="BlackRock",
        country="US",
    ),
    InstrumentDef(
        "IEF",
        "7-10 Year Treasury ETF",
        "fixed_income",
        "US",
        issuer="BlackRock",
        country="US",
    ),
    InstrumentDef(
        "SHY",
        "1-3 Year Treasury ETF",
        "fixed_income",
        "US",
        issuer="BlackRock",
        country="US",
    ),
    InstrumentDef(
        "HYG",
        "High Yield Corporate ETF",
        "fixed_income",
        "US",
        issuer="BlackRock",
        country="US",
    ),
    InstrumentDef(
        "LQD",
        "Investment Grade Corporate ETF",
        "fixed_income",
        "US",
        issuer="BlackRock",
        country="US",
    ),
    InstrumentDef("EMB", "EM Bond ETF", "fixed_income", "EM", issuer="BlackRock"),  # no single country
    InstrumentDef(
        "AGG",
        "US Aggregate Bond ETF",
        "fixed_income",
        "US",
        issuer="BlackRock",
        country="US",
    ),
    # ── Volatility (3) ────────────────────────────────────
    InstrumentDef("^VIX", "VIX Index", "vol", "US", is_tradeable=False, country="US"),
    InstrumentDef(
        "VIXY",
        "VIX Short-Term Futures ETF",
        "vol",
        "US",
        issuer="ProShares",
        country="US",
    ),
    InstrumentDef(
        "UVXY",
        "Ultra VIX Short-Term Futures ETF",
        "vol",
        "US",
        issuer="ProShares",
        country="US",
    ),
    # ── Crypto (2) ────────────────────────────────────────
    InstrumentDef("BTC-USD", "Bitcoin", "crypto", "Global", protocol="bitcoin"),
    InstrumentDef("ETH-USD", "Ethereum", "crypto", "Global", protocol="ethereum"),
)


# ── Helpers ────────────────────────────────────────────────────


def tradeable_instruments() -> list[InstrumentDef]:
    """Return only instruments that are tradeable (excludes e.g. ^VIX)."""
    return [i for i in INSTRUMENTS if i.is_tradeable]


def instruments_by_class(asset_class: str) -> list[InstrumentDef]:
    """Return instruments matching the given asset class."""
    return [i for i in INSTRUMENTS if i.asset_class == asset_class]


def ticker_to_instrument() -> dict[str, InstrumentDef]:
    """Return {ticker: InstrumentDef} lookup for all instruments."""
    return {i.ticker: i for i in INSTRUMENTS}


def cftc_code_to_ticker() -> dict[str, str]:
    """Return {CFTC_Contract_Market_Code: ticker} for instruments with cftc_code."""
    return {i.cftc_code: i.ticker for i in INSTRUMENTS if i.cftc_code}


def build_domain_company_map() -> dict[str, tuple[str, str]]:
    """Build lowercase domain-root keyword → (canonical_name, entity_id) map.

    Derived from INSTRUMENTS issuer names via ``normalize_company_name()``.
    Used by domain tools (cert_transparency, dns_monitor) to attempt
    ``domain_owned_by`` linking from a domain's base name.

    Returns:
        Dict mapping lowercase keywords to (canonical_name, company_entity_id).
        Keywords are the first word of the normalized issuer name, plus the
        full normalized name.  E.g. ``"blackrock"`` and ``"blackrock"`` both
        map to the same entity for issuer "BlackRock".
    """
    from agent.pipeline.entity import entity_id_from_key, normalize_company_name

    result: dict[str, tuple[str, str]] = {}
    for inst in INSTRUMENTS:
        if not inst.issuer:
            continue
        try:
            canon = normalize_company_name(inst.issuer)
        except ValueError:
            continue
        eid = entity_id_from_key("company", canon)
        # Map the full canonical name
        result[canon] = (canon, eid)
        # Also map the first word (e.g. "blackrock" from "blackrock")
        first_word = canon.split()[0] if canon else ""
        if first_word and first_word not in result:
            result[first_word] = (canon, eid)
    return result


# ── Cross-domain link persistence (Phase 25) ──────────────────


def _persist_instrument_links(store: PipelineStore) -> dict[str, int]:
    """Create cross-domain entity links from deterministic instrument metadata.

    Links created:
    - ``tracks_issuer``: instrument → company (ETF/stock → issuer org)
    - ``located_in``: instrument → country (primary country/market)
    - ``located_in``: company → country (issuer HQ → country, deduped)
    - ``fx_base_country``: instrument → country (FX base currency side, Phase 27)
    - ``fx_quote_country``: instrument → country (FX quote currency side, Phase 27)
    - ``tracks_protocol``: instrument → protocol (crypto → blockchain protocol, Phase 30)
    - ``exchange_country``: instrument → country (exchange domicile, Phase 34)

    Only explicit metadata is used.  Instruments without issuer/country are skipped.

    Returns:
        Counts of links created per link type.
    """
    from agent.pipeline.entity import entity_id_from_key, normalize_company_name

    counts = {
        "tracks_issuer": 0,
        "inst_country": 0,
        "issuer_country": 0,
        "fx_base_country": 0,
        "fx_quote_country": 0,
        "tracks_protocol": 0,
        "exchange_country": 0,
    }
    seen_issuers: dict[str, str] = {}  # canonical_name → entity_id (dedup)

    for inst in INSTRUMENTS:
        inst_eid = _entity_id(inst.ticker)

        # ── instrument → company (tracks_issuer) ──
        if inst.issuer:
            try:
                canon = normalize_company_name(inst.issuer)
            except ValueError:
                log.warning("Cannot normalize issuer %r for %s", inst.issuer, inst.ticker)
                continue

            if canon not in seen_issuers:
                issuer_eid = entity_id_from_key("company", canon)
                store.register_entity(
                    entity_type="company",
                    canonical_name=canon,
                    entity_id=issuer_eid,
                    metadata={"source": "instrument_universe", "raw_name": inst.issuer},
                )
                seen_issuers[canon] = issuer_eid
            else:
                issuer_eid = seen_issuers[canon]

            link_id = store.link_entities(
                entity_id_a=inst_eid,
                entity_id_b=issuer_eid,
                link_type="tracks_issuer",
                source="instrument_universe",
                confidence=1.0,
                metadata={"ticker": inst.ticker},
            )
            if link_id:
                counts["tracks_issuer"] += 1

            # ── company → country (located_in, for issuer) ──
            if inst.country:
                country_eid = entity_id_from_key("country", inst.country)
                store.register_entity(
                    entity_type="country",
                    canonical_name=inst.country,
                    entity_id=country_eid,
                )
                link_id = store.link_entities(
                    entity_id_a=issuer_eid,
                    entity_id_b=country_eid,
                    link_type="located_in",
                    source="instrument_universe",
                    confidence=1.0,
                )
                if link_id:
                    counts["issuer_country"] += 1

        # ── instrument → country (located_in) ──
        if inst.country:
            country_eid = entity_id_from_key("country", inst.country)
            store.register_entity(
                entity_type="country",
                canonical_name=inst.country,
                entity_id=country_eid,
            )
            link_id = store.link_entities(
                entity_id_a=inst_eid,
                entity_id_b=country_eid,
                link_type="located_in",
                source="instrument_universe",
                confidence=1.0,
                metadata={"ticker": inst.ticker},
            )
            if link_id:
                counts["inst_country"] += 1

        # ── FX pair two-country links (Phase 27) ──
        # Creates explicit fx_base_country / fx_quote_country edges so the
        # GNN sees both sides of every FX pair through distinct link types.
        for side, field in (
            ("fx_base_country", inst.base_country),
            ("fx_quote_country", inst.quote_country),
        ):
            if field is None:
                continue
            side_eid = entity_id_from_key("country", field)
            store.register_entity(
                entity_type="country",
                canonical_name=field,
                entity_id=side_eid,
            )
            link_id = store.link_entities(
                entity_id_a=inst_eid,
                entity_id_b=side_eid,
                link_type=side,
                source="instrument_universe",
                confidence=1.0,
                metadata={"ticker": inst.ticker},
            )
            if link_id:
                counts[side] += 1

        # ── instrument → protocol (tracks_protocol, Phase 30) ──
        if inst.protocol:
            protocol_eid = entity_id_from_key("protocol", inst.protocol)
            store.register_entity(
                entity_type="protocol",
                canonical_name=inst.protocol,
                entity_id=protocol_eid,
            )
            link_id = store.link_entities(
                entity_id_a=inst_eid,
                entity_id_b=protocol_eid,
                link_type="tracks_protocol",
                source="instrument_universe",
                confidence=1.0,
                metadata={"ticker": inst.ticker},
            )
            if link_id:
                counts["tracks_protocol"] += 1

        # ── instrument → country (exchange_country, Phase 34) ──
        if inst.primary_exchange_country:
            exc_eid = entity_id_from_key("country", inst.primary_exchange_country)
            store.register_entity(
                entity_type="country",
                canonical_name=inst.primary_exchange_country,
                entity_id=exc_eid,
            )
            link_id = store.link_entities(
                entity_id_a=inst_eid,
                entity_id_b=exc_eid,
                link_type="exchange_country",
                source="instrument_universe",
                confidence=1.0,
                metadata={"ticker": inst.ticker},
            )
            if link_id:
                counts["exchange_country"] += 1

    log.info(
        "Instrument links: %d tracks_issuer, %d inst→country, %d issuer→country, "
        "%d fx_base_country, %d fx_quote_country, %d tracks_protocol, "
        "%d exchange_country",
        counts["tracks_issuer"],
        counts["inst_country"],
        counts["issuer_country"],
        counts["fx_base_country"],
        counts["fx_quote_country"],
        counts["tracks_protocol"],
        counts["exchange_country"],
    )
    return counts


# ── Daily price ingest ─────────────────────────────────────────

# Entity ID helper (inline to avoid circular import at module level)
_ENTITY_TYPE = "instrument"
_SOURCE_TOOL = "instrument_universe"


def _entity_id(ticker: str) -> str:
    """Deterministic entity ID for an instrument ticker."""
    import hashlib

    raw = f"{_ENTITY_TYPE}:{ticker}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def ingest_daily_prices(
    store: PipelineStore,
    as_of: date | None = None,
    lookback_days: int = 30,
) -> dict[str, Any]:
    """Fetch daily prices for all tradeable instruments and store as entity observations.

    Parameters
    ----------
    store : PipelineStore to write entities/observations.
    as_of : Reference date (default: today). Only the most recent row is stored as
            today's observation, but we fetch ``lookback_days`` to compute vol/avg_volume.
    lookback_days : How many days of history to fetch for vol calculation.

    Returns
    -------
    Summary dict with keys: instruments_fetched, instruments_failed, observations_stored.
    """
    import yfinance as yf

    if as_of is None:
        as_of = date.today()

    instruments = tradeable_instruments()
    tickers = [i.ticker for i in instruments]
    ticker_map = {i.ticker: i for i in instruments}

    # ── Batch download ─────────────────────────────────────
    log.info("Downloading %d instruments, lookback=%dd", len(tickers), lookback_days)
    try:
        # yfinance batch download: returns MultiIndex columns (ticker, field)
        # when multiple tickers. Returns single-level when 1 ticker.
        raw = yf.download(
            tickers,
            period=f"{lookback_days}d",
            interval="1d",
            group_by="ticker",
            progress=False,
            threads=True,
        )
    except Exception:
        log.exception("Batch yfinance download failed")
        raise RuntimeError("Instrument price download failed entirely")

    # ── Process per ticker ─────────────────────────────────
    fetched = 0
    failed: list[str] = []
    obs_stored = 0
    observed_at = time.mktime(as_of.timetuple())

    for ticker in tickers:
        try:
            # Extract this ticker's data from the batch result
            if len(tickers) == 1:
                df = raw  # single ticker: flat columns
            else:
                if ticker not in raw.columns.get_level_values(0):
                    failed.append(ticker)
                    continue
                df = raw[ticker]

            # Drop rows with NaN close
            df = df.dropna(subset=["Close"])
            if df.empty:
                failed.append(ticker)
                continue

            # ── Compute signals from history ───────────────
            closes = df["Close"].values.astype(float)
            volumes = df["Volume"].values.astype(float) if "Volume" in df.columns else np.zeros(len(closes))
            highs = df["High"].values.astype(float) if "High" in df.columns else closes
            lows = df["Low"].values.astype(float) if "Low" in df.columns else closes

            # Log returns
            if len(closes) >= 2:
                log_returns = np.diff(np.log(closes))
                latest_return = float(log_returns[-1])
            else:
                log_returns = np.array([])
                latest_return = float("nan")

            latest_close = float(closes[-1])
            latest_volume = float(volumes[-1]) if len(volumes) > 0 else 0.0

            # 20d realized vol (annualised)
            if len(log_returns) >= 20:
                realized_vol = float(np.std(log_returns[-20:]) * math.sqrt(252))
            elif len(log_returns) >= 2:
                realized_vol = float(np.std(log_returns) * math.sqrt(252))
            else:
                realized_vol = float("nan")

            # 20d average volume
            if len(volumes) >= 20:
                avg_volume = float(np.mean(volumes[-20:]))
            elif len(volumes) >= 1:
                avg_volume = float(np.mean(volumes))
            else:
                avg_volume = 0.0

            # Intraday range (latest bar)
            if len(highs) > 0 and len(lows) > 0:
                intraday_range = float(highs[-1] - lows[-1])
            else:
                intraday_range = 0.0

            # ── Register entity ────────────────────────────
            inst = ticker_map[ticker]
            eid = _entity_id(ticker)
            store.register_entity(
                entity_type=_ENTITY_TYPE,
                canonical_name=inst.name,
                entity_id=eid,
                metadata={
                    "ticker": ticker,
                    "asset_class": inst.asset_class,
                    "region": inst.region,
                },
            )

            # ── Store observations ─────────────────────────
            if not math.isnan(latest_return):
                store.store_entity_observation(
                    entity_id=eid,
                    source_tool=_SOURCE_TOOL,
                    observed_at=observed_at,
                    observation_type="instrument_return",
                    value={"log_return": latest_return, "close": latest_close},
                    depth_level=1,
                )
                obs_stored += 1

            store.store_entity_observation(
                entity_id=eid,
                source_tool=_SOURCE_TOOL,
                observed_at=observed_at,
                observation_type="instrument_volume",
                value={"volume": latest_volume, "avg_volume_20d": avg_volume},
                depth_level=1,
            )
            obs_stored += 1

            if not math.isnan(realized_vol):
                store.store_entity_observation(
                    entity_id=eid,
                    source_tool=_SOURCE_TOOL,
                    observed_at=observed_at,
                    observation_type="instrument_volatility",
                    value={
                        "realized_vol_20d": realized_vol,
                        "intraday_range": intraday_range,
                    },
                    depth_level=1,
                )
                obs_stored += 1

            fetched += 1

        except Exception:
            log.warning("Failed to process instrument %s", ticker, exc_info=True)
            failed.append(ticker)

    # ── Check failure rate ─────────────────────────────────
    total = len(tickers)
    if len(failed) > total / 2:
        raise RuntimeError(
            f"Instrument ingest: {len(failed)}/{total} tickers failed "
            f"(>50%). Likely API issue. Failed: {failed[:10]}..."
        )

    # ── Cross-domain entity links (Phase 25) ───────────────
    try:
        link_counts = _persist_instrument_links(store)
    except Exception:
        log.warning("Instrument link persistence failed (non-fatal)", exc_info=True)
        link_counts = {}

    summary = {
        "instruments_fetched": fetched,
        "instruments_failed": failed,
        "observations_stored": obs_stored,
        "links_created": link_counts,
        "as_of": as_of.isoformat(),
    }
    log.info(
        "Instrument ingest complete: %d/%d fetched, %d observations stored",
        fetched,
        total,
        obs_stored,
    )
    return summary


# ── Historical backfill ────────────────────────────────────────


def backfill_historical_prices(
    store: PipelineStore,
    lookback_years: int = 3,
    batch_size: int = 20,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Fetch multi-year daily prices and store each day as a separate observation.

    Parameters
    ----------
    store : PipelineStore to write entities/observations.
    lookback_years : How many years of history to fetch (default: 3).
    batch_size : How many tickers to download per yfinance batch (default: 20).
    skip_existing : If True, skip instruments that already have >=100 historical
                    observations (resume capability).

    Returns
    -------
    Summary dict with keys: instruments_backfilled, instruments_skipped,
    instruments_failed, total_observations.
    """
    import yfinance as yf

    instruments = tradeable_instruments()
    ticker_map = {i.ticker: i for i in instruments}

    # ── Determine which tickers need backfill ──────────────
    tickers_to_fill: list[str] = []
    skipped: list[str] = []

    if skip_existing:
        for inst in instruments:
            eid = _entity_id(inst.ticker)
            try:
                existing = store.count_entity_observations(
                    entity_id=eid,
                    source_tool=_SOURCE_TOOL,
                )
            except Exception:
                existing = 0
            if existing >= 100:
                skipped.append(inst.ticker)
            else:
                tickers_to_fill.append(inst.ticker)
    else:
        tickers_to_fill = [i.ticker for i in instruments]

    log.info(
        "Backfill: %d tickers to fill, %d skipped (existing data)",
        len(tickers_to_fill),
        len(skipped),
    )

    # ── Process in batches ─────────────────────────────────
    total_obs = 0
    filled: list[str] = []
    failed: list[str] = []
    period = f"{lookback_years}y"

    for batch_start in range(0, len(tickers_to_fill), batch_size):
        batch_tickers = tickers_to_fill[batch_start : batch_start + batch_size]
        log.info(
            "Backfill batch %d-%d / %d",
            batch_start,
            batch_start + len(batch_tickers),
            len(tickers_to_fill),
        )

        try:
            raw = yf.download(
                batch_tickers,
                period=period,
                interval="1d",
                group_by="ticker",
                progress=False,
                threads=True,
            )
        except Exception:
            log.exception("Batch download failed for %s", batch_tickers[:5])
            failed.extend(batch_tickers)
            continue

        for ticker in batch_tickers:
            try:
                # Extract per-ticker dataframe
                if len(batch_tickers) == 1:
                    df = raw
                else:
                    if ticker not in raw.columns.get_level_values(0):
                        failed.append(ticker)
                        continue
                    df = raw[ticker]

                df = df.dropna(subset=["Close"])
                if df.empty:
                    failed.append(ticker)
                    continue

                # Register entity
                inst = ticker_map[ticker]
                eid = _entity_id(ticker)
                store.register_entity(
                    entity_type=_ENTITY_TYPE,
                    canonical_name=inst.name,
                    entity_id=eid,
                    metadata={
                        "ticker": ticker,
                        "asset_class": inst.asset_class,
                        "region": inst.region,
                    },
                )

                # Compute log returns
                closes = df["Close"].values.astype(float)
                volumes = df["Volume"].values.astype(float) if "Volume" in df.columns else np.zeros(len(closes))

                log_returns = np.diff(np.log(closes))
                dates = df.index.tolist()
                obs_count = 0

                # Store each day as a separate observation (skip day 0 — no return)
                for i in range(1, len(closes)):
                    day = dates[i]
                    if hasattr(day, "timestamp"):
                        observed_at = day.timestamp()
                    else:
                        observed_at = time.mktime(day.timetuple())

                    ret = float(log_returns[i - 1])
                    close = float(closes[i])
                    vol = float(volumes[i]) if i < len(volumes) else 0.0

                    # 20d realized vol (annualised) — rolling window
                    if i >= 20:
                        rv = float(np.std(log_returns[i - 20 : i]) * math.sqrt(252))
                    elif i >= 2:
                        rv = float(np.std(log_returns[:i]) * math.sqrt(252))
                    else:
                        rv = float("nan")

                    value = {
                        "close": close,
                        "log_return": ret,
                        "volume": vol,
                    }
                    if not math.isnan(rv):
                        value["realized_vol_20d"] = rv

                    store.store_entity_observation(
                        entity_id=eid,
                        source_tool=_SOURCE_TOOL,
                        observed_at=observed_at,
                        observation_type="instrument_daily",
                        value=value,
                        depth_level=1,
                    )
                    obs_count += 1

                total_obs += obs_count
                filled.append(ticker)
                log.debug("Backfilled %s: %d observations", ticker, obs_count)

            except Exception:
                log.warning("Failed to backfill %s", ticker, exc_info=True)
                failed.append(ticker)

        # Rate limiting between batches
        if batch_start + batch_size < len(tickers_to_fill):
            time.sleep(2)

    # Persist entity links
    try:
        _persist_instrument_links(store)
    except Exception:
        log.warning("Link persistence after backfill failed (non-fatal)", exc_info=True)

    summary = {
        "instruments_backfilled": len(filled),
        "instruments_skipped": len(skipped),
        "instruments_failed": failed,
        "total_observations": total_obs,
        "period": period,
    }
    log.info(
        "Backfill complete: %d filled, %d skipped, %d failed, %d total obs",
        len(filled),
        len(skipped),
        len(failed),
        total_obs,
    )
    return summary


# ── DAG callback ───────────────────────────────────────────────


def run_instrument_ingest(
    params: dict[str, Any],
    upstream_results: dict[str, Any],
) -> dict[str, Any]:
    """FunctionOperator callback for the daily_collection DAG.

    params:
        db_path: str — PipelineStore database path (injected by DAG builder)
    """
    from agent.pipeline.store import PipelineStore

    db_path = params.get("db_path", ".tirra_pipeline/pipeline.db")
    store = PipelineStore(db_path)
    try:
        return ingest_daily_prices(store)
    finally:
        store.close()
