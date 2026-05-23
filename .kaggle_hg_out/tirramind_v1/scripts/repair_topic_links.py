#!/usr/bin/env python3
"""repair_topic_links.py — one-shot cross-domain link repair.

Creates topic_relates_to_instrument links for all existing topic entities
in the DB using:
  1. Category-based mapping (mirrors _TOPIC_INSTRUMENT_MAP in polymarket.py)
  2. Keyword-based matching on canonical_name for uncategorized topics

Run after any tool backfill that added topic entities without wiring
instrument links (e.g., Polymarket backfill with old, narrow map).

Usage:
    python scripts/repair_topic_links.py [--db-path PATH] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.pipeline.store import PipelineStore

# ── Category → ticker map (keep in sync with polymarket.py) ──────────────────

_CATEGORY_MAP: dict[str, list[str]] = {
    "crypto":     ["BTC-USD", "ETH-USD"],
    "finance":    ["ES=F", "SPY", "ZN=F", "TLT", "XLF"],
    "politics":   ["SPY", "ES=F", "ZN=F", "TLT", "VIXY"],
    "geopolitics":["GC=F", "CL=F", "VIXY", "BZ=F", "GDX"],
    "tech":       ["QQQ", "NQ=F", "XLK"],
    "science":    ["XLV", "XLK"],
    "economics":  ["SPY", "ZN=F", "TLT", "GC=F"],
}

# ── Keyword → ticker map for uncategorized topic names ───────────────────────

_KEYWORD_MAP: list[tuple[list[str], list[str]]] = [
    # Crypto
    (["bitcoin", "btc"],                          ["BTC-USD"]),
    (["ethereum", "eth", "defi", "nft"],          ["ETH-USD"]),
    (["crypto", "cryptocurrency", "blockchain"],  ["BTC-USD", "ETH-USD"]),

    # Energy
    (["oil", "crude", "wti", "brent", "opec", "petroleum"],
                                                  ["CL=F", "BZ=F", "XLE", "USO"]),
    (["natural gas", "lng", "natgas"],            ["NG=F", "UNG"]),
    (["energy"],                                  ["XLE", "CL=F", "USO"]),

    # Metals / commodities
    (["gold", "precious metal"],                  ["GC=F", "GDX", "SLV"]),
    (["silver"],                                  ["SI=F", "SLV"]),
    (["copper", "metals", "mining"],              ["HG=F", "XLB", "GDX"]),
    (["wheat", "grain", "bread"],                 ["ZW=F"]),
    (["corn"],                                    ["ZC=F"]),
    (["soybean", "soy"],                          ["ZS=F"]),
    (["coffee"],                                  ["KC=F"]),
    (["sugar"],                                   ["SB=F"]),
    (["agriculture", "food security", "food price"],
                                                  ["ZW=F", "ZC=F", "ZS=F"]),

    # Rates / macro
    (["federal reserve", "fed rate", "fomc", "powell", "interest rate"],
                                                  ["ZN=F", "TLT", "SHY", "IEF"]),
    (["inflation", "cpi", "pce", "deflation"],    ["TLT", "GC=F"]),
    (["recession", "gdp", "growth"],              ["SPY", "TLT", "GC=F"]),
    (["dollar", "dxy", "usd index"],              ["USDCHF=X", "USDJPY=X"]),

    # Equities / indices
    (["s&p 500", "sp500", "spx", "s&p500"],       ["SPY", "ES=F"]),
    (["nasdaq"],                                  ["QQQ", "NQ=F"]),
    (["dow jones", "dow"],                        ["DIA", "YM=F"]),
    (["russell", "small cap"],                    ["IWM", "RTY=F"]),
    (["stocks", "equity", "equities", "stock market"],
                                                  ["SPY", "QQQ"]),

    # Volatility
    (["volatility", "vix", "fear index"],         ["VIXY", "UVXY"]),

    # Geopolitics / conflict
    (["war", "conflict", "military", "nato"],     ["GC=F", "CL=F", "VIXY"]),
    (["sanctions", "russia"],                     ["GC=F", "CL=F", "VIXY"]),
    (["ukraine"],                                 ["GC=F", "CL=F", "ZW=F", "VIXY"]),
    (["middle east", "iran", "israel", "saudi"],  ["CL=F", "BZ=F", "GC=F"]),

    # Elections / politics
    (["election", "president", "congress", "senate", "vote", "ballot",
      "democrat", "republican", "trump", "biden", "harris"],
                                                  ["SPY", "ZN=F", "VIXY", "TLT"]),

    # Tech / AI
    (["artificial intelligence", " ai ", "ai.", "nvidia", "semiconductor",
      "chip", "microsoft", "apple", "google", "meta", "amazon"],
                                                  ["QQQ", "XLK", "NQ=F"]),
    (["tech", "technology", "software", "internet"],
                                                  ["QQQ", "XLK"]),

    # Healthcare / pharma
    (["healthcare", "pharma", "drug", "fda", "vaccine", "cancer", "clinical"],
                                                  ["XLV"]),

    # Financials / banking
    (["bank", "banking", "financial", "wall street", "credit"],
                                                  ["XLF"]),

    # Country / region ETFs
    (["china", "chinese", "prc", "renminbi", "yuan"],
                                                  ["FXI", "EWH", "USDCNH=X"]),
    (["taiwan"],                                  ["EWT", "FXI"]),
    (["japan", "yen", "jpy", "japanese"],         ["EWJ", "USDJPY=X"]),
    (["europe", "european", "eurozone", "euro area", " eu "],
                                                  ["VGK", "EWG", "EWQ"]),
    (["germany", "german", "bundesbank"],         ["EWG", "EURUSD=X"]),
    (["uk", "britain", "british", "pound", "gbp", "brexit"],
                                                  ["EWU", "GBPUSD=X"]),
    (["france", "french"],                        ["EWQ", "EURUSD=X"]),
    (["india", "indian", "rupee"],                ["INDA"]),
    (["brazil", "brazilian", "real"],             ["EWZ"]),
    (["australia", "australian", "aud"],          ["EWA", "AUDUSD=X"]),
    (["canada", "canadian", "cad"],               ["EWC", "USDCAD=X"]),
    (["korea", "korean", "won"],                  ["EWY"]),
    (["mexico", "mexican", "peso"],               ["EWW"]),

    # Fixed income
    (["treasury", "bond", "debt", "yield curve", "10-year", "30-year"],
                                                  ["ZN=F", "ZB=F", "TLT", "IEF"]),
    (["high yield", "junk bond", "credit spread"], ["HYG", "LQD"]),
    (["emerging market", "em bond"],              ["EMB"]),
]


def _load_instruments(store: PipelineStore) -> dict[str, str]:
    """Return {ticker: entity_id} for all instrument entities in DB."""
    conn = store._get_conn()
    cur = conn.execute(
        "SELECT entity_id, metadata_json FROM entities WHERE entity_type = 'instrument'"
    )
    result: dict[str, str] = {}
    for entity_id, meta_json in cur.fetchall():
        meta = json.loads(meta_json or "{}")
        ticker = meta.get("ticker", "")
        if ticker:
            result[ticker] = entity_id
    return result


def _load_topics(store: PipelineStore) -> list[dict]:
    """Return list of {entity_id, canonical_name, category} for all topic entities."""
    conn = store._get_conn()
    cur = conn.execute(
        "SELECT entity_id, canonical_name, metadata_json FROM entities WHERE entity_type = 'topic'"
    )
    rows = []
    for entity_id, canonical_name, meta_json in cur.fetchall():
        meta = json.loads(meta_json or "{}")
        rows.append({
            "entity_id": entity_id,
            "canonical_name": (canonical_name or "").lower(),
            "category": (meta.get("category") or "").lower().strip(),
        })
    return rows


def _tickers_for_topic(canonical_name: str, category: str,
                       instrument_tickers: set[str]) -> list[str]:
    """Return list of tickers to link this topic to.

    Priority: category map first, then keyword matching.
    Filters to tickers that actually exist in the DB instrument universe.
    """
    matched: set[str] = set()

    # 1. Category map
    for ticker in _CATEGORY_MAP.get(category, []):
        if ticker in instrument_tickers:
            matched.add(ticker)

    # 2. Keyword matching on canonical_name (only if category didn't match or is empty)
    # Always run keyword matching to augment category hits
    for keywords, tickers in _KEYWORD_MAP:
        if any(kw in canonical_name for kw in keywords):
            for ticker in tickers:
                if ticker in instrument_tickers:
                    matched.add(ticker)

    return list(matched)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair topic→instrument links in DB")
    parser.add_argument("--db-path", default=".tirra_pipeline/pipeline.db")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be created without writing")
    args = parser.parse_args()

    store = PipelineStore(args.db_path)
    try:
        instruments = _load_instruments(store)
        topics = _load_topics(store)

        instrument_tickers = set(instruments.keys())
        print(f"Loaded {len(instruments)} instruments, {len(topics)} topics")

        created = 0
        skipped_sports = 0
        no_match = 0
        already_linked: dict[str, int] = {}

        for topic in topics:
            # Skip sports — no instrument relevance
            if topic["category"] == "sports":
                skipped_sports += 1
                continue

            tickers = _tickers_for_topic(
                topic["canonical_name"], topic["category"], instrument_tickers
            )

            if not tickers:
                no_match += 1
                continue

            for ticker in tickers:
                inst_eid = instruments[ticker]
                if args.dry_run:
                    print(f"  WOULD LINK: {topic['canonical_name'][:60]:60s} → {ticker}")
                    created += 1
                else:
                    result = store.link_entities(
                        entity_id_a=topic["entity_id"],
                        entity_id_b=inst_eid,
                        link_type="topic_relates_to_instrument",
                        source="repair_topic_links",
                        confidence=0.6,
                    )
                    if result is not None:
                        created += 1
                    else:
                        already_linked[ticker] = already_linked.get(ticker, 0) + 1

        print()
        print(f"=== Results ===")
        print(f"  Links {'would be ' if args.dry_run else ''}created: {created:,}")
        print(f"  Already existed (skipped):   {sum(already_linked.values()):,}")
        print(f"  Topics with no instrument match: {no_match:,}")
        print(f"  Sports topics skipped:           {skipped_sports:,}")

    finally:
        store.close()


if __name__ == "__main__":
    main()
