"""
Tool: Polymarket — Prediction Market Data

Fetches active prediction markets from Polymarket's Gamma API.
Returns current prices (implied probabilities), volume, liquidity,
and price changes. Zero cost — public REST API, no auth required.

Why this matters: prediction markets aggregate informed-money views
faster than polls, news, or traditional markets. When whale wallets
pile into a position, it often precedes real-world outcomes. This
tool gives the agent a read on what the smart money expects.

Gamma API base: https://gamma-api.polymarket.com
"""

from __future__ import annotations

import json
import logging
import time as _time
from datetime import datetime
from typing import Any

import httpx

from agent.data.cache import DataCache
from agent.data.dns_bypass import ensure_polymarket_dns
from agent.tools.base import Tool, ToolResult

try:
    from agent.pipeline.entity import entity_id_from_key
    from agent.pipeline.store import PipelineStore
except ImportError:  # pragma: no cover
    PipelineStore = None  # type: ignore[assignment,misc]
    entity_id_from_key = None  # type: ignore[assignment]

ensure_polymarket_dns()

log = logging.getLogger(__name__)

_GAMMA_BASE = "https://gamma-api.polymarket.com"

# Map Polymarket tag slugs → our normalized categories.
# Events can have multiple tags; first match wins.
_TAG_CATEGORIES: dict[str, str] = {
    "politics": "politics",
    "elections": "politics",
    "geopolitics": "geopolitics",
    "world": "geopolitics",
    "crypto": "crypto",
    "bitcoin": "crypto",
    "ethereum": "crypto",
    "finance": "finance",
    "economy": "finance",
    "stocks": "finance",
    "ipos": "finance",
    "interest-rates": "finance",
    "tech": "tech",
    "ai": "tech",
    "science": "science",
    "climate": "science",
    "sports": "sports",
}

_VALID_CATEGORIES = {
    "politics",
    "crypto",
    "finance",
    "geopolitics",
    "tech",
    "science",
    "sports",
    "all",
}

# Map our normalized categories → instrument tickers for topic→instrument links.
# Only categories with clear, tradeable instrument mappings are included.
_TOPIC_INSTRUMENT_MAP: dict[str, list[str]] = {
    "crypto": ["BTC-USD", "ETH-USD"],
    "finance": ["ES=F", "SPY", "ZN=F", "TLT", "XLF"],
    "politics": ["SPY", "ES=F", "ZN=F", "TLT", "VIXY"],
    "geopolitics": ["GC=F", "CL=F", "VIXY", "BZ=F", "GDX"],
    "tech": ["QQQ", "NQ=F", "XLK"],
    "science": ["XLV", "XLK"],
    "economics": ["SPY", "ZN=F", "TLT", "GC=F"],
}


def _iso_to_ts(s: str) -> float | None:
    """Parse an ISO 8601 date/datetime string to a Unix timestamp.

    Returns None if the string is empty or unparseable.
    Only returns a value if the resulting timestamp is in the past,
    so we never assign a future observed_at.
    """
    if not s:
        return None
    # Normalise trailing Z -> +00:00 for fromisoformat compatibility
    s_norm = s.replace("Z", "+00:00")
    # If date-only (YYYY-MM-DD) append midnight UTC
    if len(s_norm) == 10:
        s_norm += "T00:00:00+00:00"
    try:
        dt = datetime.fromisoformat(s_norm)
        ts = dt.timestamp()
    except ValueError:
        return None
    now = _time.time()
    return ts if ts < now else None


class PolymarketTool(Tool):
    name = "polymarket"

    description = (
        "Fetch active prediction markets from Polymarket. Returns current prices "
        "(implied probabilities), trading volume, liquidity, and recent price changes. "
        "Use this to see what informed money expects on elections, crypto events, "
        "geopolitics, macro policy, and more. Prediction markets often lead "
        "mainstream indicators."
    )

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": (
                    "Filter by category. Options: politics, crypto, finance, "
                    "geopolitics, tech, science, sports, all. Default: all."
                ),
                "default": "all",
            },
            "limit": {
                "type": "integer",
                "description": "Max number of markets to return. Default: 20.",
                "default": 20,
            },
            "search": {
                "type": "string",
                "description": (
                    "Optional search term to filter markets by title. E.g., 'Fed rate cut', 'Bitcoin', 'Trump'."
                ),
                "default": "",
            },
        },
        "required": [],
    }

    def __init__(
        self,
        cache: DataCache | None = None,
        *,
        pipeline_store: PipelineStore | None = None,
    ) -> None:
        self._cache = cache
        self._store = pipeline_store

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def execute(
        self,
        *,
        category: str = "all",
        limit: int = 20,
        search: str = "",
        mode: str = "active",
        days_back: int = 730,
        **_: Any,
    ) -> ToolResult:
        # --- Resolved-market backfill path ---
        if mode == "resolved":
            return self._execute_resolved(days_back=days_back)

        # --- Active-market path (original) ---
        category = category.lower().strip()
        if category not in _VALID_CATEGORIES:
            return ToolResult(
                success=False,
                output=f"Invalid category '{category}'. Must be one of: {', '.join(sorted(_VALID_CATEGORIES))}",
            )
        limit = max(1, min(limit, 100))  # clamp

        try:
            raw_events = self._fetch_events(limit=100)  # fetch more, filter after
        except Exception as exc:
            log.exception("Polymarket fetch failed")
            return ToolResult(success=False, output=f"Polymarket API error: {exc}")

        if not raw_events:
            return ToolResult(success=True, output="No active markets found.", data={"markets": []})

        markets = self._parse_markets(raw_events)

        # L2: persist topic entities + observations when PipelineStore available
        try:
            self._persist_entities(markets)
        except Exception:
            log.exception("Polymarket entity persistence failed (non-fatal)")

        # Filter by search term
        if search:
            term = search.lower()
            markets = [m for m in markets if term in m["question"].lower()]

        # Filter by category
        if category != "all":
            markets = [m for m in markets if m["category"] == category]

        # Sort by 24h volume descending (most active = most signal)
        markets.sort(key=lambda m: m["volume_24h"], reverse=True)

        # Apply limit
        markets = markets[:limit]

        if not markets:
            return ToolResult(
                success=True,
                output=f"No markets found for category='{category}'" + (f", search='{search}'" if search else "") + ".",
                data={"markets": []},
            )

        # Format human-readable output
        lines = [f"Polymarket — {len(markets)} active markets:\n"]
        for i, m in enumerate(markets, 1):
            price_str = f"YES {m['yes_price']:.0%} / NO {m['no_price']:.0%}"
            vol_str = f"${m['volume_24h']:,.0f} (24h)" if m["volume_24h"] else "no volume"
            change_parts = []
            if m["price_change_24h"] is not None:
                sign = "+" if m["price_change_24h"] >= 0 else ""
                change_parts.append(f"24h: {sign}{m['price_change_24h']:.1%}")
            if m["price_change_1wk"] is not None:
                sign = "+" if m["price_change_1wk"] >= 0 else ""
                change_parts.append(f"1wk: {sign}{m['price_change_1wk']:.1%}")
            change_str = " | ".join(change_parts) if change_parts else ""
            lines.append(
                f"  {i}. {m['question']}\n"
                f"     {price_str} | Vol: {vol_str}"
                + (f" | {change_str}" if change_str else "")
                + (f" | [{m['category']}]" if m["category"] else "")
            )

        output = "\n".join(lines)
        data = {"markets": markets, "total": len(markets)}
        return ToolResult(success=True, output=output, data=data)

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    def _execute_resolved(self, *, days_back: int) -> ToolResult:
        """Backfill path: fetch resolved markets and persist with historical timestamps."""
        cutoff_ts = _time.time() - days_back * 86400
        try:
            raw_events = self._fetch_resolved_events(limit=500)
        except Exception as exc:
            log.exception("Polymarket resolved fetch failed")
            return ToolResult(success=False, output=f"Polymarket resolved API error: {exc}")

        if not raw_events:
            return ToolResult(
                success=True,
                output="No resolved markets found.",
                data={"markets": [], "count": 0},
            )

        markets = self._parse_resolved_markets(raw_events, cutoff_ts=cutoff_ts)

        try:
            counts = self._persist_entities(markets, use_end_date=True)
        except Exception:
            log.exception("Polymarket resolved persistence failed (non-fatal)")
            counts = {"topics": 0, "observations": 0}

        return ToolResult(
            success=True,
            output=(
                f"Polymarket resolved backfill: {counts['observations']} observations "
                f"from {counts['topics']} markets (days_back={days_back})."
            ),
            data={"markets": markets, "count": len(markets)},
        )

    def _fetch_events(self, limit: int = 100) -> list[dict[str, Any]]:
        """Fetch active events from Gamma API. Returns raw event dicts."""
        cache_params = {"closed": False, "limit": limit}
        if self._cache:
            cached = self._cache.get("polymarket_events", cache_params)
            if cached is not None:
                log.debug("Cache hit for polymarket events")
                return cached

        with httpx.Client(timeout=15) as client:
            resp = client.get(
                f"{_GAMMA_BASE}/events",
                params={"closed": "false", "limit": str(limit), "active": "true"},
            )
            resp.raise_for_status()
            events = resp.json()

        if self._cache and events:
            self._cache.put("polymarket_events", cache_params, events)

        return events

    def _fetch_resolved_events(self, limit: int = 500) -> list[dict[str, Any]]:
        """Fetch resolved/closed markets from Gamma API (/markets endpoint).

        Uses volume sort so we get the most liquid, signal-rich resolved markets.
        Returns a flat list of market dicts (not event-wrapped).
        """
        cache_params = {"closed": True, "limit": limit, "order": "volume"}
        if self._cache:
            cached = self._cache.get("polymarket_resolved_markets", cache_params)
            if cached is not None:
                log.debug("Cache hit for polymarket resolved markets")
                return cached

        with httpx.Client(timeout=30) as client:
            resp = client.get(
                f"{_GAMMA_BASE}/markets",
                params={
                    "closed": "true",
                    "limit": str(limit),
                    "order": "volume",
                    "ascending": "false",
                },
            )
            resp.raise_for_status()
            markets = resp.json()

        if self._cache and markets:
            self._cache.put("polymarket_resolved_markets", cache_params, markets)

        return markets

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_resolved_markets(self, markets: list[dict[str, Any]], *, cutoff_ts: float) -> list[dict[str, Any]]:
        """Parse flat resolved market objects from the /markets endpoint.

        Only includes markets whose endDate resolves to a past timestamp >= cutoff_ts.
        """
        result: list[dict[str, Any]] = []

        for mkt in markets:
            slug = (mkt.get("slug") or "").strip()
            if not slug:
                continue

            # Prefer full ISO datetime (endDate) over date-only (endDateIso) for precision
            end_date_str = mkt.get("endDate") or mkt.get("endDateIso", "")
            end_ts = _iso_to_ts(end_date_str)
            if end_ts is None or end_ts < cutoff_ts:
                continue

            question = mkt.get("question", slug)

            prices_raw = mkt.get("outcomePrices", "")
            yes_price, no_price = self._parse_prices(prices_raw)

            volume_total = _safe_float(mkt.get("volumeNum") or mkt.get("volume"))
            volume_24h = _safe_float(mkt.get("volume24hr"))
            liquidity = _safe_float(mkt.get("liquidityNum") or mkt.get("liquidity"))

            result.append(
                {
                    "question": question,
                    "slug": slug,
                    "yes_price": yes_price or 0.0,
                    "no_price": no_price or 0.0,
                    "volume_total": volume_total or 0.0,
                    "volume_24h": volume_24h or 0.0,
                    "liquidity": liquidity or 0.0,
                    "spread": None,
                    "price_change_24h": None,
                    "price_change_1wk": None,
                    "end_date": end_date_str,
                    "end_ts": end_ts,
                    "category": "",
                    "resolved": True,
                }
            )

        return result

    def _parse_markets(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract structured market data from Gamma API events."""
        markets: list[dict[str, Any]] = []

        for event in events:
            # Determine category from tags
            category = self._categorize_event(event)

            for mkt in event.get("markets", []):
                # Skip markets with no price data (not yet deployed)
                prices_raw = mkt.get("outcomePrices", "")
                if not prices_raw:
                    continue

                yes_price, no_price = self._parse_prices(prices_raw)
                if yes_price is None:
                    continue

                # Skip closed/resolved markets
                if mkt.get("closed") or mkt.get("umaResolutionStatus") == "resolved":
                    continue

                question = mkt.get("question", event.get("title", "Unknown"))
                slug = mkt.get("slug", "")

                # Volume and liquidity — these are numeric or string fields
                volume_total = _safe_float(mkt.get("volumeNum") or mkt.get("volume"))
                volume_24h = _safe_float(mkt.get("volume24hr"))
                liquidity = _safe_float(mkt.get("liquidityNum") or mkt.get("liquidity"))

                # Spread from best bid/ask
                best_bid = _safe_float(mkt.get("bestBid"))
                best_ask = _safe_float(mkt.get("bestAsk"))
                spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else None

                # Price changes
                price_change_24h = _safe_float(mkt.get("oneDayPriceChange"))
                price_change_1wk = _safe_float(mkt.get("oneWeekPriceChange"))

                end_date = mkt.get("endDateIso", "")

                markets.append(
                    {
                        "question": question,
                        "slug": slug,
                        "yes_price": yes_price,
                        "no_price": no_price,
                        "volume_total": volume_total or 0.0,
                        "volume_24h": volume_24h or 0.0,
                        "liquidity": liquidity or 0.0,
                        "spread": spread,
                        "price_change_24h": price_change_24h,
                        "price_change_1wk": price_change_1wk,
                        "end_date": end_date,
                        "category": category,
                    }
                )

        return markets

    # ------------------------------------------------------------------
    # Entity persistence (L2)
    # ------------------------------------------------------------------

    def _persist_entities(self, markets: list[dict[str, Any]], *, use_end_date: bool = False) -> dict[str, int]:
        """Persist Polymarket markets as L2 topic entities with observations.

        Each market with a valid slug becomes a topic entity. A
        ``market_probability`` observation stores the current YES price,
        volume, liquidity, and price changes.

        When use_end_date=True, uses the market's end_ts (from endDateIso) as
        observed_at to give historical temporal depth for resolved markets.

        Skips silently if no PipelineStore is configured.
        Returns counts: {topics, observations}.
        """
        if self._store is None or entity_id_from_key is None:
            return {"topics": 0, "observations": 0}
        if not markets:
            return {"topics": 0, "observations": 0}

        try:
            return self._persist_entities_inner(markets, use_end_date=use_end_date)
        except Exception:
            log.exception("Polymarket entity persistence failed (non-fatal)")
            return {"topics": 0, "observations": 0}

    def _persist_entities_inner(self, markets: list[dict[str, Any]], *, use_end_date: bool = False) -> dict[str, int]:
        """Inner persistence logic separated for testability."""
        assert self._store is not None  # noqa: S101
        store = self._store

        counts = {"topics": 0, "observations": 0}
        seen_slugs: set[str] = set()

        for mkt in markets:
            slug = (mkt.get("slug") or "").strip()
            if not slug:
                continue

            # Register topic entity (deduped by slug)
            topic_eid = entity_id_from_key("topic", slug)
            if slug not in seen_slugs:
                seen_slugs.add(slug)
                store.register_entity(
                    entity_type="topic",
                    canonical_name=mkt.get("question", slug)[:200],
                    entity_id=topic_eid,
                    metadata={
                        "slug": slug,
                        "category": mkt.get("category", ""),
                        "source": "polymarket",
                    },
                )
                counts["topics"] += 1

                # Link topic → instruments based on category (Phase 36)
                category = mkt.get("category", "")
                for ticker in _TOPIC_INSTRUMENT_MAP.get(category, []):
                    inst_eid = entity_id_from_key("instrument", ticker)
                    store.link_entities(
                        entity_id_a=topic_eid,
                        entity_id_b=inst_eid,
                        link_type="topic_relates_to_instrument",
                        source="polymarket",
                        confidence=0.7,
                    )

            # Store market_probability observation
            # For resolved markets, use the actual resolution date so that
            # observed_at reflects when the outcome was known, not when we ingested it.
            if use_end_date:
                obs_ts = mkt.get("end_ts") or _iso_to_ts(mkt.get("end_date", "")) or _time.time()
            else:
                obs_ts = _time.time()

            store.store_entity_observation(
                entity_id=topic_eid,
                source_tool="polymarket",
                observed_at=obs_ts,
                observation_type="market_probability",
                depth_level=2,
                value={
                    "yes_price": mkt.get("yes_price"),
                    "no_price": mkt.get("no_price"),
                    "volume_24h": mkt.get("volume_24h"),
                    "volume_total": mkt.get("volume_total"),
                    "liquidity": mkt.get("liquidity"),
                    "spread": mkt.get("spread"),
                    "price_change_24h": mkt.get("price_change_24h"),
                    "price_change_1wk": mkt.get("price_change_1wk"),
                },
            )
            counts["observations"] += 1

        log.info(
            "Polymarket L2: %d topics, %d observations",
            counts["topics"],
            counts["observations"],
        )
        return counts

    def _categorize_event(self, event: dict[str, Any]) -> str:
        """Map event tags to one of our normalized categories."""
        for tag in event.get("tags", []):
            slug = tag.get("slug", "").lower()
            if slug in _TAG_CATEGORIES:
                return _TAG_CATEGORIES[slug]
        return ""

    @staticmethod
    def _parse_prices(prices_raw: str) -> tuple[float | None, float | None]:
        """Parse outcomePrices JSON string like '["0.42", "0.58"]'."""
        try:
            prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
            if isinstance(prices, list) and len(prices) >= 2:
                return float(prices[0]), float(prices[1])
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        return None, None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _safe_float(val: Any) -> float | None:
    """Convert value to float, returning None on failure."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if f == f else None  # NaN check
    except (ValueError, TypeError):
        return None
