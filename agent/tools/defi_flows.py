"""
Tool: DeFi On-Chain Flows — DefiLlama Protocol TVL, Stablecoins & DEX Volume

DefiLlama is a free, no-auth aggregator of on-chain DeFi data.
All data is derived from blockchain state — immutable and unmanipulable.

Endpoints used:
  https://api.llama.fi/protocols          — TVL by protocol (7000+ protocols)
  https://stablecoins.llama.fi/stablecoins — Stablecoin circulating supply (350+)
  https://api.llama.fi/overview/dexs       — DEX trading volume (1000+ DEXes)

Signal theory:
  - TVL drain from a protocol/chain = capital flight, pre-exploit, or confidence loss
  - Stablecoin mint (USDT/USDC supply ↑) = new liquidity entering crypto → risk-on
  - Stablecoin burn (supply ↓) = capital exiting → risk-off for crypto & correlated assets
  - DEX volume spike = on-chain panic or rotation (often precedes CEX price moves)
  - Protocol TVL concentration shift = DeFi capital rotation (e.g., from Ethereum to Solana)
  - Individual protocol TVL collapse = rug-pull or exploit early warning

Modes:
  tvl        — Top protocols by TVL, with chain/category breakdown.
  stablecoins — Stablecoin supply by issuer (circulating, peg mechanism).
  dex_volume — DEX trading volumes, sorted by 24h volume.
  chain      — TVL aggregated by chain (Ethereum, Solana, etc.).
  history    — Daily TVL time-series for top N protocols (years of history).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

try:
    from agent.pipeline.entity import entity_id_from_key
except ImportError:  # pragma: no cover
    entity_id_from_key = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

_PROTOCOLS_URL = "https://api.llama.fi/protocols"
_STABLECOINS_URL = "https://stablecoins.llama.fi/stablecoins?includePrices=true"
_DEXS_URL = "https://api.llama.fi/overview/dexs?excludeTotalDataChart=true&excludeTotalDataChartBreakdown=true&dataType=dailyVolume"
_UA = "TirraMind/0.1"
_TIMEOUT = 20

VALID_MODES = {"tvl", "stablecoins", "dex_volume", "chain", "history"}

_PROTOCOL_HISTORY_URL = "https://api.llama.fi/protocol/{slug}"

# Categories that are most market-relevant
_MARKET_CATEGORIES = {
    "Liquid Staking",
    "Lending",
    "DEX",
    "Bridge",
    "CDP",
    "Yield",
    "Derivatives",
    "RWA",
    "Restaking",
    "Yield Aggregator",
    "CEX",
}


class DefiFlowsTool(Tool):
    """Query DeFi protocol TVL, stablecoin supply, and DEX volumes from DefiLlama."""

    def __init__(
        self,
        cache: DataCache | None = None,
        *,
        pipeline_store: PipelineStore | None = None,
    ) -> None:
        self._cache = cache
        self._store = pipeline_store

    @property
    def name(self) -> str:
        return "defi_flows"

    @property
    def description(self) -> str:
        return (
            "Query on-chain DeFi data from DefiLlama: protocol TVL, "
            "stablecoin circulating supply, DEX trading volumes, and "
            "chain-level TVL. All data is derived from immutable blockchain state."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": sorted(VALID_MODES),
                    "description": (
                        "Query mode: tvl (top protocols), stablecoins (supply by issuer), "
                        "dex_volume (24h DEX volumes), chain (TVL by chain)."
                    ),
                },
                "chain": {
                    "type": "string",
                    "description": "Filter by chain name (e.g. 'Ethereum', 'Solana'). Optional.",
                },
                "category": {
                    "type": "string",
                    "description": "Filter by protocol category (e.g. 'Lending', 'DEX'). Optional.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 20, max 100).",
                },
            },
            "required": ["mode"],
        }

    # ------------------------------------------------------------------
    # Entity persistence (L2)
    # ------------------------------------------------------------------

    def _persist_entities(self, protocols: list[dict[str, Any]]) -> None:
        """Register protocol entities and store L2 TVL observations."""
        if self._store is None or entity_id_from_key is None:
            return
        if not protocols:
            return
        try:
            self._persist_entities_inner(protocols)
        except Exception:
            log.exception("Entity persistence failed (non-fatal)")

    def _persist_entities_inner(self, protocols: list[dict[str, Any]]) -> None:
        assert self._store is not None  # noqa: S101
        store = self._store

        seen: set[str] = set()
        now = time.time()
        for proto in protocols:
            name = proto.get("name", "")
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())

            protocol_eid = entity_id_from_key("protocol", name.lower())
            store.register_entity(
                entity_type="protocol",
                canonical_name=name,
                entity_id=protocol_eid,
            )
            store.add_entity_alias(protocol_eid, "protocol_name", name)

            store.store_entity_observation(
                entity_id=protocol_eid,
                source_tool="defi_flows",
                observed_at=now,
                observation_type="tvl_change",
                depth_level=2,
                value={
                    "tvl_usd": proto.get("tvl_usd", 0.0),
                    "chain": proto.get("chain", ""),
                    "chains": proto.get("chains", []),
                    "category": proto.get("category", ""),
                    "change_1d_pct": proto.get("change_1d_pct"),
                    "change_7d_pct": proto.get("change_7d_pct"),
                },
            )

    def execute(self, **kwargs: Any) -> ToolResult:
        mode = kwargs.get("mode", "")
        if mode not in VALID_MODES:
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Must be one of: {sorted(VALID_MODES)}",
            )

        limit = min(max(int(kwargs.get("limit", 20)), 1), 100)
        chain_filter = (kwargs.get("chain") or "").strip()
        category_filter = (kwargs.get("category") or "").strip()
        days_back = int(kwargs.get("days_back", 730))

        try:
            if mode == "tvl":
                return self._tvl(limit, chain_filter, category_filter)
            elif mode == "stablecoins":
                return self._stablecoins(limit)
            elif mode == "dex_volume":
                return self._dex_volume(limit, chain_filter)
            elif mode == "chain":
                return self._chain_tvl(limit)
            elif mode == "history":
                return self._tvl_history(limit, days_back, chain_filter, category_filter)
        except httpx.TimeoutException:
            return ToolResult(success=False, output="DefiLlama API timed out.")
        except httpx.HTTPError as exc:
            return ToolResult(success=False, output=f"DefiLlama API error: {exc}")
        except Exception as exc:
            log.exception("DefiFlowsTool error")
            return ToolResult(success=False, output=f"Unexpected error: {exc}")

        return ToolResult(success=False, output=f"Unhandled mode: {mode}")

    def _tvl(self, limit: int, chain_filter: str, category_filter: str) -> ToolResult:
        data = self._fetch_json(_PROTOCOLS_URL)
        if data is None:
            return ToolResult(success=False, output="Failed to fetch protocol data.")

        if chain_filter:
            cf_lower = chain_filter.lower()
            data = [p for p in data if cf_lower in [c.lower() for c in (p.get("chains") or [])]]
        if category_filter:
            cat_lower = category_filter.lower()
            data = [p for p in data if (p.get("category") or "").lower() == cat_lower]

        # Sort by TVL descending (already sorted by API, but enforce)
        data.sort(key=lambda p: p.get("tvl") or 0, reverse=True)
        top = data[:limit]

        results = []
        for p in top:
            tvl = p.get("tvl") or 0
            change_1d = p.get("change_1d")
            change_7d = p.get("change_7d")
            results.append(
                {
                    "name": p.get("name"),
                    "tvl_usd": round(tvl, 2),
                    "category": p.get("category"),
                    "chain": p.get("chain"),
                    "chains": p.get("chains", []),
                    "change_1d_pct": (round(change_1d, 2) if change_1d is not None else None),
                    "change_7d_pct": (round(change_7d, 2) if change_7d is not None else None),
                }
            )

        total_tvl = sum(p.get("tvl") or 0 for p in data)
        summary = (
            f"Top {len(results)} DeFi protocols by TVL"
            f"{f' on {chain_filter}' if chain_filter else ''}"
            f"{f' in {category_filter}' if category_filter else ''}"
            f". Total TVL: ${total_tvl:,.0f}"
        )

        self._persist_entities(results)

        return ToolResult(
            success=True,
            output=summary,
            data={
                "protocols": results,
                "total_tvl": round(total_tvl, 2),
                "count": len(results),
            },
        )

    def _stablecoins(self, limit: int) -> ToolResult:
        raw = self._fetch_json(_STABLECOINS_URL)
        if raw is None:
            return ToolResult(success=False, output="Failed to fetch stablecoin data.")

        coins = raw.get("peggedAssets", [])
        # Sort by circulating supply descending
        coins.sort(
            key=lambda c: (c.get("circulating") or {}).get("peggedUSD") or 0,
            reverse=True,
        )
        top = coins[:limit]

        results = []
        for c in top:
            circ = (c.get("circulating") or {}).get("peggedUSD") or 0
            results.append(
                {
                    "name": c.get("name"),
                    "symbol": c.get("symbol"),
                    "peg_mechanism": c.get("pegMechanism"),
                    "circulating_usd": round(circ, 2),
                    "chains": c.get("chains", []),
                }
            )

        total_supply = sum((c.get("circulating") or {}).get("peggedUSD") or 0 for c in coins)
        summary = f"Top {len(results)} stablecoins by circulating supply. Total stablecoin supply: ${total_supply:,.0f}"
        return ToolResult(
            success=True,
            output=summary,
            data={
                "stablecoins": results,
                "total_supply": round(total_supply, 2),
                "count": len(results),
            },
        )

    def _dex_volume(self, limit: int, chain_filter: str) -> ToolResult:
        raw = self._fetch_json(_DEXS_URL)
        if raw is None:
            return ToolResult(success=False, output="Failed to fetch DEX volume data.")

        protocols = raw.get("protocols", [])
        if chain_filter:
            cf_lower = chain_filter.lower()
            protocols = [p for p in protocols if cf_lower in [c.lower() for c in (p.get("chains") or [])]]

        # Sort by 24h volume descending
        protocols.sort(key=lambda p: p.get("totalVolume24h") or 0, reverse=True)
        top = protocols[:limit]

        results = []
        for p in top:
            vol = p.get("totalVolume24h") or 0
            results.append(
                {
                    "name": p.get("name"),
                    "volume_24h_usd": round(vol, 2) if vol else None,
                    "change_1d_pct": p.get("change_1d"),
                    "change_7d_pct": p.get("change_7d"),
                    "chains": p.get("chains", []),
                }
            )

        total_vol = raw.get("totalVolume", 0) or 0
        summary = (
            f"Top {len(results)} DEXes by 24h volume"
            f"{f' on {chain_filter}' if chain_filter else ''}"
            f". Total 24h DEX volume: ${total_vol:,.0f}"
        )
        return ToolResult(
            success=True,
            output=summary,
            data={
                "dexes": results,
                "total_volume_24h": round(total_vol, 2),
                "count": len(results),
            },
        )

    def _chain_tvl(self, limit: int) -> ToolResult:
        data = self._fetch_json(_PROTOCOLS_URL)
        if data is None:
            return ToolResult(success=False, output="Failed to fetch protocol data.")

        # Aggregate TVL by chain
        chain_totals: dict[str, float] = {}
        chain_counts: dict[str, int] = {}
        for p in data:
            tvl = p.get("tvl") or 0
            chains = p.get("chains") or []
            chain_tvls = p.get("chainTvls") or {}
            for chain in chains:
                chain_val = chain_tvls.get(chain) or 0
                chain_totals[chain] = chain_totals.get(chain, 0) + chain_val
                chain_counts[chain] = chain_counts.get(chain, 0) + 1

        # Sort by total TVL
        sorted_chains = sorted(chain_totals.items(), key=lambda x: x[1], reverse=True)
        top = sorted_chains[:limit]

        results = []
        for chain_name, tvl in top:
            results.append(
                {
                    "chain": chain_name,
                    "tvl_usd": round(tvl, 2),
                    "protocol_count": chain_counts.get(chain_name, 0),
                }
            )

        grand_total = sum(v for _, v in sorted_chains)
        summary = f"Top {len(results)} chains by TVL. Grand total across all chains: ${grand_total:,.0f}"
        return ToolResult(
            success=True,
            output=summary,
            data={
                "chains": results,
                "grand_total_tvl": round(grand_total, 2),
                "count": len(results),
            },
        )

    def _tvl_history(
        self,
        limit: int,
        days_back: int,
        chain_filter: str,
        category_filter: str,
    ) -> ToolResult:
        """Fetch daily TVL history for top N protocols.

        Calls /protocol/{slug} for each of the top `limit` protocols and
        writes one observation per day into the store. DeFiLlama provides
        daily data going back to protocol creation (typically 2–5 years).
        This is the key endpoint for building multi-year protocol density.
        """
        protocols_data = self._fetch_json(_PROTOCOLS_URL)
        if protocols_data is None:
            return ToolResult(success=False, output="Failed to fetch protocol list.")

        if chain_filter:
            cf_lower = chain_filter.lower()
            protocols_data = [
                p for p in protocols_data
                if cf_lower in [c.lower() for c in (p.get("chains") or [])]
            ]
        if category_filter:
            cat_lower = category_filter.lower()
            protocols_data = [
                p for p in protocols_data
                if (p.get("category") or "").lower() == cat_lower
            ]

        protocols_data.sort(key=lambda p: p.get("tvl") or 0, reverse=True)
        top_protocols = protocols_data[:limit]

        cutoff_ts = time.time() - days_back * 86400
        total_obs = 0
        protocols_processed = 0

        for proto in top_protocols:
            slug = proto.get("slug") or proto.get("name", "").lower().replace(" ", "-")
            if not slug:
                continue

            hist_url = _PROTOCOL_HISTORY_URL.format(slug=slug)
            try:
                hist = self._fetch_json(hist_url)
            except Exception as exc:
                log.debug("History fetch failed for %s: %s", slug, exc)
                continue

            if not isinstance(hist, dict):
                continue

            tvl_series = hist.get("tvl") or []
            name = hist.get("name") or proto.get("name") or slug
            category = hist.get("category") or proto.get("category") or ""
            chains = hist.get("chains") or []

            if self._store is not None and entity_id_from_key is not None:
                protocol_eid = entity_id_from_key("protocol", name.lower())
                self._store.register_entity(
                    entity_type="protocol",
                    canonical_name=name,
                    entity_id=protocol_eid,
                )
                self._store.add_entity_alias(protocol_eid, "protocol_name", name)

                for entry in tvl_series:
                    ts = entry.get("date") or 0
                    tvl_usd = entry.get("totalLiquidityUSD") or 0.0
                    if ts < cutoff_ts:
                        continue
                    try:
                        self._store.store_entity_observation(
                            entity_id=protocol_eid,
                            source_tool="defi_flows",
                            observed_at=float(ts),
                            observation_type="tvl_change",
                            depth_level=2,
                            value={
                                "tvl_usd": round(float(tvl_usd), 2),
                                "category": category,
                                "chains": chains,
                                "change_1d_pct": None,
                                "change_7d_pct": None,
                            },
                        )
                        total_obs += 1
                    except Exception:
                        log.debug("Obs write failed for %s ts=%s", name, ts)

            protocols_processed += 1

        return ToolResult(
            success=True,
            output=(
                f"DeFiLlama historical TVL: {protocols_processed} protocols, "
                f"{total_obs} daily observations (last {days_back}d)."
            ),
            data={"protocols_processed": protocols_processed, "observations_written": total_obs},
        )

    def _fetch_json(self, url: str) -> Any:
        """Fetch JSON from DefiLlama with caching."""
        cache_params = {"url": url}
        if self._cache:
            cached = self._cache.get("defi_flows", cache_params)
            if cached is not None:
                return cached

        with httpx.Client(
            timeout=_TIMEOUT,
            headers={"User-Agent": _UA},
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()

        if self._cache and data is not None:
            self._cache.put("defi_flows", cache_params, data)
        return data
