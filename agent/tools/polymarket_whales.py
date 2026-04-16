"""
Tool: Polymarket Whales — Smart Money Tracking

Queries whale activity on Polymarket: top-scored wallets, wallet detail,
market-level whale positions, and recent smart-money signals.

Reads from Pipeline DB (populated by whale_tracking and whale_scoring DAGs).
Falls back to live data-api query if DB is cold (no scores yet).

Modes:
  top_wallets    — top-N scored wallets with accuracy, volume, activity
  wallet_detail  — specific wallet's score, P&L, trade history
  market_whales  — whale activity on a specific market
  recent_signals — recent smart-money signals (consensus, whale alerts, contrarian)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from agent.data.dns_bypass import ensure_polymarket_dns
from agent.pipeline.store import PipelineStore
from agent.tools.base import Tool, ToolResult

try:
    from agent.pipeline.entity import entity_id_from_key
except ImportError:  # pragma: no cover
    entity_id_from_key = None  # type: ignore[assignment]

ensure_polymarket_dns()

log = logging.getLogger(__name__)

_DATA_API = "https://data-api.polymarket.com"

_VALID_MODES = {"top_wallets", "wallet_detail", "market_whales", "recent_signals"}
_VALID_SIGNAL_TYPES = {"consensus", "whale_alert", "contrarian", "all"}


class PolymarketWhalesTool(Tool):

    name = "polymarket_whales"

    description = (
        "Query smart-money activity on Polymarket. Shows top-scored wallets "
        "(ranked by accuracy on resolved markets), individual wallet detail, "
        "whale activity on specific markets, and recent smart-money signals "
        "(consensus bets, whale alerts, contrarian moves). Use this to see "
        "what the most accurate prediction market traders are betting on."
    )

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "description": (
                    "Query mode. Options: top_wallets (ranked by score), "
                    "wallet_detail (specific wallet), market_whales (whale "
                    "activity on a market), recent_signals (smart-money signals)."
                ),
                "enum": [
                    "top_wallets",
                    "wallet_detail",
                    "market_whales",
                    "recent_signals",
                ],
            },
            "wallet": {
                "type": "string",
                "description": "Wallet address (0x...) for wallet_detail mode.",
            },
            "market": {
                "type": "string",
                "description": (
                    "Market condition ID or search term for market_whales mode."
                ),
            },
            "signal_type": {
                "type": "string",
                "description": (
                    "Filter signals: consensus, whale_alert, contrarian, or all. "
                    "Default: all."
                ),
                "default": "all",
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return. Default: 10.",
                "default": 10,
            },
        },
        "required": ["mode"],
    }

    def __init__(self, db_path: str = ".tirra_pipeline/pipeline.db") -> None:
        self._db_path = db_path

    def execute(
        self,
        *,
        mode: str = "top_wallets",
        wallet: str = "",
        market: str = "",
        signal_type: str = "all",
        limit: int = 10,
        **_: Any,
    ) -> ToolResult:
        mode = mode.lower().strip()
        if mode not in _VALID_MODES:
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Must be one of: {', '.join(sorted(_VALID_MODES))}",
            )
        limit = max(1, min(limit, 100))

        try:
            if mode == "top_wallets":
                return self._top_wallets(limit)
            elif mode == "wallet_detail":
                return self._wallet_detail(wallet, limit)
            elif mode == "market_whales":
                return self._market_whales(market, limit)
            elif mode == "recent_signals":
                return self._recent_signals(signal_type, limit)
        except Exception as exc:
            log.exception("polymarket_whales error in mode=%s", mode)
            return ToolResult(success=False, output=f"Error: {exc}")

        return ToolResult(success=False, output=f"Unhandled mode: {mode}")

    # ── Modes ──────────────────────────────────────────────

    def _top_wallets(self, limit: int) -> ToolResult:
        store = PipelineStore(self._db_path)
        try:
            rows = store.query_data("pm_wallet_scores", limit=1)
            if not rows:
                return self._cold_start_top_wallets(limit)

            wallets = rows[0]["data"].get("wallets", [])[:limit]
            if not wallets:
                return ToolResult(
                    success=True, output="No scored wallets yet.", data={"wallets": []}
                )

            # L2: persist wallet entities
            try:
                self._persist_wallet_entities(wallets, store)
            except Exception:
                log.exception("Whale wallet persistence failed (non-fatal)")

            lines = [f"Top {len(wallets)} Polymarket wallets by composite score:\n"]
            for i, w in enumerate(wallets, 1):
                addr = w["wallet"][:10] + "..." + w["wallet"][-4:]
                lines.append(
                    f"  {i}. {addr}  score={w['composite']:.2f}  "
                    f"accuracy={w['accuracy']:.1%}  "
                    f"resolved={w['total_resolved']}  "
                    f"vol=${w['total_volume']:,.0f}  "
                    f"markets={w['markets']}"
                )

            return ToolResult(
                success=True,
                output="\n".join(lines),
                data={"wallets": wallets},
            )
        finally:
            store.close()

    def _wallet_detail(self, wallet: str, limit: int) -> ToolResult:
        if not wallet or not wallet.startswith("0x"):
            return ToolResult(
                success=False,
                output="wallet_detail mode requires a valid wallet address (0x...).",
            )

        store = PipelineStore(self._db_path)
        try:
            # Find wallet in scores
            score_rows = store.query_data("pm_wallet_scores", limit=1)
            wallet_score = None
            if score_rows:
                for ws in score_rows[0]["data"].get("wallets", []):
                    if ws["wallet"].lower() == wallet.lower():
                        wallet_score = ws
                        break

            # Get recent trades for this wallet
            recent = store.query_data("pm_trades", limit=5000)
            wallet_trades = [
                r["data"]
                for r in recent
                if r["data"].get("wallet", "").lower() == wallet.lower()
            ][:limit]

            if not wallet_score and not wallet_trades:
                return ToolResult(
                    success=False,
                    output=f"Wallet {wallet} not found in tracked data.",
                )

            lines = [f"Wallet: {wallet}\n"]
            if wallet_score:
                lines.append(
                    f"  Score: {wallet_score['composite']:.2f}  "
                    f"Accuracy: {wallet_score['accuracy']:.1%}  "
                    f"({wallet_score['correct']}/{wallet_score['total_resolved']} resolved)  "
                    f"Profit Factor: {wallet_score['profit_factor']:.2f}"
                )
                lines.append(
                    f"  Volume: ${wallet_score['total_volume']:,.0f}  "
                    f"Markets: {wallet_score['markets']}  "
                    f"Recency: {wallet_score['recency']:.2f}"
                )

            if wallet_trades:
                lines.append(f"\nRecent trades ({len(wallet_trades)}):")
                for t in wallet_trades[:limit]:
                    lines.append(
                        f"  {t.get('side', '?')} {t.get('outcome', '?')} @ "
                        f"{t.get('price', 0):.2f} — ${t.get('usdc_value', 0):,.2f} — "
                        f"{t.get('title', '')[:50]}"
                    )

            return ToolResult(
                success=True,
                output="\n".join(lines),
                data={"score": wallet_score, "trades": wallet_trades},
            )
        finally:
            store.close()

    def _market_whales(self, market: str, limit: int) -> ToolResult:
        if not market:
            return ToolResult(
                success=False,
                output="market_whales mode requires a market condition ID or search term.",
            )

        store = PipelineStore(self._db_path)
        try:
            recent = store.query_data("pm_trades", limit=10000)
            search_lower = market.lower()

            # Match by condition_id or title substring
            matched = [
                r["data"]
                for r in recent
                if r["data"].get("condition_id", "").lower() == search_lower
                or search_lower in (r["data"].get("title", "").lower())
            ]

            if not matched:
                return ToolResult(
                    success=True,
                    output=f"No whale trades found for market '{market}'.",
                    data={"trades": []},
                )

            # Load scores to rank wallets
            score_rows = store.query_data("pm_wallet_scores", limit=1)
            score_map: dict[str, dict] = {}
            if score_rows:
                for ws in score_rows[0]["data"].get("wallets", []):
                    score_map[ws["wallet"]] = ws

            # Group by wallet, sort by score
            wallet_agg: dict[str, dict] = {}
            for t in matched:
                w = t.get("wallet", "")
                if w not in wallet_agg:
                    wallet_agg[w] = {
                        "wallet": w,
                        "score": score_map.get(w, {}).get("composite", 0),
                        "accuracy": score_map.get(w, {}).get("accuracy", 0),
                        "trades": [],
                        "total_usdc": 0.0,
                    }
                wallet_agg[w]["trades"].append(t)
                wallet_agg[w]["total_usdc"] += _to_float(t.get("usdc_value"))

            ranked = sorted(
                wallet_agg.values(), key=lambda x: x["score"], reverse=True
            )[:limit]

            title = matched[0].get("title", market)
            lines = [f"Whale activity on: {title}\n"]
            for i, wa in enumerate(ranked, 1):
                addr = wa["wallet"][:10] + "..." + wa["wallet"][-4:]
                sides = {t["side"] for t in wa["trades"]}
                lines.append(
                    f"  {i}. {addr}  score={wa['score']:.2f}  "
                    f"accuracy={wa['accuracy']:.1%}  "
                    f"side={'|'.join(sides)}  "
                    f"${wa['total_usdc']:,.0f}  "
                    f"({len(wa['trades'])} trades)"
                )

            return ToolResult(
                success=True,
                output="\n".join(lines),
                data={"market": title, "whales": ranked},
            )
        finally:
            store.close()

    def _recent_signals(self, signal_type: str, limit: int) -> ToolResult:
        signal_type = signal_type.lower().strip()
        if signal_type not in _VALID_SIGNAL_TYPES:
            return ToolResult(
                success=False,
                output=f"Invalid signal_type '{signal_type}'. Must be one of: {', '.join(sorted(_VALID_SIGNAL_TYPES))}",
            )

        store = PipelineStore(self._db_path)
        try:
            signals: list[dict] = []
            types_to_query = (
                ["pm_whale_alert", "pm_consensus", "pm_contrarian"]
                if signal_type == "all"
                else [f"pm_{signal_type}"]
            )

            for stype in types_to_query:
                rows = store.query_signals(stype, limit=limit)
                for row in rows:
                    signals.append(
                        {
                            "type": stype.replace("pm_", ""),
                            "value": row.get("value"),
                            "computed_at": row.get("computed_at"),
                            "metadata": row.get("metadata"),
                        }
                    )

            # Sort by time descending
            signals.sort(key=lambda s: s.get("computed_at", 0), reverse=True)
            signals = signals[:limit]

            if not signals:
                return ToolResult(
                    success=True,
                    output="No recent signals found.",
                    data={"signals": []},
                )

            lines = [f"Recent smart-money signals ({len(signals)}):\n"]
            for s in signals:
                meta = s.get("metadata") or {}
                if s["type"] == "whale_alert":
                    lines.append(
                        f"  WHALE: {meta.get('wallet', '?')[:10]}... "
                        f"{meta.get('side', '?')} ${s['value']:,.0f} on "
                        f"{meta.get('market', '')[:40]}"
                    )
                elif s["type"] == "consensus":
                    lines.append(
                        f"  CONSENSUS: {meta.get('wallet_count', 0)} wallets "
                        f"{meta.get('side', '?')} on {meta.get('market', '')[:40]}  "
                        f"conf={s['value']:.2f}"
                    )
                elif s["type"] == "contrarian":
                    lines.append(
                        f"  CONTRARIAN: {meta.get('wallet', '?')[:10]}... "
                        f"{meta.get('side', '?')} @ {meta.get('price', 0):.2f} on "
                        f"{meta.get('market', '')[:40]}"
                    )

            return ToolResult(
                success=True,
                output="\n".join(lines),
                data={"signals": signals},
            )
        finally:
            store.close()

    # ── Entity persistence (L2) ──────────────────────────────

    def _persist_wallet_entities(
        self,
        wallets: list[dict[str, Any]],
        store: PipelineStore,
    ) -> dict[str, int]:
        """Register wallet entities and store whale_trade observations.

        Each wallet with a valid address becomes a wallet entity. A
        ``whale_trade`` observation stores the composite score, accuracy,
        volume, and trade count.

        Returns counts: {wallets, observations}.
        """
        if entity_id_from_key is None:
            return {"wallets": 0, "observations": 0}
        if not wallets:
            return {"wallets": 0, "observations": 0}

        import time as _time

        counts = {"wallets": 0, "observations": 0}
        seen: set[str] = set()

        for w in wallets:
            addr = (w.get("wallet") or "").strip().lower()
            if not addr or not addr.startswith("0x"):
                continue
            if addr in seen:
                continue
            seen.add(addr)

            wallet_eid = entity_id_from_key("wallet", addr)
            store.register_entity(
                entity_type="wallet",
                canonical_name=addr,
                entity_id=wallet_eid,
                metadata={
                    "source": "polymarket_whales",
                    "platform": "polymarket",
                },
            )
            counts["wallets"] += 1

            # Store whale_trade observation with scoring data
            store.store_entity_observation(
                entity_id=wallet_eid,
                source_tool="polymarket_whales",
                observed_at=_time.time(),
                observation_type="whale_trade",
                depth_level=2,
                value={
                    "composite_score": w.get("composite"),
                    "accuracy": w.get("accuracy"),
                    "total_volume": w.get("total_volume"),
                    "total_resolved": w.get("total_resolved"),
                    "markets": w.get("markets"),
                    "profit_factor": w.get("profit_factor"),
                },
            )
            counts["observations"] += 1

        log.info(
            "Polymarket whales L2: %d wallets, %d observations",
            counts["wallets"],
            counts["observations"],
        )
        return counts

    # ── Cold Start ─────────────────────────────────────────

    def _cold_start_top_wallets(self, limit: int) -> ToolResult:
        """Fallback when no scored wallets exist: fetch live trades and aggregate."""
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(f"{_DATA_API}/trades", params={"limit": 1000})
                resp.raise_for_status()
                raw = resp.json()
        except Exception as exc:
            return ToolResult(
                success=False,
                output=f"No scored wallets in DB, and live fallback failed: {exc}",
            )

        if not isinstance(raw, list) or not raw:
            return ToolResult(
                success=True,
                output="No scored wallets yet and no live trades available.",
                data={"wallets": []},
            )

        # Aggregate by wallet: count trades, total volume
        wallet_agg: dict[str, dict] = {}
        for t in raw:
            w = t.get("proxyWallet", "")
            if not w:
                continue
            if w not in wallet_agg:
                wallet_agg[w] = {"wallet": w, "trades": 0, "volume": 0.0}
            wallet_agg[w]["trades"] += 1
            size = _to_float(t.get("size"))
            price = _to_float(t.get("price"))
            wallet_agg[w]["volume"] += size * price

        ranked = sorted(wallet_agg.values(), key=lambda x: x["volume"], reverse=True)[
            :limit
        ]

        lines = [
            f"Top {len(ranked)} wallets by volume (cold start — no accuracy scores yet):\n"
        ]
        for i, w in enumerate(ranked, 1):
            addr = w["wallet"][:10] + "..." + w["wallet"][-4:]
            lines.append(
                f"  {i}. {addr}  trades={w['trades']}  vol=${w['volume']:,.0f}"
            )

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"wallets": ranked, "cold_start": True},
        )


def _to_float(val: Any) -> float:
    """Convert to float, returning 0.0 on failure."""
    if val is None:
        return 0.0
    try:
        f = float(val)
        return f if f == f else 0.0
    except (ValueError, TypeError):
        return 0.0
