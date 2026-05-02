"""
TirraMind — Polymarket Whale Tracking Pipeline

Two DAGs that track smart money on Polymarket:

  whale_tracking (every 15 min):
      fetch_recent_trades → index_trades → detect_signals

  whale_scoring (daily 06:00 UTC):
      track_resolutions → score_wallets

All functions follow the FunctionOperator contract:
    fn(params: dict, upstream_results: dict) -> dict

Data source: data-api.polymarket.com (public, $0, no auth).
Resolution data: gamma-api.polymarket.com/events?closed=true.
Storage: PipelineStore (pipeline_data + signals tables).
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from typing import Any

import httpx

from agent.data.dns_bypass import ensure_polymarket_dns
from agent.pipeline.dag import DAG
from agent.pipeline.store import PipelineStore

ensure_polymarket_dns()

log = logging.getLogger(__name__)

_DATA_API = "https://data-api.polymarket.com"
_GAMMA_API = "https://gamma-api.polymarket.com"

# 15-min BTC/ETH up/down noise markets — filtered by default
_MICRO_RE = re.compile(r"\bUp or Down\b", re.IGNORECASE)

# Wallet scoring constants
_MIN_RESOLVED_TRADES = 5  # minimum resolved trades to score a wallet
_RECENCY_HALF_LIFE_DAYS = 30  # exponential decay half-life
_LN2 = math.log(2)
_TOP_WALLETS_STORED = 500

# Signal detection constants
_CONSENSUS_MIN_WALLETS = 3  # min top wallets on same side
_CONSENSUS_TOP_K = 100  # wallets considered for consensus
_WHALE_ALERT_TOP_K = 50  # wallets considered for whale alerts
_WHALE_ALERT_MIN_USDC = 1000  # minimum USDC for whale alert
_CONTRARIAN_LOW = 0.30  # price below this = underdog
_CONTRARIAN_HIGH = 0.70  # price above this = favorite
_SIGNAL_LOOKBACK_H = 24  # hours of recent trades for signal detection


# ═══════════════════════════════════════════════════════════════
#  Pipeline Functions
# ═══════════════════════════════════════════════════════════════


def fetch_recent_trades(params: dict, upstream: dict) -> dict:
    """Fetch latest trades from Polymarket data-api.

    Returns {"trades": [...], "count": int} or {"trades": [], "error": ...}.
    """
    skip_micro = params.get("skip_micro", True)

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{_DATA_API}/trades", params={"limit": 1000})
            resp.raise_for_status()
            raw = resp.json()
    except Exception as exc:
        log.error("Failed to fetch trades from data-api: %s", exc)
        return {"trades": [], "count": 0, "error": str(exc)}

    if not isinstance(raw, list):
        return {
            "trades": [],
            "count": 0,
            "error": f"Unexpected response type: {type(raw).__name__}",
        }

    trades = []
    for t in raw:
        title = t.get("title") or ""
        if skip_micro and _MICRO_RE.search(title):
            continue

        tx_hash = t.get("transactionHash") or ""
        wallet = t.get("proxyWallet") or ""
        if not tx_hash or not wallet:
            continue

        size = _to_float(t.get("size"))
        price = _to_float(t.get("price"))

        trades.append(
            {
                "tx_hash": tx_hash,
                "wallet": wallet,
                "condition_id": t.get("conditionId") or "",
                "side": t.get("side") or "",
                "size": size,
                "price": price,
                "timestamp": _to_float(t.get("timestamp")),
                "outcome": t.get("outcome") or "",
                "outcome_index": t.get("outcomeIndex"),
                "title": title,
                "slug": t.get("slug") or "",
                "event_slug": t.get("eventSlug") or "",
                "usdc_value": round(size * price, 4),
                "name": t.get("name") or "",
                "pseudonym": t.get("pseudonym") or "",
            }
        )

    return {"trades": trades, "count": len(trades)}


def index_trades(params: dict, upstream: dict) -> dict:
    """Deduplicate and persist trades to pipeline_data.

    Reads upstream fetch_recent_trades output. Deduplicates by tx_hash
    against recent DB entries. Stores each new trade individually.

    Returns {"indexed": int, "duplicates": int, "total_seen": int}.
    """
    fetch_result = upstream.get("fetch_recent_trades", {})
    trades = fetch_result.get("trades", [])

    if not trades:
        return {"indexed": 0, "duplicates": 0, "total_seen": 0}

    db_path = params.get("db_path", ".tirra_pipeline/pipeline.db")
    store = PipelineStore(db_path)
    try:
        # Build dedup set from last 2h of stored trades
        seen = _recent_tx_hashes(store, hours=2)

        indexed = 0
        for trade in trades:
            tx_hash = trade["tx_hash"]
            if tx_hash in seen:
                continue

            store.store_data(
                source="pm_trades",
                params={
                    "tx_hash": tx_hash,
                    "wallet": trade["wallet"],
                    "condition_id": trade["condition_id"],
                },
                data=trade,
            )
            seen.add(tx_hash)
            indexed += 1

        duplicates = len(trades) - indexed
        return {"indexed": indexed, "duplicates": duplicates, "total_seen": len(trades)}
    finally:
        store.close()


def track_resolutions(params: dict, upstream: dict) -> dict:
    """Fetch resolved markets from Gamma API and persist outcomes.

    Stores in pipeline_data with source="pm_resolutions".
    Deduplicates by condition_id.

    Returns {"resolved": int, "new": int}.
    """
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(
                f"{_GAMMA_API}/events",
                params={"closed": "true", "limit": "100"},
            )
            resp.raise_for_status()
            events = resp.json()
    except Exception as exc:
        log.error("Failed to fetch resolved events: %s", exc)
        return {"resolved": 0, "new": 0, "error": str(exc)}

    if not isinstance(events, list):
        return {
            "resolved": 0,
            "new": 0,
            "error": f"Unexpected type: {type(events).__name__}",
        }

    db_path = params.get("db_path", ".tirra_pipeline/pipeline.db")
    store = PipelineStore(db_path)
    try:
        # Build dedup set of already-stored condition_ids
        existing = store.query_data("pm_resolutions", limit=5000)
        known_cids = {row["params"].get("condition_id") for row in existing}

        total = 0
        new = 0
        for event in events:
            for mkt in event.get("markets", []):
                condition_id = mkt.get("conditionId") or ""
                if not condition_id:
                    continue

                winning_index = _resolve_winner(mkt)
                if winning_index is None:
                    continue  # ambiguous — skip

                total += 1
                if condition_id in known_cids:
                    continue

                store.store_data(
                    source="pm_resolutions",
                    params={"condition_id": condition_id},
                    data={
                        "condition_id": condition_id,
                        "winning_index": winning_index,
                        "title": mkt.get("question") or event.get("title", ""),
                        "slug": mkt.get("slug") or "",
                    },
                )
                known_cids.add(condition_id)
                new += 1

        return {"resolved": total, "new": new}
    finally:
        store.close()


def score_wallets(params: dict, upstream: dict) -> dict:
    """Score wallets by accuracy on resolved markets.

    Reads pm_trades and pm_resolutions from pipeline DB. For each wallet
    with >= MIN_RESOLVED_TRADES resolved calls, computes:
      - Bayesian accuracy: (correct + 1) / (total + 2)
      - Profit factor: winning_pnl / max(|losing_pnl|, 0.01)
      - Composite: accuracy * log(1+volume) * recency * sqrt(markets)

    Stores top-500 in pipeline_data source="pm_wallet_scores".

    Returns {"scored": int, "top_10": [...]}.
    """
    db_path = params.get("db_path", ".tirra_pipeline/pipeline.db")
    store = PipelineStore(db_path)
    try:
        # Load resolution map: condition_id → winning_index
        resolutions = store.query_data("pm_resolutions", limit=10000)
        res_map: dict[str, int] = {}
        for row in resolutions:
            cid = row["params"].get("condition_id", "")
            widx = row["data"].get("winning_index")
            if cid and widx is not None:
                res_map[cid] = widx

        if not res_map:
            return {"scored": 0, "top_10": [], "note": "no resolutions available"}

        # Load all trades
        all_trades = store.query_data("pm_trades", limit=100000)

        # Group trades by wallet
        wallet_trades: dict[str, list[dict]] = {}
        for row in all_trades:
            trade = row["data"]
            w = trade.get("wallet", "")
            if w:
                wallet_trades.setdefault(w, []).append(trade)

        # Score each wallet
        now = time.time()
        scores: list[dict[str, Any]] = []

        for wallet, trades in wallet_trades.items():
            correct = 0
            total_resolved = 0
            winning_pnl = 0.0
            losing_pnl = 0.0
            total_volume = 0.0
            markets_seen: set[str] = set()
            latest_ts = 0.0

            for t in trades:
                cid = t.get("condition_id", "")
                usdc = _to_float(t.get("usdc_value"))
                total_volume += usdc
                markets_seen.add(cid)

                ts = _to_float(t.get("timestamp"))
                if ts > latest_ts:
                    latest_ts = ts

                # Only score against resolved markets
                if cid not in res_map:
                    continue

                total_resolved += 1
                side = t.get("side", "")
                oidx = t.get("outcome_index")
                winning_idx = res_map[cid]

                is_correct = (side == "BUY" and oidx == winning_idx) or (side == "SELL" and oidx != winning_idx)

                price = _to_float(t.get("price"))
                size = _to_float(t.get("size"))

                if is_correct:
                    correct += 1
                    winning_pnl += size * (1.0 - price)  # bought at price, settled at 1.0
                else:
                    losing_pnl += size * price  # bought at price, settled at 0.0

            if total_resolved < _MIN_RESOLVED_TRADES:
                continue

            # Bayesian accuracy: Beta(1,1) prior (Laplace smoothing)
            accuracy = (correct + 1) / (total_resolved + 2)

            # Profit factor
            profit_factor = winning_pnl / max(abs(losing_pnl), 0.01)

            # Recency: exp(-ln(2)/30 * days_since_last)
            days_since = max((now - latest_ts) / 86400.0, 0.0)
            recency = math.exp(-_LN2 / _RECENCY_HALF_LIFE_DAYS * days_since)

            # Composite score
            n_markets = len(markets_seen)
            composite = accuracy * math.log(1 + total_volume) * recency * math.sqrt(n_markets)

            scores.append(
                {
                    "wallet": wallet,
                    "composite": round(composite, 4),
                    "accuracy": round(accuracy, 4),
                    "correct": correct,
                    "total_resolved": total_resolved,
                    "profit_factor": round(profit_factor, 4),
                    "total_volume": round(total_volume, 2),
                    "markets": n_markets,
                    "recency": round(recency, 4),
                    "latest_trade_ts": latest_ts,
                }
            )

        # Sort by composite descending, take top-N
        scores.sort(key=lambda s: s["composite"], reverse=True)
        top = scores[:_TOP_WALLETS_STORED]

        # Persist wallet scores (replace previous batch)
        store.store_data(
            source="pm_wallet_scores",
            params={"batch": "latest"},
            data={"wallets": top, "scored_at": now, "total_scored": len(top)},
        )

        top_10 = [
            {
                "wallet": s["wallet"][:10] + "...",
                "score": s["composite"],
                "accuracy": s["accuracy"],
                "volume": s["total_volume"],
            }
            for s in top[:10]
        ]

        return {"scored": len(top), "top_10": top_10}
    finally:
        store.close()


def detect_signals(params: dict, upstream: dict) -> dict:
    """Detect whale signals from recent trades and wallet scores.

    Signals emitted:
      pm_whale_alert   — large trade by top-scored wallet
      pm_consensus     — 3+ top wallets on same side of same market
      pm_contrarian    — top wallet bets against market price

    Returns {"signals_emitted": int, "whale_alerts": [...],
             "consensus": [...], "contrarian": [...]}.
    """
    db_path = params.get("db_path", ".tirra_pipeline/pipeline.db")
    store = PipelineStore(db_path)
    try:
        # Load wallet scores
        score_rows = store.query_data("pm_wallet_scores", limit=1)
        if not score_rows:
            return {
                "signals_emitted": 0,
                "whale_alerts": [],
                "consensus": [],
                "contrarian": [],
                "note": "no wallet scores yet",
            }

        wallet_data = score_rows[0]["data"]
        wallets_list = wallet_data.get("wallets", [])

        # Build score lookup: wallet → score dict
        score_map: dict[str, dict] = {}
        for ws in wallets_list:
            score_map[ws["wallet"]] = ws

        top_k_wallets = {ws["wallet"] for ws in wallets_list[:_CONSENSUS_TOP_K]}
        top_alert_wallets = {ws["wallet"] for ws in wallets_list[:_WHALE_ALERT_TOP_K]}

        # Load recent trades
        since = time.time() - _SIGNAL_LOOKBACK_H * 3600
        recent_rows = store.query_data("pm_trades", since=since, limit=10000)
        recent_trades = [row["data"] for row in recent_rows]

        whale_alerts: list[dict] = []
        consensus_signals: list[dict] = []
        contrarian_signals: list[dict] = []

        # ── Whale Alerts ───────────────────────────────────
        for t in recent_trades:
            wallet = t.get("wallet", "")
            usdc = _to_float(t.get("usdc_value"))
            if wallet in top_alert_wallets and usdc >= _WHALE_ALERT_MIN_USDC:
                ws = score_map.get(wallet, {})
                alert = {
                    "wallet": wallet,
                    "market": t.get("title", ""),
                    "condition_id": t.get("condition_id", ""),
                    "side": t.get("side", ""),
                    "usdc_value": usdc,
                    "score": ws.get("composite", 0),
                    "accuracy": ws.get("accuracy", 0),
                }
                whale_alerts.append(alert)
                store.store_signal(
                    signal_name="pm_whale_alert",
                    value=usdc,
                    metadata=alert,
                )

        # ── Smart Money Consensus ──────────────────────────
        # Group recent trades by (condition_id, side) for top-K wallets
        market_sides: dict[tuple[str, str], list[dict]] = {}
        for t in recent_trades:
            wallet = t.get("wallet", "")
            if wallet not in top_k_wallets:
                continue
            cid = t.get("condition_id", "")
            side = t.get("side", "")
            if cid and side:
                key = (cid, side)
                market_sides.setdefault(key, []).append(t)

        for (cid, side), group_trades in market_sides.items():
            unique_wallets = {t["wallet"] for t in group_trades}
            if len(unique_wallets) < _CONSENSUS_MIN_WALLETS:
                continue

            # Weighted confidence: sum(score * usdc) / sum(usdc)
            total_usdc = 0.0
            weighted_score = 0.0
            prices = []
            for t in group_trades:
                usdc = _to_float(t.get("usdc_value"))
                ws = score_map.get(t["wallet"], {})
                weighted_score += ws.get("composite", 0) * usdc
                total_usdc += usdc
                prices.append(_to_float(t.get("price")))

            confidence = weighted_score / max(total_usdc, 0.01)
            avg_price = sum(prices) / max(len(prices), 1)

            signal = {
                "condition_id": cid,
                "market": group_trades[0].get("title", ""),
                "side": side,
                "wallet_count": len(unique_wallets),
                "total_usdc": round(total_usdc, 2),
                "weighted_confidence": round(confidence, 4),
                "avg_price": round(avg_price, 4),
            }
            consensus_signals.append(signal)
            store.store_signal(
                signal_name="pm_consensus",
                value=confidence,
                metadata=signal,
            )

        # ── Contrarian Smart Money ─────────────────────────
        for t in recent_trades:
            wallet = t.get("wallet", "")
            if wallet not in top_alert_wallets:
                continue

            price = _to_float(t.get("price"))
            side = t.get("side", "")
            # Contrarian: buying YES on underdog or NO on favorite
            is_contrarian = (side == "BUY" and price < _CONTRARIAN_LOW) or (side == "SELL" and price > _CONTRARIAN_HIGH)
            if not is_contrarian:
                continue

            ws = score_map.get(wallet, {})
            signal = {
                "wallet": wallet,
                "market": t.get("title", ""),
                "condition_id": t.get("condition_id", ""),
                "side": side,
                "price": price,
                "score": ws.get("composite", 0),
                "accuracy": ws.get("accuracy", 0),
            }
            contrarian_signals.append(signal)
            store.store_signal(
                signal_name="pm_contrarian",
                value=ws.get("composite", 0),
                metadata=signal,
            )

        total_signals = len(whale_alerts) + len(consensus_signals) + len(contrarian_signals)
        return {
            "signals_emitted": total_signals,
            "whale_alerts": whale_alerts,
            "consensus": consensus_signals,
            "contrarian": contrarian_signals,
        }
    finally:
        store.close()


# ═══════════════════════════════════════════════════════════════
#  DAG Builders
# ═══════════════════════════════════════════════════════════════


def build_whale_tracking_dag(db_path: str = ".tirra_pipeline/pipeline.db") -> DAG:
    """Build the whale_tracking DAG: fetch → index → detect.

    Schedule: every 15 minutes.
    """
    dag = DAG(
        name="whale_tracking",
        schedule="*/15 * * * *",
        description="Polymarket whale tracking: fetch trades, index, detect signals",
    )

    dag.add(
        "fetch_recent_trades",
        operator=fetch_recent_trades,
        params={"skip_micro": True},
        timeout=60,
        retries=2,
        store_result=False,
    )

    dag.add(
        "index_trades",
        operator=index_trades,
        params={"db_path": db_path},
        depends_on=["fetch_recent_trades"],
        timeout=30,
        retries=1,
        store_result=False,
    )

    dag.add(
        "detect_signals",
        operator=detect_signals,
        params={"db_path": db_path},
        depends_on=["index_trades"],
        timeout=30,
        retries=1,
        store_result=False,
    )

    return dag


def build_whale_scoring_dag(db_path: str = ".tirra_pipeline/pipeline.db") -> DAG:
    """Build the whale_scoring DAG: resolve → score.

    Schedule: daily at 06:00 UTC.
    """
    dag = DAG(
        name="whale_scoring",
        schedule="0 6 * * *",
        description="Polymarket whale scoring: resolve markets, compute wallet scores",
    )

    dag.add(
        "track_resolutions",
        operator=track_resolutions,
        params={"db_path": db_path},
        timeout=60,
        retries=2,
        store_result=False,
    )

    dag.add(
        "score_wallets",
        operator=score_wallets,
        params={"db_path": db_path},
        depends_on=["track_resolutions"],
        timeout=120,
        retries=1,
        store_result=False,
    )

    return dag


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════


def _to_float(val: Any) -> float:
    """Convert to float, returning 0.0 on failure."""
    if val is None:
        return 0.0
    try:
        f = float(val)
        return f if f == f else 0.0  # NaN → 0
    except (ValueError, TypeError):
        return 0.0


def _recent_tx_hashes(store: PipelineStore, hours: int = 2) -> set[str]:
    """Get tx_hashes from recent pm_trades for deduplication."""
    since = time.time() - hours * 3600
    rows = store.query_data("pm_trades", since=since, limit=5000)
    return {row["params"].get("tx_hash", "") for row in rows} - {""}


def _resolve_winner(market: dict) -> int | None:
    """Determine the winning outcome index from a resolved Gamma market.

    Returns the index of the outcome with price nearest 1.0, or None
    if the market isn't clearly resolved (ambiguous prices).
    """
    prices_raw = market.get("outcomePrices", "")
    if not prices_raw:
        return None

    try:
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        if not isinstance(prices, list) or len(prices) < 2:
            return None

        float_prices = [float(p) for p in prices]
    except (json.JSONDecodeError, ValueError, TypeError):
        return None

    # The winning outcome has price nearest 1.0
    best_idx = -1
    best_dist = 2.0
    for i, p in enumerate(float_prices):
        dist = abs(1.0 - p)
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    # Require the winner to be clearly resolved (price > 0.9)
    if best_idx >= 0 and float_prices[best_idx] > 0.9:
        return best_idx

    return None
