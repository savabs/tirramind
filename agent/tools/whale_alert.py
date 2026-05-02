"""
Tool: BTC Whale Transfer Monitoring

Two free data sources (no API key required):
  1. blockchain.com /unconfirmed-transactions — BTC mempool (leading indicator)
  2. blockchain.com /rawblock/{hash} — latest confirmed block (3000+ txs per block)

Detects large BTC transfers in mempool and confirmed blocks.

Cost: $0 — no API key required for either source.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

try:
    from agent.pipeline.entity import entity_id_from_key
except ImportError:  # pragma: no cover — entity module always available
    entity_id_from_key = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

_USER_AGENT = "TirraMind/0.1 (research; https://github.com/tirramind)"
_MEMPOOL_URL = "https://blockchain.info/unconfirmed-transactions?format=json"
_LATEST_BLOCK_URL = "https://blockchain.info/latestblock"
_RAW_BLOCK_URL = "https://blockchain.info/rawblock/{hash}"

_SATS_PER_BTC = 1e8

# Pre-computed BTC-USD instrument entity ID (Phase 30).
# Must match instrument_universe._entity_id("BTC-USD").
_BTC_INSTRUMENT_EID: str | None = None
if entity_id_from_key is not None:
    _BTC_INSTRUMENT_EID = entity_id_from_key("instrument", "BTC-USD")


class WhaleAlertTool(Tool):
    name = "whale_alert"
    description = (
        "Monitor large BTC transfers (whale movements). Two free modes, no API key needed. "
        "'mempool' scans unconfirmed transactions (leading indicator — see whale moves before confirmation). "
        "'confirmed' scans the latest confirmed block (~3000+ txs). "
        "Use min_btc to filter. Returns transactions sorted by value."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["mempool", "confirmed"],
                "default": "mempool",
                "description": (
                    "mempool = BTC unconfirmed transactions (leading indicator). "
                    "confirmed = latest confirmed block (historical, more txs)."
                ),
            },
            "min_btc": {
                "type": "number",
                "default": 10.0,
                "description": "Min BTC value to include (default 10)",
            },
            "limit": {
                "type": "integer",
                "default": 20,
                "description": "Max transactions to return",
            },
        },
    }

    def __init__(
        self,
        cache: DataCache | None = None,
        *,
        pipeline_store: PipelineStore | None = None,
    ) -> None:
        self._cache = cache
        self._store = pipeline_store

    # ── Public execute ───────────────────────────────────────────────

    def execute(
        self,
        *,
        mode: str = "mempool",
        min_btc: float = 10.0,
        limit: int = 20,
        **_: Any,
    ) -> ToolResult:
        if mode not in ("mempool", "confirmed"):
            return ToolResult(
                success=False,
                output=f"Invalid mode '{mode}'. Use 'mempool' or 'confirmed'.",
            )

        try:
            if mode == "mempool":
                txs = self._fetch_mempool(min_btc=min_btc)
            else:
                txs = self._fetch_confirmed(min_btc=min_btc)
        except Exception as exc:
            log.exception("Whale fetch failed")
            return ToolResult(success=False, output=f"Fetch error: {exc}")

        txs = txs[:limit]

        # L2: persist wallet entities + observations when PipelineStore available
        try:
            self._persist_entities(txs)
        except Exception:
            log.exception("Entity persistence failed in execute (non-fatal)")

        summary = self._compute_summary(txs, mode)
        output = self._format_output(txs, summary, mode)

        return ToolResult(
            success=True,
            output=output,
            data={"transactions": txs, "summary": summary, "mode": mode},
        )

    # ── Mempool (BTC unconfirmed) ────────────────────────────────────

    def _fetch_mempool(self, *, min_btc: float) -> list[dict[str, Any]]:
        cache_key = {"source": "blockchain_mempool"}
        if self._cache:
            cached = self._cache.get("whale_alert", cache_key)
            if cached is not None:
                log.debug("Mempool: cache hit")
                return self._filter_txs(cached, min_btc)

        with httpx.Client(timeout=15, follow_redirects=True) as client:
            resp = client.get(_MEMPOOL_URL, headers={"User-Agent": _USER_AGENT})
            resp.raise_for_status()

        data = resp.json()
        raw_txs = data.get("txs", [])
        parsed = self._parse_blockchain_txs(raw_txs, confirmed=False)

        if self._cache and parsed:
            self._cache.put("whale_alert", cache_key, parsed)

        return self._filter_txs(parsed, min_btc)

    # ── Confirmed block (latest BTC block) ───────────────────────────

    def _fetch_confirmed(self, *, min_btc: float) -> list[dict[str, Any]]:
        cache_key = {"source": "blockchain_confirmed"}
        if self._cache:
            cached = self._cache.get("whale_alert", cache_key)
            if cached is not None:
                log.debug("Confirmed block: cache hit")
                return self._filter_txs(cached, min_btc)

        headers = {"User-Agent": _USER_AGENT}

        with httpx.Client(timeout=30, follow_redirects=True) as client:
            # Get latest block hash
            resp = client.get(_LATEST_BLOCK_URL, headers=headers)
            resp.raise_for_status()
            latest = resp.json()
            block_hash = latest.get("hash", "")
            if not block_hash:
                raise RuntimeError("No block hash in latestblock response")

            # Get full block with transactions
            resp2 = client.get(_RAW_BLOCK_URL.format(hash=block_hash), headers=headers)
            resp2.raise_for_status()
            block = resp2.json()

        raw_txs = block.get("tx", [])
        block_height = block.get("height", 0)
        block_time = block.get("time", 0)

        parsed = self._parse_blockchain_txs(raw_txs, confirmed=True, block_height=block_height, block_time=block_time)

        if self._cache and parsed:
            self._cache.put("whale_alert", cache_key, parsed)

        return self._filter_txs(parsed, min_btc)

    # ── Shared parsing ───────────────────────────────────────────────

    def _parse_blockchain_txs(
        self,
        raw_txs: list[dict[str, Any]],
        *,
        confirmed: bool,
        block_height: int = 0,
        block_time: int = 0,
    ) -> list[dict[str, Any]]:
        parsed: list[dict[str, Any]] = []

        for tx in raw_txs:
            total_out_sats = sum(o.get("value", 0) for o in tx.get("out", []))
            value_btc = total_out_sats / _SATS_PER_BTC

            outputs = []
            for o in tx.get("out", []):
                addr = o.get("addr", "")
                val = o.get("value", 0) / _SATS_PER_BTC
                if addr and val > 0:
                    outputs.append({"addr": addr, "value_btc": round(val, 8)})

            inputs = []
            for inp in tx.get("inputs", []):
                prev = inp.get("prev_out", {})
                addr = prev.get("addr", "")
                val = prev.get("value", 0) / _SATS_PER_BTC
                if addr and val > 0:
                    inputs.append({"addr": addr, "value_btc": round(val, 8)})

            entry: dict[str, Any] = {
                "hash": tx.get("hash", ""),
                "time": tx.get("time", block_time),
                "value_btc": round(value_btc, 8),
                "blockchain": "bitcoin",
                "symbol": "BTC",
                "confirmed": confirmed,
                "inputs": inputs,
                "outputs": outputs,
            }
            if confirmed:
                entry["block_height"] = block_height

            # L2: entity_ids mapping for all addresses
            if entity_id_from_key is not None:
                eid_map: dict[str, str] = {}
                for inp in inputs:
                    eid_map[inp["addr"]] = entity_id_from_key("wallet", inp["addr"])
                for out in outputs:
                    eid_map[out["addr"]] = entity_id_from_key("wallet", out["addr"])
                entry["entity_ids"] = eid_map
            else:
                entry["entity_ids"] = {}

            parsed.append(entry)

        return parsed

    def _filter_txs(self, txs: list[dict[str, Any]], min_btc: float) -> list[dict[str, Any]]:
        filtered = [t for t in txs if t.get("value_btc", 0) >= min_btc]
        filtered.sort(key=lambda t: t.get("value_btc", 0), reverse=True)
        return filtered

    # ── Summary computation ──────────────────────────────────────────

    def _compute_summary(self, txs: list[dict[str, Any]], mode: str) -> dict[str, Any]:
        if not txs:
            return {
                "count": 0,
                "total_value": 0,
                "value_unit": "BTC",
                "largest": 0,
                "avg_size": 0,
                "mode": mode,
            }

        values = [t.get("value_btc", 0) for t in txs]
        return {
            "count": len(txs),
            "total_value": round(sum(values), 2),
            "value_unit": "BTC",
            "largest": round(max(values), 2) if values else 0,
            "avg_size": round(sum(values) / len(values), 2) if values else 0,
            "mode": mode,
        }

    # ── Formatting ───────────────────────────────────────────────────

    def _format_output(
        self,
        txs: list[dict[str, Any]],
        summary: dict[str, Any],
        mode: str,
    ) -> str:
        lines: list[str] = []

        if mode == "mempool":
            lines.append(f"BTC Whale Transactions (mempool) — {summary['count']} found")
        else:
            bh = txs[0].get("block_height", "?") if txs else "?"
            lines.append(f"BTC Whale Transactions (block #{bh}) — {summary['count']} found")

        lines.append(
            f"Total: {summary['total_value']:,.2f} BTC  "
            f"Largest: {summary['largest']:,.2f} BTC  "
            f"Avg: {summary['avg_size']:,.2f} BTC"
        )
        lines.append("")

        for tx in txs:
            h = tx.get("hash", "?")[:16]
            val = tx.get("value_btc", 0)
            ts = tx.get("time", 0)
            confirmed = tx.get("confirmed", False)
            status = "✓" if confirmed else "…"
            lines.append(f"  {status} {h}... {val:>12.4f} BTC  t={ts}")

        return "\n".join(lines)

    # ── L2: Entity persistence ───────────────────────────────────────

    def _persist_entities(self, txs: list[dict[str, Any]]) -> None:
        """Register wallet entities and store L2 observations.

        Skips silently if no PipelineStore is configured or if entity
        helpers are unavailable. Any persistence error is caught and
        logged — it must never prevent the tool from returning results.
        """
        if self._store is None or entity_id_from_key is None:
            return
        if not txs:
            return
        try:
            self._persist_entities_inner(txs)
        except Exception:
            log.exception("Entity persistence failed (non-fatal)")

    def _persist_entities_inner(self, txs: list[dict[str, Any]]) -> None:
        """Inner persistence logic, separated for testability."""
        assert self._store is not None  # noqa: S101 — guarded by caller
        store = self._store

        seen_wallets: set[str] = set()

        for tx in txs:
            tx_hash = tx.get("hash", "")
            tx_time = tx.get("time", 0)
            confirmed = tx.get("confirmed", False)
            block_height = tx.get("block_height")
            inputs = tx.get("inputs", [])
            outputs = tx.get("outputs", [])

            # ── Register + observe sender wallets ──
            for inp in inputs:
                addr = inp.get("addr", "")
                if not addr:
                    continue
                wallet_eid = entity_id_from_key("wallet", addr)
                if addr not in seen_wallets:
                    seen_wallets.add(addr)
                    store.register_entity(
                        entity_type="wallet",
                        canonical_name=addr,
                        entity_id=wallet_eid,
                    )
                    store.add_entity_alias(wallet_eid, "btc_address", addr)

                store.store_entity_observation(
                    entity_id=wallet_eid,
                    source_tool="whale_alert",
                    observed_at=tx_time,
                    observation_type="btc_transfer",
                    depth_level=2,
                    value={
                        "tx_hash": tx_hash,
                        "value_btc": inp.get("value_btc", 0),
                        "direction": "out",
                        "counterparty_count": len(outputs),
                        "confirmed": confirmed,
                        "block_height": block_height,
                    },
                )

            # ── Register + observe receiver wallets ──
            for out in outputs:
                addr = out.get("addr", "")
                if not addr:
                    continue
                wallet_eid = entity_id_from_key("wallet", addr)
                if addr not in seen_wallets:
                    seen_wallets.add(addr)
                    store.register_entity(
                        entity_type="wallet",
                        canonical_name=addr,
                        entity_id=wallet_eid,
                    )
                    store.add_entity_alias(wallet_eid, "btc_address", addr)

                store.store_entity_observation(
                    entity_id=wallet_eid,
                    source_tool="whale_alert",
                    observed_at=tx_time,
                    observation_type="btc_transfer",
                    depth_level=2,
                    value={
                        "tx_hash": tx_hash,
                        "value_btc": out.get("value_btc", 0),
                        "direction": "in",
                        "counterparty_count": len(inputs),
                        "confirmed": confirmed,
                        "block_height": block_height,
                    },
                )

            # ── Link sender → receiver wallets ──
            sender_addrs = [inp.get("addr", "") for inp in inputs]
            receiver_addrs = [out.get("addr", "") for out in outputs]
            for s_addr in sender_addrs:
                if not s_addr:
                    continue
                for r_addr in receiver_addrs:
                    if not r_addr or s_addr == r_addr:
                        continue
                    store.link_entities(
                        entity_id_a=entity_id_from_key("wallet", s_addr),
                        entity_id_b=entity_id_from_key("wallet", r_addr),
                        link_type="transacts_with",
                        source="whale_alert",
                        confidence=1.0,
                        metadata={"tx_hash": tx_hash},
                    )

            # ── Link wallets → BTC-USD instrument (Phase 30) ──
            if _BTC_INSTRUMENT_EID is not None:
                all_addrs = sender_addrs + receiver_addrs
                for addr in all_addrs:
                    if not addr:
                        continue
                    store.link_entities(
                        entity_id_a=entity_id_from_key("wallet", addr),
                        entity_id_b=_BTC_INSTRUMENT_EID,
                        link_type="trades_instrument",
                        source="whale_alert",
                        confidence=1.0,
                        metadata={"tx_hash": tx_hash},
                    )
