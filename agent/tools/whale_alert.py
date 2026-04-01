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
from typing import Any

import httpx

from agent.data.cache import DataCache
from agent.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

_USER_AGENT = "TirraMind/0.1 (research; https://github.com/tirramind)"
_MEMPOOL_URL = "https://blockchain.info/unconfirmed-transactions?format=json"
_LATEST_BLOCK_URL = "https://blockchain.info/latestblock"
_RAW_BLOCK_URL = "https://blockchain.info/rawblock/{hash}"

_SATS_PER_BTC = 1e8


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

    def __init__(self, cache: DataCache | None = None) -> None:
        self._cache = cache

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
            resp = client.get(
                _MEMPOOL_URL, headers={"User-Agent": _USER_AGENT}
            )
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
            resp2 = client.get(
                _RAW_BLOCK_URL.format(hash=block_hash), headers=headers
            )
            resp2.raise_for_status()
            block = resp2.json()

        raw_txs = block.get("tx", [])
        block_height = block.get("height", 0)
        block_time = block.get("time", 0)

        parsed = self._parse_blockchain_txs(
            raw_txs, confirmed=True, block_height=block_height, block_time=block_time
        )

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

            parsed.append(entry)

        return parsed

    def _filter_txs(
        self, txs: list[dict[str, Any]], min_btc: float
    ) -> list[dict[str, Any]]:
        filtered = [t for t in txs if t.get("value_btc", 0) >= min_btc]
        filtered.sort(key=lambda t: t.get("value_btc", 0), reverse=True)
        return filtered

    # ── Summary computation ──────────────────────────────────────────

    def _compute_summary(
        self, txs: list[dict[str, Any]], mode: str
    ) -> dict[str, Any]:
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
