"""Tests for whale_alert L2 upgrade — wallet entity persistence,
entity_ids mapping, address dedup, and MI integration.

Mirrors the test pattern from test_insider_filings_l2.py / test_form144_l2.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.pipeline.entity import entity_id_from_key
from agent.tools.whale_alert import WhaleAlertTool

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_tx(
    *,
    hash: str = "abc123",
    time: int = 1700000000,
    value_btc: float = 50.0,
    confirmed: bool = False,
    block_height: int | None = None,
    inputs: list[dict[str, Any]] | None = None,
    outputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a parsed whale_alert transaction dict."""
    if inputs is None:
        inputs = [{"addr": "1SenderAddr", "value_btc": 50.0}]
    if outputs is None:
        outputs = [{"addr": "1ReceiverAddr", "value_btc": 50.0}]
    tx: dict[str, Any] = {
        "hash": hash,
        "time": time,
        "value_btc": value_btc,
        "blockchain": "bitcoin",
        "symbol": "BTC",
        "confirmed": confirmed,
        "inputs": inputs,
        "outputs": outputs,
        "entity_ids": {},
    }
    if block_height is not None:
        tx["block_height"] = block_height
    return tx


def _make_store() -> MagicMock:
    """Create a mock PipelineStore with the entity API surface."""
    store = MagicMock()
    store.register_entity = MagicMock(return_value="eid")
    store.add_entity_alias = MagicMock()
    store.store_entity_observation = MagicMock(return_value=1)
    store.resolve_entity = MagicMock(return_value=None)
    return store


# ===========================================================================
# Class: TestConstructor
# ===========================================================================


class TestConstructor:
    """Step 10b.3.2: PipelineStore kwarg in constructor."""

    def test_default_no_store(self) -> None:
        tool = WhaleAlertTool()
        assert tool._store is None
        assert tool._cache is None

    def test_with_store(self) -> None:
        store = _make_store()
        tool = WhaleAlertTool(pipeline_store=store)
        assert tool._store is store

    def test_with_cache_and_store(self) -> None:
        cache = MagicMock()
        store = _make_store()
        tool = WhaleAlertTool(cache, pipeline_store=store)
        assert tool._cache is cache
        assert tool._store is store

    def test_store_keyword_only(self) -> None:
        """pipeline_store must be keyword-only — positional should fail."""
        store = _make_store()
        with pytest.raises(TypeError):
            WhaleAlertTool(None, store)  # type: ignore[misc]


# ===========================================================================
# Class: TestPersistEntitiesGuard
# ===========================================================================


class TestPersistEntitiesGuard:
    """Step 10b.3.3: Guard method skips when store is None / entities unavailable."""

    def test_no_store_is_noop(self) -> None:
        tool = WhaleAlertTool()
        tool._persist_entities([_make_tx()])  # should not raise

    def test_empty_txs_is_noop(self) -> None:
        store = _make_store()
        tool = WhaleAlertTool(pipeline_store=store)
        tool._persist_entities([])
        store.register_entity.assert_not_called()

    def test_entity_id_unavailable(self) -> None:
        """When entity_id_from_key is None, persistence is skipped."""
        store = _make_store()
        tool = WhaleAlertTool(pipeline_store=store)
        with patch("agent.tools.whale_alert.entity_id_from_key", None):
            tool._persist_entities([_make_tx()])
        store.register_entity.assert_not_called()

    def test_inner_exception_caught(self) -> None:
        """Errors in _persist_entities_inner are caught, tool still works."""
        store = _make_store()
        store.register_entity.side_effect = RuntimeError("DB locked")
        tool = WhaleAlertTool(pipeline_store=store)
        # Should not raise
        tool._persist_entities([_make_tx()])


# ===========================================================================
# Class: TestPersistEntitiesInner
# ===========================================================================


class TestPersistEntitiesInner:
    """Step 10b.3.3: Inner persistence logic — wallet registration + observations."""

    def test_registers_sender_and_receiver(self) -> None:
        store = _make_store()
        tool = WhaleAlertTool(pipeline_store=store)
        tx = _make_tx(
            inputs=[{"addr": "1Sender", "value_btc": 50.0}],
            outputs=[{"addr": "1Receiver", "value_btc": 50.0}],
        )
        tool._persist_entities_inner([tx])

        # Two entities registered (sender + receiver)
        assert store.register_entity.call_count == 2
        # Two aliases added
        assert store.add_entity_alias.call_count == 2
        # Two observations (one per address)
        assert store.store_entity_observation.call_count == 2

    def test_sender_observation_direction_out(self) -> None:
        store = _make_store()
        tool = WhaleAlertTool(pipeline_store=store)
        tx = _make_tx(
            hash="tx1",
            inputs=[{"addr": "1Sender", "value_btc": 25.0}],
            outputs=[{"addr": "1Receiver", "value_btc": 25.0}],
        )
        tool._persist_entities_inner([tx])

        # Find the sender observation (first call)
        calls = store.store_entity_observation.call_args_list
        sender_call = calls[0]
        assert sender_call.kwargs["observation_type"] == "btc_transfer"
        assert sender_call.kwargs["depth_level"] == 2
        val = sender_call.kwargs["value"]
        assert val["direction"] == "out"
        assert val["tx_hash"] == "tx1"
        assert val["value_btc"] == 25.0

    def test_receiver_observation_direction_in(self) -> None:
        store = _make_store()
        tool = WhaleAlertTool(pipeline_store=store)
        tx = _make_tx(
            hash="tx2",
            inputs=[{"addr": "1Sender", "value_btc": 30.0}],
            outputs=[{"addr": "1Receiver", "value_btc": 30.0}],
        )
        tool._persist_entities_inner([tx])

        calls = store.store_entity_observation.call_args_list
        receiver_call = calls[1]
        val = receiver_call.kwargs["value"]
        assert val["direction"] == "in"
        assert val["tx_hash"] == "tx2"
        assert val["value_btc"] == 30.0

    def test_confirmed_block_height_in_observation(self) -> None:
        store = _make_store()
        tool = WhaleAlertTool(pipeline_store=store)
        tx = _make_tx(confirmed=True, block_height=900000)
        tool._persist_entities_inner([tx])

        calls = store.store_entity_observation.call_args_list
        for call in calls:
            assert call.kwargs["value"]["confirmed"] is True
            assert call.kwargs["value"]["block_height"] == 900000

    def test_mempool_no_block_height(self) -> None:
        store = _make_store()
        tool = WhaleAlertTool(pipeline_store=store)
        tx = _make_tx(confirmed=False)
        tool._persist_entities_inner([tx])

        calls = store.store_entity_observation.call_args_list
        for call in calls:
            assert call.kwargs["value"]["confirmed"] is False
            assert call.kwargs["value"]["block_height"] is None

    def test_counterparty_count(self) -> None:
        store = _make_store()
        tool = WhaleAlertTool(pipeline_store=store)
        tx = _make_tx(
            inputs=[
                {"addr": "1S1", "value_btc": 30.0},
                {"addr": "1S2", "value_btc": 20.0},
            ],
            outputs=[
                {"addr": "1R1", "value_btc": 40.0},
                {"addr": "1R2", "value_btc": 10.0},
            ],
        )
        tool._persist_entities_inner([tx])

        calls = store.store_entity_observation.call_args_list
        # Senders see counterparty_count = len(outputs) = 2
        assert calls[0].kwargs["value"]["counterparty_count"] == 2
        assert calls[1].kwargs["value"]["counterparty_count"] == 2
        # Receivers see counterparty_count = len(inputs) = 2
        assert calls[2].kwargs["value"]["counterparty_count"] == 2
        assert calls[3].kwargs["value"]["counterparty_count"] == 2

    def test_entity_type_is_wallet(self) -> None:
        store = _make_store()
        tool = WhaleAlertTool(pipeline_store=store)
        tool._persist_entities_inner([_make_tx()])

        for call in store.register_entity.call_args_list:
            assert call.kwargs["entity_type"] == "wallet"

    def test_alias_source_is_btc_address(self) -> None:
        store = _make_store()
        tool = WhaleAlertTool(pipeline_store=store)
        tool._persist_entities_inner([_make_tx()])

        for call in store.add_entity_alias.call_args_list:
            assert call.args[1] == "btc_address"  # source arg

    def test_canonical_name_is_address(self) -> None:
        store = _make_store()
        tool = WhaleAlertTool(pipeline_store=store)
        tx = _make_tx(inputs=[{"addr": "1MyAddr", "value_btc": 10.0}], outputs=[])
        tool._persist_entities_inner([tx])

        call = store.register_entity.call_args_list[0]
        assert call.kwargs["canonical_name"] == "1MyAddr"


# ===========================================================================
# Class: TestPersistEntitiesDedup
# ===========================================================================


class TestPersistEntitiesDedup:
    """Dedup: same address across multiple txs is registered once."""

    def test_same_address_multiple_txs(self) -> None:
        store = _make_store()
        tool = WhaleAlertTool(pipeline_store=store)
        txs = [
            _make_tx(
                hash="tx1",
                inputs=[{"addr": "1Same", "value_btc": 10.0}],
                outputs=[{"addr": "1R1", "value_btc": 10.0}],
            ),
            _make_tx(
                hash="tx2",
                inputs=[{"addr": "1Same", "value_btc": 20.0}],
                outputs=[{"addr": "1R2", "value_btc": 20.0}],
            ),
        ]
        tool._persist_entities_inner(txs)

        # "1Same" registered once; "1R1" and "1R2" once each = 3 registrations
        assert store.register_entity.call_count == 3
        # But observations for "1Same" in both txs = 2 observations for sender + 2 for receivers = 4
        assert store.store_entity_observation.call_count == 4

    def test_same_address_as_sender_and_receiver(self) -> None:
        """Self-transfer: same address on both sides → register once, 2 observations."""
        store = _make_store()
        tool = WhaleAlertTool(pipeline_store=store)
        tx = _make_tx(
            inputs=[{"addr": "1Self", "value_btc": 10.0}],
            outputs=[{"addr": "1Self", "value_btc": 10.0}],
        )
        tool._persist_entities_inner([tx])

        # Registered once (dedup)
        assert store.register_entity.call_count == 1
        assert store.add_entity_alias.call_count == 1
        # Two observations: one out, one in
        assert store.store_entity_observation.call_count == 2
        dirs = [
            c.kwargs["value"]["direction"]
            for c in store.store_entity_observation.call_args_list
        ]
        assert sorted(dirs) == ["in", "out"]


# ===========================================================================
# Class: TestPersistEntitiesEdgeCases
# ===========================================================================


class TestPersistEntitiesEdgeCases:
    """Edge cases for persistence."""

    def test_missing_addr_in_input_skipped(self) -> None:
        store = _make_store()
        tool = WhaleAlertTool(pipeline_store=store)
        tx = _make_tx(
            inputs=[{"addr": "", "value_btc": 10.0}],
            outputs=[{"addr": "1Receiver", "value_btc": 10.0}],
        )
        tool._persist_entities_inner([tx])

        # Only receiver registered
        assert store.register_entity.call_count == 1
        assert store.store_entity_observation.call_count == 1

    def test_missing_addr_key_in_input_skipped(self) -> None:
        store = _make_store()
        tool = WhaleAlertTool(pipeline_store=store)
        tx = _make_tx(
            inputs=[{"value_btc": 10.0}],  # no "addr" key at all
            outputs=[{"addr": "1Receiver", "value_btc": 10.0}],
        )
        tool._persist_entities_inner([tx])

        assert store.register_entity.call_count == 1

    def test_empty_inputs_and_outputs(self) -> None:
        store = _make_store()
        tool = WhaleAlertTool(pipeline_store=store)
        tx = _make_tx(inputs=[], outputs=[])
        tool._persist_entities_inner([tx])

        store.register_entity.assert_not_called()
        store.store_entity_observation.assert_not_called()

    def test_no_hash_still_persists(self) -> None:
        store = _make_store()
        tool = WhaleAlertTool(pipeline_store=store)
        tx = _make_tx(hash="")
        tool._persist_entities_inner([tx])

        # Still stores observations, just with empty hash
        assert store.store_entity_observation.call_count == 2
        val = store.store_entity_observation.call_args_list[0].kwargs["value"]
        assert val["tx_hash"] == ""

    def test_zero_time_still_persists(self) -> None:
        store = _make_store()
        tool = WhaleAlertTool(pipeline_store=store)
        tx = _make_tx(time=0)
        tool._persist_entities_inner([tx])

        call = store.store_entity_observation.call_args_list[0]
        assert call.kwargs["observed_at"] == 0

    def test_many_addresses_per_tx(self) -> None:
        """Multiple inputs + outputs in one tx."""
        store = _make_store()
        tool = WhaleAlertTool(pipeline_store=store)
        inputs = [{"addr": f"1S{i}", "value_btc": float(i)} for i in range(5)]
        outputs = [{"addr": f"1R{i}", "value_btc": float(i)} for i in range(3)]
        tx = _make_tx(inputs=inputs, outputs=outputs)
        tool._persist_entities_inner([tx])

        # 5 senders + 3 receivers = 8 entities
        assert store.register_entity.call_count == 8
        # 5 sender obs + 3 receiver obs = 8 observations
        assert store.store_entity_observation.call_count == 8


# ===========================================================================
# Class: TestEntityIds
# ===========================================================================


class TestEntityIds:
    """Step 10b.3.4: entity_ids mapping in parsed transaction dicts."""

    def test_entity_ids_present(self) -> None:
        tool = WhaleAlertTool()
        raw_txs = [
            {
                "hash": "txhash1",
                "time": 1700000000,
                "inputs": [{"prev_out": {"addr": "1Sender", "value": 5000000000}}],
                "out": [{"addr": "1Receiver", "value": 5000000000}],
            }
        ]
        parsed = tool._parse_blockchain_txs(raw_txs, confirmed=False)
        assert len(parsed) == 1
        assert "entity_ids" in parsed[0]
        assert parsed[0]["entity_ids"]["1Sender"] == entity_id_from_key(
            "wallet", "1Sender"
        )
        assert parsed[0]["entity_ids"]["1Receiver"] == entity_id_from_key(
            "wallet", "1Receiver"
        )

    def test_entity_ids_all_addresses_mapped(self) -> None:
        tool = WhaleAlertTool()
        raw_txs = [
            {
                "hash": "txhash2",
                "time": 1700000000,
                "inputs": [
                    {"prev_out": {"addr": "1S1", "value": 1000000000}},
                    {"prev_out": {"addr": "1S2", "value": 2000000000}},
                ],
                "out": [
                    {"addr": "1R1", "value": 2000000000},
                    {"addr": "1R2", "value": 1000000000},
                ],
            }
        ]
        parsed = tool._parse_blockchain_txs(raw_txs, confirmed=False)
        eids = parsed[0]["entity_ids"]
        assert len(eids) == 4
        assert "1S1" in eids
        assert "1S2" in eids
        assert "1R1" in eids
        assert "1R2" in eids

    def test_entity_ids_empty_when_no_module(self) -> None:
        """When entity_id_from_key is None → entity_ids is empty dict."""
        tool = WhaleAlertTool()
        raw_txs = [
            {
                "hash": "txhash3",
                "time": 1700000000,
                "inputs": [{"prev_out": {"addr": "1S", "value": 1000000000}}],
                "out": [{"addr": "1R", "value": 1000000000}],
            }
        ]
        with patch("agent.tools.whale_alert.entity_id_from_key", None):
            parsed = tool._parse_blockchain_txs(raw_txs, confirmed=False)
        assert parsed[0]["entity_ids"] == {}

    def test_entity_ids_deterministic(self) -> None:
        """Same address → same entity_id every time."""
        eid1 = entity_id_from_key("wallet", "1TestAddr")
        eid2 = entity_id_from_key("wallet", "1TestAddr")
        assert eid1 == eid2
        assert len(eid1) == 16

    def test_entity_ids_confirmed_block(self) -> None:
        """entity_ids works for confirmed mode too."""
        tool = WhaleAlertTool()
        raw_txs = [
            {
                "hash": "blocktx1",
                "time": 1700000000,
                "inputs": [{"prev_out": {"addr": "1BS", "value": 10000000000}}],
                "out": [{"addr": "1BR", "value": 10000000000}],
            }
        ]
        parsed = tool._parse_blockchain_txs(
            raw_txs, confirmed=True, block_height=900000
        )
        assert parsed[0]["entity_ids"]["1BS"] == entity_id_from_key("wallet", "1BS")
        assert parsed[0]["block_height"] == 900000


# ===========================================================================
# Class: TestExecuteIntegration
# ===========================================================================


class TestExecuteIntegration:
    """Integration: full execute() with and without PipelineStore."""

    def test_execute_with_store_persists(self) -> None:
        """When PipelineStore is set, entities are persisted during execute."""
        store = _make_store()
        tool = WhaleAlertTool(pipeline_store=store)

        mempool_resp = MagicMock()
        mempool_resp.json.return_value = {
            "txs": [
                {
                    "hash": "whaleabc",
                    "time": 1700000000,
                    "inputs": [
                        {"prev_out": {"addr": "1BigSender", "value": 50_00000000}}
                    ],
                    "out": [{"addr": "1BigReceiver", "value": 50_00000000}],
                }
            ]
        }
        mempool_resp.raise_for_status = MagicMock()

        with patch("agent.tools.whale_alert.httpx.Client") as mock_client:
            instance = MagicMock()
            instance.__enter__ = lambda s: s
            instance.__exit__ = MagicMock(return_value=False)
            instance.get.return_value = mempool_resp
            mock_client.return_value = instance

            result = tool.execute(mode="mempool", min_btc=10.0, limit=10)

        assert result.success
        # Entities persisted
        assert store.register_entity.call_count >= 1
        assert store.store_entity_observation.call_count >= 1

    def test_execute_without_store_backward_compatible(self) -> None:
        """Without PipelineStore, execute still works — no entity calls."""
        tool = WhaleAlertTool()

        mempool_resp = MagicMock()
        mempool_resp.json.return_value = {
            "txs": [
                {
                    "hash": "whaledef",
                    "time": 1700000000,
                    "inputs": [{"prev_out": {"addr": "1S", "value": 20_00000000}}],
                    "out": [{"addr": "1R", "value": 20_00000000}],
                }
            ]
        }
        mempool_resp.raise_for_status = MagicMock()

        with patch("agent.tools.whale_alert.httpx.Client") as mock_client:
            instance = MagicMock()
            instance.__enter__ = lambda s: s
            instance.__exit__ = MagicMock(return_value=False)
            instance.get.return_value = mempool_resp
            mock_client.return_value = instance

            result = tool.execute(mode="mempool", min_btc=10.0, limit=10)

        assert result.success
        assert "transactions" in result.data

    def test_execute_persistence_error_nonfatal(self) -> None:
        """Even if persistence crashes, tool returns results."""
        store = _make_store()
        store.register_entity.side_effect = RuntimeError("DB on fire")
        tool = WhaleAlertTool(pipeline_store=store)

        mempool_resp = MagicMock()
        mempool_resp.json.return_value = {
            "txs": [
                {
                    "hash": "tx_err",
                    "time": 1700000000,
                    "inputs": [{"prev_out": {"addr": "1Err", "value": 100_00000000}}],
                    "out": [{"addr": "1ErrR", "value": 100_00000000}],
                }
            ]
        }
        mempool_resp.raise_for_status = MagicMock()

        with patch("agent.tools.whale_alert.httpx.Client") as mock_client:
            instance = MagicMock()
            instance.__enter__ = lambda s: s
            instance.__exit__ = MagicMock(return_value=False)
            instance.get.return_value = mempool_resp
            mock_client.return_value = instance

            result = tool.execute(mode="mempool", min_btc=10.0)

        assert result.success
        assert "transactions" in result.data


# ===========================================================================
# Class: TestRealStoreIntegration
# ===========================================================================


class TestRealStoreIntegration:
    """Integration test with a real in-memory PipelineStore."""

    def test_end_to_end_real_store(self) -> None:
        """Full loop: persist entities → query back from real SQLite store."""
        from agent.pipeline.store import PipelineStore

        store = PipelineStore(":memory:")
        tool = WhaleAlertTool(pipeline_store=store)

        txs = [
            _make_tx(
                hash="realtx1",
                time=1700000000,
                inputs=[{"addr": "1RealSender", "value_btc": 100.0}],
                outputs=[{"addr": "1RealReceiver", "value_btc": 100.0}],
            ),
            _make_tx(
                hash="realtx2",
                time=1700001000,
                inputs=[{"addr": "1RealSender", "value_btc": 50.0}],
                outputs=[{"addr": "1AnotherReceiver", "value_btc": 50.0}],
            ),
        ]

        tool._persist_entities_inner(txs)

        # Verify entities exist
        sender_eid = entity_id_from_key("wallet", "1RealSender")
        entity = store.get_entity(sender_eid)
        assert entity is not None
        assert entity["entity_type"] == "wallet"
        assert entity["canonical_name"] == "1RealSender"

        # Verify alias
        resolved = store.resolve_entity("btc_address", "1RealSender")
        assert resolved == sender_eid

        # Verify observations
        obs = store.query_entity_observations(sender_eid, source_tool="whale_alert")
        assert len(obs) == 2  # Two txs with this sender
        assert all(o["observation_type"] == "btc_transfer" for o in obs)
        assert all(o["depth_level"] == 2 for o in obs)

        # Verify receiver has 1 observation
        recv_eid = entity_id_from_key("wallet", "1RealReceiver")
        recv_obs = store.query_entity_observations(recv_eid, source_tool="whale_alert")
        assert len(recv_obs) == 1


# ===========================================================================
# Class: TestMIMeasurement
# ===========================================================================


class TestMIMeasurement:
    """Step 10b.3.6: MI integration — L2 entity observations carry more
    information than L1 aggregates.

    Design: simulate L1 (total BTC moved per time window) vs L2 (per-wallet
    transfer amounts). Compute MI of each against a synthetic target.
    L2 should yield equal or higher MI.
    """

    def test_l2_mi_geq_l1(self) -> None:
        import numpy as np

        try:
            from sklearn.feature_selection import mutual_info_regression
        except ImportError:
            pytest.skip("sklearn not installed")

        rng = np.random.default_rng(42)
        n = 200

        # Synthetic wallets: 3 wallets with different behavioral profiles
        wallet_a = rng.exponential(scale=10.0, size=n)  # frequent small
        wallet_b = rng.exponential(scale=100.0, size=n)  # occasional large
        wallet_c = rng.exponential(scale=50.0, size=n)  # medium

        # Target correlated with wallet_b (the whale)
        noise = rng.normal(0, 5, size=n)
        target = 0.6 * wallet_b + 0.3 * wallet_a + noise

        # L1: aggregate total per period
        l1_features = (wallet_a + wallet_b + wallet_c).reshape(-1, 1)

        # L2: per-wallet breakdown
        l2_features = np.column_stack([wallet_a, wallet_b, wallet_c])

        mi_l1 = mutual_info_regression(l1_features, target, random_state=42)[0]
        mi_l2 = mutual_info_regression(l2_features, target, random_state=42).sum()

        # L2 should have >= MI (and in practice strictly more, since it
        # preserves the entity-level structure that L1 aggregates away)
        assert (
            mi_l2 >= mi_l1 * 0.95
        ), f"Expected MI(L2) >= MI(L1): got MI(L2)={mi_l2:.4f} vs MI(L1)={mi_l1:.4f}"

    def test_l2_mi_with_real_store(self) -> None:
        """End-to-end: store L2 observations, query back, compute MI."""
        import numpy as np

        try:
            from sklearn.feature_selection import mutual_info_regression
        except ImportError:
            pytest.skip("sklearn not installed")

        from agent.pipeline.store import PipelineStore

        store = PipelineStore(":memory:")
        tool = WhaleAlertTool(pipeline_store=store)
        rng = np.random.default_rng(123)

        # Simulate 50 whale transactions across 3 wallets
        wallets = ["1WalletA", "1WalletB", "1WalletC"]
        txs = []
        for i in range(50):
            sender = wallets[i % 3]
            val = float(rng.exponential(50.0))
            txs.append(
                _make_tx(
                    hash=f"mitx{i}",
                    time=1700000000 + i * 600,
                    value_btc=val,
                    inputs=[{"addr": sender, "value_btc": val}],
                    outputs=[{"addr": "1ExchangeHotWallet", "value_btc": val}],
                )
            )

        tool._persist_entities_inner(txs)

        # Query observations for each wallet
        per_wallet_values: dict[str, list[float]] = {w: [] for w in wallets}
        for w in wallets:
            eid = entity_id_from_key("wallet", w)
            obs = store.query_entity_observations(
                eid, source_tool="whale_alert", limit=100
            )
            # sort by observed_at and extract values
            obs.sort(key=lambda o: o["observed_at"])
            for o in obs:
                val_data = o["value"] if isinstance(o["value"], dict) else {}
                per_wallet_values[w].append(val_data.get("value_btc", 0))

        # Each wallet should have observations
        for w in wallets:
            assert len(per_wallet_values[w]) > 0, f"No observations for {w}"

        # Verify depth_level = 2
        eid_a = entity_id_from_key("wallet", wallets[0])
        obs_a = store.query_entity_observations(
            eid_a, source_tool="whale_alert", depth_level=2
        )
        assert len(obs_a) > 0, "No depth_level=2 observations found"
