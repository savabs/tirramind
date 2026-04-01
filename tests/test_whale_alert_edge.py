"""
Edge-case test suite for WhaleAlertTool (Phase 6c — upgraded).

Tests cover both free modes: mempool + confirmed block.
No paid API paths — everything $0 with no key.
  - Mode validation (invalid mode)
  - Mempool: normal, empty, malformed, missing fields, HTTP error
  - Confirmed: normal, empty block, missing block hash, HTTP error
  - min_btc filtering + high threshold → empty
  - Limit truncation
  - Summary with zero transactions
  - Caching behavior
  - Output formatting (mempool + confirmed)
  - Sorted output by value
  - Very large values (no overflow)
  - Satoshi boundary (1 sat = 1e-8 BTC)
  - Multi-output txs sum correctly
  - Outputs with no addr excluded
  - Confirmed flag and block_height in results
  - CLI registration (tool name)
  - Bandit arm exists
"""

import sys
import unittest
from unittest.mock import MagicMock, patch
from typing import Any

sys.path.insert(0, "/home/becmachlean/2024/projects/tirramind_v1")

from agent.tools.whale_alert import WhaleAlertTool


# ──────────────────────────────────────────────────────────────────
# Mock helpers
# ──────────────────────────────────────────────────────────────────

def _make_mempool_response(txs=None):
    if txs is None:
        txs = [
            {
                "hash": "abc123def456",
                "time": 1700000000,
                "inputs": [{"prev_out": {"addr": "1Sender", "value": 5000000000}}],
                "out": [{"addr": "1Receiver", "value": 5000000000}],
            },
            {
                "hash": "bbb222ccc333",
                "time": 1700000001,
                "inputs": [{"prev_out": {"addr": "1S2", "value": 200000000}}],
                "out": [{"addr": "1R2", "value": 200000000}],
            },
        ]
    return {"txs": txs}


def _make_block_response(txs=None, height=900000, time_=1700000000):
    if txs is None:
        txs = [
            {
                "hash": "block_tx_1",
                "time": time_,
                "inputs": [{"prev_out": {"addr": "1From", "value": 10000000000}}],
                "out": [{"addr": "1To", "value": 10000000000}],
            },
            {
                "hash": "block_tx_2",
                "time": time_,
                "inputs": [{"prev_out": {"addr": "1F2", "value": 300000000}}],
                "out": [{"addr": "1T2", "value": 300000000}],
            },
        ]
    return {
        "hash": "000000000000000000001234",
        "height": height,
        "time": time_,
        "tx": txs,
    }


class MockClient:
    """Context-manager mock for httpx.Client."""
    def __init__(self, get_fn=None, **kwargs):
        self._get_fn = get_fn

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get(self, url, **kwargs):
        return self._get_fn(url, **kwargs)


def _default_get_fn(url, **kwargs):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    if "unconfirmed" in url:
        resp.json.return_value = _make_mempool_response()
    elif "latestblock" in url:
        resp.json.return_value = {"hash": "00000abc", "height": 900000}
    elif "rawblock" in url:
        resp.json.return_value = _make_block_response()
    return resp


def _make_mock_client(get_fn=None):
    fn = get_fn or _default_get_fn
    return lambda **kwargs: MockClient(get_fn=fn, **kwargs)


# ──────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────

class TestWhaleAlertEdgeCases(unittest.TestCase):

    # ── Mode validation ──────────────────────────────────────────

    def test_invalid_mode(self):
        tool = WhaleAlertTool()
        r = tool.execute(mode="whale_alert")
        self.assertFalse(r.success)
        self.assertIn("Invalid mode", r.output)

    def test_invalid_mode_random_string(self):
        tool = WhaleAlertTool()
        r = tool.execute(mode="foobar")
        self.assertFalse(r.success)

    # ── Mempool: normal ──────────────────────────────────────────

    @patch("agent.tools.whale_alert.httpx.Client", new_callable=lambda: _make_mock_client())
    def test_mempool_normal(self, _):
        tool = WhaleAlertTool()
        r = tool.execute(mode="mempool", min_btc=0.0)
        self.assertTrue(r.success)
        self.assertEqual(len(r.data["transactions"]), 2)
        self.assertEqual(r.data["mode"], "mempool")
        self.assertEqual(r.data["summary"]["value_unit"], "BTC")

    @patch("agent.tools.whale_alert.httpx.Client", new_callable=lambda: _make_mock_client())
    def test_mempool_default_mode(self, _):
        """Default mode should be mempool."""
        tool = WhaleAlertTool()
        r = tool.execute(min_btc=0.0)
        self.assertTrue(r.success)
        self.assertEqual(r.data["mode"], "mempool")

    @patch("agent.tools.whale_alert.httpx.Client", new_callable=lambda: _make_mock_client())
    def test_mempool_confirms_false(self, _):
        tool = WhaleAlertTool()
        r = tool.execute(mode="mempool", min_btc=0.0)
        for tx in r.data["transactions"]:
            self.assertFalse(tx["confirmed"])

    # ── Mempool: min_btc filter ──────────────────────────────────

    @patch("agent.tools.whale_alert.httpx.Client", new_callable=lambda: _make_mock_client())
    def test_mempool_min_btc_filter(self, _):
        tool = WhaleAlertTool()
        r = tool.execute(mode="mempool", min_btc=10.0)
        self.assertTrue(r.success)
        # 50 BTC passes, 2 BTC doesn't
        self.assertEqual(len(r.data["transactions"]), 1)
        self.assertAlmostEqual(r.data["transactions"][0]["value_btc"], 50.0)

    @patch("agent.tools.whale_alert.httpx.Client", new_callable=lambda: _make_mock_client())
    def test_mempool_high_threshold_empty(self, _):
        tool = WhaleAlertTool()
        r = tool.execute(mode="mempool", min_btc=99999.0)
        self.assertTrue(r.success)
        self.assertEqual(len(r.data["transactions"]), 0)
        self.assertEqual(r.data["summary"]["count"], 0)

    # ── Mempool: empty response ──────────────────────────────────

    @patch("agent.tools.whale_alert.httpx.Client")
    def test_mempool_empty(self, mock_cls):
        def get_fn(url, **kw):
            resp = MagicMock()
            resp.json.return_value = {"txs": []}
            resp.raise_for_status = MagicMock()
            return resp
        mock_cls.return_value = MockClient(get_fn=get_fn)
        tool = WhaleAlertTool()
        r = tool.execute(mode="mempool", min_btc=0.0)
        self.assertTrue(r.success)
        self.assertEqual(len(r.data["transactions"]), 0)

    # ── Mempool: malformed ───────────────────────────────────────

    @patch("agent.tools.whale_alert.httpx.Client")
    def test_mempool_no_txs_key(self, mock_cls):
        def get_fn(url, **kw):
            resp = MagicMock()
            resp.json.return_value = {}
            resp.raise_for_status = MagicMock()
            return resp
        mock_cls.return_value = MockClient(get_fn=get_fn)
        tool = WhaleAlertTool()
        r = tool.execute(mode="mempool", min_btc=0.0)
        self.assertTrue(r.success)
        self.assertEqual(len(r.data["transactions"]), 0)

    @patch("agent.tools.whale_alert.httpx.Client")
    def test_mempool_tx_missing_fields(self, mock_cls):
        def get_fn(url, **kw):
            resp = MagicMock()
            resp.json.return_value = {"txs": [
                {},
                {"out": [{"value": 100000000}]},
            ]}
            resp.raise_for_status = MagicMock()
            return resp
        mock_cls.return_value = MockClient(get_fn=get_fn)
        tool = WhaleAlertTool()
        r = tool.execute(mode="mempool", min_btc=0.0)
        self.assertTrue(r.success)

    # ── Mempool: HTTP error ──────────────────────────────────────

    @patch("agent.tools.whale_alert.httpx.Client")
    def test_mempool_http_error(self, mock_cls):
        def get_fn(url, **kw):
            resp = MagicMock()
            resp.raise_for_status.side_effect = Exception("503")
            return resp
        mock_cls.return_value = MockClient(get_fn=get_fn)
        tool = WhaleAlertTool()
        r = tool.execute(mode="mempool")
        self.assertFalse(r.success)
        self.assertIn("Fetch error", r.output)

    # ── Confirmed: normal ────────────────────────────────────────

    @patch("agent.tools.whale_alert.httpx.Client", new_callable=lambda: _make_mock_client())
    def test_confirmed_normal(self, _):
        tool = WhaleAlertTool()
        r = tool.execute(mode="confirmed", min_btc=0.0)
        self.assertTrue(r.success)
        self.assertEqual(len(r.data["transactions"]), 2)
        self.assertEqual(r.data["mode"], "confirmed")

    @patch("agent.tools.whale_alert.httpx.Client", new_callable=lambda: _make_mock_client())
    def test_confirmed_has_block_height(self, _):
        tool = WhaleAlertTool()
        r = tool.execute(mode="confirmed", min_btc=0.0)
        for tx in r.data["transactions"]:
            self.assertTrue(tx["confirmed"])
            self.assertEqual(tx["block_height"], 900000)

    @patch("agent.tools.whale_alert.httpx.Client", new_callable=lambda: _make_mock_client())
    def test_confirmed_min_btc_filter(self, _):
        tool = WhaleAlertTool()
        r = tool.execute(mode="confirmed", min_btc=50.0)
        # 100 BTC passes, 3 BTC doesn't
        self.assertEqual(len(r.data["transactions"]), 1)
        self.assertAlmostEqual(r.data["transactions"][0]["value_btc"], 100.0)

    # ── Confirmed: empty block ───────────────────────────────────

    @patch("agent.tools.whale_alert.httpx.Client")
    def test_confirmed_empty_block(self, mock_cls):
        def get_fn(url, **kw):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if "latestblock" in url:
                resp.json.return_value = {"hash": "0000abc", "height": 1}
            elif "rawblock" in url:
                resp.json.return_value = {"hash": "0000abc", "height": 1, "time": 0, "tx": []}
            return resp
        mock_cls.return_value = MockClient(get_fn=get_fn)
        tool = WhaleAlertTool()
        r = tool.execute(mode="confirmed", min_btc=0.0)
        self.assertTrue(r.success)
        self.assertEqual(len(r.data["transactions"]), 0)

    # ── Confirmed: missing block hash ────────────────────────────

    @patch("agent.tools.whale_alert.httpx.Client")
    def test_confirmed_no_block_hash(self, mock_cls):
        def get_fn(url, **kw):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if "latestblock" in url:
                resp.json.return_value = {}  # no hash
            return resp
        mock_cls.return_value = MockClient(get_fn=get_fn)
        tool = WhaleAlertTool()
        r = tool.execute(mode="confirmed")
        self.assertFalse(r.success)
        self.assertIn("Fetch error", r.output)

    # ── Confirmed: HTTP error on block fetch ─────────────────────

    @patch("agent.tools.whale_alert.httpx.Client")
    def test_confirmed_http_error_latestblock(self, mock_cls):
        def get_fn(url, **kw):
            resp = MagicMock()
            resp.raise_for_status.side_effect = Exception("500")
            return resp
        mock_cls.return_value = MockClient(get_fn=get_fn)
        tool = WhaleAlertTool()
        r = tool.execute(mode="confirmed")
        self.assertFalse(r.success)

    @patch("agent.tools.whale_alert.httpx.Client")
    def test_confirmed_http_error_rawblock(self, mock_cls):
        def get_fn(url, **kw):
            resp = MagicMock()
            if "latestblock" in url:
                resp.json.return_value = {"hash": "abc123"}
                resp.raise_for_status = MagicMock()
            elif "rawblock" in url:
                resp.raise_for_status.side_effect = Exception("404")
            return resp
        mock_cls.return_value = MockClient(get_fn=get_fn)
        tool = WhaleAlertTool()
        r = tool.execute(mode="confirmed")
        self.assertFalse(r.success)

    # ── Limit truncation ─────────────────────────────────────────

    @patch("agent.tools.whale_alert.httpx.Client")
    def test_limit_truncation(self, mock_cls):
        many_txs = []
        for i in range(50):
            many_txs.append({
                "hash": f"tx_{i}", "time": 1700000000 + i,
                "inputs": [], "out": [{"addr": f"a{i}", "value": 1100000000}],
            })
        def get_fn(url, **kw):
            resp = MagicMock()
            resp.json.return_value = {"txs": many_txs}
            resp.raise_for_status = MagicMock()
            return resp
        mock_cls.return_value = MockClient(get_fn=get_fn)
        tool = WhaleAlertTool()
        r = tool.execute(mode="mempool", min_btc=0.0, limit=5)
        self.assertEqual(len(r.data["transactions"]), 5)

    # ── Caching ──────────────────────────────────────────────────

    @patch("agent.tools.whale_alert.httpx.Client", new_callable=lambda: _make_mock_client())
    def test_cache_stores_and_retrieves(self, _):
        cache = MagicMock()
        cache.get.return_value = None
        tool = WhaleAlertTool(cache=cache)
        r = tool.execute(mode="mempool", min_btc=0.0)
        self.assertTrue(r.success)
        cache.put.assert_called_once()

        # Simulate cache hit
        cache.get.return_value = r.data["transactions"]
        r2 = tool.execute(mode="mempool", min_btc=0.0)
        self.assertTrue(r2.success)

    @patch("agent.tools.whale_alert.httpx.Client", new_callable=lambda: _make_mock_client())
    def test_cache_confirmed_mode(self, _):
        cache = MagicMock()
        cache.get.return_value = None
        tool = WhaleAlertTool(cache=cache)
        r = tool.execute(mode="confirmed", min_btc=0.0)
        self.assertTrue(r.success)
        cache.put.assert_called_once()

    # ── Summary zero transactions ────────────────────────────────

    @patch("agent.tools.whale_alert.httpx.Client")
    def test_summary_zero_mempool(self, mock_cls):
        def get_fn(url, **kw):
            resp = MagicMock()
            resp.json.return_value = {"txs": []}
            resp.raise_for_status = MagicMock()
            return resp
        mock_cls.return_value = MockClient(get_fn=get_fn)
        tool = WhaleAlertTool()
        r = tool.execute(mode="mempool", min_btc=0.0)
        s = r.data["summary"]
        self.assertEqual(s["count"], 0)
        self.assertEqual(s["total_value"], 0)
        self.assertEqual(s["largest"], 0)
        self.assertEqual(s["avg_size"], 0)

    # ── Sorted output ────────────────────────────────────────────

    @patch("agent.tools.whale_alert.httpx.Client", new_callable=lambda: _make_mock_client())
    def test_sorted_by_value(self, _):
        tool = WhaleAlertTool()
        r = tool.execute(mode="mempool", min_btc=0.0)
        vals = [t["value_btc"] for t in r.data["transactions"]]
        self.assertEqual(vals, sorted(vals, reverse=True))

    @patch("agent.tools.whale_alert.httpx.Client", new_callable=lambda: _make_mock_client())
    def test_confirmed_sorted_by_value(self, _):
        tool = WhaleAlertTool()
        r = tool.execute(mode="confirmed", min_btc=0.0)
        vals = [t["value_btc"] for t in r.data["transactions"]]
        self.assertEqual(vals, sorted(vals, reverse=True))

    # ── Very large values ────────────────────────────────────────

    @patch("agent.tools.whale_alert.httpx.Client")
    def test_very_large_btc(self, mock_cls):
        big_tx = {
            "hash": "bigtx", "time": 1700000000, "inputs": [],
            "out": [{"addr": "bigaddr", "value": 100000_00000000}],
        }
        def get_fn(url, **kw):
            resp = MagicMock()
            resp.json.return_value = {"txs": [big_tx]}
            resp.raise_for_status = MagicMock()
            return resp
        mock_cls.return_value = MockClient(get_fn=get_fn)
        tool = WhaleAlertTool()
        r = tool.execute(mode="mempool", min_btc=0.0)
        self.assertAlmostEqual(r.data["transactions"][0]["value_btc"], 100000.0)

    # ── Satoshi boundary ─────────────────────────────────────────

    @patch("agent.tools.whale_alert.httpx.Client")
    def test_one_satoshi(self, mock_cls):
        tiny_tx = {
            "hash": "tinytx", "time": 1700000000, "inputs": [],
            "out": [{"addr": "tinyaddr", "value": 1}],
        }
        def get_fn(url, **kw):
            resp = MagicMock()
            resp.json.return_value = {"txs": [tiny_tx]}
            resp.raise_for_status = MagicMock()
            return resp
        mock_cls.return_value = MockClient(get_fn=get_fn)
        tool = WhaleAlertTool()
        r = tool.execute(mode="mempool", min_btc=0.0)
        self.assertAlmostEqual(r.data["transactions"][0]["value_btc"], 1e-8)

    # ── Multi-output tx sums correctly ───────────────────────────

    @patch("agent.tools.whale_alert.httpx.Client")
    def test_multi_output_sum(self, mock_cls):
        tx = {
            "hash": "multi", "time": 1700000000, "inputs": [],
            "out": [
                {"addr": "a1", "value": 1000000000},   # 10 BTC
                {"addr": "a2", "value": 2000000000},   # 20 BTC
                {"addr": "a3", "value": 500000000},    # 5 BTC
            ],
        }
        def get_fn(url, **kw):
            resp = MagicMock()
            resp.json.return_value = {"txs": [tx]}
            resp.raise_for_status = MagicMock()
            return resp
        mock_cls.return_value = MockClient(get_fn=get_fn)
        tool = WhaleAlertTool()
        r = tool.execute(mode="mempool", min_btc=0.0)
        self.assertAlmostEqual(r.data["transactions"][0]["value_btc"], 35.0)

    # ── Output with no addr excluded ─────────────────────────────

    @patch("agent.tools.whale_alert.httpx.Client")
    def test_no_addr_excluded(self, mock_cls):
        tx = {
            "hash": "noaddr", "time": 1700000000, "inputs": [],
            "out": [
                {"value": 1000000000},                 # no addr
                {"addr": "real", "value": 500000000},
            ],
        }
        def get_fn(url, **kw):
            resp = MagicMock()
            resp.json.return_value = {"txs": [tx]}
            resp.raise_for_status = MagicMock()
            return resp
        mock_cls.return_value = MockClient(get_fn=get_fn)
        tool = WhaleAlertTool()
        r = tool.execute(mode="mempool", min_btc=0.0)
        outs = r.data["transactions"][0]["outputs"]
        self.assertEqual(len(outs), 1)
        self.assertEqual(outs[0]["addr"], "real")

    # ── Output formatting ────────────────────────────────────────

    @patch("agent.tools.whale_alert.httpx.Client", new_callable=lambda: _make_mock_client())
    def test_mempool_output_format(self, _):
        tool = WhaleAlertTool()
        r = tool.execute(mode="mempool", min_btc=0.0)
        self.assertIn("mempool", r.output)
        self.assertIn("Total:", r.output)
        self.assertIn("…", r.output)  # unconfirmed marker

    @patch("agent.tools.whale_alert.httpx.Client", new_callable=lambda: _make_mock_client())
    def test_confirmed_output_format(self, _):
        tool = WhaleAlertTool()
        r = tool.execute(mode="confirmed", min_btc=0.0)
        self.assertIn("block #", r.output)
        self.assertIn("✓", r.output)  # confirmed marker
        self.assertIn("Total:", r.output)

    # ── Bandit arm ───────────────────────────────────────────────

    def test_bandit_arm_exists(self):
        from agent.learning.bandit import DEFAULT_ARMS
        arm_names = [a.name for a in DEFAULT_ARMS]
        self.assertIn("crypto_whale_flows", arm_names)

    def test_bandit_arm_tools(self):
        from agent.learning.bandit import DEFAULT_ARMS
        arm = next(a for a in DEFAULT_ARMS if a.name == "crypto_whale_flows")
        self.assertIn("whale_alert", arm.tools)

    # ── Config has NO whale_alert_key ────────────────────────────

    def test_no_whale_alert_key_in_config(self):
        from agent.config.settings import AgentConfig
        cfg = AgentConfig.from_env()
        self.assertFalse(hasattr(cfg, "whale_alert_key"))

    # ── Constructor takes no api_key ─────────────────────────────

    def test_constructor_no_api_key(self):
        """WhaleAlertTool() should work with just cache= kwarg."""
        tool = WhaleAlertTool()
        self.assertIsNone(tool._cache)
        tool2 = WhaleAlertTool(cache=MagicMock())
        self.assertIsNotNone(tool2._cache)

    # ── LIVE TESTS (real network calls) ──────────────────────────

    def test_live_mempool(self):
        """Real call to blockchain.com mempool."""
        tool = WhaleAlertTool()
        r = tool.execute(mode="mempool", min_btc=0.001, limit=3)
        self.assertTrue(r.success)
        self.assertIsInstance(r.data["transactions"], list)
        self.assertEqual(r.data["mode"], "mempool")
        # Verify structure
        if r.data["transactions"]:
            tx = r.data["transactions"][0]
            self.assertIn("hash", tx)
            self.assertIn("value_btc", tx)
            self.assertFalse(tx["confirmed"])

    def test_live_confirmed(self):
        """Real call to blockchain.com latest block."""
        tool = WhaleAlertTool()
        r = tool.execute(mode="confirmed", min_btc=50.0, limit=5)
        self.assertTrue(r.success)
        self.assertEqual(r.data["mode"], "confirmed")
        # A real block should have at least some whale txs
        if r.data["transactions"]:
            tx = r.data["transactions"][0]
            self.assertTrue(tx["confirmed"])
            self.assertIn("block_height", tx)
            self.assertGreater(tx["block_height"], 0)
            self.assertGreaterEqual(tx["value_btc"], 50.0)


if __name__ == "__main__":
    unittest.main()
