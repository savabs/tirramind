"""
Edge-case tests for agent/tools/treasury_receipts.py

Covers: all 3 modes (cash_balance, deposits_withdrawals, public_debt),
invalid mode, date validation, JSON parsing, empty data, HTTP errors,
category filtering, signal computation, cache integration, tool schema.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

import httpx


# ──────────────────────────────────────────────────────────────────
# Synthetic data factories
# ──────────────────────────────────────────────────────────────────


def _make_cash_balance_response(entries=None):
    if entries is None:
        entries = [
            {
                "record_date": "2026-03-28",
                "account_type": "Federal Reserve Account",
                "close_today_bal": "750000",
                "open_month_bal": "700000",
                "open_fiscal_year_bal": "600000",
            },
            {
                "record_date": "2026-03-27",
                "account_type": "Federal Reserve Account",
                "close_today_bal": "740000",
                "open_month_bal": "700000",
                "open_fiscal_year_bal": "600000",
            },
        ]
    return {"data": entries, "meta": {"count": len(entries)}, "links": {}}


def _make_deposits_withdrawals_response(entries=None):
    if entries is None:
        entries = [
            {
                "record_date": "2026-03-28",
                "transaction_type": "Deposits",
                "transaction_catg": "Tax",
                "transaction_catg_desc": "Individual Income and Employment Taxes, Not Withheld",
                "transaction_today_amt": "5000",
                "transaction_mtd_amt": "150000",
                "transaction_fytd_amt": "1200000",
            },
            {
                "record_date": "2026-03-28",
                "transaction_type": "Withdrawals",
                "transaction_catg": "DoD",
                "transaction_catg_desc": "Dept of Defense Vendor Payments",
                "transaction_today_amt": "3000",
                "transaction_mtd_amt": "90000",
                "transaction_fytd_amt": "720000",
            },
            {
                "record_date": "2026-03-28",
                "transaction_type": "Deposits",
                "transaction_catg": "Customs",
                "transaction_catg_desc": "Customs Duties",
                "transaction_today_amt": "200",
                "transaction_mtd_amt": "6000",
                "transaction_fytd_amt": "48000",
            },
        ]
    return {"data": entries, "meta": {"count": len(entries)}, "links": {}}


def _make_public_debt_response(entries=None):
    if entries is None:
        entries = [
            {
                "record_date": "2026-03-28",
                "transaction_type": "Issues",
                "transaction_today_amt": "500000",
                "transaction_mtd_amt": "5000000",
                "transaction_fytd_amt": "30000000",
            },
            {
                "record_date": "2026-03-28",
                "transaction_type": "Redemptions",
                "transaction_today_amt": "480000",
                "transaction_mtd_amt": "4800000",
                "transaction_fytd_amt": "29000000",
            },
        ]
    return {"data": entries, "meta": {"count": len(entries)}, "links": {}}


def _mock_response(payload, status=200):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.json.return_value = payload
    resp.text = json.dumps(payload)
    if status >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status}", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


class TestTreasuryReceiptsTool(unittest.TestCase):
    """Edge-case tests for TreasuryReceiptsTool."""

    def _make_tool(self, cache=None):
        from agent.tools.treasury_receipts import TreasuryReceiptsTool

        return TreasuryReceiptsTool(cache=cache)

    # ── Mode validation ─────────────────────────────────────────────

    def test_invalid_mode_returns_error(self):
        tool = self._make_tool()
        result = tool.execute(mode="invalid")
        self.assertFalse(result.success)
        self.assertIn("Invalid mode", result.output)

    def test_empty_mode_returns_error(self):
        tool = self._make_tool()
        result = tool.execute(mode="")
        self.assertFalse(result.success)

    # ── Date validation ─────────────────────────────────────────────

    def test_invalid_start_date_format(self):
        tool = self._make_tool()
        result = tool.execute(mode="cash_balance", start_date="not-a-date")
        self.assertFalse(result.success)
        self.assertIn("Invalid start_date", result.output)

    def test_invalid_end_date_format(self):
        tool = self._make_tool()
        result = tool.execute(mode="cash_balance", end_date="2026/03/28")
        self.assertFalse(result.success)
        self.assertIn("Invalid end_date", result.output)

    def test_start_after_end_returns_error(self):
        tool = self._make_tool()
        result = tool.execute(
            mode="cash_balance", start_date="2026-04-01", end_date="2026-03-01"
        )
        self.assertFalse(result.success)
        self.assertIn("after", result.output)

    def test_valid_dates_accepted(self):
        tool = self._make_tool()
        with patch("agent.tools.treasury_receipts.httpx.Client") as mock_cls:
            client = MagicMock()
            mock_cls.return_value.__enter__ = MagicMock(return_value=client)
            mock_cls.return_value.__exit__ = MagicMock(return_value=False)
            client.get.return_value = _mock_response(_make_cash_balance_response())

            result = tool.execute(
                mode="cash_balance", start_date="2026-03-01", end_date="2026-03-28"
            )
            self.assertTrue(result.success)

    def test_leap_day_date(self):
        tool = self._make_tool()
        with patch("agent.tools.treasury_receipts.httpx.Client") as mock_cls:
            client = MagicMock()
            mock_cls.return_value.__enter__ = MagicMock(return_value=client)
            mock_cls.return_value.__exit__ = MagicMock(return_value=False)
            client.get.return_value = _mock_response(_make_cash_balance_response())

            result = tool.execute(mode="cash_balance", start_date="2024-02-29")
            self.assertTrue(result.success)

    def test_feb_29_non_leap_year_rejected(self):
        tool = self._make_tool()
        result = tool.execute(mode="cash_balance", start_date="2025-02-29")
        self.assertFalse(result.success)

    # ── Cash balance mode ───────────────────────────────────────────

    @patch("agent.tools.treasury_receipts.httpx.Client")
    def test_cash_balance_basic(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_cash_balance_response())

        tool = self._make_tool()
        result = tool.execute(mode="cash_balance")

        self.assertTrue(result.success)
        self.assertIn("TGA", result.output)
        self.assertIsNotNone(result.data)
        self.assertEqual(result.data["mode"], "cash_balance")
        self.assertEqual(len(result.data["records"]), 2)

    @patch("agent.tools.treasury_receipts.httpx.Client")
    def test_cash_balance_tga_delta_signal(self, mock_cls):
        """TGA delta computed when multiple Federal Reserve Account entries."""
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_cash_balance_response())

        tool = self._make_tool()
        result = tool.execute(mode="cash_balance")

        self.assertTrue(result.success)
        signals = result.data.get("signals", {})
        # 750000 - 740000 = 10000, delta pct = 10000/740000 * 100
        self.assertIn("tga_daily_change_abs", signals)
        self.assertAlmostEqual(signals["tga_daily_change_abs"], 10000.0)

    @patch("agent.tools.treasury_receipts.httpx.Client")
    def test_cash_balance_empty_data(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response({"data": [], "meta": {}, "links": {}})

        tool = self._make_tool()
        result = tool.execute(mode="cash_balance")

        self.assertTrue(result.success)
        self.assertIn("No data", result.output)

    @patch("agent.tools.treasury_receipts.httpx.Client")
    def test_cash_balance_missing_fields(self, mock_cls):
        """Records with missing balance fields shouldn't crash."""
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        payload = _make_cash_balance_response(
            [
                {"record_date": "2026-03-28", "account_type": "Test"},
            ]
        )
        client.get.return_value = _mock_response(payload)

        tool = self._make_tool()
        result = tool.execute(mode="cash_balance")
        self.assertTrue(result.success)

    # ── Deposits/withdrawals mode ───────────────────────────────────

    @patch("agent.tools.treasury_receipts.httpx.Client")
    def test_deposits_withdrawals_basic(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_deposits_withdrawals_response())

        tool = self._make_tool()
        result = tool.execute(mode="deposits_withdrawals")

        self.assertTrue(result.success)
        self.assertIn("Deposits", result.output)
        self.assertEqual(result.data["mode"], "deposits_withdrawals")

    @patch("agent.tools.treasury_receipts.httpx.Client")
    def test_deposits_withdrawals_net_flow_signal(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_deposits_withdrawals_response())

        tool = self._make_tool()
        result = tool.execute(mode="deposits_withdrawals")

        signals = result.data.get("signals", {})
        # deposits_today = 5000 + 200 = 5200, withdrawals_today = 3000
        self.assertEqual(signals.get("total_deposits_today"), 5200.0)
        self.assertEqual(signals.get("total_withdrawals_today"), 3000.0)
        self.assertEqual(signals.get("net_flow_today"), 2200.0)

    @patch("agent.tools.treasury_receipts.httpx.Client")
    def test_deposits_withdrawals_category_filter(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_deposits_withdrawals_response())

        tool = self._make_tool()
        result = tool.execute(mode="deposits_withdrawals", category_filter="customs")

        self.assertTrue(result.success)
        # Only the customs entry should remain
        self.assertEqual(len(result.data["records"]), 1)
        self.assertIn("Customs", result.data["records"][0]["category"])

    @patch("agent.tools.treasury_receipts.httpx.Client")
    def test_deposits_withdrawals_filter_no_match(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_deposits_withdrawals_response())

        tool = self._make_tool()
        result = tool.execute(
            mode="deposits_withdrawals", category_filter="nonexistent"
        )

        self.assertTrue(result.success)
        self.assertIn("Records: 0", result.output)

    # ── Public debt mode ────────────────────────────────────────────

    @patch("agent.tools.treasury_receipts.httpx.Client")
    def test_public_debt_basic(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_public_debt_response())

        tool = self._make_tool()
        result = tool.execute(mode="public_debt")

        self.assertTrue(result.success)
        self.assertEqual(result.data["mode"], "public_debt")
        self.assertEqual(len(result.data["records"]), 2)

    # ── HTTP errors ─────────────────────────────────────────────────

    @patch("agent.tools.treasury_receipts.httpx.Client")
    def test_http_500_error(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response({}, status=500)

        tool = self._make_tool()
        result = tool.execute(mode="cash_balance")
        self.assertFalse(result.success)
        self.assertIn("error", result.output.lower())

    @patch("agent.tools.treasury_receipts.httpx.Client")
    def test_http_404_error(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response({}, status=404)

        tool = self._make_tool()
        result = tool.execute(mode="cash_balance")
        self.assertFalse(result.success)

    @patch("agent.tools.treasury_receipts.httpx.Client")
    def test_network_timeout(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.side_effect = httpx.ConnectTimeout("Connection timed out")

        tool = self._make_tool()
        result = tool.execute(mode="cash_balance")
        self.assertFalse(result.success)
        self.assertIn("error", result.output.lower())

    @patch("agent.tools.treasury_receipts.httpx.Client")
    def test_malformed_json_response(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.side_effect = json.JSONDecodeError("bad json", "", 0)
        client.get.return_value = resp

        tool = self._make_tool()
        result = tool.execute(mode="cash_balance")
        self.assertFalse(result.success)

    # ── Amount parsing ──────────────────────────────────────────────

    def test_parse_amount_valid(self):
        from agent.tools.treasury_receipts import _parse_amount

        self.assertEqual(_parse_amount("1000"), 1000.0)
        self.assertEqual(_parse_amount("1,234,567"), 1234567.0)
        self.assertEqual(_parse_amount("  500  "), 500.0)

    def test_parse_amount_invalid(self):
        from agent.tools.treasury_receipts import _parse_amount

        self.assertIsNone(_parse_amount(None))
        self.assertIsNone(_parse_amount(""))
        self.assertIsNone(_parse_amount("-"))
        self.assertIsNone(_parse_amount("null"))
        self.assertIsNone(_parse_amount("abc"))

    def test_parse_amount_negative(self):
        from agent.tools.treasury_receipts import _parse_amount

        self.assertEqual(_parse_amount("-500"), -500.0)

    # ── top_n clamping ──────────────────────────────────────────────

    @patch("agent.tools.treasury_receipts.httpx.Client")
    def test_top_n_clamping(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        # Create 5 records
        entries = [
            {
                "record_date": f"2026-03-{20 + i:02d}",
                "account_type": "Federal Reserve Account",
                "close_today_bal": str(700000 + i * 1000),
            }
            for i in range(5)
        ]
        client.get.return_value = _mock_response(
            {"data": entries, "meta": {}, "links": {}}
        )

        tool = self._make_tool()
        result = tool.execute(mode="cash_balance", top_n=2)
        self.assertTrue(result.success)
        self.assertEqual(len(result.data["records"]), 2)

    @patch("agent.tools.treasury_receipts.httpx.Client")
    def test_top_n_zero_clamped_to_one(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_cash_balance_response())

        tool = self._make_tool()
        result = tool.execute(mode="cash_balance", top_n=0)
        self.assertTrue(result.success)
        self.assertEqual(len(result.data["records"]), 1)

    # ── Cache integration ───────────────────────────────────────────

    @patch("agent.tools.treasury_receipts.httpx.Client")
    def test_cache_miss_then_stores(self, mock_cls):
        cache = MagicMock()
        cache.get.return_value = None

        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_cash_balance_response())

        tool = self._make_tool(cache=cache)
        result = tool.execute(mode="cash_balance")

        self.assertTrue(result.success)
        cache.get.assert_called_once()
        cache.put.assert_called_once()

    def test_cache_hit_skips_http(self):
        cache = MagicMock()
        cache.get.return_value = _make_cash_balance_response()["data"]

        tool = self._make_tool(cache=cache)
        # Should NOT make any HTTP calls since cache returns data
        result = tool.execute(mode="cash_balance")

        self.assertTrue(result.success)
        cache.get.assert_called_once()

    # ── Tool schema ─────────────────────────────────────────────────

    def test_tool_schema_valid(self):
        tool = self._make_tool()
        self.assertEqual(tool.name, "treasury_receipts")
        self.assertIn("type", tool.parameters)
        self.assertIn("mode", tool.parameters["properties"])
        schema = tool.to_openai_tool()
        self.assertEqual(schema["function"]["name"], "treasury_receipts")

    def test_valid_modes_constant(self):
        from agent.tools.treasury_receipts import VALID_MODES

        self.assertEqual(
            VALID_MODES,
            frozenset({"cash_balance", "deposits_withdrawals", "public_debt"}),
        )

    # ── kwargs passthrough ──────────────────────────────────────────

    @patch("agent.tools.treasury_receipts.httpx.Client")
    def test_extra_kwargs_ignored(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_cash_balance_response())

        tool = self._make_tool()
        result = tool.execute(mode="cash_balance", unknown_param="hello")
        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()
