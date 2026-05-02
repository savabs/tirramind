"""
Edge-case tests for agent/tools/drug_regulatory.py

Covers: all 3 modes (approvals, adverse_events, labels), count mode,
invalid mode, date validation, Elasticsearch errors, HTTP 404/429/500,
drug_name convenience filter, seriousness ratio, empty results,
cache integration, tool schema.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

import httpx

# ──────────────────────────────────────────────────────────────────
# Synthetic data factories
# ──────────────────────────────────────────────────────────────────


def _make_approvals_response(results=None, total=None):
    if results is None:
        results = [
            {
                "application_number": "NDA012345",
                "sponsor_name": "PharmaCorp",
                "products": [
                    {"brand_name": "Curall", "dosage_form": "TABLET"},
                ],
                "submissions": [
                    {
                        "submission_type": "ORIG",
                        "submission_status_date": "20260315",
                        "review_priority": "PRIORITY",
                        "submission_status": "AP",
                    }
                ],
            },
            {
                "application_number": "NDA067890",
                "sponsor_name": "BioInc",
                "products": [
                    {"brand_name": "Healex", "dosage_form": "INJECTION"},
                ],
                "submissions": [
                    {
                        "submission_type": "SUPPL",
                        "submission_status_date": "20260301",
                        "review_priority": "STANDARD",
                    },
                    {
                        "submission_type": "ORIG",
                        "submission_status_date": "20240101",
                        "review_priority": "PRIORITY",
                    },
                ],
            },
        ]
    if total is None:
        total = len(results)
    return {
        "meta": {"results": {"total": total, "skip": 0, "limit": 25}},
        "results": results,
    }


def _make_adverse_events_response(results=None, total=None):
    if results is None:
        results = [
            {
                "receivedate": "20260320",
                "serious": 1,
                "seriousnessdeath": 0,
                "patient": {
                    "drug": [{"medicinalproduct": "ASPIRIN"}],
                    "reaction": [
                        {"reactionmeddrapt": "Nausea"},
                        {"reactionmeddrapt": "Headache"},
                    ],
                },
            },
            {
                "receivedate": "20260319",
                "serious": 0,
                "patient": {
                    "drug": [{"medicinalproduct": "IBUPROFEN"}],
                    "reaction": [{"reactionmeddrapt": "Rash"}],
                },
            },
            {
                "receivedate": "20260318",
                "serious": 1,
                "patient": {
                    "drug": [
                        {"medicinalproduct": "ASPIRIN"},
                        {"medicinalproduct": "WARFARIN"},
                    ],
                    "reaction": [{"reactionmeddrapt": "GI Bleed"}],
                },
            },
        ]
    if total is None:
        total = len(results)
    return {
        "meta": {"results": {"total": total, "skip": 0, "limit": 25}},
        "results": results,
    }


def _make_labels_response(results=None, total=None):
    if results is None:
        results = [
            {
                "openfda": {
                    "brand_name": ["ASPIRIN"],
                    "generic_name": ["ACETYLSALICYLIC ACID"],
                },
                "boxed_warning": ["RISK OF BLEEDING"],
                "warnings": ["Use with caution in patients with bleeding disorders."],
            },
            {
                "openfda": {
                    "brand_name": ["TYLENOL"],
                    "generic_name": ["ACETAMINOPHEN"],
                },
                "warnings": ["Liver damage may occur."],
            },
        ]
    if total is None:
        total = len(results)
    return {
        "meta": {"results": {"total": total, "skip": 0, "limit": 25}},
        "results": results,
    }


def _make_counts_response(results=None, total=None):
    if results is None:
        results = [
            {"term": "NAUSEA", "count": 15000},
            {"term": "HEADACHE", "count": 12000},
            {"term": "FATIGUE", "count": 8000},
        ]
    return {
        "meta": {"results": {"total": total or 50000}},
        "results": results,
    }


def _mock_response(payload, status=200):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.json.return_value = payload
    resp.text = json.dumps(payload)
    if status >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(f"HTTP {status}", request=MagicMock(), response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


class TestDrugRegulatoryTool(unittest.TestCase):
    """Edge-case tests for DrugRegulatoryTool."""

    def _make_tool(self, cache=None):
        from agent.tools.drug_regulatory import DrugRegulatoryTool

        return DrugRegulatoryTool(cache=cache)

    # ── Mode validation ─────────────────────────────────────────────

    def test_invalid_mode(self):
        tool = self._make_tool()
        result = tool.execute(mode="recalls")
        self.assertFalse(result.success)
        self.assertIn("Invalid mode", result.output)

    def test_empty_mode(self):
        tool = self._make_tool()
        result = tool.execute(mode="")
        self.assertFalse(result.success)

    # ── Date validation ─────────────────────────────────────────────

    def test_invalid_date_start_format(self):
        tool = self._make_tool()
        result = tool.execute(mode="approvals", date_start="2026-03-01")
        self.assertFalse(result.success)
        self.assertIn("YYYYMMDD", result.output)

    def test_invalid_date_end_not_numeric(self):
        tool = self._make_tool()
        result = tool.execute(mode="approvals", date_end="abcdefgh")
        self.assertFalse(result.success)

    def test_valid_date_format(self):
        tool = self._make_tool()
        with patch("agent.tools.drug_regulatory.httpx.Client") as mock_cls:
            client = MagicMock()
            mock_cls.return_value.__enter__ = MagicMock(return_value=client)
            mock_cls.return_value.__exit__ = MagicMock(return_value=False)
            client.get.return_value = _mock_response(_make_approvals_response())

            result = tool.execute(mode="approvals", date_start="20260101", date_end="20260331")
            self.assertTrue(result.success)

    # ── Approvals mode ──────────────────────────────────────────────

    @patch("agent.tools.drug_regulatory.httpx.Client")
    def test_approvals_basic(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_approvals_response())

        tool = self._make_tool()
        result = tool.execute(mode="approvals")

        self.assertTrue(result.success)
        self.assertEqual(result.data["mode"], "approvals")
        self.assertEqual(len(result.data["results"]), 2)

    @patch("agent.tools.drug_regulatory.httpx.Client")
    def test_approvals_priority_review_detected(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_approvals_response())

        tool = self._make_tool()
        result = tool.execute(mode="approvals")

        # First result has PRIORITY review
        self.assertEqual(result.data["results"][0]["review_priority"], "PRIORITY")

    @patch("agent.tools.drug_regulatory.httpx.Client")
    def test_approvals_latest_submission_sorted(self, mock_cls):
        """Second drug has 2 submissions — should pick the most recent."""
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_approvals_response())

        tool = self._make_tool()
        result = tool.execute(mode="approvals")

        # BioInc's latest should be SUPPL (20260301), not ORIG (20240101)
        bioinc = result.data["results"][1]
        self.assertEqual(bioinc["latest_submission_type"], "SUPPL")
        self.assertEqual(bioinc["latest_submission_date"], "20260301")

    @patch("agent.tools.drug_regulatory.httpx.Client")
    def test_approvals_empty_products(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        payload = _make_approvals_response(
            [
                {"application_number": "NDA111", "products": [], "submissions": []},
            ]
        )
        client.get.return_value = _mock_response(payload)

        tool = self._make_tool()
        result = tool.execute(mode="approvals")
        self.assertTrue(result.success)

    # ── Adverse events mode ─────────────────────────────────────────

    @patch("agent.tools.drug_regulatory.httpx.Client")
    def test_adverse_events_basic(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_adverse_events_response())

        tool = self._make_tool()
        result = tool.execute(mode="adverse_events")

        self.assertTrue(result.success)
        self.assertEqual(result.data["mode"], "adverse_events")

    @patch("agent.tools.drug_regulatory.httpx.Client")
    def test_adverse_events_seriousness_ratio(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_adverse_events_response())

        tool = self._make_tool()
        result = tool.execute(mode="adverse_events")

        signals = result.data.get("signals", {})
        # 2 serious out of 3 = 0.667
        self.assertAlmostEqual(signals["seriousness_ratio"], 0.667, places=2)
        self.assertEqual(signals["serious_count"], 2)

    @patch("agent.tools.drug_regulatory.httpx.Client")
    def test_adverse_events_no_serious(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        payload = _make_adverse_events_response(
            [
                {
                    "receivedate": "20260320",
                    "serious": 0,
                    "patient": {
                        "drug": [{"medicinalproduct": "VITAMIN_C"}],
                        "reaction": [{"reactionmeddrapt": "Nothing"}],
                    },
                }
            ]
        )
        client.get.return_value = _mock_response(payload)

        tool = self._make_tool()
        result = tool.execute(mode="adverse_events")

        signals = result.data.get("signals", {})
        self.assertEqual(signals["seriousness_ratio"], 0.0)

    # ── Labels mode ─────────────────────────────────────────────────

    @patch("agent.tools.drug_regulatory.httpx.Client")
    def test_labels_basic(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_labels_response())

        tool = self._make_tool()
        result = tool.execute(mode="labels")

        self.assertTrue(result.success)
        self.assertEqual(result.data["mode"], "labels")
        # First label has boxed warning
        self.assertTrue(result.data["results"][0]["has_boxed_warning"])
        # Second has no boxed warning
        self.assertFalse(result.data["results"][1]["has_boxed_warning"])

    @patch("agent.tools.drug_regulatory.httpx.Client")
    def test_labels_missing_openfda(self, mock_cls):
        """Label record with no openfda field shouldn't crash."""
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        payload = _make_labels_response([{"openfda": {}, "warnings": []}])
        client.get.return_value = _mock_response(payload)

        tool = self._make_tool()
        result = tool.execute(mode="labels")
        self.assertTrue(result.success)

    # ── Count mode ──────────────────────────────────────────────────

    @patch("agent.tools.drug_regulatory.httpx.Client")
    def test_count_mode(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_counts_response())

        tool = self._make_tool()
        result = tool.execute(
            mode="adverse_events",
            count_field="patient.reaction.reactionmeddrapt",
        )

        self.assertTrue(result.success)
        self.assertIn("count_field", result.data)
        self.assertEqual(len(result.data["results"]), 3)
        self.assertEqual(result.data["results"][0]["term"], "NAUSEA")

    # ── Drug name convenience filter ────────────────────────────────

    @patch("agent.tools.drug_regulatory.httpx.Client")
    def test_drug_name_filter_approvals(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_approvals_response())

        tool = self._make_tool()
        result = tool.execute(mode="approvals", drug_name="Curall")

        # Verify the search parameter was built correctly
        call_args = client.get.call_args
        params = call_args.kwargs.get("params") or call_args[1].get("params", {})
        self.assertIn("products.brand_name", params.get("search", ""))

    @patch("agent.tools.drug_regulatory.httpx.Client")
    def test_drug_name_filter_adverse_events(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_adverse_events_response())

        tool = self._make_tool()
        result = tool.execute(mode="adverse_events", drug_name="aspirin")

        call_args = client.get.call_args
        params = call_args.kwargs.get("params") or call_args[1].get("params", {})
        self.assertIn("patient.drug.medicinalproduct", params.get("search", ""))

    # ── HTTP errors ─────────────────────────────────────────────────

    @patch("agent.tools.drug_regulatory.httpx.Client")
    def test_http_404_returns_no_results(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response({}, status=404)

        tool = self._make_tool()
        result = tool.execute(mode="approvals")
        self.assertTrue(result.success)
        self.assertIn("No results", result.output)

    @patch("agent.tools.drug_regulatory.httpx.Client")
    def test_http_429_rate_limit(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response({}, status=429)

        tool = self._make_tool()
        result = tool.execute(mode="approvals")
        self.assertFalse(result.success)
        self.assertIn("rate limit", result.output.lower())

    @patch("agent.tools.drug_regulatory.httpx.Client")
    def test_http_500_error(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response({}, status=500)

        tool = self._make_tool()
        result = tool.execute(mode="approvals")
        self.assertFalse(result.success)

    @patch("agent.tools.drug_regulatory.httpx.Client")
    def test_network_timeout(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.side_effect = httpx.ConnectTimeout("Connect timeout")

        tool = self._make_tool()
        result = tool.execute(mode="approvals")
        self.assertFalse(result.success)

    # ── Empty results ───────────────────────────────────────────────

    @patch("agent.tools.drug_regulatory.httpx.Client")
    def test_empty_results(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        payload = {"meta": {"results": {"total": 0}}, "results": []}
        client.get.return_value = _mock_response(payload)

        tool = self._make_tool()
        result = tool.execute(mode="approvals")
        self.assertTrue(result.success)
        self.assertIn("No results", result.output)

    # ── Cache integration ───────────────────────────────────────────

    @patch("agent.tools.drug_regulatory.httpx.Client")
    def test_cache_miss_stores(self, mock_cls):
        cache = MagicMock()
        cache.get.return_value = None

        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_approvals_response())

        tool = self._make_tool(cache=cache)
        result = tool.execute(mode="approvals")

        self.assertTrue(result.success)
        cache.put.assert_called_once()

    def test_cache_hit_skips_http(self):
        cache = MagicMock()
        cache.get.return_value = _make_approvals_response()

        tool = self._make_tool(cache=cache)
        result = tool.execute(mode="approvals")
        self.assertTrue(result.success)

    # ── Tool schema ─────────────────────────────────────────────────

    def test_tool_schema(self):
        tool = self._make_tool()
        self.assertEqual(tool.name, "drug_regulatory")
        self.assertIn("mode", tool.parameters["properties"])
        schema = tool.to_openai_tool()
        self.assertEqual(schema["function"]["name"], "drug_regulatory")

    def test_valid_modes_constant(self):
        from agent.tools.drug_regulatory import VALID_MODES

        self.assertEqual(VALID_MODES, frozenset({"approvals", "adverse_events", "labels"}))

    # ── kwargs passthrough ──────────────────────────────────────────

    @patch("agent.tools.drug_regulatory.httpx.Client")
    def test_extra_kwargs_ignored(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_approvals_response())

        tool = self._make_tool()
        result = tool.execute(mode="approvals", unknown_field="hello")
        self.assertTrue(result.success)

    # ── Limit clamping ──────────────────────────────────────────────

    @patch("agent.tools.drug_regulatory.httpx.Client")
    def test_limit_clamped_for_regular(self, mock_cls):
        """Limit for non-count queries capped at 100."""
        client = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        client.get.return_value = _mock_response(_make_approvals_response())

        tool = self._make_tool()
        tool.execute(mode="approvals", limit=500)

        call_args = client.get.call_args
        params = call_args.kwargs.get("params") or call_args[1].get("params", {})
        self.assertLessEqual(params.get("limit", 0), 100)


if __name__ == "__main__":
    unittest.main()
