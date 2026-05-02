"""
Edge-case tests for agent/tools/sovereign_debt.py

Covers: all 5 modes (us_yields, eu_yields, jp_yields, uk_gilts, spreads),
invalid modes, month validation, XML/CSV parsing, missing data, HTTP errors,
spread computation, cache integration, tool schema, registry (35 tools, 23 arms).
"""

from __future__ import annotations

import textwrap
import unittest
from unittest.mock import MagicMock, patch

import httpx

# ──────────────────────────────────────────────────────────────────
# Synthetic data factories
# ──────────────────────────────────────────────────────────────────

_US_TREASURY_XML_TEMPLATE = textwrap.dedent(
    """\
<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"
      xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices">
  <entry>
    <content type="application/xml">
      <m:properties>
        <d:NEW_DATE>2026-03-27T00:00:00</d:NEW_DATE>
        <d:BC_1MONTH>{m1}</d:BC_1MONTH>
        <d:BC_3MONTH>{m3}</d:BC_3MONTH>
        <d:BC_2YEAR>{y2}</d:BC_2YEAR>
        <d:BC_10YEAR>{y10}</d:BC_10YEAR>
        <d:BC_30YEAR>{y30}</d:BC_30YEAR>
      </m:properties>
    </content>
  </entry>
</feed>"""
)


def _make_us_xml(m1="4.20", m3="4.30", y2="3.88", y10="4.44", y30="4.98"):
    return _US_TREASURY_XML_TEMPLATE.format(m1=m1, m3=m3, y2=y2, y10=y10, y30=y30)


_US_MULTI_ENTRY_XML = textwrap.dedent(
    """\
<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"
      xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices">
  <entry>
    <content type="application/xml">
      <m:properties>
        <d:NEW_DATE>2026-03-26T00:00:00</d:NEW_DATE>
        <d:BC_2YEAR>3.85</d:BC_2YEAR>
        <d:BC_10YEAR>4.40</d:BC_10YEAR>
      </m:properties>
    </content>
  </entry>
  <entry>
    <content type="application/xml">
      <m:properties>
        <d:NEW_DATE>2026-03-27T00:00:00</d:NEW_DATE>
        <d:BC_2YEAR>3.88</d:BC_2YEAR>
        <d:BC_10YEAR>4.44</d:BC_10YEAR>
      </m:properties>
    </content>
  </entry>
</feed>"""
)


def _make_ecb_csv(country="DE", values=None):
    """Build ECB IRS CSV for a single country."""
    if values is None:
        values = [("2026-01", 2.5), ("2026-02", 2.744)]
    header = "KEY,FREQ,REF_AREA,INSTRUMENT,MATURITY,DATA_TYPE,COUNT_AREA,CURRENCY,CALC_METHOD,DECIMALS,TIME_PERIOD,OBS_VALUE\n"
    rows = ""
    for period, val in values:
        rows += f"IRS.M.{country}.L.L40.CI.0000.EUR.N.Z,M,{country},L,L40,CI,0000,EUR,N,Z,{period},{val}\n"
    return header + rows


def _make_jp_csv(rows=None):
    """Build Japan MOF JGB yield CSV."""
    if rows is None:
        rows = [
            ("2026/3/26", "0.9", "1.5", "2.0", "2.286", "3.0", "3.489"),
            ("2026/3/27", "0.95", "1.55", "2.05", "2.30", "3.05", "3.50"),
        ]
    header0 = "Interest Rate (March 2026),,,,,,(Unit : %)\n"
    header1 = "Date,1Y,2Y,5Y,10Y,20Y,30Y\n"
    data = ""
    for r in rows:
        data += ",".join(r) + "\n"
    return header0 + header1 + data


def _make_uk_xml(auctions=None):
    """Build UK DMO gilt issuance XML."""
    if auctions is None:
        auctions = [
            {
                "name": "2% Treasury Gilt 2025",
                "isin": "GB0001234567",
                "date": "2025-11-20T00:00:00",
                "type": "Auction",
                "nominal": "3000.00",
                "price": "99.50",
                "yield": "2.05",
            },
        ]
    items = ""
    for a in auctions:
        items += (
            f'<View_Gilt_Issuance_History INSTRUMENT_NAME="{a["name"]}" '
            f'ISIN_CODE="{a["isin"]}" ACTUAL_DATE="{a["date"]}" '
            f'ISSUANCE_TYPE="{a["type"]}" NOMINAL_ISSUED="{a["nominal"]}" '
            f'ISSUE_CLEAN_PRICE="{a["price"]}" ISSUE_YIELD="{a["yield"]}"/>\n'
        )
    return f'<?xml version="1.0" encoding="utf-8"?><DataReport>{items}</DataReport>'


def _mock_response(text="", status_code=200, is_error=False):
    """Build a mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if is_error:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError("Error", request=MagicMock(), response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


# ──────────────────────────────────────────────────────────────────
# 1. Tool metadata and schema
# ──────────────────────────────────────────────────────────────────


class TestToolMetadata(unittest.TestCase):
    def test_name(self):
        from agent.tools.sovereign_debt import SovereignDebtTool

        self.assertEqual(SovereignDebtTool().name, "sovereign_debt")

    def test_description_not_empty(self):
        from agent.tools.sovereign_debt import SovereignDebtTool

        self.assertTrue(len(SovereignDebtTool().description) > 30)

    def test_parameters_schema(self):
        from agent.tools.sovereign_debt import SovereignDebtTool

        p = SovereignDebtTool().parameters
        self.assertEqual(p["type"], "object")
        self.assertIn("mode", p["properties"])
        self.assertIn("month", p["properties"])
        self.assertIn("countries", p["properties"])
        self.assertIn("limit", p["properties"])
        self.assertEqual(p["required"], ["mode"])

    def test_valid_modes_constant(self):
        from agent.tools.sovereign_debt import VALID_MODES

        self.assertEqual(VALID_MODES, {"us_yields", "eu_yields", "jp_yields", "uk_gilts", "spreads"})


# ──────────────────────────────────────────────────────────────────
# 2. Invalid mode / missing mode
# ──────────────────────────────────────────────────────────────────


class TestInvalidMode(unittest.TestCase):
    def test_invalid_mode(self):
        from agent.tools.sovereign_debt import SovereignDebtTool

        r = SovereignDebtTool().execute(mode="bogus")
        self.assertFalse(r.success)
        self.assertIn("Invalid mode", r.output)

    def test_empty_mode(self):
        from agent.tools.sovereign_debt import SovereignDebtTool

        r = SovereignDebtTool().execute(mode="")
        self.assertFalse(r.success)

    def test_no_mode(self):
        from agent.tools.sovereign_debt import SovereignDebtTool

        r = SovereignDebtTool().execute()
        self.assertFalse(r.success)


# ──────────────────────────────────────────────────────────────────
# 3. Month validation
# ──────────────────────────────────────────────────────────────────


class TestMonthValidation(unittest.TestCase):
    def test_bad_month_format(self):
        from agent.tools.sovereign_debt import SovereignDebtTool

        r = SovereignDebtTool().execute(mode="us_yields", month="03-2026")
        self.assertFalse(r.success)
        self.assertIn("Invalid month", r.output)

    def test_month_no_dash(self):
        from agent.tools.sovereign_debt import SovereignDebtTool

        r = SovereignDebtTool().execute(mode="us_yields", month="202603")
        self.assertFalse(r.success)

    def test_month_invalid_month_number(self):
        from agent.tools.sovereign_debt import SovereignDebtTool

        r = SovereignDebtTool().execute(mode="us_yields", month="2026-13")
        self.assertFalse(r.success)

    def test_month_00(self):
        from agent.tools.sovereign_debt import SovereignDebtTool

        r = SovereignDebtTool().execute(mode="us_yields", month="2026-00")
        self.assertFalse(r.success)

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_valid_month_passes(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        mock_get.return_value = _mock_response(_make_us_xml())
        r = SovereignDebtTool().execute(mode="us_yields", month="2026-03")
        self.assertTrue(r.success)


# ──────────────────────────────────────────────────────────────────
# 4. US Treasury yields — valid
# ──────────────────────────────────────────────────────────────────


class TestUSYieldsValid(unittest.TestCase):
    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_basic(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        mock_get.return_value = _mock_response(_make_us_xml())
        r = SovereignDebtTool().execute(mode="us_yields", month="2026-03")
        self.assertTrue(r.success)
        self.assertEqual(r.data["entries"], 1)
        rec = r.data["records"][0]
        self.assertEqual(rec["date"], "2026-03-27")
        self.assertAlmostEqual(rec["yields"]["2y"], 3.88)
        self.assertAlmostEqual(rec["yields"]["10y"], 4.44)

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_2s10s_spread(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        mock_get.return_value = _mock_response(_make_us_xml(y2="3.00", y10="4.00"))
        r = SovereignDebtTool().execute(mode="us_yields", month="2026-03")
        self.assertTrue(r.success)
        rec = r.data["records"][0]
        self.assertAlmostEqual(rec["curve_2s10s"], 1.0)

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_3m10y_spread(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        mock_get.return_value = _mock_response(_make_us_xml(m3="4.50", y10="4.00"))
        r = SovereignDebtTool().execute(mode="us_yields", month="2026-03")
        rec = r.data["records"][0]
        self.assertAlmostEqual(rec["curve_3m10y"], -0.5)  # inverted

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_multi_entry(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        mock_get.return_value = _mock_response(_US_MULTI_ENTRY_XML)
        r = SovereignDebtTool().execute(mode="us_yields", month="2026-03")
        self.assertTrue(r.success)
        self.assertEqual(r.data["entries"], 2)

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_default_month(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        mock_get.return_value = _mock_response(_make_us_xml())
        r = SovereignDebtTool().execute(mode="us_yields")  # no month param
        self.assertTrue(r.success)

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_output_contains_curve(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        mock_get.return_value = _mock_response(_make_us_xml())
        r = SovereignDebtTool().execute(mode="us_yields", month="2026-03")
        self.assertIn("2s10s", r.output)


# ──────────────────────────────────────────────────────────────────
# 5. US Treasury yields — edge cases
# ──────────────────────────────────────────────────────────────────


class TestUSYieldsEdge(unittest.TestCase):
    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_missing_maturity(self, mock_get):
        """Some maturities may be empty (newly introduced tenors)."""
        from agent.tools.sovereign_debt import SovereignDebtTool

        xml = _make_us_xml(m1="", m3="4.30", y2="3.88", y10="4.44", y30="")
        mock_get.return_value = _mock_response(xml)
        r = SovereignDebtTool().execute(mode="us_yields", month="2026-03")
        self.assertTrue(r.success)
        rec = r.data["records"][0]
        self.assertNotIn("1m", rec["yields"])
        self.assertNotIn("30y", rec["yields"])
        self.assertIn("2y", rec["yields"])

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_dash_value(self, mock_get):
        """US Treasury uses '-' for unavailable maturities."""
        from agent.tools.sovereign_debt import SovereignDebtTool

        xml = _make_us_xml(m1="-", y2="3.88", y10="4.44", y30="-")
        mock_get.return_value = _mock_response(xml)
        r = SovereignDebtTool().execute(mode="us_yields", month="2026-03")
        self.assertTrue(r.success)
        rec = r.data["records"][0]
        self.assertNotIn("1m", rec["yields"])

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_empty_xml(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        xml = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        mock_get.return_value = _mock_response(xml)
        r = SovereignDebtTool().execute(mode="us_yields", month="2026-03")
        self.assertFalse(r.success)
        self.assertIn("No US Treasury data", r.output)

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_malformed_xml(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        mock_get.return_value = _mock_response("<not valid xml><<<")
        r = SovereignDebtTool().execute(mode="us_yields", month="2026-03")
        self.assertFalse(r.success)

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_http_500(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        mock_get.return_value = _mock_response(status_code=500, is_error=True)
        r = SovereignDebtTool().execute(mode="us_yields", month="2026-03")
        self.assertFalse(r.success)
        self.assertIn("HTTP 500", r.output)

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_timeout(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        mock_get.side_effect = httpx.ConnectTimeout("timeout")
        r = SovereignDebtTool().execute(mode="us_yields", month="2026-03")
        self.assertFalse(r.success)
        self.assertIn("failed", r.output.lower())

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_curve_none_when_2y_missing(self, mock_get):
        """2s10s should be None when 2Y is missing."""
        from agent.tools.sovereign_debt import SovereignDebtTool

        xml = _make_us_xml(m3="4.0", y2="", y10="4.44", y30="5.0")
        mock_get.return_value = _mock_response(xml)
        r = SovereignDebtTool().execute(mode="us_yields", month="2026-03")
        self.assertTrue(r.success)
        rec = r.data["records"][0]
        self.assertIsNone(rec["curve_2s10s"])


# ──────────────────────────────────────────────────────────────────
# 6. EU yields — valid
# ──────────────────────────────────────────────────────────────────


class TestEUYieldsValid(unittest.TestCase):
    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_single_country(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        mock_get.return_value = _mock_response(_make_ecb_csv("DE"))
        r = SovereignDebtTool().execute(mode="eu_yields", countries=["DE"], month="2026-03")
        self.assertTrue(r.success)
        self.assertIn("DE", r.data["countries"])
        self.assertEqual(len(r.data["countries"]["DE"]), 2)

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_multiple_countries(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        def side_effect(url, **kwargs):
            if "M.DE." in url:
                return _mock_response(_make_ecb_csv("DE", [("2026-02", 2.744)]))
            elif "M.IT." in url:
                return _mock_response(_make_ecb_csv("IT", [("2026-02", 3.388)]))
            return _mock_response("", status_code=404, is_error=True)

        mock_get.side_effect = side_effect
        r = SovereignDebtTool().execute(mode="eu_yields", countries=["DE", "IT"], month="2026-03")
        self.assertTrue(r.success)
        self.assertIn("DE", r.data["countries"])
        self.assertIn("IT", r.data["countries"])

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_output_contains_yields(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        mock_get.return_value = _mock_response(_make_ecb_csv("DE", [("2026-02", 2.744)]))
        r = SovereignDebtTool().execute(mode="eu_yields", countries=["DE"], month="2026-03")
        self.assertIn("2.74", r.output)


# ──────────────────────────────────────────────────────────────────
# 7. EU yields — edge cases
# ──────────────────────────────────────────────────────────────────


class TestEUYieldsEdge(unittest.TestCase):
    def test_invalid_country_codes(self):
        from agent.tools.sovereign_debt import SovereignDebtTool

        r = SovereignDebtTool().execute(mode="eu_yields", countries=["XX", "ZZ"], month="2026-03")
        self.assertFalse(r.success)
        self.assertIn("No valid EU country", r.output)

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_empty_countries_uses_defaults(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        # Empty list is falsy → falls back to _EU_COUNTRIES_DEFAULT
        mock_get.return_value = _mock_response(_make_ecb_csv("DE"))
        r = SovereignDebtTool().execute(mode="eu_yields", countries=[], month="2026-03")
        # Should succeed using default country list
        self.assertTrue(r.success)

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_one_country_404(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        def side_effect(url, **kwargs):
            if "M.DE." in url:
                return _mock_response(_make_ecb_csv("DE", [("2026-02", 2.744)]))
            # EE returns 404
            resp = MagicMock()
            resp.status_code = 404
            resp.raise_for_status.return_value = None
            return resp

        mock_get.side_effect = side_effect
        r = SovereignDebtTool().execute(mode="eu_yields", countries=["DE", "EE"], month="2026-03")
        self.assertTrue(r.success)  # succeeds with partial data
        self.assertIn("DE", r.data["countries"])
        self.assertIsNotNone(r.data.get("errors"))

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_all_countries_fail(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        mock_get.side_effect = httpx.ConnectTimeout("timeout")
        r = SovereignDebtTool().execute(mode="eu_yields", countries=["DE", "FR"], month="2026-03")
        self.assertFalse(r.success)

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_ecb_empty_csv(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        header = "KEY,FREQ,REF_AREA,INSTRUMENT,MATURITY,DATA_TYPE,COUNT_AREA,CURRENCY,CALC_METHOD,DECIMALS,TIME_PERIOD,OBS_VALUE\n"
        mock_get.return_value = _mock_response(header)
        r = SovereignDebtTool().execute(mode="eu_yields", countries=["DE"], month="2026-03")
        # Header only, no data rows → no records → failure
        self.assertFalse(r.success)

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_country_case_insensitive(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        mock_get.return_value = _mock_response(_make_ecb_csv("DE"))
        r = SovereignDebtTool().execute(mode="eu_yields", countries=["de"], month="2026-03")
        self.assertTrue(r.success)


# ──────────────────────────────────────────────────────────────────
# 8. Japan yields — valid
# ──────────────────────────────────────────────────────────────────


class TestJPYieldsValid(unittest.TestCase):
    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_basic(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        mock_get.return_value = _mock_response(_make_jp_csv())
        r = SovereignDebtTool().execute(mode="jp_yields")
        self.assertTrue(r.success)
        self.assertEqual(r.data["entries"], 2)
        rec = r.data["records"][0]
        self.assertEqual(rec["date"], "2026-03-26")
        self.assertIn("10y", rec["yields"])

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_output_contains_10y(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        mock_get.return_value = _mock_response(_make_jp_csv())
        r = SovereignDebtTool().execute(mode="jp_yields")
        self.assertIn("10Y=", r.output)


# ──────────────────────────────────────────────────────────────────
# 9. Japan yields — edge cases
# ──────────────────────────────────────────────────────────────────


class TestJPYieldsEdge(unittest.TestCase):
    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_dash_values(self, mock_get):
        """Japan pre-1980s data has '-' for newer tenors."""
        from agent.tools.sovereign_debt import SovereignDebtTool

        csv = _make_jp_csv([("2026/3/26", "-", "1.5", "-", "2.286", "-", "-")])
        mock_get.return_value = _mock_response(csv)
        r = SovereignDebtTool().execute(mode="jp_yields")
        self.assertTrue(r.success)
        rec = r.data["records"][0]
        self.assertNotIn("1y", rec["yields"])
        self.assertIn("2y", rec["yields"])

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_empty_csv(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        mock_get.return_value = _mock_response("Title\nDate,1Y,2Y\n")
        r = SovereignDebtTool().execute(mode="jp_yields")
        self.assertFalse(r.success)

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_malformed_date(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        csv = _make_jp_csv([("not-a-date", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5")])
        mock_get.return_value = _mock_response(csv)
        r = SovereignDebtTool().execute(mode="jp_yields")
        self.assertFalse(r.success)  # no valid records

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_timeout(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        mock_get.side_effect = httpx.ConnectTimeout("timeout")
        r = SovereignDebtTool().execute(mode="jp_yields")
        self.assertFalse(r.success)

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_http_500(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        mock_get.return_value = _mock_response(status_code=500, is_error=True)
        r = SovereignDebtTool().execute(mode="jp_yields")
        self.assertFalse(r.success)

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_all_dash_row(self, mock_get):
        """Row where all yields are '-' should be skipped."""
        from agent.tools.sovereign_debt import SovereignDebtTool

        csv = _make_jp_csv(
            [
                ("2026/3/26", "-", "-", "-", "-", "-", "-"),
                ("2026/3/27", "0.9", "1.5", "2.0", "2.3", "3.0", "3.5"),
            ]
        )
        mock_get.return_value = _mock_response(csv)
        r = SovereignDebtTool().execute(mode="jp_yields")
        self.assertTrue(r.success)
        self.assertEqual(r.data["entries"], 1)  # only the second row


# ──────────────────────────────────────────────────────────────────
# 10. UK gilts — valid
# ──────────────────────────────────────────────────────────────────


class TestUKGiltsValid(unittest.TestCase):
    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_basic(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        mock_get.return_value = _mock_response(_make_uk_xml())
        r = SovereignDebtTool().execute(mode="uk_gilts")
        self.assertTrue(r.success)
        self.assertEqual(r.data["total_auctions"], 1)
        rec = r.data["records"][0]
        self.assertEqual(rec["date"], "2025-11-20")
        self.assertAlmostEqual(rec["yield_pct"], 2.05)

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_limit(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        auctions = [
            {
                "name": f"Gilt {i}",
                "isin": f"GB{i:010d}",
                "date": f"2025-{i + 1:02d}-15T00:00:00",
                "type": "Auction",
                "nominal": "3000.00",
                "price": "99.50",
                "yield": f"{2.0 + i * 0.1:.2f}",
            }
            for i in range(10)
        ]
        mock_get.return_value = _mock_response(_make_uk_xml(auctions))
        r = SovereignDebtTool().execute(mode="uk_gilts", limit=3)
        self.assertTrue(r.success)
        self.assertEqual(r.data["total_auctions"], 10)
        self.assertEqual(len(r.data["records"]), 3)

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_sorted_desc(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        auctions = [
            {
                "name": "Old",
                "isin": "GB001",
                "date": "2020-01-01T00:00:00",
                "type": "Auction",
                "nominal": "1000",
                "price": "99",
                "yield": "1.5",
            },
            {
                "name": "New",
                "isin": "GB002",
                "date": "2025-12-01T00:00:00",
                "type": "Auction",
                "nominal": "2000",
                "price": "100",
                "yield": "2.5",
            },
        ]
        mock_get.return_value = _mock_response(_make_uk_xml(auctions))
        r = SovereignDebtTool().execute(mode="uk_gilts")
        self.assertEqual(r.data["records"][0]["date"], "2025-12-01")


# ──────────────────────────────────────────────────────────────────
# 11. UK gilts — edge cases
# ──────────────────────────────────────────────────────────────────


class TestUKGiltsEdge(unittest.TestCase):
    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_empty_xml(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        mock_get.return_value = _mock_response('<?xml version="1.0"?><DataReport></DataReport>')
        r = SovereignDebtTool().execute(mode="uk_gilts")
        self.assertFalse(r.success)

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_malformed_xml(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        mock_get.return_value = _mock_response("<<<not xml>>>")
        r = SovereignDebtTool().execute(mode="uk_gilts")
        self.assertFalse(r.success)

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_timeout(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        mock_get.side_effect = httpx.ConnectTimeout("timeout")
        r = SovereignDebtTool().execute(mode="uk_gilts")
        self.assertFalse(r.success)

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_missing_yield(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        xml = (
            '<?xml version="1.0"?><DataReport>'
            '<View_Gilt_Issuance_History INSTRUMENT_NAME="Gilt" '
            'ISIN_CODE="GB001" ACTUAL_DATE="2025-11-20T00:00:00" '
            'ISSUANCE_TYPE="Auction" NOMINAL_ISSUED="3000" />'
            "</DataReport>"
        )
        mock_get.return_value = _mock_response(xml)
        r = SovereignDebtTool().execute(mode="uk_gilts")
        self.assertTrue(r.success)
        rec = r.data["records"][0]
        self.assertIsNone(rec["yield_pct"])

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_auction_missing_date_skipped(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        xml = (
            '<?xml version="1.0"?><DataReport>'
            '<View_Gilt_Issuance_History INSTRUMENT_NAME="Gilt" '
            'ISIN_CODE="GB001" ISSUANCE_TYPE="Auction" />'
            "</DataReport>"
        )
        mock_get.return_value = _mock_response(xml)
        r = SovereignDebtTool().execute(mode="uk_gilts")
        self.assertFalse(r.success)  # no valid records


# ──────────────────────────────────────────────────────────────────
# 12. Spreads — valid
# ──────────────────────────────────────────────────────────────────


class TestSpreadsValid(unittest.TestCase):
    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_basic_spread(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        def side_effect(url, **kwargs):
            if "M.DE." in url:
                return _mock_response(_make_ecb_csv("DE", [("2026-02", 2.744)]))
            elif "M.IT." in url:
                return _mock_response(_make_ecb_csv("IT", [("2026-02", 3.388)]))
            # US Treasury for curve
            if "treasury.gov" in url:
                return _mock_response(_make_us_xml())
            return _mock_response("", status_code=404, is_error=True)

        mock_get.side_effect = side_effect
        r = SovereignDebtTool().execute(mode="spreads", countries=["IT"], month="2026-03")
        self.assertTrue(r.success)
        self.assertAlmostEqual(r.data["de_benchmark_yield"], 2.744)
        self.assertEqual(len(r.data["spreads"]), 1)
        self.assertEqual(r.data["spreads"][0]["country"], "IT")
        self.assertAlmostEqual(r.data["spreads"][0]["spread_vs_de"], 0.644)

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_de_auto_included(self, mock_get):
        """DE should be auto-included even if not in countries list."""
        from agent.tools.sovereign_debt import SovereignDebtTool

        calls = []

        def side_effect(url, **kwargs):
            calls.append(url)
            if "M.DE." in url:
                return _mock_response(_make_ecb_csv("DE", [("2026-02", 2.744)]))
            elif "M.IT." in url:
                return _mock_response(_make_ecb_csv("IT", [("2026-02", 3.388)]))
            if "treasury.gov" in url:
                return _mock_response(_make_us_xml())
            return _mock_response("", status_code=404, is_error=True)

        mock_get.side_effect = side_effect
        r = SovereignDebtTool().execute(mode="spreads", countries=["IT"], month="2026-03")
        self.assertTrue(r.success)
        # DE should have been fetched
        de_urls = [u for u in calls if "M.DE." in u]
        self.assertTrue(len(de_urls) > 0)

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_spread_output_format(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        def side_effect(url, **kwargs):
            if "M.DE." in url:
                return _mock_response(_make_ecb_csv("DE", [("2026-02", 2.744)]))
            elif "M.GR." in url:
                return _mock_response(_make_ecb_csv("GR", [("2026-02", 3.39)]))
            if "treasury.gov" in url:
                return _mock_response(_make_us_xml())
            return _mock_response("", status_code=404, is_error=True)

        mock_get.side_effect = side_effect
        r = SovereignDebtTool().execute(mode="spreads", countries=["GR"], month="2026-03")
        self.assertIn("GR=", r.output)
        self.assertIn("2.74", r.output)  # DE benchmark

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_us_curve_included(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        def side_effect(url, **kwargs):
            if "M.DE." in url:
                return _mock_response(_make_ecb_csv("DE", [("2026-02", 2.744)]))
            elif "M.IT." in url:
                return _mock_response(_make_ecb_csv("IT", [("2026-02", 3.388)]))
            if "treasury.gov" in url:
                return _mock_response(_make_us_xml())
            return _mock_response("", status_code=404, is_error=True)

        mock_get.side_effect = side_effect
        r = SovereignDebtTool().execute(mode="spreads", countries=["IT"], month="2026-03")
        self.assertIsNotNone(r.data.get("us_curve"))
        self.assertIn("curve_2s10s", r.data["us_curve"])


# ──────────────────────────────────────────────────────────────────
# 13. Spreads — edge cases
# ──────────────────────────────────────────────────────────────────


class TestSpreadsEdge(unittest.TestCase):
    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_de_unavailable(self, mock_get):
        """Spreads should fail if DE baseline not available."""
        from agent.tools.sovereign_debt import SovereignDebtTool

        mock_get.side_effect = httpx.ConnectTimeout("timeout")
        r = SovereignDebtTool().execute(mode="spreads", countries=["IT", "GR"], month="2026-03")
        self.assertFalse(r.success)

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_us_curve_failure_doesnt_break(self, mock_get):
        """If US Treasury fetch fails, spreads still work — us_curve is None."""
        from agent.tools.sovereign_debt import SovereignDebtTool

        call_count = [0]

        def side_effect(url, **kwargs):
            if "M.DE." in url:
                return _mock_response(_make_ecb_csv("DE", [("2026-02", 2.744)]))
            elif "M.IT." in url:
                return _mock_response(_make_ecb_csv("IT", [("2026-02", 3.388)]))
            if "treasury.gov" in url:
                # US Treasury fails
                return _mock_response(status_code=500, is_error=True)
            return _mock_response("", status_code=404, is_error=True)

        mock_get.side_effect = side_effect
        r = SovereignDebtTool().execute(mode="spreads", countries=["IT"], month="2026-03")
        self.assertTrue(r.success)  # EU spreads still work
        self.assertIsNone(r.data.get("us_curve"))

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_spreads_sorted_by_level(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        def side_effect(url, **kwargs):
            if "M.DE." in url:
                return _mock_response(_make_ecb_csv("DE", [("2026-02", 2.0)]))
            elif "M.IT." in url:
                return _mock_response(_make_ecb_csv("IT", [("2026-02", 4.0)]))
            elif "M.GR." in url:
                return _mock_response(_make_ecb_csv("GR", [("2026-02", 5.0)]))
            elif "M.FR." in url:
                return _mock_response(_make_ecb_csv("FR", [("2026-02", 2.5)]))
            if "treasury.gov" in url:
                return _mock_response(_make_us_xml())
            return _mock_response("", status_code=404, is_error=True)

        mock_get.side_effect = side_effect
        r = SovereignDebtTool().execute(mode="spreads", countries=["IT", "GR", "FR"], month="2026-03")
        self.assertTrue(r.success)
        spreads = r.data["spreads"]
        # GR (3.0) > IT (2.0) > FR (0.5), sorted desc
        self.assertEqual(spreads[0]["country"], "GR")
        self.assertEqual(spreads[1]["country"], "IT")
        self.assertEqual(spreads[2]["country"], "FR")


# ──────────────────────────────────────────────────────────────────
# 14. Parsing helpers
# ──────────────────────────────────────────────────────────────────


class TestSafeFloat(unittest.TestCase):
    def test_valid(self):
        from agent.tools.sovereign_debt import _safe_float

        self.assertAlmostEqual(_safe_float("3.14"), 3.14)

    def test_none(self):
        from agent.tools.sovereign_debt import _safe_float

        self.assertIsNone(_safe_float(None))

    def test_empty(self):
        from agent.tools.sovereign_debt import _safe_float

        self.assertIsNone(_safe_float(""))

    def test_dash(self):
        from agent.tools.sovereign_debt import _safe_float

        self.assertIsNone(_safe_float("-"))

    def test_whitespace(self):
        from agent.tools.sovereign_debt import _safe_float

        self.assertAlmostEqual(_safe_float("  3.14  "), 3.14)

    def test_non_numeric(self):
        from agent.tools.sovereign_debt import _safe_float

        self.assertIsNone(_safe_float("abc"))

    def test_negative(self):
        from agent.tools.sovereign_debt import _safe_float

        self.assertAlmostEqual(_safe_float("-0.5"), -0.5)

    def test_zero(self):
        from agent.tools.sovereign_debt import _safe_float

        self.assertAlmostEqual(_safe_float("0"), 0.0)


class TestParseUSXML(unittest.TestCase):
    def test_basic(self):
        from agent.tools.sovereign_debt import _parse_us_treasury_xml

        records = _parse_us_treasury_xml(_make_us_xml())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["date"], "2026-03-27")

    def test_malformed(self):
        from agent.tools.sovereign_debt import _parse_us_treasury_xml

        records = _parse_us_treasury_xml("<<<bad>>>")
        self.assertEqual(records, [])

    def test_empty_feed(self):
        from agent.tools.sovereign_debt import _parse_us_treasury_xml

        xml = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        records = _parse_us_treasury_xml(xml)
        self.assertEqual(records, [])


class TestParseECBCSV(unittest.TestCase):
    def test_basic(self):
        from agent.tools.sovereign_debt import _parse_ecb_csv

        records = _parse_ecb_csv(_make_ecb_csv("DE"))
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["period"], "2026-01")
        self.assertAlmostEqual(records[0]["yield_pct"], 2.5)

    def test_empty(self):
        from agent.tools.sovereign_debt import _parse_ecb_csv

        records = _parse_ecb_csv("")
        self.assertEqual(records, [])

    def test_header_only(self):
        from agent.tools.sovereign_debt import _parse_ecb_csv

        header = "KEY,FREQ,REF_AREA,INSTRUMENT,MATURITY,DATA_TYPE,COUNT_AREA,CURRENCY,CALC_METHOD,DECIMALS,TIME_PERIOD,OBS_VALUE\n"
        records = _parse_ecb_csv(header)
        self.assertEqual(records, [])

    def test_missing_obs_value(self):
        from agent.tools.sovereign_debt import _parse_ecb_csv

        csv = (
            "KEY,FREQ,REF_AREA,INSTRUMENT,MATURITY,DATA_TYPE,COUNT_AREA,CURRENCY,CALC_METHOD,DECIMALS,TIME_PERIOD,OBS_VALUE\n"
            "IRS.M.DE.L.L40.CI.0000.EUR.N.Z,M,DE,L,L40,CI,0000,EUR,N,Z,2026-01,\n"
        )
        records = _parse_ecb_csv(csv)
        self.assertEqual(records, [])  # empty OBS_VALUE → _safe_float returns None → skipped


class TestParseJPCSV(unittest.TestCase):
    def test_basic(self):
        from agent.tools.sovereign_debt import _parse_jp_mof_csv

        records = _parse_jp_mof_csv(_make_jp_csv())
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["date"], "2026-03-26")

    def test_short_csv(self):
        from agent.tools.sovereign_debt import _parse_jp_mof_csv

        records = _parse_jp_mof_csv("Title\n")
        self.assertEqual(records, [])

    def test_footer_lines_skipped(self):
        """Japanese footer text with non-date first column should be skipped."""
        from agent.tools.sovereign_debt import _parse_jp_mof_csv

        csv = _make_jp_csv() + "(注)日本国債\n"
        records = _parse_jp_mof_csv(csv)
        self.assertEqual(len(records), 2)  # footer line not counted


class TestParseUKXML(unittest.TestCase):
    def test_basic(self):
        from agent.tools.sovereign_debt import _parse_uk_dmo_xml

        records = _parse_uk_dmo_xml(_make_uk_xml())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["isin"], "GB0001234567")

    def test_malformed(self):
        from agent.tools.sovereign_debt import _parse_uk_dmo_xml

        records = _parse_uk_dmo_xml("<<<bad>>>")
        self.assertEqual(records, [])

    def test_empty(self):
        from agent.tools.sovereign_debt import _parse_uk_dmo_xml

        records = _parse_uk_dmo_xml('<?xml version="1.0"?><DataReport></DataReport>')
        self.assertEqual(records, [])


# ──────────────────────────────────────────────────────────────────
# 15. Cache integration
# ──────────────────────────────────────────────────────────────────


class TestCacheIntegration(unittest.TestCase):
    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_us_cache_hit(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        cache = MagicMock()
        cache.get.return_value = {
            "month": "2026-03",
            "entries": 1,
            "records": [
                {
                    "date": "2026-03-27",
                    "yields": {"10y": 4.44},
                    "curve_2s10s": 0.56,
                    "curve_3m10y": 0.14,
                }
            ],
        }
        t = SovereignDebtTool(cache=cache)
        r = t.execute(mode="us_yields", month="2026-03")
        self.assertTrue(r.success)
        self.assertIn("cached", r.output.lower())
        mock_get.assert_not_called()

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_us_cache_miss(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        cache = MagicMock()
        cache.get.return_value = None
        mock_get.return_value = _mock_response(_make_us_xml())
        t = SovereignDebtTool(cache=cache)
        r = t.execute(mode="us_yields", month="2026-03")
        self.assertTrue(r.success)
        cache.put.assert_called_once()

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_jp_cache_hit(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        cache = MagicMock()
        cache.get.return_value = {
            "entries": 1,
            "records": [{"date": "2026-03-27", "yields": {"10y": 2.3}}],
        }
        t = SovereignDebtTool(cache=cache)
        r = t.execute(mode="jp_yields")
        self.assertTrue(r.success)
        mock_get.assert_not_called()

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_uk_cache_hit_applies_limit(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        cache = MagicMock()
        cache.get.return_value = {
            "total_auctions": 10,
            "records": [{"date": f"2025-{i + 1:02d}-01"} for i in range(10)],
        }
        t = SovereignDebtTool(cache=cache)
        r = t.execute(mode="uk_gilts", limit=3)
        self.assertTrue(r.success)
        self.assertEqual(len(r.data["records"]), 3)


# ──────────────────────────────────────────────────────────────────
# 16. Limit edge cases
# ──────────────────────────────────────────────────────────────────


class TestLimitEdge(unittest.TestCase):
    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_limit_zero_clamped(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        mock_get.return_value = _mock_response(_make_uk_xml())
        r = SovereignDebtTool().execute(mode="uk_gilts", limit=0)
        self.assertTrue(r.success)
        # limit clamped to min 1
        self.assertLessEqual(len(r.data["records"]), 1)

    @patch("agent.tools.sovereign_debt.httpx.get")
    def test_limit_over_max(self, mock_get):
        from agent.tools.sovereign_debt import SovereignDebtTool

        auctions = [
            {
                "name": f"Gilt {i}",
                "isin": f"GB{i:010d}",
                "date": f"2025-01-{i + 1:02d}T00:00:00",
                "type": "Auction",
                "nominal": "3000",
                "price": "99",
                "yield": "2.0",
            }
            for i in range(5)
        ]
        mock_get.return_value = _mock_response(_make_uk_xml(auctions))
        r = SovereignDebtTool().execute(mode="uk_gilts", limit=500)
        self.assertTrue(r.success)
        # Clamped to 100
        self.assertLessEqual(len(r.data["records"]), 100)


# ──────────────────────────────────────────────────────────────────
# 17. Constants
# ──────────────────────────────────────────────────────────────────


class TestConstants(unittest.TestCase):
    def test_us_maturity_map(self):
        from agent.tools.sovereign_debt import _US_MATURITY_MAP

        self.assertIn("BC_10YEAR", _US_MATURITY_MAP)
        self.assertEqual(_US_MATURITY_MAP["BC_10YEAR"], "10y")

    def test_jp_maturity_map(self):
        from agent.tools.sovereign_debt import _JP_MATURITY_MAP

        self.assertIn("40Y", _JP_MATURITY_MAP)
        self.assertEqual(_JP_MATURITY_MAP["40Y"], "40y")

    def test_eu_countries_all(self):
        from agent.tools.sovereign_debt import _EU_COUNTRIES_ALL

        self.assertIn("DE", _EU_COUNTRIES_ALL)
        self.assertIn("IT", _EU_COUNTRIES_ALL)
        self.assertIn("GR", _EU_COUNTRIES_ALL)

    def test_eu_countries_default(self):
        from agent.tools.sovereign_debt import _EU_COUNTRIES_DEFAULT

        self.assertIn("DE", _EU_COUNTRIES_DEFAULT)
        self.assertIn("FR", _EU_COUNTRIES_DEFAULT)


# ──────────────────────────────────────────────────────────────────
# 18. Registry integration
# ──────────────────────────────────────────────────────────────────


class TestRegistry(unittest.TestCase):
    def _build_registry(self):
        from agent.cli import build_tool_registry

        config = MagicMock()
        config.tool_timeout = 5
        config.fred_api_key = ""
        config.pipeline.db_path = ":memory:"
        return build_tool_registry(config)

    def test_tool_count(self):
        try:
            registry = self._build_registry()
        except Exception:
            import pytest

            pytest.skip("optional dependency")
            return
        names = registry.list_names()
        self.assertEqual(len(names), 60, f"Expected 60 tools, got {len(names)}: {sorted(names)}")

    def test_sovereign_debt_in_registry(self):
        try:
            registry = self._build_registry()
        except Exception:
            import pytest

            pytest.skip("optional dependency")
            return
        self.assertIn("sovereign_debt", registry.list_names())


# ──────────────────────────────────────────────────────────────────
# 19. Bandit arm integration
# ──────────────────────────────────────────────────────────────────


class TestBanditArm(unittest.TestCase):
    def test_arm_count(self):
        from agent.learning.bandit import DEFAULT_ARMS

        self.assertEqual(
            len(DEFAULT_ARMS),
            48,
            f"Expected 48 arms, got {len(DEFAULT_ARMS)}: {[a.name for a in DEFAULT_ARMS]}",
        )

    def test_sovereign_stress_arm_exists(self):
        from agent.learning.bandit import DEFAULT_ARMS

        names = [a.name for a in DEFAULT_ARMS]
        self.assertIn("sovereign_stress", names)

    def test_sovereign_stress_arm_tools(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm = next(a for a in DEFAULT_ARMS if a.name == "sovereign_stress")
        self.assertIn("sovereign_debt", arm.tools)

    def test_sovereign_stress_arm_examples(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm = next(a for a in DEFAULT_ARMS if a.name == "sovereign_stress")
        self.assertTrue(len(arm.examples) > 0)


# ──────────────────────────────────────────────────────────────────
# Phase 28: L2 sovereign-yield entity persistence
# ──────────────────────────────────────────────────────────────────


def _make_store_mock():
    """Build a mock PipelineStore for L2 persistence testing."""
    store = MagicMock()
    store.register_entity = MagicMock(side_effect=lambda **kw: kw["entity_id"])
    store.store_entity_observation = MagicMock(return_value=1)
    return store


class TestL2PersistenceNoStore(unittest.TestCase):
    """Persistence is a no-op when store is absent."""

    def test_no_store_returns_zeros(self):
        from agent.tools.sovereign_debt import SovereignDebtTool

        tool = SovereignDebtTool()
        tool._store = None
        counts = tool._persist_entities({"records": [{"date": "2026-03-27"}]}, "us_yields")
        self.assertEqual(counts, {"sovereign_yield_obs": 0})

    def test_no_entity_id_fn_returns_zeros(self):
        import agent.tools.sovereign_debt as sd_mod
        from agent.tools.sovereign_debt import SovereignDebtTool

        tool = SovereignDebtTool()
        tool._store = _make_store_mock()
        original = sd_mod._entity_id_from_key
        try:
            sd_mod._entity_id_from_key = None
            counts = tool._persist_entities({"records": [{"date": "2026-03-27"}]}, "us_yields")
            self.assertEqual(counts, {"sovereign_yield_obs": 0})
        finally:
            sd_mod._entity_id_from_key = original


class TestL2PersistenceSpreadsSkipped(unittest.TestCase):
    """Spreads mode should not persist anything (would double-count eu_yields)."""

    def test_spreads_mode_skipped(self):
        from agent.tools.sovereign_debt import SovereignDebtTool

        tool = SovereignDebtTool()
        tool._store = _make_store_mock()
        counts = tool._persist_entities({"countries": {"DE": []}}, "spreads")
        self.assertEqual(counts, {"sovereign_yield_obs": 0})
        tool._store.store_entity_observation.assert_not_called()


class TestL2PersistenceUSYields(unittest.TestCase):
    """us_yields mode persists a sovereign_yield obs on country=US."""

    def test_us_yields_persists_one_obs(self):
        from agent.tools.sovereign_debt import SovereignDebtTool

        tool = SovereignDebtTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "records": [
                {
                    "date": "2026-03-27",
                    "yields": {"2y": 3.88, "10y": 4.44},
                    "curve_2s10s": 0.56,
                },
            ],
            "entries": 1,
        }
        counts = tool._persist_entities(data, "us_yields")
        self.assertEqual(counts["sovereign_yield_obs"], 1)

    def test_us_yields_obs_type_and_depth(self):
        from agent.tools.sovereign_debt import SovereignDebtTool

        tool = SovereignDebtTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "records": [
                {
                    "date": "2026-03-27",
                    "yields": {"10y": 4.44},
                    "curve_2s10s": 0.56,
                },
            ],
        }
        tool._persist_entities(data, "us_yields")
        obs_call = store.store_entity_observation.call_args_list[0]
        self.assertEqual(obs_call.kwargs["observation_type"], "sovereign_yield")
        self.assertEqual(obs_call.kwargs["depth_level"], 2)
        self.assertEqual(obs_call.kwargs["source_tool"], "sovereign_debt")

    def test_us_yields_targets_US_country(self):
        from agent.pipeline.entity import entity_id_from_key
        from agent.tools.sovereign_debt import SovereignDebtTool

        tool = SovereignDebtTool()
        store = _make_store_mock()
        tool._store = store
        data = {"records": [{"date": "2026-03-27", "yields": {"10y": 4.44}, "curve_2s10s": 0.56}]}
        tool._persist_entities(data, "us_yields")

        us_eid = entity_id_from_key("country", "US")
        obs_call = store.store_entity_observation.call_args_list[0]
        self.assertEqual(obs_call.kwargs["entity_id"], us_eid)

    def test_us_yields_obs_value_fields(self):
        from agent.tools.sovereign_debt import SovereignDebtTool

        tool = SovereignDebtTool()
        store = _make_store_mock()
        tool._store = store
        data = {"records": [{"date": "2026-03-27", "yields": {"10y": 4.44}, "curve_2s10s": 0.56}]}
        tool._persist_entities(data, "us_yields")

        val = store.store_entity_observation.call_args_list[0].kwargs["value"]
        self.assertEqual(val["source"], "us_treasury")
        self.assertEqual(val["maturity"], "10y")
        self.assertAlmostEqual(val["yield_pct"], 4.44)
        self.assertAlmostEqual(val["curve_2s10s"], 0.56)

    def test_us_yields_empty_records(self):
        from agent.tools.sovereign_debt import SovereignDebtTool

        tool = SovereignDebtTool()
        tool._store = _make_store_mock()
        counts = tool._persist_entities({"records": []}, "us_yields")
        self.assertEqual(counts["sovereign_yield_obs"], 0)


class TestL2PersistenceEUYields(unittest.TestCase):
    """eu_yields mode persists per-country sovereign_yield obs."""

    def test_eu_yields_multiple_countries(self):
        from agent.tools.sovereign_debt import SovereignDebtTool

        tool = SovereignDebtTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "countries": {
                "DE": [{"period": "2026-02", "yield_pct": 2.744}],
                "IT": [{"period": "2026-02", "yield_pct": 3.388}],
                "FR": [{"period": "2026-02", "yield_pct": 2.900}],
            },
        }
        counts = tool._persist_entities(data, "eu_yields")
        self.assertEqual(counts["sovereign_yield_obs"], 3)

    def test_eu_yields_targets_correct_country_entities(self):
        from agent.pipeline.entity import entity_id_from_key
        from agent.tools.sovereign_debt import SovereignDebtTool

        tool = SovereignDebtTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "countries": {
                "DE": [{"period": "2026-02", "yield_pct": 2.744}],
            },
        }
        tool._persist_entities(data, "eu_yields")
        de_eid = entity_id_from_key("country", "DE")
        obs_call = store.store_entity_observation.call_args_list[0]
        self.assertEqual(obs_call.kwargs["entity_id"], de_eid)

    def test_eu_yields_source_is_ecb(self):
        from agent.tools.sovereign_debt import SovereignDebtTool

        tool = SovereignDebtTool()
        store = _make_store_mock()
        tool._store = store
        data = {"countries": {"IT": [{"period": "2026-02", "yield_pct": 3.388}]}}
        tool._persist_entities(data, "eu_yields")
        val = store.store_entity_observation.call_args_list[0].kwargs["value"]
        self.assertEqual(val["source"], "ecb")

    def test_eu_yields_skips_empty_records(self):
        from agent.tools.sovereign_debt import SovereignDebtTool

        tool = SovereignDebtTool()
        store = _make_store_mock()
        tool._store = store
        data = {"countries": {"DE": [], "FR": [{"period": "2026-02", "yield_pct": 2.9}]}}
        counts = tool._persist_entities(data, "eu_yields")
        self.assertEqual(counts["sovereign_yield_obs"], 1)


class TestL2PersistenceJPYields(unittest.TestCase):
    """jp_yields mode persists sovereign_yield on country=JP."""

    def test_jp_yields_persists_one_obs(self):
        from agent.tools.sovereign_debt import SovereignDebtTool

        tool = SovereignDebtTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "records": [{"date": "2026-03-27", "yields": {"10y": 2.30}}],
        }
        counts = tool._persist_entities(data, "jp_yields")
        self.assertEqual(counts["sovereign_yield_obs"], 1)

    def test_jp_yields_targets_JP_country(self):
        from agent.pipeline.entity import entity_id_from_key
        from agent.tools.sovereign_debt import SovereignDebtTool

        tool = SovereignDebtTool()
        store = _make_store_mock()
        tool._store = store
        data = {"records": [{"date": "2026-03-27", "yields": {"10y": 2.30}}]}
        tool._persist_entities(data, "jp_yields")
        jp_eid = entity_id_from_key("country", "JP")
        obs_call = store.store_entity_observation.call_args_list[0]
        self.assertEqual(obs_call.kwargs["entity_id"], jp_eid)

    def test_jp_yields_source_is_mof(self):
        from agent.tools.sovereign_debt import SovereignDebtTool

        tool = SovereignDebtTool()
        store = _make_store_mock()
        tool._store = store
        data = {"records": [{"date": "2026-03-27", "yields": {"10y": 2.30}}]}
        tool._persist_entities(data, "jp_yields")
        val = store.store_entity_observation.call_args_list[0].kwargs["value"]
        self.assertEqual(val["source"], "mof")


class TestL2PersistenceUKGilts(unittest.TestCase):
    """uk_gilts mode persists sovereign_yield on country=GB."""

    def test_uk_gilts_persists_one_obs(self):
        from agent.tools.sovereign_debt import SovereignDebtTool

        tool = SovereignDebtTool()
        store = _make_store_mock()
        tool._store = store
        data = {
            "records": [
                {
                    "name": "2% Treasury Gilt 2025",
                    "date": "2025-11-20",
                    "yield_pct": 2.05,
                },
            ],
        }
        counts = tool._persist_entities(data, "uk_gilts")
        self.assertEqual(counts["sovereign_yield_obs"], 1)

    def test_uk_gilts_targets_GB_country(self):
        from agent.pipeline.entity import entity_id_from_key
        from agent.tools.sovereign_debt import SovereignDebtTool

        tool = SovereignDebtTool()
        store = _make_store_mock()
        tool._store = store
        data = {"records": [{"name": "T 2%", "date": "2025-11-20", "yield_pct": 2.05}]}
        tool._persist_entities(data, "uk_gilts")
        gb_eid = entity_id_from_key("country", "GB")
        obs_call = store.store_entity_observation.call_args_list[0]
        self.assertEqual(obs_call.kwargs["entity_id"], gb_eid)


class TestL2PersistenceExceptionHandler(unittest.TestCase):
    """Inner exception caught, returns zeros."""

    def test_inner_exception_returns_zeros(self):
        from agent.tools.sovereign_debt import SovereignDebtTool

        tool = SovereignDebtTool()
        store = _make_store_mock()
        store.register_entity.side_effect = RuntimeError("DB down")
        tool._store = store
        data = {"records": [{"date": "2026-03-27", "yields": {"10y": 4.44}, "curve_2s10s": 0.56}]}
        counts = tool._persist_entities(data, "us_yields")
        self.assertEqual(counts, {"sovereign_yield_obs": 0})


if __name__ == "__main__":
    unittest.main()
