"""
Edge case tests for Form144Tool (SEC Form 144 — Insider Sell Intent Detection).

Covers: input validation, EFTS fetch (normal/empty/pagination/HTTP errors/429/500),
XML parser (normal/missing elements/namespace variations/empty/parse error),
acquisition classifier, date parsing (MM/DD/YYYY + YYYY-MM-DD + invalid),
cluster detection (normal/no clusters/single/entity filers/dedup),
gift filtering, urgency classification, cache integration, bandit arm check,
live network tests.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from agent.tools.form144 import (
    Form144Tool,
    _classify_acquisition,
    _classify_urgency,
    _extract_company_name,
    _extract_ticker,
    _normalize_name,
    _parse_date_iso,
    _parse_form144_xml,
    _parse_mmddyyyy,
    _safe_float,
    _safe_int,
)

# ─── Fixtures ────────────────────────────────────────────────────────

SAMPLE_XML_ROSS = """<?xml version="1.0" encoding="UTF-8"?>
<edgarSubmission xmlns="http://www.sec.gov/edgar/ownership" xmlns:ns2="http://www.sec.gov/edgar/common">
    <headerData><submissionType>144</submissionType></headerData>
    <formData>
        <issuerInfo>
            <issuerCik>0000745732</issuerCik>
            <issuerName>ROSS STORES, INC.</issuerName>
            <nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold>STEPHEN BRINKLEY</nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold>
            <relationshipsToIssuer>
                <relationshipToIssuer>Officer</relationshipToIssuer>
            </relationshipsToIssuer>
        </issuerInfo>
        <securitiesInformation>
            <securitiesClassTitle>Common</securitiesClassTitle>
            <brokerOrMarketmakerDetails>
                <name>Morgan Stanley Smith Barney LLC</name>
            </brokerOrMarketmakerDetails>
            <noOfUnitsSold>4154</noOfUnitsSold>
            <aggregateMarketValue>884428.56</aggregateMarketValue>
            <noOfUnitsOutstanding>323444928</noOfUnitsOutstanding>
            <approxSaleDate>03/24/2026</approxSaleDate>
            <securitiesExchangeName>NASDAQ</securitiesExchangeName>
        </securitiesInformation>
        <securitiesToBeSold>
            <securitiesClassTitle>Common</securitiesClassTitle>
            <acquiredDate>03/20/2026</acquiredDate>
            <natureOfAcquisitionTransaction>Performance Stock Units</natureOfAcquisitionTransaction>
            <nameOfPersonfromWhomAcquired>Issuer</nameOfPersonfromWhomAcquired>
            <isGiftTransaction>N</isGiftTransaction>
            <amountOfSecuritiesAcquired>1265</amountOfSecuritiesAcquired>
        </securitiesToBeSold>
        <nothingToReportFlagOnSecuritiesSoldInPast3Months>Y</nothingToReportFlagOnSecuritiesSoldInPast3Months>
        <noticeSignature>
            <noticeDate>03/24/2026</noticeDate>
            <signature>/s/ Stephen C Brinkley</signature>
        </noticeSignature>
    </formData>
</edgarSubmission>"""

SAMPLE_XML_NANO = """<?xml version="1.0" encoding="UTF-8"?>
<edgarSubmission xmlns="http://www.sec.gov/edgar/ownership" xmlns:com="http://www.sec.gov/edgar/common">
  <headerData><submissionType>144</submissionType></headerData>
  <formData>
    <issuerInfo>
      <issuerCik>0001643303</issuerCik>
      <issuerName>Nano Dimension Ltd.</issuerName>
      <nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold>Stehlin David</nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold>
      <relationshipsToIssuer>
        <relationshipToIssuer>Officer</relationshipToIssuer>
      </relationshipsToIssuer>
    </issuerInfo>
    <securitiesInformation>
      <securitiesClassTitle>Ordinary Shares</securitiesClassTitle>
      <brokerOrMarketmakerDetails><name>Oppenheimer &amp; Co. Inc.</name></brokerOrMarketmakerDetails>
      <noOfUnitsSold>22699</noOfUnitsSold>
      <aggregateMarketValue>37907.33</aggregateMarketValue>
      <noOfUnitsOutstanding>216933812</noOfUnitsOutstanding>
      <approxSaleDate>03/23/2026</approxSaleDate>
      <securitiesExchangeName>NASDAQ</securitiesExchangeName>
    </securitiesInformation>
    <securitiesToBeSold>
      <securitiesClassTitle>Ordinary</securitiesClassTitle>
      <acquiredDate>09/08/2025</acquiredDate>
      <natureOfAcquisitionTransaction>Restricted Stock Units</natureOfAcquisitionTransaction>
      <nameOfPersonfromWhomAcquired>Issuer</nameOfPersonfromWhomAcquired>
      <isGiftTransaction>N</isGiftTransaction>
      <amountOfSecuritiesAcquired>347221</amountOfSecuritiesAcquired>
    </securitiesToBeSold>
    <noticeSignature>
      <noticeDate>03/23/2026</noticeDate>
      <signature>David Stehlin</signature>
    </noticeSignature>
  </formData>
</edgarSubmission>"""

SAMPLE_XML_GIFT = """<?xml version="1.0" encoding="UTF-8"?>
<edgarSubmission xmlns="http://www.sec.gov/edgar/ownership">
  <formData>
    <issuerInfo>
      <issuerName>Test Corp</issuerName>
      <nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold>John Doe</nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold>
      <relationshipsToIssuer><relationshipToIssuer>Director</relationshipToIssuer></relationshipsToIssuer>
    </issuerInfo>
    <securitiesInformation>
      <noOfUnitsSold>1000</noOfUnitsSold>
      <aggregateMarketValue>50000</aggregateMarketValue>
      <approxSaleDate>04/01/2026</approxSaleDate>
    </securitiesInformation>
    <securitiesToBeSold>
      <natureOfAcquisitionTransaction>Gift</natureOfAcquisitionTransaction>
      <isGiftTransaction>Y</isGiftTransaction>
      <amountOfSecuritiesAcquired>1000</amountOfSecuritiesAcquired>
    </securitiesToBeSold>
    <noticeSignature><noticeDate>03/20/2026</noticeDate></noticeSignature>
  </formData>
</edgarSubmission>"""

SAMPLE_XML_OPEN_MARKET = """<?xml version="1.0" encoding="UTF-8"?>
<edgarSubmission xmlns="http://www.sec.gov/edgar/ownership">
  <formData>
    <issuerInfo>
      <issuerName>Alpha Inc.</issuerName>
      <nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold>Jane Smith</nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold>
      <relationshipsToIssuer><relationshipToIssuer>CEO</relationshipToIssuer></relationshipsToIssuer>
    </issuerInfo>
    <securitiesInformation>
      <noOfUnitsSold>10000</noOfUnitsSold>
      <aggregateMarketValue>500000</aggregateMarketValue>
      <noOfUnitsOutstanding>100000000</noOfUnitsOutstanding>
      <approxSaleDate>03/25/2026</approxSaleDate>
    </securitiesInformation>
    <securitiesToBeSold>
      <natureOfAcquisitionTransaction>Open Market Purchase</natureOfAcquisitionTransaction>
      <isGiftTransaction>N</isGiftTransaction>
      <amountOfSecuritiesAcquired>10000</amountOfSecuritiesAcquired>
    </securitiesToBeSold>
    <noticeSignature><noticeDate>03/25/2026</noticeDate></noticeSignature>
  </formData>
</edgarSubmission>"""

SAMPLE_XML_10B5_1 = """<?xml version="1.0" encoding="UTF-8"?>
<edgarSubmission xmlns="http://www.sec.gov/edgar/ownership">
  <formData>
    <issuerInfo>
      <issuerName>Beta Corp</issuerName>
      <nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold>Bob Plan</nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold>
      <relationshipsToIssuer><relationshipToIssuer>Officer</relationshipToIssuer></relationshipsToIssuer>
    </issuerInfo>
    <securitiesInformation>
      <noOfUnitsSold>5000</noOfUnitsSold>
      <aggregateMarketValue>100000</aggregateMarketValue>
      <approxSaleDate>04/15/2026</approxSaleDate>
    </securitiesInformation>
    <securitiesToBeSold>
      <natureOfAcquisitionTransaction>Restricted Stock Units</natureOfAcquisitionTransaction>
      <isGiftTransaction>N</isGiftTransaction>
      <amountOfSecuritiesAcquired>5000</amountOfSecuritiesAcquired>
    </securitiesToBeSold>
    <noticeSignature>
      <noticeDate>03/20/2026</noticeDate>
      <planAdoptionDates>01/15/2026</planAdoptionDates>
    </noticeSignature>
  </formData>
</edgarSubmission>"""

SAMPLE_XML_EMPTY_SHARES = """<?xml version="1.0" encoding="UTF-8"?>
<edgarSubmission xmlns="http://www.sec.gov/edgar/ownership">
  <formData>
    <issuerInfo>
      <issuerName>Empty Corp</issuerName>
      <nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold>Empty Person</nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold>
    </issuerInfo>
    <securitiesInformation>
      <noOfUnitsSold>0</noOfUnitsSold>
      <aggregateMarketValue>0</aggregateMarketValue>
    </securitiesInformation>
    <noticeSignature><noticeDate>03/20/2026</noticeDate></noticeSignature>
  </formData>
</edgarSubmission>"""


def _make_efts_hit(
    ticker: str,
    filer_name: str,
    issuer_cik: str = "0000111111",
    filer_cik: str = "0000222222",
    file_date: str = "2026-03-24",
    accession: str = "0001234567-26-000001",
    company: str = "Test Company",
) -> dict:
    return {
        "_source": {
            "ciks": [issuer_cik, filer_cik],
            "display_names": [
                f"{company}  ({ticker})  (CIK {issuer_cik})",
                f"{filer_name}  (CIK {filer_cik})",
            ],
            "file_date": file_date,
            "adsh": accession,
        },
        "_id": f"{accession}:primary_doc.xml",
    }


# ─── Input Validation ────────────────────────────────────────────────


class TestInputValidation:
    def test_days_back_clamped_low(self):
        tool = Form144Tool()
        with patch.object(tool, "_fetch_recent_144s", return_value=[]):
            result = tool.execute(days_back=-5)
        assert result.success

    def test_days_back_clamped_high(self):
        tool = Form144Tool()
        with patch.object(tool, "_fetch_recent_144s", return_value=[]):
            result = tool.execute(days_back=999)
        assert result.success

    def test_min_cluster_clamped(self):
        tool = Form144Tool()
        with patch.object(tool, "_fetch_recent_144s", return_value=[]):
            result = tool.execute(min_cluster_size=0)
        assert result.success

    def test_ticker_normalized(self):
        tool = Form144Tool()
        with patch.object(tool, "_fetch_recent_144s", return_value=[]):
            result = tool.execute(ticker="  aapl  ")
        assert result.success


# ─── EFTS Fetch ───────────────────────────────────────────────────────


class TestEFTSFetch:
    def test_empty_response(self):
        tool = Form144Tool()
        with patch.object(tool, "_fetch_recent_144s", return_value=[]):
            result = tool.execute(days_back=5)
        assert result.success
        assert "No Form 144" in result.output

    def test_http_error_propagates(self):
        tool = Form144Tool()
        with patch.object(
            tool, "_fetch_recent_144s", side_effect=Exception("network down")
        ):
            result = tool.execute(days_back=5)
        assert not result.success
        assert "SEC EDGAR error" in result.output

    def test_cache_hit(self):
        mock_cache = MagicMock()
        mock_cache.get.return_value = [_make_efts_hit("TEST", "John Doe")]
        tool = Form144Tool(cache=mock_cache)
        # Will use cached EFTS data, then fail on XML fetch
        with patch.object(tool, "_fetch_filing_xml", return_value=None):
            result = tool.execute(days_back=5)
        mock_cache.get.assert_called()

    def test_cache_stores_results(self):
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        tool = Form144Tool(cache=mock_cache)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"hits": {"hits": [], "total": {"value": 0}}}

        with patch("agent.tools.form144.httpx.Client") as mock_client:
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=ctx)
            ctx.__exit__ = MagicMock(return_value=False)
            ctx.get.return_value = mock_resp
            mock_client.return_value = ctx
            with patch("agent.tools.form144.time.sleep"):
                tool._fetch_recent_144s(date(2026, 3, 1), date(2026, 3, 25))
        # No hits → cache not called for put (only stores when all_hits non-empty)

    def test_429_retry(self):
        tool = Form144Tool()
        import httpx as httpx_mod

        mock_resp_429 = MagicMock()
        mock_resp_429.status_code = 429

        mock_resp_ok = MagicMock()
        mock_resp_ok.status_code = 200
        mock_resp_ok.raise_for_status = MagicMock()
        mock_resp_ok.json.return_value = {"hits": {"hits": [], "total": {"value": 0}}}

        exc_429 = httpx_mod.HTTPStatusError(
            "429", request=MagicMock(), response=mock_resp_429
        )

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise exc_429
            return mock_resp_ok

        with patch("agent.tools.form144.httpx.Client") as mock_client:
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=ctx)
            ctx.__exit__ = MagicMock(return_value=False)
            ctx.get.side_effect = side_effect
            mock_client.return_value = ctx
            with patch("agent.tools.form144.time.sleep"):
                result = tool._fetch_recent_144s(date(2026, 3, 1), date(2026, 3, 5))
        assert call_count[0] >= 2

    def test_500_returns_partial_results(self):
        """EFTS 500 error mid-pagination returns what we have so far."""
        tool = Form144Tool()
        import httpx as httpx_mod

        hit1 = _make_efts_hit("AAPL", "Tim Cook")
        mock_resp_ok = MagicMock()
        mock_resp_ok.status_code = 200
        mock_resp_ok.raise_for_status = MagicMock()
        mock_resp_ok.json.return_value = {
            "hits": {"hits": [hit1], "total": {"value": 200}}
        }

        mock_resp_500 = MagicMock()
        mock_resp_500.status_code = 500
        exc_500 = httpx_mod.HTTPStatusError(
            "500", request=MagicMock(), response=mock_resp_500
        )

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_resp_ok
            raise exc_500

        with patch("agent.tools.form144.httpx.Client") as mock_client:
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=ctx)
            ctx.__exit__ = MagicMock(return_value=False)
            ctx.get.side_effect = side_effect
            mock_client.return_value = ctx
            with patch("agent.tools.form144.time.sleep"):
                result = tool._fetch_recent_144s(date(2026, 3, 1), date(2026, 3, 5))
        assert len(result) == 1  # Got partial results


# ─── XML Parser ──────────────────────────────────────────────────────


class TestXMLParser:
    def test_ross_stores_xml(self):
        result = _parse_form144_xml(
            SAMPLE_XML_ROSS, "ROST", "ROSS STORES", "2026-03-24"
        )
        assert result is not None
        assert result["insider_name"] == "STEPHEN BRINKLEY"
        assert result["relationship"] == "Officer"
        assert result["shares_to_sell"] == 4454
        assert abs(result["dollar_value"] - 884428.56) < 0.01
        assert result["shares_outstanding"] == 323444928
        assert result["approx_sale_date"] == "2026-03-24"
        assert result["exchange"] == "NASDAQ"
        assert result["broker"] == "Morgan Stanley Smith Barney LLC"
        assert result["acquisition_type"] == "vesting"
        assert result["urgency"] == "immediate"

    def test_nano_dimension_xml(self):
        """Different namespace prefix (com: instead of ns2:)."""
        result = _parse_form144_xml(
            SAMPLE_XML_NANO, "NNDM", "Nano Dimension", "2026-03-23"
        )
        assert result is not None
        assert result["insider_name"] == "Stehlin David"
        assert result["shares_to_sell"] == 22699
        assert result["acquisition_type"] == "vesting"

    def test_gift_transaction_returns_none(self):
        result = _parse_form144_xml(SAMPLE_XML_GIFT, "TEST", "Test Corp", "2026-03-20")
        assert result is None

    def test_open_market_acquisition(self):
        result = _parse_form144_xml(
            SAMPLE_XML_OPEN_MARKET, "ALPH", "Alpha Inc.", "2026-03-25"
        )
        assert result is not None
        assert result["acquisition_type"] == "open_market"
        assert result["relationship"] == "CEO"

    def test_10b5_1_plan_detected(self):
        result = _parse_form144_xml(
            SAMPLE_XML_10B5_1, "BETA", "Beta Corp", "2026-03-20"
        )
        assert result is not None
        assert result["has_10b5_1_plan"] is True
        assert result["urgency"] == "planned"  # approxSaleDate is 04/15/2026

    def test_empty_shares_returns_none(self):
        result = _parse_form144_xml(
            SAMPLE_XML_EMPTY_SHARES, "EMPTY", "Empty Corp", "2026-03-20"
        )
        assert result is None

    def test_malformed_xml(self):
        result = _parse_form144_xml("not xml at all <garbage", "X", "X", "2026-03-20")
        assert result is None

    def test_empty_xml(self):
        result = _parse_form144_xml("", "X", "X", "2026-03-20")
        assert result is None

    def test_xml_no_form_data(self):
        xml = '<?xml version="1.0"?><edgarSubmission xmlns="http://www.sec.gov/edgar/ownership"><headerData/></edgarSubmission>'
        result = _parse_form144_xml(xml, "X", "X", "2026-03-20")
        assert result is None

    def test_xml_no_insider_name_returns_none(self):
        xml = """<?xml version="1.0"?>
        <edgarSubmission xmlns="http://www.sec.gov/edgar/ownership">
          <formData>
            <issuerInfo><issuerName>Test</issuerName></issuerInfo>
            <securitiesInformation><noOfUnitsSold>100</noOfUnitsSold><aggregateMarketValue>5000</aggregateMarketValue></securitiesInformation>
            <noticeSignature><noticeDate>03/20/2026</noticeDate></noticeSignature>
          </formData>
        </edgarSubmission>"""
        result = _parse_form144_xml(xml, "X", "X", "2026-03-20")
        assert result is None

    def test_multiple_securities_to_be_sold(self):
        """XML with multiple securitiesToBeSold entries — picks highest signal type."""
        xml = """<?xml version="1.0"?>
        <edgarSubmission xmlns="http://www.sec.gov/edgar/ownership">
          <formData>
            <issuerInfo>
              <issuerName>Multi Corp</issuerName>
              <nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold>Multi Person</nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold>
            </issuerInfo>
            <securitiesInformation>
              <noOfUnitsSold>5000</noOfUnitsSold>
              <aggregateMarketValue>250000</aggregateMarketValue>
            </securitiesInformation>
            <securitiesToBeSold>
              <natureOfAcquisitionTransaction>Restricted Stock Units</natureOfAcquisitionTransaction>
              <isGiftTransaction>N</isGiftTransaction>
              <amountOfSecuritiesAcquired>3000</amountOfSecuritiesAcquired>
            </securitiesToBeSold>
            <securitiesToBeSold>
              <natureOfAcquisitionTransaction>Open Market Purchase</natureOfAcquisitionTransaction>
              <isGiftTransaction>N</isGiftTransaction>
              <amountOfSecuritiesAcquired>2000</amountOfSecuritiesAcquired>
            </securitiesToBeSold>
            <noticeSignature><noticeDate>03/20/2026</noticeDate></noticeSignature>
          </formData>
        </edgarSubmission>"""
        result = _parse_form144_xml(xml, "MULTI", "Multi Corp", "2026-03-20")
        assert result is not None
        # Highest signal type should be open_market (weight 3.0 > vesting 0.5)
        assert result["acquisition_type"] == "open_market"
        assert len(result["acquisition_details"]) == 2


# ─── Acquisition Classifier ─────────────────────────────────────────


class TestAcquisitionClassifier:
    def test_open_market(self):
        assert _classify_acquisition("Open Market Purchase", False) == "open_market"
        assert _classify_acquisition("Market Purchase", False) == "open_market"

    def test_private_placement(self):
        assert _classify_acquisition("Private Placement", False) == "private_placement"
        assert _classify_acquisition("Private sale", False) == "private_placement"

    def test_vesting_types(self):
        assert _classify_acquisition("Performance Stock Units", False) == "vesting"
        assert _classify_acquisition("Restricted Stock Units", False) == "vesting"
        assert _classify_acquisition("Stock Option Exercise", False) == "vesting"
        assert _classify_acquisition("Incentive Plan Award", False) == "vesting"
        assert _classify_acquisition("Compensation", False) == "vesting"

    def test_gift(self):
        assert _classify_acquisition("Gift", True) == "gift"
        assert _classify_acquisition("Whatever", True) == "gift"

    def test_other(self):
        assert _classify_acquisition("Something Unknown", False) == "other"
        assert _classify_acquisition("", False) == "other"


# ─── Date Parsing ────────────────────────────────────────────────────


class TestDateParsing:
    def test_mmddyyyy(self):
        assert _parse_mmddyyyy("03/24/2026") == "2026-03-24"
        assert _parse_mmddyyyy("12/31/2025") == "2025-12-31"
        assert _parse_mmddyyyy("01/01/2020") == "2020-01-01"

    def test_already_iso(self):
        assert _parse_mmddyyyy("2026-03-24") == "2026-03-24"

    def test_invalid_date(self):
        assert _parse_mmddyyyy("not-a-date") == ""
        assert _parse_mmddyyyy("") == ""

    def test_iso_parse(self):
        d = _parse_date_iso("2026-03-24")
        assert d == date(2026, 3, 24)

    def test_iso_parse_none(self):
        assert _parse_date_iso("") is None
        assert _parse_date_iso("garbage") is None


# ─── Urgency Classification ─────────────────────────────────────────


class TestUrgency:
    def test_immediate(self):
        assert _classify_urgency("2026-03-24", "2026-03-24") == "immediate"
        assert _classify_urgency("2026-03-24", "2026-03-25") == "immediate"

    def test_near_term(self):
        assert _classify_urgency("2026-03-20", "2026-03-25") == "near_term"

    def test_planned(self):
        assert _classify_urgency("2026-03-20", "2026-04-15") == "planned"

    def test_unknown(self):
        assert _classify_urgency("", "2026-03-24") == "unknown"
        assert _classify_urgency("2026-03-24", "") == "unknown"


# ─── Helper Functions ────────────────────────────────────────────────


class TestHelpers:
    def test_extract_ticker(self):
        assert _extract_ticker("ROSS STORES, INC.  (ROST)  (CIK 0000745732)") == "ROST"
        assert (
            _extract_ticker("Nano Dimension Ltd.  (NNDM)  (CIK 0001643303)") == "NNDM"
        )
        assert (
            _extract_ticker("Ramaco Resources, Inc.  (METC, METCB)  (CIK 000)")
            == "METC"
        )
        assert _extract_ticker("Some Person  (CIK 0001234567)") == ""

    def test_extract_company_name(self):
        assert (
            _extract_company_name("ROSS STORES, INC.  (ROST)  (CIK 000)")
            == "ROSS STORES, INC."
        )
        assert _extract_company_name("No Parens") == "No Parens"

    def test_normalize_name(self):
        assert (
            _normalize_name("Yorktown Energy Partners IX, L.P.")
            == "YORKTOWN ENERGY PARTNERS IX"
        )
        assert _normalize_name("Silver Lake, LLC") == "SILVER LAKE"
        assert _normalize_name("  John Doe  ") == "JOHN DOE"

    def test_safe_float(self):
        assert _safe_float("884428.56") == 884428.56
        assert _safe_float("1,234,567.89") == 1234567.89
        assert _safe_float(None) == 0.0
        assert _safe_float("not-a-num") == 0.0

    def test_safe_int(self):
        assert _safe_int("4154") == 4454
        assert _safe_int("323,444,928") == 323444928
        assert _safe_int(None) == 0
        assert _safe_int("not-a-num") == 0
        assert _safe_int("4154.0") == 4454


# ─── Cluster Detection ──────────────────────────────────────────────


class TestClusterDetection:
    def test_basic_cluster(self):
        tool = Form144Tool()
        filings = [
            {
                "ticker": "AAPL",
                "company": "Apple",
                "insider_name": "Alice",
                "filing_date": "2026-03-20",
                "dollar_value": 100000,
                "shares_to_sell": 1000,
                "shares_outstanding": 10000000,
                "acquisition_type": "vesting",
                "relationship": "Officer",
                "urgency": "immediate",
            },
            {
                "ticker": "AAPL",
                "company": "Apple",
                "insider_name": "Bob",
                "filing_date": "2026-03-22",
                "dollar_value": 200000,
                "shares_to_sell": 2000,
                "shares_outstanding": 10000000,
                "acquisition_type": "open_market",
                "relationship": "Director",
                "urgency": "immediate",
            },
        ]
        clusters = tool._detect_sell_clusters(filings, min_size=2)
        assert len(clusters) == 1
        assert clusters[0]["ticker"] == "AAPL"
        assert clusters[0]["insider_count"] == 2
        assert clusters[0]["total_value"] == 300000
        assert clusters[0]["has_voluntary_sells"] is True

    def test_no_cluster_single_insider(self):
        tool = Form144Tool()
        filings = [
            {
                "ticker": "AAPL",
                "company": "Apple",
                "insider_name": "Alice",
                "filing_date": "2026-03-20",
                "dollar_value": 100000,
                "shares_to_sell": 1000,
                "shares_outstanding": 10000000,
                "acquisition_type": "vesting",
                "relationship": "Officer",
                "urgency": "immediate",
            },
        ]
        clusters = tool._detect_sell_clusters(filings, min_size=2)
        assert len(clusters) == 0

    def test_cluster_dedup_same_insider(self):
        """Same insider filing twice shouldn't count as 2 distinct insiders."""
        tool = Form144Tool()
        filings = [
            {
                "ticker": "AAPL",
                "company": "Apple",
                "insider_name": "Alice",
                "filing_date": "2026-03-20",
                "dollar_value": 100000,
                "shares_to_sell": 1000,
                "shares_outstanding": 10000000,
                "acquisition_type": "vesting",
                "relationship": "Officer",
                "urgency": "immediate",
            },
            {
                "ticker": "AAPL",
                "company": "Apple",
                "insider_name": "ALICE",
                "filing_date": "2026-03-22",
                "dollar_value": 200000,
                "shares_to_sell": 2000,
                "shares_outstanding": 10000000,
                "acquisition_type": "vesting",
                "relationship": "Officer",
                "urgency": "immediate",
            },
        ]
        clusters = tool._detect_sell_clusters(filings, min_size=2)
        assert len(clusters) == 0  # Same person (case-insensitive)

    def test_cluster_window_boundary(self):
        """Filings 15 days apart should NOT cluster (14-day window)."""
        tool = Form144Tool()
        filings = [
            {
                "ticker": "AAPL",
                "company": "Apple",
                "insider_name": "Alice",
                "filing_date": "2026-03-01",
                "dollar_value": 100000,
                "shares_to_sell": 1000,
                "shares_outstanding": 10000000,
                "acquisition_type": "vesting",
                "relationship": "Officer",
                "urgency": "immediate",
            },
            {
                "ticker": "AAPL",
                "company": "Apple",
                "insider_name": "Bob",
                "filing_date": "2026-03-16",
                "dollar_value": 200000,
                "shares_to_sell": 2000,
                "shares_outstanding": 10000000,
                "acquisition_type": "vesting",
                "relationship": "Director",
                "urgency": "near_term",
            },
        ]
        clusters = tool._detect_sell_clusters(filings, min_size=2)
        assert len(clusters) == 0

    def test_multiple_tickers(self):
        tool = Form144Tool()
        filings = [
            {
                "ticker": "AAPL",
                "company": "Apple",
                "insider_name": "Alice",
                "filing_date": "2026-03-20",
                "dollar_value": 100000,
                "shares_to_sell": 1000,
                "shares_outstanding": 10000000,
                "acquisition_type": "vesting",
                "relationship": "Officer",
                "urgency": "immediate",
            },
            {
                "ticker": "AAPL",
                "company": "Apple",
                "insider_name": "Bob",
                "filing_date": "2026-03-22",
                "dollar_value": 200000,
                "shares_to_sell": 2000,
                "shares_outstanding": 10000000,
                "acquisition_type": "vesting",
                "relationship": "Director",
                "urgency": "immediate",
            },
            {
                "ticker": "MSFT",
                "company": "Microsoft",
                "insider_name": "Carol",
                "filing_date": "2026-03-20",
                "dollar_value": 500000,
                "shares_to_sell": 3000,
                "shares_outstanding": 50000000,
                "acquisition_type": "open_market",
                "relationship": "CEO",
                "urgency": "immediate",
            },
            {
                "ticker": "MSFT",
                "company": "Microsoft",
                "insider_name": "Dave",
                "filing_date": "2026-03-21",
                "dollar_value": 600000,
                "shares_to_sell": 4000,
                "shares_outstanding": 50000000,
                "acquisition_type": "private_placement",
                "relationship": "Officer",
                "urgency": "near_term",
            },
        ]
        clusters = tool._detect_sell_clusters(filings, min_size=2)
        assert len(clusters) == 2
        # MSFT should rank higher (more dollar value * open_market weight)
        assert clusters[0]["ticker"] == "MSFT"

    def test_entity_name_dedup(self):
        """Entity names with suffixes like L.P. should dedup correctly."""
        tool = Form144Tool()
        filings = [
            {
                "ticker": "DELL",
                "company": "Dell",
                "insider_name": "Silver Lake Partners, L.P.",
                "filing_date": "2026-03-20",
                "dollar_value": 1000000,
                "shares_to_sell": 5000,
                "shares_outstanding": 50000000,
                "acquisition_type": "private_placement",
                "relationship": "10% Stockholder",
                "urgency": "immediate",
            },
            {
                "ticker": "DELL",
                "company": "Dell",
                "insider_name": "Silver Lake Partners, LP",
                "filing_date": "2026-03-21",
                "dollar_value": 2000000,
                "shares_to_sell": 10000,
                "shares_outstanding": 50000000,
                "acquisition_type": "private_placement",
                "relationship": "10% Stockholder",
                "urgency": "immediate",
            },
        ]
        clusters = tool._detect_sell_clusters(filings, min_size=2)
        # "Silver Lake Partners, L.P." and "Silver Lake Partners, LP" → same after normalization
        assert len(clusters) == 0

    def test_conviction_high_voluntary_3plus(self):
        tool = Form144Tool()
        filings = [
            {
                "ticker": "X",
                "company": "X",
                "insider_name": f"Person{i}",
                "filing_date": "2026-03-20",
                "dollar_value": 100000,
                "shares_to_sell": 1000,
                "shares_outstanding": 1000000,
                "acquisition_type": "open_market",
                "relationship": "Officer",
                "urgency": "immediate",
            }
            for i in range(3)
        ]
        clusters = tool._detect_sell_clusters(filings, min_size=2)
        assert clusters[0]["conviction"] == "high"

    def test_pct_of_outstanding(self):
        tool = Form144Tool()
        filings = [
            {
                "ticker": "TINY",
                "company": "Tiny Corp",
                "insider_name": "Big Seller",
                "filing_date": "2026-03-20",
                "dollar_value": 50000,
                "shares_to_sell": 500000,
                "shares_outstanding": 10000000,
                "acquisition_type": "vesting",
                "relationship": "Officer",
                "urgency": "immediate",
            },
            {
                "ticker": "TINY",
                "company": "Tiny Corp",
                "insider_name": "Also Selling",
                "filing_date": "2026-03-21",
                "dollar_value": 30000,
                "shares_to_sell": 300000,
                "shares_outstanding": 10000000,
                "acquisition_type": "vesting",
                "relationship": "Director",
                "urgency": "immediate",
            },
        ]
        clusters = tool._detect_sell_clusters(filings, min_size=2)
        assert len(clusters) == 1
        # (500000 + 300000) / 10000000 * 100 = 8%
        assert clusters[0]["pct_of_outstanding"] == 8.0


# ─── Full Pipeline (Mocked) ─────────────────────────────────────────


class TestFullPipeline:
    def test_no_filings_found(self):
        tool = Form144Tool()
        with patch.object(tool, "_fetch_recent_144s", return_value=[]):
            result = tool.execute(days_back=5)
        assert result.success
        assert "No Form 144" in result.output

    def test_filings_but_no_clusters(self):
        tool = Form144Tool()
        hits = [_make_efts_hit("AAPL", "Alice", accession="0001-26-001")]
        with patch.object(tool, "_fetch_recent_144s", return_value=hits):
            with patch.object(tool, "_fetch_filing_xml", return_value=None):
                result = tool.execute(days_back=5, min_cluster_size=2)
        assert result.success

    def test_ticker_filter(self):
        tool = Form144Tool()
        hits = [
            _make_efts_hit("AAPL", "Alice", accession="0001-26-001"),
            _make_efts_hit("AAPL", "Bob", accession="0001-26-002"),
            _make_efts_hit("MSFT", "Carol", accession="0001-26-003"),
        ]
        with patch.object(tool, "_fetch_recent_144s", return_value=hits):
            with patch.object(
                tool, "_fetch_filing_xml", return_value=SAMPLE_XML_OPEN_MARKET
            ):
                result = tool.execute(days_back=5, ticker="msft", min_cluster_size=2)
        assert result.success


# ─── Bandit + Config Integration ─────────────────────────────────────


class TestIntegration:
    def test_bandit_arm_has_form144(self):
        from agent.learning.bandit import DEFAULT_ARMS

        insider_arm = next(a for a in DEFAULT_ARMS if a.name == "insider_flow")
        assert "form144" in insider_arm.tools

    def test_cli_registers_form144(self):
        try:
            from agent.cli import build_tool_registry
            from agent.config.settings import AgentConfig
        except (ImportError, ModuleNotFoundError):
            pytest.skip("hmmlearn or other dep not installed")
        config = AgentConfig()
        registry = build_tool_registry(config)
        names = registry.list_names()
        assert "form144" in names

    def test_tool_count(self):
        """Verify total registered tool count after adding form144."""
        try:
            from agent.cli import build_tool_registry
            from agent.config.settings import AgentConfig
        except (ImportError, ModuleNotFoundError):
            pytest.skip("hmmlearn or other dep not installed")
        config = AgentConfig()
        registry = build_tool_registry(config)
        names = registry.list_names()
        assert (
            len(names) == 47
        )  # Was 27, +3 for defi/gov_contracts/academic_preprints, +1 sanctions_monitor, +1 cert_transparency, +1 sovereign_debt, +1 central_bank_balance, +1 foia_requests

    def test_openai_schema(self):
        tool = Form144Tool()
        schema = tool.to_openai_tool()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "form144"
        props = schema["function"]["parameters"]["properties"]
        assert "days_back" in props
        assert "ticker" in props
        assert "min_cluster_size" in props


# ─── Live Network Tests ──────────────────────────────────────────────


class TestLiveNetwork:
    @pytest.mark.skipif(
        not __import__("os").environ.get("TIRRA_LIVE_TESTS", ""),
        reason="Live tests disabled (set TIRRA_LIVE_TESTS=1)",
    )
    def test_live_efts_search(self):
        """Live EFTS search returns Form 144 filings."""
        tool = Form144Tool()
        today = date.today()
        start = today - timedelta(days=5)
        hits = tool._fetch_recent_144s(start, today)
        assert len(hits) > 0
        # Check structure
        src = hits[0].get("_source", {})
        assert "ciks" in src
        assert "display_names" in src
        assert "file_date" in src

    @pytest.mark.skipif(
        not __import__("os").environ.get("TIRRA_LIVE_TESTS", ""),
        reason="Live tests disabled (set TIRRA_LIVE_TESTS=1)",
    )
    def test_live_full_scan(self):
        """Live full scan finds at least some sell-intent filings."""
        tool = Form144Tool()
        result = tool.execute(days_back=5, min_cluster_size=2)
        assert result.success
        assert result.data is not None
        assert result.data["total_filings"] > 0
