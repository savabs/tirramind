"""Tests for insider_filings L2 upgrade — Phase 10b.1.

Step 10b.1.1: reporter_cik and issuer_cik threading through the parser.
Step 10b.1.2: Optional PipelineStore in constructor.
Step 10b.1.3: Entity registration + observation storage (_persist_entities).
Step 10b.1.4: CIK-based dedup in _find_best_cluster.
Step 10b.1.5: entity_ids mapping in cluster data.
Step 10b.1.6: Edge case test suite.
"""

from __future__ import annotations

import textwrap
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.pipeline.entity import entity_id_from_key
from agent.pipeline.store import PipelineStore
from agent.tools.insider_filings import InsiderFilingsTool, _parse_form4_xml

# ── Fixtures ────────────────────────────────────────────────────────────


def _make_form4_xml(
    *,
    ticker: str = "AAPL",
    company: str = "Apple Inc.",
    issuer_cik: str = "0000320193",
    reporter_name: str = "COOK TIMOTHY D",
    reporter_cik: str = "0001214156",
    role: str = "Chief Executive Officer",
    txn_code: str = "P",
    shares: str = "10000",
    price: str = "150.00",
    txn_date: str = "2026-03-15",
    with_namespace: bool = False,
) -> str:
    """Build a minimal but valid Form 4 XML for testing."""
    ns_open = ' xmlns="http://www.sec.gov/edgar/form4"' if with_namespace else ""
    return textwrap.dedent(
        f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <ownershipDocument{ns_open}>
            <issuer>
                <issuerCik>{issuer_cik}</issuerCik>
                <issuerName>{company}</issuerName>
                <issuerTradingSymbol>{ticker}</issuerTradingSymbol>
            </issuer>
            <reportingOwner>
                <reportingOwnerId>
                    <rptOwnerCik>{reporter_cik}</rptOwnerCik>
                    <rptOwnerName>{reporter_name}</rptOwnerName>
                </reportingOwnerId>
                <reportingOwnerRelationship>
                    <officerTitle>{role}</officerTitle>
                </reportingOwnerRelationship>
            </reportingOwner>
            <nonDerivativeTable>
                <nonDerivativeTransaction>
                    <transactionDate><value>{txn_date}</value></transactionDate>
                    <transactionCoding>
                        <transactionCode>{txn_code}</transactionCode>
                    </transactionCoding>
                    <transactionAmounts>
                        <transactionShares><value>{shares}</value></transactionShares>
                        <transactionPricePerShare><value>{price}</value></transactionPricePerShare>
                    </transactionAmounts>
                </nonDerivativeTransaction>
            </nonDerivativeTable>
        </ownershipDocument>
    """
    )


def _make_efts_hit(
    *,
    reporter_cik: str = "0001214156",
    issuer_cik: str = "0000320193",
    reporter_display: str = "COOK TIMOTHY D (CIK 0001214156)",
    issuer_display: str = "Apple Inc. (CIK 0000320193)",
    accession: str = "0001214156-26-001234",
    file_date: str = "2026-03-16",
    primary_doc: str = "form4.xml",
) -> dict[str, Any]:
    """Build a mock EFTS hit for _parse_filings()."""
    return {
        "_source": {
            "ciks": [reporter_cik, issuer_cik],
            "display_names": [reporter_display, issuer_display],
            "file_date": file_date,
            "adsh": accession,
        },
        "_id": f"0000320193:{primary_doc}",
    }


# ═══════════════════════════════════════════════════════════════════════
# 10b.1.1 — CIK fields present in transaction dicts
# ═══════════════════════════════════════════════════════════════════════


class TestParseForm4XmlCIKFields:
    """_parse_form4_xml returns reporter_cik and issuer_cik in every txn."""

    def test_cik_from_xml_authoritative(self):
        """XML-embedded CIKs populate the transaction dict."""
        xml = _make_form4_xml(reporter_cik="0001214156", issuer_cik="0000320193")
        txns = _parse_form4_xml(xml, "FALLBACK NAME", "", "Fallback Co", "2026-01-01")
        assert len(txns) == 1
        assert txns[0]["reporter_cik"] == "0001214156"
        assert txns[0]["issuer_cik"] == "0000320193"

    def test_cik_from_efts_fallback_when_xml_missing(self):
        """When XML lacks CIK elements, EFTS-passed values are used."""
        # Build XML without CIK elements
        xml = textwrap.dedent(
            """\
            <?xml version="1.0" encoding="UTF-8"?>
            <ownershipDocument>
                <issuer>
                    <issuerName>Acme Corp</issuerName>
                    <issuerTradingSymbol>ACME</issuerTradingSymbol>
                </issuer>
                <reportingOwner>
                    <reportingOwnerId>
                        <rptOwnerName>DOE JANE Q</rptOwnerName>
                    </reportingOwnerId>
                    <reportingOwnerRelationship>
                        <officerTitle>CFO</officerTitle>
                    </reportingOwnerRelationship>
                </reportingOwner>
                <nonDerivativeTable>
                    <nonDerivativeTransaction>
                        <transactionDate><value>2026-02-10</value></transactionDate>
                        <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
                        <transactionAmounts>
                            <transactionShares><value>500</value></transactionShares>
                            <transactionPricePerShare><value>25.00</value></transactionPricePerShare>
                        </transactionAmounts>
                    </nonDerivativeTransaction>
                </nonDerivativeTable>
            </ownershipDocument>
        """
        )
        txns = _parse_form4_xml(
            xml,
            "FALLBACK",
            "",
            "Fallback",
            "2026-01-01",
            reporter_cik="0009999999",
            issuer_cik="0008888888",
        )
        assert len(txns) == 1
        assert txns[0]["reporter_cik"] == "0009999999"
        assert txns[0]["issuer_cik"] == "0008888888"

    def test_xml_cik_overrides_efts_cik(self):
        """XML CIK takes precedence over EFTS-provided CIK."""
        xml = _make_form4_xml(reporter_cik="0001111111", issuer_cik="0002222222")
        txns = _parse_form4_xml(
            xml,
            "NAME",
            "",
            "Co",
            "2026-01-01",
            reporter_cik="0009999999",
            issuer_cik="0008888888",
        )
        assert txns[0]["reporter_cik"] == "0001111111"
        assert txns[0]["issuer_cik"] == "0002222222"

    def test_empty_cik_defaults(self):
        """With no EFTS fallback and no XML CIK, fields are empty strings."""
        xml = textwrap.dedent(
            """\
            <?xml version="1.0" encoding="UTF-8"?>
            <ownershipDocument>
                <issuer>
                    <issuerTradingSymbol>XYZ</issuerTradingSymbol>
                </issuer>
                <reportingOwner>
                    <reportingOwnerId>
                        <rptOwnerName>SMITH JOHN</rptOwnerName>
                    </reportingOwnerId>
                </reportingOwner>
                <nonDerivativeTable>
                    <nonDerivativeTransaction>
                        <transactionDate><value>2026-01-05</value></transactionDate>
                        <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
                        <transactionAmounts>
                            <transactionShares><value>100</value></transactionShares>
                            <transactionPricePerShare><value>10.00</value></transactionPricePerShare>
                        </transactionAmounts>
                    </nonDerivativeTransaction>
                </nonDerivativeTable>
            </ownershipDocument>
        """
        )
        txns = _parse_form4_xml(xml, "SMITH JOHN", "", "", "2026-01-01")
        assert txns[0]["reporter_cik"] == ""
        assert txns[0]["issuer_cik"] == ""

    def test_cik_with_namespace(self):
        """CIK extraction works with namespaced XML."""
        xml = _make_form4_xml(
            reporter_cik="0003333333",
            issuer_cik="0004444444",
            with_namespace=True,
        )
        txns = _parse_form4_xml(xml, "N", "", "C", "2026-01-01")
        assert txns[0]["reporter_cik"] == "0003333333"
        assert txns[0]["issuer_cik"] == "0004444444"

    def test_multiple_purchases_same_ciks(self):
        """All transactions from one filing share the same CIKs."""
        xml = textwrap.dedent(
            """\
            <?xml version="1.0" encoding="UTF-8"?>
            <ownershipDocument>
                <issuer>
                    <issuerCik>0000320193</issuerCik>
                    <issuerName>Apple Inc.</issuerName>
                    <issuerTradingSymbol>AAPL</issuerTradingSymbol>
                </issuer>
                <reportingOwner>
                    <reportingOwnerId>
                        <rptOwnerCik>0001214156</rptOwnerCik>
                        <rptOwnerName>COOK TIMOTHY D</rptOwnerName>
                    </reportingOwnerId>
                </reportingOwner>
                <nonDerivativeTable>
                    <nonDerivativeTransaction>
                        <transactionDate><value>2026-03-10</value></transactionDate>
                        <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
                        <transactionAmounts>
                            <transactionShares><value>5000</value></transactionShares>
                            <transactionPricePerShare><value>148.50</value></transactionPricePerShare>
                        </transactionAmounts>
                    </nonDerivativeTransaction>
                    <nonDerivativeTransaction>
                        <transactionDate><value>2026-03-11</value></transactionDate>
                        <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
                        <transactionAmounts>
                            <transactionShares><value>3000</value></transactionShares>
                            <transactionPricePerShare><value>151.00</value></transactionPricePerShare>
                        </transactionAmounts>
                    </nonDerivativeTransaction>
                </nonDerivativeTable>
            </ownershipDocument>
        """
        )
        txns = _parse_form4_xml(xml, "N", "", "C", "2026-01-01")
        assert len(txns) == 2
        for t in txns:
            assert t["reporter_cik"] == "0001214156"
            assert t["issuer_cik"] == "0000320193"

    def test_non_purchase_excluded_still(self):
        """Sales (S) and other codes still excluded — CIK doesn't change filtering."""
        xml = _make_form4_xml(txn_code="S", shares="5000")
        txns = _parse_form4_xml(xml, "N", "", "C", "2026-01-01")
        assert len(txns) == 0

    def test_zero_shares_excluded_still(self):
        """Zero-share transactions still filtered out."""
        xml = _make_form4_xml(shares="0")
        txns = _parse_form4_xml(xml, "N", "", "C", "2026-01-01")
        assert len(txns) == 0

    def test_malformed_xml_returns_empty(self):
        """Malformed XML returns empty list, not crash."""
        txns = _parse_form4_xml(
            "<<<not xml>>>",
            "N",
            "",
            "C",
            "2026-01-01",
            reporter_cik="0001111111",
            issuer_cik="0002222222",
        )
        assert txns == []

    def test_backward_compat_no_keyword_args(self):
        """Calling without CIK keyword args still works (default empty strings)."""
        xml = _make_form4_xml()
        txns = _parse_form4_xml(xml, "N", "", "C", "2026-01-01")
        # CIKs come from XML in this case
        assert txns[0]["reporter_cik"] == "0001214156"
        assert txns[0]["issuer_cik"] == "0000320193"


# ═══════════════════════════════════════════════════════════════════════
# 10b.1.1 — _parse_filings passes CIKs through
# ═══════════════════════════════════════════════════════════════════════


class TestParseFilingsCIKPassthrough:
    """_parse_filings extracts EFTS CIKs and passes them to _parse_form4_xml."""

    def _make_tool(self) -> InsiderFilingsTool:
        return InsiderFilingsTool(cache=None)

    def test_cik_passed_to_parser(self):
        """EFTS reporter_cik (ciks[0]) and issuer_cik (ciks[1]) reach _parse_form4_xml."""
        tool = self._make_tool()
        xml = _make_form4_xml(reporter_cik="0001214156", issuer_cik="0000320193")
        hit = _make_efts_hit(reporter_cik="0001214156", issuer_cik="0000320193")

        with patch.object(tool, "_fetch_filing_xml", return_value=xml):
            txns = tool._parse_filings([hit])

        assert len(txns) == 1
        assert txns[0]["reporter_cik"] == "0001214156"
        assert txns[0]["issuer_cik"] == "0000320193"

    def test_cik_from_multiple_filings(self):
        """Each filing gets its own CIK pair, not leftover state from prior ones."""
        tool = self._make_tool()

        xml_a = _make_form4_xml(
            reporter_cik="0001111111",
            issuer_cik="0002222222",
            ticker="AAA",
            company="Aaa Inc.",
            reporter_name="ALICE A",
        )
        xml_b = _make_form4_xml(
            reporter_cik="0003333333",
            issuer_cik="0004444444",
            ticker="BBB",
            company="Bbb Inc.",
            reporter_name="BOB B",
        )

        hit_a = _make_efts_hit(
            reporter_cik="0001111111",
            issuer_cik="0002222222",
            accession="0001111111-26-000001",
        )
        hit_b = _make_efts_hit(
            reporter_cik="0003333333",
            issuer_cik="0004444444",
            accession="0003333333-26-000002",
        )

        xmls = {"0002222222": xml_a, "0004444444": xml_b}

        def fetch_xml(cik, accession, primary_doc):
            return xmls.get(cik)

        with patch.object(tool, "_fetch_filing_xml", side_effect=fetch_xml):
            txns = tool._parse_filings([hit_a, hit_b])

        assert len(txns) == 2
        a = next(t for t in txns if t["ticker"] == "AAA")
        b = next(t for t in txns if t["ticker"] == "BBB")
        assert a["reporter_cik"] == "0001111111"
        assert a["issuer_cik"] == "0002222222"
        assert b["reporter_cik"] == "0003333333"
        assert b["issuer_cik"] == "0004444444"

    def test_efts_hit_with_fewer_than_2_ciks_skipped(self):
        """EFTS hits with < 2 CIKs are skipped (existing behavior, unchanged)."""
        tool = self._make_tool()
        hit = {
            "_source": {
                "ciks": ["0001111111"],
                "display_names": ["PERSON A"],
                "file_date": "2026-03-01",
                "adsh": "0001111111-26-000001",
            },
            "_id": "0001111111:form4.xml",
        }
        with patch.object(tool, "_fetch_filing_xml") as mock_fetch:
            txns = tool._parse_filings([hit])
        assert txns == []
        mock_fetch.assert_not_called()

    def test_efts_hit_with_empty_accession_skipped(self):
        """EFTS hits with no accession number are skipped (existing behavior)."""
        tool = self._make_tool()
        hit = _make_efts_hit(accession="")
        with patch.object(tool, "_fetch_filing_xml") as mock_fetch:
            txns = tool._parse_filings([hit])
        assert txns == []
        mock_fetch.assert_not_called()

    def test_fetch_returns_none_no_crash(self):
        """When XML fetch fails (returns None), no transactions and no crash."""
        tool = self._make_tool()
        hit = _make_efts_hit()
        with patch.object(tool, "_fetch_filing_xml", return_value=None):
            txns = tool._parse_filings([hit])
        assert txns == []


# ═══════════════════════════════════════════════════════════════════════
# 10b.1.1 — Cluster detection ignores new fields (backward compat)
# ═══════════════════════════════════════════════════════════════════════


class TestClusterDetectionBackwardCompat:
    """New CIK fields don't break cluster detection (which uses name-based dedup)."""

    def test_clusters_still_work_with_cik_fields(self):
        """Cluster detection uses name-based dedup; extra CIK keys are harmless."""
        txns = [
            {
                "ticker": "AAPL",
                "company": "Apple Inc.",
                "name": f"INSIDER_{i}",
                "role": "Director",
                "type": "P",
                "shares": 1000,
                "price": 150.0,
                "date": f"2026-03-{10 + i:02d}",
                "reporter_cik": f"000{i:07d}",
                "issuer_cik": "0000320193",
            }
            for i in range(4)
        ]
        tool = InsiderFilingsTool()
        clusters = tool._detect_clusters(txns, min_size=3)
        assert len(clusters) == 1
        assert clusters[0]["insider_count"] == 4
        # CIK fields are carried through in the insider list
        for ins in clusters[0]["insiders"]:
            assert "reporter_cik" in ins
            assert "issuer_cik" in ins

    def test_cluster_conviction_unchanged(self):
        """Conviction scoring still works — CIK fields don't interfere."""
        txns = [
            {
                "ticker": "MSFT",
                "company": "Microsoft",
                "name": "CEO PERSON",
                "role": "Chief Executive Officer",
                "type": "P",
                "shares": 50000,
                "price": 300.0,
                "date": "2026-03-10",
                "reporter_cik": "0001000001",
                "issuer_cik": "0000789012",
            },
        ] + [
            {
                "ticker": "MSFT",
                "company": "Microsoft",
                "name": f"DIRECTOR_{i}",
                "role": "Director",
                "type": "P",
                "shares": 5000,
                "price": 300.0,
                "date": f"2026-03-{11 + i:02d}",
                "reporter_cik": f"000200000{i}",
                "issuer_cik": "0000789012",
            }
            for i in range(3)
        ]
        tool = InsiderFilingsTool()
        clusters = tool._detect_clusters(txns, min_size=3)
        assert len(clusters) == 1
        # Has C-suite + 4 total → "high"
        assert clusters[0]["conviction"] == "high"


# ═══════════════════════════════════════════════════════════════════════
# 10b.1.2 — Optional PipelineStore in constructor
# ═══════════════════════════════════════════════════════════════════════


class TestConstructorPipelineStore:
    """InsiderFilingsTool accepts an optional pipeline_store kwarg."""

    def test_default_no_store(self):
        """Without pipeline_store kwarg, _store is None."""
        tool = InsiderFilingsTool()
        assert tool._store is None
        assert tool._cache is None

    def test_with_cache_only(self):
        """Providing only cache still works, _store stays None."""
        mock_cache = MagicMock()
        tool = InsiderFilingsTool(cache=mock_cache)
        assert tool._cache is mock_cache
        assert tool._store is None

    def test_with_pipeline_store(self):
        """Providing pipeline_store sets _store attribute."""
        mock_store = MagicMock()
        tool = InsiderFilingsTool(pipeline_store=mock_store)
        assert tool._store is mock_store
        assert tool._cache is None

    def test_with_both_cache_and_store(self):
        """Both cache and pipeline_store can be provided together."""
        mock_cache = MagicMock()
        mock_store = MagicMock()
        tool = InsiderFilingsTool(cache=mock_cache, pipeline_store=mock_store)
        assert tool._cache is mock_cache
        assert tool._store is mock_store

    def test_pipeline_store_is_keyword_only(self):
        """pipeline_store cannot be passed positionally."""
        mock_store = MagicMock()
        with pytest.raises(TypeError):
            InsiderFilingsTool(None, mock_store)  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════
# 10b.1.3 — _persist_entities: entity registration + observations
# ═══════════════════════════════════════════════════════════════════════


def _make_txn(
    *,
    ticker: str = "AAPL",
    company: str = "Apple Inc.",
    name: str = "COOK TIMOTHY D",
    role: str = "CEO",
    shares: float = 10000,
    price: float = 150.0,
    date: str = "2026-03-15",
    reporter_cik: str = "0001214156",
    issuer_cik: str = "0000320193",
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "company": company,
        "name": name,
        "role": role,
        "type": "P",
        "shares": shares,
        "price": price,
        "date": date,
        "reporter_cik": reporter_cik,
        "issuer_cik": issuer_cik,
    }


class TestPersistEntities:
    """_persist_entities registers companies, insiders, and observations."""

    def _make_tool_with_store(self) -> tuple[InsiderFilingsTool, PipelineStore]:
        store = PipelineStore(db_path=":memory:")
        tool = InsiderFilingsTool(pipeline_store=store)
        return tool, store

    def test_registers_company_and_insider(self):
        tool, store = self._make_tool_with_store()
        txns = [_make_txn()]
        tool._persist_entities(txns)

        company_eid = entity_id_from_key("company", "0000320193")
        insider_eid = entity_id_from_key("person", "0001214156")

        company = store.get_entity(company_eid)
        assert company is not None
        assert company["entity_type"] == "company"

        insider = store.get_entity(insider_eid)
        assert insider is not None
        assert insider["entity_type"] == "person"

    def test_stores_observation_for_insider(self):
        tool, store = self._make_tool_with_store()
        txns = [_make_txn()]
        tool._persist_entities(txns)

        insider_eid = entity_id_from_key("person", "0001214156")
        obs = store.query_entity_observations(insider_eid, source_tool="insider_filings")
        assert len(obs) == 1
        assert obs[0]["observation_type"] == "insider_trade"
        assert obs[0]["depth_level"] == 2
        val = obs[0]["value"]
        assert val["ticker"] == "AAPL"
        assert val["shares"] == 10000

    def test_company_aliases_cik_and_ticker(self):
        tool, store = self._make_tool_with_store()
        txns = [_make_txn(ticker="AAPL", issuer_cik="0000320193")]
        tool._persist_entities(txns)

        company_eid = entity_id_from_key("company", "0000320193")
        # Check CIK alias
        resolved = store.resolve_entity("sec_cik", "0000320193")
        assert resolved == company_eid
        # Check ticker alias
        resolved_t = store.resolve_entity("ticker", "AAPL")
        assert resolved_t == company_eid

    def test_insider_alias_cik(self):
        tool, store = self._make_tool_with_store()
        txns = [_make_txn(reporter_cik="0001214156")]
        tool._persist_entities(txns)

        insider_eid = entity_id_from_key("person", "0001214156")
        resolved = store.resolve_entity("sec_cik", "0001214156")
        assert resolved == insider_eid

    def test_deduplicates_same_company(self):
        """Two transactions with same issuer_cik → company registered once."""
        tool, store = self._make_tool_with_store()
        txns = [
            _make_txn(name="A", reporter_cik="0001111111", issuer_cik="0000320193"),
            _make_txn(name="B", reporter_cik="0002222222", issuer_cik="0000320193"),
        ]
        tool._persist_entities(txns)
        # Should still resolve (no duplicate errors)
        eid = store.resolve_entity("sec_cik", "0000320193")
        assert eid == entity_id_from_key("company", "0000320193")

    def test_deduplicates_same_insider(self):
        """Two transactions with same reporter_cik → insider registered once, two observations."""
        tool, store = self._make_tool_with_store()
        txns = [
            _make_txn(date="2026-03-10", reporter_cik="0001111111"),
            _make_txn(date="2026-03-11", reporter_cik="0001111111"),
        ]
        tool._persist_entities(txns)

        insider_eid = entity_id_from_key("person", "0001111111")
        obs = store.query_entity_observations(insider_eid, source_tool="insider_filings")
        assert len(obs) == 2

    def test_skips_when_no_store(self):
        """No store → no-op, no error."""
        tool = InsiderFilingsTool()
        tool._persist_entities([_make_txn()])  # should not raise

    def test_skips_empty_transactions(self):
        tool, store = self._make_tool_with_store()
        tool._persist_entities([])  # should not raise

    def test_skips_missing_reporter_cik(self):
        """Transaction with empty reporter_cik → no insider registered, no observation."""
        tool, store = self._make_tool_with_store()
        txns = [_make_txn(reporter_cik="", issuer_cik="0000320193")]
        tool._persist_entities(txns)

        # Company still registered
        assert store.resolve_entity("sec_cik", "0000320193") is not None

    def test_skips_missing_issuer_cik(self):
        """Transaction with empty issuer_cik → no company registered, insider still registered."""
        tool, store = self._make_tool_with_store()
        txns = [_make_txn(issuer_cik="", reporter_cik="0001111111")]
        tool._persist_entities(txns)

        insider_eid = entity_id_from_key("person", "0001111111")
        assert store.get_entity(insider_eid) is not None

    def test_persistence_failure_non_fatal(self):
        """PipelineStore write failure → tool continues, no exception raised."""
        mock_store = MagicMock()
        mock_store.register_entity.side_effect = RuntimeError("DB locked")
        tool = InsiderFilingsTool(pipeline_store=mock_store)
        # Must not raise
        tool._persist_entities([_make_txn()])

    def test_normalize_failure_uses_raw_name(self):
        """If company name normalization fails, raw name is used."""
        tool, store = self._make_tool_with_store()
        # A company name that normalizes to empty after suffix removal
        txns = [_make_txn(company="Inc.", issuer_cik="0009999999")]
        tool._persist_entities(txns)

        eid = entity_id_from_key("company", "0009999999")
        entity = store.get_entity(eid)
        assert entity is not None
        # Should have fallen back to raw name "Inc." or the CIK
        assert entity["canonical_name"] in ("Inc.", "0009999999")

    def test_unicode_insider_name(self):
        """Unicode characters in insider name → registered without error."""
        tool, store = self._make_tool_with_store()
        txns = [_make_txn(name="MÜLLER HANS-JÖRG", reporter_cik="0005555555")]
        tool._persist_entities(txns)

        eid = entity_id_from_key("person", "0005555555")
        entity = store.get_entity(eid)
        assert entity is not None
        assert "MÜLLER" in entity["canonical_name"]

    def test_zero_shares_still_persisted(self):
        """Zero-share/zero-price txns are still stored as observations (amendments)."""
        tool, store = self._make_tool_with_store()
        txns = [_make_txn(shares=0, price=0)]
        # Note: _parse_form4_xml filters zero shares, but _persist_entities doesn't
        # care — it stores whatever transactions it receives.
        tool._persist_entities(txns)

        eid = entity_id_from_key("person", "0001214156")
        obs = store.query_entity_observations(eid)
        assert len(obs) == 1
        assert obs[0]["value"]["shares"] == 0

    def test_multiple_companies_across_transactions(self):
        """Three different companies → three entity registrations."""
        tool, store = self._make_tool_with_store()
        txns = [
            _make_txn(ticker="AAPL", company="Apple", issuer_cik="0001", reporter_cik="0010"),
            _make_txn(
                ticker="MSFT",
                company="Microsoft",
                issuer_cik="0002",
                reporter_cik="0020",
            ),
            _make_txn(
                ticker="GOOG",
                company="Alphabet",
                issuer_cik="0003",
                reporter_cik="0030",
            ),
        ]
        tool._persist_entities(txns)

        for cik in ("0001", "0002", "0003"):
            assert store.resolve_entity("sec_cik", cik) is not None

    def test_observation_timestamp_from_date(self):
        """observed_at is a valid timestamp derived from the transaction date."""
        tool, store = self._make_tool_with_store()
        txns = [_make_txn(date="2026-03-15")]
        tool._persist_entities(txns)

        eid = entity_id_from_key("person", "0001214156")
        obs = store.query_entity_observations(eid)
        assert obs[0]["observed_at"] > 0  # valid timestamp

    def test_observation_bad_date_uses_zero(self):
        """Invalid date string → observed_at = 0.0."""
        tool, store = self._make_tool_with_store()
        txns = [_make_txn(date="not-a-date")]
        tool._persist_entities(txns)

        eid = entity_id_from_key("person", "0001214156")
        obs = store.query_entity_observations(eid)
        assert obs[0]["observed_at"] == 0.0


# ═══════════════════════════════════════════════════════════════════════
# 10b.1.4 — CIK-based dedup in _find_best_cluster
# ═══════════════════════════════════════════════════════════════════════


class TestCIKBasedDedup:
    """_find_best_cluster uses reporter_cik for dedup when available."""

    def test_same_name_different_cik_counted_as_two(self):
        """Two insiders with identical names but different CIKs → 2 distinct."""
        buys = [
            _make_txn(name="JOHN SMITH", reporter_cik="0001111111", date="2026-03-10"),
            _make_txn(name="JOHN SMITH", reporter_cik="0002222222", date="2026-03-11"),
            _make_txn(name="JOHN SMITH", reporter_cik="0003333333", date="2026-03-12"),
        ]
        result = InsiderFilingsTool._find_best_cluster(buys, window_days=14, min_size=3)
        assert result is not None
        assert result["insider_count"] == 3

    def test_same_cik_different_names_counted_as_one(self):
        """Same CIK with variant name spellings → counted as 1 insider."""
        buys = [
            _make_txn(name="SMITH JOHN A", reporter_cik="0001111111", date="2026-03-10"),
            _make_txn(name="SMITH JOHN", reporter_cik="0001111111", date="2026-03-11"),
            _make_txn(name="SMITH, JOHN A.", reporter_cik="0001111111", date="2026-03-12"),
        ]
        result = InsiderFilingsTool._find_best_cluster(buys, window_days=14, min_size=2)
        # Only 1 distinct insider (same CIK), so can't form a cluster of 2
        assert result is None

    def test_empty_cik_falls_back_to_name(self):
        """No reporter_cik → falls back to name-based dedup."""
        buys = [
            _make_txn(name="ALICE A", reporter_cik="", date="2026-03-10"),
            _make_txn(name="BOB B", reporter_cik="", date="2026-03-11"),
            _make_txn(name="CAROL C", reporter_cik="", date="2026-03-12"),
        ]
        result = InsiderFilingsTool._find_best_cluster(buys, window_days=14, min_size=3)
        assert result is not None
        assert result["insider_count"] == 3

    def test_mixed_cik_and_no_cik(self):
        """Mix of CIK and no-CIK transactions deduplicates correctly."""
        buys = [
            _make_txn(name="ALICE", reporter_cik="0001111111", date="2026-03-10"),
            _make_txn(name="ALICE", reporter_cik="", date="2026-03-11"),  # no CIK → name dedup
            _make_txn(name="BOB", reporter_cik="0002222222", date="2026-03-12"),
        ]
        result = InsiderFilingsTool._find_best_cluster(buys, window_days=14, min_size=3)
        # "0001111111", "ALICE" (name fallback), "0002222222" → 3 distinct
        assert result is not None
        assert result["insider_count"] == 3


# ═══════════════════════════════════════════════════════════════════════
# 10b.1.5 — entity_ids mapping in cluster data
# ═══════════════════════════════════════════════════════════════════════


class TestEntityIdsInClusters:
    """Cluster dicts contain entity_ids mapping {name: entity_id}."""

    def test_entity_ids_present(self):
        txns = [
            _make_txn(
                name=f"INSIDER_{i}",
                reporter_cik=f"000{i:07d}",
                date=f"2026-03-{10 + i:02d}",
            )
            for i in range(3)
        ]
        tool = InsiderFilingsTool()
        clusters = tool._detect_clusters(txns, min_size=3)
        assert len(clusters) == 1
        eid_map = clusters[0]["entity_ids"]
        assert isinstance(eid_map, dict)
        assert len(eid_map) == 3
        for i in range(3):
            name = f"INSIDER_{i}"
            assert name in eid_map
            assert eid_map[name] == entity_id_from_key("person", f"000{i:07d}")

    def test_entity_ids_empty_when_no_cik(self):
        """When all transactions lack CIKs, entity_ids is empty dict."""
        txns = [_make_txn(name=f"PERSON_{i}", reporter_cik="", date=f"2026-03-{10 + i:02d}") for i in range(3)]
        tool = InsiderFilingsTool()
        clusters = tool._detect_clusters(txns, min_size=3)
        assert clusters[0]["entity_ids"] == {}

    def test_entity_ids_partial(self):
        """Mix of CIK and no-CIK → only CIK-bearing insiders appear in entity_ids."""
        txns = [
            _make_txn(name="ALICE", reporter_cik="0001111111", date="2026-03-10"),
            _make_txn(name="BOB", reporter_cik="", date="2026-03-11"),
            _make_txn(name="CAROL", reporter_cik="0003333333", date="2026-03-12"),
        ]
        tool = InsiderFilingsTool()
        clusters = tool._detect_clusters(txns, min_size=3)
        eid_map = clusters[0]["entity_ids"]
        assert "ALICE" in eid_map
        assert "CAROL" in eid_map
        assert "BOB" not in eid_map


# ═══════════════════════════════════════════════════════════════════════
# 10b.1.6 — Additional edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Comprehensive edge cases across all L2 changes."""

    def test_execute_calls_persist_entities(self):
        """execute() calls _persist_entities after parsing."""
        tool = InsiderFilingsTool()
        with (
            patch.object(tool, "_fetch_recent_filings", return_value=[]),
            patch.object(tool, "_persist_entities") as mock_persist,
        ):
            tool.execute(days_back=7)
        # No filings → early return before _persist_entities
        mock_persist.assert_not_called()

    def test_execute_calls_persist_with_transactions(self):
        """When filings parse to transactions, _persist_entities is called."""
        xml = _make_form4_xml()
        hit = _make_efts_hit()
        tool = InsiderFilingsTool()
        with (
            patch.object(tool, "_fetch_recent_filings", return_value=[hit]),
            patch.object(tool, "_fetch_filing_xml", return_value=xml),
            patch.object(tool, "_persist_entities") as mock_persist,
        ):
            tool.execute(days_back=7)
        mock_persist.assert_called_once()
        txns = mock_persist.call_args[0][0]
        assert len(txns) == 1
        assert txns[0]["reporter_cik"] == "0001214156"

    def test_persist_failure_does_not_break_execute(self):
        """If _persist_entities raises, execute() still returns ToolResult."""
        xml = _make_form4_xml()
        hit = _make_efts_hit()
        tool = InsiderFilingsTool()
        with (
            patch.object(tool, "_fetch_recent_filings", return_value=[hit]),
            patch.object(tool, "_fetch_filing_xml", return_value=xml),
            patch.object(tool, "_persist_entities", side_effect=Exception("boom")),
        ):
            result = tool.execute(days_back=7)
        # Should still return a result (no clusters since only 1 insider)
        assert result.success is True

    def test_idempotent_entity_registration(self):
        """Re-persisting the same transactions doesn't error (INSERT OR IGNORE)."""
        store = PipelineStore(db_path=":memory:")
        tool = InsiderFilingsTool(pipeline_store=store)
        txns = [_make_txn()]
        tool._persist_entities(txns)
        tool._persist_entities(txns)  # second call should not raise

        eid = entity_id_from_key("person", "0001214156")
        obs = store.query_entity_observations(eid, source_tool="insider_filings")
        # Observations stored twice (amendments are allowed)
        assert len(obs) == 2

    def test_same_insider_multiple_companies(self):
        """One insider filing at two companies → two observations, insider registered once."""
        store = PipelineStore(db_path=":memory:")
        tool = InsiderFilingsTool(pipeline_store=store)
        txns = [
            _make_txn(
                ticker="AAPL",
                company="Apple",
                issuer_cik="0000320193",
                reporter_cik="0001214156",
            ),
            _make_txn(
                ticker="MSFT",
                company="Microsoft",
                issuer_cik="0000789012",
                reporter_cik="0001214156",
            ),
        ]
        tool._persist_entities(txns)

        eid = entity_id_from_key("person", "0001214156")
        obs = store.query_entity_observations(eid, source_tool="insider_filings")
        assert len(obs) == 2
        tickers = {o["value"]["ticker"] for o in obs}
        assert tickers == {"AAPL", "MSFT"}

    def test_company_name_empty_uses_cik_as_canonical(self):
        """Empty company name → uses CIK as canonical_name fallback."""
        store = PipelineStore(db_path=":memory:")
        tool = InsiderFilingsTool(pipeline_store=store)
        txns = [_make_txn(company="", issuer_cik="0009999999")]
        tool._persist_entities(txns)

        eid = entity_id_from_key("company", "0009999999")
        entity = store.get_entity(eid)
        assert entity is not None
        assert entity["canonical_name"] == "0009999999"
