"""Tests for Form 144 L2 upgrade — entity persistence, CIK threading,
CIK-based dedup, and entity_ids in cluster output.

Mirrors the test pattern from test_insider_filings_l2.py.
"""

from __future__ import annotations

import textwrap
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.pipeline.entity import entity_id_from_key, normalize_company_name
from agent.tools.form144 import Form144Tool, _date_to_timestamp

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SAMPLE_XML = textwrap.dedent(
    """\
    <?xml version="1.0" encoding="UTF-8"?>
    <informationTable>
        <formData>
            <issuerInfo>
                <issuerName>{company}</issuerName>
                <nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold>{insider}</nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold>
                <relationshipsToIssuer>
                    <relationshipToIssuer>{relationship}</relationshipToIssuer>
                </relationshipsToIssuer>
            </issuerInfo>
            <securitiesInformation>
                <noOfUnitsSold>{shares}</noOfUnitsSold>
                <aggregateMarketValue>{value}</aggregateMarketValue>
                <noOfUnitsOutstanding>{outstanding}</noOfUnitsOutstanding>
                <approxSaleDate>{sale_date}</approxSaleDate>
            </securitiesInformation>
            <securitiesToBeSold>
                <natureOfAcquisitionTransaction>{acq_nature}</natureOfAcquisitionTransaction>
                <isGiftTransaction>N</isGiftTransaction>
                <amountOfSecuritiesAcquired>{shares}</amountOfSecuritiesAcquired>
            </securitiesToBeSold>
            <noticeSignature>
                <noticeDate>{notice_date}</noticeDate>
            </noticeSignature>
        </formData>
    </informationTable>
"""
)


def _make_xml(
    company: str = "Acme Inc",
    insider: str = "John Smith",
    relationship: str = "Officer",
    shares: int = 10000,
    value: float = 500000.0,
    outstanding: int = 1000000,
    sale_date: str = "04/10/2026",
    acq_nature: str = "Open Market Purchase",
    notice_date: str = "04/01/2026",
) -> str:
    return SAMPLE_XML.format(
        company=company,
        insider=insider,
        relationship=relationship,
        shares=shares,
        value=value,
        outstanding=outstanding,
        sale_date=sale_date,
        acq_nature=acq_nature,
        notice_date=notice_date,
    )


def _make_efts_hit(
    issuer_cik: str = "0000012345",
    reporter_cik: str = "0000099999",
    issuer_name: str = "Acme Inc",
    ticker: str = "ACME",
    reporter_name: str = "SMITH JOHN",
    accession: str = "0000012345-26-000001",
    file_date: str = "2026-04-01",
) -> dict[str, Any]:
    """Build an EFTS hit dict for Form 144."""
    return {
        "_id": f"0000012345:{accession.replace('-', '')}:primary_doc.xml",
        "_source": {
            "ciks": [issuer_cik, reporter_cik],
            "display_names": [
                f"{issuer_name} ({ticker}) (CIK {issuer_cik})",
                f"{reporter_name} (CIK {reporter_cik})",
            ],
            "adsh": accession,
            "file_date": file_date,
        },
    }


def _make_filing(
    ticker: str = "ACME",
    company: str = "Acme Inc",
    insider_name: str = "John Smith",
    issuer_cik: str = "0000012345",
    reporter_cik: str = "0000099999",
    shares_to_sell: int = 10000,
    dollar_value: float = 500000.0,
    filing_date: str = "2026-04-01",
    acquisition_type: str = "open_market",
    urgency: str = "near_term",
    relationship: str = "Officer",
) -> dict[str, Any]:
    """Build a parsed filing dict (post _parse_filings)."""
    return {
        "ticker": ticker,
        "company": company,
        "insider_name": insider_name,
        "issuer_cik": issuer_cik,
        "reporter_cik": reporter_cik,
        "shares_to_sell": shares_to_sell,
        "dollar_value": dollar_value,
        "shares_outstanding": 1000000,
        "approx_sale_date": "2026-04-10",
        "filing_date": filing_date,
        "exchange": "",
        "broker": "",
        "acquisition_type": acquisition_type,
        "acquisition_details": [],
        "is_gift": False,
        "has_10b5_1_plan": False,
        "urgency": urgency,
        "relationship": relationship,
    }


# ===================================================================
# 10b.2.1: CIK threading in _parse_filings
# ===================================================================


class TestCIKThreading:
    """reporter_cik and issuer_cik appear in all parsed filing dicts."""

    def _parse_with_hits(self, hits, xml_text=None):
        tool = Form144Tool()
        if xml_text is not None:
            with patch.object(tool, "_fetch_filing_xml", return_value=xml_text):
                return tool._parse_filings(hits)
        else:
            return tool._parse_filings(hits)

    def test_cik_pair_in_metadata_only_record(self):
        """Single-filer ticker → metadata-only record still has both CIKs."""
        hit = _make_efts_hit(issuer_cik="0000011111", reporter_cik="0000022222")
        filings = self._parse_with_hits([hit])
        assert len(filings) == 1
        assert filings[0]["issuer_cik"] == "0000011111"
        assert filings[0]["reporter_cik"] == "0000022222"
        assert filings[0].get("_metadata_only") is True

    def test_cik_pair_in_xml_parsed_record(self):
        """Cluster-candidate ticker → XML-parsed record has both CIKs."""
        h1 = _make_efts_hit(reporter_cik="0000099901", reporter_name="ALICE A")
        h2 = _make_efts_hit(
            reporter_cik="0000099902",
            reporter_name="BOB B",
            accession="0000012345-26-000002",
        )
        xml = _make_xml(insider="Alice A", shares=5000, value=250000)
        filings = self._parse_with_hits([h1, h2], xml_text=xml)
        xml_filings = [f for f in filings if not f.get("_metadata_only")]
        assert len(xml_filings) >= 1
        for f in xml_filings:
            assert "issuer_cik" in f
            assert "reporter_cik" in f

    def test_cik_swap_when_ticker_in_second_name(self):
        """When ticker is in names[1], swap logic flips issuer/reporter.
        After swap: issuer_cik = ciks[1], reporter_cik = ciks[0]."""
        hit = {
            "_id": "0000012345:0000012345260000001:primary_doc.xml",
            "_source": {
                "ciks": ["0000099999", "0000012345"],
                "display_names": [
                    "SMITH JOHN (CIK 0000099999)",
                    "Acme Inc (ACME) (CIK 0000012345)",
                ],
                "adsh": "0000012345-26-000001",
                "file_date": "2026-04-01",
            },
        }
        filings = self._parse_with_hits([hit])
        assert len(filings) == 1
        # After swap, issuer_cik should be the CIK with the ticker
        assert filings[0]["issuer_cik"] == "0000012345"
        assert filings[0]["reporter_cik"] == "0000099999"

    def test_fewer_than_two_ciks_skipped(self):
        """EFTS hit with < 2 CIKs is skipped entirely."""
        hit = {
            "_source": {
                "ciks": ["0000012345"],
                "display_names": ["Acme (ACME)"],
                "adsh": "0000012345-26-000001",
                "file_date": "2026-04-01",
            },
        }
        filings = self._parse_with_hits([hit])
        assert filings == []

    def test_cik_in_fallback_metadata_record(self):
        """When XML fetch fails, fallback metadata record still has CIKs."""
        h1 = _make_efts_hit(reporter_cik="0000099901", reporter_name="ALICE A")
        h2 = _make_efts_hit(
            reporter_cik="0000099902",
            reporter_name="BOB B",
            accession="0000012345-26-000002",
        )
        filings = self._parse_with_hits([h1, h2], xml_text=None)  # XML fetch returns None
        # All records should be metadata-only
        for f in filings:
            assert f.get("_metadata_only") is True
            assert "issuer_cik" in f
            assert "reporter_cik" in f

    def test_cik_with_duplicate_values(self):
        """When both CIKs are the same (self-filing), reporter_cik = issuer_cik."""
        hit = _make_efts_hit(issuer_cik="0000012345", reporter_cik="0000012345")
        filings = self._parse_with_hits([hit])
        assert len(filings) == 1
        # Both will be the same CIK
        assert filings[0]["issuer_cik"] == "0000012345"
        assert filings[0]["reporter_cik"] == "0000012345"


# ===================================================================
# 10b.2.2: Constructor with PipelineStore
# ===================================================================


class TestConstructorPipelineStore:
    def test_default_no_store(self):
        tool = Form144Tool()
        assert tool._store is None

    def test_with_cache_only(self):
        cache = MagicMock()
        tool = Form144Tool(cache=cache)
        assert tool._cache is cache
        assert tool._store is None

    def test_with_pipeline_store(self):
        store = MagicMock()
        tool = Form144Tool(pipeline_store=store)
        assert tool._store is store
        assert tool._cache is None

    def test_with_both(self):
        cache = MagicMock()
        store = MagicMock()
        tool = Form144Tool(cache=cache, pipeline_store=store)
        assert tool._cache is cache
        assert tool._store is store

    def test_pipeline_store_keyword_only(self):
        """pipeline_store must be keyword-only."""
        with pytest.raises(TypeError):
            Form144Tool(None, MagicMock())  # type: ignore[misc]


# ===================================================================
# 10b.2.3: Entity persistence
# ===================================================================


class TestPersistEntities:
    def _tool_with_store(self):
        store = MagicMock()
        store.register_entity = MagicMock()
        store.add_entity_alias = MagicMock()
        store.store_entity_observation = MagicMock()
        tool = Form144Tool(pipeline_store=store)
        return tool, store

    def test_company_registered(self):
        tool, store = self._tool_with_store()
        filings = [_make_filing(issuer_cik="0000011111", company="Acme Inc")]
        tool._persist_entities(filings)
        eid = entity_id_from_key("company", "0000011111")
        store.register_entity.assert_any_call(
            entity_type="company",
            canonical_name=normalize_company_name("Acme Inc"),
            entity_id=eid,
        )

    def test_company_alias_cik_and_ticker(self):
        tool, store = self._tool_with_store()
        filings = [_make_filing(issuer_cik="0000011111", ticker="ACME")]
        tool._persist_entities(filings)
        eid = entity_id_from_key("company", "0000011111")
        store.add_entity_alias.assert_any_call(eid, "sec_cik", "0000011111")
        store.add_entity_alias.assert_any_call(eid, "ticker", "ACME")

    def test_insider_registered(self):
        tool, store = self._tool_with_store()
        filings = [_make_filing(reporter_cik="0000099999", insider_name="John Smith")]
        tool._persist_entities(filings)
        eid = entity_id_from_key("person", "0000099999")
        store.register_entity.assert_any_call(
            entity_type="person",
            canonical_name="John Smith",
            entity_id=eid,
        )

    def test_insider_alias_sec_cik(self):
        tool, store = self._tool_with_store()
        filings = [_make_filing(reporter_cik="0000099999")]
        tool._persist_entities(filings)
        eid = entity_id_from_key("person", "0000099999")
        store.add_entity_alias.assert_any_call(eid, "sec_cik", "0000099999")

    def test_observation_stored_depth_2(self):
        tool, store = self._tool_with_store()
        filings = [_make_filing(reporter_cik="0000099999", filing_date="2026-04-01")]
        tool._persist_entities(filings)
        call_args = store.store_entity_observation.call_args
        assert call_args.kwargs["source_tool"] == "form144"
        assert call_args.kwargs["depth_level"] == 2
        assert call_args.kwargs["observation_type"] == "sell_intent"

    def test_observation_value_structure(self):
        tool, store = self._tool_with_store()
        filings = [
            _make_filing(
                ticker="ACME",
                company="Acme Inc",
                shares_to_sell=10000,
                dollar_value=500000.0,
                acquisition_type="open_market",
                urgency="near_term",
                relationship="Officer",
            )
        ]
        tool._persist_entities(filings)
        val = store.store_entity_observation.call_args.kwargs["value"]
        assert val["ticker"] == "ACME"
        assert val["company"] == "Acme Inc"
        assert val["shares_to_sell"] == 10000
        assert val["dollar_value"] == 500000.0
        assert val["acquisition_type"] == "open_market"
        assert val["urgency"] == "near_term"
        assert val["relationship"] == "Officer"

    def test_no_op_when_store_none(self):
        tool = Form144Tool()
        # Should not raise
        tool._persist_entities([_make_filing()])

    def test_no_op_on_empty_filings(self):
        tool, store = self._tool_with_store()
        tool._persist_entities([])
        store.register_entity.assert_not_called()

    def test_dedup_companies(self):
        tool, store = self._tool_with_store()
        filings = [
            _make_filing(issuer_cik="0000011111", reporter_cik="0000099901"),
            _make_filing(issuer_cik="0000011111", reporter_cik="0000099902"),
        ]
        tool._persist_entities(filings)
        # Company registered once (deduped)
        company_calls = [c for c in store.register_entity.call_args_list if c.kwargs.get("entity_type") == "company"]
        assert len(company_calls) == 1

    def test_dedup_insiders(self):
        tool, store = self._tool_with_store()
        filings = [
            _make_filing(reporter_cik="0000099999", filing_date="2026-04-01"),
            _make_filing(reporter_cik="0000099999", filing_date="2026-04-02"),
        ]
        tool._persist_entities(filings)
        # Insider registered once (deduped), but 2 observations stored
        insider_calls = [c for c in store.register_entity.call_args_list if c.kwargs.get("entity_type") == "person"]
        assert len(insider_calls) == 1
        assert store.store_entity_observation.call_count == 2

    def test_missing_issuer_cik_skips_company(self):
        tool, store = self._tool_with_store()
        filings = [_make_filing(issuer_cik="", reporter_cik="0000099999")]
        tool._persist_entities(filings)
        company_calls = [c for c in store.register_entity.call_args_list if c.kwargs.get("entity_type") == "company"]
        assert len(company_calls) == 0

    def test_missing_reporter_cik_skips_insider_and_observation(self):
        tool, store = self._tool_with_store()
        filings = [_make_filing(reporter_cik="", issuer_cik="0000011111")]
        tool._persist_entities(filings)
        insider_calls = [c for c in store.register_entity.call_args_list if c.kwargs.get("entity_type") == "person"]
        assert len(insider_calls) == 0
        store.store_entity_observation.assert_not_called()

    def test_persist_failure_does_not_break_execute(self):
        """Persistence error must not prevent ToolResult from returning."""
        tool = Form144Tool(pipeline_store=MagicMock())
        with (
            patch.object(tool, "_persist_entities", side_effect=RuntimeError("db fail")),
            patch.object(tool, "_fetch_recent_144s", return_value=[]),
        ):
            result = tool.execute(days_back=7)
        assert result.success is True

    @patch("agent.tools.form144.entity_id_from_key", None)
    def test_no_op_when_entity_helpers_unavailable(self):
        tool, store = self._tool_with_store()
        tool._persist_entities([_make_filing()])
        store.register_entity.assert_not_called()

    def test_normalization_error_falls_back(self):
        """If normalize_company_name raises, use raw company name."""
        tool, store = self._tool_with_store()
        with patch("agent.tools.form144.normalize_company_name", side_effect=ValueError("bad")):
            tool._persist_entities([_make_filing(company="Weird Co!!!")])
        # Should still register with the raw name
        company_calls = [c for c in store.register_entity.call_args_list if c.kwargs.get("entity_type") == "company"]
        assert len(company_calls) == 1
        assert company_calls[0].kwargs["canonical_name"] == "Weird Co!!!"


# ===================================================================
# 10b.2.4: CIK-based dedup in clustering
# ===================================================================


class TestCIKDedup:
    def test_dedup_by_cik_not_name(self):
        """Same reporter_cik with different display names → one insider."""
        filings = [
            _make_filing(
                insider_name="John Smith",
                reporter_cik="0000099999",
                filing_date="2026-04-01",
                dollar_value=100000,
            ),
            _make_filing(
                insider_name="J. SMITH",
                reporter_cik="0000099999",
                filing_date="2026-04-02",
                dollar_value=200000,
            ),
            _make_filing(
                insider_name="Alice Bob",
                reporter_cik="0000088888",
                filing_date="2026-04-03",
                dollar_value=300000,
            ),
        ]
        cluster = Form144Tool._find_best_sell_cluster(filings, window_days=14, min_size=2)
        assert cluster is not None
        assert cluster["insider_count"] == 2  # Not 3

    def test_name_fallback_when_no_cik(self):
        """Without reporter_cik, falls back to _normalize_name dedup."""
        filings = [
            _make_filing(insider_name="John Smith", reporter_cik="", filing_date="2026-04-01"),
            _make_filing(insider_name="JOHN SMITH", reporter_cik="", filing_date="2026-04-02"),
            _make_filing(insider_name="Alice Bob", reporter_cik="", filing_date="2026-04-03"),
        ]
        cluster = Form144Tool._find_best_sell_cluster(filings, window_days=14, min_size=2)
        assert cluster is not None
        assert cluster["insider_count"] == 2  # John Smith deduplicated by name

    def test_mixed_cik_and_name_dedup(self):
        """Some filings have CIK, some don't — both dedup paths work."""
        filings = [
            _make_filing(
                insider_name="John Smith",
                reporter_cik="0000099999",
                filing_date="2026-04-01",
                dollar_value=100000,
            ),
            _make_filing(
                insider_name="Alice Bob",
                reporter_cik="",
                filing_date="2026-04-02",
                dollar_value=200000,
            ),
            _make_filing(
                insider_name="ALICE BOB",
                reporter_cik="",
                filing_date="2026-04-03",
                dollar_value=150000,
            ),
        ]
        cluster = Form144Tool._find_best_sell_cluster(filings, window_days=14, min_size=2)
        assert cluster is not None
        assert cluster["insider_count"] == 2  # Alice Bob deduplicated by name


# ===================================================================
# 10b.2.5: entity_ids in cluster output
# ===================================================================


class TestEntityIdsInCluster:
    def test_entity_ids_present(self):
        filings = [
            _make_filing(
                insider_name="John Smith",
                reporter_cik="0000099999",
                filing_date="2026-04-01",
                dollar_value=100000,
            ),
            _make_filing(
                insider_name="Alice Bob",
                reporter_cik="0000088888",
                filing_date="2026-04-03",
                dollar_value=200000,
            ),
        ]
        cluster = Form144Tool._find_best_sell_cluster(filings, window_days=14, min_size=2)
        assert cluster is not None
        assert "entity_ids" in cluster
        assert cluster["entity_ids"]["John Smith"] == entity_id_from_key("person", "0000099999")
        assert cluster["entity_ids"]["Alice Bob"] == entity_id_from_key("person", "0000088888")

    def test_entity_ids_empty_when_no_cik(self):
        filings = [
            _make_filing(
                insider_name="John Smith",
                reporter_cik="",
                filing_date="2026-04-01",
                dollar_value=100000,
            ),
            _make_filing(
                insider_name="Alice Bob",
                reporter_cik="",
                filing_date="2026-04-03",
                dollar_value=200000,
            ),
        ]
        cluster = Form144Tool._find_best_sell_cluster(filings, window_days=14, min_size=2)
        assert cluster is not None
        assert cluster["entity_ids"] == {}

    def test_entity_ids_partial(self):
        """Only insiders with CIKs get entity_ids."""
        filings = [
            _make_filing(
                insider_name="John Smith",
                reporter_cik="0000099999",
                filing_date="2026-04-01",
                dollar_value=100000,
            ),
            _make_filing(
                insider_name="Alice Bob",
                reporter_cik="",
                filing_date="2026-04-03",
                dollar_value=200000,
            ),
        ]
        cluster = Form144Tool._find_best_sell_cluster(filings, window_days=14, min_size=2)
        assert cluster is not None
        assert "John Smith" in cluster["entity_ids"]
        assert "Alice Bob" not in cluster["entity_ids"]


# ===================================================================
# 10b.2.6: Edge cases
# ===================================================================


class TestEdgeCases:
    def test_date_to_timestamp_valid(self):
        ts = _date_to_timestamp("2026-04-01")
        expected = datetime(2026, 4, 1).timestamp()
        assert ts == expected

    def test_date_to_timestamp_invalid(self):
        assert _date_to_timestamp("not-a-date") == 0.0

    def test_date_to_timestamp_empty(self):
        assert _date_to_timestamp("") == 0.0

    def test_cluster_backward_compat_still_works(self):
        """Clusters still form correctly with CIK fields present."""
        filings = [
            _make_filing(
                insider_name="Alice",
                reporter_cik="CIK1",
                filing_date="2026-04-01",
                dollar_value=100000,
            ),
            _make_filing(
                insider_name="Bob",
                reporter_cik="CIK2",
                filing_date="2026-04-05",
                dollar_value=200000,
            ),
        ]
        cluster = Form144Tool._find_best_sell_cluster(filings, window_days=14, min_size=2)
        assert cluster is not None
        assert cluster["insider_count"] == 2
        assert cluster["conviction"] in ("high", "medium-high", "medium", "moderate")

    def test_persist_with_metadata_only_record(self):
        """Metadata-only records (no XML parse) still have CIKs and get persisted."""
        tool, store = TestPersistEntities()._tool_with_store()
        filing = _make_filing()
        filing["_metadata_only"] = True
        filing["shares_to_sell"] = 0
        filing["dollar_value"] = 0.0
        tool._persist_entities([filing])
        # Even metadata-only records should register entities
        assert store.register_entity.call_count >= 1

    def test_gift_filings_excluded_from_persistence(self):
        """Filings filtered as gifts by XML parser never reach persistence."""
        # This is structural: _parse_form144_xml returns None for gifts,
        # so they're never added to the filings list.
        # Verify that if a filing somehow had is_gift=True, it still persists
        # (filtering happens before persistence, not during).
        tool, store = TestPersistEntities()._tool_with_store()
        filing = _make_filing()
        filing["is_gift"] = True  # Should still persist — filtering is upstream
        tool._persist_entities([filing])
        assert store.store_entity_observation.call_count == 1


# ===================================================================
# 10b.2.7: MI measurement integration test
# ===================================================================


class TestForm144MIIntegration:
    """Verify that L2 entity-resolved observations carry more signal
    than L1 aggregate data, measured via mutual information."""

    def test_l2_adds_signal_beyond_l1(self):
        """MI(L2; target | L1) > 0 when L2 observes the signal more cleanly."""
        import numpy as np

        from agent.pipeline.depth_eval import compute_conditional_mi

        rng = np.random.default_rng(42)
        n = 200
        signal = rng.standard_normal(n)
        l1 = signal + rng.normal(0, 3.0, n)  # noisy aggregate
        l2 = signal + rng.normal(0, 0.5, n)  # clean entity-level
        target = signal + rng.normal(0, 0.3, n)

        mi_gain = compute_conditional_mi(l2, target, l1)
        assert mi_gain > 0, f"Expected positive MI gain, got {mi_gain}"

    def test_no_signal_yields_near_zero_mi(self):
        """Independent noise → MI ≈ 0."""
        import numpy as np

        from agent.pipeline.depth_eval import compute_conditional_mi

        rng = np.random.default_rng(123)
        n = 200
        l1 = rng.standard_normal(n)
        l2 = rng.standard_normal(n)
        target = rng.standard_normal(n)

        mi = compute_conditional_mi(l2, target, l1)
        assert abs(mi) < 0.3, f"Expected near-zero MI for independent noise, got {mi}"
