"""Tests for Phase 17 — Entity Linking Layer.

17a: works_for links in insider_filings + form144.
Later phases (17b–17e) will extend this file.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent.pipeline.entity import entity_id_from_key
from agent.pipeline.store import PipelineStore
from agent.tools.insider_filings import InsiderFilingsTool
from agent.tools.form144 import Form144Tool
from agent.tools.whale_alert import WhaleAlertTool
from agent.tools.gdelt import GDELTTool
from agent.tools.ais_vessel import AISVesselTool
from agent.tools.lobbying import LobbyingTool
from agent.tools.interconnection_queue import InterconnectionQueueTool
from agent.tools.patent_filings import PatentFilingsTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_filing(
    *,
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


def _make_tool_with_store(tool_cls):
    """Create a tool backed by an in-memory PipelineStore."""
    store = PipelineStore(db_path=":memory:")
    tool = tool_cls(pipeline_store=store)
    return tool, store


# ═══════════════════════════════════════════════════════════════════════
# 17a — works_for links (insider_filings + form144)
# ═══════════════════════════════════════════════════════════════════════


class TestInsiderFilingsWorksForLink:
    """insider_filings creates a works_for link from person → company."""

    def test_normal_link_created(self):
        """A transaction with both CIKs creates a works_for link."""
        tool, store = _make_tool_with_store(InsiderFilingsTool)
        txns = [_make_txn()]
        tool._persist_entities(txns)

        insider_eid = entity_id_from_key("person", "0001214156")
        company_eid = entity_id_from_key("company", "0000320193")

        links = store.query_entity_links(insider_eid, link_type="works_for")
        assert len(links) == 1
        link = links[0]
        assert link["entity_id_a"] == insider_eid
        assert link["entity_id_b"] == company_eid
        assert link["link_type"] == "works_for"
        assert link["source"] == "insider_filings"
        assert link["confidence"] == 1.0

    def test_link_metadata_contains_role(self):
        """The works_for link metadata includes the insider's role."""
        tool, store = _make_tool_with_store(InsiderFilingsTool)
        txns = [_make_txn(role="Chief Executive Officer")]
        tool._persist_entities(txns)

        insider_eid = entity_id_from_key("person", "0001214156")
        links = store.query_entity_links(insider_eid, link_type="works_for")
        assert len(links) == 1
        assert links[0]["metadata"]["relationship"] == "Chief Executive Officer"

    def test_link_dedup_same_person_company(self):
        """Multiple transactions for the same person→company produce one link."""
        tool, store = _make_tool_with_store(InsiderFilingsTool)
        txns = [
            _make_txn(date="2026-03-10"),
            _make_txn(date="2026-03-15"),
            _make_txn(date="2026-03-20"),
        ]
        tool._persist_entities(txns)

        insider_eid = entity_id_from_key("person", "0001214156")
        links = store.query_entity_links(insider_eid, link_type="works_for")
        assert len(links) == 1  # idempotent INSERT OR IGNORE

    def test_missing_reporter_cik_no_link(self):
        """No link created when reporter_cik is empty."""
        tool, store = _make_tool_with_store(InsiderFilingsTool)
        txns = [_make_txn(reporter_cik="")]
        tool._persist_entities(txns)

        company_eid = entity_id_from_key("company", "0000320193")
        links = store.query_entity_links(company_eid, link_type="works_for")
        assert len(links) == 0

    def test_missing_issuer_cik_no_link(self):
        """No link created when issuer_cik is empty."""
        tool, store = _make_tool_with_store(InsiderFilingsTool)
        txns = [_make_txn(issuer_cik="")]
        tool._persist_entities(txns)

        insider_eid = entity_id_from_key("person", "0001214156")
        links = store.query_entity_links(insider_eid, link_type="works_for")
        assert len(links) == 0

    def test_both_ciks_missing_no_link(self):
        """No link or entity crash when both CIKs are missing."""
        tool, store = _make_tool_with_store(InsiderFilingsTool)
        txns = [_make_txn(issuer_cik="", reporter_cik="")]
        tool._persist_entities(txns)
        # No entities, no links
        all_links = store.query_all_entity_links()
        assert len(all_links) == 0

    def test_multiple_insiders_same_company(self):
        """Multiple insiders at one company → separate works_for links."""
        tool, store = _make_tool_with_store(InsiderFilingsTool)
        txns = [
            _make_txn(reporter_cik="0001111111", name="Alice"),
            _make_txn(reporter_cik="0002222222", name="Bob"),
        ]
        tool._persist_entities(txns)

        company_eid = entity_id_from_key("company", "0000320193")
        links = store.query_entity_links(
            company_eid, link_type="works_for", direction="incoming"
        )
        assert len(links) == 2
        link_sources = {link["entity_id_a"] for link in links}
        assert entity_id_from_key("person", "0001111111") in link_sources
        assert entity_id_from_key("person", "0002222222") in link_sources

    def test_one_insider_multiple_companies(self):
        """One insider filing at multiple companies → separate links."""
        tool, store = _make_tool_with_store(InsiderFilingsTool)
        txns = [
            _make_txn(issuer_cik="0000320193", company="Apple Inc."),
            _make_txn(issuer_cik="0000789019", company="Microsoft Corp"),
        ]
        tool._persist_entities(txns)

        insider_eid = entity_id_from_key("person", "0001214156")
        links = store.query_entity_links(
            insider_eid, link_type="works_for", direction="outgoing"
        )
        assert len(links) == 2

    def test_empty_role_in_metadata(self):
        """Empty role still produces valid metadata."""
        tool, store = _make_tool_with_store(InsiderFilingsTool)
        txns = [_make_txn(role="")]
        tool._persist_entities(txns)

        insider_eid = entity_id_from_key("person", "0001214156")
        links = store.query_entity_links(insider_eid, link_type="works_for")
        assert len(links) == 1
        assert links[0]["metadata"]["relationship"] == ""

    def test_link_direction_outgoing_from_person(self):
        """works_for link is outgoing from person (entity_id_a = person)."""
        tool, store = _make_tool_with_store(InsiderFilingsTool)
        txns = [_make_txn()]
        tool._persist_entities(txns)

        insider_eid = entity_id_from_key("person", "0001214156")
        out = store.query_entity_links(
            insider_eid, link_type="works_for", direction="outgoing"
        )
        assert len(out) == 1
        inc = store.query_entity_links(
            insider_eid, link_type="works_for", direction="incoming"
        )
        assert len(inc) == 0


class TestForm144WorksForLink:
    """form144 creates a works_for link from person → company."""

    def test_normal_link_created(self):
        """A filing with both CIKs creates a works_for link."""
        tool, store = _make_tool_with_store(Form144Tool)
        filings = [_make_filing()]
        tool._persist_entities(filings)

        insider_eid = entity_id_from_key("person", "0000099999")
        company_eid = entity_id_from_key("company", "0000012345")

        links = store.query_entity_links(insider_eid, link_type="works_for")
        assert len(links) == 1
        link = links[0]
        assert link["entity_id_a"] == insider_eid
        assert link["entity_id_b"] == company_eid
        assert link["link_type"] == "works_for"
        assert link["source"] == "form144"
        assert link["confidence"] == 1.0

    def test_link_metadata_contains_relationship(self):
        """The works_for link metadata includes the relationship field."""
        tool, store = _make_tool_with_store(Form144Tool)
        filings = [_make_filing(relationship="Director")]
        tool._persist_entities(filings)

        insider_eid = entity_id_from_key("person", "0000099999")
        links = store.query_entity_links(insider_eid, link_type="works_for")
        assert links[0]["metadata"]["relationship"] == "Director"

    def test_link_dedup_same_person_company(self):
        """Multiple filings for the same person→company produce one link."""
        tool, store = _make_tool_with_store(Form144Tool)
        filings = [
            _make_filing(filing_date="2026-03-01"),
            _make_filing(filing_date="2026-04-01"),
        ]
        tool._persist_entities(filings)

        insider_eid = entity_id_from_key("person", "0000099999")
        links = store.query_entity_links(insider_eid, link_type="works_for")
        assert len(links) == 1

    def test_missing_reporter_cik_no_link(self):
        """No link created when reporter_cik is empty."""
        tool, store = _make_tool_with_store(Form144Tool)
        filings = [_make_filing(reporter_cik="")]
        tool._persist_entities(filings)

        company_eid = entity_id_from_key("company", "0000012345")
        links = store.query_entity_links(company_eid, link_type="works_for")
        assert len(links) == 0

    def test_missing_issuer_cik_no_link(self):
        """No link created when issuer_cik is empty."""
        tool, store = _make_tool_with_store(Form144Tool)
        filings = [_make_filing(issuer_cik="")]
        tool._persist_entities(filings)

        insider_eid = entity_id_from_key("person", "0000099999")
        links = store.query_entity_links(insider_eid, link_type="works_for")
        assert len(links) == 0

    def test_both_ciks_missing_no_link(self):
        """No link or crash when both CIKs are missing."""
        tool, store = _make_tool_with_store(Form144Tool)
        filings = [_make_filing(issuer_cik="", reporter_cik="")]
        tool._persist_entities(filings)
        all_links = store.query_all_entity_links()
        assert len(all_links) == 0

    def test_multiple_insiders_same_company(self):
        """Multiple insiders at one company → separate works_for links."""
        tool, store = _make_tool_with_store(Form144Tool)
        filings = [
            _make_filing(reporter_cik="0000011111", insider_name="Alice"),
            _make_filing(reporter_cik="0000022222", insider_name="Bob"),
        ]
        tool._persist_entities(filings)

        company_eid = entity_id_from_key("company", "0000012345")
        links = store.query_entity_links(
            company_eid, link_type="works_for", direction="incoming"
        )
        assert len(links) == 2

    def test_one_insider_multiple_companies(self):
        """One insider filing at multiple companies → separate links."""
        tool, store = _make_tool_with_store(Form144Tool)
        filings = [
            _make_filing(issuer_cik="0000012345", company="Acme Inc"),
            _make_filing(issuer_cik="0000067890", company="Beta Corp"),
        ]
        tool._persist_entities(filings)

        insider_eid = entity_id_from_key("person", "0000099999")
        links = store.query_entity_links(
            insider_eid, link_type="works_for", direction="outgoing"
        )
        assert len(links) == 2

    def test_empty_relationship_in_metadata(self):
        """Empty relationship still produces valid metadata."""
        tool, store = _make_tool_with_store(Form144Tool)
        filings = [_make_filing(relationship="")]
        tool._persist_entities(filings)

        insider_eid = entity_id_from_key("person", "0000099999")
        links = store.query_entity_links(insider_eid, link_type="works_for")
        assert len(links) == 1
        assert links[0]["metadata"]["relationship"] == ""


class TestCrossToolEntityConsistency:
    """Entity IDs match when the same CIK appears in both tools."""

    def test_same_company_cik_same_entity_id(self):
        """insider_filings and form144 produce the same company entity_id for the same CIK."""
        insider_tool, insider_store = _make_tool_with_store(InsiderFilingsTool)
        form144_tool, form144_store = _make_tool_with_store(Form144Tool)

        # Same issuer CIK
        cik = "0000320193"
        insider_tool._persist_entities([_make_txn(issuer_cik=cik)])
        form144_tool._persist_entities([_make_filing(issuer_cik=cik)])

        # Both produce the same entity_id
        assert entity_id_from_key("company", cik) == entity_id_from_key("company", cik)
        # And specifically the links point to the same target
        insider_eid_if = entity_id_from_key("person", "0001214156")
        insider_eid_f144 = entity_id_from_key("person", "0000099999")

        links_if = insider_store.query_entity_links(
            insider_eid_if, link_type="works_for"
        )
        links_f144 = form144_store.query_entity_links(
            insider_eid_f144, link_type="works_for"
        )
        assert links_if[0]["entity_id_b"] == links_f144[0]["entity_id_b"]

    def test_same_person_cik_same_entity_id(self):
        """Same reporter CIK in both tools → same person entity_id."""
        cik = "0001111111"
        eid = entity_id_from_key("person", cik)

        insider_tool, insider_store = _make_tool_with_store(InsiderFilingsTool)
        insider_tool._persist_entities([_make_txn(reporter_cik=cik)])
        links = insider_store.query_entity_links(eid, link_type="works_for")
        assert links[0]["entity_id_a"] == eid

        form144_tool, form144_store = _make_tool_with_store(Form144Tool)
        form144_tool._persist_entities([_make_filing(reporter_cik=cik)])
        links = form144_store.query_entity_links(eid, link_type="works_for")
        assert links[0]["entity_id_a"] == eid

    def test_shared_store_both_tools_single_link(self):
        """When both tools share one store, same CIK pair → one link (dedup)."""
        store = PipelineStore(db_path=":memory:")
        insider_tool = InsiderFilingsTool(pipeline_store=store)
        form144_tool = Form144Tool(pipeline_store=store)

        shared_issuer = "0000320193"
        shared_reporter = "0001214156"

        insider_tool._persist_entities(
            [_make_txn(issuer_cik=shared_issuer, reporter_cik=shared_reporter)]
        )
        form144_tool._persist_entities(
            [_make_filing(issuer_cik=shared_issuer, reporter_cik=shared_reporter)]
        )

        person_eid = entity_id_from_key("person", shared_reporter)
        links = store.query_entity_links(person_eid, link_type="works_for")
        # INSERT OR IGNORE — both tools write same (a, b, link_type) → 1 row
        assert len(links) == 1


class TestNoLinkWithoutPipeline:
    """Tools with no pipeline_store don't crash on the link call."""

    def test_insider_filings_no_store(self):
        """InsiderFilingsTool without a pipeline_store doesn't crash."""
        tool = InsiderFilingsTool()
        tool._persist_entities([_make_txn()])  # no-op, should not raise

    def test_form144_no_store(self):
        """Form144Tool without a pipeline_store doesn't crash."""
        tool = Form144Tool()
        tool._persist_entities([_make_filing()])  # no-op, should not raise


# ═══════════════════════════════════════════════════════════════════════
# Helpers for 17b
# ═══════════════════════════════════════════════════════════════════════


def _make_whale_tx(
    *,
    tx_hash: str = "abc123",
    time: int = 1712700000,
    confirmed: bool = True,
    block_height: int = 800000,
    inputs: list[dict] | None = None,
    outputs: list[dict] | None = None,
) -> dict[str, Any]:
    if inputs is None:
        inputs = [{"addr": "1SenderAddr", "value_btc": 5.0}]
    if outputs is None:
        outputs = [{"addr": "1ReceiverAddr", "value_btc": 5.0}]
    return {
        "hash": tx_hash,
        "time": time,
        "confirmed": confirmed,
        "block_height": block_height,
        "inputs": inputs,
        "outputs": outputs,
    }


def _make_gdelt_event(
    *,
    event_id: str = "EVT001",
    date: str = "2026-04-01",
    actor1_country: str = "US",
    actor1_name: str = "United States",
    actor2_country: str = "CN",
    actor2_name: str = "China",
    event_root: str = "04",
    event_description: str = "Consult",
    goldstein: float = 1.0,
    quad_class: int = 1,
    num_mentions: int = 5,
) -> dict[str, Any]:
    return {
        "id": event_id,
        "date": date,
        "actor1": {"country": actor1_country, "name": actor1_name, "type": "GOV"},
        "actor2": {"country": actor2_country, "name": actor2_name, "type": "GOV"},
        "event_root": event_root,
        "event_description": event_description,
        "goldstein": goldstein,
        "quad_class": quad_class,
        "num_mentions": num_mentions,
        "location": {"country": "US"},
    }


# ═══════════════════════════════════════════════════════════════════════
# 17b.1 — transacts_with links (whale_alert)
# ═══════════════════════════════════════════════════════════════════════


class TestWhaleAlertTransactsWith:
    """whale_alert creates transacts_with links from sender → receiver wallets."""

    def test_normal_link_created(self):
        """Single sender → single receiver creates one transacts_with link."""
        tool, store = _make_tool_with_store(WhaleAlertTool)
        tool._persist_entities([_make_whale_tx()])

        sender_eid = entity_id_from_key("wallet", "1SenderAddr")
        receiver_eid = entity_id_from_key("wallet", "1ReceiverAddr")

        links = store.query_entity_links(
            sender_eid, link_type="transacts_with", direction="outgoing"
        )
        assert len(links) == 1
        assert links[0]["entity_id_a"] == sender_eid
        assert links[0]["entity_id_b"] == receiver_eid
        assert links[0]["source"] == "whale_alert"
        assert links[0]["confidence"] == 1.0

    def test_link_metadata_contains_tx_hash(self):
        """The transacts_with link metadata includes the tx_hash."""
        tool, store = _make_tool_with_store(WhaleAlertTool)
        tool._persist_entities([_make_whale_tx(tx_hash="deadbeef")])

        sender_eid = entity_id_from_key("wallet", "1SenderAddr")
        links = store.query_entity_links(sender_eid, link_type="transacts_with")
        assert links[0]["metadata"]["tx_hash"] == "deadbeef"

    def test_multiple_senders_multiple_receivers(self):
        """N senders × M receivers → N*M links (minus self-links)."""
        tool, store = _make_tool_with_store(WhaleAlertTool)
        tx = _make_whale_tx(
            inputs=[
                {"addr": "sA", "value_btc": 3.0},
                {"addr": "sB", "value_btc": 2.0},
            ],
            outputs=[
                {"addr": "rX", "value_btc": 4.0},
                {"addr": "rY", "value_btc": 1.0},
            ],
        )
        tool._persist_entities([tx])

        # 2 senders × 2 receivers = 4 links
        all_links = store.query_all_entity_links()
        transact_links = [l for l in all_links if l["link_type"] == "transacts_with"]
        assert len(transact_links) == 4

    def test_self_send_skipped(self):
        """Same address in inputs and outputs → no self-link."""
        tool, store = _make_tool_with_store(WhaleAlertTool)
        tx = _make_whale_tx(
            inputs=[{"addr": "1Same", "value_btc": 5.0}],
            outputs=[{"addr": "1Same", "value_btc": 5.0}],
        )
        tool._persist_entities([tx])

        eid = entity_id_from_key("wallet", "1Same")
        links = store.query_entity_links(eid, link_type="transacts_with")
        assert len(links) == 0

    def test_empty_sender_addr_skipped(self):
        """Empty sender address produces no link."""
        tool, store = _make_tool_with_store(WhaleAlertTool)
        tx = _make_whale_tx(
            inputs=[{"addr": "", "value_btc": 5.0}],
            outputs=[{"addr": "1ReceiverAddr", "value_btc": 5.0}],
        )
        tool._persist_entities([tx])

        receiver_eid = entity_id_from_key("wallet", "1ReceiverAddr")
        links = store.query_entity_links(receiver_eid, link_type="transacts_with")
        assert len(links) == 0

    def test_empty_receiver_addr_skipped(self):
        """Empty receiver address produces no link."""
        tool, store = _make_tool_with_store(WhaleAlertTool)
        tx = _make_whale_tx(
            inputs=[{"addr": "1SenderAddr", "value_btc": 5.0}],
            outputs=[{"addr": "", "value_btc": 5.0}],
        )
        tool._persist_entities([tx])

        sender_eid = entity_id_from_key("wallet", "1SenderAddr")
        links = store.query_entity_links(
            sender_eid, link_type="transacts_with", direction="outgoing"
        )
        assert len(links) == 0

    def test_dedup_same_sender_receiver_pair(self):
        """Two transactions between same wallets → one link (INSERT OR IGNORE)."""
        tool, store = _make_tool_with_store(WhaleAlertTool)
        tool._persist_entities(
            [
                _make_whale_tx(tx_hash="tx1"),
                _make_whale_tx(tx_hash="tx2"),
            ]
        )

        sender_eid = entity_id_from_key("wallet", "1SenderAddr")
        links = store.query_entity_links(sender_eid, link_type="transacts_with")
        assert len(links) == 1

    def test_no_inputs_no_links(self):
        """Transaction with no inputs → no links."""
        tool, store = _make_tool_with_store(WhaleAlertTool)
        tx = _make_whale_tx(inputs=[], outputs=[{"addr": "1Recv", "value_btc": 5.0}])
        tool._persist_entities([tx])

        all_links = store.query_all_entity_links()
        assert len(all_links) == 0

    def test_no_outputs_no_links(self):
        """Transaction with no outputs → no links."""
        tool, store = _make_tool_with_store(WhaleAlertTool)
        tx = _make_whale_tx(inputs=[{"addr": "1Send", "value_btc": 5.0}], outputs=[])
        tool._persist_entities([tx])

        all_links = store.query_all_entity_links()
        assert len(all_links) == 0

    def test_no_store_no_crash(self):
        """WhaleAlertTool without a pipeline_store doesn't crash."""
        tool = WhaleAlertTool()
        tool._persist_entities([_make_whale_tx()])  # no-op


# ═══════════════════════════════════════════════════════════════════════
# 17b.2 — event_involves links (gdelt)
# ═══════════════════════════════════════════════════════════════════════


class TestGDELTEventInvolves:
    """gdelt creates event_involves links between bilateral country pairs."""

    def test_normal_link_created(self):
        """An event with two different countries creates an event_involves link."""
        tool, store = _make_tool_with_store(GDELTTool)
        tool._persist_entities([_make_gdelt_event()])

        us_eid = entity_id_from_key("country", "US")
        cn_eid = entity_id_from_key("country", "CN")

        links = store.query_entity_links(us_eid, link_type="event_involves")
        assert len(links) == 1
        assert links[0]["entity_id_a"] == us_eid
        assert links[0]["entity_id_b"] == cn_eid
        assert links[0]["source"] == "gdelt"
        assert links[0]["confidence"] == 0.9

    def test_link_metadata_contains_event_info(self):
        """The event_involves link metadata includes event_id and event_root."""
        tool, store = _make_tool_with_store(GDELTTool)
        tool._persist_entities([_make_gdelt_event(event_id="EVT123", event_root="14")])

        us_eid = entity_id_from_key("country", "US")
        links = store.query_entity_links(us_eid, link_type="event_involves")
        assert links[0]["metadata"]["event_id"] == "EVT123"
        assert links[0]["metadata"]["event_root"] == "14"

    def test_same_country_no_link(self):
        """Event where actor1 and actor2 are the same country → no link."""
        tool, store = _make_tool_with_store(GDELTTool)
        ev = _make_gdelt_event(actor1_country="US", actor2_country="US")
        tool._persist_entities([ev])

        us_eid = entity_id_from_key("country", "US")
        links = store.query_entity_links(us_eid, link_type="event_involves")
        assert len(links) == 0

    def test_missing_actor1_country_no_link(self):
        """Event with empty actor1 country → no link."""
        tool, store = _make_tool_with_store(GDELTTool)
        ev = _make_gdelt_event(actor1_country="", actor2_country="CN")
        tool._persist_entities([ev])

        cn_eid = entity_id_from_key("country", "CN")
        links = store.query_entity_links(cn_eid, link_type="event_involves")
        assert len(links) == 0

    def test_missing_actor2_country_no_link(self):
        """Event with empty actor2 country → no link."""
        tool, store = _make_tool_with_store(GDELTTool)
        ev = _make_gdelt_event(actor1_country="US", actor2_country="")
        tool._persist_entities([ev])

        us_eid = entity_id_from_key("country", "US")
        links = store.query_entity_links(us_eid, link_type="event_involves")
        assert len(links) == 0

    def test_both_countries_missing_no_link(self):
        """Event with both countries empty → no link, no crash."""
        tool, store = _make_tool_with_store(GDELTTool)
        ev = _make_gdelt_event(actor1_country="", actor2_country="")
        tool._persist_entities([ev])

        all_links = store.query_all_entity_links()
        assert len(all_links) == 0

    def test_dedup_same_country_pair(self):
        """Multiple events between same countries → one link (INSERT OR IGNORE)."""
        tool, store = _make_tool_with_store(GDELTTool)
        tool._persist_entities(
            [
                _make_gdelt_event(event_id="EVT001"),
                _make_gdelt_event(event_id="EVT002"),
            ]
        )

        us_eid = entity_id_from_key("country", "US")
        links = store.query_entity_links(us_eid, link_type="event_involves")
        assert len(links) == 1

    def test_multiple_country_pairs(self):
        """Events between different country pairs → separate links."""
        tool, store = _make_tool_with_store(GDELTTool)
        tool._persist_entities(
            [
                _make_gdelt_event(actor1_country="US", actor2_country="CN"),
                _make_gdelt_event(
                    actor1_country="US", actor2_country="RU", event_id="E2"
                ),
                _make_gdelt_event(
                    actor1_country="FR", actor2_country="DE", event_id="E3"
                ),
            ]
        )

        us_eid = entity_id_from_key("country", "US")
        links = store.query_entity_links(
            us_eid, link_type="event_involves", direction="outgoing"
        )
        assert len(links) == 2

        fr_eid = entity_id_from_key("country", "FR")
        links_fr = store.query_entity_links(
            fr_eid, link_type="event_involves", direction="outgoing"
        )
        assert len(links_fr) == 1

    def test_whitespace_country_stripped(self):
        """Country codes with whitespace are stripped before comparison."""
        tool, store = _make_tool_with_store(GDELTTool)
        ev = _make_gdelt_event(actor1_country=" US ", actor2_country=" US ")
        tool._persist_entities([ev])

        # Same country after strip → no link
        all_links = store.query_all_entity_links()
        event_links = [l for l in all_links if l["link_type"] == "event_involves"]
        assert len(event_links) == 0

    def test_no_store_no_crash(self):
        """GDELTTool without a pipeline_store doesn't crash."""
        tool = GDELTTool()
        tool._persist_entities([_make_gdelt_event()])  # no-op


# ═══════════════════════════════════════════════════════════════════════
# Helpers for 17c
# ═══════════════════════════════════════════════════════════════════════


def _make_vessel(
    *,
    mmsi: int = 230999000,
    imo: int | None = 9999999,
    name: str = "NORDIC STAR",
    destination: str = "ROTTERDAM",
    lat: float = 60.1,
    lon: float = 24.9,
    sog: float = 12.0,
    cog: float = 180.0,
    ship_type: str = "Tanker",
    timestamp: str = "2026-04-01T12:00:00Z",
) -> dict[str, Any]:
    return {
        "mmsi": mmsi,
        "imo": imo,
        "name": name,
        "destination": destination,
        "lat": lat,
        "lon": lon,
        "sog": sog,
        "cog": cog,
        "ship_type": ship_type,
        "timestamp": timestamp,
    }


def _make_port_call(
    *,
    mmsi: int = 230999000,
    imoLloyds: int | None = 9999999,
    vesselName: str = "NORDIC STAR",
    portToVisit: str = "HELSINKI",
    prevPort: str = "STOCKHOLM",
    nextPort: str = "TALLINN",
    eta: str = "2026-04-02T08:00:00Z",
) -> dict[str, Any]:
    return {
        "mmsi": mmsi,
        "imoLloyds": imoLloyds,
        "vesselName": vesselName,
        "portToVisit": portToVisit,
        "prevPort": prevPort,
        "nextPort": nextPort,
        "eta": eta,
    }


def _make_lobby_filing(
    *,
    registrant_name: str = "Akin Gump",
    registrant_id: str = "300429",
    client_name: str = "Amazon.com",
    amount: int = 5000000,
    filing_year: int = 2026,
    filing_period: str = "Q1",
    dt_posted: str = "2026-04-01T00:00:00Z",
    issue_codes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "registrant_name": registrant_name,
        "registrant_id": registrant_id,
        "client_name": client_name,
        "amount": amount,
        "filing_year": filing_year,
        "filing_period": filing_period,
        "dt_posted": dt_posted,
        "issue_codes": issue_codes or ["TAX"],
    }


# ═══════════════════════════════════════════════════════════════════════
# 17c.1 — port_call_to links (ais_vessel)
# ═══════════════════════════════════════════════════════════════════════


class TestAISVesselPortCallTo:
    """ais_vessel creates port_call_to links from vessel → country."""

    def test_normal_link_from_destination(self):
        """Vessel with known destination creates a port_call_to link."""
        tool, store = _make_tool_with_store(AISVesselTool)
        tool._persist_entities([_make_vessel(destination="ROTTERDAM")])

        vessel_eid = entity_id_from_key("vessel", "9999999")
        nl_eid = entity_id_from_key("country", "NL")

        links = store.query_entity_links(
            vessel_eid, link_type="port_call_to", direction="outgoing"
        )
        assert len(links) == 1
        assert links[0]["entity_id_b"] == nl_eid
        assert links[0]["source"] == "ais_vessel"
        assert links[0]["confidence"] == 0.8

    def test_link_metadata_has_raw_destination(self):
        """Metadata contains the original destination string."""
        tool, store = _make_tool_with_store(AISVesselTool)
        tool._persist_entities([_make_vessel(destination="HELSINKI")])

        vessel_eid = entity_id_from_key("vessel", "9999999")
        links = store.query_entity_links(vessel_eid, link_type="port_call_to")
        assert links[0]["metadata"]["destination_raw"] == "HELSINKI"

    def test_country_entity_created(self):
        """The destination country entity is registered."""
        tool, store = _make_tool_with_store(AISVesselTool)
        tool._persist_entities([_make_vessel(destination="SINGAPORE")])

        sg_eid = entity_id_from_key("country", "SG")
        entity = store.get_entity(sg_eid)
        assert entity is not None
        assert entity["entity_type"] == "country"

    def test_unknown_destination_no_link(self):
        """An unrecognized destination produces no link."""
        tool, store = _make_tool_with_store(AISVesselTool)
        tool._persist_entities([_make_vessel(destination="UNKNOWN PORT 42")])

        vessel_eid = entity_id_from_key("vessel", "9999999")
        links = store.query_entity_links(vessel_eid, link_type="port_call_to")
        assert len(links) == 0

    def test_empty_destination_no_link(self):
        """Empty destination produces no link."""
        tool, store = _make_tool_with_store(AISVesselTool)
        tool._persist_entities([_make_vessel(destination="")])

        vessel_eid = entity_id_from_key("vessel", "9999999")
        links = store.query_entity_links(vessel_eid, link_type="port_call_to")
        assert len(links) == 0

    def test_case_insensitive_destination(self):
        """Destination matching is case-insensitive (uppercased)."""
        tool, store = _make_tool_with_store(AISVesselTool)
        tool._persist_entities([_make_vessel(destination="rotterdam")])

        vessel_eid = entity_id_from_key("vessel", "9999999")
        links = store.query_entity_links(vessel_eid, link_type="port_call_to")
        assert len(links) == 1

    def test_dedup_same_vessel_same_destination(self):
        """Same vessel→country pair from dedup → one link."""
        tool, store = _make_tool_with_store(AISVesselTool)
        # Same vessel, same destination — eid dedup prevents second persist
        tool._persist_entities(
            [
                _make_vessel(destination="ROTTERDAM"),
                _make_vessel(destination="ROTTERDAM"),
            ]
        )

        vessel_eid = entity_id_from_key("vessel", "9999999")
        links = store.query_entity_links(vessel_eid, link_type="port_call_to")
        assert len(links) == 1

    def test_port_call_mode_creates_link(self):
        """Port call entities also create port_call_to links."""
        tool, store = _make_tool_with_store(AISVesselTool)
        tool._persist_port_call_entities([_make_port_call(portToVisit="HELSINKI")])

        vessel_eid = entity_id_from_key("vessel", "9999999")
        links = store.query_entity_links(vessel_eid, link_type="port_call_to")
        assert len(links) == 1
        assert links[0]["entity_id_b"] == entity_id_from_key("country", "FI")

    def test_mmsi_only_vessel_gets_link(self):
        """Vessel with MMSI only (no IMO) still gets linked."""
        tool, store = _make_tool_with_store(AISVesselTool)
        tool._persist_entities(
            [_make_vessel(mmsi=123456789, imo=None, destination="HOUSTON")]
        )

        vessel_eid = entity_id_from_key("vessel", "mmsi:123456789")
        links = store.query_entity_links(vessel_eid, link_type="port_call_to")
        assert len(links) == 1
        assert links[0]["entity_id_b"] == entity_id_from_key("country", "US")

    def test_no_store_no_crash(self):
        """AISVesselTool without a pipeline_store doesn't crash."""
        tool = AISVesselTool()
        tool._persist_entities([_make_vessel()])  # no-op


# ═══════════════════════════════════════════════════════════════════════
# 17c.2 — lobbies_for links (lobbying)
# ═══════════════════════════════════════════════════════════════════════


class TestLobbyingLobbiesFor:
    """lobbying creates lobbies_for links from registrant → client company."""

    def test_normal_link_created(self):
        """Filing with registrant and client creates a lobbies_for link."""
        tool, store = _make_tool_with_store(LobbyingTool)
        tool._persist_entities([_make_lobby_filing()])

        from agent.pipeline.entity import normalize_company_name

        reg_canon = normalize_company_name("Akin Gump")
        client_canon = normalize_company_name("Amazon.com")
        reg_eid = entity_id_from_key("company", reg_canon)
        client_eid = entity_id_from_key("company", client_canon)

        links = store.query_entity_links(
            reg_eid, link_type="lobbies_for", direction="outgoing"
        )
        assert len(links) == 1
        assert links[0]["entity_id_b"] == client_eid
        assert links[0]["source"] == "lobbying"
        assert links[0]["confidence"] == 0.9

    def test_client_entity_created(self):
        """The client company entity is registered in the store."""
        tool, store = _make_tool_with_store(LobbyingTool)
        tool._persist_entities([_make_lobby_filing(client_name="Tesla Inc")])

        from agent.pipeline.entity import normalize_company_name

        client_canon = normalize_company_name("Tesla Inc")
        client_eid = entity_id_from_key("company", client_canon)
        entity = store.get_entity(client_eid)
        assert entity is not None
        assert entity["entity_type"] == "company"

    def test_same_registrant_and_client_no_link(self):
        """Self-lobbying (registrant == client) produces no lobbies_for link."""
        tool, store = _make_tool_with_store(LobbyingTool)
        tool._persist_entities(
            [_make_lobby_filing(registrant_name="Acme Corp", client_name="Acme Corp")]
        )

        from agent.pipeline.entity import normalize_company_name

        reg_canon = normalize_company_name("Acme Corp")
        reg_eid = entity_id_from_key("company", reg_canon)
        links = store.query_entity_links(reg_eid, link_type="lobbies_for")
        assert len(links) == 0

    def test_empty_client_no_link(self):
        """Empty client_name produces no link."""
        tool, store = _make_tool_with_store(LobbyingTool)
        tool._persist_entities([_make_lobby_filing(client_name="")])

        from agent.pipeline.entity import normalize_company_name

        reg_canon = normalize_company_name("Akin Gump")
        reg_eid = entity_id_from_key("company", reg_canon)
        links = store.query_entity_links(reg_eid, link_type="lobbies_for")
        assert len(links) == 0

    def test_dedup_same_registrant_client(self):
        """Multiple filings with same registrant→client → one link."""
        tool, store = _make_tool_with_store(LobbyingTool)
        tool._persist_entities(
            [
                _make_lobby_filing(filing_period="Q1"),
                _make_lobby_filing(filing_period="Q2"),
            ]
        )

        from agent.pipeline.entity import normalize_company_name

        reg_canon = normalize_company_name("Akin Gump")
        reg_eid = entity_id_from_key("company", reg_canon)
        links = store.query_entity_links(reg_eid, link_type="lobbies_for")
        assert len(links) == 1

    def test_multiple_clients(self):
        """One registrant lobbying for multiple clients → separate links."""
        tool, store = _make_tool_with_store(LobbyingTool)
        tool._persist_entities(
            [
                _make_lobby_filing(client_name="Amazon.com"),
                _make_lobby_filing(client_name="Google LLC"),
            ]
        )

        from agent.pipeline.entity import normalize_company_name

        reg_canon = normalize_company_name("Akin Gump")
        reg_eid = entity_id_from_key("company", reg_canon)
        links = store.query_entity_links(
            reg_eid, link_type="lobbies_for", direction="outgoing"
        )
        assert len(links) == 2

    def test_whitespace_client_stripped(self):
        """Client name with only whitespace produces no link."""
        tool, store = _make_tool_with_store(LobbyingTool)
        tool._persist_entities([_make_lobby_filing(client_name="   ")])

        from agent.pipeline.entity import normalize_company_name

        reg_canon = normalize_company_name("Akin Gump")
        reg_eid = entity_id_from_key("company", reg_canon)
        links = store.query_entity_links(reg_eid, link_type="lobbies_for")
        assert len(links) == 0

    def test_no_store_no_crash(self):
        """LobbyingTool without a pipeline_store doesn't crash."""
        tool = LobbyingTool()
        tool._persist_entities([_make_lobby_filing()])  # no-op


# ===========================================================================
# Helpers: InterconnectionQueue + PatentFilings
# ===========================================================================


def _make_iq_record(
    entity_name: str = "Acme Solar LLC",
    plant_name: str = "Acme Solar Farm",
    capacity_mw: float = 100.0,
    energy_source: str = "SUN",
    state: str = "TX",
    status: str = "planned",
    technology: str = "solar",
) -> dict[str, Any]:
    return {
        "entityName": entity_name,
        "plantName": plant_name,
        "nameplate-capacity-mw": capacity_mw,
        "energy-source-code": energy_source,
        "stateid": state,
        "status": status,
        "technology": technology,
    }


def _make_patent(
    assignee: str = "Big Corp Inc",
    patent_number: str = "US12345678",
    patent_title: str = "System and method for thing",
    patent_date: str = "2025-01-15",
    cpc_subgroup_id: str = "H04L67/00",
) -> dict[str, Any]:
    return {
        "assignee_organization": assignee,
        "patent_number": patent_number,
        "patent_title": patent_title,
        "patent_date": patent_date,
        "cpc_subgroup_id": cpc_subgroup_id,
    }


# ===========================================================================
# Phase 17d: InterconnectionQueue — located_in links
# ===========================================================================


class TestInterconnectionQueueLocatedIn:
    """Phase 17d.1: interconnection_queue company → US country."""

    def test_normal_link_created(self):
        tool, store = _make_tool_with_store(InterconnectionQueueTool)
        tool._persist_entities([_make_iq_record()])

        from agent.pipeline.entity import normalize_company_name

        canon = normalize_company_name("Acme Solar LLC")
        company_eid = entity_id_from_key("company", canon)
        us_eid = entity_id_from_key("country", "US")
        links = store.query_entity_links(company_eid, link_type="located_in")
        assert len(links) == 1
        link = links[0]
        assert link["entity_id_b"] == us_eid
        assert link["link_type"] == "located_in"
        assert link["source"] == "interconnection_queue"
        assert link["confidence"] == 1.0

    def test_us_country_entity_created(self):
        tool, store = _make_tool_with_store(InterconnectionQueueTool)
        tool._persist_entities([_make_iq_record()])

        us_eid = entity_id_from_key("country", "US")
        entities = store.query_all_entities()
        country_entities = [e for e in entities if e["entity_type"] == "country"]
        assert len(country_entities) == 1
        assert country_entities[0]["entity_id"] == us_eid
        assert country_entities[0]["canonical_name"] == "US"

    def test_link_metadata_contains_state(self):
        tool, store = _make_tool_with_store(InterconnectionQueueTool)
        tool._persist_entities([_make_iq_record(state="CA")])

        from agent.pipeline.entity import normalize_company_name

        canon = normalize_company_name("Acme Solar LLC")
        company_eid = entity_id_from_key("company", canon)
        links = store.query_entity_links(company_eid, link_type="located_in")
        assert links[0]["metadata"]["state"] == "CA"

    def test_multiple_companies_all_linked_to_us(self):
        tool, store = _make_tool_with_store(InterconnectionQueueTool)
        records = [
            _make_iq_record(entity_name="Alpha Energy"),
            _make_iq_record(entity_name="Beta Power"),
            _make_iq_record(entity_name="Gamma Wind"),
        ]
        tool._persist_entities(records)

        us_eid = entity_id_from_key("country", "US")
        links = store.query_entity_links(
            us_eid, link_type="located_in", direction="incoming"
        )
        assert len(links) == 3
        assert all(lnk["entity_id_b"] == us_eid for lnk in links)

    def test_dedup_same_company_single_link(self):
        tool, store = _make_tool_with_store(InterconnectionQueueTool)
        records = [
            _make_iq_record(entity_name="Acme Solar LLC"),
            _make_iq_record(entity_name="Acme Solar LLC", plant_name="Acme Farm 2"),
        ]
        tool._persist_entities(records)

        us_eid = entity_id_from_key("country", "US")
        links = store.query_entity_links(
            us_eid, link_type="located_in", direction="incoming"
        )
        # Deduped by entity name → one company → one link (INSERT OR IGNORE)
        assert len(links) == 1

    def test_missing_entity_name_no_link(self):
        tool, store = _make_tool_with_store(InterconnectionQueueTool)
        tool._persist_entities([{"plantName": "Ghost Plant"}])

        us_eid = entity_id_from_key("country", "US")
        links = store.query_entity_links(
            us_eid, link_type="located_in", direction="incoming"
        )
        assert len(links) == 0

    def test_empty_entity_name_no_link(self):
        tool, store = _make_tool_with_store(InterconnectionQueueTool)
        tool._persist_entities([_make_iq_record(entity_name="")])

        us_eid = entity_id_from_key("country", "US")
        links = store.query_entity_links(
            us_eid, link_type="located_in", direction="incoming"
        )
        assert len(links) == 0

    def test_no_store_no_crash(self):
        tool = InterconnectionQueueTool()
        tool._persist_entities([_make_iq_record()])  # no-op


# ===========================================================================
# Phase 17d: PatentFilings — patents_in links
# ===========================================================================


class TestPatentFilingsPatentsIn:
    """Phase 17d.2: patent_filings company → US country."""

    def test_normal_link_created(self):
        tool, store = _make_tool_with_store(PatentFilingsTool)
        tool._persist_entities([_make_patent()])

        from agent.pipeline.entity import normalize_company_name

        canon = normalize_company_name("Big Corp Inc")
        company_eid = entity_id_from_key("company", canon)
        us_eid = entity_id_from_key("country", "US")
        links = store.query_entity_links(company_eid, link_type="patents_in")
        assert len(links) == 1
        link = links[0]
        assert link["entity_id_b"] == us_eid
        assert link["link_type"] == "patents_in"
        assert link["source"] == "patent_filings"
        assert link["confidence"] == 1.0

    def test_us_country_entity_created(self):
        tool, store = _make_tool_with_store(PatentFilingsTool)
        tool._persist_entities([_make_patent()])

        us_eid = entity_id_from_key("country", "US")
        entities = store.query_all_entities()
        country_entities = [e for e in entities if e["entity_type"] == "country"]
        assert len(country_entities) == 1
        assert country_entities[0]["entity_id"] == us_eid

    def test_link_metadata_contains_patent_number(self):
        tool, store = _make_tool_with_store(PatentFilingsTool)
        tool._persist_entities([_make_patent(patent_number="US99999999")])

        from agent.pipeline.entity import normalize_company_name

        canon = normalize_company_name("Big Corp Inc")
        company_eid = entity_id_from_key("company", canon)
        links = store.query_entity_links(company_eid, link_type="patents_in")
        assert links[0]["metadata"]["patent_number"] == "US99999999"

    def test_multiple_assignees_all_linked_to_us(self):
        tool, store = _make_tool_with_store(PatentFilingsTool)
        patents = [
            _make_patent(assignee="Alpha Inc", patent_number="US001"),
            _make_patent(assignee="Beta Corp", patent_number="US002"),
            _make_patent(assignee="Gamma Ltd", patent_number="US003"),
        ]
        tool._persist_entities(patents)

        us_eid = entity_id_from_key("country", "US")
        links = store.query_entity_links(
            us_eid, link_type="patents_in", direction="incoming"
        )
        assert len(links) == 3
        assert all(lnk["entity_id_b"] == us_eid for lnk in links)

    def test_dedup_same_assignee_single_link(self):
        tool, store = _make_tool_with_store(PatentFilingsTool)
        patents = [
            _make_patent(assignee="Big Corp Inc", patent_number="US001"),
            _make_patent(assignee="Big Corp Inc", patent_number="US002"),
        ]
        tool._persist_entities(patents)

        us_eid = entity_id_from_key("country", "US")
        links = store.query_entity_links(
            us_eid, link_type="patents_in", direction="incoming"
        )
        # Same assignee deduped → one company → one link (INSERT OR IGNORE)
        assert len(links) == 1

    def test_missing_assignee_no_link(self):
        tool, store = _make_tool_with_store(PatentFilingsTool)
        tool._persist_entities([{"patent_number": "US001"}])

        us_eid = entity_id_from_key("country", "US")
        links = store.query_entity_links(
            us_eid, link_type="patents_in", direction="incoming"
        )
        assert len(links) == 0

    def test_empty_assignee_no_link(self):
        tool, store = _make_tool_with_store(PatentFilingsTool)
        tool._persist_entities([_make_patent(assignee="")])

        us_eid = entity_id_from_key("country", "US")
        links = store.query_entity_links(
            us_eid, link_type="patents_in", direction="incoming"
        )
        assert len(links) == 0

    def test_list_assignee_uses_first(self):
        """assignee_organization can be a list — should use first element."""
        tool, store = _make_tool_with_store(PatentFilingsTool)
        patent = _make_patent()
        patent["assignee_organization"] = ["FirstCo", "SecondCo"]
        tool._persist_entities([patent])

        from agent.pipeline.entity import normalize_company_name

        canon = normalize_company_name("FirstCo")
        eid = entity_id_from_key("company", canon)
        links = store.query_entity_links(entity_id=eid, link_type="patents_in")
        assert len(links) == 1

    def test_no_store_no_crash(self):
        tool = PatentFilingsTool()
        tool._persist_entities([_make_patent()])  # no-op


# ===========================================================================
# Phase 17e.1: Cross-tool entity consistency
# ===========================================================================


class TestCrossToolEntityConsistencyExtended:
    """17e.1: Same entity referenced from multiple tools → same entity_id."""

    def test_same_company_from_lobbying_and_patents(self):
        """Same company name through lobbying + patent_filings → same entity_id."""
        store = PipelineStore(db_path=":memory:")
        lobby = LobbyingTool(pipeline_store=store)
        patent = PatentFilingsTool(pipeline_store=store)

        lobby._persist_entities(
            [_make_lobby_filing(registrant_name="Acme Corp", client_name="Other Inc")]
        )
        patent._persist_entities([_make_patent(assignee="Acme Corp")])

        from agent.pipeline.entity import normalize_company_name

        canon = normalize_company_name("Acme Corp")
        eid = entity_id_from_key("company", canon)
        # Both tools produced the same entity
        entities = store.query_all_entities()
        company_entities = [e for e in entities if e["entity_id"] == eid]
        assert len(company_entities) == 1  # registered once, not duplicated

    def test_same_company_from_patent_and_iq(self):
        """Same company via patent_filings + interconnection_queue → same entity_id."""
        store = PipelineStore(db_path=":memory:")
        patent = PatentFilingsTool(pipeline_store=store)
        iq = InterconnectionQueueTool(pipeline_store=store)

        patent._persist_entities([_make_patent(assignee="Tesla Inc")])
        iq._persist_entities([_make_iq_record(entity_name="Tesla Inc")])

        from agent.pipeline.entity import normalize_company_name

        canon = normalize_company_name("Tesla Inc")
        eid = entity_id_from_key("company", canon)
        entities = store.query_all_entities()
        company_entities = [e for e in entities if e["entity_id"] == eid]
        assert len(company_entities) == 1

    def test_same_us_country_from_patent_and_iq(self):
        """US country entity created by both patent + IQ → same entity_id, registered once."""
        store = PipelineStore(db_path=":memory:")
        patent = PatentFilingsTool(pipeline_store=store)
        iq = InterconnectionQueueTool(pipeline_store=store)

        patent._persist_entities([_make_patent()])
        iq._persist_entities([_make_iq_record()])

        us_eid = entity_id_from_key("country", "US")
        entities = store.query_all_entities()
        us_entities = [e for e in entities if e["entity_id"] == us_eid]
        assert len(us_entities) == 1

    def test_cross_tool_links_for_same_company(self):
        """Same company linked to US from both patents_in and located_in."""
        store = PipelineStore(db_path=":memory:")
        patent = PatentFilingsTool(pipeline_store=store)
        iq = InterconnectionQueueTool(pipeline_store=store)

        patent._persist_entities([_make_patent(assignee="Google LLC")])
        iq._persist_entities([_make_iq_record(entity_name="Google LLC")])

        from agent.pipeline.entity import normalize_company_name

        canon = normalize_company_name("Google LLC")
        eid = entity_id_from_key("company", canon)
        links = store.query_entity_links(eid, direction="outgoing")
        link_types = {lnk["link_type"] for lnk in links}
        assert "patents_in" in link_types
        assert "located_in" in link_types


# ===========================================================================
# Phase 17e.2: Graph builder integration — links → edge_index
# ===========================================================================


class TestGraphBuilderEdgeIntegration:
    """17e.2: Entity links appear as edge_index in HeteroData."""

    def test_works_for_edges_in_graph(self):
        """InsiderFilings works_for links → (person, works_for, company) edge."""
        store = PipelineStore(db_path=":memory:")
        tool = InsiderFilingsTool(pipeline_store=store)
        tool._persist_entities([_make_txn()])

        from agent.models.gnn.graph_builder import GraphBuilder

        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        triplet = ("person", "works_for", "company")
        assert triplet in data.edge_types
        assert data[triplet].edge_index.shape[0] == 2
        assert data[triplet].edge_index.shape[1] >= 1

    def test_located_in_edges_in_graph(self):
        """InterconnectionQueue located_in → (company, located_in, country) edge."""
        store = PipelineStore(db_path=":memory:")
        tool = InterconnectionQueueTool(pipeline_store=store)
        tool._persist_entities([_make_iq_record()])

        from agent.models.gnn.graph_builder import GraphBuilder

        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        triplet = ("company", "located_in", "country")
        assert triplet in data.edge_types
        assert data[triplet].edge_index.shape[1] == 1

    def test_patents_in_edges_in_graph(self):
        """PatentFilings patents_in → (company, patents_in, country) edge."""
        store = PipelineStore(db_path=":memory:")
        tool = PatentFilingsTool(pipeline_store=store)
        tool._persist_entities([_make_patent()])

        from agent.models.gnn.graph_builder import GraphBuilder

        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        triplet = ("company", "patents_in", "country")
        assert triplet in data.edge_types
        assert data[triplet].edge_index.shape[1] == 1

    def test_edge_attr_contains_confidence(self):
        """Edge attributes include confidence and age."""
        store = PipelineStore(db_path=":memory:")
        tool = InsiderFilingsTool(pipeline_store=store)
        tool._persist_entities([_make_txn()])

        from agent.models.gnn.graph_builder import GraphBuilder

        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        triplet = ("person", "works_for", "company")
        edge_attr = data[triplet].edge_attr
        assert edge_attr.shape[1] == 2  # [confidence, age_days]
        assert edge_attr[0, 0].item() == 1.0  # confidence

    def test_multiple_edge_types_coexist(self):
        """Multiple link types from different tools → multiple edge types in graph."""
        store = PipelineStore(db_path=":memory:")
        InsiderFilingsTool(pipeline_store=store)._persist_entities([_make_txn()])
        InterconnectionQueueTool(pipeline_store=store)._persist_entities(
            [_make_iq_record()]
        )
        PatentFilingsTool(pipeline_store=store)._persist_entities([_make_patent()])

        from agent.models.gnn.graph_builder import GraphBuilder

        builder = GraphBuilder(store)
        data, id_map, events = builder.build()

        assert ("person", "works_for", "company") in data.edge_types
        assert ("company", "located_in", "country") in data.edge_types
        assert ("company", "patents_in", "country") in data.edge_types


# ===========================================================================
# Phase 17e.3: Edge case suite
# ===========================================================================


class TestEntityLinkEdgeCases:
    """17e.3: Edge cases for link_entities and the linking layer."""

    def test_self_link_raises_value_error(self):
        """link_entities with same entity on both sides → ValueError."""
        store = PipelineStore(db_path=":memory:")
        eid = entity_id_from_key("company", "test")
        store.register_entity(
            entity_type="company", canonical_name="test", entity_id=eid
        )
        import pytest

        with pytest.raises(ValueError, match="Cannot link an entity to itself"):
            store.link_entities(
                entity_id_a=eid,
                entity_id_b=eid,
                link_type="self_ref",
                source="test",
            )

    def test_idempotent_link_insertion(self):
        """Inserting the same link twice → only one link stored."""
        store = PipelineStore(db_path=":memory:")
        eid_a = entity_id_from_key("person", "A")
        eid_b = entity_id_from_key("company", "B")
        store.register_entity(entity_type="person", canonical_name="A", entity_id=eid_a)
        store.register_entity(
            entity_type="company", canonical_name="B", entity_id=eid_b
        )

        result1 = store.link_entities(eid_a, eid_b, "works_for", "test", 1.0)
        result2 = store.link_entities(eid_a, eid_b, "works_for", "test", 1.0)

        assert result1 is not None  # first insert returns ID
        assert result2 is None  # second is suppressed
        links = store.query_entity_links(eid_a, link_type="works_for")
        assert len(links) == 1

    def test_long_metadata_stored_correctly(self):
        """Large metadata dict is serialized and round-tripped."""
        store = PipelineStore(db_path=":memory:")
        eid_a = entity_id_from_key("company", "A")
        eid_b = entity_id_from_key("country", "US")
        store.register_entity(
            entity_type="company", canonical_name="A", entity_id=eid_a
        )
        store.register_entity(
            entity_type="country", canonical_name="US", entity_id=eid_b
        )

        big_meta = {f"key_{i}": f"value_{i}" for i in range(100)}
        store.link_entities(eid_a, eid_b, "located_in", "test", 1.0, metadata=big_meta)

        links = store.query_entity_links(eid_a, link_type="located_in")
        assert len(links) == 1
        assert links[0]["metadata"] == big_meta

    def test_no_pipeline_all_tools_safe(self):
        """All tools with no pipeline_store don't crash on persist."""
        for ToolCls in [
            InsiderFilingsTool,
            Form144Tool,
            WhaleAlertTool,
            GDELTTool,
            AISVesselTool,
            LobbyingTool,
            InterconnectionQueueTool,
            PatentFilingsTool,
        ]:
            tool = ToolCls()
            # Calling _persist_entities with no store should be a no-op
            tool._persist_entities([{}])

    def test_different_link_types_same_entity_pair(self):
        """Two different link types between the same entity pair → both stored."""
        store = PipelineStore(db_path=":memory:")
        eid_a = entity_id_from_key("company", "A")
        eid_b = entity_id_from_key("country", "US")
        store.register_entity(
            entity_type="company", canonical_name="A", entity_id=eid_a
        )
        store.register_entity(
            entity_type="country", canonical_name="US", entity_id=eid_b
        )

        store.link_entities(eid_a, eid_b, "located_in", "iq", 1.0)
        store.link_entities(eid_a, eid_b, "patents_in", "patent", 1.0)

        links = store.query_entity_links(eid_a, direction="outgoing")
        assert len(links) == 2
        link_types = {lnk["link_type"] for lnk in links}
        assert link_types == {"located_in", "patents_in"}

    def test_confidence_filter_works(self):
        """query with min_confidence filters out low-confidence links."""
        store = PipelineStore(db_path=":memory:")
        eid_a = entity_id_from_key("vessel", "V1")
        eid_b = entity_id_from_key("country", "EE")
        store.register_entity(
            entity_type="vessel", canonical_name="V1", entity_id=eid_a
        )
        store.register_entity(
            entity_type="country", canonical_name="EE", entity_id=eid_b
        )

        store.link_entities(eid_a, eid_b, "port_call_to", "ais", 0.5)

        links_all = store.query_entity_links(
            eid_a, link_type="port_call_to", min_confidence=0.0
        )
        assert len(links_all) == 1
        links_high = store.query_entity_links(
            eid_a, link_type="port_call_to", min_confidence=0.9
        )
        assert len(links_high) == 0

    def test_query_direction_outgoing_only(self):
        """direction='outgoing' only returns links where entity is source."""
        store = PipelineStore(db_path=":memory:")
        eid_a = entity_id_from_key("company", "A")
        eid_b = entity_id_from_key("country", "US")
        store.register_entity(
            entity_type="company", canonical_name="A", entity_id=eid_a
        )
        store.register_entity(
            entity_type="country", canonical_name="US", entity_id=eid_b
        )
        store.link_entities(eid_a, eid_b, "located_in", "test", 1.0)

        assert len(store.query_entity_links(eid_a, direction="outgoing")) == 1
        assert len(store.query_entity_links(eid_a, direction="incoming")) == 0
        assert len(store.query_entity_links(eid_b, direction="incoming")) == 1
        assert len(store.query_entity_links(eid_b, direction="outgoing")) == 0
