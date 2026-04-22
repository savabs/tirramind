"""
Phase 36 — Edge case tests for disconnected entity linking.

Tests cover:
- polymarket.py: topic_relates_to_instrument links by category
- cert_transparency.py: domain_owned_by links via company keyword map
- dns_monitor.py: domain_owned_by links via company keyword map
- instrument_universe.py: build_domain_company_map() helper
- SyntheticGraphGenerator: domain/topic link generation + defaults

Edge cases: missing slugs, unmatched domains, empty categories,
deduplication, entity ID consistency, no-dot domains, subdomain
extraction, unmapped categories, graceful fallback on no match.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.pipeline.entity import entity_id_from_key
from agent.tools.cert_transparency import CertTransparencyTool
from agent.tools.dns_monitor import DnsMonitorTool
from agent.tools.instrument_universe import build_domain_company_map
from agent.tools.polymarket import PolymarketTool, _TOPIC_INSTRUMENT_MAP


# ── Helpers ───────────────────────────────────────────────────


def _make_store() -> MagicMock:
    store = MagicMock()
    store.register_entity = MagicMock(
        side_effect=lambda *args, **kw: kw.get("entity_id")
        or (args[2] if len(args) > 2 else "eid")
    )
    store.store_entity_observation = MagicMock(return_value=1)
    store.link_entities = MagicMock(return_value=1)
    store.add_entity_alias = MagicMock()
    store.close = MagicMock()
    store.query_all_entity_links = MagicMock(return_value=[])
    store.query_all_entities = MagicMock(return_value=[])
    store.query_all_observations = MagicMock(return_value=[])
    return store


def _make_market(
    slug: str = "will-btc-reach-100k",
    question: str = "Will BTC reach $100K?",
    category: str = "crypto",
    **overrides: Any,
) -> dict[str, Any]:
    mkt: dict[str, Any] = {
        "slug": slug,
        "question": question,
        "category": category,
        "yes_price": 0.65,
        "no_price": 0.35,
        "volume_24h": 50000.0,
        "volume_total": 500000.0,
        "liquidity": 100000.0,
        "spread": 0.02,
        "price_change_24h": 0.05,
        "price_change_1wk": -0.02,
        "end_date": "2025-12-31",
    }
    mkt.update(overrides)
    return mkt


# ══════════════════════════════════════════════════════════════
# Polymarket: topic_relates_to_instrument links
# ══════════════════════════════════════════════════════════════


class TestPolymarketTopicInstrumentLinks:
    """Tests for topic→instrument linking in polymarket._persist_entities_inner()."""

    def test_crypto_category_links_to_btc_and_eth(self) -> None:
        """Crypto topics link to BTC-USD and ETH-USD."""
        store = _make_store()
        tool = PolymarketTool.__new__(PolymarketTool)
        tool._store = store

        markets = [_make_market(category="crypto")]
        tool._persist_entities_inner(markets)

        link_calls = store.link_entities.call_args_list
        assert len(link_calls) == 2

        linked_tickers = set()
        for call in link_calls:
            assert call.kwargs["link_type"] == "topic_relates_to_instrument"
            assert call.kwargs["source"] == "polymarket"
            assert call.kwargs["confidence"] == 0.7
            linked_tickers.add(call.kwargs["entity_id_b"])

        expected = {
            entity_id_from_key("instrument", "BTC-USD"),
            entity_id_from_key("instrument", "ETH-USD"),
        }
        assert linked_tickers == expected

    def test_finance_category_links_to_correct_instruments(self) -> None:
        """Finance topics link to ES=F, SPY, ZN=F."""
        store = _make_store()
        tool = PolymarketTool.__new__(PolymarketTool)
        tool._store = store

        markets = [_make_market(slug="fed-rate-hike", category="finance")]
        tool._persist_entities_inner(markets)

        link_calls = store.link_entities.call_args_list
        expected_tickers = _TOPIC_INSTRUMENT_MAP["finance"]
        assert len(link_calls) == len(expected_tickers)

        linked_ids = {call.kwargs["entity_id_b"] for call in link_calls}
        expected_ids = {entity_id_from_key("instrument", t) for t in expected_tickers}
        assert linked_ids == expected_ids

    def test_politics_category_creates_no_links(self) -> None:
        """Politics topics have no instrument mapping → no links."""
        store = _make_store()
        tool = PolymarketTool.__new__(PolymarketTool)
        tool._store = store

        markets = [_make_market(slug="us-election", category="politics")]
        tool._persist_entities_inner(markets)

        store.link_entities.assert_not_called()

    def test_empty_category_creates_no_links(self) -> None:
        """Empty category → no links."""
        store = _make_store()
        tool = PolymarketTool.__new__(PolymarketTool)
        tool._store = store

        markets = [_make_market(slug="unknown-market", category="")]
        tool._persist_entities_inner(markets)

        store.link_entities.assert_not_called()

    def test_missing_slug_skips_entirely(self) -> None:
        """Market with no slug → no entity, no links."""
        store = _make_store()
        tool = PolymarketTool.__new__(PolymarketTool)
        tool._store = store

        markets = [_make_market(slug="", category="crypto")]
        tool._persist_entities_inner(markets)

        store.register_entity.assert_not_called()
        store.link_entities.assert_not_called()

    def test_duplicate_slug_links_only_once(self) -> None:
        """Two markets with same slug → entity registered once, links once."""
        store = _make_store()
        tool = PolymarketTool.__new__(PolymarketTool)
        tool._store = store

        markets = [
            _make_market(slug="btc-100k", category="crypto"),
            _make_market(slug="btc-100k", category="crypto", yes_price=0.70),
        ]
        tool._persist_entities_inner(markets)

        # Entity registered once
        assert store.register_entity.call_count == 1
        # Links created once (2 for crypto: BTC-USD, ETH-USD)
        assert store.link_entities.call_count == 2

    def test_entity_id_matches_instrument_universe_scheme(self) -> None:
        """Entity IDs used in links match entity_id_from_key('instrument', ticker)."""
        store = _make_store()
        tool = PolymarketTool.__new__(PolymarketTool)
        tool._store = store

        markets = [_make_market(category="crypto")]
        tool._persist_entities_inner(markets)

        for call in store.link_entities.call_args_list:
            inst_eid = call.kwargs["entity_id_b"]
            # Must be a valid 16-char hex string
            assert len(inst_eid) == 16
            assert all(c in "0123456789abcdef" for c in inst_eid)

    def test_topic_entity_id_is_correct(self) -> None:
        """Topic entity ID in link matches entity_id_from_key('topic', slug)."""
        store = _make_store()
        tool = PolymarketTool.__new__(PolymarketTool)
        tool._store = store

        slug = "my-test-market"
        markets = [_make_market(slug=slug, category="crypto")]
        tool._persist_entities_inner(markets)

        expected_topic_eid = entity_id_from_key("topic", slug)
        for call in store.link_entities.call_args_list:
            assert call.kwargs["entity_id_a"] == expected_topic_eid

    def test_unmapped_categories_create_no_links(self) -> None:
        """Categories not in _TOPIC_INSTRUMENT_MAP produce no links."""
        store = _make_store()
        tool = PolymarketTool.__new__(PolymarketTool)
        tool._store = store

        for cat in ["geopolitics", "tech", "science", "sports"]:
            store.reset_mock()
            markets = [_make_market(slug=f"test-{cat}", category=cat)]
            tool._persist_entities_inner(markets)
            store.link_entities.assert_not_called()

    def test_multiple_markets_different_categories(self) -> None:
        """Markets with different categories each get their own links."""
        store = _make_store()
        tool = PolymarketTool.__new__(PolymarketTool)
        tool._store = store

        markets = [
            _make_market(slug="crypto-market", category="crypto"),
            _make_market(slug="finance-market", category="finance"),
            _make_market(slug="politics-market", category="politics"),
        ]
        tool._persist_entities_inner(markets)

        # crypto: 2 links, finance: 3 links, politics: 0
        expected_link_count = len(_TOPIC_INSTRUMENT_MAP["crypto"]) + len(
            _TOPIC_INSTRUMENT_MAP["finance"]
        )
        assert store.link_entities.call_count == expected_link_count


# ══════════════════════════════════════════════════════════════
# build_domain_company_map()
# ══════════════════════════════════════════════════════════════


class TestBuildDomainCompanyMap:
    """Tests for instrument_universe.build_domain_company_map()."""

    def test_returns_dict(self) -> None:
        result = build_domain_company_map()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_blackrock_in_map(self) -> None:
        """BlackRock is a major issuer and should be in the map."""
        result = build_domain_company_map()
        assert "blackrock" in result
        canon, eid = result["blackrock"]
        assert canon == "blackrock"
        assert len(eid) == 16

    def test_invesco_in_map(self) -> None:
        """Invesco is QQQ issuer."""
        result = build_domain_company_map()
        assert "invesco" in result

    def test_entity_ids_are_consistent(self) -> None:
        """Entity IDs match entity_id_from_key('company', canonical)."""
        result = build_domain_company_map()
        for keyword, (canon, eid) in result.items():
            expected = entity_id_from_key("company", canon)
            assert eid == expected, f"Mismatch for {keyword}: {eid} != {expected}"

    def test_pure_function_returns_same_result(self) -> None:
        """Calling twice returns same dict (deterministic)."""
        r1 = build_domain_company_map()
        r2 = build_domain_company_map()
        assert r1 == r2


# ══════════════════════════════════════════════════════════════
# CertTransparency: domain_owned_by links
# ══════════════════════════════════════════════════════════════


class TestCertTransparencyDomainLinks:
    """Tests for domain→company linking in cert_transparency."""

    def test_matching_domain_creates_link(self) -> None:
        """Domain matching a known company → link created."""
        store = _make_store()
        tool = CertTransparencyTool.__new__(CertTransparencyTool)
        tool._store = store

        # Use a company that's in INSTRUMENTS issuers
        domain = "invesco.com"
        certs = [
            {
                "entry_timestamp": "2025-01-01T00:00:00Z",
                "common_name": domain,
                "issuer_name": "DigiCert",
                "is_expired": False,
            }
        ]
        tool._persist_entities_inner(domain, certs)

        # Should have link_entities called for domain_owned_by
        link_calls = [
            c
            for c in store.link_entities.call_args_list
            if c.kwargs.get("link_type") == "domain_owned_by"
        ]
        assert len(link_calls) == 1
        assert link_calls[0].kwargs["source"] == "cert_transparency"
        assert link_calls[0].kwargs["confidence"] == 0.8

    def test_unmatched_domain_no_link(self) -> None:
        """Domain not matching any known company → no link."""
        store = _make_store()
        tool = CertTransparencyTool.__new__(CertTransparencyTool)
        tool._store = store

        domain = "randomnocompany.com"
        certs = [
            {
                "entry_timestamp": "2025-01-01T00:00:00Z",
                "common_name": domain,
                "issuer_name": "DigiCert",
                "is_expired": False,
            }
        ]
        tool._persist_entities_inner(domain, certs)

        link_calls = [
            c
            for c in store.link_entities.call_args_list
            if c.kwargs.get("link_type") == "domain_owned_by"
        ]
        assert len(link_calls) == 0

    def test_subdomain_extraction(self) -> None:
        """Subdomain like api.invesco.com → base 'invesco' matches."""
        store = _make_store()
        tool = CertTransparencyTool.__new__(CertTransparencyTool)
        tool._store = store

        domain = "api.invesco.com"
        certs = [
            {
                "entry_timestamp": "2025-01-01T00:00:00Z",
                "common_name": domain,
                "issuer_name": "DigiCert",
                "is_expired": False,
            }
        ]
        tool._persist_entities_inner(domain, certs)

        link_calls = [
            c
            for c in store.link_entities.call_args_list
            if c.kwargs.get("link_type") == "domain_owned_by"
        ]
        assert len(link_calls) == 1

    def test_no_dot_domain_handled(self) -> None:
        """Single-word domain (no dots) → graceful, no crash."""
        store = _make_store()
        tool = CertTransparencyTool.__new__(CertTransparencyTool)
        tool._store = store

        domain = "localhost"
        certs = []
        tool._persist_entities_inner(domain, certs)
        # No crash. May or may not create link (depends on whether
        # "localhost" matches a company name — it shouldn't).

    def test_empty_domain_no_crash(self) -> None:
        """Empty domain string handled by _persist_entities (outer)."""
        store = _make_store()
        tool = CertTransparencyTool.__new__(CertTransparencyTool)
        tool._store = store

        # _persist_entities guards against empty domain
        tool._persist_entities("", [])
        store.link_entities.assert_not_called()

    def test_domain_entity_id_in_link_is_correct(self) -> None:
        """The domain entity ID used in the link matches entity_id_from_key."""
        store = _make_store()
        tool = CertTransparencyTool.__new__(CertTransparencyTool)
        tool._store = store

        domain = "invesco.com"
        certs = [
            {
                "entry_timestamp": "2025-01-01T00:00:00Z",
                "common_name": domain,
                "issuer_name": "DigiCert",
                "is_expired": False,
            }
        ]
        tool._persist_entities_inner(domain, certs)

        expected_domain_eid = entity_id_from_key("domain", domain)
        link_calls = [
            c
            for c in store.link_entities.call_args_list
            if c.kwargs.get("link_type") == "domain_owned_by"
        ]
        assert link_calls[0].kwargs["entity_id_a"] == expected_domain_eid


# ══════════════════════════════════════════════════════════════
# DnsMonitor: domain_owned_by links
# ══════════════════════════════════════════════════════════════


class TestDnsMonitorDomainLinks:
    """Tests for domain→company linking in dns_monitor."""

    def test_matching_domain_creates_link(self) -> None:
        """Domain matching a known company → link created."""
        store = _make_store()
        tool = DnsMonitorTool.__new__(DnsMonitorTool)
        tool._store = store

        domain = "invesco.com"
        analysis = {"cloud_providers": ["aws"], "record_count": 5}
        tool._persist_entities_inner(domain, analysis)

        link_calls = [
            c
            for c in store.link_entities.call_args_list
            if c.kwargs.get("link_type") == "domain_owned_by"
        ]
        assert len(link_calls) == 1
        assert link_calls[0].kwargs["source"] == "dns_monitor"
        assert link_calls[0].kwargs["confidence"] == 0.8

    def test_unmatched_domain_no_link(self) -> None:
        """Unknown domain → no link."""
        store = _make_store()
        tool = DnsMonitorTool.__new__(DnsMonitorTool)
        tool._store = store

        domain = "randomsite.com"
        analysis = {"cloud_providers": [], "record_count": 1}
        tool._persist_entities_inner(domain, analysis)

        link_calls = [
            c
            for c in store.link_entities.call_args_list
            if c.kwargs.get("link_type") == "domain_owned_by"
        ]
        assert len(link_calls) == 0

    def test_subdomain_extraction(self) -> None:
        """Subdomain extraction works for dns_monitor too."""
        store = _make_store()
        tool = DnsMonitorTool.__new__(DnsMonitorTool)
        tool._store = store

        domain = "cdn.invesco.com"
        analysis = {"cloud_providers": ["cloudflare"], "record_count": 3}
        tool._persist_entities_inner(domain, analysis)

        link_calls = [
            c
            for c in store.link_entities.call_args_list
            if c.kwargs.get("link_type") == "domain_owned_by"
        ]
        assert len(link_calls) == 1


# ══════════════════════════════════════════════════════════════
# SyntheticGraphGenerator: domain/topic link coverage
# ══════════════════════════════════════════════════════════════


class TestSyntheticGeneratorPhase36:
    """Tests for Phase 36 additions to SyntheticGraphGenerator."""

    def test_default_includes_topics_and_domains(self) -> None:
        """Default num_topics and num_domains are now > 0."""
        from agent.models.gnn.trainer import SyntheticGraphGenerator

        gen = SyntheticGraphGenerator()
        assert gen.num_entities.get("topic", 0) == 3
        assert gen.num_entities.get("domain", 0) == 3

    def test_generates_domain_owned_by_links(self) -> None:
        """Generator produces domain_owned_by links."""
        from agent.models.gnn.trainer import SyntheticGraphGenerator

        store = _make_store()
        # Need companies and domains
        gen = SyntheticGraphGenerator(
            num_companies=3,
            num_countries=2,
            num_domains=3,
            num_vessels=0,
            num_wallets=0,
        )
        gen.generate(store)

        domain_links = [
            c
            for c in store.link_entities.call_args_list
            if len(c.args) >= 3 and c.args[2] == "domain_owned_by"
        ]
        assert len(domain_links) == 3  # one per domain

    def test_generates_topic_relates_to_instrument_links(self) -> None:
        """Generator produces topic_relates_to_instrument links."""
        from agent.models.gnn.trainer import SyntheticGraphGenerator

        store = _make_store()
        gen = SyntheticGraphGenerator(
            num_companies=2,
            num_countries=2,
            num_topics=3,
            num_instruments=4,
            num_vessels=0,
            num_wallets=0,
        )
        gen.generate(store)

        topic_links = [
            c
            for c in store.link_entities.call_args_list
            if len(c.args) >= 3 and c.args[2] == "topic_relates_to_instrument"
        ]
        # Each topic links to min(2, num_instruments) instruments
        assert len(topic_links) == 6  # 3 topics × 2 instruments each

    def test_no_topic_links_without_instruments(self) -> None:
        """If no instruments exist, no topic_relates_to_instrument links."""
        from agent.models.gnn.trainer import SyntheticGraphGenerator

        store = _make_store()
        gen = SyntheticGraphGenerator(
            num_companies=2,
            num_countries=2,
            num_topics=3,
            num_instruments=0,
            num_vessels=0,
            num_wallets=0,
        )
        gen.generate(store)

        topic_links = [
            c
            for c in store.link_entities.call_args_list
            if len(c.args) >= 3 and c.args[2] == "topic_relates_to_instrument"
        ]
        assert len(topic_links) == 0

    def test_no_domain_links_without_companies(self) -> None:
        """If no companies exist, no domain_owned_by links."""
        from agent.models.gnn.trainer import SyntheticGraphGenerator

        store = _make_store()
        gen = SyntheticGraphGenerator(
            num_companies=0,
            num_countries=2,
            num_domains=3,
            num_vessels=0,
            num_wallets=0,
        )
        gen.generate(store)

        domain_links = [
            c
            for c in store.link_entities.call_args_list
            if len(c.args) >= 3 and c.args[2] == "domain_owned_by"
        ]
        assert len(domain_links) == 0

    def test_link_count_in_stats(self) -> None:
        """Stats include domain and topic links in the total count."""
        from agent.models.gnn.trainer import SyntheticGraphGenerator

        store = _make_store()
        gen = SyntheticGraphGenerator(
            num_companies=2,
            num_countries=2,
            num_domains=2,
            num_topics=2,
            num_instruments=3,
            num_vessels=0,
            num_wallets=0,
        )
        stats = gen.generate(store)

        assert stats["links"] > 0
        assert stats["links"] == store.link_entities.call_count


# ══════════════════════════════════════════════════════════════
# _TOPIC_INSTRUMENT_MAP consistency
# ══════════════════════════════════════════════════════════════


class TestTopicInstrumentMap:
    """Verify _TOPIC_INSTRUMENT_MAP entries are valid INSTRUMENTS tickers."""

    def test_all_tickers_exist_in_instruments(self) -> None:
        """Every ticker in _TOPIC_INSTRUMENT_MAP must be in INSTRUMENTS."""
        from agent.tools.instrument_universe import INSTRUMENTS

        valid_tickers = {i.ticker for i in INSTRUMENTS}
        for category, tickers in _TOPIC_INSTRUMENT_MAP.items():
            for ticker in tickers:
                assert ticker in valid_tickers, (
                    f"Ticker {ticker!r} in category {category!r} "
                    f"not found in INSTRUMENTS"
                )

    def test_crypto_maps_to_crypto_instruments(self) -> None:
        """Crypto category maps to BTC-USD and ETH-USD."""
        assert "BTC-USD" in _TOPIC_INSTRUMENT_MAP["crypto"]
        assert "ETH-USD" in _TOPIC_INSTRUMENT_MAP["crypto"]

    def test_no_politics_mapping(self) -> None:
        """Politics has no instrument mapping (deliberate)."""
        assert "politics" not in _TOPIC_INSTRUMENT_MAP
