"""Tests for Phase 13b: L2 entity persistence for digital infrastructure tools.

Covers cert_transparency, dns_monitor, and wikipedia_pageviews L2 upgrades:
    - Optional pipeline_store in constructor
    - Entity registration (domain/topic types)
    - Observation storage (cert_issued/dns_change/pageview_spike)
    - Error isolation (persistence failure doesn't crash tool)
    - Empty results handled gracefully
    - Deduplication in batch operations
    - Missing fields / edge cases
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.pipeline.entity import entity_id_from_key
from agent.pipeline.store import PipelineStore
from agent.tools.cert_transparency import CertTransparencyTool
from agent.tools.dns_monitor import DnsMonitorTool
from agent.tools.wikipedia_pageviews import WikipediaPageviewsTool

# ── Fixtures ───────────────────────────────────────────────────


@pytest.fixture()
def store() -> PipelineStore:
    s = PipelineStore(db_path=":memory:")
    yield s
    s.close()


# ── CertTransparencyTool L2 Tests ──────────────────────────────


class TestCertTransparencyL2:
    """L2 entity persistence for CertTransparencyTool."""

    def test_constructor_accepts_pipeline_store(self, store):
        tool = CertTransparencyTool(pipeline_store=store)
        assert tool._store is store

    def test_constructor_without_pipeline_store(self):
        tool = CertTransparencyTool()
        assert tool._store is None

    def test_persist_entities_registers_domain(self, store):
        tool = CertTransparencyTool(pipeline_store=store)
        certs = [
            {
                "entry_timestamp": "2025-01-15T12:00:00Z",
                "is_expired": False,
                "common_name": "*.example.com",
                "issuer_name": "Let's Encrypt",
            }
        ]
        tool._persist_entities("example.com", certs)

        entities = store.query_all_entities()
        assert len(entities) == 1
        assert entities[0]["entity_type"] == "domain"
        assert entities[0]["canonical_name"] == "example.com"

    def test_persist_entities_stores_cert_observation(self, store):
        tool = CertTransparencyTool(pipeline_store=store)
        certs = [
            {
                "entry_timestamp": "2025-01-15T12:00:00Z",
                "is_expired": False,
                "common_name": "api.example.com",
                "issuer_name": "DigiCert",
            }
        ]
        tool._persist_entities("example.com", certs)

        domain_eid = entity_id_from_key("domain", "example.com")
        obs = store.query_entity_observations(domain_eid, source_tool="cert_transparency")
        assert len(obs) == 1
        assert obs[0]["observation_type"] == "cert_issued"
        assert obs[0]["depth_level"] == 2
        assert obs[0]["value"]["common_name"] == "api.example.com"
        assert obs[0]["value"]["issuer_name"] == "DigiCert"
        assert obs[0]["value"]["is_expired"] is False

    def test_persist_entities_multiple_certs(self, store):
        tool = CertTransparencyTool(pipeline_store=store)
        certs = [
            {"entry_timestamp": "2025-01-15T12:00:00Z", "common_name": "a.example.com"},
            {"entry_timestamp": "2025-01-16T12:00:00Z", "common_name": "b.example.com"},
            {"entry_timestamp": "2025-01-17T12:00:00Z", "common_name": "c.example.com"},
        ]
        tool._persist_entities("example.com", certs)

        domain_eid = entity_id_from_key("domain", "example.com")
        obs = store.query_entity_observations(domain_eid, source_tool="cert_transparency")
        assert len(obs) == 3

    def test_persist_entities_empty_certs(self, store):
        """Empty cert list should not crash or create entities."""
        tool = CertTransparencyTool(pipeline_store=store)
        tool._persist_entities("example.com", [])
        entities = store.query_all_entities()
        # Domain still registered even with empty certs
        assert len(entities) == 1

    def test_persist_entities_no_store(self):
        """No store configured — should silently skip."""
        tool = CertTransparencyTool()
        tool._persist_entities("example.com", [{"entry_timestamp": "2025-01-01T00:00:00Z"}])
        # No exception raised

    def test_persist_entities_bad_timestamp(self, store):
        """Invalid timestamp defaults to current time."""
        tool = CertTransparencyTool(pipeline_store=store)
        certs = [{"entry_timestamp": "not-a-date", "common_name": "test.com"}]
        tool._persist_entities("test.com", certs)

        domain_eid = entity_id_from_key("domain", "test.com")
        obs = store.query_entity_observations(domain_eid, source_tool="cert_transparency")
        assert len(obs) == 1
        # Should have a reasonable timestamp (not crash)
        assert obs[0]["observed_at"] > 0

    def test_persist_entities_empty_domain(self, store):
        """Empty domain string should skip persistence."""
        tool = CertTransparencyTool(pipeline_store=store)
        tool._persist_entities("", [{"entry_timestamp": "2025-01-01T00:00:00Z"}])
        entities = store.query_all_entities()
        assert len(entities) == 0

    def test_persist_entities_error_isolation(self, store):
        """Store error should be caught, not propagated."""
        tool = CertTransparencyTool(pipeline_store=store)
        with patch.object(store, "register_entity", side_effect=RuntimeError("DB error")):
            # Should not raise
            tool._persist_entities("example.com", [{"entry_timestamp": "2025-01-01T00:00:00Z"}])

    def test_domain_alias_stored(self, store):
        tool = CertTransparencyTool(pipeline_store=store)
        tool._persist_entities("example.com", [])
        domain_eid = entity_id_from_key("domain", "example.com")
        aliases = store.query_entity_aliases(domain_eid)
        assert any(a["source"] == "domain_name" and a["external_id"] == "example.com" for a in aliases)


# ── DnsMonitorTool L2 Tests ────────────────────────────────────


class TestDnsMonitorL2:
    """L2 entity persistence for DnsMonitorTool."""

    def test_constructor_accepts_pipeline_store(self, store):
        tool = DnsMonitorTool(pipeline_store=store)
        assert tool._store is store

    def test_constructor_without_pipeline_store(self):
        tool = DnsMonitorTool()
        assert tool._store is None

    def test_persist_entities_registers_domain(self, store):
        tool = DnsMonitorTool(pipeline_store=store)
        analysis = {
            "cloud_providers": ["aws"],
            "mx_provider": "google",
            "ns_provider": "cloudflare",
            "min_ttl": 300,
            "low_ttl_warning": False,
            "record_count": 5,
        }
        tool._persist_entities("example.com", analysis)

        entities = store.query_all_entities()
        assert len(entities) == 1
        assert entities[0]["entity_type"] == "domain"
        assert entities[0]["canonical_name"] == "example.com"

    def test_persist_entities_stores_dns_observation(self, store):
        tool = DnsMonitorTool(pipeline_store=store)
        analysis = {
            "cloud_providers": ["aws", "cloudflare"],
            "mx_provider": "google",
            "ns_provider": "cloudflare",
            "min_ttl": 60,
            "low_ttl_warning": True,
            "record_count": 12,
        }
        tool._persist_entities("example.com", analysis)

        domain_eid = entity_id_from_key("domain", "example.com")
        obs = store.query_entity_observations(domain_eid, source_tool="dns_monitor")
        assert len(obs) == 1
        assert obs[0]["observation_type"] == "dns_change"
        assert obs[0]["depth_level"] == 2
        val = obs[0]["value"]
        assert val["cloud_providers"] == ["aws", "cloudflare"]
        assert val["mx_provider"] == "google"
        assert val["low_ttl_warning"] is True
        assert val["min_ttl"] == 60

    def test_persist_entities_empty_domain(self, store):
        tool = DnsMonitorTool(pipeline_store=store)
        tool._persist_entities("", {"record_count": 0})
        entities = store.query_all_entities()
        assert len(entities) == 0

    def test_persist_entities_no_store(self):
        tool = DnsMonitorTool()
        tool._persist_entities("example.com", {"record_count": 5})
        # No exception

    def test_persist_entities_error_isolation(self, store):
        tool = DnsMonitorTool(pipeline_store=store)
        with patch.object(store, "register_entity", side_effect=RuntimeError("DB error")):
            tool._persist_entities("example.com", {"record_count": 5})

    def test_persist_entities_empty_analysis(self, store):
        tool = DnsMonitorTool(pipeline_store=store)
        tool._persist_entities("example.com", {})

        domain_eid = entity_id_from_key("domain", "example.com")
        obs = store.query_entity_observations(domain_eid, source_tool="dns_monitor")
        assert len(obs) == 1
        val = obs[0]["value"]
        assert val["cloud_providers"] == []
        assert val["low_ttl_warning"] is False

    def test_domain_alias_stored(self, store):
        tool = DnsMonitorTool(pipeline_store=store)
        tool._persist_entities("example.com", {})
        domain_eid = entity_id_from_key("domain", "example.com")
        aliases = store.query_entity_aliases(domain_eid)
        assert any(a["source"] == "domain_name" and a["external_id"] == "example.com" for a in aliases)

    def test_same_domain_from_cert_and_dns_same_entity(self, store):
        """Domain registered by cert_transparency and dns_monitor should be the same entity."""
        ct = CertTransparencyTool(pipeline_store=store)
        dns = DnsMonitorTool(pipeline_store=store)

        ct._persist_entities("example.com", [{"entry_timestamp": "2025-01-01T00:00:00Z"}])
        dns._persist_entities("example.com", {"record_count": 5})

        entities = store.query_all_entities()
        # Should be exactly 1 domain entity (deduplicated by entity_id)
        domain_entities = [e for e in entities if e["entity_type"] == "domain"]
        assert len(domain_entities) == 1

        domain_eid = entity_id_from_key("domain", "example.com")
        obs = store.query_entity_observations(domain_eid)
        assert len(obs) == 2
        obs_types = {o["observation_type"] for o in obs}
        assert obs_types == {"cert_issued", "dns_change"}


# ── WikipediaPageviewsTool L2 Tests ────────────────────────────


class TestWikipediaPageviewsL2:
    """L2 entity persistence for WikipediaPageviewsTool."""

    def test_constructor_accepts_pipeline_store(self, store):
        tool = WikipediaPageviewsTool(pipeline_store=store)
        assert tool._store is store

    def test_constructor_without_pipeline_store(self):
        tool = WikipediaPageviewsTool()
        assert tool._store is None

    def test_persist_entities_registers_topic(self, store):
        tool = WikipediaPageviewsTool(pipeline_store=store)
        spikes = [
            {
                "article": "Tesla,_Inc.",
                "z_score": 3.5,
                "latest_views": 100000,
                "mean_views": 20000.0,
                "spike_ratio": 5.0,
                "date": "20250115",
                "project": "en.wikipedia",
            }
        ]
        tool._persist_entities(spikes)

        entities = store.query_all_entities()
        assert len(entities) == 1
        assert entities[0]["entity_type"] == "topic"
        # Underscores replaced with spaces in canonical name
        assert entities[0]["canonical_name"] == "Tesla, Inc."

    def test_persist_entities_stores_pageview_observation(self, store):
        tool = WikipediaPageviewsTool(pipeline_store=store)
        spikes = [
            {
                "article": "Elon_Musk",
                "z_score": 4.2,
                "latest_views": 500000,
                "mean_views": 50000.0,
                "spike_ratio": 10.0,
                "date": "20250120",
                "project": "en.wikipedia",
            }
        ]
        tool._persist_entities(spikes)

        topic_eid = entity_id_from_key("topic", "Elon_Musk")
        obs = store.query_entity_observations(topic_eid, source_tool="wikipedia_pageviews")
        assert len(obs) == 1
        assert obs[0]["observation_type"] == "pageview_spike"
        assert obs[0]["depth_level"] == 2
        val = obs[0]["value"]
        assert val["z_score"] == 4.2
        assert val["latest_views"] == 500000
        assert val["spike_ratio"] == 10.0
        assert val["project"] == "en.wikipedia"

    def test_persist_entities_multiple_spikes(self, store):
        tool = WikipediaPageviewsTool(pipeline_store=store)
        spikes = [
            {
                "article": "Tesla,_Inc.",
                "z_score": 3.0,
                "latest_views": 80000,
                "date": "20250115",
            },
            {
                "article": "Elon_Musk",
                "z_score": 4.0,
                "latest_views": 200000,
                "date": "20250115",
            },
            {
                "article": "SpaceX",
                "z_score": 2.5,
                "latest_views": 60000,
                "date": "20250115",
            },
        ]
        tool._persist_entities(spikes)

        entities = store.query_all_entities()
        assert len(entities) == 3
        assert all(e["entity_type"] == "topic" for e in entities)

    def test_persist_entities_deduplicates_articles(self, store):
        """Same article appearing twice should only register once."""
        tool = WikipediaPageviewsTool(pipeline_store=store)
        spikes = [
            {"article": "Tesla,_Inc.", "z_score": 3.0, "date": "20250115"},
            {"article": "Tesla,_Inc.", "z_score": 3.5, "date": "20250116"},
        ]
        tool._persist_entities(spikes)

        entities = store.query_all_entities()
        assert len(entities) == 1
        # Only first spike creates observation (dedup by seen set)
        topic_eid = entity_id_from_key("topic", "Tesla,_Inc.")
        obs = store.query_entity_observations(topic_eid, source_tool="wikipedia_pageviews")
        assert len(obs) == 1

    def test_persist_entities_empty_spikes(self, store):
        tool = WikipediaPageviewsTool(pipeline_store=store)
        tool._persist_entities([])
        entities = store.query_all_entities()
        assert len(entities) == 0

    def test_persist_entities_no_store(self):
        tool = WikipediaPageviewsTool()
        tool._persist_entities([{"article": "Test", "date": "20250101"}])
        # No exception

    def test_persist_entities_bad_date(self, store):
        """Invalid date string defaults to current time."""
        tool = WikipediaPageviewsTool(pipeline_store=store)
        spikes = [{"article": "Test_Article", "z_score": 2.0, "date": "invalid"}]
        tool._persist_entities(spikes)

        topic_eid = entity_id_from_key("topic", "Test_Article")
        obs = store.query_entity_observations(topic_eid, source_tool="wikipedia_pageviews")
        assert len(obs) == 1
        assert obs[0]["observed_at"] > 0

    def test_persist_entities_missing_article(self, store):
        """Spike with empty article should be skipped."""
        tool = WikipediaPageviewsTool(pipeline_store=store)
        spikes = [{"article": "", "z_score": 3.0, "date": "20250115"}]
        tool._persist_entities(spikes)
        entities = store.query_all_entities()
        assert len(entities) == 0

    def test_persist_entities_error_isolation(self, store):
        tool = WikipediaPageviewsTool(pipeline_store=store)
        with patch.object(store, "register_entity", side_effect=RuntimeError("DB error")):
            tool._persist_entities([{"article": "Test", "date": "20250101"}])

    def test_topic_alias_stored(self, store):
        tool = WikipediaPageviewsTool(pipeline_store=store)
        spikes = [{"article": "Tesla,_Inc.", "z_score": 3.0, "date": "20250115"}]
        tool._persist_entities(spikes)

        topic_eid = entity_id_from_key("topic", "Tesla,_Inc.")
        aliases = store.query_entity_aliases(topic_eid)
        assert any(a["source"] == "wikipedia_article" and a["external_id"] == "Tesla,_Inc." for a in aliases)


# ── Cross-Tool Integration ─────────────────────────────────────


class TestCrossToolEntityDedup:
    """Verify that entities from different tools are properly deduplicated."""

    def test_domain_shared_between_cert_and_dns(self, store):
        """cert_transparency and dns_monitor both register 'example.com' as the same domain entity."""
        ct = CertTransparencyTool(pipeline_store=store)
        dns = DnsMonitorTool(pipeline_store=store)

        ct._persist_entities(
            "stripe.com",
            [
                {
                    "entry_timestamp": "2025-06-01T00:00:00Z",
                    "common_name": "*.stripe.com",
                },
            ],
        )
        dns._persist_entities(
            "stripe.com",
            {
                "cloud_providers": ["aws"],
                "record_count": 8,
            },
        )

        # Single domain entity
        domain_entities = [e for e in store.query_all_entities() if e["entity_type"] == "domain"]
        assert len(domain_entities) == 1

        # Two observations from different tools
        eid = entity_id_from_key("domain", "stripe.com")
        obs = store.query_entity_observations(eid)
        assert len(obs) == 2
        tools = {o["source_tool"] for o in obs}
        assert tools == {"cert_transparency", "dns_monitor"}
