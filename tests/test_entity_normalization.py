"""Tests for entity name normalization and SEC seed loader."""

from __future__ import annotations

import json

import pytest

from agent.pipeline.entity import (
    entity_id_from_key,
    load_sec_company_tickers,
    normalize_company_name,
)
from agent.pipeline.store import PipelineStore

# ── Fixtures ───────────────────────────────────────────────────


@pytest.fixture
def store():
    s = PipelineStore(db_path=":memory:")
    yield s
    s.close()


@pytest.fixture
def sec_tickers_file(tmp_path):
    """Create a minimal company_tickers.json for testing."""
    data = {
        "0": {"cik_str": "320193", "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": "789019", "ticker": "MSFT", "title": "MICROSOFT CORP"},
        "2": {"cik_str": "1318605", "ticker": "TSLA", "title": "Tesla, Inc."},
        "3": {"cik_str": "1652044", "ticker": "GOOGL", "title": "Alphabet Inc."},
        "4": {"cik_str": "0000000", "ticker": "", "title": "No Ticker LLC"},
    }
    path = tmp_path / "company_tickers.json"
    path.write_text(json.dumps(data))
    return path


# ── normalize_company_name ─────────────────────────────────────


class TestNormalizeCompanyName:
    def test_basic_normalization(self):
        assert normalize_company_name("Apple Inc.") == "apple"

    def test_strip_corp(self):
        assert normalize_company_name("MICROSOFT CORP") == "microsoft"

    def test_strip_ltd(self):
        assert normalize_company_name("British Petroleum Ltd.") == "british petroleum"

    def test_strip_llc(self):
        assert normalize_company_name("Cool Startup LLC") == "cool startup"

    def test_strip_incorporated(self):
        assert normalize_company_name("Acme Incorporated") == "acme"

    def test_strip_corporation(self):
        assert normalize_company_name("Foo Corporation") == "foo"

    def test_strip_limited(self):
        assert normalize_company_name("Bar Limited") == "bar"

    def test_strip_holdings(self):
        assert normalize_company_name("Mega Holdings") == "mega"

    def test_strip_company(self):
        assert normalize_company_name("General Electric Company") == "general electric"

    def test_strip_plc(self):
        assert normalize_company_name("AstraZeneca PLC") == "astrazeneca"

    def test_strip_lp(self):
        assert normalize_company_name("Some Fund LP") == "some fund"

    def test_strip_co(self):
        assert normalize_company_name("Smith & Co.") == "smith"

    def test_collapse_whitespace(self):
        assert normalize_company_name("  Extra   Spaces   Inc  ") == "extra spaces"

    def test_unicode_accents(self):
        # Accented chars should be decomposed
        result = normalize_company_name("Nestlé S.A.")
        assert "nestle" in result

    def test_preserve_hyphens(self):
        result = normalize_company_name("Rolls-Royce Holdings")
        assert "rolls-royce" in result

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            normalize_company_name("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            normalize_company_name("   ")

    def test_suffix_only_name_raises(self):
        """A name that is ONLY a suffix should raise ValueError."""
        with pytest.raises(ValueError, match="normalizes to empty"):
            normalize_company_name("Inc.")

    def test_comma_in_name(self):
        """Tesla, Inc. → tesla"""
        assert normalize_company_name("Tesla, Inc.") == "tesla"

    def test_multiple_suffixes(self):
        result = normalize_company_name("FooBar Holdings Corp.")
        assert result == "foobar"

    def test_case_insensitive(self):
        assert normalize_company_name("aPpLe INC") == normalize_company_name("APPLE inc.")

    def test_deterministic(self):
        """Same input always produces same output."""
        name = "Berkshire Hathaway Inc."
        assert normalize_company_name(name) == normalize_company_name(name)


# ── entity_id_from_key ─────────────────────────────────────────


class TestEntityIdFromKey:
    def test_deterministic(self):
        a = entity_id_from_key("company", "320193")
        b = entity_id_from_key("company", "320193")
        assert a == b

    def test_different_keys(self):
        a = entity_id_from_key("company", "320193")
        b = entity_id_from_key("company", "789019")
        assert a != b

    def test_different_types(self):
        a = entity_id_from_key("company", "320193")
        b = entity_id_from_key("person", "320193")
        assert a != b

    def test_length(self):
        eid = entity_id_from_key("company", "test")
        assert len(eid) == 16

    def test_hex_chars_only(self):
        eid = entity_id_from_key("vessel", "mmsi_12345")
        assert all(c in "0123456789abcdef" for c in eid)


# ── load_sec_company_tickers ───────────────────────────────────


class TestLoadSecCompanyTickers:
    def test_basic_load(self, store: PipelineStore, sec_tickers_file):
        count = load_sec_company_tickers(store, json_path=sec_tickers_file)
        assert count == 5  # all 5 entries

    def test_entities_created(self, store: PipelineStore, sec_tickers_file):
        load_sec_company_tickers(store, json_path=sec_tickers_file)

        # Apple should be resolvable by CIK
        eid = store.resolve_entity("sec_cik", "320193")
        assert eid is not None

        entity = store.get_entity(eid)
        assert entity is not None
        assert entity["entity_type"] == "company"
        assert entity["canonical_name"] == "apple"

    def test_ticker_aliases_created(self, store: PipelineStore, sec_tickers_file):
        load_sec_company_tickers(store, json_path=sec_tickers_file)

        # Should be resolvable by ticker
        eid = store.resolve_entity("ticker", "AAPL")
        assert eid is not None

        # CIK and ticker should resolve to same entity
        eid_cik = store.resolve_entity("sec_cik", "320193")
        assert eid == eid_cik

    def test_idempotent(self, store: PipelineStore, sec_tickers_file):
        count1 = load_sec_company_tickers(store, json_path=sec_tickers_file)
        count2 = load_sec_company_tickers(store, json_path=sec_tickers_file)

        # Both should return same count (entities are INSERT OR IGNORE)
        assert count1 == count2

        # Only one set of entities
        eid = store.resolve_entity("sec_cik", "320193")
        entity = store.get_entity(eid)
        assert entity is not None

    def test_empty_ticker_skipped(self, store: PipelineStore, sec_tickers_file):
        load_sec_company_tickers(store, json_path=sec_tickers_file)

        # Entity with empty ticker should still be created (via CIK)
        eid = store.resolve_entity("sec_cik", "0000000")
        assert eid is not None

        # But no ticker alias
        eid_ticker = store.resolve_entity("ticker", "")
        # Empty ticker should still create an alias since it's not None
        # The function does `if ticker:` — empty string is falsy, so no alias
        assert eid_ticker is None

    def test_file_not_found(self, store: PipelineStore):
        with pytest.raises(FileNotFoundError):
            load_sec_company_tickers(store, json_path="/nonexistent/path.json")

    def test_metadata_preserved(self, store: PipelineStore, sec_tickers_file):
        load_sec_company_tickers(store, json_path=sec_tickers_file)

        eid = store.resolve_entity("sec_cik", "320193")
        entity = store.get_entity(eid)
        assert entity["metadata"]["sec_title"] == "Apple Inc."
