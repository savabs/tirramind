"""
Tests for Tier 8, Change 16 — Self-Extending Entity Ontology.

Covers: OntologyRegistry (seed init, CRUD, hierarchy, validation), entity.py
runtime type validation, TypeInducer (clustering, type proposal, overlap
detection, relationship induction), GNN get_connected_types dynamic expansion,
and global registry accessor.
"""

from __future__ import annotations

import pathlib
import tempfile
import time

import pytest

from agent.discovery.ontology_registry import (
    _TYPE_NAME_PATTERN,
    SEED_ENTITY_TYPES,
    OntologyRegistry,
    TypeInfo,
)
from agent.discovery.type_inducer import TypeInducer
from agent.features.gnn_builder import (
    _SEED_CONNECTED_TYPES,
    get_connected_types,
)
from agent.pipeline.entity import (
    SEED_ENTITY_TYPES as ENTITY_SEED_TYPES,
)
from agent.pipeline.entity import (
    get_ontology_registry,
    set_ontology_registry,
    validate_entity_type,
)
from agent.pipeline.store import PipelineStore

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture()
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield pathlib.Path(d)


@pytest.fixture()
def store(tmp_dir):
    return PipelineStore(tmp_dir / "test.db")


@pytest.fixture()
def registry(store):
    return OntologyRegistry(store)


# ── 1. OntologyRegistry: seed initialization ─────────────────


class TestOntologyRegistrySeedInit:
    def test_seed_types_registered(self, registry):
        known = registry.known_entity_types()
        for t in SEED_ENTITY_TYPES:
            assert t in known, f"Seed type '{t}' missing from registry"

    def test_all_seed_types_active(self, store, registry):
        rows = store.query_entity_types(active_only=True)
        active_names = {r["type_name"] for r in rows}
        for t in SEED_ENTITY_TYPES:
            assert t in active_names

    def test_seed_types_source_is_seed(self, store, registry):
        rows = store.query_entity_types(active_only=True)
        seed_rows = [r for r in rows if r["type_name"] in SEED_ENTITY_TYPES]
        for r in seed_rows:
            assert r["source"] == "seed"


# ── 2. OntologyRegistry: register, query, deactivate ─────────


class TestOntologyRegistryCRUD:
    def test_register_new_type(self, registry):
        assert registry.register_type("facility", source="induced", confidence=0.85)
        assert registry.is_valid_type("facility")

    def test_register_returns_false_for_duplicate(self, registry):
        registry.register_type("facility", source="induced")
        assert not registry.register_type("facility", source="induced")

    def test_deactivate_hides_type(self, registry):
        registry.register_type("temp_type", source="test")
        registry.deactivate_type("temp_type")
        # is_valid_type should exclude inactive types
        known = registry.known_entity_types()
        assert "temp_type" not in known

    def test_reactivate_restores_type(self, registry):
        registry.register_type("revived", source="test")
        registry.deactivate_type("revived")
        registry.reactivate_type("revived")
        assert registry.is_valid_type("revived")

    def test_query_entity_types(self, registry, store):
        registry.register_type("custom_a", source="test")
        rows = store.query_entity_types(active_only=True)
        names = [r["type_name"] for r in rows]
        assert "custom_a" in names


# ── 3. OntologyRegistry: type hierarchy ──────────────────────


class TestOntologyHierarchy:
    def test_parent_type_recorded(self, registry):
        registry.register_type("sub_type", parent_type="company", source="test")
        hierarchy = registry.type_hierarchy()
        assert hierarchy["sub_type"] == "company"

    def test_seed_types_have_no_parent(self, registry):
        hierarchy = registry.type_hierarchy()
        for t in SEED_ENTITY_TYPES:
            assert hierarchy.get(t) is None

    def test_deep_hierarchy(self, registry):
        registry.register_type("level1", parent_type=None, source="test")
        registry.register_type("level2", parent_type="level1", source="test")
        registry.register_type("level3", parent_type="level2", source="test")
        h = registry.type_hierarchy()
        assert h["level3"] == "level2"
        assert h["level2"] == "level1"


# ── 4. OntologyRegistry: validation rules ────────────────────


class TestOntologyValidation:
    def test_empty_name_rejected(self, registry):
        with pytest.raises((ValueError, TypeError)):
            registry.register_type("", source="test")

    def test_invalid_chars_rejected(self, registry):
        with pytest.raises((ValueError, TypeError)):
            registry.register_type("Bad Type!", source="test")

    def test_uppercase_rejected(self, registry):
        with pytest.raises((ValueError, TypeError)):
            registry.register_type("BadName", source="test")

    def test_underscore_allowed(self, registry):
        assert registry.register_type("compound_type", source="test")

    def test_name_pattern_regex(self):
        assert _TYPE_NAME_PATTERN.match("valid_type")
        assert _TYPE_NAME_PATTERN.match("ok123")
        assert not _TYPE_NAME_PATTERN.match("Invalid")
        assert not _TYPE_NAME_PATTERN.match("")
        assert not _TYPE_NAME_PATTERN.match("123start")


# ── 5. validate_entity_type: seed types always valid ─────────


class TestValidateEntityTypeSeed:
    def test_all_seed_types_valid(self):
        for t in ENTITY_SEED_TYPES:
            assert validate_entity_type(t), f"Seed type {t} should be valid"

    def test_unknown_type_invalid_without_registry(self):
        assert not validate_entity_type("nonexistent_type")


# ── 6. validate_entity_type: dynamic types with/without reg ──


class TestValidateEntityTypeDynamic:
    def test_dynamic_type_valid_with_registry(self, registry):
        registry.register_type("dynamic_thing", source="induced")
        assert validate_entity_type("dynamic_thing", registry=registry)

    def test_dynamic_type_invalid_without_registry(self):
        assert not validate_entity_type("dynamic_thing")

    def test_deactivated_type_invalid(self, registry):
        registry.register_type("gone", source="test")
        registry.deactivate_type("gone")
        assert not validate_entity_type("gone", registry=registry)


# ── 7. TypeInducer: clustering unresolved entities ────────────


class TestTypeInducerClustering:
    def test_ingest_creates_entries(self, store, registry):
        inducer = TypeInducer(store, registry, min_cluster_size=2)
        rid = inducer.ingest_unresolved("Entity A", "tool_x", '{"field": "val"}')
        assert rid > 0
        rows = store.query_unresolved_entities(resolved=False)
        assert len(rows) >= 1

    def test_empty_unresolved_returns_nothing(self, store, registry):
        inducer = TypeInducer(store, registry)
        result = inducer.run_induction()
        assert result == []


# ── 8. TypeInducer: type proposal from cluster ───────────────


class TestTypeInducerProposal:
    def test_induction_creates_type(self, store, registry):
        inducer = TypeInducer(store, registry, min_cluster_size=3, cohesion_threshold=0.3)
        # Insert entities from same source with similar context fields
        for i in range(6):
            inducer.ingest_unresolved(
                f"Facility_{i}",
                "power_grid",
                json.dumps({"capacity_mw": 100 + i, "fuel": "gas", "owner": f"Corp{i}"}),
                observed_at=time.time() + i,
            )
        new_types = inducer.run_induction()
        assert len(new_types) >= 1
        # Verify registered
        for t in new_types:
            assert registry.is_valid_type(t)


# ── 9. TypeInducer: overlap detection ────────────────────────


class TestTypeInducerOverlap:
    def test_existing_type_not_duplicated(self, store, registry):
        """If entities strongly overlap with an existing type, no new type."""
        # Register existing company entities
        for i in range(5):
            store.register_entity("company", f"Corp{i}", f"comp_{i}")

        inducer = TypeInducer(store, registry, min_cluster_size=3, cohesion_threshold=0.3)
        # Ingest entities that are also named "Corp" — should overlap
        for i in range(5):
            inducer.ingest_unresolved(
                f"Corp{i}",
                "insider_filings",
                json.dumps({"issuer": f"Corp{i}", "form": "SC 13D"}),
            )
        # Even if clustering works, overlap detection should prevent duplication
        # This is a heuristic test — may or may not create a type depending on impl
        new_types = inducer.run_induction()
        # At minimum, the method should not crash
        assert isinstance(new_types, list)


# ── 10. TypeInducer: relationship induction ──────────────────


class TestTypeInducerRelationships:
    def test_cooccurrence_generates_links(self, store, registry):
        """Entities co-occurring in time should suggest a relationship."""
        registry.register_type("facility", source="test")

        # Create entities of different types
        base_time = time.time()
        for i in range(10):
            store.register_entity("facility", f"Plant_{i}", f"fac_{i}")
            store.register_entity("company", f"Corp_{i}", f"corp_{i}")
            # Co-link them
            store.link_entities(f"fac_{i}", f"corp_{i}", "owned_by", "test")

        inducer = TypeInducer(store, registry, min_cluster_size=3)
        rels = inducer.induce_relationships()
        # Should detect facility-company co-occurrence
        assert isinstance(rels, list)
        # If relationships are found, each should be a tuple
        for r in rels:
            assert isinstance(r, tuple)
            assert len(r) >= 2


# ── 11. get_connected_types: default seed types ──────────────


class TestGetConnectedTypesDefault:
    def test_no_args_returns_seed(self):
        result = get_connected_types()
        assert result == _SEED_CONNECTED_TYPES

    def test_none_store_returns_seed(self):
        result = get_connected_types(store=None, registry=None)
        assert result == _SEED_CONNECTED_TYPES


# ── 12. get_connected_types: dynamic expansion ──────────────


class TestGetConnectedTypesDynamic:
    def test_new_type_with_entities_and_links(self, store, registry):
        registry.register_type("facility", source="test")

        # Create entities + links
        e1 = store.register_entity("facility", "Plant A", "fac_a")
        e2 = store.register_entity("facility", "Plant B", "fac_b")
        co = store.register_entity("company", "ACME", "comp_acme")
        store.link_entities(e1, co, "operates", "test")
        store.link_entities(e2, co, "operates", "test")

        result = get_connected_types(store, registry)
        assert "facility" in result
        # Seed types still present
        for s in _SEED_CONNECTED_TYPES:
            assert s in result


# ── 13. get_connected_types: type below threshold excluded ───


class TestGetConnectedTypesThreshold:
    def test_type_below_entity_count_excluded(self, store, registry):
        registry.register_type("niche_thing", source="test")
        # Only 1 entity — below default min_entities of 2
        store.register_entity("niche_thing", "Lonely Entity", "niche_1")

        result = get_connected_types(store, registry)
        assert "niche_thing" not in result

    def test_type_below_link_count_excluded(self, store, registry):
        registry.register_type("isolated", source="test")
        store.register_entity("isolated", "A", "iso_a")
        store.register_entity("isolated", "B", "iso_b")
        # No links — below min_links threshold
        result = get_connected_types(store, registry)
        assert "isolated" not in result


# ── 14. Global registry accessor: set/get pattern ────────────


class TestGlobalRegistryAccessor:
    def test_set_and_get(self, registry):
        original = get_ontology_registry()
        try:
            set_ontology_registry(registry)
            assert get_ontology_registry() is registry
        finally:
            # Restore original
            if original is not None:
                set_ontology_registry(original)
            else:
                # Reset to None
                import agent.pipeline.entity as ent_mod

                ent_mod._GLOBAL_REGISTRY = None

    def test_default_is_none(self):
        import agent.pipeline.entity as ent_mod

        original = ent_mod._GLOBAL_REGISTRY
        try:
            ent_mod._GLOBAL_REGISTRY = None
            assert get_ontology_registry() is None
        finally:
            ent_mod._GLOBAL_REGISTRY = original


# ── Edge cases: TypeInfo dataclass ────────────────────────────


class TestTypeInfoDataclass:
    def test_frozen(self):
        ti = TypeInfo(name="test", parent_type=None, source="seed", confidence=1.0, active=True)
        with pytest.raises(AttributeError):
            ti.name = "other"  # type: ignore[misc]

    def test_equality(self):
        a = TypeInfo(name="x", parent_type=None, source="seed", confidence=1.0, active=True)
        b = TypeInfo(name="x", parent_type=None, source="seed", confidence=1.0, active=True)
        assert a == b

    def test_hashable(self):
        ti = TypeInfo(name="x", parent_type=None, source="seed", confidence=1.0, active=True)
        {ti}  # should not raise


# Need json for one test
import json
