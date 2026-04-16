"""TirraMind — Dynamic Entity Type & Link Type Registry (Change 16)

Runtime registry that replaces the hardcoded ``Literal`` entity type vocabulary.
Initialized from the DB's existing entity types and the ``entity_type_registry``
table.  New types are added by the TypeInducer or manually.

The registry is the single source of truth for what entity types the system
recognises at runtime.  The 9 seed types are always valid.

Reference: Spec step 16.1 in [[tier8_autonomous_discovery_spec]].
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)

# Seed entity types — always valid even without DB
SEED_ENTITY_TYPES: frozenset[str] = frozenset(
    {
        "company",
        "country",
        "domain",
        "organization",
        "person",
        "protocol",
        "topic",
        "vessel",
        "wallet",
    }
)

_TYPE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True)
class TypeInfo:
    """Metadata about a registered entity type."""

    name: str
    parent_type: str | None
    source: str  # 'seed' | 'induced' | 'manual'
    confidence: float
    active: bool


class OntologyRegistry:
    """Dynamic registry of entity types and relationship (link) types.

    Parameters
    ----------
    store : PipelineStore
        Backing store for persistence.  If *None*, the registry operates
        in memory-only mode with just the seed types.
    """

    def __init__(self, store: PipelineStore | None = None) -> None:
        self._store = store
        self._types: dict[str, TypeInfo] = {}
        self._link_types: set[str] = set()
        self._init_from_store()

    # ── initialisation ─────────────────────────────────────────

    def _init_from_store(self) -> None:
        """Load seed types, then overlay DB-registered types."""
        # Always register seed types
        for t in SEED_ENTITY_TYPES:
            self._types[t] = TypeInfo(
                name=t,
                parent_type=None,
                source="seed",
                confidence=1.0,
                active=True,
            )

        if self._store is None:
            return

        # Persist seed types if not already in DB
        for t in SEED_ENTITY_TYPES:
            self._store.register_entity_type(t, source="seed", confidence=1.0)

        # Load any additional types from the registry table
        for row in self._store.query_entity_types(active_only=False):
            name = row["type_name"]
            if name in self._types and self._types[name].source == "seed":
                continue  # seed types take precedence
            self._types[name] = TypeInfo(
                name=name,
                parent_type=row.get("parent_type"),
                source=row["source"],
                confidence=row["confidence"],
                active=bool(row["active"]),
            )

        # Discover link types already in the DB
        try:
            conn = self._store._get_conn()  # noqa: SLF001
            rows = conn.execute(
                "SELECT DISTINCT link_type FROM entity_links"
            ).fetchall()
            self._link_types = {r[0] for r in rows}
        except Exception:
            pass  # table may be empty

    # ── queries ────────────────────────────────────────────────

    def known_entity_types(self, *, active_only: bool = True) -> frozenset[str]:
        """Return all known entity type names."""
        if active_only:
            return frozenset(t.name for t in self._types.values() if t.active)
        return frozenset(self._types)

    def known_link_types(self) -> frozenset[str]:
        """Return all known relationship (link) type names."""
        return frozenset(self._link_types)

    def is_valid_type(self, type_name: str) -> bool:
        """Check whether *type_name* is a known, active entity type."""
        info = self._types.get(type_name)
        return info is not None and info.active

    def get_type_info(self, type_name: str) -> TypeInfo | None:
        return self._types.get(type_name)

    def type_hierarchy(self) -> dict[str, str | None]:
        """Return ``{type_name: parent_type}`` mapping."""
        return {t.name: t.parent_type for t in self._types.values() if t.active}

    # ── mutations ──────────────────────────────────────────────

    def register_type(
        self,
        type_name: str,
        *,
        parent_type: str | None = None,
        source: str = "induced",
        confidence: float = 1.0,
    ) -> bool:
        """Register a new entity type.  Returns *True* if newly created.

        Raises ``ValueError`` if the name is invalid.
        """
        if not _TYPE_NAME_PATTERN.match(type_name):
            raise ValueError(
                f"Invalid entity type name {type_name!r}: must match {_TYPE_NAME_PATTERN.pattern}"
            )
        if type_name in self._types:
            return False

        info = TypeInfo(
            name=type_name,
            parent_type=parent_type,
            source=source,
            confidence=confidence,
            active=True,
        )
        self._types[type_name] = info

        if self._store is not None:
            self._store.register_entity_type(
                type_name,
                parent_type=parent_type,
                source=source,
                confidence=confidence,
            )
        log.info("Registered new entity type: %s (source=%s)", type_name, source)
        return True

    def register_link_type(self, link_type: str) -> bool:
        """Register a relationship type.  Returns *True* if newly created."""
        if not link_type or not link_type.strip():
            raise ValueError("link_type must be non-empty")
        if link_type in self._link_types:
            return False
        self._link_types.add(link_type)
        log.info("Registered new link type: %s", link_type)
        return True

    def deactivate_type(self, type_name: str) -> None:
        """Deactivate an entity type (never delete)."""
        if type_name in SEED_ENTITY_TYPES:
            raise ValueError(f"Cannot deactivate seed type: {type_name}")
        info = self._types.get(type_name)
        if info is None:
            return
        self._types[type_name] = TypeInfo(
            name=info.name,
            parent_type=info.parent_type,
            source=info.source,
            confidence=info.confidence,
            active=False,
        )
        if self._store is not None:
            self._store.deactivate_entity_type(type_name)

    def reactivate_type(self, type_name: str) -> None:
        """Re-activate a previously deactivated entity type."""
        info = self._types.get(type_name)
        if info is None:
            return
        self._types[type_name] = TypeInfo(
            name=info.name,
            parent_type=info.parent_type,
            source=info.source,
            confidence=info.confidence,
            active=True,
        )
        if self._store is not None:
            self._store.reactivate_entity_type(type_name)
