"""TirraMind — Entity-to-Asset Mapper

Resolves entity_id → tradeable ticker symbol using the existing
entity_aliases table in PipelineStore (source='ticker').

Currently, only 'company' entity_type is directly tradeable.
Other entity types (vessel, wallet, person) may become tradeable
in future phases via inference chains.
"""

from __future__ import annotations

import logging

from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)


class AssetMapper:
    """Map entities to tradeable asset tickers.

    Uses PipelineStore.entity_aliases where source='ticker', restricted
    to entity_type='company'.  Results are cached for the lifetime of
    the mapper instance (entity→ticker mapping is stable within a session).
    """

    def __init__(self, store: PipelineStore) -> None:
        self._store = store
        self._cache: dict[str, str | None] = {}  # entity_id → ticker | None
        self._all_tradeable: dict[str, str] | None = None  # lazy

    def resolve(self, entity_id: str) -> str | None:
        """Return ticker symbol for *entity_id*, or None if not tradeable."""
        if entity_id in self._cache:
            return self._cache[entity_id]

        aliases = self._store.query_entity_aliases(entity_id)
        ticker = None
        for alias in aliases:
            if alias["source"] == "ticker":
                ticker = alias["external_id"]
                break

        self._cache[entity_id] = ticker
        return ticker

    def resolve_batch(self, entity_ids: list[str]) -> dict[str, str]:
        """Return {entity_id: ticker} for all resolvable entities."""
        result: dict[str, str] = {}
        for eid in entity_ids:
            t = self.resolve(eid)
            if t is not None:
                result[eid] = t
        return result

    def tradeable_entities(self) -> dict[str, str]:
        """Return all entities with known tickers.

        Queries all entities, filters to those with ticker aliases.
        Cached after first call.
        """
        if self._all_tradeable is not None:
            return dict(self._all_tradeable)

        all_entities = self._store.query_all_entities()
        result: dict[str, str] = {}
        for ent in all_entities:
            eid = ent["entity_id"]
            t = self.resolve(eid)
            if t is not None:
                result[eid] = t

        self._all_tradeable = result
        return dict(result)

    def clear_cache(self) -> None:
        """Clear the internal cache (e.g., after new aliases are added)."""
        self._cache.clear()
        self._all_tradeable = None
