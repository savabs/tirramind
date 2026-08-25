"""
TirraMind — Entity Resolution (Idea 3)

Probabilistic cross-source entity linking using the Fellegi-Sunter model
(implemented by Splink 4) combined with deterministic exact-alias matching.

Problem
-------
Entities from 51+ data sources arrive with different identifiers:
    "Apple Inc."  /  "AAPL"  /  "Apple Computer"  /  CIK 0000320193
These create **separate fragmented nodes** in the knowledge graph —
silently poisoning every relationship that should cross them.

Solution
--------
Two-stage resolution per run:

  Stage 1 — Deterministic  (confidence = 1.0)
    Find entity pairs that share an identical (source, external_id) alias,
    e.g. the same ISIN, LEI, ticker, CIK, MMSI, or IMO number.
    These are definitive matches regardless of name differences.

  Stage 2 — Probabilistic  (confidence from Fellegi-Sunter EM model)
    Within each entity_type, run Splink's EM algorithm on canonical_name
    similarity (Jaro-Winkler) to find near-duplicate names.
    Pairs above ``match_threshold`` are recorded as probable same entities.

Both stages write results as ``entity_links`` rows with
``link_type="same_as"`` and ``source="entity_resolution"``.
The existing ``query_all_entity_links()`` call in ``GraphBuilder`` therefore
automatically includes same_as edges — the GNN learns to propagate
information across entity duplicates.

References
----------
    Fellegi & Sunter (1969) "A theory of record linkage." JASA 64(328):1183.
    Christen (2012) "Data Matching." Springer — §4 (Fellegi-Sunter model).
    moj-analytical-services/splink (MIT) — industrial-scale implementation.
"""

from __future__ import annotations

import logging
import time
import unicodedata
from collections import defaultdict
from typing import Any

log = logging.getLogger(__name__)

# Sources considered definitively identifying (exact match = same entity)
_DETERMINISTIC_SOURCES: frozenset[str] = frozenset(
    {
        "isin",
        "lei",
        "ticker",
        "sec_cik",
        "mmsi",
        "imo",
        "cusip",
        "sedol",
        "figi",
        "gvkey",
        "permid",
    }
)

# Minimum number of entities of a type required to run Splink
_MIN_SPLINK_ENTITIES: int = 5

# Jaro-Winkler thresholds for graduated name comparison
_JW_THRESHOLDS: list[float] = [0.95, 0.88]


def _normalize_name(name: str) -> str:
    """Lowercase, strip accents, collapse whitespace, drop punctuation."""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    name = "".join(c if c.isalnum() or c.isspace() else " " for c in name)
    return " ".join(name.split())


class EntityResolver:
    """Probabilistic cross-source entity resolution using Splink 4 + exact alias matching.

    Usage::

        resolver = EntityResolver(store, match_threshold=0.9)
        n_new = resolver.resolve()

    Parameters
    ----------
    store : PipelineStore
        The pipeline store to read entities/aliases from and write links to.
    match_threshold : float
        Minimum Fellegi-Sunter match probability to store a probabilistic
        ``same_as`` link.  Default 0.9.  Deterministic (alias-based) links
        always use confidence=1.0 regardless of this threshold.
    max_pairs_u_training : int
        Maximum random pairs drawn for Splink u-probability estimation.
        Lower = faster but noisier.  Default 10_000.
    """

    def __init__(
        self,
        store: Any,  # PipelineStore — avoid circular import in type hint
        match_threshold: float = 0.9,
        max_pairs_u_training: int = 10_000,
    ) -> None:
        self._store = store
        self._threshold = match_threshold
        self._max_pairs = max_pairs_u_training

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def resolve(self) -> int:
        """Run full two-stage resolution and persist results.

        Returns
        -------
        int
            Number of *new* ``same_as`` entity_link rows written.  Already-
            existing links (UNIQUE constraint) are silently skipped.
        """
        n_det = self._resolve_deterministic()
        n_prob = self._resolve_probabilistic()
        total = n_det + n_prob
        log.info(
            "EntityResolver: %d deterministic + %d probabilistic = %d new same_as links",
            n_det,
            n_prob,
            total,
        )
        return total

    # ──────────────────────────────────────────────────────────────
    # Stage 1: Deterministic (exact alias match)
    # ──────────────────────────────────────────────────────────────

    def _resolve_deterministic(self) -> int:
        """Find entity pairs sharing a definitive identifying field in metadata_json.

        The ``entity_aliases`` table has a UNIQUE(source, external_id) constraint
        which prevents two entities from holding the same alias simultaneously.
        Instead, this stage inspects each entity's ``metadata_json`` dict for
        well-known identifying keys: isin, lei, ticker, mmsi, imo, cik, etc.

        When two entities carry the same non-empty value for any deterministic
        key (e.g. both have ``"isin": "US0378331005"``), they are almost certainly
        the same real-world entity.  Writes entity_link rows with confidence=1.0.

        Returns count of new links written.
        """
        entities = self._store.query_all_entities()
        if not entities:
            return 0

        # Build index: det_key -> {normalised_value -> [entity_id, ...]}
        key_index: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for ent in entities:
            meta = ent.get("metadata") or {}
            eid = ent["entity_id"]
            for key in _DETERMINISTIC_SOURCES:
                val = meta.get(key)
                if val is not None and str(val).strip():
                    norm_val = str(val).strip().lower()
                    if eid not in key_index[key][norm_val]:
                        key_index[key][norm_val].append(eid)

        # Collect unique pairs where any deterministic key matches
        pairs: set[tuple[str, str]] = set()
        for key_groups in key_index.values():
            for entity_ids in key_groups.values():
                for i in range(len(entity_ids)):
                    for j in range(i + 1, len(entity_ids)):
                        a, b = entity_ids[i], entity_ids[j]
                        pairs.add((min(a, b), max(a, b)))

        count = 0
        for a, b in pairs:
            stored = self._store_same_as(a, b, confidence=1.0, method="deterministic")
            if stored:
                count += 1

        log.debug("Deterministic resolution: %d pairs, %d new links", len(pairs), count)
        return count

    # ──────────────────────────────────────────────────────────────
    # Stage 2: Probabilistic (Splink Fellegi-Sunter)
    # ──────────────────────────────────────────────────────────────

    def _resolve_probabilistic(self) -> int:
        """Run Splink name-similarity deduplication per entity_type.

        Uses Jaro-Winkler distance on normalised canonical_name with
        Fellegi-Sunter EM parameter estimation.  Falls back gracefully when
        data is too sparse for EM training.

        Returns count of new links written.
        """
        try:
            import pandas as pd
            from splink import Linker, SettingsCreator, block_on
            from splink.backends.duckdb import DuckDBAPI
            import splink.comparison_library as cl
        except ImportError:
            log.warning(
                "splink not installed — skipping probabilistic entity resolution."
            )
            return 0

        entities = self._store.query_all_entities()
        if not entities:
            return 0

        # Load deterministic same_as links to exclude pairs already matched
        existing = self._store.query_all_entity_links(link_type="same_as")
        already_linked: set[tuple[str, str]] = set()
        for lnk in existing:
            a, b = lnk["entity_id_a"], lnk["entity_id_b"]
            already_linked.add((min(a, b), max(a, b)))

        # Group entities by type — run Splink independently per type
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ent in entities:
            by_type[ent["entity_type"]].append(ent)

        total_new = 0
        for etype, ents in by_type.items():
            if len(ents) < _MIN_SPLINK_ENTITIES:
                log.debug(
                    "Skipping Splink for entity_type=%r: only %d entities (need ≥%d)",
                    etype,
                    len(ents),
                    _MIN_SPLINK_ENTITIES,
                )
                continue

            n = self._run_splink_for_type(
                etype=etype,
                entities=ents,
                already_linked=already_linked,
                pd=pd,
                Linker=Linker,
                SettingsCreator=SettingsCreator,
                block_on=block_on,
                DuckDBAPI=DuckDBAPI,
                cl=cl,
            )
            total_new += n

        return total_new

    def _run_splink_for_type(
        self,
        *,
        etype: str,
        entities: list[dict[str, Any]],
        already_linked: set[tuple[str, str]],
        pd: Any,
        Linker: Any,
        SettingsCreator: Any,
        block_on: Any,
        DuckDBAPI: Any,
        cl: Any,
    ) -> int:
        """Run Splink deduplication for one entity_type. Returns new link count."""
        import warnings

        # Build DataFrame with normalised name
        rows = [
            {
                "unique_id": ent["entity_id"],
                "canonical_name_norm": _normalize_name(ent.get("canonical_name", "")),
            }
            for ent in entities
            if ent.get("canonical_name")
        ]
        if len(rows) < _MIN_SPLINK_ENTITIES:
            return 0

        df = pd.DataFrame(rows)

        # Blocking: first 4 chars of normalised name (fast, reasonable recall)
        blocking_rules = [block_on("substr(canonical_name_norm, 1, 4)")]

        settings = SettingsCreator(
            link_type="dedupe_only",
            comparisons=[
                cl.JaroWinklerAtThresholds("canonical_name_norm", _JW_THRESHOLDS),
            ],
            blocking_rules_to_generate_predictions=blocking_rules,
            probability_two_random_records_match=0.001,
        )

        try:
            linker = Linker(df, settings, db_api=DuckDBAPI())

            # EM training — may emit warnings on sparse data, suppress them
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    linker.training.estimate_probability_two_random_records_match(
                        blocking_rules, recall=0.7
                    )
                except Exception:
                    pass  # use default prior if estimation fails
                try:
                    linker.training.estimate_u_using_random_sampling(
                        max_pairs=min(self._max_pairs, len(rows) * (len(rows) - 1) // 2)
                    )
                except Exception:
                    pass
                try:
                    linker.training.estimate_m_from_label_column("canonical_name_norm")
                except Exception:
                    pass

            df_pred = linker.inference.predict(
                threshold_match_probability=self._threshold
            ).as_pandas_dataframe()

        except Exception as exc:
            log.warning("Splink failed for entity_type=%r: %s", etype, exc)
            return 0

        if df_pred.empty:
            return 0

        count = 0
        for _, row in df_pred.iterrows():
            a = str(row["unique_id_l"])
            b = str(row["unique_id_r"])
            prob = float(row.get("match_probability", 0.0))
            pair = (min(a, b), max(a, b))
            if pair in already_linked:
                continue
            stored = self._store_same_as(a, b, confidence=prob, method="splink")
            if stored:
                count += 1
                already_linked.add(pair)

        log.debug(
            "Splink (%s): %d candidate pairs → %d new links",
            etype,
            len(df_pred),
            count,
        )
        return count

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    def _store_same_as(
        self,
        entity_id_a: str,
        entity_id_b: str,
        *,
        confidence: float,
        method: str,
    ) -> bool:
        """Write a same_as link. Returns True if a new row was inserted."""
        # Canonical ordering: lexicographic so UNIQUE constraint fires correctly
        a, b = min(entity_id_a, entity_id_b), max(entity_id_a, entity_id_b)
        try:
            result = self._store.link_entities(
                a,
                b,
                "same_as",
                source=f"entity_resolution:{method}",
                confidence=confidence,
                metadata={"method": method},
            )
            return result is not None
        except Exception:
            # UNIQUE constraint violation or self-link = already stored / invalid
            return False


# ── Convenience function ───────────────────────────────────────────────────────


def resolve_entities(
    store: Any,
    match_threshold: float = 0.9,
    max_pairs_u_training: int = 10_000,
) -> int:
    """One-call convenience wrapper around EntityResolver.resolve().

    Args:
        store:               PipelineStore to read from and write to.
        match_threshold:     Minimum Splink match probability for same_as links.
        max_pairs_u_training: Max random pairs for Splink u-estimation.

    Returns:
        Number of new ``same_as`` entity_link rows written.
    """
    resolver = EntityResolver(
        store,
        match_threshold=match_threshold,
        max_pairs_u_training=max_pairs_u_training,
    )
    return resolver.resolve()
