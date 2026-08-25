"""
TirraMind — TS2Vec Contrastive Pretraining Encoder (Idea 5)

Learns universal embeddings for all 51+ data source time series without labels,
then uses them as enriched initial node features in the GNN.

Problem
-------
The GNN trains from scratch on every run.  Entities with sparse history have
nearly zero-information initial features: a one-hot type vector plus a handful
of aggregate statistics.  Cold-start nodes — a newly listed instrument, a vessel
entering a new region — produce unreliable predictions until they accumulate
enough observations to drive gradient updates.

Solution — TS2Vec Contrastive Pretraining
------------------------------------------
TS2Vec (Yue et al., AAAI 2022) learns universal time-series representations via
hierarchical contrastive learning:

  1. **Instance-level contrast**: the same entity's representation should be
     consistent regardless of which sub-segment of its history is seen.
  2. **Temporal contrast**: adjacent timestamps of the same entity should have
     similar representations; distant ones may differ.

The model is a dilated causal TCN with exponentially growing receptive field
(2^depth positions).  Training requires **no labels** — the contrastive loss is
self-supervised across random timestamp crops.

After pretraining, calling ``encode(data, encoding_window='full_series')``
produces one fixed-dim vector per entity that summarises its entire history.
These vectors become extra node features in the GNN, giving every entity a
richer starting point regardless of how sparse its observations are.

Architecture
------------
For each entity type with ≥2 entities:

    observations → uniform time-bin grid → (N, T, C) numpy array
    TS2Vec(input_dims=C, output_dims=ts2vec_dim).fit(X, n_iters=N)
    embs = model.encode(X, encoding_window='full_series')  # (N, ts2vec_dim)

Channel layout (C=2):
    channel 0 — normalised event count in bin  (0-1, captures activity level)
    channel 1 — tanh-normalised mean value in bin  (captures magnitude shape)

The two channels are complementary: channel 0 captures *when* activity
happens (timing patterns), channel 1 captures *what* magnitudes look like
(value patterns).  Together they give TS2Vec enough signal to learn
entity-type-specific dynamics without access to labels.

Integration
-----------
``TS2VecEncoder.fit_and_encode(store)`` is called once in ``Trainer.build_model()``
when ``use_ts2vec=True``.  The returned
``{entity_type: {entity_id: ndarray(ts2vec_dim,)}}`` dict is stored in
``Trainer._ts2vec_embeddings`` and passed to every ``GraphBuilder.build()`` and
``build_from_cached()`` call for the lifetime of that trainer.

Node features expand from ``in_channels[ntype]`` to
``in_channels[ntype] + ts2vec_dim``.  ``in_channels`` is automatically re-read
from the built ``HeteroData.x`` tensor in ``build_model()``, so the HetTGN
``type_projections`` are constructed with the correct extended dimension.

Entities whose type has fewer than 2 entities (cannot form a contrastive batch),
or whose type fails TS2Vec training for any reason, receive a zero embedding of
the correct dimension — they participate in the GNN with unchanged base features.

References
----------
    Yue, Z. et al. (2022). "TS2Vec: Towards Universal Representation of Time
        Series." AAAI 2022. arXiv:2106.10466.
    zhihanyue/ts2vec (MIT) — reference implementation. pip install ts2vec.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Minimum number of entities of a type to attempt TS2Vec pretraining.
# TS2Vec needs ≥2 samples for a meaningful contrastive batch.
_MIN_ENTITIES: int = 2

# Value extraction key priority order (mirrors signature_encoder.py)
_VALUE_KEYS = (
    "close",
    "usd_amount",
    "value",
    "estimated_value",
    "goldstein_scale",
    "btc_amount",
    "log_return",
    "num_articles",
)


# ═══════════════════════════════════════════════════════════════════════════
# TS2VecEncoder
# ═══════════════════════════════════════════════════════════════════════════


class TS2VecEncoder:
    """Pretrain TS2Vec on all entity time series and encode to fixed-dim vectors.

    Parameters
    ----------
    output_dims : int
        Embedding dimension per entity.  Default 32 — small enough for fast
        pretraining, large enough to capture temporal patterns.
    n_iters : int
        TS2Vec training iterations.  200 is the paper default; reduce to 50-100
        for faster development cycles; increase to 500+ for production quality.
    time_bins : int
        Number of uniform time bins used to discretise each entity's history.
        32 bins over the full observation window → ~11 days per bin for a
        1-year history.  Larger values increase resolution but require more
        TS2Vec capacity (increase depth).
    depth : int
        TS2Vec dilated TCN depth.  Receptive field = 2^depth timesteps.
        depth=6 → 64-bin receptive field (covers 2 × time_bins=32).
        depth=10 (paper default) is fine but slower for short sequences.
    device : str
        PyTorch device string ('cpu', 'cuda', 'cuda:0').
    """

    def __init__(
        self,
        output_dims: int = 32,
        n_iters: int = 200,
        time_bins: int = 32,
        depth: int = 6,
        device: str = "cpu",
    ) -> None:
        self.output_dims = output_dims
        self.n_iters = n_iters
        self.time_bins = time_bins
        self.depth = depth
        self.device = device

        # Populated by fit_and_encode()
        self._embeddings: dict[str, dict[str, np.ndarray]] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def fit_and_encode(
        self,
        store: Any,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Pretrain TS2Vec per entity type and return embeddings.

        Args:
            store: PipelineStore instance.  Uses ``query_all_entities()``
                   and ``query_all_observations()``.

        Returns:
            Nested dict ``{entity_type: {entity_id: ndarray(output_dims,)}}``.
            Entity types with fewer than ``_MIN_ENTITIES`` entities are absent.
            Within a present type, all entities have an embedding (zero-vector
            for entities with no observations).
        """
        entities = store.query_all_entities()
        observations = store.query_all_observations()

        # Group observations by entity_id
        obs_by_entity: dict[str, list[dict[str, Any]]] = {}
        for obs in observations:
            eid = obs.get("entity_id", "")
            if eid:
                obs_by_entity.setdefault(eid, []).append(obs)

        # Group entity_ids by type
        entities_by_type: dict[str, list[str]] = {}
        for ent in entities:
            etype = ent.get("entity_type", "")
            eid = ent.get("entity_id", "")
            if etype and eid:
                entities_by_type.setdefault(etype, []).append(eid)

        result: dict[str, dict[str, np.ndarray]] = {}

        for etype, eids in entities_by_type.items():
            if len(eids) < _MIN_ENTITIES:
                log.debug(
                    "TS2Vec: skipping type %r (%d entities < min=%d)",
                    etype, len(eids), _MIN_ENTITIES,
                )
                continue

            type_embs = self._encode_type(etype, eids, obs_by_entity)
            if type_embs is not None:
                result[etype] = type_embs
                log.debug(
                    "TS2Vec: encoded %d %r entities → dim=%d",
                    len(eids), etype, self.output_dims,
                )

        self._embeddings = result
        log.info(
            "TS2Vec pretraining complete: %d types, %d entities total",
            len(result),
            sum(len(v) for v in result.values()),
        )
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _encode_type(
        self,
        etype: str,
        eids: list[str],
        obs_by_entity: dict[str, list[dict[str, Any]]],
    ) -> dict[str, np.ndarray] | None:
        """Build time series array, train TS2Vec, encode, return id→embedding."""
        # ── 1. Determine time range from all observations of this type ─────
        t_min = math.inf
        t_max = -math.inf
        for eid in eids:
            for obs in obs_by_entity.get(eid, []):
                t = obs.get("observed_at")
                if t is not None:
                    t = float(t)
                    if t < t_min:
                        t_min = t
                    if t > t_max:
                        t_max = t

        if math.isinf(t_min) or (t_max - t_min) < 1.0:
            log.debug("TS2Vec: no valid observations for type %r — skipping", etype)
            return None

        t_span = t_max - t_min
        T = self.time_bins
        N = len(eids)

        # ── 2. Build (N, T, 2) array ──────────────────────────────────────
        X = np.zeros((N, T, 2), dtype=np.float32)

        for i, eid in enumerate(eids):
            X[i] = self._build_series(
                obs_by_entity.get(eid, []), t_min, t_span, T
            )

        # ── 3. Train TS2Vec ───────────────────────────────────────────────
        try:
            from ts2vec import TS2Vec  # noqa: PLC0415

            model = TS2Vec(
                input_dims=2,
                output_dims=self.output_dims,
                hidden_dims=64,
                depth=self.depth,
                device=self.device,
                batch_size=min(16, N),
            )
            model.fit(X, n_iters=self.n_iters, verbose=False)
        except Exception as exc:
            log.warning(
                "TS2Vec training failed for entity type %r: %s — using zero embeddings",
                etype, exc,
            )
            return {eid: np.zeros(self.output_dims, dtype=np.float32) for eid in eids}

        # ── 4. Encode full series ─────────────────────────────────────────
        try:
            embs = model.encode(X, encoding_window="full_series")  # (N, output_dims)
        except Exception as exc:
            log.warning(
                "TS2Vec encoding failed for entity type %r: %s — using zero embeddings",
                etype, exc,
            )
            return {eid: np.zeros(self.output_dims, dtype=np.float32) for eid in eids}

        return {eid: embs[i].astype(np.float32) for i, eid in enumerate(eids)}

    def _build_series(
        self,
        observations: list[dict[str, Any]],
        t_min: float,
        t_span: float,
        T: int,
    ) -> np.ndarray:
        """Build a (T, 2) time-binned feature array for one entity.

        Channel 0: normalised event count per bin  (bin_count / max_bin_count)
        Channel 1: tanh-normalised mean observation value per bin
        """
        bin_counts = np.zeros(T, dtype=np.float64)
        bin_value_sums = np.zeros(T, dtype=np.float64)
        bin_value_counts = np.zeros(T, dtype=np.float64)

        for obs in observations:
            t = obs.get("observed_at")
            if t is None:
                continue
            t = float(t)
            frac = (t - t_min) / max(t_span, 1.0)
            bin_idx = min(int(frac * T), T - 1)
            bin_counts[bin_idx] += 1.0

            v = obs.get("value", {})
            if isinstance(v, dict):
                for k in _VALUE_KEYS:
                    if k in v:
                        try:
                            val = float(v[k])
                            if math.isfinite(val):
                                bin_value_sums[bin_idx] += val
                                bin_value_counts[bin_idx] += 1.0
                        except (TypeError, ValueError):
                            pass
                        break

        out = np.zeros((T, 2), dtype=np.float32)

        # Channel 0: normalised counts
        max_count = bin_counts.max()
        if max_count > 0:
            out[:, 0] = (bin_counts / max_count).astype(np.float32)

        # Channel 1: tanh-normalised mean values
        mask = bin_value_counts > 0
        if mask.any():
            mean_vals = np.where(mask, bin_value_sums / np.maximum(bin_value_counts, 1.0), 0.0)
            out[:, 1] = np.tanh(mean_vals / (np.abs(mean_vals) + 1.0)).astype(np.float32)

        return out
