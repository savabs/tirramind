"""
TirraMind — Barra-Style Signal Attribution (Idea 12)

For every instrument node whose return is predicted by the GNN, decomposes
the HGT attention into per-source-type fractional contributions — answering
"Why did this prediction happen?"

Example output for a copper futures entity:
    vessel           : 0.62   (maritime routing signals)
    commodity        : 0.21   (commodity peer cross-attention)
    country          : 0.12   (macro / geographic signals)
    company          : 0.05   (corporate earnings / filings)

Problem
-------
``return_pred_head`` produces a raw log-return scalar.  Callers have no
visibility into which data sources drove the prediction.  Institutional
consumers require factor attribution — "We're long copper because maritime
signals contribute 62%" — for both trust and debugging.

The existing ``AttentionCapturingHGTConv`` already records per-edge
softmax-normalised attention weights (α_ij).  These weights are exactly the
Barra "factor exposure" proxy: summing α_ij over all source nodes of type T
gives the total information contribution from data-source family T to node i
(Hu et al. 2020, §3.2).

Algorithm
---------
1. Enable ``capture_attention=True`` on all HGT layers.
2. Run **one** ``torch.no_grad()`` forward pass (identical cost to inference).
3. For each HGT layer L:
   a. Call ``get_edge_attention()`` → ``{(src, rel, dst): Tensor(E,)}``.
      Each value is per-edge mean attention across attention heads.
   b. For each edge type with ``dst_type == target_type``:
      - Retrieve destination local indices from ``data[etype].edge_index[1]``.
      - Scatter-add: ``raw[dst_local][src_type] += attn[edge_idx]``.
4. Average raw scores across layers (L contributions → 1 score per entity).
5. Normalise per entity so scores sum to 1.0.
6. Capture is disabled in a ``finally`` block — zero overhead after return.

CPU Safety Guarantees
---------------------
- **Hard cap**: ``max_entities`` (default 200).  If more target entities
  exist, the first ``max_entities`` in alphabetical order are attributed.
  Prevents O(E × N) scatters from overwhelming a CPU-only machine.
- ``torch.no_grad()`` wraps the entire forward pass — no gradient tape.
- Attention tensors are already ``detach()``-ed inside
  ``AttentionCapturingHGTConv.message()``.
- No matrix inversions or dense covariance computation.
- Disabled by default (``TrainerConfig.use_attribution = False``).

References
----------
Hu, W. et al. (2020). Heterogeneous Graph Transformer. arXiv:2003.01332.
    §3.2: the softmax attention α_ij quantifies source-to-destination
    information flow.  Aggregating by source type gives factor exposures.

Grinold, R.C. & Kahn, R.N. (1999). Active Portfolio Management, 2nd ed.,
    Ch. 3. McGraw-Hill.  Barra factor decomposition: return_i = Σ_k β_ik·f_k
    where β = factor exposure and f = factor return.  We use ΣαT as β.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from agent.models.gnn.het_tgn import HetTGN
    from agent.models.gnn.graph_builder import IDMap

log = logging.getLogger(__name__)

_EPS: float = 1e-10
_DEFAULT_TARGET_TYPE: str = "instrument"
_DEFAULT_MAX_ENTITIES: int = 200


# ═══════════════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AttributionResult:
    """Per-entity Barra-style factor attribution.

    Attributes
    ----------
    entity_id : str
        The target entity whose prediction is being explained.
    entity_type : str
        Node type of the target (typically "instrument").
    factor_contributions : dict[str, float]
        Source-type name → fractional attention contribution ∈ [0, 1].
        Values sum to 1.0.
    dominant_factor : str
        Source type with the highest contribution.
    top_factors : list[tuple[str, float]]
        All factors sorted descending by contribution, as (name, share).
    n_layers_averaged : int
        Number of HGT layers whose attention was averaged.
    computed_at : float
        Unix timestamp.
    """

    entity_id: str
    entity_type: str
    factor_contributions: dict[str, float]
    dominant_factor: str
    top_factors: list[tuple[str, float]]
    n_layers_averaged: int
    computed_at: float
    details: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# BarraAttribution
# ═══════════════════════════════════════════════════════════════════════════


class BarraAttribution:
    """Decomposes GNN predictions by data-source-type contribution.

    Parameters
    ----------
    target_type : str
        The node type whose predictions are explained.  Default "instrument".
    max_entities : int
        Hard CPU-safety cap.  At most this many entities are attributed
        per call.  If the graph has more, the first ``max_entities``
        (sorted alphabetically by entity_id) are processed.
        Default 200.
    min_attention : float
        Source types whose total raw attention across all edges to a target
        is below this threshold are collapsed into an "other" bucket.
        Set 0 to keep all.  Default 0.0.
    """

    def __init__(
        self,
        target_type: str = _DEFAULT_TARGET_TYPE,
        max_entities: int = _DEFAULT_MAX_ENTITIES,
        min_attention: float = 0.0,
    ) -> None:
        self.target_type = target_type
        self.max_entities = max_entities
        self.min_attention = min_attention

    # ── Public API ─────────────────────────────────────────────────────────

    def compute(
        self,
        model: "HetTGN",
        data: Any,
        id_map: "IDMap",
        target_entity_ids: list[str] | None = None,
    ) -> dict[str, AttributionResult]:
        """Compute per-entity factor attribution via HGT attention capture.

        Args:
            model: Trained HetTGN model.
            data: HeteroData graph snapshot.
            id_map: IDMap from the same graph build.
            target_entity_ids: Restrict attribution to these entity IDs.
                If None, all entities of ``target_type`` are attributed
                (subject to ``max_entities`` cap).

        Returns:
            dict[entity_id, AttributionResult].  Empty if the graph has
            no target-type nodes or attention capture yields nothing.
        """
        local_map = id_map.type_local.get(self.target_type, {})
        if not local_map:
            log.info(
                "BarraAttribution: no '%s' nodes in graph — skipping.",
                self.target_type,
            )
            return {}

        # Determine which entities to attribute
        if target_entity_ids is not None:
            candidates = [e for e in target_entity_ids if e in local_map]
        else:
            candidates = sorted(local_map.keys())

        # CPU safety cap
        if len(candidates) > self.max_entities:
            log.warning(
                "BarraAttribution: %d entities exceeds max_entities=%d; "
                "truncating to first %d (alphabetical).",
                len(candidates), self.max_entities, self.max_entities,
            )
            candidates = candidates[: self.max_entities]

        if not candidates:
            return {}

        # Capture attention from one forward pass
        raw_per_layer = self._capture_attention(model, data, id_map)
        if not raw_per_layer:
            log.info("BarraAttribution: no attention captured — skipping.")
            return {}

        # Aggregate across layers: entity_local → {src_type → sum}
        n_layers = len(raw_per_layer)
        aggregated = self._aggregate_layers(
            raw_per_layer, data, n_layers
        )  # dict[dst_local_int → dict[src_type → float]]

        # Build results for requested entities
        now = time.time()
        results: dict[str, AttributionResult] = {}

        for eid in candidates:
            local_idx = local_map[eid]
            raw = aggregated.get(local_idx, {})
            if not raw:
                continue

            factor_contributions = self._normalize(raw)
            if not factor_contributions:
                continue

            top = sorted(factor_contributions.items(), key=lambda x: -x[1])
            results[eid] = AttributionResult(
                entity_id=eid,
                entity_type=self.target_type,
                factor_contributions=factor_contributions,
                dominant_factor=top[0][0],
                top_factors=top,
                n_layers_averaged=n_layers,
                computed_at=now,
            )

        log.info(
            "BarraAttribution: attributed %d/%d entities (%d layers, %d edge types).",
            len(results),
            len(candidates),
            n_layers,
            sum(len(d) for d in raw_per_layer),
        )
        return results

    def store_results(
        self,
        store: Any,
        results: dict[str, "AttributionResult"],
    ) -> int:
        """Persist attribution signals to the pipeline store.

        Signal name: ``attribution.{entity_id}.{src_type}``
        Value:       fractional contribution ∈ [0, 1]

        Returns:
            Number of signal writes attempted.
        """
        n_written = 0
        for eid, result in results.items():
            for src_type, contribution in result.factor_contributions.items():
                signal_name = f"attribution.{eid}.{src_type}"
                try:
                    store.store_signal(
                        signal_name=signal_name,
                        value=contribution,
                        observed_at=result.computed_at,
                        source_tool="barra_attribution",
                    )
                    n_written += 1
                except Exception:
                    log.warning(
                        "BarraAttribution: failed to store %s", signal_name,
                        exc_info=True,
                    )
        log.info("BarraAttribution: stored %d signals.", n_written)
        return n_written

    # ── Internal ───────────────────────────────────────────────────────────

    def _capture_attention(
        self,
        model: "HetTGN",
        data: Any,
        id_map: "IDMap",
    ) -> list[dict[tuple[str, str, str], torch.Tensor]]:
        """Run one no_grad forward pass and collect per-layer attention.

        Returns:
            List of per-layer dicts: [(src,rel,dst) → Tensor(E,)].
            Tensor values are per-edge mean attention (already detached).
            Returns empty list on any failure.
        """
        was_training = model.training
        for hgt in model.hgt_layers:
            hgt.capture_attention = True
        model.eval()

        try:
            with torch.no_grad():
                model(data, id_map)

            layers_attention: list[dict[tuple[str, str, str], torch.Tensor]] = []
            for hgt in model.hgt_layers:
                edge_attn = hgt.get_edge_attention()
                if edge_attn:
                    layers_attention.append(edge_attn)
            return layers_attention

        except Exception:
            log.warning(
                "BarraAttribution: attention capture failed.", exc_info=True
            )
            return []

        finally:
            for hgt in model.hgt_layers:
                hgt.capture_attention = False
            if was_training:
                model.train()

    def _aggregate_layers(
        self,
        raw_per_layer: list[dict[tuple[str, str, str], torch.Tensor]],
        data: Any,
        n_layers: int,
    ) -> dict[int, dict[str, float]]:
        """Scatter-add attention to destination nodes, average across layers.

        Returns:
            dict[dst_local_int → {src_type → averaged_attention_sum}]
        """
        # Accumulate: dst_local → {src_type → total_attention}
        accum: dict[int, dict[str, float]] = {}

        for layer_attn in raw_per_layer:
            for etype, attn_tensor in layer_attn.items():
                src_type, _rel, dst_type = etype
                if dst_type != self.target_type:
                    continue
                if etype not in data.edge_types:
                    continue

                try:
                    edge_index = data[etype].edge_index  # (2, E)
                except AttributeError:
                    continue

                if edge_index.size(1) == 0:
                    continue

                # Ensure attn matches edge count (guard against shape mismatch)
                n_edges = edge_index.size(1)
                if attn_tensor.size(0) != n_edges:
                    continue

                dst_locals = edge_index[1]  # (E,) — destination local indices

                # CPU-safe scatter: iterate edges
                for e_idx in range(n_edges):
                    dst = int(dst_locals[e_idx].item())
                    a = float(attn_tensor[e_idx].item())
                    if not (a >= 0.0):  # guards NaN/negative
                        continue
                    node_dict = accum.setdefault(dst, {})
                    node_dict[src_type] = node_dict.get(src_type, 0.0) + a

        # Average across layers
        if n_layers > 1:
            for dst_dict in accum.values():
                for k in dst_dict:
                    dst_dict[k] /= n_layers

        return accum

    def _normalize(self, raw: dict[str, float]) -> dict[str, float]:
        """Normalise raw attention sums to fractional contributions.

        Optionally collapses weak sources into an "other" bucket if
        ``min_attention > 0``.
        """
        if not raw:
            return {}

        total = sum(raw.values()) + _EPS
        normalised = {k: v / total for k, v in raw.items()}

        if self.min_attention > 0.0:
            main = {k: v for k, v in normalised.items() if v >= self.min_attention}
            other_sum = sum(v for k, v in normalised.items() if v < self.min_attention)
            if other_sum > _EPS:
                main["other"] = other_sum
            normalised = main

        # Re-normalise after "other" grouping
        total2 = sum(normalised.values()) + _EPS
        return {k: v / total2 for k, v in normalised.items()}
