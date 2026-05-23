"""
TirraMind — Pattern Extraction from Trained HetTGN (Phase 14)

Provides:
    MetaPathPattern         — Scored meta-path discovered by the model.
    CrystallizedPattern     — Production-ready rule config (compatible
                              with cross_entity.py).
    ValidationResult        — Statistical validation of a crystallized pattern.
    PatternExtractor        — Extracts meta-path importance from HGT
                              attention weights (1-hop and 2-hop).
    extract_temporal_lags() — Compute lag distributions for top patterns.
    crystallize()           — Convert patterns → production rules with
                              obs-type conditioned co-occurrence and
                              Fisher's exact test validation.

The extraction pipeline:
    1. Run trained model on graph snapshots, capture HGT attention.
    2. Score 1-hop meta-paths by mean_attention × log(frequency).
    3. Score 2-hop meta-paths by attn(hop1) × attn(hop2) × log2(1 + freq).
    4. For top patterns, analyze source→target temporal lag distributions.
    5. Crystallize using per-edge co-occurrence tables (not global obs freq).
    6. Validate with Fisher's exact test + BH FDR correction.

References:
    HGT (Hu et al. 2020) — attention weights as edge-type importance.
    Fisher (1935) — exact test for 2×2 contingency tables.
    Benjamini & Hochberg (1995) — FDR-controlling procedure.
    Spec steps: 14a–14d.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass

import torch
from torch_geometric.data import HeteroData

from agent.models.gnn.graph_builder import (
    GraphBuilder,
    IDMap,
)
from agent.models.gnn.het_tgn import HetTGN
from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════


@dataclass
class MetaPathPattern:
    """A scored meta-path pattern discovered by the model."""

    src_type: str
    edge_type: str
    dst_type: str
    score: float
    mean_attention: float
    frequency: int  # how many edges of this type
    hops: int = 1
    mean_lag: float = 0.0
    lag_std: float = 0.0
    lag_p25: float = 0.0
    lag_p75: float = 0.0


@dataclass
class CrystallizedPattern:
    """Production rule format compatible with cross_entity.py.

    This is the output of crystallization — a concrete pattern that
    can be evaluated by the pipeline to look for cross-entity signal.
    """

    source_type: str
    target_type: str
    via_edge: str
    obs_type_a: str
    obs_type_b: str
    window_seconds: float
    min_score: float = 0.5
    source: str = "auto_gnn"


@dataclass
class ValidationResult:
    """Statistical validation of a CrystallizedPattern.

    Uses a 2×2 contingency table:
        | source_event & target_in_window | source_event & no_target |
        | no_source      & target_in_window | no_source & no_target   |

    Fisher's exact test (Fisher 1935) gives the p-value under the null
    hypothesis that source and target events are independent.  Lift =
    hit_rate / baseline_rate measures the strength of association.
    The one-sided test (``alternative="greater"``) detects only
    over-representation — patterns where hits occur *more* than
    expected by chance.
    Benjamini–Hochberg (1995) FDR correction is applied across all
    patterns in ``validate_patterns()``.

    Attributes:
        hit_rate: P(target_in_window | source_event).
        baseline_rate: P(target_in_window) overall.
        lift: hit_rate / baseline_rate.
        p_value: Fisher's exact test (one-sided, greater).
        significant: True if p_value < alpha after BH correction.
    """

    hit_rate: float
    baseline_rate: float
    lift: float
    p_value: float
    significant: bool = False


# ═══════════════════════════════════════════════════════════════
# PatternExtractor
# ═══════════════════════════════════════════════════════════════


class PatternExtractor:
    """Extract meta-path importance from a trained HetTGN.

    Runs the model on the full graph (or a snapshot), hooks into HGT
    attention weights, and scores each (src_type, edge_type, dst_type)
    triplet.
    """

    def __init__(
        self,
        model: HetTGN,
        store: PipelineStore,
    ) -> None:
        self.model = model
        self.store = store
        self._graph_builder = GraphBuilder(store)

    def extract_metapath_importance(
        self,
        since: float | None = None,
        until: float | None = None,
    ) -> list[MetaPathPattern]:
        """Score all meta-paths by attention × log(frequency).

        Strategy: register forward hooks on HGT layers to capture
        attention weights. For each (src, rel, dst) edge type,
        compute the mean attention across all edges of that type.

        If attention weights are unavailable (e.g. no-edge graph,
        or HGT internals change), fall back to embedding-based
        scoring: the mean cosine similarity between connected
        src/dst pairs as a proxy for learned importance.

        Returns:
            Sorted list of MetaPathPattern (highest score first).
        """
        self.model.eval()
        data, id_map, events = self._graph_builder.build(since=since, until=until)

        if not data.node_types:
            return []

        # Collect edge counts per type
        edge_counts: dict[tuple[str, str, str], int] = {}
        for etype in self.model.edge_types:
            if etype in data.edge_types:
                edge_counts[etype] = data[etype].edge_index.size(1)
            else:
                edge_counts[etype] = 0

        # Try attention-hook approach first
        attention_scores = self._extract_attention_hooks(data, id_map)

        # Fallback: embedding-based scoring
        if not attention_scores:
            attention_scores = self._extract_embedding_scores(data, id_map)

        # Build 1-hop patterns
        patterns: list[MetaPathPattern] = []
        for etype, mean_attn in attention_scores.items():
            freq = edge_counts.get(etype, 0)
            if freq == 0:
                continue
            score = mean_attn * math.log1p(freq)
            src_type, rel, dst_type = etype
            patterns.append(
                MetaPathPattern(
                    src_type=src_type,
                    edge_type=rel,
                    dst_type=dst_type,
                    score=score,
                    mean_attention=mean_attn,
                    frequency=freq,
                    hops=1,
                )
            )

        # Build 2-hop patterns
        twohop = self._score_2hop_metapaths(attention_scores, edge_counts)
        patterns.extend(twohop)

        patterns.sort(key=lambda p: p.score, reverse=True)
        return patterns

    def _score_2hop_metapaths(
        self,
        attention_scores: dict[tuple[str, str, str], float],
        edge_counts: dict[tuple[str, str, str], int],
    ) -> list[MetaPathPattern]:
        """Enumerate and score 2-hop meta-paths: src→mid→dst.

        A 2-hop path exists when hop1.dst_type == hop2.src_type.  The
        combined score is::

            attn(hop1) × attn(hop2) × log2(1 + freq(hop1) × freq(hop2))

        Only considers edge types that have both attention scores and
        non-zero edge counts.

        Returns:
            List of MetaPathPattern with ``hops=2`` and
            ``edge_type="hop1_rel_via_hop2_rel"``.
        """
        # Build lookup of edge types that have score and edges
        valid: list[tuple[tuple[str, str, str], float, int]] = []
        for etype, attn in attention_scores.items():
            freq = edge_counts.get(etype, 0)
            if freq > 0:
                valid.append((etype, attn, freq))

        patterns: list[MetaPathPattern] = []
        for et1, attn1, freq1 in valid:
            src1, rel1, dst1 = et1
            for et2, attn2, freq2 in valid:
                src2, rel2, dst2 = et2
                if dst1 != src2:
                    continue
                # Avoid trivial self-loops: A→B→A via same relation
                if src1 == dst2 and rel1 == rel2:
                    continue
                combined_attn = attn1 * attn2
                combined_freq = freq1 * freq2
                score = combined_attn * math.log2(1 + combined_freq)
                patterns.append(
                    MetaPathPattern(
                        src_type=src1,
                        edge_type=f"{rel1}_via_{rel2}",
                        dst_type=dst2,
                        score=score,
                        mean_attention=combined_attn,
                        frequency=combined_freq,
                        hops=2,
                    )
                )
        return patterns

    def _extract_attention_hooks(
        self,
        data: HeteroData,
        id_map: IDMap,
    ) -> dict[tuple[str, str, str], float]:
        """Extract per-edge-type mean attention via model.get_attention_weights().

        Uses AttentionCapturingHGTConv's built-in capture mechanism
        instead of fragile forward hooks.  Falls back to empty dict
        if anything goes wrong (caller will use embedding fallback).
        """
        try:
            return self.model.get_attention_weights(data, id_map)
        except Exception as e:
            log.warning("Attention extraction failed: %s", e)
            return {}

    def _extract_embedding_scores(
        self,
        data: HeteroData,
        id_map: IDMap,
    ) -> dict[tuple[str, str, str], float]:
        """Fallback: score edge types by cosine similarity of endpoints.

        For each edge type, compute mean cosine similarity between
        connected source-destination pairs. Higher similarity means
        the model learned to align these node types through the edge.
        """
        with torch.no_grad():
            embeddings = self.model(data, id_map)

        scores: dict[tuple[str, str, str], float] = {}

        for etype in self.model.edge_types:
            if etype not in data.edge_types:
                continue
            src_type, rel, dst_type = etype
            if src_type not in embeddings or dst_type not in embeddings:
                continue

            edge_index = data[etype].edge_index
            if edge_index.size(1) == 0:
                continue

            src_emb = embeddings[src_type][edge_index[0]]  # (E, hidden)
            dst_emb = embeddings[dst_type][edge_index[1]]  # (E, hidden)

            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(src_emb, dst_emb, dim=-1)
            # Shift to [0, 1] range for scoring
            mean_sim = ((cos_sim + 1.0) / 2.0).mean().item()
            scores[etype] = mean_sim

        return scores


# ═══════════════════════════════════════════════════════════════
# Temporal lag extraction
# ═══════════════════════════════════════════════════════════════


def extract_temporal_lags(
    patterns: list[MetaPathPattern],
    store: PipelineStore,
    top_k: int = 10,
) -> list[MetaPathPattern]:
    """For top-K patterns, compute temporal lag distributions.

    For each (src_type → dst_type), collect all pairs of observations
    on linked entities and compute the lag distribution.

    Modifies patterns in-place (adds mean_lag, lag_std, lag_p25, lag_p75).

    Args:
        patterns: Scored MetaPathPatterns from extract_metapath_importance.
        store: PipelineStore with observations and links.
        top_k: How many patterns to analyze.

    Returns:
        Input patterns with lag fields populated (top_k only).
    """
    all_entities = store.query_all_entities()
    eid_to_type = {e["entity_id"]: e["entity_type"] for e in all_entities}

    all_obs = store.query_all_observations()
    # Index observations by entity_id
    obs_by_entity: dict[str, list[dict]] = {}
    for o in all_obs:
        eid = o.get("entity_id")
        if eid:
            obs_by_entity.setdefault(eid, []).append(o)

    all_links = store.query_all_entity_links()
    # Index links by (entity_id_a, link_type)
    link_index: dict[tuple[str, str], list[str]] = {}
    for lnk in all_links:
        key = (lnk["entity_id_a"], lnk["link_type"])
        link_index.setdefault(key, []).append(lnk["entity_id_b"])

    for pattern in patterns[:top_k]:
        lags = _compute_lags_for_pattern(
            pattern,
            eid_to_type,
            obs_by_entity,
            link_index,
        )
        if lags:
            lags_t = torch.tensor(lags, dtype=torch.float)
            pattern.mean_lag = lags_t.mean().item()
            pattern.lag_std = lags_t.std().item() if len(lags) > 1 else 0.0
            pattern.lag_p25 = lags_t.quantile(0.25).item() if len(lags) >= 4 else lags_t.min().item()
            pattern.lag_p75 = lags_t.quantile(0.75).item() if len(lags) >= 4 else lags_t.max().item()

    return patterns[:top_k]


def _compute_lags_for_pattern(
    pattern: MetaPathPattern,
    eid_to_type: dict[str, str],
    obs_by_entity: dict[str, list[dict]],
    link_index: dict[tuple[str, str], list[str]],
) -> list[float]:
    """Compute observed temporal lags for a single meta-path pattern.

    For each entity of src_type linked to an entity of dst_type via
    the pattern's edge_type, find all (src_obs_time, dst_obs_time)
    pairs where dst occurs after src, and record the lag.
    """
    lags: list[float] = []

    # Find all source entities of the right type
    for (eid_a, link_type), targets in link_index.items():
        if link_type != pattern.edge_type:
            continue
        src_type = eid_to_type.get(eid_a)
        if src_type != pattern.src_type:
            continue

        src_obs = obs_by_entity.get(eid_a, [])
        src_times = sorted(o.get("observed_at", 0.0) for o in src_obs)

        for eid_b in targets:
            dst_type = eid_to_type.get(eid_b)
            if dst_type != pattern.dst_type:
                continue

            dst_obs = obs_by_entity.get(eid_b, [])
            dst_times = sorted(o.get("observed_at", 0.0) for o in dst_obs)

            # For each src event, find the nearest subsequent dst event
            di = 0
            for st in src_times:
                while di < len(dst_times) and dst_times[di] <= st:
                    di += 1
                if di < len(dst_times):
                    lag = dst_times[di] - st
                    lags.append(lag)

    return lags


# ═══════════════════════════════════════════════════════════════
# Crystallization
# ═══════════════════════════════════════════════════════════════


def crystallize(
    patterns: list[MetaPathPattern],
    store: PipelineStore,
    threshold: float = 0.1,
    validate: bool = True,
    alpha: float = 0.05,
) -> list[CrystallizedPattern]:
    """Convert scored meta-paths into production rule configs.

    For each pattern above threshold:
        - Build a (src_obs_type, dst_obs_type) co-occurrence table
          on linked entities, picking the pair that co-occurs most
          within a candidate time window.
        - Use the pattern's lag distribution to set window_seconds.
        - Optionally validate with Fisher's exact test + BH FDR.

    Args:
        patterns: MetaPathPattern list (ideally with lags populated).
        store: PipelineStore for observation data.
        threshold: Minimum score to crystallize.
        validate: If True, run validate_patterns() and keep only
            significant patterns.
        alpha: Significance level for validation (after BH correction).

    Returns:
        List of CrystallizedPattern configs.
    """
    filtered = [p for p in patterns if p.score >= threshold]
    if not filtered:
        return []

    # Build data indices
    all_entities = store.query_all_entities()
    eid_to_type = {e["entity_id"]: e["entity_type"] for e in all_entities}

    all_obs = store.query_all_observations()
    obs_by_entity: dict[str, list[dict]] = {}
    for o in all_obs:
        eid = o.get("entity_id")
        if eid:
            obs_by_entity.setdefault(eid, []).append(o)

    all_links = store.query_all_entity_links()
    link_index: dict[tuple[str, str], list[str]] = {}
    for lnk in all_links:
        key = (lnk["entity_id_a"], lnk["link_type"])
        link_index.setdefault(key, []).append(lnk["entity_id_b"])

    results: list[CrystallizedPattern] = []
    for p in filtered:
        # Window = lag_p75 or 2x mean_lag, clamped to [1 hour, 7 days]
        if p.lag_p75 > 0:
            window = p.lag_p75
        elif p.mean_lag > 0:
            window = 2.0 * p.mean_lag
        else:
            window = 86400.0  # default 1 day
        window = max(3600.0, min(window, 7 * 86400.0))

        # Obs-type conditioned co-occurrence
        cooccurrence = _build_cooccurrence_table(
            p,
            eid_to_type,
            obs_by_entity,
            link_index,
            window,
        )
        if cooccurrence:
            # Pick (src_obs, dst_obs) with highest co-occurrence count
            (src_obs, dst_obs), _count = cooccurrence.most_common(1)[0]
        else:
            # Fallback: global most-common obs type per entity type
            type_obs_freq: dict[str, dict[str, int]] = {}
            for o in all_obs:
                eid_ = o.get("entity_id")
                if eid_ is None:
                    continue
                etype_ = eid_to_type.get(eid_)
                if etype_ is None:
                    continue
                ot = o.get("observation_type", "")
                fd = type_obs_freq.setdefault(etype_, {})
                fd[ot] = fd.get(ot, 0) + 1
            src_obs = _most_common_obs_type(type_obs_freq.get(p.src_type, {}))
            dst_obs = _most_common_obs_type(type_obs_freq.get(p.dst_type, {}))

        if not src_obs or not dst_obs:
            continue

        results.append(
            CrystallizedPattern(
                source_type=p.src_type,
                target_type=p.dst_type,
                via_edge=p.edge_type,
                obs_type_a=src_obs,
                obs_type_b=dst_obs,
                window_seconds=window,
                min_score=p.score,
            )
        )

    if validate and results:
        validated = validate_patterns(results, store, alpha=alpha)
        results = [cp for cp, vr in zip(results, validated) if vr.significant]

    return results


def _build_cooccurrence_table(
    pattern: MetaPathPattern,
    eid_to_type: dict[str, str],
    obs_by_entity: dict[str, list[dict]],
    link_index: dict[tuple[str, str], list[str]],
    window: float,
) -> Counter[tuple[str, str]]:
    """Count (src_obs_type, dst_obs_type) co-occurrences within *window*.

    For each linked (src, dst) entity pair matching the pattern's types
    and edge, find all src-obs → dst-obs pairs where dst occurs within
    *window* seconds after the src observation.

    Returns:
        Counter mapping (src_obs_type, dst_obs_type) → count.
    """
    counts: Counter[tuple[str, str]] = Counter()

    for (eid_a, link_type), targets in link_index.items():
        if link_type != pattern.edge_type:
            continue
        if eid_to_type.get(eid_a) != pattern.src_type:
            continue

        src_obs_list = obs_by_entity.get(eid_a, [])
        if not src_obs_list:
            continue

        for eid_b in targets:
            if eid_to_type.get(eid_b) != pattern.dst_type:
                continue
            dst_obs_list = obs_by_entity.get(eid_b, [])
            if not dst_obs_list:
                continue

            # Sort by time
            src_sorted = sorted(src_obs_list, key=lambda o: o.get("observed_at", 0.0))
            dst_sorted = sorted(dst_obs_list, key=lambda o: o.get("observed_at", 0.0))

            di = 0
            for so in src_sorted:
                st = so.get("observed_at", 0.0)
                s_ot = so.get("observation_type", "")
                # Advance dst pointer past events before st
                while di < len(dst_sorted) and dst_sorted[di].get("observed_at", 0.0) <= st:
                    di += 1
                # Collect dst events within window
                for j in range(di, len(dst_sorted)):
                    dt = dst_sorted[j].get("observed_at", 0.0)
                    if dt - st > window:
                        break
                    d_ot = dst_sorted[j].get("observation_type", "")
                    if s_ot and d_ot:
                        counts[(s_ot, d_ot)] += 1

    return counts


def validate_patterns(
    patterns: list[CrystallizedPattern],
    store: PipelineStore,
    alpha: float = 0.05,
) -> list[ValidationResult]:
    """Statistically validate crystallized patterns.

    For each pattern, builds a 2×2 contingency table:

    +-------------------+-----------+-----------+
    |                   | target_in | no_target |
    +-------------------+-----------+-----------+
    | source_event      |     a     |     b     |
    +-------------------+-----------+-----------+
    | no_source         |     c     |     d     |
    +-------------------+-----------+-----------+

    Where:
        - source_event = an observation of obs_type_a on a source entity
        - target_in = obs_type_b observed on a linked target entity
          within window_seconds after the source event
        - Baseline computed from unlinked target entities

    Uses Fisher's exact test (scipy.stats.fisher_exact, one-sided
    ``alternative="greater"``) to detect over-representation only.
    Significant under-representation (hit_rate < baseline) is not
    interesting for pattern discovery.  Applies Benjamini–Hochberg
    FDR correction across all patterns.

    Args:
        patterns: CrystallizedPattern list.
        store: PipelineStore.
        alpha: Nominal significance level (default 0.05).

    Returns:
        List of ValidationResult, one per input pattern, same order.
    """
    from scipy.stats import fisher_exact

    all_entities = store.query_all_entities()
    eid_to_type = {e["entity_id"]: e["entity_type"] for e in all_entities}

    all_obs = store.query_all_observations()
    obs_by_entity: dict[str, list[dict]] = {}
    for o in all_obs:
        eid = o.get("entity_id")
        if eid:
            obs_by_entity.setdefault(eid, []).append(o)

    all_links = store.query_all_entity_links()
    link_index: dict[tuple[str, str], list[str]] = {}
    for lnk in all_links:
        key = (lnk["entity_id_a"], lnk["link_type"])
        link_index.setdefault(key, []).append(lnk["entity_id_b"])

    results: list[ValidationResult] = []

    for cp in patterns:
        a, b, c, d = 0, 0, 0, 0  # contingency cells

        # Count source events and hits
        for (eid_a, lt), targets in link_index.items():
            if lt != cp.via_edge:
                continue
            if eid_to_type.get(eid_a) != cp.source_type:
                continue

            src_obs = [o for o in obs_by_entity.get(eid_a, []) if o.get("observation_type") == cp.obs_type_a]
            if not src_obs:
                continue

            for eid_b in targets:
                if eid_to_type.get(eid_b) != cp.target_type:
                    continue
                dst_obs = sorted(
                    [o for o in obs_by_entity.get(eid_b, []) if o.get("observation_type") == cp.obs_type_b],
                    key=lambda o: o.get("observed_at", 0.0),
                )

                for so in src_obs:
                    st = so.get("observed_at", 0.0)
                    hit = any(0 < (do.get("observed_at", 0.0) - st) <= cp.window_seconds for do in dst_obs)
                    if hit:
                        a += 1
                    else:
                        b += 1

        # Baseline: target obs_type_b frequency on target entities
        # (events on target-type entities regardless of link)
        target_entities = [eid for eid, etype in eid_to_type.items() if etype == cp.target_type]
        total_target_obs = 0
        target_b_count = 0
        for te in target_entities:
            for o in obs_by_entity.get(te, []):
                total_target_obs += 1
                if o.get("observation_type") == cp.obs_type_b:
                    target_b_count += 1

        baseline_rate = target_b_count / max(total_target_obs, 1)
        total_source_events = a + b
        hit_rate = a / max(total_source_events, 1)
        lift = hit_rate / max(baseline_rate, 1e-9)

        # Construct contingency: c and d from baseline
        # c = expected hits if independent, d = remainder
        c = max(target_b_count - a, 0)
        d = max(total_target_obs - target_b_count - b, 0)

        # Fisher's exact test
        if total_source_events == 0 or total_target_obs == 0:
            p_value = 1.0
        else:
            _, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")

        results.append(
            ValidationResult(
                hit_rate=hit_rate,
                baseline_rate=baseline_rate,
                lift=lift,
                p_value=p_value,
                significant=False,  # set after BH correction
            )
        )

    # Benjamini–Hochberg FDR correction
    if results:
        n = len(results)
        # Sort by p-value, apply BH
        indexed = sorted(enumerate(results), key=lambda x: x[1].p_value)
        for rank, (idx, vr) in enumerate(indexed, start=1):
            bh_threshold = alpha * rank / n
            if vr.p_value <= bh_threshold:
                vr.significant = True
            else:
                # Once one fails, all higher p-values also fail
                break

    return results


def _most_common_obs_type(freq: dict[str, int]) -> str | None:
    """Return the most frequent observation type, or None."""
    if not freq:
        return None
    return max(freq, key=freq.get)  # type: ignore[arg-type]
