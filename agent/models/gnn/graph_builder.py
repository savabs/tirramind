"""
TirraMind — Graph Builder (Phase 12a)

Converts the PipelineStore entity graph into a PyG HeteroData object
suitable for training heterogeneous temporal graph networks.

Design:
    1. Query all entities → typed node feature tensors.
    2. Query all entity_links → typed edge_index tensors.
    3. Query all observations → per-node temporal event lists.
    4. Build bidirectional (type, entity_id) ↔ global int ID mappings.

The resulting HeteroData has:
    - data[node_type].x         — feature matrix [N_type, feat_dim]
    - data[node_type].node_ids  — original entity_id strings
    - data[src, rel, dst].edge_index — [2, E] connectivity
    - data[src, rel, dst].edge_attr  — [E, edge_feat_dim]

Temporal events are returned separately as a sorted list of dicts,
not embedded in the static graph (the trainer consumes them
sequentially).

References:
    HGT (Hu et al. 2020, arXiv:2003.01332) — motivates heterogeneous typing.
    PyG HeteroData — torch_geometric.data.HeteroData API.
    Spec step 12a.3.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

# Ignore corrupt future timestamps when choosing graph reference time
# (e.g. gov_contracts bad start_date years). Allows 1-day clock slack.
_REFERENCE_TIME_SLACK_SEC = 86400.0
from torch_geometric.data import HeteroData

from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────

# Canonical ordering — determines one-hot position in node features.
ENTITY_TYPES: list[str] = [
    "cftc_contract",
    "company",
    "country",
    "domain",
    "instrument",
    "maritime_area",
    "organization",
    "person",
    "protocol",
    "topic",
    "vessel",
    "wallet",
]

OBSERVATION_TYPES: list[str] = [
    "area_daily_activity",
    "baltic_activity_proxy",
    "bankruptcy_status",
    "border_throughput",
    "btc_transfer",
    "campaign_finance",
    "capital_flow",
    "cb_balance_sheet",
    "cb_policy_rate",
    "cert_issued",
    "consumer_confidence",
    "contract_award",
    "creditor_filing",
    "cross_entity_pattern",
    "dividend",
    "dns_change",
    "drug_approval",
    "economic_activity",
    "food_security",
    "form144_filing",
    "futures_positioning",
    "futures_positioning_derived",
    "geopolitical_event",
    "grid_demand",
    "insider_trade",
    "instrument_daily",
    "instrument_return",
    "instrument_volatility",
    "instrument_volume",
    "internet_disruption",
    "investigation_signal",
    "lobbying_spend",
    "market_probability",
    "migration_pressure",
    "options_chain_eod",
    "pageview_spike",
    "patent_filing",
    "pathogen_level",
    "petroleum_inventory",
    "port_call",
    "price_movement",
    "project_status",
    "regulatory_velocity",
    "research_velocity",
    "sanctions_listing",
    "sell_intent",
    "short_interest",
    "sovereign_yield",
    "trade_flow",
    "tvl_change",
    "vessel_position",
    "whale_trade",
]

_ENTITY_TYPE_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(ENTITY_TYPES)}
_OBS_TYPE_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(OBSERVATION_TYPES)}


def _links_as_of(links: list[dict[str, Any]], until: float | None) -> list[dict[str, Any]]:
    """Drop links that did not exist yet at `until` (future-blindness, F-04).

    A link created after the window's end is information from the future. Left
    unfiltered it produced the worst kind of leakage — silent, and flattering:
    backtests improve, and the improvement is entirely spurious.

    `until=None` means "live/current", so everything is in scope.

    Links with no `created_at` are KEPT: some were written before the column
    existed, and dropping them would silently shrink historical graphs. That is
    a deliberate trade — note it if backtest results look suspicious.
    """
    if until is None:
        return links
    return [lk for lk in links if lk.get("created_at") is None or float(lk["created_at"]) <= until]


class SchemaDriftError(ValueError):
    """Raised when the store holds entity/observation types the model can't encode.

    One-hot positions derive from list index in ENTITY_TYPES / OBSERVATION_TYPES,
    so a type present in the DB but absent from those lists cannot be represented
    at all. Feature building degrades gracefully (all-zero type one-hot) so that
    live collection of a brand-new type never crashes, but anything that trains
    or scores against weights must refuse to run on a schema it cannot encode.
    """


def validate_schema_against_store(store, *, strict: bool = True) -> dict[str, list[str]]:
    """Compare live DB entity/observation types against the code's registries.

    This is the guard that was missing when the instrument feature vector grew
    23 → 49 and `maritime_area` appeared in the DB: both drifted silently for
    months because nothing ever compared the two.

    Args:
        store: PipelineStore to inspect.
        strict: raise SchemaDriftError on drift. False → return the report only.

    Returns:
        {"unknown_entity_types": [...], "unknown_observation_types": [...]}

    Raises:
        SchemaDriftError: if strict and either list is non-empty.
    """
    conn = store._get_conn()  # noqa: SLF001 — store exposes no type-listing API
    db_entity_types = {r[0] for r in conn.execute("SELECT DISTINCT entity_type FROM entities")}
    db_obs_types = {r[0] for r in conn.execute("SELECT DISTINCT observation_type FROM entity_observations")}

    report = {
        "unknown_entity_types": sorted(db_entity_types - set(ENTITY_TYPES)),
        "unknown_observation_types": sorted(db_obs_types - set(OBSERVATION_TYPES)),
    }

    if strict and (report["unknown_entity_types"] or report["unknown_observation_types"]):
        raise SchemaDriftError(
            "Store contains types the model cannot encode.\n"
            f"  unknown entity types      : {report['unknown_entity_types']}\n"
            f"  unknown observation types : {report['unknown_observation_types']}\n"
            "Add them to ENTITY_TYPES / OBSERVATION_TYPES in "
            "agent/models/gnn/graph_builder.py (alphabetical order — one-hot "
            "positions derive from list index) and retrain: inserting a type "
            "shifts every later index and invalidates existing checkpoints."
        )
    return report


# ── ID mapping ─────────────────────────────────────────────────


@dataclass
class IDMap:
    """Bidirectional mapping between (entity_type, entity_id) and global int."""

    typed_to_global: dict[tuple[str, str], int] = field(default_factory=dict)
    global_to_typed: dict[int, tuple[str, str]] = field(default_factory=dict)
    # Per-type local indices: entity_type → {entity_id: local_idx}
    type_local: dict[str, dict[str, int]] = field(default_factory=dict)
    _next_global: int = 0

    def add(self, entity_type: str, entity_id: str) -> int:
        """Register an entity, return its global ID.  Idempotent."""
        key = (entity_type, entity_id)
        if key in self.typed_to_global:
            return self.typed_to_global[key]
        gid = self._next_global
        self._next_global += 1
        self.typed_to_global[key] = gid
        self.global_to_typed[gid] = key
        local = self.type_local.setdefault(entity_type, {})
        local[entity_id] = len(local)
        return gid

    def global_id(self, entity_type: str, entity_id: str) -> int | None:
        """Look up global ID.  Returns None if not registered."""
        return self.typed_to_global.get((entity_type, entity_id))

    def local_id(self, entity_type: str, entity_id: str) -> int | None:
        """Per-type local index (0-based within that type)."""
        local = self.type_local.get(entity_type)
        if local is None:
            return None
        return local.get(entity_id)

    @property
    def num_nodes(self) -> int:
        return self._next_global

    def num_nodes_of_type(self, entity_type: str) -> int:
        return len(self.type_local.get(entity_type, {}))


# ── Observation statistics helper ──────────────────────────────


def _compute_obs_stats(
    observations: list[dict[str, Any]],
    entity_id: str,
    current_time: float,
) -> dict[str, float]:
    """Aggregate observation statistics for a single entity.

    Returns: count, recency (seconds since last obs), mean_value.
    """
    ent_obs = [o for o in observations if o.get("entity_id") == entity_id]
    count = len(ent_obs)
    if count == 0:
        return {"count": 0.0, "recency": 0.0, "mean_value": 0.0}

    latest_t = max(o.get("observed_at", 0.0) for o in ent_obs)
    recency = current_time - latest_t if current_time > 0 else 0.0

    values: list[float] = []
    for o in ent_obs:
        v = o.get("value", {})
        if isinstance(v, dict):
            # Try common value fields
            for k in (
                "usd_amount",
                "btc_amount",
                "value",
                "estimated_value",
                "goldstein_scale",
                "num_articles",
            ):
                if k in v:
                    try:
                        values.append(float(v[k]))
                    except (TypeError, ValueError):
                        pass
                    break
    mean_val = sum(values) / len(values) if values else 0.0

    return {"count": float(count), "recency": recency, "mean_value": mean_val}


# ── Node feature builder ──────────────────────────────────────


def _reference_time(observations: list[dict[str, Any]]) -> float:
    """Latest sane observation timestamp for recency / causal cutoffs.

    Filters observed_at values beyond now + 1 day so one corrupt row
    (e.g. year 9019 in gov_contracts) cannot zero-out all node features.
    """
    if not observations:
        return 0.0
    ceiling = time.time() + _REFERENCE_TIME_SLACK_SEC
    sane = [float(o.get("observed_at", 0.0)) for o in observations if 0.0 < float(o.get("observed_at", 0.0)) <= ceiling]
    if sane:
        return max(sane)
    return max(float(o.get("observed_at", 0.0)) for o in observations)


# ── Enrichment feature dimensions ──────────────────────────────
# When enrichment is provided, these extra features are appended:
#   cusum_state (1) + hawkes_intensity (1) + event_study_score (1) +
#   bocpd_prob (1) + value_variance (1) + value_min (1) + value_max (1) +
#   value_iqr (1) + num_source_tools (1) + obs_type_dist (len(OBSERVATION_TYPES))
#
# MUST stay derived. This was hardcoded to 55 — correct only while
# len(OBSERVATION_TYPES) == 46. When the registry grew to 48, the writer at
# `offset + 9 + ot_idx` (see _build_node_features) ran past the allocated width:
# with BASE_FEAT_DIM=14 the tensor was 14+55=69 wide and ot_idx=46 addressed
# index 69 — "index 69 is out of bounds for dimension 1 with size 69", which
# crashed the entity_scoring DAG. For instrument nodes the same overflow instead
# silently corrupted the price-feature block that follows it.
_ENRICHMENT_SCALAR_DIM = 9  # cusum, hawkes, event_study, bocpd, var, min, max, iqr, num_tools
ENRICHMENT_DIM = _ENRICHMENT_SCALAR_DIM + len(OBSERVATION_TYPES)
BASE_FEAT_DIM = len(ENTITY_TYPES) + 3  # one-hot type + count + recency + mean_val
PRICE_FEAT_DIM = 9  # momentum(3) + volatility(2) + volume(2) + max_dd + sharpe
# Idea 2 — depth-3 path signature dim (PATH_CHANNELS=3): 3+9+27 = 39
# Imported lazily inside _build_node_features to avoid circular import.
_SIGNATURE_DIM_CACHE: int | None = None
# Idea 5 — TS2Vec output dim (set by GraphBuilder.build() callers; 0 = disabled)
# Stored here so _build_node_features can read it without an extra argument when
# ts2vec_embeddings is provided but ts2vec_dim is not yet wired through.
# Actual value is always passed explicitly — this is just documentation.
_TS2VEC_DIM: int = 0
# M9 — Microstructure features (11 dims)
# spread_cs(1) + spread_roll(1) + ofi_zscore(1) + vpin(1) + vpin_regime(3) + kyle_lambda(1) + lambda_regime(3)
MICROSTRUCTURE_DIM = 11
# M14.1 — options(7) + rate(5) + dividend(3) — see agent/quant/gnn_quant_features.py
M15_QUANT_DIM = 15


def xsnorm_price_feats(feats: torch.Tensor) -> torch.Tensor:
    """Cross-sectionally z-score the price feature block before return_raw_head.

    Normalises only dims [BASE_FEAT_DIM : BASE_FEAT_DIM + PRICE_FEAT_DIM] so
    that all 9 price features are on the same scale within each evaluation
    window.  Without this, avg_vol_20d (range [37-100]) dominates the gradient
    signal and the head cannot learn momentum / sharpe factors.

    Must be called at EVERY call site that invokes return_raw_head — both at
    training time (trainer.py) and inference time (quant_benchmark.py, ic_check.py).
    The standalone trainer (train_raw_head.py) does this internally in build_panel.

    Args:
        feats: [N, D] float tensor, N >= 2 instruments in the same window.
    Returns:
        [N, D] tensor; price block z-scored, all other dims unchanged.
    """

    if feats.shape[0] < 2:
        return feats
    out = feats.clone()
    block = out[:, BASE_FEAT_DIM : BASE_FEAT_DIM + PRICE_FEAT_DIM]
    mean = block.mean(0, keepdim=True)
    std = block.std(0, keepdim=True).clamp(min=1e-8)
    out[:, BASE_FEAT_DIM : BASE_FEAT_DIM + PRICE_FEAT_DIM] = (block - mean) / std
    return out


def _signature_dim() -> int:
    """Return SIGNATURE_DIM, importing lazily to avoid circular imports."""
    global _SIGNATURE_DIM_CACHE
    if _SIGNATURE_DIM_CACHE is None:
        from agent.models.gnn.signature_encoder import SIGNATURE_DIM  # noqa: PLC0415

        _SIGNATURE_DIM_CACHE = SIGNATURE_DIM
    return _SIGNATURE_DIM_CACHE


def _compute_distributional_features(
    observations: list[dict[str, Any]],
) -> dict[str, float]:
    """Extract distributional features from an entity's observations.

    Returns: variance, min, max, iqr of observation values, number of
    distinct source tools, and obs_type frequency distribution.
    """
    import math

    values: list[float] = []
    tools: set[str] = set()
    obs_type_counts: dict[str, int] = {}

    for o in observations:
        # Collect value
        v = o.get("value", {})
        if isinstance(v, dict):
            for k in (
                "usd_amount",
                "btc_amount",
                "value",
                "estimated_value",
                "goldstein_scale",
                "num_articles",
            ):
                if k in v:
                    try:
                        values.append(float(v[k]))
                    except (TypeError, ValueError):
                        pass
                    break
        # Collect source tool
        tool = o.get("source_tool", "")
        if tool:
            tools.add(tool)
        # Collect obs_type — store returns "observation_type" (DB column name).
        # "obs_type" is a legacy alias kept for backward compat with any callers
        # that build obs dicts by hand.  Without this fallback the obs_type_dist
        # enrichment features (35 dims in ENRICHMENT_DIM) are always zero because
        # the key lookup silently returns "".
        ot = o.get("observation_type", "") or o.get("obs_type", "")
        obs_type_counts[ot] = obs_type_counts.get(ot, 0) + 1

    # Value distribution
    if len(values) >= 2:
        mean_v = sum(values) / len(values)
        variance = sum((x - mean_v) ** 2 for x in values) / (len(values) - 1)
        val_min = min(values)
        val_max = max(values)
        sorted_v = sorted(values)
        q1 = sorted_v[len(sorted_v) // 4]
        q3 = sorted_v[3 * len(sorted_v) // 4]
        iqr = q3 - q1
    elif len(values) == 1:
        variance = 0.0
        val_min = val_max = values[0]
        iqr = 0.0
    else:
        variance = val_min = val_max = iqr = 0.0

    # Obs type distribution (normalized to sum=1)
    total_obs = sum(obs_type_counts.values()) or 1
    obs_type_dist: dict[str, float] = {}
    for ot in OBSERVATION_TYPES:
        obs_type_dist[ot] = obs_type_counts.get(ot, 0) / total_obs

    return {
        "variance": variance,
        "min": val_min if math.isfinite(val_min) else 0.0,
        "max": val_max if math.isfinite(val_max) else 0.0,
        "iqr": iqr,
        "num_tools": float(len(tools)),
        "obs_type_dist": obs_type_dist,  # type: ignore[dict-item]
    }


def _compute_price_features(
    entity_id: str,
    observations: list[dict[str, Any]],
    current_time: float,
) -> list[float]:
    """Compute price-derived features for an instrument from instrument_daily obs.

    Features (9 dims, all computed from close prices / log_returns up to current_time):
        momentum_1m, momentum_3m, momentum_6m,
        volatility_20d, volatility_60d,
        avg_volume_20d, volume_trend,
        max_drawdown_60d, sharpe_60d

    All features are computed without forward-looking bias — only observations
    with observed_at <= current_time are used.  Returns zeros when insufficient data.
    """
    import math

    # Extract instrument_daily data for this entity, sorted by time
    daily_data: list[dict[str, float]] = []
    for o in observations:
        if o.get("entity_id") != entity_id:
            continue
        if o.get("observation_type") != "instrument_daily":
            continue
        ts = o.get("observed_at", 0.0)
        if ts > current_time:
            continue
        v = o.get("value", {})
        if not isinstance(v, dict):
            continue
        close = v.get("close")
        volume = v.get("volume")
        log_ret = v.get("log_return")
        if close is None:
            continue
        try:
            daily_data.append(
                {
                    "ts": float(ts),
                    "close": float(close),
                    "volume": float(volume) if volume is not None else 0.0,
                    "log_return": float(log_ret) if log_ret is not None else 0.0,
                }
            )
        except (TypeError, ValueError):
            continue

    if len(daily_data) < 2:
        return [0.0] * PRICE_FEAT_DIM

    daily_data.sort(key=lambda x: x["ts"])
    closes = [d["close"] for d in daily_data]
    log_rets = [d["log_return"] for d in daily_data]
    volumes = [d["volume"] for d in daily_data]
    n = len(closes)

    def _momentum(lookback: int) -> float:
        if n <= lookback:
            return 0.0
        past_close = closes[n - 1 - lookback]
        if abs(past_close) < 1e-8:
            return 0.0
        return (closes[-1] - past_close) / abs(past_close)

    def _volatility(lookback: int) -> float:
        window = log_rets[-min(lookback, n) :]
        if len(window) < 2:
            return 0.0
        mean = sum(window) / len(window)
        var = sum((r - mean) ** 2 for r in window) / (len(window) - 1)
        return math.sqrt(var) if var > 0 else 0.0

    def _avg_volume(lookback: int) -> float:
        window = volumes[-min(lookback, n) :]
        if not window:
            return 0.0
        return sum(window) / len(window)

    def _max_drawdown(lookback: int) -> float:
        window = closes[-min(lookback, n) :]
        if len(window) < 2:
            return 0.0
        peak = window[0]
        max_dd = 0.0
        for price in window[1:]:
            if price > peak:
                peak = price
            dd = (price - peak) / peak if peak > 0 else 0.0
            if dd < max_dd:
                max_dd = dd
        return max_dd

    def _sharpe(lookback: int) -> float:
        window = log_rets[-min(lookback, n) :]
        if len(window) < 2:
            return 0.0
        mean = sum(window) / len(window)
        var = sum((r - mean) ** 2 for r in window) / (len(window) - 1)
        if var < 1e-12:
            return 0.0
        return mean / math.sqrt(var)

    mom_1m = _momentum(21)
    mom_3m = _momentum(63)
    mom_6m = _momentum(126)
    vol_20d = _volatility(20)
    vol_60d = _volatility(60)
    avg_vol_20d = _avg_volume(20)
    avg_vol_60d = _avg_volume(60)
    vol_trend = (avg_vol_20d / avg_vol_60d - 1.0) if avg_vol_60d > 0 else 0.0
    max_dd_60d = _max_drawdown(60)
    sharpe_60d = _sharpe(60)

    features = [
        mom_1m,
        mom_3m,
        mom_6m,
        vol_20d,
        vol_60d,
        avg_vol_20d,
        vol_trend,
        max_dd_60d,
        sharpe_60d,
    ]
    # Clamp extreme values and replace NaN/Inf with 0
    result: list[float] = []
    for f in features:
        if not math.isfinite(f):
            result.append(0.0)
        else:
            result.append(max(-100.0, min(100.0, f)))
    return result


def _build_node_features(
    entity_type: str,
    entity_ids: list[str],
    observations: list[dict[str, Any]],
    current_time: float,
    enrichment: dict[str, dict[str, float]] | None = None,
    use_signatures: bool = False,
    ts2vec_embeddings: dict[str, dict[str, np.ndarray]] | None = None,
    ts2vec_dim: int = 0,
    *,
    zero_price_feats: bool = False,
) -> torch.Tensor:
    """Build feature matrix for one node type.

    Base features per node (dim = len(ENTITY_TYPES) + 3):
        [one_hot_entity_type..., obs_count, recency, mean_value]

    When enrichment is provided, additional features are appended:
        [cusum, hawkes, event_study, bocpd, variance, min, max, iqr,
         num_tools, obs_type_dist(18)] → 27 extra dims.

    When use_signatures=True (Idea 2), depth-3 path signatures are appended:
        [S^1(3 dims) + S^2(9 dims) + S^3(27 dims)] = 39 extra dims.
        Captures shape, curvature and texture of the event stream — a
        provably universal feature map (Lyons & McLeod 2022).

    When ts2vec_embeddings is provided (Idea 5), TS2Vec pretraining vectors
    are appended (ts2vec_dim extra dims per node).
    """
    type_idx = _ENTITY_TYPE_TO_IDX.get(entity_type)
    if type_idx is None:
        # Do NOT fall back to index 0. That silently one-hot-encodes the unknown
        # type as ENTITY_TYPES[0], so the model trains and scores it as a
        # different entity kind entirely — `maritime_area` was mislabelled as
        # `cftc_contract` this way for months behind a log line nobody read.
        #
        # Runtime discovery of genuinely-new types is a supported feature
        # (see GraphBuilder.build), so this stays non-fatal: encode an all-zero
        # type one-hot, which claims no identity rather than the wrong one.
        # Drift is caught loudly instead by validate_schema_against_store(),
        # which training and model loading call before touching weights.
        log.warning(
            "Unknown entity type %r — encoding all-zero type one-hot (claims no "
            "type). Register it in ENTITY_TYPES and retrain to give it an "
            "identity; see validate_schema_against_store().",
            entity_type,
        )
    type_dim = len(ENTITY_TYPES)
    _is_instrument = entity_type == "instrument"
    _price_dim = PRICE_FEAT_DIM if _is_instrument else 0
    _micro_dim = MICROSTRUCTURE_DIM if _is_instrument else 0
    _m15_dim = M15_QUANT_DIM if _is_instrument else 0
    _sig_dim = _signature_dim() if use_signatures else 0
    # TS2Vec extra dims: only non-zero if caller passed embeddings AND dim > 0
    _type_embs: dict[str, np.ndarray] | None = ts2vec_embeddings.get(entity_type) if ts2vec_embeddings else None
    _ts_dim = ts2vec_dim if (_type_embs is not None and ts2vec_dim > 0) else 0
    feat_dim = (
        BASE_FEAT_DIM
        + (ENRICHMENT_DIM if enrichment is not None else 0)
        + _price_dim
        + _micro_dim
        + _m15_dim
        + _sig_dim
        + _ts_dim
    )

    if not entity_ids:
        return torch.zeros(0, feat_dim)

    us_country_eid: str | None = None
    if _is_instrument:
        from agent.pipeline.entity import entity_id_from_key  # noqa: PLC0415

        us_country_eid = entity_id_from_key("country", "US")

    # Pre-group observations by entity_id for efficiency
    obs_by_entity: dict[str, list[dict[str, Any]]] = {}
    for o in observations:
        eid = o.get("entity_id", "")
        obs_by_entity.setdefault(eid, []).append(o)

    features = torch.zeros(len(entity_ids), feat_dim)
    for local_idx, eid in enumerate(entity_ids):
        # One-hot entity type. type_idx is None for an unregistered type — leave
        # the whole one-hot block zero rather than claiming ENTITY_TYPES[0].
        if type_idx is not None:
            features[local_idx, type_idx] = 1.0
        # Observation stats
        ent_obs = obs_by_entity.get(eid, [])
        stats = _compute_obs_stats(ent_obs, eid, current_time)
        features[local_idx, type_dim] = stats["count"]
        features[local_idx, type_dim + 1] = stats["recency"]
        features[local_idx, type_dim + 2] = stats["mean_value"]

        # Enrichment features (when provided)
        if enrichment is not None:
            offset = BASE_FEAT_DIM
            ent_enrich = enrichment.get(eid, {})
            # Statistical monitor states
            features[local_idx, offset] = ent_enrich.get("cusum", 0.0)
            features[local_idx, offset + 1] = ent_enrich.get("hawkes", 0.0)
            features[local_idx, offset + 2] = ent_enrich.get("event_study", 0.0)
            features[local_idx, offset + 3] = ent_enrich.get("bocpd", 0.0)
            # Distributional features from observations
            dist_feats = _compute_distributional_features(ent_obs)
            features[local_idx, offset + 4] = dist_feats["variance"]
            features[local_idx, offset + 5] = dist_feats["min"]
            features[local_idx, offset + 6] = dist_feats["max"]
            features[local_idx, offset + 7] = dist_feats["iqr"]
            features[local_idx, offset + 8] = dist_feats["num_tools"]
            # Obs type distribution — one slot per OBSERVATION_TYPES entry.
            # ENRICHMENT_DIM is derived from the same list, so this can never
            # run past the allocated block (see the ENRICHMENT_DIM comment).
            for ot_idx, ot_name in enumerate(OBSERVATION_TYPES):
                features[local_idx, offset + _ENRICHMENT_SCALAR_DIM + ot_idx] = dist_feats["obs_type_dist"][ot_name]

        # Price-derived features (instruments only — Phase 50)
        if _is_instrument:
            price_offset = BASE_FEAT_DIM + (ENRICHMENT_DIM if enrichment is not None else 0)
            if zero_price_feats:
                price_feats = [0.0] * PRICE_FEAT_DIM
            else:
                price_feats = _compute_price_features(eid, observations, current_time)
            for pf_idx, pf_val in enumerate(price_feats):
                features[local_idx, price_offset + pf_idx] = pf_val

        # Microstructure features (M9 — instruments only, daily instrument_daily)
        if _is_instrument:
            from agent.quant.microstructure_signals import (  # noqa: PLC0415
                compute_gnn_micro_features,
            )

            micro_offset = BASE_FEAT_DIM + (ENRICHMENT_DIM if enrichment is not None else 0) + _price_dim
            micro_feats = compute_gnn_micro_features(eid, observations, current_time)
            for mf_idx, mf_val in enumerate(micro_feats):
                features[local_idx, micro_offset + mf_idx] = mf_val

        # M14.1 — options / rates / dividends (M15 pipeline data)
        if _is_instrument:
            from agent.quant.gnn_quant_features import (  # noqa: PLC0415
                _latest_close,
                compute_gnn_m15_features,
            )

            m15_offset = BASE_FEAT_DIM + (ENRICHMENT_DIM if enrichment is not None else 0) + _price_dim + _micro_dim
            spot = _latest_close(eid, observations, current_time)
            m15_feats = compute_gnn_m15_features(
                eid,
                observations,
                current_time,
                spot=spot,
                us_country_eid=us_country_eid,
            )
            for i, val in enumerate(m15_feats):
                features[local_idx, m15_offset + i] = val

        # Path signature features (Idea 2 — optional)
        if use_signatures and ent_obs:
            from agent.models.gnn.signature_encoder import (
                compute_entity_signature,
            )

            sig_offset = (
                BASE_FEAT_DIM + (ENRICHMENT_DIM if enrichment is not None else 0) + _price_dim + _micro_dim + _m15_dim
            )
            sig = compute_entity_signature(ent_obs)
            sig_len = min(sig.size(0), _sig_dim)
            features[local_idx, sig_offset : sig_offset + sig_len] = sig[:sig_len]

        # TS2Vec pretraining features (Idea 5 — optional)
        if _ts_dim > 0 and _type_embs is not None:
            ts_offset = (
                BASE_FEAT_DIM
                + (ENRICHMENT_DIM if enrichment is not None else 0)
                + _price_dim
                + _micro_dim
                + _m15_dim
                + _sig_dim
            )
            emb = _type_embs.get(eid)
            if emb is not None:
                emb_t = torch.tensor(emb[:_ts_dim], dtype=torch.float32)
                fill_len = min(emb_t.size(0), _ts_dim)
                features[local_idx, ts_offset : ts_offset + fill_len] = emb_t[:fill_len]

    return features


# ── Edge builder ───────────────────────────────────────────────


def _build_edge_data(
    links: list[dict[str, Any]],
    id_map: IDMap,
    reference_time: float | None = None,
) -> dict[tuple[str, str, str], dict[str, torch.Tensor]]:
    """Group entity_links by (src_type, link_type, dst_type) and build tensors.

    Returns dict mapping edge-type triplet → {"edge_index": [2,E], "edge_attr": [E,2]}
    edge_attr columns: [confidence, age_days]

    Args:
        reference_time: the "now" of the window being built. `age_days` is
            measured against this, NOT wall-clock. Using wall-clock leaks the
            present into every historical snapshot: replaying a 2023 window in
            2026 stamped every edge as ~1000 days old, a value the model could
            never have observed at that point in time. Defaults to wall-clock
            only for live (non-replay) callers that pass nothing.
    """
    import time as _time

    now = _time.time() if reference_time is None else reference_time
    grouped: dict[tuple[str, str, str], dict[str, list[Any]]] = {}

    for link in links:
        eid_a = link.get("entity_id_a", "")
        eid_b = link.get("entity_id_b", "")
        ltype = link.get("link_type", "unknown")

        # Look up typed info from id_map
        gid_a = None
        gid_b = None
        type_a = type_b = None
        for (etype, eid), gid in id_map.typed_to_global.items():
            if eid == eid_a and gid_a is None:
                gid_a = gid
                type_a = etype
            if eid == eid_b and gid_b is None:
                gid_b = gid
                type_b = etype
            if gid_a is not None and gid_b is not None:
                break

        if type_a is None or type_b is None:
            log.debug("Skipping link %s→%s: entity not in id_map", eid_a, eid_b)
            continue

        local_a = id_map.local_id(type_a, eid_a)
        local_b = id_map.local_id(type_b, eid_b)
        if local_a is None or local_b is None:
            continue

        triplet = (type_a, ltype, type_b)
        bucket = grouped.setdefault(triplet, {"src": [], "dst": [], "conf": [], "age": []})
        bucket["src"].append(local_a)
        bucket["dst"].append(local_b)
        bucket["conf"].append(link.get("confidence", 1.0))
        created = link.get("created_at", now)
        age_days = max(0.0, (now - created) / 86400.0)
        bucket["age"].append(age_days)

    result: dict[tuple[str, str, str], dict[str, torch.Tensor]] = {}
    for triplet, bucket in grouped.items():
        src = torch.tensor(bucket["src"], dtype=torch.long)
        dst = torch.tensor(bucket["dst"], dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)
        edge_attr = torch.tensor(
            list(zip(bucket["conf"], bucket["age"])),
            dtype=torch.float,
        )
        result[triplet] = {"edge_index": edge_index, "edge_attr": edge_attr}

    return result


# ── GraphBuilder ───────────────────────────────────────────────


class GraphBuilder:
    """Converts PipelineStore into PyG HeteroData + ID mappings.

    Usage::

        builder = GraphBuilder(store)
        data, id_map, events = builder.build()
    """

    def __init__(
        self,
        store: PipelineStore,
        *,
        zero_price_feats: bool = False,
    ) -> None:
        self._store = store
        self._zero_price_feats = zero_price_feats

    def build(
        self,
        *,
        since: float | None = None,
        until: float | None = None,
        enrichment: dict[str, dict[str, float]] | None = None,
        use_signatures: bool = False,
        ts2vec_embeddings: dict | None = None,
        ts2vec_dim: int = 0,
    ) -> tuple[HeteroData, IDMap, list[dict[str, Any]]]:
        """Build the full heterogeneous graph.

        Args:
            since: Only include observations from this timestamp onward.
            until: Only include observations up to this timestamp.
            enrichment: Optional per-entity enrichment features.
                Maps entity_id → {"cusum": float, "hawkes": float,
                "event_study": float, "bocpd": float}.
                When provided, node features expand from 12d to 39d.
            ts2vec_embeddings: Optional TS2Vec pretraining embeddings
                (Idea 5).  Maps entity_type → {entity_id → ndarray}.
                When provided, ts2vec_dim extra dims are appended to
                every node's feature vector.
            ts2vec_dim: Embedding dimension added per node when
                ts2vec_embeddings is provided.  0 = disabled.

        Returns:
            (HeteroData, IDMap, events) where events is a time-sorted list
            of observation dicts for the trainer to consume sequentially.
        """
        # 1. Fetch all raw data
        entities = self._store.query_all_entities()
        observations = self._store.query_all_observations(since=since, until=until)
        # Links MUST be future-blind. Observations are filtered by `until`, but
        # links were not — so every historical snapshot received the complete
        # present-day link set (all 16,870 of them, including 2023 windows).
        # The model saw relationships that had not yet been discovered at the
        # time it was supposedly predicting, which inflates any backtest and
        # voids the eval entirely (LESSONS.md F-04).
        links = _links_as_of(self._store.query_all_entity_links(), until)

        # 2. Build ID map
        id_map = IDMap()
        for ent in entities:
            id_map.add(ent["entity_type"], ent["entity_id"])

        # 3. Determine current_time for recency features (sanitized)
        current_time = _reference_time(observations)

        # 4. Build HeteroData
        data = HeteroData()

        # Node features per type — iterate all types present in id_map,
        # not just ENTITY_TYPES, so dynamically-added types get features too.
        all_types = sorted(set(ENTITY_TYPES) | set(id_map.type_local.keys()))
        for etype in all_types:
            local_map = id_map.type_local.get(etype, {})
            if not local_map:
                continue
            # Reconstruct ordered list by local index
            ordered_ids = [""] * len(local_map)
            for eid, lidx in local_map.items():
                ordered_ids[lidx] = eid
            features = _build_node_features(
                etype,
                ordered_ids,
                observations,
                current_time,
                enrichment=enrichment,
                use_signatures=use_signatures,
                ts2vec_embeddings=ts2vec_embeddings,
                ts2vec_dim=ts2vec_dim,
                zero_price_feats=self._zero_price_feats,
            )
            data[etype].x = features
            data[etype].node_ids = ordered_ids

        # Edge data per (src_type, link_type, dst_type).
        # `current_time` is the window's reference clock, so edge age is
        # measured as of the snapshot rather than as of today (F-04).
        edge_data = _build_edge_data(links, id_map, reference_time=current_time)
        for triplet, tensors in edge_data.items():
            data[triplet].edge_index = tensors["edge_index"]
            data[triplet].edge_attr = tensors["edge_attr"]

        # 5. Events = observations sorted by time (for sequential training)
        events = sorted(observations, key=lambda o: o.get("observed_at", 0.0))

        node_ct = sum(id_map.num_nodes_of_type(t) for t in all_types)
        edge_ct = sum(tensors["edge_index"].size(1) for tensors in edge_data.values())
        log.info(
            "Built HeteroData: %d node types, %d nodes, %d edge types, %d edges, %d events",
            len([t for t in all_types if id_map.num_nodes_of_type(t) > 0]),
            node_ct,
            len(edge_data),
            edge_ct,
            len(events),
        )

        return data, id_map, events

    def prepare_static(
        self,
    ) -> tuple[IDMap, list[dict[str, Any]], list[dict[str, Any]]]:
        """Pre-fetch entities and links for caching across multiple build() calls.

        Returns:
            (id_map, entities, links) — reusable across windows that share
            the same entity/link set.
        """
        entities = self._store.query_all_entities()
        links = self._store.query_all_entity_links()
        id_map = IDMap()
        for ent in entities:
            id_map.add(ent["entity_type"], ent["entity_id"])
        return id_map, entities, links

    def prefetch_observations(
        self,
        *,
        since: float | None = None,
        until: float | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch all observations once, sorted by time.

        Use with build_from_cached(observations=...) to avoid per-window
        DB queries. Caller can slice by time using bisect.
        """
        obs = self._store.query_all_observations(since=since, until=until)
        obs.sort(key=lambda o: o.get("observed_at", 0.0))
        return obs

    def build_from_cached(
        self,
        id_map: IDMap,
        links: list[dict[str, Any]],
        *,
        since: float | None = None,
        until: float | None = None,
        observations: list[dict[str, Any]] | None = None,
        enrichment: dict[str, dict[str, float]] | None = None,
        use_signatures: bool = False,
        ts2vec_embeddings: dict | None = None,
        ts2vec_dim: int = 0,
    ) -> tuple[HeteroData, IDMap, list[dict[str, Any]]]:
        """Build graph reusing pre-fetched entities/links (skips 2 of 3 DB queries).

        Use with prepare_static() for training loops where entities/links
        are constant across windows.

        If ``observations`` is provided, skips the DB observation query too
        (caller is responsible for time-filtering).
        """
        if observations is None:
            observations = self._store.query_all_observations(since=since, until=until)

        current_time = _reference_time(observations or [])

        # Same future-blindness requirement as build(). This is the path the
        # TRAINING loop uses (via prepare_static), so leakage here contaminates
        # the model itself rather than just an eval — the caller pre-fetches
        # links once and reuses them across every window, which is exactly how
        # present-day edges ended up inside 2023 snapshots.
        links = _links_as_of(links, until)

        data = HeteroData()

        all_types = sorted(set(ENTITY_TYPES) | set(id_map.type_local.keys()))
        for etype in all_types:
            local_map = id_map.type_local.get(etype, {})
            if not local_map:
                continue
            ordered_ids = [""] * len(local_map)
            for eid, lidx in local_map.items():
                ordered_ids[lidx] = eid
            features = _build_node_features(
                etype,
                ordered_ids,
                observations,
                current_time,
                enrichment=enrichment,
                use_signatures=use_signatures,
                ts2vec_embeddings=ts2vec_embeddings,
                ts2vec_dim=ts2vec_dim,
                zero_price_feats=self._zero_price_feats,
            )
            data[etype].x = features
            data[etype].node_ids = ordered_ids

        edge_data = _build_edge_data(links, id_map, reference_time=current_time)
        for triplet, tensors in edge_data.items():
            data[triplet].edge_index = tensors["edge_index"]
            data[triplet].edge_attr = tensors["edge_attr"]

        events = sorted(observations, key=lambda o: o.get("observed_at", 0.0))

        return data, id_map, events
