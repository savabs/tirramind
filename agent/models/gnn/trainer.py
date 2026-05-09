"""
TirraMind — Self-Supervised Training & Outcome Fine-Tuning for HetTGN
(Phases 12d, 15a–c, 19a)

Provides:
    SyntheticGraphGenerator — Generates realistic entity graphs with
                              injected known temporal patterns for
                              training loop development and validation.
    TrainerConfig           — Hyperparameter dataclass.
    Trainer                 — Walk-forward training loop with:
                              (1) next-event obs_type prediction (CE)
                              (2) next-event time_delta prediction (MSE)
                              (3) contrastive link loss (margin)
                              Plus: infer(), save_model(), load_model()
                              for production inference (Phase 19a).
    evaluate()              — Walk-forward evaluation (no leakage).
    OutcomeLabel            — Binary co-occurrence label for fine-tuning.
    generate_outcome_labels — Create outcome labels from CrystallizedPatterns.
    FineTuner               — Supervised fine-tuning loop (Phase 15c).
    evaluate_supervised()   — AUROC, precision, recall, F1 (Phase 15c).

Self-supervised signal:
    Given an entity graph snapshot up to time T, predict what happens
    next: which entity gets an observation, what type, and when.
    No market outcome label required for pre-training.

Supervised signal (Phase 15):
    For each CrystallizedPattern, label (src, dst) entity pairs as
    positive when target obs_type occurs within window after source
    obs_type, negative otherwise.  Fine-tune the supervised bilinear
    head on these labels.

References:
    TGN (Rossi et al. 2020, arXiv:2006.10637) — temporal training recipe.
    InfoNCE contrastive loss — linked entity pairs vs random negatives.
    Pre-training GNNs (Hu et al. ICLR 2020, arXiv:1905.12265) — two-phase.
    Spec steps: 12d.1–12d.3, 15a.1–15c.4.
"""

from __future__ import annotations

import bisect
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm as _tqdm

    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from agent.models.gnn.alignment import load_alignment_weights
from agent.models.gnn.ewc import EWCState, compute_fisher, ewc_penalty
from agent.models.gnn.graph_builder import (
    OBSERVATION_TYPES,
    GraphBuilder,
    IDMap,
)
from agent.models.gnn.het_tgn import HetTGN
from agent.pipeline.entity import entity_id_from_key
from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# SyntheticGraphGenerator
# ═══════════════════════════════════════════════════════════════


@dataclass
class InjectedPattern:
    """A known temporal pattern to inject for validation.

    Whenever an entity of ``source_type`` receives an observation of
    ``source_obs_type``, an entity of ``target_type`` (linked via
    ``via_edge``) will receive ``target_obs_type`` within
    ``lag_seconds`` ± ``lag_jitter`` seconds afterward.
    """

    source_type: str
    source_obs_type: str
    target_type: str
    target_obs_type: str
    via_edge: str
    lag_seconds: float = 3600.0
    lag_jitter: float = 600.0


# ═══════════════════════════════════════════════════════════════
# ListNet ranking loss (Phase 41b)
# ═══════════════════════════════════════════════════════════════


def _listnet_loss(
    scores: torch.Tensor, targets: torch.Tensor, tau: float = 1.0
) -> torch.Tensor:
    """ListNet top-1 approximation (Cao et al. 2007, ICML).

    Minimises KL(p_target || p_pred) where p = softmax(x / tau).
    Directly optimises cross-sectional rank ordering (IC metric),
    unlike Huber/MSE which minimise absolute prediction error and
    allow IC=0 (predict-the-mean) as a valid optimum.

    Args:
        scores:  Model predicted scores, shape (N,).  N must be >= 2.
        targets: Realised returns (or any continuous ranking target), shape (N,).
        tau:     Softmax temperature. 1.0 = standard; lower = harder ranking.

    Returns:
        Scalar loss tensor (non-negative).

    Reference:
        Cao et al. 2007 "Learning to Rank: From Pairwise Approach to
        Listwise Approach" — top-1 probability formulation, ICML.
    """
    p_target = F.softmax(targets / tau, dim=0)
    log_p_pred = F.log_softmax(scores / tau, dim=0)
    return -(p_target * log_p_pred).sum()


class SyntheticGraphGenerator:
    """Generate synthetic entity graphs with known temporal patterns.

    The generated data is inserted directly into a PipelineStore,
    which can then be consumed by GraphBuilder and HetTGN.

    Covers all 11 entity types, 45 observation types, and 21 link
    types in the TirraMind schema (expanded in Phase 36).
    """

    def __init__(
        self,
        num_companies: int = 8,
        num_countries: int = 5,
        num_vessels: int = 4,
        num_wallets: int = 4,
        num_instruments: int = 0,
        num_persons: int = 0,
        num_cftc_contracts: int = 0,
        num_organizations: int = 0,
        num_protocols: int = 0,
        num_topics: int = 3,
        num_domains: int = 3,
        time_span: float = 86400.0 * 30,  # 30 days
        base_event_rate: float = 0.001,  # events per entity per second
        seed: int = 42,
        patterns: list[InjectedPattern] | None = None,
    ) -> None:
        self.num_entities = {
            k: v
            for k, v in {
                "company": num_companies,
                "country": num_countries,
                "vessel": num_vessels,
                "wallet": num_wallets,
                "instrument": num_instruments,
                "person": num_persons,
                "cftc_contract": num_cftc_contracts,
                "organization": num_organizations,
                "protocol": num_protocols,
                "topic": num_topics,
                "domain": num_domains,
            }.items()
            if v > 0
        }
        self.time_span = time_span
        self.base_event_rate = base_event_rate
        self.seed = seed
        self.patterns = patterns or []
        self._rng = random.Random(seed)

    def generate(self, store: PipelineStore) -> dict[str, Any]:
        """Populate the store with synthetic entities, links, and observations.

        Returns metadata about generated data: entity counts, link counts,
        observation count, and ground truth pattern instances.
        """
        self._rng = random.Random(self.seed)
        entities: dict[str, list[str]] = {}
        stats: dict[str, Any] = {
            "entities": {},
            "links": 0,
            "observations": 0,
            "pattern_instances": [],
        }

        # ── Create entities ──────────────────────────────
        for etype, count in self.num_entities.items():
            ids = []
            for i in range(count):
                name = f"{etype}_{i}"
                eid = entity_id_from_key(etype, name)
                store.register_entity(etype, name, eid)
                ids.append(eid)
            entities[etype] = ids
            stats["entities"][etype] = count

        # ── Create links ─────────────────────────────────
        link_count = 0
        # company → country (headquartered_in)
        for cid in entities.get("company", []):
            co_id = self._rng.choice(entities.get("country", ["default"]))
            store.link_entities(
                cid, co_id, "headquartered_in", "synthetic", confidence=0.9
            )
            link_count += 1

        # company → country (operates_in) — subset of companies
        for cid in entities.get("company", [])[
            : max(1, len(entities.get("company", [])) // 2)
        ]:
            co_id = self._rng.choice(entities.get("country", ["default"]))
            store.link_entities(cid, co_id, "operates_in", "synthetic", confidence=0.8)
            link_count += 1

        # company → country (market_authorized_in) — pharma companies
        for cid in entities.get("company", [])[
            : max(1, len(entities.get("company", [])) // 3)
        ]:
            co_id = self._rng.choice(entities.get("country", ["default"]))
            store.link_entities(
                cid, co_id, "market_authorized_in", "synthetic", confidence=0.85
            )
            link_count += 1

        # company → company (lobbies_for) — some business relationships
        comps = entities.get("company", [])
        for i in range(min(3, len(comps) - 1)):
            store.link_entities(
                comps[i], comps[i + 1], "lobbies_for", "synthetic", confidence=0.7
            )
            link_count += 1

        # company → company (debtor_of) — creditor relationships
        for i in range(min(2, len(comps) - 1)):
            src, tgt = comps[i], comps[-(i + 1)]
            if src == tgt:
                continue
            store.link_entities(src, tgt, "debtor_of", "synthetic", confidence=0.75)
            link_count += 1

        # company → organization (awarded_by) — government contracts
        for cid in entities.get("company", [])[
            : max(1, len(entities.get("company", [])) // 2)
        ]:
            if entities.get("organization"):
                org_id = self._rng.choice(entities["organization"])
                store.link_entities(
                    cid, org_id, "awarded_by", "synthetic", confidence=0.9
                )
                link_count += 1

        # person → company (works_for)
        for pid in entities.get("person", []):
            if entities.get("company"):
                cid = self._rng.choice(entities["company"])
                store.link_entities(pid, cid, "works_for", "synthetic", confidence=0.95)
                link_count += 1

        # vessel → country (port_call_to)
        for vid in entities.get("vessel", []):
            co_id = self._rng.choice(entities.get("country", ["default"]))
            store.link_entities(vid, co_id, "port_call_to", "synthetic", confidence=0.8)
            link_count += 1

        # wallet → country (exchange_based_in)
        for wid in entities.get("wallet", []):
            co_id = self._rng.choice(entities.get("country", ["default"]))
            store.link_entities(
                wid, co_id, "exchange_based_in", "synthetic", confidence=0.7
            )
            link_count += 1

        # wallet → wallet (transacts_with) — inter-wallet transfers
        wallets = entities.get("wallet", [])
        for i in range(min(3, len(wallets) - 1)):
            store.link_entities(
                wallets[i],
                wallets[i + 1],
                "transacts_with",
                "synthetic",
                confidence=0.8,
            )
            link_count += 1

        # wallet → instrument (trades_instrument) — crypto wallets
        for wid in entities.get("wallet", []):
            if entities.get("instrument"):
                # Pick a crypto-like instrument
                inst_id = self._rng.choice(entities["instrument"])
                store.link_entities(
                    wid, inst_id, "trades_instrument", "synthetic", confidence=0.75
                )
                link_count += 1

        # instrument → company (tracks_issuer) — stocks/ETFs
        for inst_id in entities.get("instrument", [])[
            : max(1, len(entities.get("instrument", [])) * 2 // 3)
        ]:
            if entities.get("company"):
                cid = self._rng.choice(entities["company"])
                store.link_entities(
                    inst_id, cid, "tracks_issuer", "synthetic", confidence=0.95
                )
                link_count += 1

        # instrument → country (located_in) — domicile
        for inst_id in entities.get("instrument", []):
            co_id = self._rng.choice(entities.get("country", ["default"]))
            store.link_entities(
                inst_id, co_id, "located_in", "synthetic", confidence=0.9
            )
            link_count += 1

        # instrument → country (fx_base_country / fx_quote_country) — subset
        fx_insts = entities.get("instrument", [])[
            : max(1, len(entities.get("instrument", [])) // 3)
        ]
        for inst_id in fx_insts:
            countries = entities.get("country", [])
            if len(countries) >= 2:
                base_co, quote_co = self._rng.sample(countries, 2)
                store.link_entities(
                    inst_id, base_co, "fx_base_country", "synthetic", confidence=0.95
                )
                store.link_entities(
                    inst_id, quote_co, "fx_quote_country", "synthetic", confidence=0.95
                )
                link_count += 2

        # instrument → country (exchange_country) — commodity futures
        for inst_id in entities.get("instrument", [])[
            -max(1, len(entities.get("instrument", [])) // 3) :
        ]:
            co_id = self._rng.choice(entities.get("country", ["default"]))
            store.link_entities(
                inst_id, co_id, "exchange_country", "synthetic", confidence=0.95
            )
            link_count += 1

        # instrument → protocol (tracks_protocol) — crypto instruments
        for inst_id in entities.get("instrument", [])[
            : max(1, len(entities.get("instrument", [])) // 5)
        ]:
            if entities.get("protocol"):
                proto_id = self._rng.choice(entities["protocol"])
                store.link_entities(
                    inst_id, proto_id, "tracks_protocol", "synthetic", confidence=0.9
                )
                link_count += 1

        # cftc_contract → instrument (cftc_tracks)
        for cid in entities.get("cftc_contract", []):
            if entities.get("instrument"):
                inst_id = self._rng.choice(entities["instrument"])
                store.link_entities(
                    cid, inst_id, "cftc_tracks", "synthetic", confidence=0.95
                )
                link_count += 1

        # country → country (sanctioned_under) — geopolitical
        countries = entities.get("country", [])
        if len(countries) >= 2:
            # 1-2 sanction relationships
            for _ in range(min(2, len(countries) - 1)):
                a, b = self._rng.sample(countries, 2)
                store.link_entities(
                    a, b, "sanctioned_under", "synthetic", confidence=0.85
                )
                link_count += 1

        # domain → company (domain_owned_by) — Phase 36
        for did in entities.get("domain", []):
            if entities.get("company"):
                cid = self._rng.choice(entities["company"])
                store.link_entities(
                    did, cid, "domain_owned_by", "synthetic", confidence=0.8
                )
                link_count += 1

        # topic → instrument (topic_relates_to_instrument) — Phase 36
        for tid in entities.get("topic", []):
            if entities.get("instrument"):
                # Each topic links to 1-2 instruments
                n_links = min(2, len(entities["instrument"]))
                targets = self._rng.sample(entities["instrument"], n_links)
                for inst_id in targets:
                    store.link_entities(
                        tid,
                        inst_id,
                        "topic_relates_to_instrument",
                        "synthetic",
                        confidence=0.7,
                    )
                    link_count += 1

        stats["links"] = link_count

        # ── Generate base observations (Poisson-like) ────
        all_obs: list[dict] = []
        t_start = 0.0
        t_end = self.time_span

        for etype, eids in entities.items():
            # Assign plausible obs types per entity type
            type_obs_map = self._obs_types_for(etype)
            for eid in eids:
                t = t_start
                while t < t_end:
                    dt = self._rng.expovariate(self.base_event_rate)
                    t += dt
                    if t >= t_end:
                        break
                    obs_type = self._rng.choice(type_obs_map)
                    value = {"synthetic": True, "amount": self._rng.uniform(100, 10000)}
                    store.store_entity_observation(
                        entity_id=eid,
                        source_tool="synthetic",
                        observed_at=t,
                        observation_type=obs_type,
                        value=value,
                    )
                    all_obs.append(
                        {
                            "entity_id": eid,
                            "entity_type": etype,
                            "observed_at": t,
                            "observation_type": obs_type,
                        }
                    )

        # ── Inject patterns ──────────────────────────────
        # Build link index for pattern injection
        links = store.query_all_entity_links()
        link_index: dict[str, list[dict]] = {}
        for lnk in links:
            key = (lnk["entity_id_a"], lnk["link_type"])
            link_index.setdefault(key, []).append(lnk)

        # Build entity type index
        all_entities = store.query_all_entities()
        entity_type_map: dict[str, str] = {
            e["entity_id"]: e["entity_type"] for e in all_entities
        }

        for pattern in self.patterns:
            # Find source observations matching the pattern
            source_obs = [
                o
                for o in all_obs
                if o["entity_type"] == pattern.source_type
                and o["observation_type"] == pattern.source_obs_type
            ]
            for src_ob in source_obs:
                src_id = src_ob["entity_id"]
                # Find linked targets via the pattern edge type
                linked = link_index.get((src_id, pattern.via_edge), [])
                for lnk in linked:
                    tgt_id = lnk["entity_id_b"]
                    tgt_type = entity_type_map.get(tgt_id, "")
                    if tgt_type != pattern.target_type:
                        continue
                    # Inject correlated target observation
                    lag = pattern.lag_seconds + self._rng.gauss(0, pattern.lag_jitter)
                    lag = max(1.0, lag)  # ensure positive
                    tgt_time = src_ob["observed_at"] + lag
                    if tgt_time > t_end:
                        continue
                    tgt_value = {
                        "synthetic": True,
                        "injected_pattern": True,
                        "source_entity": src_id,
                    }
                    store.store_entity_observation(
                        entity_id=tgt_id,
                        source_tool="synthetic_pattern",
                        observed_at=tgt_time,
                        observation_type=pattern.target_obs_type,
                        value=tgt_value,
                    )
                    stats["pattern_instances"].append(
                        {
                            "pattern": pattern,
                            "source_entity": src_id,
                            "target_entity": tgt_id,
                            "source_time": src_ob["observed_at"],
                            "target_time": tgt_time,
                            "actual_lag": tgt_time - src_ob["observed_at"],
                        }
                    )

        all_obs_final = store.query_all_observations()
        stats["observations"] = len(all_obs_final)

        return stats

    @staticmethod
    def _obs_types_for(entity_type: str) -> list[str]:
        """Return plausible observation types for an entity type.

        Covers all 45 OBSERVATION_TYPES across all 11 ENTITY_TYPES.
        Mapping reflects real tool outputs from the L2 expansion
        (Phases 17–34).
        """
        mapping: dict[str, list[str]] = {
            "cftc_contract": ["futures_positioning"],
            "company": [
                "insider_trade",
                "form144_filing",
                "sell_intent",
                "patent_filing",
                "contract_award",
                "creditor_filing",
                "lobbying_spend",
                "drug_approval",
                "short_interest",
                "bankruptcy_status",
                "investigation_signal",
            ],
            "country": [
                "geopolitical_event",
                "sanctions_listing",
                "cb_balance_sheet",
                "cb_policy_rate",
                "economic_activity",
                "capital_flow",
                "sovereign_yield",
                "consumer_confidence",
                "food_security",
                "internet_disruption",
                "migration_pressure",
                "trade_flow",
                "border_throughput",
                "pathogen_level",
                "campaign_finance",
                "grid_demand",
            ],
            "domain": ["cert_issued", "dns_change"],
            "instrument": [
                "instrument_return",
                "instrument_volatility",
                "instrument_volume",
                "price_movement",
            ],
            "organization": ["regulatory_velocity", "contract_award"],
            "person": ["insider_trade", "sell_intent", "campaign_finance"],
            "protocol": ["tvl_change"],
            "topic": [
                "pageview_spike",
                "market_probability",
                "research_velocity",
                "price_movement",
            ],
            "vessel": ["port_call", "vessel_position"],
            "wallet": ["btc_transfer", "whale_trade"],
        }
        return mapping.get(entity_type, ["cross_entity_pattern"])


# ═══════════════════════════════════════════════════════════════
# TrainerConfig
# ═══════════════════════════════════════════════════════════════


@dataclass
class TrainerConfig:
    """Hyperparameters for HetTGN self-supervised training."""

    hidden_dim: int = 64
    memory_dim: int = 64
    message_dim: int = 64
    time_dim: int = 16
    num_heads: int = 2
    num_layers: int = 2
    learning_rate: float = 1e-3
    epochs: int = 10
    window_size: float = 86400.0  # 1 day
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    # test_ratio = 1 - train_ratio - val_ratio
    obs_type_weight: float = 1.0
    time_delta_weight: float = 0.1
    contrastive_weight: float = 0.5
    contrastive_margin: float = 1.0
    num_negative_samples: int = 5
    value_weight: float = 0.3
    return_weight: float = 1.0
    """Weight for the instrument log_return auxiliary loss (Phase 41).
    Higher than value_weight because this is the primary financial signal
    we want the embedding to encode. Set to 0.0 to disable."""
    use_listnet_return_loss: bool = False
    """When True, replace Huber return loss with ListNet cross-entropy ranking
    loss (Cao et al. 2007, ICML).  Requires >= 2 instrument observations per
    window; windows with < 2 finite-return instruments are skipped for the
    return loss.  Directly optimises cross-sectional IC (Spearman rank
    correlation) rather than minimising absolute prediction error."""
    listnet_temperature: float = 1.0
    """Softmax temperature tau for ListNet.  Higher = softer target distribution
    (less peaked on best instrument).  1.0 is standard; lower values approach
    hard argmax ranking.  Has no effect when use_listnet_return_loss=False."""
    gdelt_subsample_frac: float = 1.0
    """Fraction of geopolitical_event observations to keep during training.
    GDELT makes up ~92% of the DB (901K rows) and causes OOM during snapshot
    pre-building. Set to e.g. 0.05 to keep 5% (~45K) while retaining all
    other observation types. Applied once after prefetch_observations()."""
    max_windows: int = 0
    """Cap the number of training windows per epoch. 0 = use all windows.
    Takes the LAST N windows (most recent data). Bounds peak RAM to
    O(N * avg_graph_size). Recommended: 200 for <4 GB RAM usage."""
    auto_tune_loss_weights: bool = False
    """When True, use learnable uncertainty-based loss weighting
    (Kendall et al. 2018 "Multi-Task Learning Using Uncertainty
    to Weigh Losses") instead of fixed config weights."""
    log_var_min: float = -3.0
    """Lower clamp for the learnable log-variance parameters s_k = ln(sigma_k^2)
    used by the uncertainty-weighted loss. Effective task weight is
    exp(-s_k), so this bounds the weight ABOVE at exp(-log_var_min).
    Prevents the Phase 40 divergence where s_k -> -inf when a component
    loss reaches 0 (see Kendall et al. 2018, Liebel & Körner 2018)."""
    log_var_max: float = 3.0
    """Upper clamp for s_k = ln(sigma_k^2). Bounds the effective weight
    BELOW at exp(-log_var_max)."""
    return_log_var_max: float = 0.0
    """Tighter upper clamp for the return component log-variance only.
    Defaults to 0.0 → effective return weight ≥ exp(-0) = 1.0, preventing
    auto-tune from silencing the return head. Set to log_var_max (3.0)
    to disable the special handling and treat return like other losses."""
    obs_since: float | None = None
    """When set, exclude observations with observed_at < obs_since
    from training/val/test splits. Useful for skipping sparse early
    data (e.g. GDELT 1970-era timestamps)."""
    ewc_lambda: float = 1000.0
    """EWC regularisation strength λ (Kirkpatrick et al. 2017).
    L_total = L_new + λ · Σ F_i (θ_i − θ_i*)²
    Higher → more conservative (less forgetting, less plasticity).
    1000.0 is the value used in the original paper for Permuted MNIST;
    the correct value for this domain should be validated empirically
    once Phase 47 historical data is available."""
    online_batch_threshold: int = 100
    device: str = "cpu"
    """Torch device string for model and data tensors.  Use 'cuda' to
    enable GPU acceleration (recommended when a CUDA GPU is available).
    Use 'cpu' when no GPU is present or for debugging."""
    """Minimum number of new observations accumulated since the last
    online_update (or full retrain) before the DAG operator triggers
    another EWC gradient step.  Set higher to reduce compute overhead;
    set lower to adapt more frequently to incoming data streams."""
    checkpoint_dir: str | None = None
    """Directory to save per-epoch checkpoints (epoch_001.pt, epoch_002.pt …).
    If None, no per-epoch checkpoints are written."""
    resume_from_epoch: int = 0
    """Skip the first N epochs by loading the checkpoint for epoch N from
    checkpoint_dir before training begins.  0 = start from scratch."""
    use_direction_loss: bool = False
    """When True, add a binary direction (sign) cross-entropy loss alongside
    the return loss.  Loss formulation: treat the predicted scalar return as a
    logit and apply BCE(sigmoid(pred), (target > 0).float()).  This directly
    penalises sign errors independently of magnitude, providing a complementary
    gradient to ListNet (which optimises ranking order) and Huber (which
    minimises absolute error).  Weight controlled by direction_loss_weight."""
    direction_loss_weight: float = 0.3
    """Multiplier for the direction BCE loss when use_direction_loss=True.
    Added as: ret_loss += direction_loss_weight * direction_bce_loss."""

    # ── Weights & Biases streaming (optional) ──────────────────────────────
    wandb_project: str | None = None
    """W&B project name. If None, wandb logging is disabled."""
    wandb_run_name: str | None = None
    """W&B run display name (e.g. 'h-a-epoch31-40'). Auto-generated if None."""
    wandb_tags: list[str] | None = None
    """Optional list of tags for the W&B run (e.g. ['h-a', 'phase43'])."""


# ═══════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════


class Trainer:
    """Self-supervised walk-forward trainer for HetTGN.

    Training signal:
        For each temporal window W_t, build graph snapshot, run forward,
        then predict what observation types occur in W_{t+1} for each
        entity that has activity. Four loss components:

        1. obs_type CE:    cross-entropy on next observation type per entity
        2. time_delta MSE: mean squared error on time-to-next-event
        3. contrastive:    linked pairs should be closer than random pairs
        4. value Huber:    Huber loss on predicted vs actual observation value
    """

    def __init__(
        self,
        store: PipelineStore,
        config: TrainerConfig | None = None,
    ) -> None:
        self.store = store
        self.config = config or TrainerConfig()
        self._device = torch.device(self.config.device)
        self._model: HetTGN | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._log_vars: dict[str, torch.nn.Parameter] | None = None
        self._graph_builder = GraphBuilder(store)
        self._ewc_state: EWCState | None = None
        self._wandb_run = (
            None  # W&B run handle; initialised by train() if wandb_project is set
        )
        """Populated by train() after the final epoch.
        Holds the Fisher diagonal + anchor weights for continual learning.
        Persisted through save_model / load_model."""

    @property
    def model(self) -> HetTGN:
        if self._model is None:
            raise RuntimeError("Call build_model() first.")
        return self._model

    def build_model(self) -> HetTGN:
        """Build HetTGN from current store contents.

        Must be called after store is populated and before train().
        """
        cfg = self.config
        # Build full graph to determine metadata and node counts
        data, id_map, _ = self._graph_builder.build()

        metadata = data.metadata()
        in_channels = {}
        for ntype in metadata[0]:
            if ntype in data.node_types and hasattr(data[ntype], "x"):
                in_channels[ntype] = data[ntype].x.size(1)
            else:
                in_channels[ntype] = cfg.hidden_dim

        self._model = HetTGN(
            metadata=metadata,
            in_channels=in_channels,
            hidden_dim=cfg.hidden_dim,
            time_dim=cfg.time_dim,
            memory_dim=cfg.memory_dim,
            message_dim=cfg.message_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            num_nodes=id_map.num_nodes,
        )

        # Learnable log-variance params for uncertainty-weighted multi-task
        # loss (Kendall et al. 2018).  Initialised from config weights:
        #   fixed_weight ≈ exp(-log_var) → log_var ≈ -ln(fixed_weight)
        # Stored as plain Parameters (not part of the model) so they can
        # be independently inspected / serialised.
        self._log_vars: dict[str, torch.nn.Parameter] | None = None
        if cfg.auto_tune_loss_weights:
            self._log_vars = {
                "obs_type": torch.nn.Parameter(
                    torch.tensor(-math.log(max(cfg.obs_type_weight, 1e-6)))
                ),
                "time_delta": torch.nn.Parameter(
                    torch.tensor(-math.log(max(cfg.time_delta_weight, 1e-6)))
                ),
                "contrastive": torch.nn.Parameter(
                    torch.tensor(-math.log(max(cfg.contrastive_weight, 1e-6)))
                ),
                "value": torch.nn.Parameter(
                    torch.tensor(-math.log(max(cfg.value_weight, 1e-6)))
                ),
                "return": torch.nn.Parameter(
                    torch.tensor(-math.log(max(cfg.return_weight, 1e-6)))
                ),
            }

        # Move model and log-var tensors to target device
        self._model = self._model.to(self._device)
        if self._log_vars is not None:
            self._log_vars = {
                k: torch.nn.Parameter(v.to(self._device))
                for k, v in self._log_vars.items()
            }

        # Build optimizer — include log-var params when auto-tuning
        opt_params = list(self._model.parameters())
        if self._log_vars is not None:
            opt_params.extend(self._log_vars.values())
        self._optimizer = torch.optim.Adam(opt_params, lr=cfg.learning_rate)
        return self._model

    def _split_observations(
        self,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Chronological 70/15/15 split of all observations."""
        all_obs = self.store.query_all_observations()
        all_obs.sort(key=lambda o: o.get("observed_at", 0.0))
        # Apply obs_since filter if configured
        if self.config.obs_since is not None:
            all_obs = [
                o for o in all_obs if o.get("observed_at", 0.0) >= self.config.obs_since
            ]
        n = len(all_obs)
        n_train = int(n * self.config.train_ratio)
        n_val = int(n * (self.config.train_ratio + self.config.val_ratio))
        return all_obs[:n_train], all_obs[n_train:n_val], all_obs[n_val:]

    def _make_windows(
        self,
        observations: list[dict],
    ) -> list[tuple[float, float, list[dict]]]:
        """Split observations into fixed-length temporal windows.

        Returns list of (start, end, obs_in_window).
        Observations must be sorted by observed_at (ascending).
        Uses O(n) single-pass bucketing instead of O(n*w) scanning.
        """
        if not observations:
            return []
        ws = self.config.window_size
        # Bucket observations by window index, single pass O(n)
        buckets: dict[int, list[dict]] = {}
        for o in observations:
            t = o.get("observed_at", 0.0)
            bucket_idx = int(t // ws)
            buckets.setdefault(bucket_idx, []).append(o)
        # Convert to sorted (start, end, obs) tuples
        windows = []
        for idx in sorted(buckets):
            t_start = idx * ws
            windows.append((t_start, t_start + ws, buckets[idx]))
        return windows

    def _compute_targets(
        self,
        current_window_obs: list[dict],
        next_window_obs: list[dict],
        id_map: IDMap,
    ) -> tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build supervision targets from next-window observations.

        For each entity that appears in next_window_obs, the target is:
            - obs_type index (for CE loss)
            - time delta from current window end (for MSE loss)
            - observation value magnitude (for Huber loss)

        Returns:
            (global_ids, obs_type_targets, time_delta_targets, value_targets)
        """
        # Group by entity — take first obs per entity in next window
        entity_next: dict[str, dict] = {}
        for o in next_window_obs:
            eid = o.get("entity_id")
            if eid and eid not in entity_next:
                entity_next[eid] = o

        # Also need entity_type for each entity_id — use cache if available
        eid_to_type = getattr(self, "_eid_to_type_cache", None)
        if eid_to_type is None:
            all_entities = self.store.query_all_entities()
            eid_to_type = {e["entity_id"]: e["entity_type"] for e in all_entities}

        global_ids = []
        obs_types = []
        time_deltas = []
        values = []

        # Current window end
        if current_window_obs:
            t_end = max(o.get("observed_at", 0.0) for o in current_window_obs)
        else:
            t_end = 0.0

        for eid, o in entity_next.items():
            etype = eid_to_type.get(eid)
            if etype is None:
                continue
            gid = id_map.global_id(etype, eid)
            if gid is None:
                continue
            obs_type = o.get("observation_type", "")
            obs_idx = None
            for i, ot in enumerate(OBSERVATION_TYPES):
                if ot == obs_type:
                    obs_idx = i
                    break
            if obs_idx is None:
                continue

            dt = max(0.0, o.get("observed_at", 0.0) - t_end)
            global_ids.append(gid)
            obs_types.append(obs_idx)
            time_deltas.append(math.log1p(dt))  # log(1+dt) for numerical stability

            # Extract value for value prediction target
            val = 0.0
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
                            val = float(v[k])
                        except (TypeError, ValueError):
                            pass
                        break
            values.append(val)

        return (
            global_ids,
            (
                torch.tensor(obs_types, dtype=torch.long)
                if obs_types
                else torch.zeros(0, dtype=torch.long)
            ),
            (
                torch.tensor(time_deltas, dtype=torch.float)
                if time_deltas
                else torch.zeros(0)
            ),
            (torch.tensor(values, dtype=torch.float) if values else torch.zeros(0)),
        )

    def _contrastive_loss(
        self,
        embeddings: dict[str, torch.Tensor],
        id_map: IDMap,
    ) -> torch.Tensor:
        """Margin-based contrastive loss on entity links.

        Linked pairs should be closer than random pairs by a margin.

        Embeddings are L2-normalised before distance computation so that
        the margin is scale-invariant.  Without normalisation, embedding
        magnitudes driven by the time_delta / value MSE heads grow to
        10^4–10^5 scale, making margin=1.0 permanently inactive.

        Negative samples are drawn randomly (not front-loaded) to avoid
        the degenerate case where the first k nodes of a type are always
        chosen, biasing the neg_mean toward entities with similar features.
        """
        import random

        links = self.store.query_all_entity_links()
        if not links:
            return torch.tensor(0.0)

        # Use cached type lookup if available (set by train() at epoch start)
        if hasattr(self, "_eid_to_type_cache") and self._eid_to_type_cache:
            eid_to_type = self._eid_to_type_cache
        else:
            all_entities = self.store.query_all_entities()
            eid_to_type = {e["entity_id"]: e["entity_type"] for e in all_entities}

        # Pre-normalise all embedding matrices once (scale-invariant distances)
        norm_embeddings: dict[str, torch.Tensor] = {
            ntype: F.normalize(emb, p=2, dim=-1) for ntype, emb in embeddings.items()
        }

        pos_scores = []
        neg_scores = []
        margin = self.config.contrastive_margin

        for lnk in links:
            a_id = lnk["entity_id_a"]
            b_id = lnk["entity_id_b"]
            a_type = eid_to_type.get(a_id)
            b_type = eid_to_type.get(b_id)
            if a_type is None or b_type is None:
                continue
            if a_type not in norm_embeddings or b_type not in norm_embeddings:
                continue

            a_local = id_map.local_id(a_type, a_id)
            b_local = id_map.local_id(b_type, b_id)
            if a_local is None or b_local is None:
                continue

            emb_a_n = norm_embeddings[a_type][a_local]
            emb_b_n = norm_embeddings[b_type][b_local]

            pos_dist = F.pairwise_distance(
                emb_a_n.unsqueeze(0),
                emb_b_n.unsqueeze(0),
            ).squeeze()
            pos_scores.append(pos_dist)

            # Negative: randomly sampled entity of same type as b
            b_embs_n = norm_embeddings[b_type]
            n_nodes = b_embs_n.size(0)
            if n_nodes > 1:
                pool = [j for j in range(n_nodes) if j != b_local]
                neg_indices = random.sample(
                    pool, min(self.config.num_negative_samples, len(pool))
                )
                for neg_idx in neg_indices:
                    neg_dist = F.pairwise_distance(
                        emb_a_n.unsqueeze(0),
                        b_embs_n[neg_idx].unsqueeze(0),
                    ).squeeze()
                    neg_scores.append(neg_dist)

        if not pos_scores or not neg_scores:
            return torch.tensor(0.0)

        pos_mean = torch.stack(pos_scores).mean()
        neg_mean = torch.stack(neg_scores).mean()
        # Margin loss: positive pairs should be closer than negative pairs
        loss = F.relu(pos_mean - neg_mean + margin)
        return loss

    def train(self) -> dict[str, list[float]]:
        """Run walk-forward self-supervised training.

        Returns:
            Dict with loss curves: 'total', 'obs_type', 'time_delta', 'contrastive'.
        """
        import os

        model = self.model
        optimizer = self._optimizer
        cfg = self.config
        model.train()

        # ── W&B initialisation (optional) ────────────────────────────────
        self._wandb_run = None
        if cfg.wandb_project:
            try:
                import wandb as _wandb  # type: ignore[import]
                import dataclasses

                self._wandb_run = _wandb.init(
                    project=cfg.wandb_project,
                    name=cfg.wandb_run_name or None,
                    tags=cfg.wandb_tags or None,
                    resume="allow",
                    config={
                        k: v
                        for k, v in dataclasses.asdict(cfg).items()
                        if k not in ("wandb_project", "wandb_run_name", "wandb_tags")
                    },
                )
                log.info(
                    "W&B run started: %s",
                    self._wandb_run.url if self._wandb_run else "?",
                )
            except Exception as _wb_init_exc:
                log.warning(
                    "W&B init failed (non-fatal) — continuing without streaming: %s",
                    _wb_init_exc,
                )
                self._wandb_run = None

        # ── Resume from checkpoint if requested ───────────────────────────
        start_epoch = 0
        history: dict[str, list[float]] = {
            "total": [],
            "obs_type": [],
            "time_delta": [],
            "contrastive": [],
            "value": [],
            "return": [],
        }
        if cfg.resume_from_epoch > 0 and cfg.checkpoint_dir:
            ckpt_path = os.path.join(
                cfg.checkpoint_dir, f"epoch_{cfg.resume_from_epoch:03d}.pt"
            )
            print(f"[RESUME] Looking for checkpoint: {ckpt_path}")
            print(f"[RESUME] File exists: {os.path.exists(ckpt_path)}")
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=self._device)
                # ── Handle entity-count growth between checkpoints ───────
                # strict=False skips missing/unexpected keys but STILL crashes
                # on shape mismatches for keys that are present.  The only safe
                # approach: pop the variable-size memory buffers out of the
                # state dict before calling load_state_dict, then copy them
                # manually with zero-padding for any new rows.
                ckpt_state = dict(
                    ckpt["model_state"]
                )  # shallow copy — don't mutate ckpt
                MEMORY_BUFFER_KEYS = ("memory.memory", "memory.last_update")
                saved_buffers = {
                    k: ckpt_state.pop(k) for k in MEMORY_BUFFER_KEYS if k in ckpt_state
                }
                # Filter out keys whose shape doesn't match the current model.
                # strict=False already handles missing/extra keys, but PyTorch
                # still raises RuntimeError on size mismatches even with strict=False.
                current_state = model.state_dict()
                shape_mismatches = [
                    k
                    for k, v in ckpt_state.items()
                    if k in current_state and v.shape != current_state[k].shape
                ]
                if shape_mismatches:
                    log.warning(
                        "Skipping %d checkpoint keys with shape mismatches (e.g. hidden_dim changed). First few: %s",
                        len(shape_mismatches),
                        shape_mismatches[:4],
                    )
                    for k in shape_mismatches:
                        ckpt_state.pop(k)
                missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
                arch_changed = bool(shape_mismatches or missing)
                if missing or unexpected:
                    log.warning(
                        "Checkpoint state_dict mismatch — %d missing keys "
                        "(e.g. new Phase 41 heads), %d unexpected keys.",
                        len(missing),
                        len(unexpected),
                    )
                # Restore memory buffers with zero-padding for any new entities
                if "memory.memory" in saved_buffers:
                    ckpt_mem = saved_buffers["memory.memory"].to(self._device)
                    ckpt_lu = saved_buffers["memory.last_update"].to(self._device)
                    ckpt_n = ckpt_mem.shape[0]
                    cur_n = model.memory.memory.shape[0]
                    copy_n = min(ckpt_n, cur_n)
                    model.memory.memory[:copy_n].copy_(ckpt_mem[:copy_n])
                    model.memory.last_update[:copy_n].copy_(ckpt_lu[:copy_n])
                    model.memory.num_nodes = cur_n
                    if ckpt_n != cur_n:
                        log.warning(
                            "Entity count changed since checkpoint: "
                            "checkpoint=%d, current=%d — zero-padded memory buffers.",
                            ckpt_n,
                            cur_n,
                        )
                if arch_changed:
                    # The model architecture changed (new or reshaped parameters).
                    # PyTorch optimizer state is positional — inserting new params
                    # in the middle shifts all subsequent indices, causing Adam to
                    # apply stale exp_avg/exp_avg_sq tensors with wrong shapes to
                    # new params, resulting in the "output with shape [] doesn't
                    # match broadcast shape [1]" crash at optimizer.step().
                    # Safest resolution: skip optimizer state entirely.
                    log.warning(
                        "Architecture change detected (%d shape mismatches, "
                        "%d new keys) — skipping optimizer state to avoid "
                        "index-shift corruption. Model weights loaded OK, "
                        "optimizer starts fresh.",
                        len(shape_mismatches),
                        len(missing),
                    )
                else:
                    try:
                        optimizer.load_state_dict(ckpt["optimizer_state"])
                    except (ValueError, RuntimeError) as exc:
                        log.warning(
                            "Optimizer state incompatible (%s) — starting fresh. Model weights loaded OK.",
                            exc,
                        )
                if self._log_vars is not None and "log_vars" in ckpt:
                    for k, v in ckpt["log_vars"].items():
                        if k in self._log_vars:
                            self._log_vars[k].data.copy_(v.to(self._device))
                history = ckpt.get("history", history)
                # Backfill any keys added after the checkpoint was saved
                # (e.g. "return" added in Phase 41).  Without this,
                # history[k].append() raises KeyError on the first epoch.
                for _hk in (
                    "total",
                    "obs_type",
                    "time_delta",
                    "contrastive",
                    "value",
                    "return",
                ):
                    if _hk not in history:
                        history[_hk] = []
                start_epoch = cfg.resume_from_epoch
                log.info(
                    "Resumed from checkpoint %s (epoch %d)",
                    ckpt_path,
                    start_epoch,
                )
            else:
                raise FileNotFoundError(
                    f"[RESUME] Checkpoint not found: {ckpt_path}\n"
                    f"  checkpoint_dir={cfg.checkpoint_dir}\n"
                    f"  resume_from_epoch={cfg.resume_from_epoch}\n"
                    "  Attach the tirramind-h-d-ckpt dataset containing epoch_018.pt"
                )

        train_obs, _, _ = self._split_observations()
        windows = self._make_windows(train_obs)

        # ── max_windows cap (Phase 41 resource guard) ─────────────────────────
        # Take the LAST max_windows windows (most recent temporal data).
        # This bounds peak RAM to O(max_windows * avg_graph_size) regardless
        # of total DB size. 0 = use all windows (original behaviour).
        if cfg.max_windows > 0 and len(windows) > cfg.max_windows + 1:
            windows = windows[
                -(cfg.max_windows + 1) :
            ]  # +1 because we need windows[i+1] as next
            log.info(
                "max_windows=%d: truncated to last %d windows (%.1f%% of training data)",
                cfg.max_windows,
                cfg.max_windows,
                100.0 * cfg.max_windows / (len(windows)),
            )

        # Cache entity type lookups — entities don't change during training
        all_entities = self.store.query_all_entities()
        self._eid_to_type_cache = {
            e["entity_id"]: e["entity_type"] for e in all_entities
        }

        if cfg.checkpoint_dir:
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)

        # Pre-fetch static graph structure (entities + links) once
        cached_id_map, _, cached_links = self._graph_builder.prepare_static()

        # Pre-fetch ALL observations once, sorted by time.
        # Per-window slicing via bisect eliminates the DB query per window.
        # NOTE: No since= filter here — graph features need full history
        # (original code used build(since=None, until=t_end)).
        all_prefetched_obs = self._graph_builder.prefetch_observations()

        # ── GDELT subsampling (Phase 41) ──────────────────────────────────
        # geopolitical_event rows are 92% of the DB (~901K rows).  Loading
        # and snapshot-building all of them at once causes OOM on CPU.
        # Subsample deterministically (seed=42) so snapshots are reproducible.
        # All non-GDELT obs are always kept; only GDELT rows are thinned.
        if 0.0 < cfg.gdelt_subsample_frac < 1.0:
            import random as _random

            _rng = _random.Random(42)
            _gdelt_kept: list[dict] = []
            _other_kept: list[dict] = []
            for _o in all_prefetched_obs:
                if _o.get("observation_type") == "geopolitical_event":
                    if _rng.random() < cfg.gdelt_subsample_frac:
                        _gdelt_kept.append(_o)
                else:
                    _other_kept.append(_o)
            _n_before = len(all_prefetched_obs)
            all_prefetched_obs = sorted(
                _gdelt_kept + _other_kept,
                key=lambda o: o.get("observed_at", 0.0),
            )
            log.info(
                "GDELT subsample frac=%.3f: %d → %d obs (kept %d GDELT + %d other)",
                cfg.gdelt_subsample_frac,
                _n_before,
                len(all_prefetched_obs),
                len(_gdelt_kept),
                len(_other_kept),
            )
        _obs_timestamps = [o.get("observed_at", 0.0) for o in all_prefetched_obs]

        # ── Fix: dynamic return upscaling ────────────────────────────────────
        # The return head sees only instrument_daily obs with log_return (~53
        # entities) while obs_type sees all ~2,145 entities.  Without upscaling,
        # the return gradient is ~40× weaker than every other head, causing the
        # return head to get drowned out and silenced by auto-tune.
        # We compute the ratio once here and apply it as a multiplier to ret_loss
        # every window.  Ratio = n_unique_entities / n_return_labeled_entities.
        _return_entities = {
            o["entity_id"]
            for o in all_prefetched_obs
            if o.get("observation_type") == "instrument_daily"
            and isinstance(o.get("value"), dict)
            and "log_return" in o["value"]
            and o.get("entity_id") is not None
        }
        _n_return_entities = max(len(_return_entities), 1)
        _n_total_entities = max(len(all_entities), 1)
        _return_upscale = _n_total_entities / _n_return_entities
        log.info(
            "Return upscale: %d total entities / %d return-labeled = %.1f×",
            _n_total_entities,
            _n_return_entities,
            _return_upscale,
        )

        # ── Phase 49: Load GNN alignment weights ──────────────────────────
        # Alignment weights tell the training loop which entity types are
        # not yet well-aligned with the world model (low belief sharpening
        # → high weight → more training emphasis).  Loaded once per
        # train() call — constant across all epochs and windows.
        # Returns None if no alignment signals are stored (uniform weights).
        entity_types_in_graph = list(
            {e.get("entity_type") for e in all_entities if e.get("entity_type")}
        )
        try:
            _alignment_weights: dict[str, float] | None = load_alignment_weights(
                self.store, entity_types_in_graph
            )
        except Exception as exc:
            log.debug("Phase 49: failed to load alignment weights: %s", exc)
            _alignment_weights = None

        # ── Pre-build all window graph snapshots (runs ONCE, reused each epoch) ──
        # build_from_cached() is O(N_obs) Python per window — not the GPU forward
        # pass.  Snapshots are static (depend only on obs up to t_end, not model
        # state), so rebuilding identically each epoch wastes 49×(N_epochs-1) builds.
        # Cache on CPU; first data.to(device) in epoch 1 moves tensors to GPU where
        # they stay, so epochs 2-N incur zero CPU→GPU transfer overhead.
        _total_windows = len(windows) - 1
        log.info(
            "Pre-building %d window graph snapshots (once; reused each epoch) ...",
            _total_windows,
        )
        _window_snapshots: list = []
        _snap_iter = range(_total_windows)
        if _HAS_TQDM:
            _snap_iter = _tqdm(
                _snap_iter,
                total=_total_windows,
                desc="Building snapshots",
                unit="win",
                dynamic_ncols=True,
                leave=False,
            )
        for _snap_i in _snap_iter:
            _t_end_snap = windows[_snap_i][1]
            _cutoff_snap = bisect.bisect_right(_obs_timestamps, _t_end_snap)
            _snap_data, _, _ = self._graph_builder.build_from_cached(
                cached_id_map,
                cached_links,
                observations=all_prefetched_obs[:_cutoff_snap],
            )
            _window_snapshots.append(_snap_data if _snap_data.node_types else None)
        log.info("Snapshots ready — beginning training epochs.")

        for epoch in range(start_epoch, cfg.epochs):
            model.reset_memory()
            epoch_losses = {
                "total": 0.0,
                "obs_type": 0.0,
                "time_delta": 0.0,
                "contrastive": 0.0,
                "value": 0.0,
                "return": 0.0,
            }
            n_windows = 0
            total_windows = _total_windows

            _window_iter = range(total_windows)
            if _HAS_TQDM:
                _window_iter = _tqdm(
                    _window_iter,
                    total=total_windows,
                    desc=f"Epoch {epoch + 1}/{cfg.epochs}",
                    unit="win",
                    dynamic_ncols=True,
                    leave=False,
                )

            for i in _window_iter:
                if not _HAS_TQDM and ((i + 1) % 50 == 0 or i == 0):
                    log.info(
                        "  Epoch %d/%d — window %d/%d",
                        epoch + 1,
                        cfg.epochs,
                        i + 1,
                        total_windows,
                    )
                t_start, t_end, curr_obs = windows[i]
                _, _, next_obs = windows[i + 1]

                # Load pre-built snapshot (no rebuild per epoch)
                data = _window_snapshots[i]
                id_map = cached_id_map
                if data is None:
                    continue

                # Move graph snapshot to target device (no-op after epoch 1 —
                # tensors are already on device from the previous epoch's .to())
                data = data.to(self._device)

                # Forward
                embeddings = model(data, id_map)

                # Supervision targets from next window
                global_ids, obs_targets, dt_targets, val_targets = (
                    self._compute_targets(
                        curr_obs,
                        next_obs,
                        id_map,
                    )
                )

                # Move supervision targets to device
                obs_targets = obs_targets.to(self._device)
                dt_targets = dt_targets.to(self._device)
                val_targets = val_targets.to(self._device)

                # ── obs_type loss ────────────────────────
                obs_loss = torch.tensor(0.0, device=self._device)
                target_embs = []
                valid_indices = []
                target_emb_tensor = None
                # Also track per-example entity types for alignment weighting
                _valid_ntypes: list[str] = []
                if len(obs_targets) > 0:
                    # Gather embeddings for target entities — use cached entity types
                    eid_to_type = self._eid_to_type_cache

                    target_embs = []
                    valid_indices = []
                    for idx, gid in enumerate(global_ids):
                        typed = id_map.global_to_typed.get(gid)
                        if typed is None:
                            continue
                        ntype, eid = typed
                        local_idx = id_map.local_id(ntype, eid)
                        if local_idx is None or ntype not in embeddings:
                            continue
                        if local_idx >= embeddings[ntype].size(0):
                            continue
                        target_embs.append(embeddings[ntype][local_idx])
                        valid_indices.append(idx)
                        _valid_ntypes.append(ntype)

                    if target_embs:
                        target_emb_tensor = torch.stack(target_embs)
                        logits = model.obs_type_head(target_emb_tensor)
                        valid_targets = obs_targets[valid_indices]

                        # Phase 49: apply per-entity-type alignment weights
                        if _alignment_weights is not None and _valid_ntypes:
                            per_example_w = torch.tensor(
                                [
                                    _alignment_weights.get(nt, 1.0)
                                    for nt in _valid_ntypes
                                ],
                                dtype=torch.float32,
                                device=logits.device,
                            )
                            raw_ce = F.cross_entropy(
                                logits, valid_targets, reduction="none"
                            )
                            obs_loss = (raw_ce * per_example_w).mean()
                        else:
                            obs_loss = F.cross_entropy(logits, valid_targets)

                # ── time_delta loss ──────────────────────
                # Targets are log1p(seconds) ≈ 0–14 for weekly windows.
                # Clamp predictions to [-20, 20] before loss to prevent gradient
                # explosions when the head outputs large values mid-training.
                # Huber loss (delta=1.0) further limits gradient magnitude on
                # outlier predictions, replacing the raw MSE that caused spikes.
                # Fix: also clamp TARGETS to [0, 20] so any bad DB row with a
                # far-future observed_at can't produce a target outside the
                # prediction range and cause runaway loss (observed: 168K at ep1).
                dt_loss = torch.tensor(0.0, device=self._device)
                if target_embs:
                    dt_pred = (
                        model.time_delta_head(target_emb_tensor)
                        .squeeze(-1)
                        .clamp(-20.0, 20.0)
                    )
                    valid_dt = dt_targets[valid_indices].clamp(0.0, 20.0)
                    dt_loss = F.huber_loss(dt_pred, valid_dt, delta=1.0)

                # ── value prediction loss ────────────────
                # Fix: clamp predictions to [-1e4, 1e4].  Raw financial values
                # (usd_amount, whale txn amounts) can be $1B+.  Without a clamp
                # the value head produces gradient spikes that propagate backwards
                # and corrupt all other heads (observed: val_loss = 1,094,629
                # at epoch 24 in H-G run).
                val_loss = torch.tensor(0.0, device=self._device)
                if target_embs:
                    val_pred = (
                        model.value_pred_head(target_emb_tensor)
                        .squeeze(-1)
                        .clamp(-1e4, 1e4)
                    )
                    valid_val = val_targets[valid_indices]
                    val_loss = F.huber_loss(val_pred, valid_val)

                # ── contrastive loss ─────────────────────
                c_loss = self._contrastive_loss(embeddings, id_map)

                # ── return auxiliary loss (Phase 41) ──────────────────────────────────
                # Directly supervise instrument embeddings on log_return.
                # Filters next_obs to instrument_daily with log_return only.
                # This pushes the embedding to encode return-relevant info,
                # separate from the generic value_pred_head which sees all types.
                ret_loss = torch.tensor(0.0, device=self._device)
                if cfg.return_weight > 0.0 and "instrument" in embeddings:
                    _ret_embs: list[torch.Tensor] = []
                    _ret_targets: list[float] = []
                    for _o in next_obs:
                        if _o.get("observation_type") != "instrument_daily":
                            continue
                        _v = _o.get("value", {})
                        if not isinstance(_v, dict) or "log_return" not in _v:
                            continue
                        try:
                            _lr = float(_v["log_return"])
                        except (TypeError, ValueError):
                            continue
                        _eid = _o.get("entity_id")
                        if _eid is None:
                            continue
                        _local_idx = id_map.local_id("instrument", _eid)
                        if _local_idx is None:
                            continue
                        _inst_emb = embeddings["instrument"]
                        if _local_idx >= _inst_emb.size(0):
                            continue
                        _ret_embs.append(_inst_emb[_local_idx])
                        _ret_targets.append(_lr)
                    if _ret_embs:
                        _ret_emb_t = torch.stack(_ret_embs)
                        _ret_tgt_t = torch.tensor(
                            _ret_targets, dtype=torch.float32, device=self._device
                        )
                        # Guard: filter out any NaN/Inf targets that came from
                        # bad DB rows (stock splits, missing prices, etc.).
                        # A single NaN target propagates through huber_loss →
                        # total loss → backward → all weights become NaN silently.
                        # ListNet additionally requires >= 2 items to rank;
                        # a single-item softmax is trivially 1.0 → loss = 0.
                        _finite_mask = torch.isfinite(_ret_tgt_t)
                        _n_valid = int(_finite_mask.sum().item())
                        _min_required = 2 if cfg.use_listnet_return_loss else 1
                        if _n_valid >= _min_required:
                            _ret_pred = model.return_pred_head(_ret_emb_t).squeeze(-1)
                            if cfg.use_listnet_return_loss:
                                ret_loss = _listnet_loss(
                                    _ret_pred[_finite_mask],
                                    _ret_tgt_t[_finite_mask],
                                    tau=cfg.listnet_temperature,
                                )
                            else:
                                ret_loss = F.huber_loss(
                                    _ret_pred[_finite_mask], _ret_tgt_t[_finite_mask]
                                )
                            # Direction BCE loss (Phase H-D/H-F hypothesis).
                            # Treat predicted scalar as logit; penalises sign
                            # errors independently of magnitude.  Provides a
                            # complementary gradient to ListNet's rank ordering.
                            if cfg.use_direction_loss:
                                _dir_tgt = (_ret_tgt_t[_finite_mask] > 0).float()
                                _dir_loss = F.binary_cross_entropy_with_logits(
                                    _ret_pred[_finite_mask], _dir_tgt
                                )
                                ret_loss = (
                                    ret_loss + cfg.direction_loss_weight * _dir_loss
                                )
                        # Fix: ListNet loss is KL(p_target || p_pred).
                        # In theory non-negative, but floating-point rounding
                        # in softmax can produce tiny negatives.  A negative
                        # loss rewards the model for being wrong — clamp it out.
                        # Also apply return upscaling: the return head sees only
                        # ~53 instruments while obs_type sees ~2,145 entities.
                        # Without upscaling the return gradient is ~40× weaker
                        # and auto-tune silences it within 10 epochs.
                        ret_loss = ret_loss.clamp(min=0.0) * _return_upscale

                # ── total loss ───────────────────────────
                if self._log_vars is not None:
                    # Uncertainty-weighted multi-task loss
                    # (Kendall et al. 2018): L_k / (2 * sigma_k^2) + ln(sigma_k)
                    # With log_var = ln(sigma^2): exp(-log_var) * L_k + log_var
                    #
                    # Phase 41 hardening: clamp log_var to [log_var_min,
                    # log_var_max]. Without this, when any L_k -> 0 the
                    # optimizer drives log_var -> -inf, the total loss
                    # collapses to -inf, and effective weights explode
                    # (see Phase 40 Run 2 post-mortem; Liebel & Körner 2018).
                    # Clamping still lets gradients flow inside the interval
                    # and saturates at the bounds without mutating the
                    # stored Parameter tensors.
                    lv = self._log_vars
                    lv_min = cfg.log_var_min
                    lv_max = cfg.log_var_max
                    # Use a tighter upper clamp for 'return' so auto-tune
                    # cannot silence the return head (cfg.return_log_var_max
                    # defaults to 0.0 → weight ≥ exp(-0) = 1.0).
                    clamped = {
                        k: torch.clamp(
                            p,
                            min=lv_min,
                            max=(cfg.return_log_var_max if k == "return" else lv_max),
                        )
                        for k, p in lv.items()
                    }
                    total = (
                        torch.exp(-clamped["obs_type"]) * obs_loss
                        + clamped["obs_type"]
                        + torch.exp(-clamped["time_delta"]) * dt_loss
                        + clamped["time_delta"]
                        + torch.exp(-clamped["contrastive"]) * c_loss
                        + clamped["contrastive"]
                        + torch.exp(-clamped["value"]) * val_loss
                        + clamped["value"]
                        + torch.exp(-clamped["return"]) * ret_loss
                        + clamped["return"]
                    )
                else:
                    total = (
                        cfg.obs_type_weight * obs_loss
                        + cfg.time_delta_weight * dt_loss
                        + cfg.contrastive_weight * c_loss
                        + cfg.value_weight * val_loss
                        + cfg.return_weight * ret_loss
                    )

                if total.requires_grad:
                    optimizer.zero_grad()
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                # Update memory from current window events
                with torch.no_grad():
                    # Recompute embeddings after optimizer step
                    embeddings_detached = model(data, id_map)
                model.update_memory_from_events(curr_obs, embeddings_detached, id_map)

                epoch_losses["total"] += total.item()
                epoch_losses["obs_type"] += obs_loss.item()
                epoch_losses["time_delta"] += dt_loss.item()
                epoch_losses["contrastive"] += c_loss.item()
                epoch_losses["value"] += val_loss.item()
                epoch_losses["return"] += ret_loss.item()
                n_windows += 1

            # Average over windows
            for k in epoch_losses:
                avg = epoch_losses[k] / max(n_windows, 1)
                history[k].append(avg)

            log.info(
                "Epoch %d/%d — loss: %.4f (obs_type: %.4f, dt: %.4f, contrastive: %.4f, return: %.4f)",
                epoch + 1,
                cfg.epochs,
                history["total"][-1],
                history["obs_type"][-1],
                history["time_delta"][-1],
                history["contrastive"][-1],
                history["return"][-1],
            )

            # ── Optimization-target validator (Phase AR.4) ───────────────
            # Warn when dt loss dominates return head by >50× for 3+ epochs.
            # This indicates the return head is receiving near-zero gradient
            # and IC will stay flat regardless of architecture improvements.
            if len(history["time_delta"]) >= 3 and len(history["return"]) >= 3:
                _dt_avg = sum(history["time_delta"][-3:]) / 3
                _ret_avg = sum(history["return"][-3:]) / 3
                _ratio = _dt_avg / (_ret_avg + 1e-8)
                if _ratio > 50:
                    log.warning(
                        "OPTIMIZATION TARGET WARNING: return head receiving <2%% of "
                        "gradient budget (dt/ret loss ratio=%.0fx over last 3 epochs). "
                        "IC is likely to stay flat. Consider: --return-weight, "
                        "--gdelt-frac, or checking data balance.",
                        _ratio,
                    )
            if self._log_vars is not None:
                eff = self.effective_loss_weights()
                log.info(
                    "  Effective loss weights: obs=%.3f dt=%.3f contr=%.3f val=%.3f ret=%.3f",
                    eff["obs_type"],
                    eff["time_delta"],
                    eff["contrastive"],
                    eff["value"],
                    eff["return"],
                )

            # ── W&B streaming ─────────────────────────────────────────────
            if self._wandb_run is not None:
                try:
                    _wmetrics: dict = {
                        "epoch": epoch + 1,
                        "loss/total": history["total"][-1],
                        "loss/obs_type": history["obs_type"][-1],
                        "loss/time_delta": history["time_delta"][-1],
                        "loss/contrastive": history["contrastive"][-1],
                        "loss/value": history["value"][-1],
                        "loss/return": (
                            history["return"][-1] if history["return"] else float("nan")
                        ),
                    }
                    if self._log_vars is not None:
                        _weff = self.effective_loss_weights()
                        for _k, _v in _weff.items():
                            _wmetrics[f"weight/{_k}"] = _v
                    self._wandb_run.log(_wmetrics, step=epoch + 1)
                except Exception as _wandb_exc:
                    log.debug("W&B log failed (non-fatal): %s", _wandb_exc)

            # ── Per-epoch checkpoint ──────────────────────────────────────
            if cfg.checkpoint_dir:
                ckpt_path = os.path.join(
                    cfg.checkpoint_dir, f"epoch_{epoch + 1:03d}.pt"
                )
                ckpt_payload: dict = {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "history": history,
                }
                if self._log_vars is not None:
                    ckpt_payload["log_vars"] = {
                        k: v.data.cpu() for k, v in self._log_vars.items()
                    }
                torch.save(ckpt_payload, ckpt_path)
                log.info("  Checkpoint saved → %s", ckpt_path)
                # ── Per-epoch metrics.jsonl (lightweight, no torch to read) ──
                import json as _json_m  # noqa: PLC0415
                _ret_loss = history["return"][-1] if history.get("return") else float("nan")
                _dt_loss = history["time_delta"][-1] if history.get("time_delta") else float("nan")
                _dt_ret_ratio = (
                    _dt_loss / max(_ret_loss, 1e-8)
                    if not (math.isnan(_ret_loss) or math.isnan(_dt_loss) or _ret_loss < 1e-10)
                    else float("nan")
                )
                _epoch_record = {
                    "epoch": epoch + 1,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "loss": {
                        "total": history["total"][-1] if history.get("total") else float("nan"),
                        "return": _ret_loss,
                        "dt": _dt_loss,
                        "obs": history["obs_type"][-1] if history.get("obs_type") else float("nan"),
                        "contrastive": history["contrastive"][-1] if history.get("contrastive") else float("nan"),
                        "value": history["value"][-1] if history.get("value") else float("nan"),
                    },
                    "dt_ret_ratio": _dt_ret_ratio,
                    "config": {
                        "lr": cfg.learning_rate,
                        "return_weight": float(cfg.return_weight),
                        "gdelt_frac": float(getattr(cfg, "gdelt_subsample_frac", 1.0)),
                        "listnet_temp": float(getattr(cfg, "listnet_temperature", 1.0)),
                        "auto_tune": bool(cfg.auto_tune_loss_weights),
                        "use_listnet": bool(getattr(cfg, "use_listnet_return_loss", False)),
                    },
                }
                _metrics_path = os.path.join(cfg.checkpoint_dir, "metrics.jsonl")
                with open(_metrics_path, "a") as _mf:
                    _mf.write(_json_m.dumps(_epoch_record) + "\n")
        # After all epochs complete, approximate F_i ≈ E[(dL/dθ_i)²] on
        # the last training window pair.  This is the Laplace approximation
        # of the posterior p(θ | data_old) that EWC uses to protect
        # parameters important to previously learned tasks.
        # Ref: Kirkpatrick et al. 2017, arXiv:1612.00796, Section 2.
        if len(windows) >= 2 and all_prefetched_obs:
            log.info("Computing Fisher Information diagonal for EWC (Phase 46) ...")
            last_curr_obs = windows[-2][2]
            last_next_obs = windows[-1][2]
            last_t_end = windows[-2][1]
            fisher_cutoff = bisect.bisect_right(_obs_timestamps, last_t_end)
            fisher_window_obs = all_prefetched_obs[:fisher_cutoff]
            fisher_data, fisher_id_map, _ = self._graph_builder.build_from_cached(
                cached_id_map,
                cached_links,
                observations=fisher_window_obs,
            )
            if fisher_data.node_types:
                # Move fisher snapshot to same device as model before closure.
                fisher_data = fisher_data.to(self._device)

                # Closure: zero-arg callable that rebuilds gradients from
                # the fixed last-window snapshot.  Passed to compute_fisher
                # so the EWC module stays decoupled from all data logic.
                def _fisher_loss_fn(
                    _fd=fisher_data,
                    _fm=fisher_id_map,
                    _co=last_curr_obs,
                    _no=last_next_obs,
                ) -> torch.Tensor:
                    return self._loss_from_window(_fd, _fm, _co, _no)

                fisher_diag = compute_fisher(model, _fisher_loss_fn, n_samples=1)
                self._ewc_state = EWCState(
                    fisher=fisher_diag,
                    anchor={
                        n: p.data.clone().cpu() for n, p in model.named_parameters()
                    },
                    lambda_=cfg.ewc_lambda,
                    last_update_ts=time.time(),
                    obs_count_at_update=len(all_prefetched_obs),
                )
                log.info(
                    "EWC state computed: %d params in Fisher, lambda=%.1f, obs_count=%d",
                    len(fisher_diag),
                    cfg.ewc_lambda,
                    len(all_prefetched_obs),
                )
            else:
                log.warning(
                    "Fisher computation skipped — last training window produced an empty graph (no nodes)."
                )
        else:
            log.warning(
                "Fisher computation skipped — need ≥ 2 training windows "
                "(got %d). EWC will not be available until more data "
                "accumulates.",
                len(windows),
            )

        # ── W&B finish ───────────────────────────────────────────────────
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception:
                pass
            self._wandb_run = None

        return history

    def effective_loss_weights(self) -> dict[str, float]:
        """Return current effective loss weights.

        When auto_tune_loss_weights is on, these are exp(-clamp(log_var))
        for each task — i.e. the learned precision after applying the
        [log_var_min, log_var_max] clamp so that the reported weights
        always match what the training step actually applied. Otherwise
        returns the fixed config weights.
        """
        if self._log_vars is not None:
            cfg = self.config
            return {
                k: math.exp(-max(cfg.log_var_min, min(cfg.log_var_max, p.item())))
                for k, p in self._log_vars.items()
            }
        cfg = self.config
        return {
            "obs_type": cfg.obs_type_weight,
            "time_delta": cfg.time_delta_weight,
            "contrastive": cfg.contrastive_weight,
            "value": cfg.value_weight,
            "return": cfg.return_weight,
        }

    # ── Inference (Phase 19a) ─────────────────────────────────

    def infer(
        self,
        *,
        until: float | None = None,
    ) -> tuple[dict[str, torch.Tensor], IDMap]:
        """Run a forward pass on the entity graph and return embeddings.

        If the model has not been trained, returns embeddings from a
        randomly-initialized model (useful for testing the downstream
        pipeline before real training data accumulates).

        Args:
            until: Optional cutoff time for point-in-time graph building.
                   Entities/observations after this time are excluded.

        Returns:
            (embeddings, id_map) where embeddings is
            dict[entity_type → Tensor[N_type, hidden_dim]] and id_map
            maps entity IDs to node indices.
        """
        if self._model is None:
            self.build_model()

        model = self.model
        model.eval()

        with torch.no_grad():
            data, id_map, _ = self._graph_builder.build(until=until)
            if not data.node_types:
                return {}, id_map
            # Resize memory if entity count grew since last training
            if id_map.num_nodes > model.memory.num_nodes:
                log.warning(
                    "Entity count grew %d → %d since last training; resizing GNN memory buffer.",
                    model.memory.num_nodes,
                    id_map.num_nodes,
                )
                model.memory.resize(id_map.num_nodes)
            embeddings = model(data, id_map)

        return embeddings, id_map

    # ── Phase 46: continual learning helpers ─────────────────────────────

    def _loss_from_window(
        self,
        data,
        id_map: IDMap,
        curr_obs: list[dict],
        next_obs: list[dict],
    ) -> torch.Tensor:
        """Compute the full multi-task self-supervised loss for one window pair.

        Identical loss formulation to the training loop (obs_type CE,
        time_delta MSE, value Huber, contrastive margin), with the same
        auto-tuning branch when ``config.auto_tune_loss_weights`` is True.

        Used by:
          - ``train()`` — to build the Fisher loss closure after final epoch.
          - ``online_update()`` — as the L_new term in the EWC objective.

        The training loop has its own identical inline copy and is left
        unchanged for backward compatibility.

        Args:
            data:     HeteroData graph snapshot (already built by caller).
            id_map:   IDMap matching ``data``.
            curr_obs: Observations in the current window (for contrastive
                      and memory update context).
            next_obs: Observations in the next window (supervision targets).

        Returns:
            Scalar Tensor with requires_grad=True when the graph contains
            at least one target entity; a detached zero Tensor otherwise.
        """
        model = self.model
        cfg = self.config

        embeddings = model(data, id_map)
        global_ids, obs_targets, dt_targets, val_targets = self._compute_targets(
            curr_obs, next_obs, id_map
        )

        # ── obs_type loss ────────────────────────────────────────────────
        obs_loss = torch.tensor(0.0)
        target_embs: list[torch.Tensor] = []
        valid_indices: list[int] = []
        target_emb_tensor: torch.Tensor | None = None

        if len(obs_targets) > 0:
            for idx, gid in enumerate(global_ids):
                typed = id_map.global_to_typed.get(gid)
                if typed is None:
                    continue
                ntype, eid = typed
                local_idx = id_map.local_id(ntype, eid)
                if local_idx is None or ntype not in embeddings:
                    continue
                if local_idx >= embeddings[ntype].size(0):
                    continue
                target_embs.append(embeddings[ntype][local_idx])
                valid_indices.append(idx)

            if target_embs:
                target_emb_tensor = torch.stack(target_embs)
                logits = model.obs_type_head(target_emb_tensor)
                valid_targets = obs_targets[valid_indices].to(logits.device)
                obs_loss = F.cross_entropy(logits, valid_targets)

        # ── time_delta loss ──────────────────────────────────────────────
        dt_loss = torch.tensor(0.0, device=self._device)
        if target_embs and target_emb_tensor is not None:
            dt_pred = model.time_delta_head(target_emb_tensor).squeeze(-1)
            valid_dt = dt_targets[valid_indices].to(dt_pred.device)
            dt_loss = F.mse_loss(dt_pred, valid_dt)

        # ── value loss ───────────────────────────────────────────────────
        val_loss = torch.tensor(0.0, device=self._device)
        if target_embs and target_emb_tensor is not None:
            val_pred = model.value_pred_head(target_emb_tensor).squeeze(-1)
            valid_val = val_targets[valid_indices].to(val_pred.device)
            val_loss = F.huber_loss(val_pred, valid_val)

        # ── contrastive loss ─────────────────────────────────────────────
        c_loss = self._contrastive_loss(embeddings, id_map)

        # ── combine (mirror the training loop's auto-tune branch) ────────
        if self._log_vars is not None:
            lv = self._log_vars
            lv_min = cfg.log_var_min
            lv_max = cfg.log_var_max
            clamped = {k: torch.clamp(p, min=lv_min, max=lv_max) for k, p in lv.items()}
            total = (
                torch.exp(-clamped["obs_type"]) * obs_loss
                + clamped["obs_type"]
                + torch.exp(-clamped["time_delta"]) * dt_loss
                + clamped["time_delta"]
                + torch.exp(-clamped["contrastive"]) * c_loss
                + clamped["contrastive"]
                + torch.exp(-clamped["value"]) * val_loss
                + clamped["value"]
            )
        else:
            total = (
                cfg.obs_type_weight * obs_loss
                + cfg.time_delta_weight * dt_loss
                + cfg.contrastive_weight * c_loss
                + cfg.value_weight * val_loss
            )

        return total

    def online_update(self, new_events: list[dict]) -> dict[str, float]:
        """Apply a single EWC-regularised gradient step on new observations.

        This is the online continual learning path (Phase 46).  One gradient
        step minimises:

            L_total = L_new(new_events) + λ · Σ_i F_i (θ_i − θ_i*)²

        where L_new is the full multi-task self-supervised loss (same
        formulation as full training), and the second term is the EWC
        penalty that prevents catastrophic forgetting.  High-Fisher
        parameters (those critical to previously learned tasks) resist large
        updates; low-Fisher parameters can freely adapt to new signals.

        When to call:
            After ``train()`` completes at least once.  Designed to be
            called periodically by the ``gnn_inference`` DAG operator
            whenever at least ``config.online_batch_threshold`` new
            observations have accumulated since the last update.

        Args:
            new_events: Non-empty list of observation dicts with the same
                        schema as ``PipelineStore.query_all_observations()``.
                        Must have at least 1 entry.

        Returns:
            Dict with 4 float keys::

                {
                    "loss_new":   multi-task loss on new data (no regularisation),
                    "loss_ewc":   EWC penalty term only,
                    "loss_total": sum of above (what was back-propagated),
                    "n_events":   float(len(new_events)),
                }

        Raises:
            RuntimeError: If model is not built, EWC state is not computed,
                          new_events is empty, or the graph has no nodes.
        """
        # ── Guards ───────────────────────────────────────────────────────
        if self._model is None:
            raise RuntimeError("Model not built — call train() before online_update().")
        if self._ewc_state is None:
            raise RuntimeError(
                "EWC state not computed — train() must complete at least once "
                "before calling online_update(). The Fisher diagonal is "
                "computed after the final training epoch."
            )
        if not new_events:
            raise RuntimeError(
                "new_events must be non-empty — there is nothing to learn from an empty observation batch."
            )

        model = self._model
        optimizer = self._optimizer
        cfg = self.config

        # Sort new events chronologically (same invariant the training loop
        # assumes; prevents temporal leakage in target construction).
        new_events_sorted = sorted(new_events, key=lambda o: o.get("observed_at", 0.0))

        # ── Build temporal windows from the new events ───────────────────
        # _make_windows uses config.window_size (default 1 day). If all
        # new events arrive within a single day, we get 0 or 1 windows —
        # the degenerate case is handled by treating all events as curr_obs
        # with an empty next_obs.  In this case obs_type/dt/value losses
        # are zero; contrastive loss still fires if entity links exist.
        windows_new = self._make_windows(new_events_sorted)

        if len(windows_new) >= 2:
            curr_obs = windows_new[-2][2]
            next_obs = windows_new[-1][2]
            t_end = windows_new[-2][1]
        else:
            # Sub-window batch: use all events for context, no supervision
            curr_obs = new_events_sorted
            next_obs = []
            t_end = new_events_sorted[-1].get("observed_at", 0.0)

        # ── Build full graph snapshot up to t_end ────────────────────────
        # Use the live DB (not the pre-cached static snapshot) so that
        # newly stored entities and links are included in the context.
        # This is intentionally slower than build_from_cached — online
        # updates run infrequently (~daily) so the cost is acceptable.
        data, id_map, _ = self._graph_builder.build(until=t_end)

        if not data.node_types:
            raise RuntimeError(
                f"online_update: graph has no nodes up to t={t_end:.0f}. "
                "Ensure entities and observations are stored before calling "
                "online_update()."
            )

        # Resize model memory buffer if entity population grew since the
        # last full retrain (new entities start with zero memory state).
        if id_map.num_nodes > model.memory.num_nodes:
            log.warning(
                "online_update: entity count grew %d → %d since last full "
                "retrain. Resizing GNN memory buffer — new entities start "
                "with zero memory state.",
                model.memory.num_nodes,
                id_map.num_nodes,
            )
            model.memory.resize(id_map.num_nodes)

        # ── Forward: compute L_new ───────────────────────────────────────
        model.train()
        loss_new: torch.Tensor = self._loss_from_window(
            data, id_map, curr_obs, next_obs
        )

        # ── EWC penalty: λ · Σ F_i (θ_i − θ_i*)² ───────────────────────
        loss_ewc: torch.Tensor = ewc_penalty(model, self._ewc_state)

        # ── Backward pass: optimise L_total ─────────────────────────────
        loss_total: torch.Tensor = loss_new + loss_ewc

        if loss_total.requires_grad:
            optimizer.zero_grad()
            loss_total.backward()
            # Same grad-clipping as the training loop (prevents gradient
            # explosion on small online batches).
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            log.warning(
                "online_update: loss_total.requires_grad=False — skipping "
                "optimizer step. This usually means the graph is too sparse "
                "to produce supervised targets and no entity links exist for "
                "contrastive loss. Update will be a no-op."
            )

        # ── Bookkeeping ──────────────────────────────────────────────────
        # Update EWC state timestamps so the DAG operator can decide when
        # the next online update is due.
        self._ewc_state.last_update_ts = time.time()
        self._ewc_state.obs_count_at_update += len(new_events)

        result = {
            "loss_new": loss_new.item(),
            "loss_ewc": loss_ewc.item(),
            "loss_total": loss_total.item(),
            "n_events": float(len(new_events)),
        }
        log.info(
            "online_update: loss_new=%.4f loss_ewc=%.4f loss_total=%.4f n_events=%d obs_count_at_update=%d",
            result["loss_new"],
            result["loss_ewc"],
            result["loss_total"],
            len(new_events),
            self._ewc_state.obs_count_at_update,
        )
        return result

    def save_model(self, path: str | Path) -> None:
        """Persist trained model state to disk.

        Saves both the model state_dict and the config/metadata needed
        to reconstruct it. Creates parent directories if needed.

        Args:
            path: File path for the saved checkpoint.
        """
        if self._model is None:
            raise RuntimeError(
                "No model to save — call build_model() or train() first."
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build graph once to capture metadata for reconstruction
        data, id_map, _ = self._graph_builder.build()
        metadata = data.metadata()
        in_channels: dict[str, int] = {}
        for ntype in metadata[0]:
            if ntype in data.node_types and hasattr(data[ntype], "x"):
                in_channels[ntype] = data[ntype].x.size(1)
            else:
                in_channels[ntype] = self.config.hidden_dim

        checkpoint = {
            "model_state_dict": self._model.state_dict(),
            "config": {
                "hidden_dim": self.config.hidden_dim,
                "memory_dim": self.config.memory_dim,
                "message_dim": self.config.message_dim,
                "time_dim": self.config.time_dim,
                "num_heads": self.config.num_heads,
                "num_layers": self.config.num_layers,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
                "window_size": self.config.window_size,
                "train_ratio": self.config.train_ratio,
                "val_ratio": self.config.val_ratio,
                "obs_type_weight": self.config.obs_type_weight,
                "time_delta_weight": self.config.time_delta_weight,
                "contrastive_weight": self.config.contrastive_weight,
                "contrastive_margin": self.config.contrastive_margin,
                "num_negative_samples": self.config.num_negative_samples,
                "value_weight": self.config.value_weight,
                "auto_tune_loss_weights": self.config.auto_tune_loss_weights,
                # Phase 46 — EWC / online learning config
                "ewc_lambda": self.config.ewc_lambda,
                "online_batch_threshold": self.config.online_batch_threshold,
            },
            "metadata_node_types": metadata[0],
            "metadata_edge_types": [list(t) for t in metadata[1]],
            "in_channels": in_channels,
            "num_nodes": id_map.num_nodes,
        }

        # Phase 46: persist EWC state when present.
        # Old checkpoints that pre-date Phase 46 simply lack these keys;
        # load_model treats their absence as _ewc_state=None (backward compat).
        if self._ewc_state is not None:
            checkpoint["ewc_fisher"] = self._ewc_state.fisher
            checkpoint["ewc_anchor"] = self._ewc_state.anchor
            checkpoint["ewc_lambda"] = self._ewc_state.lambda_
            checkpoint["ewc_last_update_ts"] = self._ewc_state.last_update_ts
            checkpoint["ewc_obs_count_at_update"] = self._ewc_state.obs_count_at_update
            log.info(
                "EWC state serialised: %d Fisher params.",
                len(self._ewc_state.fisher),
            )

        torch.save(checkpoint, path)
        log.info("Model saved to %s (%d nodes).", path, id_map.num_nodes)

    @classmethod
    def load_model(cls, path: str | Path, store: PipelineStore) -> Trainer:
        """Load a previously saved model checkpoint.

        Reconstructs the HetTGN from saved metadata and loads the
        state_dict. The returned Trainer is ready for infer() calls.

        Args:
            path: File path to the saved checkpoint.
            store: PipelineStore (needed for GraphBuilder and future ops).

        Returns:
            Trainer with loaded model.

        Raises:
            FileNotFoundError: If path does not exist.
            RuntimeError: If checkpoint is corrupt or incompatible.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")

        checkpoint = torch.load(path, weights_only=False)

        config = TrainerConfig(**checkpoint["config"])
        trainer = cls(store, config)

        metadata = (
            checkpoint["metadata_node_types"],
            [tuple(t) for t in checkpoint["metadata_edge_types"]],
        )
        in_channels = checkpoint["in_channels"]
        num_nodes = checkpoint["num_nodes"]

        trainer._model = HetTGN(
            metadata=metadata,
            in_channels=in_channels,
            hidden_dim=config.hidden_dim,
            time_dim=config.time_dim,
            memory_dim=config.memory_dim,
            message_dim=config.message_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            num_nodes=num_nodes,
        )

        missing, unexpected = trainer._model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        if missing or unexpected:
            log.warning(
                "load_model: %d missing keys, %d unexpected keys (strict=False).",
                len(missing),
                len(unexpected),
            )
        trainer._optimizer = torch.optim.Adam(
            trainer._model.parameters(), lr=config.learning_rate
        )

        # Phase 46: restore EWC state if present (absent = pre-Phase-46 checkpoint).
        if "ewc_fisher" in checkpoint:
            trainer._ewc_state = EWCState(
                fisher=checkpoint["ewc_fisher"],
                anchor=checkpoint["ewc_anchor"],
                lambda_=checkpoint.get("ewc_lambda", 1000.0),
                last_update_ts=checkpoint.get("ewc_last_update_ts", 0.0),
                obs_count_at_update=checkpoint.get("ewc_obs_count_at_update", 0),
            )
            log.info(
                "EWC state restored from checkpoint: %d Fisher params, lambda=%.1f.",
                len(trainer._ewc_state.fisher),
                trainer._ewc_state.lambda_,
            )
        else:
            log.info(
                "No EWC state in checkpoint (pre-Phase-46 model). Run train() to compute Fisher diagonal."
            )

        log.info("Model loaded from %s.", path)
        return trainer


# ═══════════════════════════════════════════════════════════════
# Walk-forward evaluation
# ═══════════════════════════════════════════════════════════════


def evaluate(
    model: HetTGN,
    store: PipelineStore,
    config: TrainerConfig | None = None,
    split: str = "val",
) -> dict[str, float]:
    """Evaluate HetTGN on validation or test split.

    Walk-forward: for each window in the split, predict next window's
    observation types and time deltas.

    Args:
        model: Trained HetTGN.
        store: PipelineStore with data.
        config: Same config used for training (for splits/windows).
        split: 'val' or 'test'.

    Returns:
        Dict with metrics: obs_type_acc_top1, obs_type_acc_top5,
        time_delta_mae, num_predictions.
    """
    cfg = config or TrainerConfig()
    graph_builder = GraphBuilder(store)
    model.eval()
    model.reset_memory()

    # Get split observations
    all_obs = store.query_all_observations()
    all_obs.sort(key=lambda o: o.get("observed_at", 0.0))
    # Apply obs_since filter if configured
    if cfg.obs_since is not None:
        all_obs = [o for o in all_obs if o.get("observed_at", 0.0) >= cfg.obs_since]
    n = len(all_obs)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * (cfg.train_ratio + cfg.val_ratio))

    if split == "val":
        eval_obs = all_obs[n_train:n_val]
    elif split == "test":
        eval_obs = all_obs[n_val:]
    else:
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")

    if not eval_obs:
        return {
            "obs_type_acc_top1": 0.0,
            "obs_type_acc_top5": 0.0,
            "time_delta_mae": 0.0,
            "num_predictions": 0,
        }

    # Build windows — O(n) single-pass bucketing
    ws = cfg.window_size
    buckets: dict[int, list[dict]] = {}
    for o in eval_obs:
        t = o.get("observed_at", 0.0)
        bucket_idx = int(t // ws)
        buckets.setdefault(bucket_idx, []).append(o)
    windows = []
    for idx in sorted(buckets):
        t_start = idx * ws
        windows.append((t_start, t_start + ws, buckets[idx]))

    all_entities = store.query_all_entities()
    eid_to_type = {e["entity_id"]: e["entity_type"] for e in all_entities}

    correct_top1 = 0
    correct_top5 = 0
    total_dt_ae = 0.0
    num_preds = 0

    # Pre-fetch static graph structure for cached builds
    cached_id_map, _, cached_links = graph_builder.prepare_static()

    # Pre-fetch ALL observations sorted by time for bisect slicing
    # No since= filter — graph features need full history.
    all_prefetched_obs = graph_builder.prefetch_observations()
    _obs_timestamps = [o.get("observed_at", 0.0) for o in all_prefetched_obs]

    with torch.no_grad():
        for i in range(len(windows) - 1):
            t_start, t_end, curr_obs = windows[i]
            _, _, next_obs = windows[i + 1]

            # Build graph up to current window end (fully cached)
            cutoff = bisect.bisect_right(_obs_timestamps, t_end)
            window_obs = all_prefetched_obs[:cutoff]
            data, id_map, events = graph_builder.build_from_cached(
                cached_id_map,
                cached_links,
                observations=window_obs,
            )
            if not data.node_types:
                continue

            embeddings = model(data, id_map)

            # Group next window by entity (first obs per entity)
            entity_next: dict[str, dict] = {}
            for o in next_obs:
                eid = o.get("entity_id")
                if eid and eid not in entity_next:
                    entity_next[eid] = o

            t_ref = max(o.get("observed_at", 0.0) for o in curr_obs)

            for eid, o in entity_next.items():
                etype = eid_to_type.get(eid)
                if etype is None or etype not in embeddings:
                    continue
                local_idx = id_map.local_id(etype, eid)
                if local_idx is None or local_idx >= embeddings[etype].size(0):
                    continue

                true_type = o.get("observation_type", "")
                true_idx = None
                for j, ot in enumerate(OBSERVATION_TYPES):
                    if ot == true_type:
                        true_idx = j
                        break
                if true_idx is None:
                    continue

                emb = embeddings[etype][local_idx]
                logits = model.obs_type_head(emb.unsqueeze(0)).squeeze(0)
                pred_type = logits.argmax().item()
                top5 = logits.topk(min(5, len(OBSERVATION_TYPES))).indices.tolist()

                if pred_type == true_idx:
                    correct_top1 += 1
                if true_idx in top5:
                    correct_top5 += 1

                dt_pred = model.time_delta_head(emb.unsqueeze(0)).squeeze().item()
                dt_true = math.log1p(max(0.0, o.get("observed_at", 0.0) - t_ref))
                total_dt_ae += abs(dt_pred - dt_true)
                num_preds += 1

            # Update memory from current window
            model.update_memory_from_events(curr_obs, embeddings, id_map)

    return {
        "obs_type_acc_top1": correct_top1 / max(num_preds, 1),
        "obs_type_acc_top5": correct_top5 / max(num_preds, 1),
        "time_delta_mae": total_dt_ae / max(num_preds, 1),
        "num_predictions": num_preds,
    }


# ═══════════════════════════════════════════════════════════════
# Phase 15a: Outcome Label Generation
# ═══════════════════════════════════════════════════════════════


@dataclass
class OutcomeLabel:
    """Binary co-occurrence label for supervised fine-tuning.

    Attributes:
        src_entity_id: Source entity ID.
        dst_entity_id: Destination entity ID.
        src_type: Source entity type.
        dst_type: Destination entity type.
        pattern_edge: Edge type from the CrystallizedPattern.
        timestamp: Timestamp of the source observation.
        label: 1 if target obs occurs within window, 0 otherwise.
    """

    src_entity_id: str
    dst_entity_id: str
    src_type: str
    dst_type: str
    pattern_edge: str
    timestamp: float
    label: int  # 0 or 1


def generate_outcome_labels(
    patterns: list,  # list[CrystallizedPattern] — avoid circular import
    store: PipelineStore,
    *,
    since: float | None = None,
    until: float | None = None,
    max_neg_ratio: float = 3.0,
    seed: int = 42,
) -> list[OutcomeLabel]:
    """Generate binary outcome labels from CrystallizedPatterns.

    For each pattern, for each linked (src, dst) entity pair, for each
    source observation of ``obs_type_a``: label as 1 (positive) if a
    ``obs_type_b`` observation on the destination entity occurs within
    ``window_seconds`` after the source observation, 0 otherwise.

    Balanced subsampling limits negatives to ``max_neg_ratio`` × positives
    to prevent severe class imbalance.

    Args:
        patterns: CrystallizedPattern list.
        store: PipelineStore.
        since: Optional start time filter.
        until: Optional end time filter.
        max_neg_ratio: Maximum negative-to-positive ratio (default 3:1).
        seed: Random seed for subsampling.

    Returns:
        List of OutcomeLabel.
    """
    rng = random.Random(seed)

    all_entities = store.query_all_entities()
    eid_to_type = {e["entity_id"]: e["entity_type"] for e in all_entities}

    all_obs = store.query_all_observations(since=since, until=until)
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

    positives: list[OutcomeLabel] = []
    negatives: list[OutcomeLabel] = []

    for cp in patterns:
        for (eid_a, lt), targets in link_index.items():
            if lt != cp.via_edge:
                continue
            if eid_to_type.get(eid_a) != cp.source_type:
                continue

            src_obs = sorted(
                [
                    o
                    for o in obs_by_entity.get(eid_a, [])
                    if o.get("observation_type") == cp.obs_type_a
                ],
                key=lambda o: o.get("observed_at", 0.0),
            )
            if not src_obs:
                continue

            for eid_b in targets:
                if eid_to_type.get(eid_b) != cp.target_type:
                    continue
                dst_obs = sorted(
                    [
                        o
                        for o in obs_by_entity.get(eid_b, [])
                        if o.get("observation_type") == cp.obs_type_b
                    ],
                    key=lambda o: o.get("observed_at", 0.0),
                )

                for so in src_obs:
                    st = so.get("observed_at", 0.0)
                    hit = any(
                        0 < (do.get("observed_at", 0.0) - st) <= cp.window_seconds
                        for do in dst_obs
                    )
                    lbl = OutcomeLabel(
                        src_entity_id=eid_a,
                        dst_entity_id=eid_b,
                        src_type=cp.source_type,
                        dst_type=cp.target_type,
                        pattern_edge=cp.via_edge,
                        timestamp=st,
                        label=1 if hit else 0,
                    )
                    if hit:
                        positives.append(lbl)
                    else:
                        negatives.append(lbl)

    # Balanced subsampling
    max_neg = max(1, int(len(positives) * max_neg_ratio))
    if len(negatives) > max_neg:
        negatives = rng.sample(negatives, max_neg)

    labels = positives + negatives
    labels.sort(key=lambda lbl: lbl.timestamp)
    return labels


# ═══════════════════════════════════════════════════════════════
# Phase 15c: Fine-Tuning Loop
# ═══════════════════════════════════════════════════════════════


class FineTuner:
    """Supervised fine-tuning for outcome prediction.

    Two-phase training:
        1. Pre-training: self-supervised (existing Trainer).
        2. Fine-tuning: freeze HGT + memory, train supervised head.

    The supervised head is a bilinear scorer: σ(src^T W dst + b).
    Loss is weighted Binary Cross-Entropy (weight inversely proportional
    to class frequency).

    Args:
        model: Pre-trained HetTGN.
        store: PipelineStore.
        labels: Outcome labels from generate_outcome_labels().
        lr: Learning rate for fine-tuning.
        epochs: Number of fine-tuning epochs.
        freeze_backbone: If True, freeze HGT + combiner (train head only).
    """

    def __init__(
        self,
        model: HetTGN,
        store: PipelineStore,
        labels: list[OutcomeLabel],
        lr: float = 1e-3,
        epochs: int = 10,
        freeze_backbone: bool = True,
    ) -> None:
        self.model = model
        self.store = store
        self.labels = labels
        self.lr = lr
        self.epochs = epochs
        self.freeze_backbone = freeze_backbone
        self._graph_builder = GraphBuilder(store)

    def finetune(self) -> dict[str, list[float]]:
        """Run fine-tuning loop.

        Returns:
            Dict with 'loss' and 'accuracy' per epoch.
        """
        if not self.labels:
            log.warning("No outcome labels — skipping fine-tuning.")
            return {"loss": [], "accuracy": []}

        model = self.model
        model.train()

        # Freeze backbone if requested
        if self.freeze_backbone:
            for name, param in model.named_parameters():
                if "supervised_head" not in name:
                    param.requires_grad = False

        # Optimizer: only trainable parameters
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            log.warning("No trainable parameters — skipping fine-tuning.")
            return {"loss": [], "accuracy": []}
        optimizer = torch.optim.Adam(trainable, lr=self.lr)

        # Compute class weights for BCE
        n_pos = sum(1 for lbl in self.labels if lbl.label == 1)
        n_neg = len(self.labels) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float)

        history: dict[str, list[float]] = {"loss": [], "accuracy": []}

        # Group labels by entity type for efficient embedding lookup
        all_entities = self.store.query_all_entities()
        eid_to_type = {e["entity_id"]: e["entity_type"] for e in all_entities}

        for epoch in range(self.epochs):
            model.reset_memory()

            # Build full graph
            data, id_map, _ = self._graph_builder.build()
            if not data.node_types:
                continue

            embeddings = model(data, id_map)

            # Gather embeddings for label pairs
            src_embs = []
            dst_embs = []
            targets = []

            for lbl in self.labels:
                src_local = id_map.local_id(lbl.src_type, lbl.src_entity_id)
                dst_local = id_map.local_id(lbl.dst_type, lbl.dst_entity_id)
                if src_local is None or dst_local is None:
                    continue
                if lbl.src_type not in embeddings or lbl.dst_type not in embeddings:
                    continue
                if src_local >= embeddings[lbl.src_type].size(0):
                    continue
                if dst_local >= embeddings[lbl.dst_type].size(0):
                    continue

                src_embs.append(embeddings[lbl.src_type][src_local])
                dst_embs.append(embeddings[lbl.dst_type][dst_local])
                targets.append(float(lbl.label))

            if not src_embs:
                continue

            src_tensor = torch.stack(src_embs)
            dst_tensor = torch.stack(dst_embs)
            target_tensor = torch.tensor(targets, dtype=torch.float)

            # Forward through supervised head
            probs = model.predict_outcome(src_tensor, dst_tensor)

            # Weighted BCE loss
            loss = F.binary_cross_entropy(
                probs,
                target_tensor,
                weight=torch.where(target_tensor == 1, pos_weight, torch.ones(1)),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            # Accuracy
            preds = (probs > 0.5).float()
            acc = (preds == target_tensor).float().mean().item()

            history["loss"].append(loss.item())
            history["accuracy"].append(acc)

            log.info(
                "Fine-tune epoch %d/%d — loss: %.4f, accuracy: %.4f",
                epoch + 1,
                self.epochs,
                loss.item(),
                acc,
            )

        # Unfreeze backbone
        if self.freeze_backbone:
            for param in model.parameters():
                param.requires_grad = True

        return history


def evaluate_supervised(
    model: HetTGN,
    store: PipelineStore,
    labels: list[OutcomeLabel],
) -> dict[str, float]:
    """Evaluate supervised head on outcome labels.

    Computes AUROC, precision, recall, and F1 on the given labels.

    Args:
        model: Fine-tuned HetTGN.
        store: PipelineStore.
        labels: Outcome labels (typically from a held-out time period).

    Returns:
        Dict with auroc, precision, recall, f1, num_samples.
    """
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    if not labels:
        return {
            "auroc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "num_samples": 0,
        }

    model.eval()
    graph_builder = GraphBuilder(store)
    data, id_map, _ = graph_builder.build()

    if not data.node_types:
        return {
            "auroc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "num_samples": 0,
        }

    with torch.no_grad():
        embeddings = model(data, id_map)

    all_probs: list[float] = []
    all_labels: list[int] = []

    for lbl in labels:
        src_local = id_map.local_id(lbl.src_type, lbl.src_entity_id)
        dst_local = id_map.local_id(lbl.dst_type, lbl.dst_entity_id)
        if src_local is None or dst_local is None:
            continue
        if lbl.src_type not in embeddings or lbl.dst_type not in embeddings:
            continue
        if src_local >= embeddings[lbl.src_type].size(0):
            continue
        if dst_local >= embeddings[lbl.dst_type].size(0):
            continue

        src_emb = embeddings[lbl.src_type][src_local].unsqueeze(0)
        dst_emb = embeddings[lbl.dst_type][dst_local].unsqueeze(0)

        with torch.no_grad():
            prob = model.predict_outcome(src_emb, dst_emb).item()

        all_probs.append(prob)
        all_labels.append(lbl.label)

    if not all_probs or len(set(all_labels)) < 2:
        return {
            "auroc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "num_samples": len(all_probs),
        }

    preds = [1 if p > 0.5 else 0 for p in all_probs]
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = 0.0

    return {
        "auroc": auroc,
        "precision": precision_score(all_labels, preds, zero_division=0.0),
        "recall": recall_score(all_labels, preds, zero_division=0.0),
        "f1": f1_score(all_labels, preds, zero_division=0.0),
        "num_samples": len(all_probs),
    }
