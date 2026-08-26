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
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    xsnorm_price_feats,
)
from agent.models.gnn.het_tgn import HetTGN
from agent.pipeline.entity import entity_id_from_key
from agent.pipeline.store import PipelineStore

if TYPE_CHECKING:
    from agent.models.gnn.continuous_world_model import ContinuousWorldModel

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


def _listnet_loss(scores: torch.Tensor, targets: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
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


def _vicreg_loss(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Variance + covariance regularization (Bardes et al. 2021, arxiv:2105.04906).

    Prevents dimensional embedding collapse without negative pairs.
    Invariance term omitted — single-view instrument embeddings per window.
    """
    if z.size(0) < 2:
        return z.new_zeros(())
    z = z - z.mean(dim=0)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
    var_loss = F.relu(gamma - std).pow(2).mean()
    n, d = z.shape
    cov = (z.T @ z) / max(n - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = off_diag.pow(2).sum() / d
    return var_loss + cov_loss


def _log_transform_loss(loss: torch.Tensor) -> torch.Tensor:
    """Log transformation for loss-scale balancing (Lin et al. HKUST 2026).

    Transforms L → log(L + 1) so all task losses have comparable scale,
    preventing large-scale losses (e.g. value: 1653) from dominating
    small-scale losses (e.g. return: 210).

    Part of Dual-Balancing MTL: log transform (loss) + max-norm (gradient).
    """
    return torch.log(loss + 1.0)


def _scaled_task_loss(loss: torch.Tensor, cfg: TrainerConfig) -> torch.Tensor:
    """Apply log transform when cfg.use_log_loss is enabled."""
    return _log_transform_loss(loss) if cfg.use_log_loss else loss


def _pcgrad_projection(task_grads: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """PCGrad: Project conflicting gradients (Yu et al. NeurIPS 2020, arXiv:2001.06782).

    When task gradients have negative cosine similarity (conflict), projects
    each gradient onto the normal plane of the other, removing the destructive
    interference component while preserving constructive directions.

    Algorithm:
        for each pair (i, j):
            if dot(g_i, g_j) < 0:  # Conflict
                g_i = g_i - (dot(g_i, g_j) / ||g_j||^2) * g_j  # Project

    Args:
        task_grads: Dict mapping task_name → flattened gradient tensor.

    Returns:
        Dict mapping task_name → projected gradient tensor.
    """
    task_names = list(task_grads.keys())
    n_tasks = len(task_names)
    if n_tasks < 2:
        return task_grads  # No conflicts possible

    # Clone gradients for projection
    projected = {k: v.clone() for k, v in task_grads.items()}

    # Pairwise conflict resolution
    for i in range(n_tasks):
        for j in range(n_tasks):
            if i == j:
                continue
            task_i, task_j = task_names[i], task_names[j]
            g_i = projected[task_i]
            g_j = task_grads[task_j]  # Use original, not projected

            dot_ij = torch.dot(g_i, g_j)
            if dot_ij < 0:  # Conflict detected
                # Project g_i onto normal plane of g_j
                g_j_norm_sq = torch.dot(g_j, g_j).clamp(min=1e-8)
                proj = (dot_ij / g_j_norm_sq) * g_j
                projected[task_i] = g_i - proj

    return projected


def _flatten_param_grads(params: list[torch.nn.Parameter]) -> torch.Tensor:
    """Concatenate per-param gradients; zero-fill when a task did not touch a param."""
    parts: list[torch.Tensor] = []
    for p in params:
        if p.grad is not None:
            parts.append(p.grad.flatten())
        else:
            parts.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
    return torch.cat(parts)


def _scatter_flat_grad(params: list[torch.nn.Parameter], flat: torch.Tensor) -> None:
    """Write a flat gradient vector back into param.grad tensors."""
    offset = 0
    for p in params:
        numel = p.numel()
        p.grad = flat[offset : offset + numel].view_as(p).clone()
        offset += numel


def _pcgrad_optimizer_step(
    optimizer: torch.optim.Optimizer,
    params: list[torch.nn.Parameter],
    task_losses: dict[str, torch.Tensor],
) -> bool:
    """Per-task backward + PCGrad projection. Returns False if fallback needed."""
    active = {
        k: v for k, v in task_losses.items() if v.requires_grad and torch.isfinite(v).all() and v.detach().item() > 0
    }
    if len(active) < 2:
        return False

    task_grads: dict[str, torch.Tensor] = {}
    for name, loss in active.items():
        optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        task_grads[name] = _flatten_param_grads(params).detach().clone()

    projected = _pcgrad_projection(task_grads)
    merged = sum(projected.values())
    optimizer.zero_grad(set_to_none=True)
    _scatter_flat_grad(params, merged)
    return True


class SyntheticGraphGenerator:
    """Generate synthetic entity graphs with known temporal patterns.

    The generated data is inserted directly into a PipelineStore,
    which can then be consumed by GraphBuilder and HetTGN.

    Covers all 12 entity types, the full observation-type registry, and 21 link
    types in the TirraMind schema (expanded in Phase 36; `maritime_area` added
    2026-08-26 when it was registered in ENTITY_TYPES).
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
        num_maritime_areas: int = 0,
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
                "maritime_area": num_maritime_areas,
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
            store.link_entities(cid, co_id, "headquartered_in", "synthetic", confidence=0.9)
            link_count += 1

        # company → country (operates_in) — subset of companies
        for cid in entities.get("company", [])[: max(1, len(entities.get("company", [])) // 2)]:
            co_id = self._rng.choice(entities.get("country", ["default"]))
            store.link_entities(cid, co_id, "operates_in", "synthetic", confidence=0.8)
            link_count += 1

        # company → country (market_authorized_in) — pharma companies
        for cid in entities.get("company", [])[: max(1, len(entities.get("company", [])) // 3)]:
            co_id = self._rng.choice(entities.get("country", ["default"]))
            store.link_entities(cid, co_id, "market_authorized_in", "synthetic", confidence=0.85)
            link_count += 1

        # company → company (lobbies_for) — some business relationships
        comps = entities.get("company", [])
        for i in range(min(3, len(comps) - 1)):
            store.link_entities(comps[i], comps[i + 1], "lobbies_for", "synthetic", confidence=0.7)
            link_count += 1

        # company → company (debtor_of) — creditor relationships
        for i in range(min(2, len(comps) - 1)):
            src, tgt = comps[i], comps[-(i + 1)]
            if src == tgt:
                continue
            store.link_entities(src, tgt, "debtor_of", "synthetic", confidence=0.75)
            link_count += 1

        # company → organization (awarded_by) — government contracts
        for cid in entities.get("company", [])[: max(1, len(entities.get("company", [])) // 2)]:
            if entities.get("organization"):
                org_id = self._rng.choice(entities["organization"])
                store.link_entities(cid, org_id, "awarded_by", "synthetic", confidence=0.9)
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
            store.link_entities(wid, co_id, "exchange_based_in", "synthetic", confidence=0.7)
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
                store.link_entities(wid, inst_id, "trades_instrument", "synthetic", confidence=0.75)
                link_count += 1

        # instrument → company (tracks_issuer) — stocks/ETFs
        for inst_id in entities.get("instrument", [])[: max(1, len(entities.get("instrument", [])) * 2 // 3)]:
            if entities.get("company"):
                cid = self._rng.choice(entities["company"])
                store.link_entities(inst_id, cid, "tracks_issuer", "synthetic", confidence=0.95)
                link_count += 1

        # instrument → country (located_in) — domicile
        for inst_id in entities.get("instrument", []):
            co_id = self._rng.choice(entities.get("country", ["default"]))
            store.link_entities(inst_id, co_id, "located_in", "synthetic", confidence=0.9)
            link_count += 1

        # instrument → country (fx_base_country / fx_quote_country) — subset
        fx_insts = entities.get("instrument", [])[: max(1, len(entities.get("instrument", [])) // 3)]
        for inst_id in fx_insts:
            countries = entities.get("country", [])
            if len(countries) >= 2:
                base_co, quote_co = self._rng.sample(countries, 2)
                store.link_entities(inst_id, base_co, "fx_base_country", "synthetic", confidence=0.95)
                store.link_entities(inst_id, quote_co, "fx_quote_country", "synthetic", confidence=0.95)
                link_count += 2

        # instrument → country (exchange_country) — commodity futures
        for inst_id in entities.get("instrument", [])[-max(1, len(entities.get("instrument", [])) // 3) :]:
            co_id = self._rng.choice(entities.get("country", ["default"]))
            store.link_entities(inst_id, co_id, "exchange_country", "synthetic", confidence=0.95)
            link_count += 1

        # instrument → protocol (tracks_protocol) — crypto instruments
        for inst_id in entities.get("instrument", [])[: max(1, len(entities.get("instrument", [])) // 5)]:
            if entities.get("protocol"):
                proto_id = self._rng.choice(entities["protocol"])
                store.link_entities(inst_id, proto_id, "tracks_protocol", "synthetic", confidence=0.9)
                link_count += 1

        # cftc_contract → instrument (cftc_tracks)
        for cid in entities.get("cftc_contract", []):
            if entities.get("instrument"):
                inst_id = self._rng.choice(entities["instrument"])
                store.link_entities(cid, inst_id, "cftc_tracks", "synthetic", confidence=0.95)
                link_count += 1

        # country → country (sanctioned_under) — geopolitical
        countries = entities.get("country", [])
        if len(countries) >= 2:
            # 1-2 sanction relationships
            for _ in range(min(2, len(countries) - 1)):
                a, b = self._rng.sample(countries, 2)
                store.link_entities(a, b, "sanctioned_under", "synthetic", confidence=0.85)
                link_count += 1

        # domain → company (domain_owned_by) — Phase 36
        for did in entities.get("domain", []):
            if entities.get("company"):
                cid = self._rng.choice(entities["company"])
                store.link_entities(did, cid, "domain_owned_by", "synthetic", confidence=0.8)
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
        entity_type_map: dict[str, str] = {e["entity_id"]: e["entity_type"] for e in all_entities}

        for pattern in self.patterns:
            # Find source observations matching the pattern
            source_obs = [
                o
                for o in all_obs
                if o["entity_type"] == pattern.source_type and o["observation_type"] == pattern.source_obs_type
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
    defi_subsample_frac: float = 1.0
    """Fraction of tvl_change (defi_flows) observations to keep. N1 POC:
    0.05–0.10 so DeFi does not dominate prefetch after GDELT cap."""
    zero_price_feats: bool = False
    """When True, zero the PRICE_FEAT block on instrument nodes (outcome≠input)."""
    embedding_only_return: bool = False
    """When True, predict returns from GNN embeddings only (return_pred_head),
    not return_concat_head / return_raw_head price shortcuts."""
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
    contrastive_log_var_min: float = -1.0
    """Tighter lower clamp for the contrastive/CSRC log-variance only (V50).
    Effective contrastive weight = exp(-log_var) ≤ exp(1) ≈ 2.72 when -1.0,
    preventing auto-tune from exploding contrastive weight (V49 peaked ~6.8)."""
    vicreg_weight: float = 0.0
    """Weight for VICReg variance+covariance regularization on instrument
    embeddings (Bardes et al. 2021). 0 = disabled. Fights dimensional collapse
    alongside CSRC; see LESSONS R-01."""
    vicreg_var_gamma: float = 1.0
    """Minimum per-dimension std threshold γ in VICReg variance term."""
    use_contranorm: bool = False
    """When True, add ContraNorm layers after each HGT layer to prevent
    dimensional collapse (Guo et al. PKU 2023, arXiv:2303.06562).  ContraNorm
    is an architectural fix (zero parameters) that complements VICReg loss-based
    regularization.  Together they provide stronger anti-collapse signal than
    either alone.  See LESSONS R-01 and docs/research/training_efficiency_v61_solution.html."""
    use_log_loss: bool = False
    """When True, apply log transformation to all task losses before combining:
    L_total = sum(w_k * log(L_k + 1)) instead of sum(w_k * L_k).  This is the
    loss-scale balancing component of Dual-Balancing MTL (Lin et al. HKUST 2026).
    Prevents large-scale losses (e.g. value: 1653) from dominating small-scale
    losses (e.g. return: 210) and silencing their gradients.  Applied in both
    fixed-weight and auto-tune branches via _scaled_task_loss()."""
    use_pcgrad: bool = False
    """When True, apply PCGrad (Projecting Conflicting Gradients, Yu et al.
    NeurIPS 2020) to resolve gradient conflicts between tasks.  When task
    gradients have negative cosine similarity, projects each onto the normal
    plane of the other, removing destructive interference.  Complements log_loss
    (which balances scales) by handling conflicting gradient directions.  Model-
    agnostic, zero overhead when gradients don't conflict.  See Yu et al. 2020
    arXiv:2001.06782 and docs/research/training_efficiency_v61_solution.html."""
    return_pred_clamp: float = 5.0
    """Symmetric clamp on return predictions before ListNet/Huber.  Prevents
    extreme logits from destabilising softmax ranking.  V64 showed hard ±5
    clamp causes saturation (pred_std→0).  Set to 0 to disable; V65+ uses 50."""
    use_concat_batchnorm: bool = False
    """When True, apply LayerNorm on concat-head input [xsnorm_raw || gnn_emb]
    before return_concat_head to keep activations bounded without hard clamp."""
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
    use_forward_returns: bool = True
    """When True, replace the daily log_return supervision target with the
    N-day (forward_return_horizon) forward return computed from instrument_daily
    close prices.  This is the correct oracle for cross-sectional IC:
    we want the model to learn which instruments outperform over the next
    21 trading days, not what yesterday's return was.

    Falls back to daily log_return when no forward return is found in the
    lookup (e.g. instrument is delisted before the horizon date).

    Reference: Lewellen (2015) "The Cross-section of Expected Stock Returns";
    forward holding-period returns are the standard IC denominator."""
    forward_return_horizon: int = 21
    """Holding period in TRADING DAYS for forward return computation.
    21 trading days ≈ 1 calendar month.  Aligns with the monthly rebalancing
    cadence assumed in the walk-forward backtest."""
    use_residual_returns: bool = False
    """When True, cross-sectionally demean the forward return targets before
    computing the return loss.  This strips out the market-wide component
    (which the model cannot predict from entity-level features) and forces
    the model to learn only cross-sectional variation — exactly what
    Spearman IC measures.  Has no effect when return_weight=0."""

    use_cde: bool = False
    """When True, replace the per-window GRU memory update with a Neural
    Controlled Differential Equation (CDE) integration over each node's
    event path (Kidger et al. NeurIPS 2020).  Requires torchcde and
    torchdiffeq to be installed (pip install torchdiffeq torchcde).
    Falls back to GRU when libraries are unavailable.  Default False
    preserves existing behaviour with no performance regression."""

    use_signatures: bool = False
    """When True, append depth-3 path signature features (Idea 2) to each
    node's feature vector.  The signature encodes the shape, curvature, and
    higher-order texture of the entity's event stream — a provably universal
    feature map from rough path theory (Lyons & McLeod 2022, Theorem 3.1).
    Adds SIGNATURE_DIM=39 extra dims to every node type's input features.
    Pure-PyTorch implementation — no external libraries required.
    Default False preserves existing in_channels dimensions."""

    use_mamba: bool = False
    """When True, replace the per-window GRU memory update with a Mamba
    selective State Space Model block (Gu & Dao, NeurIPS 2023).  Mamba
    processes the full sequence of messages a node receives in each window
    with input-selective state transitions — it can learn to forget routine
    pings and retain rare anomalies across arbitrarily long sequences.
    Requires mambapy (pip install mambapy, pure-PyTorch, no CUDA kernels).
    Falls back silently to GRU if mambapy is unavailable.
    Cannot be combined with use_cde; Mamba takes priority if both are True.
    Default False preserves existing GRU behaviour."""

    use_ts2vec: bool = False
    """When True, pretrain a TS2Vec contrastive encoder (Idea 5) on all
    entity time series before building the GNN (Yue et al., AAAI 2022).
    The encoder requires no labels — it learns universal embeddings via
    hierarchical contrastive loss across timestamp and instance levels.
    Adds ts2vec_dim extra dimensions to every node type's feature vector,
    giving cold-start entities a richer initial representation and enabling
    cross-domain similarity structure in the learned embedding space.
    Requires ts2vec (pip install ts2vec).  Default False leaves in_channels
    unchanged."""

    ts2vec_dim: int = 32
    """TS2Vec output embedding dimension appended to every node's features.
    Only used when use_ts2vec=True.  Larger values capture more temporal
    structure but increase HetTGN input dim and training cost.  Default 32."""

    ts2vec_n_iters: int = 200
    """TS2Vec pretraining iterations per entity type.  200 is the paper
    default; reduce to 50-100 for fast development; increase to 500+ for
    production-quality embeddings.  Only used when use_ts2vec=True."""

    use_wasserstein: bool = False
    """When True, run the Wasserstein Distribution Shift Monitor (Idea 8)
    before building the GNN.  For each data source tool, computes a
    normalised Wasserstein-1 distance between the last short_days window
    and the long_days baseline.  Drift scores are stored as
    ``wasserstein.<tool>.drift`` signals in the pipeline store and
    alarm warnings are logged for tools that exceed the threshold.
    No external dependencies — uses pure-numpy W1.  Optionally uses
    Sinkhorn distance if ``pip install POT`` is available.
    Default False — monitoring is opt-in."""

    wasserstein_threshold: float = 1.0
    """Normalised W1 drift score above which an alarm is raised.
    The score is divided by the baseline standard deviation, so threshold=1.0
    means one standard-deviation shift in the daily activity distribution.
    Lower = more sensitive.  Only used when use_wasserstein=True."""

    wasserstein_short_days: int = 30
    """Short rolling window length in days for distribution comparison.
    Only used when use_wasserstein=True."""

    use_hawkes: bool = False
    """When True, run the Neural Hawkes Process encoder (Idea 6) before
    building the GNN.  Trains a continuous-time LSTM on the full observation
    history, then predicts per-event-type intensities for the next
    ``hawkes_forecast_hours`` hours.  Results are stored as
    ``hawkes.<event_type>.intensity_<H>h`` signals in the pipeline store so
    the ConvergenceDetector can use them as learned leading indicators.
    Requires PyTorch (already a dependency).  Default False."""

    hawkes_hidden_dim: int = 64
    """LSTM hidden dimension for the Neural Hawkes model.  Larger = more
    expressive but slower.  Default 64 matches Mei & Eisner (2017)."""

    hawkes_n_iters: int = 200
    """Training iterations for the Neural Hawkes LSTM.  200 is sufficient
    for most datasets; increase to 500+ for production-quality models.
    Only used when use_hawkes=True."""

    hawkes_forecast_hours: float = 72.0
    """Forecast horizon in hours for the intensity probability computation.
    P(event_k in [0, T]) ≈ 1 − exp(−λ*_k · T).  Default 72h."""

    use_vine_copula: bool = False
    """When True, run the Vine Copula Tail-Dependence Encoder (Idea 7) before
    building the GNN.  Fits bivariate Clayton + Gumbel copulas to every linked
    entity pair and stores λ_L (lower tail) and λ_U (upper tail) dependence
    coefficients as pipeline signals.  These capture co-crash and co-spike
    probabilities that linear correlation cannot express.  Results are stored
    as ``copula.<pair_key>.lambda_lower/upper`` signals.  Default False."""

    vine_copula_min_obs: int = 20
    """Minimum aligned time-bins required to fit a copula for a pair.
    Pairs with fewer joint observations are skipped.
    Only used when use_vine_copula=True."""

    vine_copula_n_bins: int = 60
    """Number of time bins to divide the lookback window into.
    With lookback_days=365 and n_bins=60, each bin spans ≈6 days.
    Only used when use_vine_copula=True."""

    use_gdn: bool = False
    """When True, run the Graph Deviation Network monitor (Idea 10) before
    building the GNN.  Trains a GDN on TirraMind's entity observation history
    to learn expected inter-entity co-movement patterns, then scores each
    entity's deviation from its graph-predicted behaviour.  High deviation
    scores indicate structural breaks (bankruptcy, port closure, sanctions)
    that precede market events by days.  Results stored as
    ``graph_structure.<entity_id>.deviation`` signals.  Default False."""

    gdn_hidden_dim: int = 64
    """GDN graph attention hidden dimension.  Default 64."""

    gdn_n_iters: int = 100
    """GDN training iterations.  Default 100."""

    gdn_anomaly_threshold: float = 3.0
    """Normalised deviation z-score above which is_anomaly=True.  Default 3.0
    (roughly 3-sigma; lower = more sensitive).  Only used when use_gdn=True."""

    use_entity_resolution: bool = False
    """When True, run probabilistic entity resolution (Idea 3) via Splink
    before building the model.  Identifies cross-source duplicates —
    "Apple Inc." / "AAPL" / "Apple Computer" — and stores them as
    same_as edges in entity_links.  The GNN then propagates information
    across duplicate nodes, increasing effective graph connectivity.
    Requires splink>=4.0 (pip install splink).  Runs once per build_model()
    call; subsequent train() calls reuse the stored same_as links.
    Default False leaves entity deduplication to the caller."""

    entity_resolution_threshold: float = 0.9
    """Minimum Fellegi-Sunter match probability (0-1) for storing a
    probabilistic same_as link.  Only used when use_entity_resolution=True.
    Default 0.9 keeps precision high; lower to increase recall."""

    portfolio_delta: float = 2.5
    """Black-Litterman risk aversion coefficient (δ).
    Higher = less aggressive weighting toward high-return assets.  Default 2.5."""

    portfolio_tilt_factor: float = 0.5
    """BL-return tilt magnitude α ∈ [0,1].  0 = pure HRP; 1 = strong tilt
    toward high-return assets.  Default 0.5."""

    portfolio_turnover_lambda: float = 0.3
    """Turnover smoothing λ: w_final = (1-λ)·w_new + λ·w_prev.
    0 = no smoothing; 1 = freeze at previous weights.  Default 0.3."""

    portfolio_lookback_days: int = 365
    """Historical price window (days) for covariance estimation.  Default 365."""

    portfolio_min_history: int = 20
    """Minimum price bins required per asset to include it.  Default 20."""

    use_attribution: bool = False
    """When True, run Barra-Style Signal Attribution (Idea 12) after
    inference.  Decomposes each instrument node's GNN prediction into
    per-source-type fractional attention contributions — answering
    "Why did this prediction happen?".  Disabled by default to avoid
    any CPU overhead during training.  Results stored as
    ``attribution.{entity_id}.{src_type}`` pipeline signals."""

    attribution_max_entities: int = 200
    """Hard CPU-safety cap for attribution.  At most this many instrument
    entities are attributed per call.  Default 200."""

    attribution_min_attention: float = 0.0
    """Source types with total attention below this threshold are collapsed
    into an 'other' bucket.  0 = keep all.  Default 0.0."""

    use_data_catalog: bool = False
    """When True, enable the Data Governance Catalog (Idea 13).
    Tracks freshness SLAs for all 51+ data sources against the
    pipeline store, emits ``catalog.{tool}.freshness_hours`` and
    ``catalog.{tool}.sla_breach`` signals.  Disabled by default."""

    catalog_max_tools: int = 200
    """Hard cap on how many tools are checked per freshness scan.
    Default 200 (covers all current tools with headroom)."""

    catalog_sla_multiplier: float = 1.0
    """Scale factor applied to every tool's SLA threshold.
    1.0 = use manifest defaults.  Set >1 to relax, <1 to tighten."""

    # ── Weights & Biases streaming (optional) ──────────────────────────────
    wandb_project: str | None = None
    """W&B project name. If None, wandb logging is disabled."""
    wandb_run_name: str | None = None
    """W&B run display name (e.g. 'h-a-epoch31-40'). Auto-generated if None."""
    wandb_tags: list[str] | None = None
    """Optional list of tags for the W&B run (e.g. ['h-a', 'phase43'])."""

    freeze_backbone: bool = False
    """When True, freeze all parameters except return_raw_head and train only
    that head.  This completely isolates the ranking signal from obs_type
    explosion spikes that otherwise starve return_raw_head gradients through
    the shared gradient clip.  Use after a warm-start checkpoint once the
    backbone embeddings are stable."""

    use_concat_head: bool = False
    """Option B: when True, use return_concat_head([xsnorm_raw || gnn_emb])
    as the return predictor instead of return_raw_head(raw_only).  Creates a
    direct gradient path from the GNN backbone into the ranking signal.
    Use when GNN embeddings show positive IC delta over the raw head."""

    use_csrc_loss: bool = True
    """When True, use Cross-Sectional Ranking Contrastive loss instead of
    entity-identity contrastive.  Positive pairs = same return decile,
    negative pairs = opposite deciles.  Forces backbone to encode
    return-relevant features.  Default True (replaces broken contrastive)."""

    csrc_temperature: float = 0.1
    """Temperature for CSRC InfoNCE loss.  Lower = sharper distinction
    between deciles.  0.1 is a strong separation (same as ListNet temp)."""

    csrc_n_deciles: int = 5
    """Number of return deciles for CSRC positive/negative pair assignment.
    5 deciles gives ~18 instruments per decile with 89 instruments."""

    # ── M1: Continuous-Time Heterogeneous World Model ─────────────────────
    use_continuous_world_model: bool = False
    """When True, replace the per-window GRU/CDE memory update with the M1
    Neural SDE integration (ContinuousWorldModel).  The CDE solve runs once
    per entity per window before the regular HetTGN forward pass.
    Requires torchcde>=0.2 to be installed.  Default False preserves the
    existing update_memory_from_events() behaviour with no regression."""

    cwm_curriculum_phase: str = "B"
    """M1 curriculum phase controlling which components are active.
    'B' — CDE only, no graph context, no Mamba, no diffusion (safest start).
    'C' — add Mamba drift context and graph message from prev window memory.
    'D' — add Hawkes hidden states + path signatures to control path Z(t).
    'E' — add DiagonalDiffusionHead stochastic term + KL loss.
    Progress from B→C→D→E over consecutive training runs once each phase
    converges (IC stabilises).  Default 'B'."""

    cwm_n_euler_steps: int = 20
    """Number of Euler-Maruyama integration steps per window.
    More steps = better approximation of the CDE solution, slower training.
    20 is a practical default for daily windows; reduce to 5 for fast
    debugging, increase to 50 for final production runs."""

    cwm_ctrl_time_dim: int = 16
    """Time2Vec output channels for the CDE control path Z(t).
    Higher = richer temporal encoding, larger d_z, slower.  Default 16."""

    cwm_ctrl_msg_dim: int = 32
    """Projected message dimension for the CDE control path Z(t).
    d_z = cwm_ctrl_time_dim + cwm_ctrl_msg_dim [+ sig_dim if Phase D+]."""

    cwm_lambda_kl: float = 0.01
    """Weight for the KL divergence regularisation loss from the M1
    diffusion head (Phase E only).  Annealed from 0 → cwm_lambda_kl over
    cwm_kl_warmup_epochs epochs.  Has no effect when Phase < E."""

    cwm_kl_warmup_epochs: int = 10
    """Number of epochs over which to anneal the KL weight from 0 to
    cwm_lambda_kl.  Prevents KL collapse in early training (Phase E)."""

    cwm_sig_proj_dim: int = 4
    """Signature path projection dimension for Phase D.
    Log-sig output dim = iisig.logsiglength(cwm_sig_proj_dim, depth).
    Default 4; larger values give richer signatures but slower training."""

    cwm_sig_depth: int = 3
    """Log-signature depth for Phase D.  Requires iisignature; falls back
    to depth-2 manual implementation.  Depth-3 gives 30 extra d_z dims
    for proj_dim=4.  Default 3."""


# ═══════════════════════════════════════════════════════════════
# Forward return lookup (Phase 47 — A2 fix)
# ═══════════════════════════════════════════════════════════════


def _build_forward_return_lookup(
    observations: list[dict],
    horizon_days: int = 21,
) -> dict[tuple[str, int], float]:
    """Backward-compatible wrapper — canonical impl: agent.quant.forward_returns."""
    from agent.quant.forward_returns import build_forward_return_lookup

    return build_forward_return_lookup(observations, horizon_days=horizon_days)


# ═══════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════


def _resolve_torch_device(device: str | torch.device) -> torch.device:
    """Map config device to a runtime device (CPU when CUDA unavailable)."""
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        log.warning("config.device=%r but CUDA unavailable; using CPU.", device)
        return torch.device("cpu")
    return dev


def _checkpoint_map_location(device: torch.device | None = None) -> torch.device:
    """Device target for torch.load (CPU when no GPU)."""
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_state_dict_skip_shape_mismatch(
    module: torch.nn.Module,
    state: dict[str, torch.Tensor],
    *,
    log_prefix: str = "",
) -> tuple[list[str], list[str], list[str]]:
    """Load weights, skipping keys absent from the model or with shape mismatches.

    Used when checkpoint obs_type cardinality drifts vs current pipeline.db
    (e.g. 46 vs 48 observation types) but return/embedding heads still load.
    """
    model_sd = module.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped_shape: list[str] = []
    for key, tensor in state.items():
        if key not in model_sd:
            continue
        if model_sd[key].shape != tensor.shape:
            skipped_shape.append(key)
            continue
        compatible[key] = tensor
    missing, unexpected = module.load_state_dict(compatible, strict=False)
    if skipped_shape:
        log.warning(
            "%sSkipped %d keys due to shape mismatch (first: %s).",
            log_prefix,
            len(skipped_shape),
            ", ".join(skipped_shape[:6]),
        )
    return missing, unexpected, skipped_shape


class CheckpointSchemaDriftError(RuntimeError):
    """Checkpoint weights cannot encode the current feature schema."""


def _describe_checkpoint_schema_drift(
    *,
    checkpoint: dict,
    model: torch.nn.Module,
    skipped: list[str],
) -> str | None:
    """Human-readable diagnosis of checkpoint-vs-live feature drift, or None.

    A skipped `type_projections.<t>.weight` means that entity type's input
    width changed since training, so the loaded model carries a randomly
    initialised projection for it — the forward pass will then fail with an
    opaque "mat1 and mat2 shapes cannot be multiplied" deep inside torch.
    Surface it here instead, naming the type and both widths.
    """
    if not skipped:
        return None

    model_sd = model.state_dict()
    ckpt_sd = checkpoint.get("model_state_dict") or {}
    drift: list[str] = []
    for key in skipped:
        if not key.startswith("type_projections.") or not key.endswith(".weight"):
            continue
        etype = key.split(".")[1]
        live_w = model_sd.get(key)
        ckpt_w = ckpt_sd.get(key)
        if live_w is None or ckpt_w is None:
            continue
        drift.append(f"    {etype}: trained_weights={ckpt_w.shape[1]} " f"expected_by_model={live_w.shape[1]}")

    if not drift:
        return None
    return (
        "Checkpoint's trained weights disagree with the architecture rebuilt "
        "from its own metadata — the feature schema changed since training. "
        "Input widths per entity type:\n"
        + "\n".join(sorted(drift))
        + "\nThe affected projections are randomly initialised, so inference "
        "will produce garbage or fail with a shape error. Retrain against the "
        "current schema (see validate_schema_against_store)."
    )


def _warn_on_checkpoint_schema_drift(
    *,
    checkpoint: dict,
    model: torch.nn.Module,
    skipped: list[str],
    missing: list[str],
    path: Path,
) -> None:
    """Log an actionable diagnosis when a checkpoint no longer matches the schema.

    This is the guard that was missing while the instrument feature vector grew
    23 → 49: `load_model` skipped the mismatched projection, logged a generic
    "skipped N keys" line, and let inference fail later with an opaque torch
    error that named no entity type. Warn (not raise) so deliberate partial-load
    workflows still work; the DAG chain validates strictly before running.
    """
    diagnosis = _describe_checkpoint_schema_drift(checkpoint=checkpoint, model=model, skipped=skipped)
    if diagnosis:
        log.warning("load_model(%s): %s", path.name, diagnosis)


def _het_tgn_kwargs_from_checkpoint(
    checkpoint: dict,
    config: TrainerConfig,
) -> dict[str, int | bool]:
    """Resolve HetTGN head flags so load_model matches build_model architecture.

    Older checkpoints omit ``use_concat_head`` in ``config`` but still carry
    ``return_concat_head.*`` weights — infer from state dict keys so eval can
    register GNN-ConcatReturnHead.
    """
    in_channels = checkpoint["in_channels"]
    state = checkpoint.get("model_state_dict") or {}
    keys = state.keys() if isinstance(state, dict) else []

    use_concat_head = getattr(config, "use_concat_head", False) or any(
        k.startswith("return_concat_head.") for k in keys
    )
    use_contranorm = getattr(config, "use_contranorm", False) or any(k.startswith("contranorm_layers.") for k in keys)
    instrument_raw_dim = int(in_channels.get("instrument", 0))

    if use_concat_head and instrument_raw_dim <= 0:
        log.warning(
            "Checkpoint has return_concat_head weights but instrument in_channels=0; "
            "concat head will not be instantiated."
        )
        use_concat_head = False

    config.use_concat_head = use_concat_head
    config.use_contranorm = use_contranorm

    return {
        "instrument_raw_dim": instrument_raw_dim,
        "use_concat_head": use_concat_head,
        "use_contranorm": use_contranorm,
    }


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
        self._device = _resolve_torch_device(self.config.device)
        self._model: HetTGN | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._log_vars: dict[str, torch.nn.Parameter] | None = None
        self._graph_builder = GraphBuilder(
            store,
            zero_price_feats=self.config.zero_price_feats,
        )
        self._ts2vec_embeddings: dict | None = None
        self._ewc_state: EWCState | None = None
        self._cwm: ContinuousWorldModel | None = None  # M1: set by build_model()
        self._wandb_run = None  # W&B run handle; initialised by train() if wandb_project is set
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

        # Idea 8: run Wasserstein distribution shift monitor
        self._wasserstein_results: dict = {}
        if cfg.use_wasserstein:
            from agent.convergence.wasserstein_monitor import (
                WassersteinMonitor,
            )

            _wmon = WassersteinMonitor(
                short_days=cfg.wasserstein_short_days,
                alarm_threshold=cfg.wasserstein_threshold,
            )
            self._wasserstein_results = _wmon.run(self.store)
            _wmon.store_results(self._wasserstein_results, self.store)
            _n_alarms = sum(1 for r in self._wasserstein_results.values() if r.is_alarm)
            log.info(
                "Wasserstein monitor: %d tools scored, %d alarm(s)",
                len(self._wasserstein_results),
                _n_alarms,
            )

        # Idea 6: run Neural Hawkes Process encoder
        self._hawkes_results: dict = {}
        if cfg.use_hawkes:
            from agent.convergence.neural_hawkes import (
                NeuralHawkesEncoder,
            )

            _hawkes_enc = NeuralHawkesEncoder(
                hidden_dim=cfg.hawkes_hidden_dim,
                n_iters=cfg.hawkes_n_iters,
                forecast_hours=cfg.hawkes_forecast_hours,
            )
            self._hawkes_results = _hawkes_enc.run(self.store)
            _hawkes_enc.store_results(self._hawkes_results, self.store)
            log.info(
                "Neural Hawkes: predicted intensities for %d event types.",
                len(self._hawkes_results),
            )

        # Idea 7: run Vine Copula tail-dependence encoder
        self._copula_results: dict = {}
        if cfg.use_vine_copula:
            from agent.convergence.vine_copula import VineCopulaEncoder  # noqa: PLC0415

            _vine = VineCopulaEncoder(
                min_joint_obs=cfg.vine_copula_min_obs,
                n_bins=cfg.vine_copula_n_bins,
            )
            self._copula_results = _vine.run(self.store)
            _vine.store_results(self._copula_results, self.store)
            log.info(
                "Vine Copula: fitted copulas for %d entity pairs.",
                len(self._copula_results),
            )

        # Idea 10: run Graph Deviation Network monitor
        self._gdn_results: dict = {}
        if cfg.use_gdn:
            from agent.convergence.gdn_monitor import GDNMonitor  # noqa: PLC0415

            _gdn = GDNMonitor(
                hidden_dim=cfg.gdn_hidden_dim,
                n_iters=cfg.gdn_n_iters,
                anomaly_threshold=cfg.gdn_anomaly_threshold,
            )
            self._gdn_results = _gdn.run(self.store)
            _gdn.store_results(self._gdn_results, self.store)
            _n_anomalies = sum(1 for r in self._gdn_results.values() if r.is_anomaly)
            log.info(
                "GDN monitor: scored %d entities, %d anomaly/anomalies.",
                len(self._gdn_results),
                _n_anomalies,
            )

        # Idea 3: run entity resolution before building the graph
        if cfg.use_entity_resolution:
            from agent.pipeline.entity_resolver import resolve_entities  # noqa: PLC0415

            n_links = resolve_entities(
                self.store,
                match_threshold=cfg.entity_resolution_threshold,
            )
            log.info("Entity resolution complete: %d new same_as links", n_links)

        # Idea 5: pretrain TS2Vec encoder on full observation history
        if cfg.use_ts2vec:
            from agent.models.gnn.ts2vec_encoder import TS2VecEncoder  # noqa: PLC0415

            _ts2vec_device = cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
            _ts2vec_enc = TS2VecEncoder(
                output_dims=cfg.ts2vec_dim,
                n_iters=cfg.ts2vec_n_iters,
                device=_ts2vec_device,
            )
            self._ts2vec_embeddings = _ts2vec_enc.fit_and_encode(self.store)
            log.info(
                "TS2Vec pretraining complete — %d entity types encoded",
                len(self._ts2vec_embeddings),
            )

        # Build full graph to determine metadata and node counts
        data, id_map, _ = self._graph_builder.build(
            use_signatures=cfg.use_signatures,
            ts2vec_embeddings=self._ts2vec_embeddings,
            ts2vec_dim=cfg.ts2vec_dim if cfg.use_ts2vec else 0,
        )

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
            use_cde=cfg.use_cde,
            use_mamba=cfg.use_mamba,
            instrument_raw_dim=in_channels.get("instrument", 0),
            use_concat_head=getattr(cfg, "use_concat_head", False),
            use_contranorm=getattr(cfg, "use_contranorm", False),
        )

        # Learnable log-variance params for uncertainty-weighted multi-task
        # loss (Kendall et al. 2018).  Initialised from config weights:
        #   fixed_weight ≈ exp(-log_var) → log_var ≈ -ln(fixed_weight)
        # Stored as plain Parameters (not part of the model) so they can
        # be independently inspected / serialised.
        self._log_vars: dict[str, torch.nn.Parameter] | None = None
        if cfg.auto_tune_loss_weights:
            self._log_vars = {
                "obs_type": torch.nn.Parameter(torch.tensor(-math.log(max(cfg.obs_type_weight, 1e-6)))),
                "time_delta": torch.nn.Parameter(torch.tensor(-math.log(max(cfg.time_delta_weight, 1e-6)))),
                "contrastive": torch.nn.Parameter(torch.tensor(-math.log(max(cfg.contrastive_weight, 1e-6)))),
                "value": torch.nn.Parameter(torch.tensor(-math.log(max(cfg.value_weight, 1e-6)))),
                "return": torch.nn.Parameter(torch.tensor(-math.log(max(cfg.return_weight, 1e-6)))),
            }

        # Move model and log-var tensors to target device
        self._model = self._model.to(self._device)
        if self._log_vars is not None:
            self._log_vars = {k: torch.nn.Parameter(v.to(self._device)) for k, v in self._log_vars.items()}

        # M1: Instantiate ContinuousWorldModel when requested
        self._cwm: ContinuousWorldModel | None = None
        if cfg.use_continuous_world_model:
            from agent.models.gnn.continuous_world_model import (  # noqa: PLC0415
                ContinuousWorldModel,
            )
            from agent.models.gnn.signature_path import (  # noqa: PLC0415
                SignaturePathBuilder,
            )

            _phase = cfg.cwm_curriculum_phase.upper()
            _use_sigs = _phase in ("D", "E")
            _use_mamba = _phase in ("C", "D", "E")
            _use_diff = _phase == "E"

            _sig_builder = None
            if _use_sigs:
                _sig_builder = SignaturePathBuilder(
                    message_dim=(cfg.ctrl_msg_dim if hasattr(cfg, "ctrl_msg_dim") else cfg.cwm_ctrl_msg_dim),
                    proj_dim=cfg.cwm_sig_proj_dim,
                    depth=cfg.cwm_sig_depth,
                ).to(self._device)

            _mamba_enc = None
            if _use_mamba and cfg.use_mamba:
                _mamba_enc = getattr(self._model, "memory_encoder", None)

            self._cwm = ContinuousWorldModel(
                hidden_dim=cfg.hidden_dim,
                ctrl_time_dim=cfg.cwm_ctrl_time_dim,
                ctrl_msg_dim=cfg.cwm_ctrl_msg_dim,
                n_euler_steps=cfg.cwm_n_euler_steps,
                use_signatures=_use_sigs,
                use_mamba_ctx=_use_mamba,
                use_diffusion=_use_diff,
                mamba_encoder=_mamba_enc,
                sig_builder=_sig_builder,
            ).to(self._device)

            log.info(
                "ContinuousWorldModel built — phase=%s d_z=%d euler_steps=%d",
                _phase,
                self._cwm.d_z,
                cfg.cwm_n_euler_steps,
            )

        # Optionally freeze backbone — only return_raw_head trains
        if getattr(cfg, "freeze_backbone", False):
            frozen, trainable = 0, 0
            for name, param in self._model.named_parameters():
                if "return_raw_head" in name:
                    param.requires_grad = True
                    trainable += 1
                else:
                    param.requires_grad = False
                    frozen += 1
            log.info(
                "freeze_backbone=True: froze %d params, keeping %d return_raw_head params trainable",
                frozen,
                trainable,
            )

        # Build optimizer — include log-var params and CWM params when active
        opt_params = [p for p in self._model.parameters() if p.requires_grad]
        if self._log_vars is not None:
            opt_params.extend(self._log_vars.values())
        if self._cwm is not None:
            opt_params.extend(p for p in self._cwm.parameters() if p.requires_grad)
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
            all_obs = [o for o in all_obs if o.get("observed_at", 0.0) >= self.config.obs_since]
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
            (torch.tensor(obs_types, dtype=torch.long) if obs_types else torch.zeros(0, dtype=torch.long)),
            (torch.tensor(time_deltas, dtype=torch.float) if time_deltas else torch.zeros(0)),
            (torch.tensor(values, dtype=torch.float) if values else torch.zeros(0)),
        )

    def _contrastive_loss(
        self,
        embeddings: dict[str, torch.Tensor],
        id_map: IDMap,
    ) -> torch.Tensor:
        """Margin-based contrastive loss on entity links — DISABLED.

        Replaced by _cross_sectional_ranking_contrastive() which uses
        return-decile-based pairs instead of entity-identity links.
        The old loss collapsed all instrument embeddings to a single
        cluster because it only enforced same-entity similarity, never
        cross-instrument differentiation.

        Kept as a no-op for backward compatibility with checkpoints
        that still reference 'contrastive' in their loss keys.
        """
        return torch.tensor(0.0, device=self._device)

    def _cross_sectional_ranking_contrastive(
        self,
        instrument_embeddings: torch.Tensor,
        return_targets: torch.Tensor,
        temperature: float = 0.1,
        n_deciles: int = 5,
    ) -> torch.Tensor:
        """Cross-Sectional Ranking Contrastive (CSRC) loss.

        Replaces entity-identity contrastive with return-rank-based pairs:
          - Positive pair: instruments in the SAME return decile
          - Negative pair: instruments in OPPOSITE deciles (top vs bottom)

        This forces the backbone to encode return-relevant features because
        high-return instruments must have different embeddings from low-return
        ones, while similar-return instruments cluster together.

        Uses InfoNCE-style loss with decile-based positive/negative sampling.
        Only active when >= 2 deciles have >= 2 instruments each.

        Reference: Inspired by Contrastive Learning of Asset Embeddings
        (ICAIF 2024, arXiv:2407.18645) — return-correlation-based pairs
        preserve cross-sectional structure.
        """
        n = instrument_embeddings.size(0)
        if n < 4:
            return torch.tensor(0.0, device=self._device)

        # Filter to finite targets only
        finite_mask = torch.isfinite(return_targets)
        if finite_mask.sum() < 4:
            return torch.tensor(0.0, device=self._device)

        emb = instrument_embeddings[finite_mask]
        tgt = return_targets[finite_mask]
        n_valid = emb.size(0)

        # Assign deciles (1 = lowest return, n_deciles = highest)
        # Use quantile-based binning for robustness to outliers
        sorted_tgt, _ = tgt.sort()
        decile_size = max(2, n_valid // n_deciles)
        n_actual_deciles = min(n_deciles, n_valid // decile_size)
        if n_actual_deciles < 2:
            return torch.tensor(0.0, device=self._device)

        decile_assignments = torch.zeros(n_valid, dtype=torch.long, device=self._device)
        for d in range(n_actual_deciles):
            start = d * decile_size
            end = start + decile_size if d < n_actual_deciles - 1 else n_valid
            decile_assignments[start:end] = d

        # L2-normalise embeddings for cosine similarity
        emb_n = F.normalize(emb, p=2, dim=-1)

        # Compute cosine similarity matrix
        sim = torch.mm(emb_n, emb_n.t()) / temperature

        # Build positive mask: same decile, exclude self
        pos_mask = torch.zeros((n_valid, n_valid), device=self._device)
        for d in range(n_actual_deciles):
            in_decile = (decile_assignments == d).nonzero(as_tuple=True)[0]
            if in_decile.size(0) >= 2:
                for i in in_decile:
                    pos_mask[i, in_decile] = 1.0
        pos_mask.fill_diagonal_(0.0)

        # Build negative mask: top decile vs bottom decile
        neg_mask = torch.zeros((n_valid, n_valid), device=self._device)
        if n_actual_deciles >= 2:
            top_decile = (decile_assignments == n_actual_deciles - 1).nonzero(as_tuple=True)[0]
            bot_decile = (decile_assignments == 0).nonzero(as_tuple=True)[0]
            for i in top_decile:
                neg_mask[i, bot_decile] = 1.0
            for i in bot_decile:
                neg_mask[i, top_decile] = 1.0

        # If no valid pairs, fall back to all-different-decile negatives
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=self._device)

        # InfoNCE: -log( sum(exp(pos)) / (sum(exp(pos)) + sum(exp(neg))) )
        pos_exp = (sim * pos_mask).exp()
        neg_exp = (sim * neg_mask).exp()
        pos_sum = pos_exp.sum(dim=1)
        neg_sum = neg_exp.sum(dim=1)

        # Only compute loss for rows that have both positive and negative pairs
        valid_rows = (pos_sum > 0) & (neg_sum > 0)
        if valid_rows.sum() == 0:
            return torch.tensor(0.0, device=self._device)

        loss_per_row = -torch.log(pos_sum[valid_rows] / (pos_sum[valid_rows] + neg_sum[valid_rows]))
        return loss_per_row.mean()

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
                import dataclasses

                import wandb as _wandb  # type: ignore[import]

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
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"epoch_{cfg.resume_from_epoch:03d}.pt")
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
                ckpt_state = dict(ckpt["model_state"])  # shallow copy — don't mutate ckpt
                MEMORY_BUFFER_KEYS = ("memory.memory", "memory.last_update")
                saved_buffers = {k: ckpt_state.pop(k) for k in MEMORY_BUFFER_KEYS if k in ckpt_state}
                # Filter out keys whose shape doesn't match the current model.
                # strict=False already handles missing/extra keys, but PyTorch
                # still raises RuntimeError on size mismatches even with strict=False.
                current_state = model.state_dict()
                shape_mismatches = [
                    k for k, v in ckpt_state.items() if k in current_state and v.shape != current_state[k].shape
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
                    # Raw bypass head is new (not in the old checkpoint).  The old
                    # return_pred_head was trained on collapsed embeddings and
                    # outputs near-constant, slightly anti-correlated scores.
                    # Zero its output layer so both heads start from the entropy
                    # floor instead of above it — cleaner, faster convergence.
                    if model.return_raw_head is not None and any(k.startswith("return_raw_head.") for k in missing):
                        torch.nn.init.zeros_(model.return_pred_head[-1].weight)
                        torch.nn.init.zeros_(model.return_pred_head[-1].bias)
                        log.info(
                            "Zeroed return_pred_head output layer — raw bypass head "
                            "is new; both heads now start from entropy floor."
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
                # ── Align history lengths ────────────────────────────
                # If a key was added mid-training (e.g. "return" in Phase 47),
                # its history is shorter than "total".  Pad with NaN at the
                # FRONT so that history[k][i] corresponds to epoch i+1 for all k.
                # Without this, the loss-curve table renders return values
                # shifted up by the gap size (observed: 10-row shift in V40).
                _n_total = len(history.get("total", []))
                for _hk in ("return", "value", "contrastive", "time_delta", "obs_type"):
                    _h = history.get(_hk)
                    if _h is not None and len(_h) < _n_total:
                        _pad = _n_total - len(_h)
                        history[_hk] = [float("nan")] * _pad + _h
                start_epoch = cfg.resume_from_epoch
                log.info(
                    "Resumed from checkpoint %s (epoch %d)",
                    ckpt_path,
                    start_epoch,
                )
                # ── Restore EWC state from sidecar if present ─────────────
                # The per-epoch checkpoint does NOT contain EWC state (Fisher
                # is only computed after the full block finishes).  A separate
                # sidecar file ewc_state.pt is written by train() after Fisher
                # computation so subsequent blocks can resume with full EWC
                # regularisation instead of starting cold (λ=0 → spike).
                _ewc_sidecar = os.path.join(cfg.checkpoint_dir, "ewc_state.pt")
                if os.path.exists(_ewc_sidecar):
                    try:
                        _ewc_ckpt = torch.load(_ewc_sidecar, map_location=self._device)
                        self._ewc_state = EWCState(
                            fisher=_ewc_ckpt["ewc_fisher"],
                            anchor={k: v.to(self._device) for k, v in _ewc_ckpt["ewc_anchor"].items()},
                            lambda_=_ewc_ckpt.get("ewc_lambda", cfg.ewc_lambda),
                            last_update_ts=_ewc_ckpt.get("ewc_last_update_ts", 0.0),
                            obs_count_at_update=_ewc_ckpt.get("ewc_obs_count_at_update", 0),
                        )
                        log.info(
                            "EWC sidecar loaded: %d Fisher params, lambda=%.1f "
                            "(EWC regularisation active from epoch 1).",
                            len(self._ewc_state.fisher),
                            self._ewc_state.lambda_,
                        )
                    except Exception as _ewc_exc:
                        log.warning(
                            "EWC sidecar load failed (%s) — starting block without EWC.",
                            _ewc_exc,
                        )
                else:
                    log.info("No EWC sidecar found — first block, EWC will be computed after training.")
            else:
                raise FileNotFoundError(
                    f"[RESUME] Checkpoint not found: {ckpt_path}\n"
                    f"  checkpoint_dir={cfg.checkpoint_dir}\n"
                    f"  resume_from_epoch={cfg.resume_from_epoch}\n"
                    "  Attach the tirramind-h-d-ckpt dataset containing epoch_018.pt"
                )

        train_obs, val_obs, test_obs = self._split_observations()
        windows = self._make_windows(train_obs)
        n_windows_full = len(windows)

        # ── max_windows cap (Phase 41 resource guard) ─────────────────────────
        # Take the LAST max_windows windows (most recent temporal data).
        # This bounds peak RAM to O(max_windows * avg_graph_size) regardless
        # of total DB size. 0 = use all windows (original behaviour).
        if cfg.max_windows > 0 and len(windows) > cfg.max_windows + 1:
            windows = windows[-(cfg.max_windows + 1) :]  # +1 because we need windows[i+1] as next
            _used = len(windows) - 1
            _full = max(n_windows_full - 1, 1)
            pct = 100.0 * _used / _full
            log.info(
                "max_windows=%d: using last %d of %d train windows (%.1f%% of train timeline)",
                cfg.max_windows,
                _used,
                n_windows_full - 1,
                pct,
            )
        else:
            log.info(
                "max_windows=0: using all %d train windows (full train timeline)",
                max(len(windows) - 1, 0),
            )

        def _obs_iso_range(obs: list[dict]) -> tuple[str, str]:
            if not obs:
                return ("—", "—")
            ts = [o.get("observed_at", 0.0) for o in obs]
            lo = datetime.fromtimestamp(min(ts), tz=UTC).date().isoformat()
            hi = datetime.fromtimestamp(max(ts), tz=UTC).date().isoformat()
            return (lo, hi)

        _tr_lo, _tr_hi = _obs_iso_range(train_obs)
        _va_lo, _va_hi = _obs_iso_range(val_obs)
        _te_lo, _te_hi = _obs_iso_range(test_obs)
        log.info(
            "TRAINING_AUDIT: chronological 70/15/15 obs split — " "train [%s → %s]  val [%s → %s]  test [%s → %s]",
            _tr_lo,
            _tr_hi,
            _va_lo,
            _va_hi,
            _te_lo,
            _te_hi,
        )

        # Cache entity type lookups — entities don't change during training
        all_entities = self.store.query_all_entities()
        self._eid_to_type_cache = {e["entity_id"]: e["entity_type"] for e in all_entities}

        if cfg.checkpoint_dir:
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)

        # Pre-fetch static graph structure (entities + links) once
        cached_id_map, _, cached_links = self._graph_builder.prepare_static()

        # Pre-fetch ALL observations once, sorted by time.
        # Per-window slicing via bisect eliminates the DB query per window.
        # NOTE: No since= filter here — graph features need full history
        # (original code used build(since=None, until=t_end)).
        all_prefetched_obs = self._graph_builder.prefetch_observations()

        # ── Modal subsampling (Phase 41 + N1 doctrine) ───────────────────
        from agent.models.gnn.obs_subsample import apply_training_obs_subsample

        _n_before = len(all_prefetched_obs)
        all_prefetched_obs, _sub_stats = apply_training_obs_subsample(
            all_prefetched_obs,
            gdelt_subsample_frac=cfg.gdelt_subsample_frac,
            defi_subsample_frac=cfg.defi_subsample_frac,
        )
        if cfg.zero_price_feats or cfg.embedding_only_return:
            log.info(
                "N1 doctrine flags: zero_price_feats=%s embedding_only_return=%s",
                cfg.zero_price_feats,
                cfg.embedding_only_return,
            )
        if _sub_stats.get("dropped_total", 0) > 0:
            log.info(
                "Obs subsample gdelt_frac=%.3f defi_frac=%.3f: %d → %d "
                "(gdelt kept/drop=%s/%s defi kept/drop=%s/%s floor=%s)",
                cfg.gdelt_subsample_frac,
                cfg.defi_subsample_frac,
                _n_before,
                len(all_prefetched_obs),
                _sub_stats.get("gdelt_kept", 0),
                _sub_stats.get("gdelt_dropped", 0),
                _sub_stats.get("defi_kept", 0),
                _sub_stats.get("defi_dropped", 0),
                _sub_stats.get("floor_kept", 0),
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

        # ── A2: Precompute 21-day forward return lookup (Phase 47) ──────────
        # Replaces daily log_return (near-zero IC) with N-day forward return
        # (the correct oracle for cross-sectional IC at the backtest horizon).
        # Computed once from all prefetched instrument_daily obs; O(1) lookup
        # per obs in the return loss loop below.
        _forward_returns: dict[tuple[str, int], float] = {}
        if cfg.use_forward_returns and cfg.return_weight > 0.0:
            _forward_returns = _build_forward_return_lookup(all_prefetched_obs, horizon_days=cfg.forward_return_horizon)
            log.info(
                "A2: Precomputed %d forward-return labels (horizon=%dd)",
                len(_forward_returns),
                cfg.forward_return_horizon,
            )

        # ── Phase 49: Load GNN alignment weights ──────────────────────────
        # Alignment weights tell the training loop which entity types are
        # not yet well-aligned with the world model (low belief sharpening
        # → high weight → more training emphasis).  Loaded once per
        # train() call — constant across all epochs and windows.
        # Returns None if no alignment signals are stored (uniform weights).
        entity_types_in_graph = list({e.get("entity_type") for e in all_entities if e.get("entity_type")})
        try:
            _alignment_weights: dict[str, float] | None = load_alignment_weights(self.store, entity_types_in_graph)
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
        _snap_build_t0 = time.perf_counter()
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
                use_signatures=cfg.use_signatures,
                ts2vec_embeddings=self._ts2vec_embeddings,
                ts2vec_dim=cfg.ts2vec_dim if cfg.use_ts2vec else 0,
            )
            _window_snapshots.append(_snap_data if _snap_data.node_types else None)
        log.info("Snapshots ready — beginning training epochs.")
        log.info(
            "LOSS_MODE: auto_tune=%s log_loss=%s value_w=%.3f return_w=%.3f "
            "obs_w=%.3f vicreg=%.3f contranorm=%s concat_head=%s",
            cfg.auto_tune_loss_weights,
            cfg.use_log_loss,
            cfg.value_weight,
            cfg.return_weight,
            cfg.obs_type_weight,
            cfg.vicreg_weight,
            getattr(cfg, "use_contranorm", False),
            getattr(cfg, "use_concat_head", False),
        )
        if cfg.auto_tune_loss_weights and cfg.use_log_loss:
            log.warning(
                "use_log_loss + auto_tune both enabled — log transform applied "
                "inside auto_tune branch (V61 bug: log_loss was a no-op)."
            )
        _snap_build_s = time.perf_counter() - _snap_build_t0
        log.info(
            "TRAINING_AUDIT: %d gradient windows/epoch (snapshot build %.0fs, once). "
            "Epoch 1 ≈ snapshot_build + %d×forward; epochs 2+ ≈ %d×forward only "
            "(~1 min/epoch on GPU is expected after the first epoch — not a shortcut).",
            _total_windows,
            _snap_build_s,
            _total_windows,
            _total_windows,
        )

        # ── Active return head diagnostic ─────────────────────────────────
        # Reflect the *runtime* return-loss path (cfg), not merely which heads
        # exist on the model (return_raw_head is always built when raw_dim>0).
        if cfg.embedding_only_return:
            log.info(
                "[HEAD] return_pred_head ACTIVE (embedding_only_return) — "
                "GNN embeddings only → return. N1 doctrine path. ✓"
            )
        elif getattr(model, "return_concat_head", None) is not None:
            log.info(
                "[HEAD] return_concat_head ACTIVE — GNN embeddings + raw features → return. "
                "Ghost patterns will contribute to IC. ✓"
            )
        elif getattr(model, "return_raw_head", None) is not None:
            log.warning(
                "[HEAD] return_raw_head ACTIVE — raw features ONLY → return. "
                "GNN embeddings are BYPASSED. IC = raw features baseline only. "
                "Add --use-concat-head to enable ghost pattern contribution."
            )
        else:
            log.info("[HEAD] return_pred_head ACTIVE — GNN embeddings only → return.")

        for epoch in range(start_epoch, cfg.epochs):
            _epoch_t0 = time.perf_counter()
            model.reset_memory()
            epoch_losses = {
                "total": 0.0,
                "obs_type": 0.0,
                "time_delta": 0.0,
                "contrastive": 0.0,
                "value": 0.0,
                "return": 0.0,
            }
            _epoch_ntype_obs: dict[str, list[float]] = {}
            _epoch_grad_diag: dict = {
                "preds": [],
                "tgts": [],
                "raw_preds": [],
                "n_clamped_upper": 0,
                "n_clamped_lower": 0,
                "n_pred_samples": 0,
                "active_head": None,
            }
            _epoch_emb_diag: dict = {}
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

                vicreg_loss = torch.tensor(0.0, device=self._device)
                if cfg.vicreg_weight > 0.0 and "instrument" in embeddings:
                    _inst_emb = embeddings["instrument"]
                    if _inst_emb.size(0) >= 2:
                        vicreg_loss = _vicreg_loss(_inst_emb, gamma=cfg.vicreg_var_gamma)

                # Supervision targets from next window
                global_ids, obs_targets, dt_targets, val_targets = self._compute_targets(
                    curr_obs,
                    next_obs,
                    id_map,
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
                                [_alignment_weights.get(nt, 1.0) for nt in _valid_ntypes],
                                dtype=torch.float32,
                                device=logits.device,
                            )
                            raw_ce = F.cross_entropy(logits, valid_targets, reduction="none").clamp(max=20.0)
                            obs_loss = (raw_ce * per_example_w).mean()
                        else:
                            raw_ce = F.cross_entropy(logits, valid_targets, reduction="none").clamp(max=20.0)
                            obs_loss = raw_ce.mean()
                        # Per-node-type CE accumulation (for metrics.jsonl)
                        for _nt, _ce_v in zip(_valid_ntypes, raw_ce.detach().tolist()):
                            _epoch_ntype_obs.setdefault(_nt, []).append(_ce_v)

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
                    dt_pred = model.time_delta_head(target_emb_tensor).squeeze(-1).clamp(-20.0, 20.0)
                    valid_dt_raw = dt_targets[valid_indices]
                    # Mask NaN/Inf targets (same-timestamp obs → log1p(0)=0 is fine
                    # but negative dt from bad DB rows → log1p(dt<-1) = NaN).
                    # Clamp AFTER masking so clamp(NaN) can't propagate.
                    _dt_finite = torch.isfinite(valid_dt_raw)
                    if _dt_finite.any():
                        valid_dt = valid_dt_raw[_dt_finite].clamp(0.0, 20.0)
                        dt_loss = F.huber_loss(dt_pred[_dt_finite], valid_dt, delta=1.0)
                        if not torch.isfinite(dt_loss):
                            dt_loss = torch.tensor(0.0, device=self._device)

                # ── value prediction loss ────────────────
                # Fix: clamp predictions to [-1e4, 1e4].  Raw financial values
                # (usd_amount, whale txn amounts) can be $1B+.  Without a clamp
                # the value head produces gradient spikes that propagate backwards
                # and corrupt all other heads (observed: val_loss = 1,094,629
                # at epoch 24 in H-G run).
                val_loss = torch.tensor(0.0, device=self._device)
                if target_embs:
                    val_pred = model.value_pred_head(target_emb_tensor).squeeze(-1).clamp(-1e4, 1e4)
                    valid_val = val_targets[valid_indices]
                    val_loss = F.huber_loss(val_pred, valid_val)

                # ── contrastive loss (CSRC: return-decile-based) ──
                # Old entity-link contrastive is disabled (returns 0.0).
                # CSRC is computed inside the return section below where
                # instrument embeddings and return targets are available.
                c_loss = self._contrastive_loss(embeddings, id_map)

                # ── return auxiliary loss (Phase 47 — A2 fix) ────────────────────────
                # Supervise instrument embeddings on N-day FORWARD return.
                # Using daily log_return (the old target) gives near-zero IC
                # because daily returns have AR(1) ≈ 0 at cross-section.
                # 21-day forward returns have measurable persistence and align
                # with the monthly rebalancing cadence in the backtest.
                # Falls back to daily log_return when no forward label exists.
                ret_loss = torch.tensor(0.0, device=self._device)
                if cfg.return_weight > 0.0 and "instrument" in embeddings:
                    _ret_embs: list[torch.Tensor] = []
                    _ret_raw_feats: list[torch.Tensor] = []
                    _has_raw = (
                        model.return_raw_head is not None
                        and "instrument" in data.node_types
                        and hasattr(data["instrument"], "x")
                    )
                    _ret_targets: list[float] = []
                    for _o in next_obs:
                        if _o.get("observation_type") != "instrument_daily":
                            continue
                        _eid = _o.get("entity_id")
                        if _eid is None:
                            continue
                        # ── A2: prefer forward return over daily log_return ──
                        _ts_key = int(_o.get("observed_at", 0.0))
                        _lr: float | None = None
                        if cfg.use_forward_returns:
                            _lr = _forward_returns.get((_eid, _ts_key))
                        if _lr is None:
                            # Fallback: daily log_return (backward compat)
                            _v = _o.get("value", {})
                            if not isinstance(_v, dict) or "log_return" not in _v:
                                continue
                            try:
                                _lr = float(_v["log_return"])
                            except (TypeError, ValueError):
                                continue
                        _local_idx = id_map.local_id("instrument", _eid)
                        if _local_idx is None:
                            continue
                        _inst_emb = embeddings["instrument"]
                        if _local_idx >= _inst_emb.size(0):
                            continue
                        _ret_embs.append(_inst_emb[_local_idx])
                        if _has_raw:
                            _ret_raw_feats.append(data["instrument"].x[_local_idx])
                        _ret_targets.append(_lr)
                    if _ret_embs:
                        _ret_emb_t = torch.stack(_ret_embs)
                        _ret_tgt_t = torch.tensor(_ret_targets, dtype=torch.float32, device=self._device)
                        # ── CSRC: cross-sectional ranking contrastive ──
                        # Computed BEFORE residual demeaning so deciles
                        # reflect raw return magnitude, not relative rank.
                        if cfg.use_csrc_loss:
                            _csrc = self._cross_sectional_ranking_contrastive(
                                _ret_emb_t,
                                _ret_tgt_t,
                                temperature=cfg.csrc_temperature,
                                n_deciles=cfg.csrc_n_deciles,
                            )
                            c_loss = c_loss + _csrc
                        # ── Phase 50: residual returns ──────────────────
                        # Cross-sectionally demean targets so the model
                        # learns only relative outperformance, not the
                        # market-wide component.  This directly aligns the
                        # training objective with Spearman IC evaluation.
                        if cfg.use_residual_returns:
                            _ret_tgt_t = _ret_tgt_t - _ret_tgt_t.mean()
                        # Guard: filter out any NaN/Inf targets that came from
                        # bad DB rows (stock splits, missing prices, etc.).
                        # A single NaN target propagates through huber_loss →
                        # total loss → backward → all weights become NaN silently.
                        # ListNet additionally requires >= 2 items to rank;
                        # a single-item softmax is trivially 1.0 → loss = 0.
                        _finite_mask = torch.isfinite(_ret_tgt_t)
                        _n_valid = int(_finite_mask.sum().item())
                        _min_required = 2 if cfg.use_listnet_return_loss else 1
                        if _n_valid < _min_required:
                            # Debug: log why return loss was skipped
                            _n_total = len(_ret_tgt_t)
                            _n_nan = _n_total - _n_valid
                            if _n_nan > 0:
                                log.warning(
                                    "Return loss skipped: %d/%d targets NaN/Inf "
                                    "(min_required=%d).  First 3 targets: %s",
                                    _n_nan,
                                    _n_total,
                                    _min_required,
                                    _ret_tgt_t[:3].tolist(),
                                )
                        if _n_valid >= _min_required:
                            if cfg.embedding_only_return:
                                _ret_pred = model.return_pred_head(_ret_emb_t).squeeze(-1)
                            elif _has_raw and _ret_raw_feats and getattr(model, "return_concat_head", None) is not None:
                                # CONCAT mode (Option B): concatenate xsnorm raw
                                # features with GNN instrument embeddings so that
                                # backbone gradients flow into the return predictor.
                                _raw_feat_t = xsnorm_price_feats(torch.stack(_ret_raw_feats))
                                _concat_in = torch.cat([_raw_feat_t, _ret_emb_t], dim=-1)
                                if cfg.use_concat_batchnorm:
                                    _concat_in = F.layer_norm(_concat_in, _concat_in.shape[-1:])
                                _ret_pred = model.return_concat_head(_concat_in).squeeze(-1)
                            elif _has_raw and _ret_raw_feats:
                                # REPLACE mode: raw features are the sole return
                                # predictor.  The backbone path is excluded because
                                # obs_type training spikes corrupt instrument
                                # embeddings every few epochs, adding oscillating
                                # noise that decays the raw head's learned signal.
                                _raw_feat_t = xsnorm_price_feats(torch.stack(_ret_raw_feats))
                                _ret_pred = model.return_raw_head(_raw_feat_t).squeeze(-1)
                            else:
                                _ret_pred = model.return_pred_head(_ret_emb_t).squeeze(-1)
                            _ret_pred_raw = _ret_pred
                            if cfg.return_pred_clamp > 0:
                                _ret_pred = _ret_pred.clamp(-cfg.return_pred_clamp, cfg.return_pred_clamp)
                            if cfg.embedding_only_return:
                                _active_head = "pred"
                            elif _has_raw and _ret_raw_feats and getattr(model, "return_concat_head", None) is not None:
                                _active_head = "concat"
                            elif _has_raw and _ret_raw_feats:
                                _active_head = "raw"
                            else:
                                _active_head = "pred"
                            if _epoch_grad_diag["active_head"] is None:
                                _epoch_grad_diag["active_head"] = _active_head
                            _pred_finite = _ret_pred[_finite_mask].detach()
                            _tgt_finite = _ret_tgt_t[_finite_mask].detach()
                            _raw_finite = _ret_pred_raw[_finite_mask].detach()
                            _epoch_grad_diag["preds"].extend(_pred_finite.cpu().tolist())
                            _epoch_grad_diag["tgts"].extend(_tgt_finite.cpu().tolist())
                            _epoch_grad_diag["raw_preds"].extend(_raw_finite.cpu().tolist())
                            _n_batch = int(_raw_finite.numel())
                            _epoch_grad_diag["n_pred_samples"] += _n_batch
                            if cfg.return_pred_clamp > 0 and _n_batch > 0:
                                _epoch_grad_diag["n_clamped_upper"] += int(
                                    (_raw_finite > cfg.return_pred_clamp).sum().item()
                                )
                                _epoch_grad_diag["n_clamped_lower"] += int(
                                    (_raw_finite < -cfg.return_pred_clamp).sum().item()
                                )
                            if cfg.use_listnet_return_loss:
                                ret_loss = _listnet_loss(
                                    _ret_pred[_finite_mask],
                                    _ret_tgt_t[_finite_mask],
                                    tau=cfg.listnet_temperature,
                                )
                            else:
                                ret_loss = F.huber_loss(_ret_pred[_finite_mask], _ret_tgt_t[_finite_mask])
                            # Direction BCE loss (Phase H-D/H-F hypothesis).
                            # Treat predicted scalar as logit; penalises sign
                            # errors independently of magnitude.  Provides a
                            # complementary gradient to ListNet's rank ordering.
                            if cfg.use_direction_loss:
                                _dir_tgt = (_ret_tgt_t[_finite_mask] > 0).float()
                                _dir_loss = F.binary_cross_entropy_with_logits(_ret_pred[_finite_mask], _dir_tgt)
                                ret_loss = ret_loss + cfg.direction_loss_weight * _dir_loss
                        # Fix: ListNet loss is KL(p_target || p_pred).
                        # In theory non-negative, but floating-point rounding
                        # in softmax can produce tiny negatives.  A negative
                        # loss rewards the model for being wrong — clamp it out.
                        # Also apply return upscaling: the return head sees only
                        # ~53 instruments while obs_type sees ~2,145 entities.
                        # Without upscaling the return gradient is ~40× weaker
                        # and auto-tune silences it within 10 epochs.
                        ret_loss = ret_loss.clamp(min=0.0) * _return_upscale

                # M1: CDE memory update — runs before loss combination so the
                # KL term (Phase E only) is included in this window's backward pass.
                # For Phase B/C/D the KL is 0 and this is a pure memory update.
                _cwm_kl = torch.tensor(0.0, device=self._device)
                if self._cwm is not None:
                    _cwm_out = self._cwm.update_memories(
                        curr_obs,
                        model.memory,
                        id_map,
                        embeddings,
                        training=True,
                    )
                    _cwm_kl = _cwm_out["kl_loss"]

                # ── total loss ───────────────────────────
                _task_losses: dict[str, torch.Tensor] = {}
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
                            min=(cfg.contrastive_log_var_min if k == "contrastive" else lv_min),
                            max=(cfg.return_log_var_max if k == "return" else lv_max),
                        )
                        for k, p in lv.items()
                    }
                    _obs_l = _scaled_task_loss(obs_loss, cfg)
                    _dt_l = _scaled_task_loss(dt_loss, cfg)
                    _c_l = _scaled_task_loss(c_loss, cfg)
                    _val_l = _scaled_task_loss(val_loss, cfg)
                    _ret_l = _scaled_task_loss(ret_loss, cfg)
                    _task_losses = {
                        "obs_type": torch.exp(-clamped["obs_type"]) * _obs_l + clamped["obs_type"],
                        "time_delta": torch.exp(-clamped["time_delta"]) * _dt_l + clamped["time_delta"],
                        "contrastive": torch.exp(-clamped["contrastive"]) * _c_l + clamped["contrastive"],
                        "value": torch.exp(-clamped["value"]) * _val_l + clamped["value"],
                        "return": torch.exp(-clamped["return"]) * _ret_l + clamped["return"],
                    }
                    total = sum(_task_losses.values())
                    if cfg.vicreg_weight > 0.0:
                        _task_losses["vicreg"] = cfg.vicreg_weight * vicreg_loss
                        total = total + _task_losses["vicreg"]
                else:
                    _task_losses = {
                        "obs_type": cfg.obs_type_weight * _scaled_task_loss(obs_loss, cfg),
                        "time_delta": cfg.time_delta_weight * _scaled_task_loss(dt_loss, cfg),
                        "contrastive": cfg.contrastive_weight * _scaled_task_loss(c_loss, cfg),
                        "value": cfg.value_weight * _scaled_task_loss(val_loss, cfg),
                        "return": cfg.return_weight * _scaled_task_loss(ret_loss, cfg),
                    }
                    total = sum(_task_losses.values())
                    if cfg.vicreg_weight > 0.0:
                        _task_losses["vicreg"] = cfg.vicreg_weight * vicreg_loss
                        total = total + _task_losses["vicreg"]

                # M1 Phase E: anneal KL weight 0 → cwm_lambda_kl over warmup epochs
                if self._cwm is not None and cfg.cwm_curriculum_phase.upper() == "E":
                    _kl_w = cfg.cwm_lambda_kl * min(1.0, (epoch + 1) / max(cfg.cwm_kl_warmup_epochs, 1))
                    total = total + _kl_w * _cwm_kl

                if total.requires_grad:
                    _pcgrad_params = [p for p in model.parameters() if p.requires_grad]
                    if self._log_vars is not None:
                        _pcgrad_params.extend(p for p in self._log_vars.values() if p.requires_grad)
                    _used_pcgrad = False
                    if cfg.use_pcgrad:
                        try:
                            _used_pcgrad = _pcgrad_optimizer_step(optimizer, _pcgrad_params, _task_losses)
                        except RuntimeError as _pcgrad_err:
                            log.warning(
                                "PCGrad failed (%s) — falling back to standard backward",
                                _pcgrad_err,
                            )
                    if not _used_pcgrad:
                        optimizer.zero_grad(set_to_none=True)
                        total.backward()
                    # Clip backbone and return_raw_head separately so that
                    # obs_type explosion spikes never starve the raw head.
                    _backbone_p = [
                        p for n, p in model.named_parameters() if "return_raw_head" not in n and p.requires_grad
                    ]
                    _raw_p = [p for n, p in model.named_parameters() if "return_raw_head" in n and p.requires_grad]
                    if _backbone_p:
                        torch.nn.utils.clip_grad_norm_(_backbone_p, 1.0)
                    if _raw_p:
                        torch.nn.utils.clip_grad_norm_(_raw_p, 1.0)
                    optimizer.step()

                # Update memory from current window events
                # Skip when CWM is active — memories already updated above
                if self._cwm is None:
                    with torch.no_grad():
                        # Recompute embeddings after optimizer step
                        embeddings_detached = model(data, id_map)
                    model.update_memory_from_events(
                        curr_obs,
                        embeddings_detached,
                        id_map,
                        t_start=t_start,
                        t_end=t_end,
                    )

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

            _epoch_s = time.perf_counter() - _epoch_t0
            _wps = n_windows / max(_epoch_s, 1e-6)
            log.info(
                "Epoch %d/%d — loss: %.4f (obs_type: %.4f, dt: %.4f, contrastive: %.4f, return: %.4f) "
                "| %d windows in %.1fs (%.2f win/s)",
                epoch + 1,
                cfg.epochs,
                history["total"][-1],
                history["obs_type"][-1],
                history["time_delta"][-1],
                history["contrastive"][-1],
                history["return"][-1],
                n_windows,
                _epoch_s,
                _wps,
            )
            if epoch == start_epoch and _epoch_s < 120 and _total_windows >= 100:
                log.warning(
                    "TRAINING_AUDIT: epoch 1 completed in %.0fs for %d windows — "
                    "verify max_windows and snapshot build ran (expected: long epoch 1, "
                    "then faster epochs 2+).",
                    _epoch_s,
                    _total_windows,
                )

            # ── Embedding diversity check (LESSONS.md F-01 prevention) ──
            # Compute std of instrument embeddings on the last window snapshot.
            # std < 0.05 → embeddings are collapsing → CSRC is not working.
            # Runs in no_grad on the last snapshot only (cheap). Every epoch
            # when grad-flow diagnostics are enabled (V64+ loop).
            if _window_snapshots and "instrument" in (
                _window_snapshots[-1].node_types if _window_snapshots[-1] is not None else []
            ):
                try:
                    with torch.no_grad():
                        _last_snap = _window_snapshots[-1].to(self._device)
                        _diag_embs = model(_last_snap, cached_id_map)
                        if "instrument" in _diag_embs:
                            _ie = _diag_embs["instrument"]
                            _emb_std = _ie.std(dim=0).mean().item()
                            _emb_rank = min(_ie.shape[0], _ie.shape[1])
                            # Effective rank: exp(entropy of normalised singular values)
                            _sv = torch.linalg.svdvals(_ie.float())
                            _sv_norm = _sv / (_sv.sum() + 1e-8)
                            _eff_rank = torch.exp(-(_sv_norm * (_sv_norm + 1e-8).log()).sum()).item()
                            _epoch_emb_diag = {
                                "emb_std": _emb_std,
                                "eff_rank": _eff_rank,
                                "emb_dim": _emb_rank,
                                "collapse_risk": _emb_std < 0.05,
                            }
                            if _emb_std < 0.05:
                                log.warning(
                                    "[COLLAPSE] Instrument embedding std=%.4f < 0.05 "
                                    "— embeddings are collapsing! "
                                    "CSRC loss not differentiating instruments. "
                                    "effective_rank=%.1f/%d",
                                    _emb_std,
                                    _eff_rank,
                                    _emb_rank,
                                )
                            elif (epoch + 1) % 5 == 0:
                                log.info(
                                    "[EMB] Instrument embedding std=%.4f  effective_rank=%.1f/%d",
                                    _emb_std,
                                    _eff_rank,
                                    _emb_rank,
                                )
                except Exception as _diag_e:
                    log.debug("Embedding diversity check skipped: %s", _diag_e)

            # ── Grad-flow diagnostic (V64+ autonomous loop) ───────────────
            _grad_flow: dict = {}
            if _epoch_grad_diag["preds"]:
                import statistics as _stats  # noqa: PLC0415

                _preds = _epoch_grad_diag["preds"]
                _tgts = _epoch_grad_diag["tgts"]
                _pred_std = _stats.pstdev(_preds) if len(_preds) > 1 else 0.0
                _tgt_std = _stats.pstdev(_tgts) if len(_tgts) > 1 else 0.0
                _ret_hist = history.get("return", [])
                _plateau_eps = 0.01
                _listnet_floor = len(_ret_hist) >= 1 and _ret_hist[-1] > 200.0
                _return_flat = len(_ret_hist) >= 3 and max(_ret_hist[-3:]) - min(_ret_hist[-3:]) < _plateau_eps
                _n_samples = max(_epoch_grad_diag.get("n_pred_samples", 0), 1)
                _pct_clamped_upper = 100.0 * _epoch_grad_diag.get("n_clamped_upper", 0) / _n_samples
                _pct_clamped_lower = 100.0 * _epoch_grad_diag.get("n_clamped_lower", 0) / _n_samples
                _raw_preds = _epoch_grad_diag.get("raw_preds") or _preds
                _raw_std = _stats.pstdev(_raw_preds) if len(_raw_preds) > 1 else 0.0
                _in_sample_ic = None
                if len(_preds) >= 3 and len(_tgts) >= 3:
                    try:
                        from scipy import stats as _scipy_stats  # noqa: PLC0415

                        _ic_res = _scipy_stats.spearmanr(_preds, _tgts)
                        _in_sample_ic = float(_ic_res.statistic)
                    except Exception as _ic_err:
                        log.debug("In-sample IC skipped: %s", _ic_err)
                _grad_flow = {
                    "active_head": _epoch_grad_diag.get("active_head"),
                    "return_pred_std": _pred_std,
                    "return_raw_pred_std": _raw_std,
                    "return_tgt_std": _tgt_std,
                    "return_pred_mean": sum(_preds) / len(_preds),
                    "n_return_samples": len(_preds),
                    "pct_clamped_upper": _pct_clamped_upper,
                    "pct_clamped_lower": _pct_clamped_lower,
                    "in_sample_ic": _in_sample_ic,
                    "listnet_floor": _listnet_floor,
                    "return_loss_flat_3ep": _return_flat,
                    **_epoch_emb_diag,
                }
                log.info(
                    "[GRAD_FLOW] ep%d head=%s pred_std=%.4f raw_std=%.4f tgt_std=%.4f "
                    "pred_mean=%.4f clamp_u=%.1f%% clamp_l=%.1f%% in_sample_ic=%s "
                    "emb_std=%s eff_rank=%s listnet_floor=%s",
                    epoch + 1,
                    _grad_flow.get("active_head", "?"),
                    _pred_std,
                    _raw_std,
                    _tgt_std,
                    _grad_flow["return_pred_mean"],
                    _pct_clamped_upper,
                    _pct_clamped_lower,
                    (f"{_in_sample_ic:.4f}" if _in_sample_ic is not None else "n/a"),
                    (f"{_grad_flow['emb_std']:.4f}" if "emb_std" in _grad_flow else "n/a"),
                    (f"{_grad_flow['eff_rank']:.1f}" if "eff_rank" in _grad_flow else "n/a"),
                    _listnet_floor,
                )
                if _in_sample_ic is not None:
                    log.info(
                        "[IN_SAMPLE_IC] ep%d spearman=%.4f n=%d",
                        epoch + 1,
                        _in_sample_ic,
                        len(_preds),
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
                        "loss/return": (history["return"][-1] if history["return"] else float("nan")),
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
                ckpt_path = os.path.join(cfg.checkpoint_dir, f"epoch_{epoch + 1:03d}.pt")
                ckpt_payload: dict = {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "history": history,
                }
                if self._log_vars is not None:
                    ckpt_payload["log_vars"] = {k: v.data.cpu() for k, v in self._log_vars.items()}
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
                # Collect any trainer warnings for this epoch so auto_improve
                # and the advisor LLM can read them without parsing log text.
                _epoch_warnings: list[str] = []
                if len(history["time_delta"]) >= 3 and len(history["return"]) >= 3:
                    _w_dt = sum(history["time_delta"][-3:]) / 3
                    _w_ret = sum(history["return"][-3:]) / 3
                    _w_ratio = _w_dt / (_w_ret + 1e-8)
                    if _w_ratio > 50:
                        _epoch_warnings.append(
                            f"return_head_starved: dt/ret ratio={_w_ratio:.0f}x "
                            "— return head receiving <2% of gradient budget"
                        )
                _obs_losses = history.get("obs_type", [])
                if _obs_losses and len(_obs_losses) >= 2:
                    _obs_last = _obs_losses[-1]
                    _obs_prev = _obs_losses[-2]
                    if _obs_last > _obs_prev * 5 and _obs_last > 10:
                        _epoch_warnings.append(
                            f"obs_type_spike: loss jumped {_obs_prev:.2f} → {_obs_last:.2f} "
                            "— likely rare obs_type batch; may steal gradient budget"
                        )
                _epoch_record = {
                    "epoch": epoch + 1,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "loss": {
                        "total": (history["total"][-1] if history.get("total") else float("nan")),
                        "return": _ret_loss,
                        "dt": _dt_loss,
                        "obs_type": (history["obs_type"][-1] if history.get("obs_type") else float("nan")),
                        "contrastive": (history["contrastive"][-1] if history.get("contrastive") else float("nan")),
                        "value": (history["value"][-1] if history.get("value") else float("nan")),
                    },
                    "dt_ret_ratio": _dt_ret_ratio,
                    "grad_flow": _grad_flow,
                    "warnings": _epoch_warnings,
                    "entity_type_losses": {nt: sum(vs) / len(vs) for nt, vs in _epoch_ntype_obs.items() if vs},
                    "entity_type_counts": {nt: len(vs) for nt, vs in _epoch_ntype_obs.items()},
                    "config": {
                        "lr": cfg.learning_rate,
                        "return_weight": float(cfg.return_weight),
                        "obs_type_weight": float(getattr(cfg, "obs_type_weight", 1.0)),
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
                use_signatures=cfg.use_signatures,
                ts2vec_embeddings=self._ts2vec_embeddings,
                ts2vec_dim=cfg.ts2vec_dim if cfg.use_ts2vec else 0,
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
                    anchor={n: p.data.clone().cpu() for n, p in model.named_parameters()},
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
                # ── Persist EWC state as sidecar for next block ───────────
                # The per-epoch .pt checkpoints do NOT include EWC state.
                # Writing a separate sidecar here means the next block's
                # train(resume=N) call will load it and start with full
                # EWC regularisation, preventing the loss spike on restart.
                if cfg.checkpoint_dir:
                    _ewc_sidecar = os.path.join(cfg.checkpoint_dir, "ewc_state.pt")
                    _ewc_payload = {
                        "ewc_fisher": {k: v.cpu() for k, v in self._ewc_state.fisher.items()},
                        "ewc_anchor": {k: v.cpu() for k, v in self._ewc_state.anchor.items()},
                        "ewc_lambda": self._ewc_state.lambda_,
                        "ewc_last_update_ts": self._ewc_state.last_update_ts,
                        "ewc_obs_count_at_update": self._ewc_state.obs_count_at_update,
                    }
                    torch.save(_ewc_payload, _ewc_sidecar)
                    log.info(
                        "EWC sidecar saved → %s (%d Fisher params).",
                        _ewc_sidecar,
                        len(self._ewc_state.fisher),
                    )
            else:
                log.warning("Fisher computation skipped — last training window produced an empty graph (no nodes).")
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
            out: dict[str, float] = {}
            for k, p in self._log_vars.items():
                mn = cfg.contrastive_log_var_min if k == "contrastive" else cfg.log_var_min
                mx = cfg.return_log_var_max if k == "return" else cfg.log_var_max
                clamped = max(mn, min(mx, p.item()))
                out[k] = math.exp(-clamped)
            return out
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
                from agent.preflight import FeaturePreflight  # noqa: PLC0415

                _, pf = FeaturePreflight.for_gnn_inference(model=model, store=self.store, min_entity_types=1)
                if not pf.ok:
                    log.warning("infer: empty graph — %s", pf.user_message)
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

    # ── Idea 11: Portfolio construction ───────────────────────────────────

    def compute_portfolio(
        self,
        return_preds: dict[str, float] | None = None,
        quality_scores: dict[str, float] | None = None,
        prev_weights: dict[str, float] | None = None,
        as_of: float | None = None,
    ):
        """Run GNN inference + BL/HRP portfolio construction.

        If ``return_preds`` is not provided, runs a forward pass on the
        current graph and uses ``model.predict_return()`` on instrument
        embeddings as the view vector.

        Args:
            return_preds: Optional override dict[entity_id, log_return].
                If None, derived from model.predict_return() on instruments.
            quality_scores: Optional dict[entity_id, confidence ∈ (0,1]].
            prev_weights: Optional previous portfolio weights for smoothing.
            as_of: Reference time for the graph snapshot.

        Returns:
            PortfolioWeights or None if insufficient data.
        """
        from agent.portfolio.constructor import PortfolioConstructor  # noqa: PLC0415
        from agent.preflight import FeaturePreflight  # noqa: PLC0415

        cfg = self.config

        # If no explicit predictions, run inference
        if return_preds is None:
            embeddings, id_map = self.infer(until=as_of)
            return_preds = {}

            instr_embs = embeddings.get("instrument")
            if instr_embs is not None:
                with torch.no_grad():
                    preds = self.model.predict_return(instr_embs)  # (N, 1)
                instr_local = id_map.type_local.get("instrument", {})
                for eid, local_idx in instr_local.items():
                    return_preds[eid] = float(preds[local_idx, 0])

        ok, pf = FeaturePreflight.for_portfolio(
            store=self.store,
            return_preds=return_preds,
            min_assets=2,
        )
        if not ok:
            log.warning("compute_portfolio preflight: %s", pf.user_message)
            return None

        pc = PortfolioConstructor(
            delta=cfg.portfolio_delta,
            tilt_factor=cfg.portfolio_tilt_factor,
            turnover_lambda=cfg.portfolio_turnover_lambda,
            lookback_days=cfg.portfolio_lookback_days,
            min_history=cfg.portfolio_min_history,
        )
        pw = pc.build_weights(
            store=self.store,
            return_preds=return_preds,
            quality_scores=quality_scores,
            prev_weights=prev_weights,
            as_of=as_of,
        )
        if pw is not None:
            log.info("compute_portfolio: built weights for %d assets.", pw.n_assets)
        return pw

    # ── Idea 12: Barra-style attribution ──────────────────────────────────

    def compute_attribution(
        self,
        target_entity_ids: list[str] | None = None,
        as_of: float | None = None,
    ) -> dict[str, Any]:
        """Decompose GNN predictions into per-source-type contributions.

        Runs one no_grad forward pass with HGT attention capture enabled,
        then aggregates attention by source node type for every instrument
        node (or the subset given by ``target_entity_ids``).

        CPU safety: capped at ``config.attribution_max_entities`` (default
        200).  No gradient computation.  Disabled by default
        (``use_attribution=False``).

        Args:
            target_entity_ids: Restrict attribution to these IDs.
                If None, all instrument nodes (up to cap) are attributed.
            as_of: Optional cutoff time for the graph snapshot.

        Returns:
            dict[entity_id, AttributionResult] — empty dict if the model
            is not built or the graph has no instrument nodes.
        """
        from agent.models.gnn.attribution import BarraAttribution  # noqa: PLC0415
        from agent.preflight import FeaturePreflight  # noqa: PLC0415

        ok, pf = FeaturePreflight.for_attribution(model=self._model)
        if not ok:
            log.warning("compute_attribution preflight: %s", pf.user_message)
            return {}

        cfg = self.config
        ba = BarraAttribution(
            target_type="instrument",
            max_entities=cfg.attribution_max_entities,
            min_attention=cfg.attribution_min_attention,
        )

        with torch.no_grad():
            data, id_map, _ = self._graph_builder.build(until=as_of)

        if not data.node_types:
            log.info("compute_attribution: empty graph — skipping.")
            return {}

        return ba.compute(
            model=self.model,
            data=data,
            id_map=id_map,
            target_entity_ids=target_entity_ids,
        )

    # ── Idea 13: Data Governance Catalog ──────────────────────────────────

    def check_data_freshness(
        self,
        store_signals: bool = False,
    ) -> Any:
        """Run a freshness SLA check across all registered data sources.

        Queries the pipeline store for the most recent observation timestamp
        per tool and compares against each tool's SLA window.

        CPU Safety: only SQLite MAX queries — no torch, no matrix ops.
        Disabled by default (``use_data_catalog=False``).

        Args:
            store_signals: If True, persist freshness and breach signals
                to the pipeline store.

        Returns:
            CatalogReport with per-tool FreshnessStatus entries, or None
            if ``use_data_catalog`` is False.
        """
        from agent.data_catalog.catalog import DataCatalog  # noqa: PLC0415
        from agent.preflight import FeaturePreflight  # noqa: PLC0415

        cfg = self.config
        ok, pf = FeaturePreflight.for_data_catalog(
            store=self.store,
            use_data_catalog=cfg.use_data_catalog,
        )
        if not ok:
            log.info("check_data_freshness preflight: %s", pf.user_message)
            return None

        catalog = DataCatalog(max_tools=cfg.catalog_max_tools)

        # Apply SLA multiplier if not default
        if abs(cfg.catalog_sla_multiplier - 1.0) > 1e-9:
            from agent.data_catalog.catalog import ToolMeta  # noqa: PLC0415

            scaled = [
                ToolMeta(
                    name=tm.name,
                    category=tm.category,
                    frequency_hours=tm.frequency_hours,
                    sla_hours=tm.sla_hours * cfg.catalog_sla_multiplier,
                    signals=tm.signals,
                    description=tm.description,
                )
                for tm in catalog._registry.values()
            ]
            catalog._registry.clear()
            for tm in scaled:
                catalog._registry[tm.name] = tm

        report = catalog.check_freshness(self.store)

        if report.n_breached > 0:
            log.warning(
                "check_data_freshness: %d/%d tools breaching SLA: %s",
                report.n_breached,
                len(report.statuses),
                report.breached_tools[:10],
            )

        if store_signals:
            n = catalog.store_freshness_signals(self.store, report)
            log.info("check_data_freshness: stored %d freshness signals.", n)

        return report

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
        global_ids, obs_targets, dt_targets, val_targets = self._compute_targets(curr_obs, next_obs, id_map)

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
            dt_pred = model.time_delta_head(target_emb_tensor).squeeze(-1).clamp(-20.0, 20.0)
            valid_dt_raw = dt_targets[valid_indices].to(dt_pred.device)
            _dt_finite = torch.isfinite(valid_dt_raw)
            if _dt_finite.any():
                valid_dt = valid_dt_raw[_dt_finite].clamp(0.0, 20.0)
                dt_loss = F.huber_loss(dt_pred[_dt_finite], valid_dt, delta=1.0)
                if not torch.isfinite(dt_loss):
                    dt_loss = torch.tensor(0.0, device=self._device)

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
        loss_new: torch.Tensor = self._loss_from_window(data, id_map, curr_obs, next_obs)

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
            raise RuntimeError("No model to save — call build_model() or train() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build graph once to capture metadata (node/edge types, node count)
        # for reconstruction. This bare call is fine for *those* fields —
        # node/edge types don't depend on use_signatures/ts2vec — but must
        # NOT be used for in_channels (see below).
        data, id_map, _ = self._graph_builder.build()
        metadata = data.metadata()

        # in_channels MUST come from the actual model instance being saved,
        # not from this bare build() call. A bare call omits
        # use_signatures=True / ts2vec_embeddings, which the real
        # build_model() applies when configured (agent/models/gnn/graph_
        # builder.py adds SIGNATURE_DIM/ts2vec_dim extra dims per node type).
        # Deriving in_channels from `data` here would then record a NARROWER
        # width than the model's real type_projections — precisely the
        # self-contradicting checkpoint this module was built to catch
        # (see checkpoint_store.py: in_channels['instrument']=49 recorded
        # against trained weights of width 23). Reading straight off
        # self._model.type_projections instead ties the recorded width to
        # the weights actually being serialised, by construction.
        in_channels: dict[str, int] = {
            ntype: self._model.type_projections[ntype].in_features for ntype in self._model.node_types
        }
        for ntype in metadata[0]:
            if ntype not in in_channels:
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
                # Phase 50 — head architecture flags (must match build_model())
                "use_concat_head": getattr(self.config, "use_concat_head", False),
                "use_contranorm": getattr(self.config, "use_contranorm", False),
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

        checkpoint = torch.load(
            path,
            map_location=_checkpoint_map_location(),
            weights_only=False,
        )

        config = TrainerConfig(**checkpoint["config"])
        trainer = cls(store, config)
        trainer._device = _resolve_torch_device(trainer.config.device)
        if str(trainer._device) != trainer.config.device:
            trainer.config.device = str(trainer._device)

        metadata = (
            checkpoint["metadata_node_types"],
            [tuple(t) for t in checkpoint["metadata_edge_types"]],
        )
        in_channels = checkpoint["in_channels"]
        num_nodes = checkpoint["num_nodes"]
        _head_kw = _het_tgn_kwargs_from_checkpoint(checkpoint, config)

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
            instrument_raw_dim=_head_kw["instrument_raw_dim"],
            use_concat_head=_head_kw["use_concat_head"],
            use_contranorm=_head_kw["use_contranorm"],
        )
        if _head_kw["use_concat_head"]:
            log.info(
                "Restored return_concat_head (instrument_raw_dim=%d).",
                _head_kw["instrument_raw_dim"],
            )

        missing, unexpected, skipped = _load_state_dict_skip_shape_mismatch(
            trainer._model,
            checkpoint["model_state_dict"],
            log_prefix="load_model: ",
        )
        if missing or unexpected:
            log.warning(
                "load_model: %d missing keys, %d unexpected keys (strict=False).",
                len(missing),
                len(unexpected),
            )
        _warn_on_checkpoint_schema_drift(
            checkpoint=checkpoint,
            model=trainer._model,
            skipped=skipped,
            missing=missing,
            path=path,
        )
        trainer._model = trainer._model.to(trainer._device)
        trainer._optimizer = torch.optim.Adam(trainer._model.parameters(), lr=config.learning_rate)

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
            log.info("No EWC state in checkpoint (pre-Phase-46 model). Run train() to compute Fisher diagonal.")

        log.info("Model loaded from %s.", path)
        return trainer

    @classmethod
    def load_model_with_epoch_weights(
        cls,
        full_checkpoint_path: str | Path,
        per_epoch_path: str | Path,
        store: PipelineStore,
    ) -> Trainer:
        """Rebuild HetTGN from a full ``save_model`` checkpoint, then swap in
        weights from ``epoch_NNN.pt``.

        Per-epoch files only store ``model_state`` (plus optimizer/history);
        they lack ``config`` / ``metadata_*`` / ``in_channels`` needed to
        construct the graph and tensors.  Use this for IC/backtests when the
        exported ``gnn_model_*.pt`` lags the last completed training epoch.

        Args:
            full_checkpoint_path: Path to ``save_model`` output (e.g. ``gnn_model_h_g.pt``).
            per_epoch_path: Path to ``epoch_052.pt`` from the same training run.
            store: Pipeline store for graph building.

        Returns:
            Trainer with architecture from ``full_checkpoint_path`` and weights
            from ``per_epoch_path``.
        """
        full_checkpoint_path = Path(full_checkpoint_path)
        per_epoch_path = Path(per_epoch_path)
        if not full_checkpoint_path.exists():
            raise FileNotFoundError(f"Full checkpoint not found: {full_checkpoint_path}")
        if not per_epoch_path.exists():
            raise FileNotFoundError(f"Per-epoch checkpoint not found: {per_epoch_path}")

        checkpoint = torch.load(
            full_checkpoint_path,
            map_location=_checkpoint_map_location(),
            weights_only=False,
        )
        ep = torch.load(per_epoch_path, map_location="cpu", weights_only=False)
        if not isinstance(ep, dict):
            raise RuntimeError(f"Unexpected checkpoint type in {per_epoch_path}")

        epoch_state = ep.get("model_state") or ep.get("model_state_dict")
        if epoch_state is None:
            raise KeyError(
                f"{per_epoch_path} has neither 'model_state' nor 'model_state_dict' "
                "(not a per-epoch or full trainer checkpoint)."
            )

        arch_ckpt = dict(checkpoint)
        merged_state = dict(checkpoint.get("model_state_dict", {}))
        merged_state.update(epoch_state)
        arch_ckpt["model_state_dict"] = merged_state

        config = TrainerConfig(**checkpoint["config"])
        trainer = cls(store, config)
        trainer._device = _resolve_torch_device(trainer.config.device)
        if str(trainer._device) != trainer.config.device:
            trainer.config.device = str(trainer._device)

        metadata = (
            checkpoint["metadata_node_types"],
            [tuple(t) for t in checkpoint["metadata_edge_types"]],
        )
        in_channels = checkpoint["in_channels"]
        num_nodes = checkpoint["num_nodes"]
        _head_kw = _het_tgn_kwargs_from_checkpoint(arch_ckpt, config)

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
            instrument_raw_dim=_head_kw["instrument_raw_dim"],
            use_concat_head=_head_kw["use_concat_head"],
            use_contranorm=_head_kw["use_contranorm"],
        )
        if _head_kw["use_concat_head"]:
            log.info(
                "Restored return_concat_head from epoch weights (instrument_raw_dim=%d).",
                _head_kw["instrument_raw_dim"],
            )

        _load_state_dict_skip_shape_mismatch(
            trainer._model,
            checkpoint["model_state_dict"],
            log_prefix="load_model_with_epoch_weights (full): ",
        )
        missing, unexpected, _ = _load_state_dict_skip_shape_mismatch(
            trainer._model,
            epoch_state,
            log_prefix="load_model_with_epoch_weights (epoch): ",
        )
        if missing or unexpected:
            log.warning(
                "load_model_with_epoch_weights: %d missing keys, %d unexpected keys " "(strict=False).",
                len(missing),
                len(unexpected),
            )
        trainer._model = trainer._model.to(trainer._device)
        trainer._optimizer = torch.optim.Adam(trainer._model.parameters(), lr=config.learning_rate)
        ep_num = ep.get("epoch", "?")
        log.info(
            "Overlayed per-epoch weights from %s (checkpoint epoch=%s) onto " "architecture from %s.",
            per_epoch_path,
            ep_num,
            full_checkpoint_path,
        )
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
    device = next(model.parameters()).device
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

    _lo = datetime.fromtimestamp(eval_obs[0].get("observed_at", 0.0), tz=UTC).date().isoformat()
    _hi = datetime.fromtimestamp(eval_obs[-1].get("observed_at", 0.0), tz=UTC).date().isoformat()
    log.info(
        "evaluate(split=%s): %d obs, date range [%s → %s] (no future obs in this split)",
        split,
        len(eval_obs),
        _lo,
        _hi,
    )

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
            # evaluate() is a module-level helper (no Trainer instance), so
            # TS2Vec cache must come from the passed model/config, not `self`.
            data, id_map, events = graph_builder.build_from_cached(
                cached_id_map,
                cached_links,
                observations=window_obs,
                ts2vec_embeddings=getattr(model, "_ts2vec_embeddings", None),
                ts2vec_dim=cfg.ts2vec_dim if cfg.use_ts2vec else 0,
            )
            if not data.node_types:
                continue

            data = data.to(device)
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
                [o for o in obs_by_entity.get(eid_a, []) if o.get("observation_type") == cp.obs_type_a],
                key=lambda o: o.get("observed_at", 0.0),
            )
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
        self._graph_builder = GraphBuilder(
            store,
            zero_price_feats=self.config.zero_price_feats,
        )

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
