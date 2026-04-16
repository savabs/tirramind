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

import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.models.gnn.graph_builder import (
    ENTITY_TYPES,
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


class SyntheticGraphGenerator:
    """Generate synthetic entity graphs with known temporal patterns.

    The generated data is inserted directly into a PipelineStore,
    which can then be consumed by GraphBuilder and HetTGN.
    """

    def __init__(
        self,
        num_companies: int = 8,
        num_countries: int = 3,
        num_vessels: int = 4,
        num_wallets: int = 4,
        time_span: float = 86400.0 * 30,  # 30 days
        base_event_rate: float = 0.001,  # events per entity per second
        seed: int = 42,
        patterns: list[InjectedPattern] | None = None,
    ) -> None:
        self.num_entities = {
            "company": num_companies,
            "country": num_countries,
            "vessel": num_vessels,
            "wallet": num_wallets,
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
        """Return plausible observation types for an entity type."""
        mapping = {
            "company": ["insider_trade", "form144_filing", "sell_intent"],
            "country": ["geopolitical_event"],
            "vessel": ["port_call", "vessel_position"],
            "wallet": ["btc_transfer"],
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
    auto_tune_loss_weights: bool = False
    """When True, use learnable uncertainty-based loss weighting
    (Kendall et al. 2018 "Multi-Task Learning Using Uncertainty
    to Weigh Losses") instead of fixed config weights."""


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
        self._model: HetTGN | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._graph_builder = GraphBuilder(store)

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
        """
        if not observations:
            return []
        t_min = observations[0].get("observed_at", 0.0)
        t_max = observations[-1].get("observed_at", 0.0)
        ws = self.config.window_size
        windows = []
        t = t_min
        while t < t_max:
            t_end = t + ws
            win_obs = [
                o for o in observations if t <= o.get("observed_at", 0.0) < t_end
            ]
            if win_obs:
                windows.append((t, t_end, win_obs))
            t = t_end
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

        # Also need entity_type for each entity_id
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
            time_deltas.append(dt)

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
        """
        links = self.store.query_all_entity_links()
        if not links:
            return torch.tensor(0.0)

        all_entities = self.store.query_all_entities()
        eid_to_type = {e["entity_id"]: e["entity_type"] for e in all_entities}

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
            if a_type not in embeddings or b_type not in embeddings:
                continue

            a_local = id_map.local_id(a_type, a_id)
            b_local = id_map.local_id(b_type, b_id)
            if a_local is None or b_local is None:
                continue

            emb_a = embeddings[a_type][a_local]
            emb_b = embeddings[b_type][b_local]

            pos_dist = F.pairwise_distance(
                emb_a.unsqueeze(0),
                emb_b.unsqueeze(0),
            ).squeeze()
            pos_scores.append(pos_dist)

            # Negative: random entity of same type as b
            b_embs = embeddings[b_type]
            n_nodes = b_embs.size(0)
            if n_nodes > 1:
                neg_indices = [j for j in range(n_nodes) if j != b_local][
                    : self.config.num_negative_samples
                ]
                for neg_idx in neg_indices:
                    neg_dist = F.pairwise_distance(
                        emb_a.unsqueeze(0),
                        b_embs[neg_idx].unsqueeze(0),
                    ).squeeze()
                    neg_scores.append(neg_dist)

        if not pos_scores or not neg_scores:
            return torch.tensor(0.0)

        pos_mean = torch.stack(pos_scores).mean()
        neg_mean = torch.stack(neg_scores).mean()
        # Margin loss: positive pairs should be closer
        loss = F.relu(pos_mean - neg_mean + margin)
        return loss

    def train(self) -> dict[str, list[float]]:
        """Run walk-forward self-supervised training.

        Returns:
            Dict with loss curves: 'total', 'obs_type', 'time_delta', 'contrastive'.
        """
        model = self.model
        optimizer = self._optimizer
        cfg = self.config
        model.train()

        train_obs, _, _ = self._split_observations()
        windows = self._make_windows(train_obs)

        history: dict[str, list[float]] = {
            "total": [],
            "obs_type": [],
            "time_delta": [],
            "contrastive": [],
            "value": [],
        }

        for epoch in range(cfg.epochs):
            model.reset_memory()
            epoch_losses = {
                "total": 0.0,
                "obs_type": 0.0,
                "time_delta": 0.0,
                "contrastive": 0.0,
                "value": 0.0,
            }
            n_windows = 0

            for i in range(len(windows) - 1):
                t_start, t_end, curr_obs = windows[i]
                _, _, next_obs = windows[i + 1]

                # Build graph snapshot for current window
                data, id_map, events = self._graph_builder.build(
                    since=None,
                    until=t_end,
                )
                if not data.node_types:
                    continue

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

                # ── obs_type loss ────────────────────────
                obs_loss = torch.tensor(0.0)
                if len(obs_targets) > 0:
                    # Gather embeddings for target entities
                    all_entities = self.store.query_all_entities()
                    eid_to_type = {
                        e["entity_id"]: e["entity_type"] for e in all_entities
                    }

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

                    if target_embs:
                        target_emb_tensor = torch.stack(target_embs)
                        logits = model.obs_type_head(target_emb_tensor)
                        valid_targets = obs_targets[valid_indices]
                        obs_loss = F.cross_entropy(logits, valid_targets)

                # ── time_delta loss ──────────────────────
                dt_loss = torch.tensor(0.0)
                if target_embs:
                    dt_pred = model.time_delta_head(target_emb_tensor).squeeze(-1)
                    valid_dt = dt_targets[valid_indices]
                    dt_loss = F.mse_loss(dt_pred, valid_dt)

                # ── value prediction loss ────────────────
                val_loss = torch.tensor(0.0)
                if target_embs:
                    val_pred = model.value_pred_head(target_emb_tensor).squeeze(-1)
                    valid_val = val_targets[valid_indices]
                    val_loss = F.huber_loss(val_pred, valid_val)

                # ── contrastive loss ─────────────────────
                c_loss = self._contrastive_loss(embeddings, id_map)

                # ── total loss ───────────────────────────
                if self._log_vars is not None:
                    # Uncertainty-weighted multi-task loss
                    # (Kendall et al. 2018): L_k / (2 * sigma_k^2) + ln(sigma_k)
                    # With log_var = ln(sigma^2): exp(-log_var) * L_k + log_var
                    lv = self._log_vars
                    total = (
                        torch.exp(-lv["obs_type"]) * obs_loss
                        + lv["obs_type"]
                        + torch.exp(-lv["time_delta"]) * dt_loss
                        + lv["time_delta"]
                        + torch.exp(-lv["contrastive"]) * c_loss
                        + lv["contrastive"]
                        + torch.exp(-lv["value"]) * val_loss
                        + lv["value"]
                    )
                else:
                    total = (
                        cfg.obs_type_weight * obs_loss
                        + cfg.time_delta_weight * dt_loss
                        + cfg.contrastive_weight * c_loss
                        + cfg.value_weight * val_loss
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
                n_windows += 1

            # Average over windows
            for k in epoch_losses:
                avg = epoch_losses[k] / max(n_windows, 1)
                history[k].append(avg)

            log.info(
                "Epoch %d/%d — loss: %.4f (obs_type: %.4f, dt: %.4f, contrastive: %.4f)",
                epoch + 1,
                cfg.epochs,
                history["total"][-1],
                history["obs_type"][-1],
                history["time_delta"][-1],
                history["contrastive"][-1],
            )
            if self._log_vars is not None:
                eff = self.effective_loss_weights()
                log.info(
                    "  Effective loss weights: obs=%.3f dt=%.3f contr=%.3f val=%.3f",
                    eff["obs_type"],
                    eff["time_delta"],
                    eff["contrastive"],
                    eff["value"],
                )

        return history

    def effective_loss_weights(self) -> dict[str, float]:
        """Return current effective loss weights.

        When auto_tune_loss_weights is on, these are exp(-log_var)
        for each task (the learned precision).  Otherwise returns
        the fixed config weights.
        """
        if self._log_vars is not None:
            return {k: math.exp(-p.item()) for k, p in self._log_vars.items()}
        cfg = self.config
        return {
            "obs_type": cfg.obs_type_weight,
            "time_delta": cfg.time_delta_weight,
            "contrastive": cfg.contrastive_weight,
            "value": cfg.value_weight,
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
            embeddings = model(data, id_map)

        return embeddings, id_map

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
            },
            "metadata_node_types": metadata[0],
            "metadata_edge_types": [list(t) for t in metadata[1]],
            "in_channels": in_channels,
            "num_nodes": id_map.num_nodes,
        }
        torch.save(checkpoint, path)
        log.info("Model saved to %s (%d nodes).", path, id_map.num_nodes)

    @classmethod
    def load_model(cls, path: str | Path, store: PipelineStore) -> "Trainer":
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

        trainer._model.load_state_dict(checkpoint["model_state_dict"])
        trainer._optimizer = torch.optim.Adam(
            trainer._model.parameters(), lr=config.learning_rate
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

    # Build windows
    ws = cfg.window_size
    t_min = eval_obs[0].get("observed_at", 0.0)
    t_max = eval_obs[-1].get("observed_at", 0.0)
    windows = []
    t = t_min
    while t < t_max:
        t_end = t + ws
        win_obs = [o for o in eval_obs if t <= o.get("observed_at", 0.0) < t_end]
        if win_obs:
            windows.append((t, t_end, win_obs))
        t = t_end

    all_entities = store.query_all_entities()
    eid_to_type = {e["entity_id"]: e["entity_type"] for e in all_entities}

    correct_top1 = 0
    correct_top5 = 0
    total_dt_ae = 0.0
    num_preds = 0

    with torch.no_grad():
        for i in range(len(windows) - 1):
            t_start, t_end, curr_obs = windows[i]
            _, _, next_obs = windows[i + 1]

            # Build graph up to current window end (includes train data)
            data, id_map, events = graph_builder.build(since=None, until=t_end)
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
                dt_true = max(0.0, o.get("observed_at", 0.0) - t_ref)
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
