"""
TirraMind — Graph Deviation Network Monitor (Idea 10)

Detects structural anomalies in TirraMind's entity graph by learning the
expected co-movement patterns between entities and flagging deviations.

Problem
-------
TirraMind's GNN produces predictions from the *current* graph structure.
When a relationship structurally breaks — a company goes bankrupt, a port
closes, a trade route shifts, a sanctions regime changes — the graph
continues producing predictions based on stale topology.  This silent
structural mismatch is undetectable without a dedicated monitor.

Worse, structural breaks often **precede** market events by days:
  - A vessel stops pinging AIS before the sanctions announcement.
  - A company's supplier links go quiet before the earnings miss.
  - A port's freight volume drops before the commodity price spike.

These are early-warning signals hiding in graph structure deviation.

Solution — Graph Deviation Network (Deng & Hooi, KDD 2021)
-----------------------------------------------------------
GDN learns the expected relationship between each entity and its graph
neighbors using **learned graph attention**, then computes per-entity
deviation scores at inference time.

Architecture:

1. **Sensor embeddings**: Each entity i has a learnable embedding e_i ∈ ℝ^d.

2. **Graph structure learning**: Top-K neighbors selected by cosine similarity
   of sensor embeddings:
       A_ij = cos(e_i, e_j) = (e_i · e_j) / (||e_i|| ||e_j||)
   Top-K mask applied per row.  Learned → adapts to observation co-movement.

3. **Graph attention**: For entity i, attention-weighted neighbor embedding:
       α_ij = softmax_j(v^T · LeakyReLU(W · [x_i || x_j || e_i || e_j]))
       z_i  = Σ_{j∈N(i)} α_ij · x_j

4. **Prediction head**: Concatenate own window + neighbor context:
       h_i  = ReLU(W_out · concat(x_i, z_i))
       x̂_i  = W_pred · h_i           (predict next-period mean value)

5. **Deviation score**: Normalised squared error over the test window:
       a_i  = (x̂_i - x_true_i)² / (σ_i² + ε)   (z-score²)

Training: minimise MSE between predicted and actual next-period mean over
the training window, using Adam optimiser.

Output
------
``GDNMonitor.run(store, as_of)`` returns
``dict[entity_id, GDNResult]`` where each result contains:
  - ``deviation_score``: normalised deviation (higher = more anomalous)
  - ``is_anomaly``: deviation_score > threshold
  - ``entity_type``, ``entity_id``, ``computed_at``

Results are stored as signals:
  ``graph_structure.<entity_id>.deviation``   — per-entity L2 signal
  ``graph_structure.<entity_type>.avg_deviation`` — per-type L1 aggregate

The ConvergenceDetector registers these under the ``"anomaly"`` category
as structural break early-warning signals.

References
----------
    Deng, A. & Hooi, B. (2021).
        Graph Neural Network-Based Anomaly Detection in Multivariate
        Time Series. AAAI 2021.
        https://arxiv.org/abs/2106.06947

    KatieBuc/gnnad — MIT licence.
        Used as API reference; this implementation is independent.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

_DAY: float = 86_400.0
_EPS: float = 1e-8
_MIN_ENTITIES: int = 2   # need at least 2 entities for graph attention
_MIN_OBS: int = 4        # minimum bins per entity to train

_VALUE_KEYS = (
    "close", "usd_amount", "value", "estimated_value",
    "goldstein_scale", "btc_amount", "log_return", "num_articles",
)


# ═══════════════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class GDNResult:
    """Per-entity deviation score from the Graph Deviation Network.

    Attributes
    ----------
    entity_id : str
        Entity identifier.
    entity_type : str
        Entity taxonomy type (``"company"``, ``"vessel"``, etc.).
    deviation_score : float
        Normalised deviation from graph-predicted value.
        0 = perfectly expected behaviour.  Higher = more anomalous.
    is_anomaly : bool
        True when ``deviation_score > threshold``.
    computed_at : float
        Unix timestamp of computation.
    """

    entity_id: str
    entity_type: str
    deviation_score: float
    is_anomaly: bool
    computed_at: float
    details: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# PyTorch Model
# ═══════════════════════════════════════════════════════════════════════════


class _GDNModel(nn.Module):
    """Graph Deviation Network (Deng & Hooi 2021, simplified).

    Learns expected inter-entity co-movement via graph attention, then
    predicts each entity's next-period value from its neighbors.

    Parameters
    ----------
    n_nodes : int
        Number of entities (graph nodes).
    window : int
        Input feature window length (number of time bins).
    hidden_dim : int
        Graph attention hidden dimension.
    emb_dim : int
        Sensor embedding dimension.
    top_k : int
        Max neighbors per entity in the learned graph.
    """

    def __init__(
        self,
        n_nodes: int,
        window: int,
        hidden_dim: int = 64,
        emb_dim: int = 16,
        top_k: int = 5,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.window = window
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.top_k = min(top_k, n_nodes - 1)

        # Learnable sensor embedding per entity
        self.node_emb = nn.Embedding(n_nodes, emb_dim)

        # Graph attention: concat(x_i, x_j, e_i, e_j) → scalar attention
        attn_in = window * 2 + emb_dim * 2
        self.attn_v = nn.Parameter(torch.empty(hidden_dim))
        nn.init.uniform_(self.attn_v, -1.0 / math.sqrt(hidden_dim), 1.0 / math.sqrt(hidden_dim))
        self.attn_W = nn.Linear(attn_in, hidden_dim, bias=False)

        # Output: concat(x_i, z_i) → prediction
        self.out_proj = nn.Linear(window * 2, hidden_dim)
        self.pred_head = nn.Linear(hidden_dim, 1)

    def _learned_graph(self, device: torch.device) -> torch.Tensor:
        """Build top-K adjacency mask from sensor embedding cosine similarity.

        Returns:
            adj: (n_nodes, n_nodes) bool mask, True = neighbor.
        """
        embs = self.node_emb.weight          # (n, emb_dim)
        # Cosine similarity matrix
        norm = embs.norm(dim=-1, keepdim=True).clamp(min=_EPS)
        normed = embs / norm
        sim = normed @ normed.T              # (n, n)
        # Zero out diagonal (self-connections)
        sim = sim.fill_diagonal_(-1e9)
        # Top-K mask
        k = self.top_k
        topk_vals, _ = sim.topk(k, dim=-1)  # (n, k)
        threshold = topk_vals[:, -1].unsqueeze(-1)  # (n, 1)
        adj = sim >= threshold               # (n, n) bool
        return adj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (n_nodes, window) — windowed mean-value features.

        Returns:
            x_hat: (n_nodes,) — predicted next-period value per entity.
        """
        n = self.n_nodes
        device = x.device

        node_ids = torch.arange(n, device=device)
        embs = self.node_emb(node_ids)          # (n, emb_dim)

        adj = self._learned_graph(device)        # (n, n) bool

        # Graph attention: for each node i, attend over neighbors j
        # Expand x to (n, n, window): row i = x_i broadcast, col j = x_j
        x_i = x.unsqueeze(1).expand(n, n, self.window)     # (n, n, w)
        x_j = x.unsqueeze(0).expand(n, n, self.window)     # (n, n, w)
        e_i = embs.unsqueeze(1).expand(n, n, self.emb_dim) # (n, n, d)
        e_j = embs.unsqueeze(0).expand(n, n, self.emb_dim) # (n, n, d)

        # Attention input: concat(x_i, x_j, e_i, e_j)
        attn_in = torch.cat([x_i, x_j, e_i, e_j], dim=-1)  # (n, n, w*2+d*2)
        h = F.leaky_relu(self.attn_W(attn_in))               # (n, n, hidden)
        scores = (h * self.attn_v).sum(dim=-1)                # (n, n)

        # Mask out non-neighbors with -inf
        scores = scores.masked_fill(~adj, float("-inf"))

        # Softmax over neighbors (rows with all -inf → uniform via clamp)
        alpha = torch.softmax(scores, dim=-1)                 # (n, n)
        alpha = torch.nan_to_num(alpha, nan=0.0)

        # Neighbor aggregation: z_i = sum_j alpha_ij * x_j
        z = alpha @ x                                         # (n, window)

        # Prediction: concat(x_i, z_i) → hidden → scalar
        h_out = F.relu(self.out_proj(torch.cat([x, z], dim=-1)))  # (n, hidden)
        x_hat = self.pred_head(h_out).squeeze(-1)                 # (n,)

        return x_hat


# ═══════════════════════════════════════════════════════════════════════════
# GDNMonitor — high-level API
# ═══════════════════════════════════════════════════════════════════════════


class GDNMonitor:
    """Train GDN on TirraMind's entity observation history and score deviations.

    Parameters
    ----------
    hidden_dim : int
        Graph attention hidden dimension.  Default 64.
    emb_dim : int
        Sensor embedding dimension.  Default 16.
    top_k : int
        Learned graph neighbours per entity.  Default 5.
    n_iters : int
        Training iterations.  Default 100.
    lr : float
        Adam learning rate.  Default 1e-3.
    lookback_days : int
        Observation history window.  Default 60.
    n_bins : int
        Time bins in the lookback window.  Default 30.
    window : int
        Number of bins used as input features.  Default 10.
    anomaly_threshold : float
        Deviation z-score above which ``is_anomaly = True``.  Default 3.0.
    device : str | None
        Torch device.  Auto-selects CUDA if available.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        emb_dim: int = 16,
        top_k: int = 5,
        n_iters: int = 100,
        lr: float = 1e-3,
        lookback_days: int = 60,
        n_bins: int = 30,
        window: int = 10,
        anomaly_threshold: float = 3.0,
        device: str | None = None,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.top_k = top_k
        self.n_iters = n_iters
        self.lr = lr
        self.lookback_days = lookback_days
        self.n_bins = n_bins
        self.window = window
        self.anomaly_threshold = anomaly_threshold
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._model: _GDNModel | None = None
        self._entity_index: list[tuple[str, str]] = []   # (entity_type, entity_id)
        self._train_std: np.ndarray | None = None         # per-entity normalisation

    # ── Public API ─────────────────────────────────────────────────────────

    def run(
        self,
        store: Any,
        as_of: float | None = None,
    ) -> dict[str, GDNResult]:
        """Train GDN and score all entities.

        Args:
            store: PipelineStore instance.
            as_of: Reference time.  Defaults to ``time.time()``.

        Returns:
            Dict mapping entity_id → GDNResult.
        """
        if as_of is None:
            as_of = time.time()

        t_start = as_of - self.lookback_days * _DAY

        # Load entities + observations
        entities = store.query_all_entities()
        if not entities:
            log.info("GDNMonitor: no entities found.")
            return {}

        obs_all = store.query_all_observations(since=t_start, until=as_of)
        if not obs_all:
            log.info("GDNMonitor: no observations in window.")
            return {}

        # Build entity → observation index
        obs_by_entity: dict[str, list[dict]] = {}
        for o in obs_all:
            eid = o.get("entity_id", "")
            if eid:
                obs_by_entity.setdefault(eid, []).append(o)

        # Keep only entities with enough observations
        valid = [
            (e.get("entity_type", "unknown"), e.get("entity_id", ""))
            for e in entities
            if e.get("entity_id", "") in obs_by_entity
            and len(obs_by_entity[e.get("entity_id", "")]) >= _MIN_OBS
        ]
        if len(valid) < _MIN_ENTITIES:
            log.info(
                "GDNMonitor: only %d entities with sufficient observations (need ≥ %d).",
                len(valid), _MIN_ENTITIES,
            )
            return {}

        self._entity_index = valid
        n_nodes = len(valid)

        # Build feature matrix: (n_nodes, n_bins)
        feat = self._build_feature_matrix(
            valid, obs_by_entity, t_start, as_of
        )   # (n_nodes, n_bins)

        # Train on first (n_bins - window - 1) steps, score on last window+1
        train_steps = self.n_bins - self.window - 1
        if train_steps < 1:
            log.info("GDNMonitor: not enough bins to train (n_bins=%d, window=%d).", self.n_bins, self.window)
            return {}

        self._train(feat, n_nodes, train_steps)
        results = self._score(feat, valid, as_of)

        log.info(
            "GDNMonitor: scored %d entities, %d anomalies detected.",
            len(results),
            sum(1 for r in results.values() if r.is_anomaly),
        )
        return results

    def store_results(
        self,
        results: dict[str, GDNResult],
        store: Any,
    ) -> int:
        """Persist deviation scores as pipeline signals.

        Per entity:   ``graph_structure.<entity_id>.deviation``
        Per type agg: ``graph_structure.<entity_type>.avg_deviation``
        """
        count = 0
        type_scores: dict[str, list[float]] = {}

        for entity_id, res in results.items():
            sig_name = f"graph_structure.{entity_id}.deviation"
            try:
                store.store_signal(
                    signal_name=sig_name,
                    value=res.deviation_score,
                    metadata={
                        "entity_id": entity_id,
                        "entity_type": res.entity_type,
                        "is_anomaly": res.is_anomaly,
                        "computed_at": res.computed_at,
                    },
                )
                count += 1
            except Exception:
                log.warning("GDNMonitor: failed to store %r", sig_name, exc_info=True)

            type_scores.setdefault(res.entity_type, []).append(res.deviation_score)

        # L1 aggregates per entity type
        for etype, scores in type_scores.items():
            avg_score = float(np.mean(scores))
            sig_name = f"graph_structure.{etype}.avg_deviation"
            try:
                store.store_signal(
                    signal_name=sig_name,
                    value=avg_score,
                    metadata={"entity_type": etype, "n_entities": len(scores)},
                )
                count += 1
            except Exception:
                log.warning("GDNMonitor: failed to store %r", sig_name, exc_info=True)

        log.info("GDNMonitor: stored %d signals.", count)
        return count

    # ── Feature matrix ─────────────────────────────────────────────────────

    def _build_feature_matrix(
        self,
        valid: list[tuple[str, str]],
        obs_by_entity: dict[str, list[dict]],
        t_start: float,
        t_end: float,
    ) -> np.ndarray:
        """Build (n_nodes, n_bins) feature matrix of daily mean values."""
        span = max(t_end - t_start, 1.0)
        bin_dur = span / self.n_bins
        n = len(valid)
        feat = np.full((n, self.n_bins), np.nan)

        for i, (_, eid) in enumerate(valid):
            obs = obs_by_entity.get(eid, [])
            sums = np.zeros(self.n_bins)
            cnts = np.zeros(self.n_bins)
            for o in obs:
                t = float(o.get("observed_at", t_start))
                b = min(int((t - t_start) / bin_dur), self.n_bins - 1)
                v = _extract_value(o.get("value", {}))
                if v is not None and math.isfinite(v):
                    sums[b] += v
                    cnts[b] += 1.0
            with np.errstate(invalid="ignore", divide="ignore"):
                row = np.where(cnts > 0, sums / cnts, np.nan)
            # Forward-fill NaNs for continuity
            feat[i] = _forward_fill(row)

        # Normalise per entity: zero-mean, unit-variance
        self._train_std = np.nanstd(feat, axis=1, keepdims=True).clip(min=_EPS)
        train_mean = np.nanmean(feat, axis=1, keepdims=True)
        feat = (feat - train_mean) / self._train_std
        feat = np.nan_to_num(feat, nan=0.0)
        return feat

    # ── Training ───────────────────────────────────────────────────────────

    def _train(
        self,
        feat: np.ndarray,
        n_nodes: int,
        train_steps: int,
    ) -> None:
        """Train _GDNModel to predict next bin value from window of bins."""
        window = self.window
        self._model = _GDNModel(
            n_nodes=n_nodes,
            window=window,
            hidden_dim=self.hidden_dim,
            emb_dim=self.emb_dim,
            top_k=self.top_k,
        ).to(self.device)

        optimiser = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        self._model.train()

        feat_t = torch.tensor(feat, dtype=torch.float32, device=self.device)

        for iteration in range(self.n_iters):
            total_loss = 0.0
            n_steps = 0

            for t in range(train_steps):
                x = feat_t[:, t : t + window]          # (n, window)
                y = feat_t[:, t + window]               # (n,) — next bin

                optimiser.zero_grad()
                y_hat = self._model(x)                  # (n,)
                loss = F.mse_loss(y_hat, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimiser.step()

                total_loss += loss.item()
                n_steps += 1

            if n_steps > 0 and (iteration + 1) % 25 == 0:
                log.debug(
                    "GDN iter %d/%d — avg_loss=%.4f",
                    iteration + 1, self.n_iters, total_loss / n_steps,
                )

        self._model.eval()

    # ── Scoring ────────────────────────────────────────────────────────────

    def _score(
        self,
        feat: np.ndarray,
        valid: list[tuple[str, str]],
        as_of: float,
    ) -> dict[str, GDNResult]:
        """Score each entity on the most recent window."""
        if self._model is None:
            return {}

        window = self.window
        # Use the last `window` bins as input, next bin as ground truth
        x_np = feat[:, -(window + 1) : -1]          # (n, window)
        y_np = feat[:, -1]                           # (n,)

        x_t = torch.tensor(x_np, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            y_hat = self._model(x_t).cpu().numpy()   # (n,)

        # Per-entity normalised deviation: (y - ŷ)² / (train_std² + ε)
        residuals = (y_np - y_hat) ** 2
        # Use training std for normalisation (from _build_feature_matrix)
        std = self._train_std.squeeze(-1) if self._train_std is not None else np.ones(len(valid))
        dev_scores = residuals / (std ** 2 + _EPS)

        # Anomaly threshold in units of normalised deviation
        threshold = self.anomaly_threshold

        results: dict[str, GDNResult] = {}
        for i, (etype, eid) in enumerate(valid):
            score = float(dev_scores[i])
            results[eid] = GDNResult(
                entity_id=eid,
                entity_type=etype,
                deviation_score=score,
                is_anomaly=score > threshold,
                computed_at=as_of,
            )

        return results


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _extract_value(value_dict: Any) -> float | None:
    """Extract numeric scalar from observation value dict."""
    if not isinstance(value_dict, dict):
        return None
    for k in _VALUE_KEYS:
        if k in value_dict:
            try:
                v = float(value_dict[k])
                return v if math.isfinite(v) else None
            except (TypeError, ValueError):
                pass
    return None


def _forward_fill(arr: np.ndarray) -> np.ndarray:
    """Forward-fill NaN values in a 1D array."""
    out = arr.copy()
    last = 0.0
    for i in range(len(out)):
        if not math.isnan(out[i]):
            last = out[i]
        else:
            out[i] = last
    return out
