"""
TirraMind — Neural Hawkes Process (Idea 6)

Replaces the hand-coded causal template library in ``ConvergenceDetector``
with a learned multivariate temporal point process.

Problem
-------
``ConvergenceDetector`` classifies detected signal cliques by matching them
against a 50-template hand-coded library of causal chains:

    "sanctions → shipping diversion → commodity spike → PMI decline"

The template library:
  - Is static — new causal patterns (e.g., a new geopolitical regime) go
    undetected until someone manually writes a new template.
  - Assigns a match score but not a forward probability — "this looks like
    template X" vs. "P(commodity spike in next 72h) = 0.73".
  - Cannot learn from past outcomes — errors never self-correct.

Solution — Neural Hawkes Process
---------------------------------
A Hawkes process models **self-exciting events**: past events raise the
probability of future events with a learned excitation kernel.

Neural Hawkes (Mei & Eisner, NeurIPS 2017) replaces the fixed exponential
kernel α·exp(−β·Δt) with an LSTM that learns the full excitation history:

    h_t, c_t = LSTM(concat(TypeEmb[k], [Δt]), h_{t−1}, c_{t−1})

Between events the hidden state decays exponentially:
    h(t) = h_n ⊙ exp(−softplus(w_decay) · (t − t_n))

Conditional intensity at time t after the last event:
    λ*_k(t) = softplus(W_out · h(t) + b_out)_k    ∀ event type k

Forecast probability for event type k in the next T seconds:
    P(k in [0,T]) ≈ 1 − exp(−λ*_k · T)     (Poisson approximation)

Training
--------
The LSTM is trained by maximum likelihood on the observed event sequence.
We use the **simplified NLL**:

    L = −Σ_i log P(type = k_i | history up to t_i)
      = CrossEntropy(intensity_logits[:-1], types[1:])

This is the classification-only component of the full NHP log-likelihood.
It learns *which events follow which other events* and *with what timing*,
which is sufficient for TirraMind's forward-probability application.

The full Mei & Eisner log-likelihood adds a survival term
``−∫₀^T Σ_k λ*_k(s) ds`` that requires Monte Carlo integration.
This adds correctness for timing prediction but significant complexity.
The simplified version captures the self-exciting structure and causal
ordering that TirraMind needs.

Output
------
``NeuralHawkesEncoder.run(store, as_of)`` returns
``dict[event_type_str, float]`` — probability in (0,1) for each observed
event type occurring in the next ``forecast_hours`` hours.

Results are stored as ``hawkes.<event_type>.intensity_<H>h`` signals in the
pipeline store.  The ``ConvergenceDetector`` registers these signals under
the ``"event_prediction"`` category as learned leading indicators.

References
----------
    Mei, H. & Eisner, J. (2017).
        The Neural Hawkes Process: A Neurally Self-Modulating Multivariate
        Point Process. NeurIPS 2017.
        https://arxiv.org/abs/1612.09328

    Hongrui24/NeuralHawkesPytorch — MIT licence.
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

_SECS_PER_HOUR: float = 3_600.0
_MIN_EVENTS: int = 4  # minimum events per sequence for training
_MIN_VOCAB_SIZE: int = 2  # need at least 2 event types for a meaningful model
_PAD_IDX: int = 0  # index reserved for padding (never trained)


# ═══════════════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class HawkesResult:
    """Per-event-type intensity forecast from Neural Hawkes.

    Attributes
    ----------
    event_type : str
        Observation type string (e.g., ``"ais_position"``, ``"price"``).
    intensity : float
        Conditional intensity λ*_k at ``as_of`` (events per second).
        Dimensionless — units match the training event rate.
    prob_72h : float
        Poisson-approximated probability of event k in next 72 hours.
        ``P = 1 − exp(−λ*_k · 72 × 3600)``
    forecast_hours : float
        Forecast window used for prob computation.
    computed_at : float
        Unix timestamp of computation.
    """

    event_type: str
    intensity: float
    prob_72h: float
    forecast_hours: float
    computed_at: float
    details: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# PyTorch Model
# ═══════════════════════════════════════════════════════════════════════════


class _NHPModel(nn.Module):
    """Continuous-time LSTM with per-dimension hidden-state decay.

    Architecture (Mei & Eisner 2017, simplified):
      - Type embedding for each event category.
      - Δt (time since last event) appended to the embedding.
      - LSTMCell processes one event at a time.
      - Between events: hidden state decays as h·exp(−softplus(w)·Δt).
      - Conditional intensity: softplus(W_out·h + b_out).
    """

    def __init__(self, n_types: int, hidden_dim: int = 64, emb_dim: int = 16) -> None:
        super().__init__()
        self.n_types = n_types
        self.hidden_dim = hidden_dim

        # +1: type 0 is reserved for the BOS (begin-of-sequence) token
        self.type_emb = nn.Embedding(n_types + 1, emb_dim, padding_idx=_PAD_IDX)

        # Input: [type_emb | log(Δt + 1)] — log-scale for numerical stability
        self.lstm_cell = nn.LSTMCell(emb_dim + 1, hidden_dim)

        # Per-dimension decay rate (one scalar per hidden unit)
        self.log_decay = nn.Parameter(torch.zeros(hidden_dim))

        # Output: conditional intensity logits, one per event type
        # Index 0 is unused (padding); output dim = n_types (indices 1..n_types)
        self.intensity_head = nn.Linear(hidden_dim, n_types)

    def _decay_hidden(self, h: torch.Tensor, dt: float) -> torch.Tensor:
        """Apply per-dimension exponential decay to hidden state."""
        # decay_rate: softplus ensures it's positive
        rate = F.softplus(self.log_decay)  # (hidden_dim,)
        return h * torch.exp(-rate * dt)

    def forward(
        self,
        types: torch.Tensor,  # (T,) long — event type indices 1..n_types
        delta_ts: torch.Tensor,  # (T,) float — seconds since previous event
    ) -> torch.Tensor:
        """Forward pass over one event sequence.

        Returns
        -------
        torch.Tensor  shape (T, n_types)
            Intensity logits at each event step (before softplus).
            Row i is the intensity state *after* processing event i.
        """
        T = types.shape[0]
        h = torch.zeros(1, self.hidden_dim, device=types.device)
        c = torch.zeros(1, self.hidden_dim, device=types.device)

        all_h: list[torch.Tensor] = []

        for i in range(T):
            dt = float(delta_ts[i].item())
            # Decay hidden state by dt before processing this event
            h = self._decay_hidden(h, dt)

            emb = self.type_emb(types[i].unsqueeze(0))  # (1, emb_dim)
            log_dt = torch.tensor(
                [[math.log(dt + 1.0)]], dtype=torch.float32, device=types.device
            )
            x = torch.cat([emb, log_dt], dim=-1)  # (1, emb+1)
            h, c = self.lstm_cell(x, (h, c))
            all_h.append(h)

        stacked = torch.cat(all_h, dim=0)  # (T, hidden_dim)
        logits = self.intensity_head(stacked)  # (T, n_types)
        return logits

    def predict_at(
        self,
        types: torch.Tensor,
        delta_ts: torch.Tensor,
        forecast_dt: float,
    ) -> torch.Tensor:
        """Return per-type intensity after processing history + forecast_dt decay.

        Parameters
        ----------
        types, delta_ts : history sequence (T,)
        forecast_dt : seconds into the future

        Returns
        -------
        torch.Tensor  shape (n_types,)
            Per-type conditional intensity λ*_k (softplus-activated).
        """
        with torch.no_grad():
            logits = self.forward(types, delta_ts)  # (T, n_types)
            h_last = self._h_at_step(types, delta_ts)  # (1, hidden_dim)
            h_future = self._decay_hidden(h_last, forecast_dt)
            intensity_logits = self.intensity_head(h_future)  # (1, n_types)
            return F.softplus(intensity_logits.squeeze(0))  # (n_types,)

    def _h_at_step(
        self,
        types: torch.Tensor,
        delta_ts: torch.Tensor,
    ) -> torch.Tensor:
        """Run LSTM forward and return the final hidden state (1, hidden_dim)."""
        h = torch.zeros(1, self.hidden_dim, device=types.device)
        c = torch.zeros(1, self.hidden_dim, device=types.device)
        for i in range(types.shape[0]):
            dt = float(delta_ts[i].item())
            h = self._decay_hidden(h, dt)
            emb = self.type_emb(types[i].unsqueeze(0))
            log_dt = torch.tensor(
                [[math.log(dt + 1.0)]], dtype=torch.float32, device=types.device
            )
            x = torch.cat([emb, log_dt], dim=-1)
            h, c = self.lstm_cell(x, (h, c))
        return h

    def get_hidden_sequence(
        self,
        types: torch.Tensor,
        delta_ts: torch.Tensor,
    ) -> torch.Tensor:
        """Return LSTM hidden states h(t_k) for every event k.

        Same computation as forward() but returns the raw hidden states
        instead of running them through the intensity head.  Used by
        ContinuousWorldModel to build the temporal skeleton of the
        control path Z(t) for the M1 Neural SDE.

        Args:
            types:     (T,) long — event type indices 1..n_types.
            delta_ts:  (T,) float — seconds since previous event.

        Returns:
            (T, hidden_dim) — h(t_k) after processing event k.
        """
        T = types.shape[0]
        h = torch.zeros(1, self.hidden_dim, device=types.device)
        c = torch.zeros(1, self.hidden_dim, device=types.device)
        all_h: list[torch.Tensor] = []

        for i in range(T):
            dt = float(delta_ts[i].item())
            h = self._decay_hidden(h, dt)
            emb = self.type_emb(types[i].unsqueeze(0))
            log_dt = torch.tensor(
                [[math.log(dt + 1.0)]], dtype=torch.float32, device=types.device
            )
            x = torch.cat([emb, log_dt], dim=-1)
            h, c = self.lstm_cell(x, (h, c))
            all_h.append(h)  # each h: (1, hidden_dim)

        return torch.cat(all_h, dim=0)  # (T, hidden_dim)


# ═══════════════════════════════════════════════════════════════════════════
# NeuralHawkesEncoder — high-level API
# ═══════════════════════════════════════════════════════════════════════════


class NeuralHawkesEncoder:
    """Train Neural Hawkes on TirraMind's observation stream and produce
    forward-looking per-event-type probability forecasts.

    Parameters
    ----------
    hidden_dim : int
        LSTM hidden dimension.  Default 64.
    emb_dim : int
        Type embedding dimension.  Default 16.
    n_iters : int
        Training iterations (passes over all sequences).  Default 200.
    lr : float
        Adam learning rate.  Default 1e-3.
    forecast_hours : float
        Forecast horizon for prob_72h computation.  Default 72.
    session_days : int
        Each training sequence spans this many days.  Default 7.
    device : str | None
        Torch device.  Auto-selects CUDA if available.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        emb_dim: int = 16,
        n_iters: int = 200,
        lr: float = 1e-3,
        forecast_hours: float = 72.0,
        session_days: int = 7,
        device: str | None = None,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.n_iters = n_iters
        self.lr = lr
        self.forecast_hours = forecast_hours
        self.session_days = session_days
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._model: _NHPModel | None = None
        self._vocab: dict[str, int] = {}  # event_type_str → 1-indexed int
        self._inv_vocab: dict[int, str] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def run(
        self,
        store: Any,
        as_of: float | None = None,
    ) -> dict[str, HawkesResult]:
        """Full pipeline: extract history from store, train, predict.

        Args:
            store: PipelineStore instance.
            as_of: Reference time.  Defaults to ``time.time()``.

        Returns:
            Dict mapping event_type_str → HawkesResult.
        """
        if as_of is None:
            as_of = time.time()

        # Load all observations (up to 365 days)
        obs = store.query_all_observations(since=as_of - 365 * 86_400.0, until=as_of)
        if not obs:
            log.info("NeuralHawkesEncoder: no observations — skipping.")
            return {}

        # Build vocabulary and sequences
        self._build_vocab(obs)
        if len(self._vocab) < _MIN_VOCAB_SIZE:
            log.info(
                "NeuralHawkesEncoder: only %d event type(s) — need ≥ %d.",
                len(self._vocab),
                _MIN_VOCAB_SIZE,
            )
            return {}

        sequences = self._build_sequences(obs)
        if not sequences:
            log.info("NeuralHawkesEncoder: no sequences long enough to train on.")
            return {}

        # Train
        self._train(sequences)

        # Predict
        results = self._predict(obs, as_of)
        log.info(
            "NeuralHawkesEncoder: predicted intensities for %d event types.",
            len(results),
        )
        return results

    def get_hidden_sequence(
        self,
        obs: list[dict[str, Any]],
    ) -> torch.Tensor | None:
        """Return LSTM hidden states for a list of observation dicts.

        Converts raw observation dicts to the (types, delta_ts) format,
        then calls _NHPModel.get_hidden_sequence().  Used by
        ContinuousWorldModel to build the Hawkes temporal skeleton for Z(t).

        Args:
            obs: List of observation dicts with keys ``observation_type``
                 and ``observed_at``.  Must be in chronological order.

        Returns:
            (n, hidden_dim) tensor, or None if model not trained / no events.
        """
        if self._model is None or not obs:
            return None

        types_list: list[int] = []
        delta_ts_list: list[float] = []
        prev_t: float | None = None

        for o in sorted(obs, key=lambda x: float(x.get("observed_at", 0.0))):
            t = float(o.get("observed_at", 0.0))
            etype = o.get("observation_type", "unknown")
            idx = self._vocab.get(etype)
            if idx is None:
                continue
            dt = (t - prev_t) if prev_t is not None else 0.0
            types_list.append(idx)
            delta_ts_list.append(max(dt, 0.0))
            prev_t = t

        if not types_list:
            return None

        types_t = torch.tensor(types_list, dtype=torch.long, device=self.device)
        dt_t = torch.tensor(delta_ts_list, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self._model.get_hidden_sequence(types_t, dt_t)  # (n, hidden_dim)

    def store_results(
        self,
        results: dict[str, HawkesResult],
        store: Any,
    ) -> int:
        """Persist intensity forecasts as pipeline signals.

        Signal names: ``hawkes.<event_type>.intensity_<H>h``
        Value: ``prob_72h`` (probability in [0,1])
        """
        count = 0
        for event_type, result in results.items():
            h = int(self.forecast_hours)
            signal_name = f"hawkes.{event_type}.intensity_{h}h"
            try:
                store.store_signal(
                    signal_name=signal_name,
                    value=result.prob_72h,
                    metadata={
                        "intensity": result.intensity,
                        "forecast_hours": result.forecast_hours,
                        "computed_at": result.computed_at,
                        "event_type": event_type,
                    },
                )
                count += 1
            except Exception:
                log.warning(
                    "NeuralHawkesEncoder: failed to store signal for %r",
                    event_type,
                    exc_info=True,
                )
        log.info("NeuralHawkesEncoder: stored %d intensity signals.", count)
        return count

    # ── Vocabulary ─────────────────────────────────────────────────────────

    def _build_vocab(self, obs: list[dict[str, Any]]) -> None:
        """Build event type vocabulary from observation_type field."""
        types = sorted({o.get("observation_type", "unknown") for o in obs})
        # Index 0 reserved for padding; types start at 1
        self._vocab = {t: i + 1 for i, t in enumerate(types)}
        self._inv_vocab = {v: k for k, v in self._vocab.items()}
        log.debug("NeuralHawkesEncoder: vocabulary size = %d types", len(self._vocab))

    # ── Sequence construction ──────────────────────────────────────────────

    def _build_sequences(
        self,
        obs: list[dict[str, Any]],
    ) -> list[tuple[list[int], list[float]]]:
        """Partition observations into fixed-duration sessions.

        Returns list of (types_list, delta_ts_list) for training.
        Each session spans ``session_days`` days.
        """
        if not obs:
            return []

        sorted_obs = sorted(obs, key=lambda o: o.get("observed_at", 0.0))
        t_start = float(sorted_obs[0].get("observed_at", 0.0))
        t_end = float(sorted_obs[-1].get("observed_at", 0.0))

        session_secs = self.session_days * 86_400.0
        if t_end - t_start < session_secs:
            # Not enough history for a full session — use what we have
            session_secs = max(t_end - t_start, 1.0)

        sequences: list[tuple[list[int], list[float]]] = []
        cursor = t_start

        while cursor < t_end:
            window_obs = [
                o
                for o in sorted_obs
                if cursor <= o.get("observed_at", 0.0) < cursor + session_secs
            ]
            cursor += session_secs

            if len(window_obs) < _MIN_EVENTS:
                continue

            types: list[int] = []
            delta_ts: list[float] = []
            prev_t: float | None = None

            for o in sorted(window_obs, key=lambda x: x.get("observed_at", 0.0)):
                t = float(o.get("observed_at", 0.0))
                etype = o.get("observation_type", "unknown")
                idx = self._vocab.get(etype)
                if idx is None:
                    continue
                dt = (t - prev_t) if prev_t is not None else 0.0
                types.append(idx)
                delta_ts.append(max(dt, 0.0))
                prev_t = t

            if len(types) >= _MIN_EVENTS:
                sequences.append((types, delta_ts))

        return sequences

    # ── Training ───────────────────────────────────────────────────────────

    def _train(
        self,
        sequences: list[tuple[list[int], list[float]]],
    ) -> None:
        """Train _NHPModel on observed sequences using cross-entropy NLL."""
        n_types = len(self._vocab)
        self._model = _NHPModel(
            n_types=n_types,
            hidden_dim=self.hidden_dim,
            emb_dim=self.emb_dim,
        ).to(self.device)

        optimiser = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        self._model.train()

        for iteration in range(self.n_iters):
            total_loss = 0.0
            n_batches = 0

            for types_list, delta_ts_list in sequences:
                if len(types_list) < 2:
                    continue

                types_t = torch.tensor(types_list, dtype=torch.long, device=self.device)
                dt_t = torch.tensor(
                    delta_ts_list, dtype=torch.float32, device=self.device
                )

                optimiser.zero_grad()

                logits = self._model(types_t, dt_t)  # (T, n_types)

                # Predict type of event i+1 from history up to i
                # types_t uses 1-based indices; CE expects 0-based
                input_logits = logits[:-1]  # (T-1, n_types)
                target = types_t[1:] - 1  # (T-1,) 0-based

                loss = F.cross_entropy(input_logits, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimiser.step()

                total_loss += loss.item()
                n_batches += 1

            if n_batches > 0 and (iteration + 1) % 50 == 0:
                log.debug(
                    "NeuralHawkes iter %d/%d — avg_loss=%.4f",
                    iteration + 1,
                    self.n_iters,
                    total_loss / n_batches,
                )

        self._model.eval()

    # ── Prediction ─────────────────────────────────────────────────────────

    def _predict(
        self,
        obs: list[dict[str, Any]],
        as_of: float,
    ) -> dict[str, HawkesResult]:
        """Run prediction on the recent history up to as_of."""
        if self._model is None:
            return {}

        # Use the most recent session_days of observations as context
        context_start = as_of - self.session_days * 86_400.0
        context_obs = sorted(
            [o for o in obs if float(o.get("observed_at", 0.0)) >= context_start],
            key=lambda o: o.get("observed_at", 0.0),
        )

        if not context_obs:
            return {}

        types_list: list[int] = []
        delta_ts_list: list[float] = []
        prev_t: float | None = None

        for o in context_obs:
            t = float(o.get("observed_at", 0.0))
            etype = o.get("observation_type", "unknown")
            idx = self._vocab.get(etype)
            if idx is None:
                continue
            dt = (t - prev_t) if prev_t is not None else 0.0
            types_list.append(idx)
            delta_ts_list.append(max(dt, 0.0))
            prev_t = t

        if len(types_list) < 1:
            return {}

        types_t = torch.tensor(types_list, dtype=torch.long, device=self.device)
        dt_t = torch.tensor(delta_ts_list, dtype=torch.float32, device=self.device)
        forecast_dt = self.forecast_hours * _SECS_PER_HOUR

        intensity = self._model.predict_at(types_t, dt_t, forecast_dt)  # (n_types,)
        intensity_np = intensity.cpu().numpy()

        results: dict[str, HawkesResult] = {}
        for idx_0based, lam in enumerate(intensity_np):
            idx_1based = idx_0based + 1
            event_type = self._inv_vocab.get(idx_1based)
            if event_type is None:
                continue
            prob = float(1.0 - math.exp(-float(lam) * forecast_dt))
            prob = min(max(prob, 0.0), 1.0)
            results[event_type] = HawkesResult(
                event_type=event_type,
                intensity=float(lam),
                prob_72h=prob,
                forecast_hours=self.forecast_hours,
                computed_at=as_of,
            )

        return results
