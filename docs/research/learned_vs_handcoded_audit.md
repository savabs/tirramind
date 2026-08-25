---
title: "Research: Learned vs Hand-Coded Audit — Full Subsystem Scorecard"
tags:
  - doc/research
  - phase/24
  - topic/learning-agent
  - topic/self-improving
  - layer/surveillance
  - layer/feature-engineering
  - layer/world-model
  - layer/fusion
  - layer/learning
  - layer/adversarial
  - layer/llm-support
---

# Research: Learned vs Hand-Coded Audit

**Date:** 2026-04-14
**Scope:** Every subsystem in the TirraMind codebase, classified as learned (parameters update from data/experience) vs hand-coded (fixed logic, manual constants, expert rules, heuristics).

---

## Executive Summary

**TirraMind is ~25% learned, ~75% hand-coded.** The learned parts are real and substantive (GNN weights, Bayesian posteriors, RL policy parameters, SAC actor/critic). But the agent's overall behavior is still dominated by engineering: fixed observation matrices, expert-authored causal graphs, hand-tuned thresholds, LLM prompt templates, and deterministic orchestration logic.

The system follows a clear pattern: **observational and structural knowledge is hand-specified; predictive behavior is learned**. This is a reasonable starting point, but it means the agent cannot yet improve its own structure, discover new causal relationships, or adapt its own reward function.

---

## Layer-by-Layer Scorecard

### Layer 1: Surveillance Surface — `agent/tools/` (57 tools)

| Component | Learned | Hand-Coded | Notes |
|-----------|---------|------------|-------|
| API wrappers (57 tools) | 0% | 100% | Deterministic fetch → parse → ToolResult |
| Parameter schemas (JSON) | 0% | 100% | Every tool's params hand-specified |
| Output normalization | 0% | 100% | Fixed parsing per API |
| L2 entity extraction | 0% | 100% | entity_id_from_key() with hand-coded type mapping |
| DataCache (6hr TTL) | 0% | 100% | Infrastructure, no learning |

**Layer 1 total: 0% learned / 100% hand-coded**

This is correct by design. Data fetching should be deterministic. The question is whether the *choice* of which tools to run and when should be learned — and currently that's handled by the bandit (Layer 5) choosing goal categories, not by the tools themselves.

---

### Layer 2: Feature Engineering — `agent/quant/` + `agent/features/`

| Component | Learned | Hand-Coded | Notes |
|-----------|---------|------------|-------|
| EngineeredFeature protocol | 0% | 100% | Frozen dataclass, naming taxonomy, 12 units, 10 horizons |
| Feature builders | 0% | 100% | Fixed mappings from tool outputs → features |
| BOCPD (changepoint.py) | 60% | 40% | NIG conjugate posterior updates from data; fixed hazard λ=200, priors |
| HMM (regime.py) | 70% | 30% | EM-learned means/variances/transitions; fixed n_states=3 |
| Scoring (scoring.py) | 0% | 100% | Textbook Sharpe/Sortino/VaR formulas |
| Spectral (spectral.py) | 0% | 100% | FFT + CWT Morlet, fixed wavelet params |
| GNN feature builder | 0% | 100% | Fixed aggregation of entity observations → node features |

**Layer 2 total: ~25% learned / ~75% hand-coded**

BOCPD and HMM are genuinely statistical learners. But the feature *schema* (what gets computed, how it's named, what horizon) is entirely hand-designed. The GNN feature builder (`gnn_builder.py`) maps entity observations to 39-dim node features via a fixed layout: base(12) + enrichment(27).

---

### Layer 3: World Model — `agent/models/`

| Component | Learned | Hand-Coded | Notes |
|-----------|---------|------------|-------|
| **HetTGN** (het_tgn.py) | **80%** | 20% | Self-supervised NN: per-type projections, HGT attention, GRU memory, 5 prediction heads — all trained. Fixed: architecture hyperparams (layers, heads, hidden_dim) |
| Graph builder (graph_builder.py) | 20% | 80% | Entity/obs type taxonomy, edge definitions, IDMap allocation — all hand-coded. Only embeddings are learned |
| Trainer (trainer.py) | 70% | 30% | Walk-forward training loop, 4-loss weighting (obs_type CE 1.0, time_delta MSE 0.1, contrastive 0.5, value Huber 0.3) — weights are fixed |
| **Expert causal DAG** (initial_graph.py) | 0% | **100%** | 20 nodes, 19 edges, CPDs — all expert-authored. "Weakly informative priors" are hand-set, not learned |
| **Belief propagator** (propagator.py) | 0% | **100%** | pgmpy VariableElimination on expert DAG. Discretization bin edges hand-set |
| **Kalman filter** (state_filter.py) | 0% | **100%** | F, Q per regime, H, R — **all hand-specified**. 3 latent states, 17 obs dims, observation matrix H explicitly maps features to states |
| **WorldModel orchestrator** (world_model.py) | 0% | **100%** | DAG→Kalman coupling, feature→obs mapping, regime extraction — all fixed logic |
| **World model update DAG** | 0% | **100%** | `_FEATURE_TO_OBS_INDEX` maps 17 features to obs vector positions by hand. H matrix entries (0.5, 0.3, 0.4) hand-tuned |
| BeliefState protocol (belief.py) | 0% | 100% | Frozen dataclass, validation rules — correct by design |

**Layer 3 total: ~35% learned / ~65% hand-coded**

This is the critical finding. The GNN is genuinely learned (80%), but the causal world model that sits on top is entirely expert-constructed. The observation matrix H, noise R, transition dynamics F/Q, graph edges, CPDs, and feature→state mappings are all hand-tuned constants. The "world model" is really an expert Bayesian network + Kalman filter that consumes learned GNN features as inputs but doesn't learn its own structure or parameters from data.

---

### Layer 4: Signal Fusion — `agent/fusion/` + `agent/convergence/`

| Component | Learned | Hand-Coded | Notes |
|-----------|---------|------------|-------|
| **Surprise extractor** (surprise.py) | **70%** | 30% | 5 surprise signals derived from trained GNN predictions. But composite weights (0.3, 0.15, 0.25, 0.2, 0.1) are fixed |
| CUSUM (cusum.py) | 0% | 100% | Classical SPC: k=0.5, h=5.0 hand-set |
| Hawkes process (hawkes.py) | 0% | 100% | μ=0.1, α=0.5, β=1.0 hand-set exponential kernel |
| Entity baseline (entity_baseline.py) | 0% | 100% | Rolling stats + z-score threshold |
| Entity scorer (entity_scorer.py) | 40% | 60% | Orchestras GNN→surprise pipeline; all ScorerConfig params manual |
| Convergence detector (detector.py) | 2% | 98% | z_threshold=2.0, p_threshold=0.05, fdr_q=0.05, min_clique_size=3 — all manual. Only correlations computed from data |
| **50 causal templates** (templates.py) | 0% | **100%** | Expert-encoded causal chains (sanctions→shipping→commodity→PMI etc.) |
| Atomic signals (atomic_signals.py) | 0% | 100% | Rolling z-score, percentile, threshold-based anomaly flags |

**Layer 4 total: ~20% learned / ~80% hand-coded**

The surprise extractor is the best part — it turns GNN prediction error into anomaly signal, which is genuinely learned behavior. But everything around it (CUSUM params, Hawkes params, convergence thresholds, causal templates) is hand-tuned domain expertise.

---

### Layer 5: RL Policy — `agent/learning/`

| Component | Learned | Hand-Coded | Notes |
|-----------|---------|------------|-------|
| **Thompson Sampling bandit** (bandit.py) | **100%** | 0% | α/β posterior updates from rewards. Pure RL |
| **SAC actor** (sac.py GaussianActor) | **100%** | 0% | Tanh-squashed Gaussian policy, fully trained via backprop |
| **SAC twin critic** (sac.py TwinCritic) | **100%** | 0% | Clipped double-Q, Polyak soft update — fully trained |
| **Alpha scheduler** (sac.py) | **100%** | 0% | Auto-tuned entropy temperature from Haarnoja 2018b |
| GoalArm definitions (bandit.py) | 0% | 100% | 45 arms: name, description, tools, examples — all hand-authored |
| State assembler (state_assembler.py) | 0% | **100%** | Tensor layout, top-K truncation by composite_surprise, zero-padding, feature grouping — all manual design |
| Replay buffer (replay_buffer.py) | 0% | 100% | Infrastructure (circular numpy buffer) |
| Reward function (reward.py) | 0% | **100%** | Weights: eval=0.4, sharpe=0.3, facts=0.2, novelty=0.1, dead_end=-0.3 — all hand-coded |
| Evaluator (evaluator.py) | 50% | 50% | LLM-based scoring + hand-coded prompt + heuristic fallback |
| Goal generator (goal_generator.py) | 30% | 70% | LLM fills details; prompt templates and dedup logic hand-coded |
| Reflector (reflection.py) | 50% | 50% | LLM reflection; prompt templates and schemas hand-coded |

**Layer 5 total: ~55% learned / ~45% hand-coded**

The actual RL components (bandit θ, SAC actor/critic/alpha) are fully learned. But they sit on top of a heavily engineered foundation: the state assembly, reward computation, arm definitions, and goal generation are all manual. The policy learns *within* a hand-coded action space and reward landscape.

---

### Layer 6: Adversarial — `agent/adversarial/`

| Component | Learned | Hand-Coded | Notes |
|-----------|---------|------------|-------|
| VPIN (vpin.py) | 0% | 100% | BVC formula, spike_threshold, n_buckets — domain knowledge |
| Edge decay monitor (edge_decay.py) | 50% | 50% | BOCPD inference on rolling Sharpe; decay_threshold hand-set |
| Crowding detection (crowding.py) | 0% | 100% | Heuristic threshold-based flags |
| AdversarialFlag protocol | 0% | 100% | Frozen dataclass |

**Layer 6 total: ~15% learned / ~85% hand-coded**

---

### Layer 7: LLM Support — `agent/reasoning/`

| Component | Learned | Hand-Coded | Notes |
|-----------|---------|------------|-------|
| LLM client (llm_client.py) | 0% | 100% | OpenAI-compatible API wrapper |
| System prompts | 0% | 100% | 500+ word identity/worldview prompt |

**Layer 7 total: 0% learned (locally) / 100% hand-coded**

The external LLM (GPT-4/Groq/Ollama) is trained by someone else. The local code is a pass-through.

---

### Meta: Orchestration & Infrastructure

| Component | Learned | Hand-Coded | Notes |
|-----------|---------|------------|-------|
| Orchestrator (orchestrator.py) | 0% | 100% | Research→Plan→Execute→Synthesize pipeline, max_steps=30, replan_limit=2 |
| Task planner (task_planner.py) | 0% | 100% | LLM-driven task decomposition; schema and prompts hand-coded |
| Autonomous loop (autonomous.py) | 40% | 60% | Bandit selects arm (learned); loop structure, reflection, goal gen (hand-coded) |
| Pipeline DAGs (pipeline/dags/) | 0% | 100% | 9 DAGs with fixed schedules, node ordering, dependency graph |
| Pipeline store (pipeline/store.py) | 0% | 100% | SQLite schema (14 tables), deterministic CRUD |

**Meta total: ~8% learned / ~92% hand-coded**

---

## Aggregate Scorecard

| Layer | Description | % Learned | % Hand-Coded | Key Learned Components |
|-------|-------------|-----------|-------------|----------------------|
| L1 | Surveillance Surface | 0% | 100% | — |
| L2 | Feature Engineering | 25% | 75% | BOCPD posteriors, HMM EM params |
| L3 | World Model | 35% | 65% | HetTGN weights (80% of GNN), but expert DAG/Kalman 100% manual |
| L4 | Signal Fusion | 20% | 80% | GNN-derived surprise vectors |
| L5 | RL Policy | 55% | 45% | SAC actor/critic, bandit α/β |
| L6 | Adversarial | 15% | 85% | BOCPD on edge decay |
| L7 | LLM Support | 0% | 100% | — |
| Meta | Orchestration | 8% | 92% | Bandit arm selection |
| **TOTAL** | **All layers** | **~25%** | **~75%** | GNN + SAC + Bandit + BOCPD/HMM |

---

## What Is Genuinely Learned (parameters update from data)

1. **HetTGN weights** — per-type projections, HGT attention heads, GRU memory, 5 prediction heads. Self-supervised from entity-observation sequences. ~150K parameters.
2. **SAC policy** — GaussianActor (tanh-squashed), TwinCritic (double-Q), AlphaScheduler. Trained via RL on portfolio outcomes.
3. **Thompson Sampling bandit** — 45 arms × (α, β) posterior parameters. Updated from scalar rewards after each autonomous iteration.
4. **BOCPD posteriors** — Normal-Inverse-Gamma conjugate updates in changepoint detection and edge decay monitoring.
5. **HMM parameters** — Means, variances, transition matrix learned via EM for regime detection.

## What Is Hand-Coded but Pretends to Be Adaptive

1. **LLM-based evaluation/reflection/goal-gen** — These use an external LLM (GPT-4/Groq), which *is* a trained model, but the local code treats it as a black-box oracle. The prompts, schemas, fallback logic, and scoring rubrics are all hand-authored. The system doesn't fine-tune the LLM or learn to prompt it better.
2. **World model beliefs consumed by SAC** — The BeliefState protocol exists and has `entity_id` fields, but in `inference.py` line 299-302, beliefs are stubbed: `beliefs: list[BeliefState] = []` with a comment "pass empty beliefs — the assembler zero-pads." The learned world model exists but isn't wired into the learned policy.

## What Is Hand-Coded and Should Stay Hand-Coded

1. **Data tool implementations** — API wrappers should be deterministic. The *choice* of which tools to use can be learned, but `yfinance.download()` shouldn't be.
2. **Feature/belief schemas** — Frozen dataclass contracts are infrastructure. Immutability is a safety invariant.
3. **Mathematical formulas** — Sharpe ratio, VaR, Kalman predict/update equations are textbook. No learning needed.
4. **Pipeline infrastructure** — DAG executor, SQLite store, cache TTL — these are engineering.

## What Is Hand-Coded and Shouldn't Be

This is the punchline. These are the components where manual tuning should give way to learned behavior:

### Critical (blocks end-to-end learning)

| What | Where | Why It Matters |
|------|-------|---------------|
| World model structure (DAG edges, CPDs) | `initial_graph.py` | 19 edges and all CPDs are expert-authored. The system can't discover new causal relationships |
| Observation matrix H | `world_model_update.py` | 17×3 matrix hand-set. Feature→state mapping is frozen |
| Kalman dynamics F, Q, R | `world_model_update.py`, `state_filter.py` | Per-regime transition/noise matrices are hand-tuned constants |
| Reward function weights | `reward.py` | eval=0.4, sharpe=0.3, facts=0.2, novelty=0.1 — the reward landscape that drives all bandit learning is itself not learned |
| State assembler layout | `state_assembler.py` | Top-K truncation, feature grouping, zero-padding strategy — all manual decisions that determine what the policy can see |
| Surprise composite weights | `surprise.py` | (0.3, 0.15, 0.25, 0.2, 0.1) — determines how anomaly signals are fused |

### Important (limits adaptation)

| What | Where | Why It Matters |
|------|-------|---------------|
| GoalArm definitions (45 arms) | `bandit.py` | The bandit learns *which* arm to pull, but the arm set itself is static. Can't discover new goal categories |
| Convergence causal templates (50) | `templates.py` | Expert-encoded chains. System can't discover new causal patterns |
| CUSUM/Hawkes/baseline params | `fusion/*.py` | k, h, μ, α, β — all hand-tuned. Could be calibrated online |
| Loss weights in GNN trainer | `trainer.py` | obs_type CE 1.0, time_delta MSE 0.1, contrastive 0.5, value Huber 0.3 — fixed |
| Convergence thresholds | `detector.py` | z_threshold=2.0, p_threshold=0.05, fdr_q=0.05 — could be Bayesian-optimized |
| VPIN parameters | `vpin.py` | spike_threshold, n_buckets — could be learned from order flow |

---

## What "Self-Improving" Would Mean

A self-improving TirraMind would:

1. **Learn its own world model structure** — discover causal edges from data, not use expert-drawn graphs
2. **Learn observation→state mappings** — H, R matrices fit from data, not hand-specified
3. **Learn its own reward function** — meta-RL or inverse RL to adapt what "good" means
4. **Learn state representation** — replace manual state assembly with learned aggregation
5. **Discover goal categories** — generate new bandit arms from experience, not use a fixed set
6. **Calibrate detector thresholds** — online Bayesian optimization of CUSUM/Hawkes/convergence params
7. **Learn surprise fusion weights** — data-driven weighting of the 5 surprise channels
8. **Close the inference loop** — world model beliefs actually flow into SAC at runtime

The gap between current state and self-improving agent is detailed in [[learned_vs_handcoded_architecture_spec]].

---

## Related

- [[learned_vs_handcoded_architecture_spec]] — Architectural changes to make it end-to-end learned
- [[e2e_global_integration]] — Phase 24 research (current phase)
- [[signal_fusion]] — Phase 20 research (surprise is the best learned component)
- [[world_model_bridge]] — Phase 19 research (GNN→world model bridge)
- [[rl_policy]] — Phase 21 research (SAC implementation)
- [[project_memory]] — Persistent architectural memory
