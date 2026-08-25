---
title: "Checkpoint: 2026-04-22 Strategy Session — Cost, Density Mandate, Transformer Vision, Phase 48"
tags:
  - doc/checkpoint
  - phase/46
  - phase/47
  - phase/48
  - topic/transformer
  - topic/cost
  - topic/density
  - status/done
---

# Checkpoint: 2026-04-22 — Strategy Session II
## Cost, Data Density, Transformer World Model, and the Road to Production

---

## What this session was really about

This session started as a simple cost question — "how much to go live?" — and ended as a complete strategic architecture conversation. By the end, we had defined the exact target architecture (transformer world model + Dreamer RL), the exact reason we build through it in four phases rather than jumping straight to it, the exact data quality gate that must pass before a single line of transformer code is written, and the hard economics from $0 dev to $41/month with 100 paying users at 99.9% margin.

This is the session where TirraMind stopped being "a project with a plan" and became "a company with a technical roadmap locked to first principles."

---

## 1. The cost reality — cheaper than a Netflix subscription to start

The headline number: **under $55 total from today to end of month 1 in live production.**

That number is not aspirational. It is the sum of: one Hetzner CX22 server at $14/month, a domain and SSL at $15/month, one GPU retrain costing $0.60, and $0 for everything else — LLM (Ollama local), all 51 data tools (free public APIs), monitoring (BetterUptime free tier), GNN online updates (CPU, 41 seconds), Bayesian world model updates (milliseconds per DAG run), SAC RL updates (CPU), Thompson bandit (pure Python).

The reason the number is this low is architectural: the heavy computation runs **once** per day and writes results to the database. User queries read pre-computed answers. Adding users does not trigger the GNN or the Bayesian update — it triggers a database read. The intelligence is decoupled from the serving layer.

### The 100-user calculation

100 users on this system costs $12/month extra — a server upgrade from CX22 to CX32 (4 vCPU, 8GB RAM). That is the only change. FastAPI serving, Nginx, Supabase auth (free to 50K users), Cloudflare CDN (free) — all $0. The architecture has nearly free marginal cost per user because of that computation/serving decoupling.

| Users | Monthly infra | Delta |
|---|---|---|
| 1 (solo dev) | $29/mo | — |
| 100 | $41/mo | +$12 |
| 1,000 | $41/mo | $0 (Postgres on same box) |
| 10,000 | $60/mo | +$19 (2nd server) |
| 100,000 | $300–800/mo | architecture change |

100 B2B users at $500/month each = $50K MRR on $41/month of infrastructure. **99.9% margin.** That is the compounding advantage of a prediction engine that runs once and serves many.

### What drives costs up (only two triggers)

1. Entity count exceeds 100K → upgrade to 16GB RAM server (+$24/mo)
2. Phase 40 walk-forward proves a specific paid data source would materially improve signal → buy it

Both triggers are data-driven. No guessing, no premature spend.

---

## 2. The training data question — 5 years is right, but not with equal weight

The user correctly identified the risk: training on 5 years of raw historical data means the GNN sees COVID supply chain collapse, 2021 liquidity flood, 2022 rate shock, and 2023 normalisation all with equal weight. That teaches the model that those regime extremes are "normal." The posterior gets contaminated.

The solution is exponential time-decay weighting during training:

$$w_t = e^{-\lambda (T_{now} - t)}$$

With $\lambda = 0.001$ per day:
- Data from last month → weight ~0.97 (almost full signal)
- Data from 1 year ago → weight 0.69
- Data from 3 years ago → weight 0.33
- Data from 5 years ago → weight 0.16

**Why not just cut to 1–2 years:** The GNN has never seen a crisis pattern if you train on 90 days. Supply chain cascades, sanctions waves, and liquidity crises only exist in the historical record. You need the vocabulary. The decay weighting gives you the vocabulary without letting 2021 dominate the gradient updates.

**Practical decision locked in:** Collect all 5 years (Phase 47, free API calls). Train with exponential decay weighting. The GNN trainer must accept a `time_decay_lambda` parameter — this is a hard requirement on Phase 47 spec and Phase 40 training code.

---

## 3. The escalation ladder — what happens if Phase 40 shows no edge

The user asked the honest question: if we don't get edge, we need a bigger model, which drives cost. True. Here is the pre-mapped escalation path:

| Trigger | What to upgrade | One-time cost | Monthly delta |
|---|---|---|---|
| GNN underfitting | HetTGN: hidden_dim 64→128, layers 2→4 | $2–5 GPU run | $0 |
| World model too sparse | pgmpy DAG → PyMC variational (full posterior) | $0 (code change) | $0 |
| RL policy plateau | SAC hidden_dim 128→256 | $0 (still CPU) | $0 |
| Still no edge | Add Polygon.io ($29) + Glassnode ($39) | $0 | +$68/mo |
| Still no edge | Transformer world model (Phase 48, planned) | $50–200 GPU run | +$20–50/mo |
| Still no edge | Hire quant to audit signal stack | $0 in code | people cost |

**The strategic principle behind this table:** Phase 40 must run first so you know *which* layer failed. Throwing a bigger model at an unknown failure mode wastes money. The GNN might be fine — the failure might be a timestamp bug in one backfill tool corrupting 30% of the training signal. No transformer fixes a data bug. Diagnose first, scale after.

---

## 4. Phase 48 — why transformer is the right target architecture

This is the core strategic decision of the session.

The user pushed back on "simple models" and asked why a transformer. The answer operates on three levels simultaneously:

### Level 1: The causal structure argument

The current world model (pgmpy Bayesian DAG) has hand-coded edges. The human defines which nodes cause which. HillClimbSearch can discover some structure from data, but it is brittle on sparse data and cannot scale beyond a few hundred nodes without becoming computationally intractable.

A transformer world model learns the causal structure from the sequence of observations. The attention weights *are* the learned causal graph — and they update automatically as the world changes. No human-coded edges. No structure learning heuristics. The model discovers that "vessel tracking + CFTC positioning + GDELT political events → commodity price move" because it saw that sequence happen in the data, not because someone drew that edge in a DAG.

### Level 2: The scaling law argument

Bayesian DAGs asymptote. Add 10× more data to a pgmpy model and you get marginal improvement in posterior precision. Add 10× more data to a transformer and you get genuinely better predictions — because the transformer has learned more latent structure, more rare event patterns, more cross-domain correlations that only appear in dense history.

This is the fundamental transformer property: **performance scales predictably with data volume.** For a company that is about to have 5 years × 51 tools × 1,087+ entities of observation history, that scaling law is the competitive moat.

### Level 3: The Dreamer RL argument

The current SAC policy learns from real transitions only — it acts, observes the result, updates. It cannot plan ahead. It cannot simulate "what happens if a sanctions cascade starts in country X while shipping routes through Y are already disrupted."

Dreamer-style model-based RL agents plan by simulating millions of imagined futures *inside* the transformer world model before acting. They learn a policy over imagined rollouts, not just real ones. The policy gets orders of magnitude more experience per unit of real-world interaction because most of the learning happens in simulation.

For a prediction system watching geopolitical events with low base rates and high consequence, the ability to plan over imagined scenarios is not a nice-to-have. It is the core intelligence.

### Side-by-side comparison

| | Current Stack | Phase 48 Target |
|---|---|---|
| World model | pgmpy Bayesian DAG — hand-coded causal edges | Transformer over entity-observation sequences — attention IS the learned causal graph |
| Causal discovery | HillClimbSearch (brittle on sparse data) | Learned from sequence. Scales with data. |
| Working memory | GRU per entity (fixed hidden state) | Full attention over context window — every entity attends to every other |
| RL policy | SAC MLP — learns from real transitions only | Dreamer — plans over imagined transformer rollouts |
| Scaling with data | Asymptotes | Improves predictably with 10× more data |
| Cross-entity reasoning | Kalman fusion + GNN (sequential) | All entities attend simultaneously at prediction time |
| Why it matters | Correct scaffold for current scale | The prediction engine becomes the product |

---

## 5. The GNN is not replaced — it feeds the transformer

A critical clarification from this session: the HetTGN GNN is **not** being retired in Phase 48. It provides entity embeddings that become the input token stream for the transformer world model.

```
GNN entity embeddings → transformer world model → Dreamer RL policy
     (Phase 40 trains)       (Phase 48 builds)        (Phase 48 builds)
```

This means Phase 40 (GNN retrain on real data) is not wasted work before Phase 48. The GNN embeddings are the vocabulary tokens that the transformer reads. Better GNN training = richer token representations = better transformer world model. The phases are cumulative, not sequential replacements.

---

## 6. The density mandate — the gate that protects Phase 48 from itself

The user identified the core transformer vulnerability: they are data-hungry. Sparse input → broken attention → garbage predictions. A transformer trained on patchy, thin observations does not produce calibrated uncertainty — it hallucinates confident predictions on entity types it has barely seen.

**The density mandate is now a standing rule in the task file:**

> Transformers are data-hungry. Sparse input = broken attention = garbage predictions. Every phase from 47 onward must track observation density per entity type. Any entity type below threshold gets extended backfill, synthetic augmentation, or additional tool wiring before Phase 48 begins. Density is a first-class exit condition, not an afterthought.

### Hard gate on Phase 48

Phase 48 cannot begin until three conditions are simultaneously true:

1. **Density audit passes:** ≥500 observations per entity type average, no entity type below 100. Run `scripts/density_audit.py` at end of Phase 47.

2. **Phase 40 walk-forward complete:** failure modes identified per layer. Which entities are sparse? Which tools have weak signal? Which cross-domain edges are missing?

3. **Phase 40 ceiling confirmed:** the current stack has hit its performance ceiling on at least one layer. This is the empirical justification for the transformer investment — not enthusiasm, not architecture aesthetics.

### The density repair protocol

If the density audit fails:
1. Extend backfill window from 5 years (`days_back=1825`) to 10 years (`days_back=3650`) on all confirmed Group A tools
2. Wire additional tools to sparse entity types (identify which entity types are thin and which of the 51 tools observe them)
3. Targeted synthetic augmentation for entity types where no historical API exists (e.g., newly-created political entities, recently-formed companies)

Only after the audit passes does Phase 48 research begin.

---

## 7. What Phase 40 tells you before you write Phase 48

Phase 40 (GNN retrain on real history) is the diagnostic run that informs every Phase 48 architecture decision:

- **Data volume report:** how many observations per entity type after backfill? Determines transformer context window size and whether you need augmentation.
- **Failure mode map:** which layer fails first? DAG inference, GNN embedding quality, Kalman fusion noise floor, RL exploration coverage? The transformer architecture fixes specific things — know which things need fixing.
- **Attention starvation map:** which entity types have thin cross-domain edges? The transformer will have the same attention starvation problem as the GNN if those edges are not filled.
- **Scaling curve:** does performance improve monotonically with more data, or plateau? If it plateaus on the current architecture, the transformer will break through it. If it doesn't plateau, the bottleneck is not model capacity.

**The transformer is not a leap of faith. It is a conclusion that Phase 40 will either confirm or modify.**

---

## 8. Final phase sequence — locked

```
Phase 46   Phase 47              Phase 40              Phase 48
─────────  ──────────────────    ──────────────────    ──────────────────────
EWC        Backfill 51 tools     GNN retrain on        Transformer world model
online     5 years history       real years of         + Dreamer model-based RL
learning   + density audit       data + diagnostic     (GATED: density + ceiling
(CPU, $0)  + patch sparse        walk-forward          confirmed by Phase 40)
<1 week    entities              report failure        Target production arch.
           1–2 days              modes + ceiling
```

No phase starts until its predecessor's exit conditions are met. Density is not negotiable.

---

## 9. Files changed this session

| File | What changed |
|---|---|
| `[[quant_training_ground]]` | Sequence header: density audit added to Phase 47, Phase 48 gated on density + ceiling. Density mandate standing rule added. Phase 47 entry: density audit as mandatory exit step. Phase 48 entry: three hard prerequisites spelled out. Model agnosticism doctrine restored (had been undone by user). |
| `[[chat_checkpoint_2026-04-22_final_session]]` | Full cost breakdown + complete system state checkpoint (written earlier this session) |
| `[[chat_checkpoint_2026-04-22_strategy_transformer]]` | This file |

---

## 10. What the next session opens with

1. **Read this checkpoint + `[[quant_training_ground]]`** — that is sufficient context to start cold.
2. **Phase 46 preflight** — create `[[living_system_online_gnn]]` first. No code until that exists.
3. **Key constraint to remember:** Phase 46 (EWC) is a leaf node implementation — small blast radius, independent of transformer plans. It makes the GNN living regardless of what architecture comes later. It is not wasted work even if Phase 48 changes everything.
4. **`time_decay_lambda` parameter** — must be wired into Phase 47 spec and Phase 40 training code. Exponential decay on historical observations, $\lambda = 0.001$/day default, configurable.
5. **`scripts/density_audit.py`** — must be built as part of Phase 47 exit, not as an afterthought. Report: observations per entity type (count, min, max, mean, p10, p90), flag types below 100, recommendation per flagged type.

---

## 11. The one-paragraph summary

TirraMind is a predictive AI company building a real-world prediction engine that watches physical reality, human decisions, and information flows across every country simultaneously. The current system (51 tools, 29-node DAG, HetTGN GNN, Bayesian world model, SAC RL, Thompson bandit) is architecturally complete but data-starved. The four-phase plan to reach the target production architecture is: Phase 46 adds online learning so the GNN evolves continuously; Phase 47 floods the DB with 5 years of real history across all 51 tools and audits density per entity type; Phase 40 retrains the GNN on real data and produces the diagnostic report that tells us exactly where the current stack hits its ceiling; Phase 48 replaces the Bayesian DAG with a transformer world model and the SAC MLP with a Dreamer model-based RL agent — but only after density is proven and the ceiling is confirmed. Total cost from today to live production: under $55. Total cost for 100 users: $41/month. The transformer is not a dream — it is the architecture that this data density and this entity graph scale will naturally support, once Phase 47 fills it in.

## Related

- [[quant_training_ground]]
- [[historical_backfill]]
- [[chat_checkpoint_2026-04-22_final_session]]
- [[transformer_world_model]]
- [[living_system_online_gnn]]
