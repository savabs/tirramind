---
title: TirraMind
tags:
  - layer/adversarial
  - layer/feature-engineering
  - layer/fusion
  - layer/learning
  - layer/llm-support
  - layer/surveillance
  - layer/world-model
---

# TirraMind

**Technologically advanced information-arbitrage system.** Finds unique, cheap/free data sources that leak predictive signal and applies SOTA math + CS to extract asymmetric edge.

> Math on common data is commoditized. Math on unique data is the moat.

---

## Architecture

```
Layer 1: Surveillance Surface   → agent/tools/      (60+ data tools, free APIs)
Layer 2: Feature Engineering     → agent/quant/      (BOCPD, HMM, spectral, signals)
Layer 3: World Model             → agent/models/     (Bayesian network, causal graph)
Layer 4: Signal Fusion           → agent/fusion/     (Kalman, particle filter)
Layer 5: RL Policy               → agent/learning/   (model-based RL, portfolio opt)
Layer 6: Adversarial             → agent/adversarial/ (manipulation, edge decay)
Layer 7: LLM Support             → agent/reasoning/  (text parsing, narration only)

Pipeline Layer (deterministic)   → agent/pipeline/   (DAG scheduler, no LLM)
Knowledge Wiki (compiled docs)   → wiki/             (persistent synthesized markdown)
```

Two execution engines run in parallel:
- **Agent Layer** (LLM-driven): orchestrator, exploration, research, hypothesis generation
- **Pipeline Layer** (deterministic): DAG scheduler, fetch → feature → model → signal, scheduled triggers

The LLM explains what the math decided. It doesn't decide.

---

## Quick Start

```bash
# Clone
git clone <repo-url> && cd tirramind_v1

# Install (editable, with dev + quant extras)
make setup          # or: pip install -e ".[dev,quant]"

# Run tests
make test           # all tests
make test-fast      # skip slow/live/integration tests

# Lint
make lint           # ruff check
make format         # ruff format

# Full CI check
make check          # lint + test
```

---

## Run the Intelligence Brief (one command)

> **You can run this from any folder — it always uses the repo root.** No setup
> beyond the install above.

```bash
# Refresh live data + build + deliver the brief, then serve it over HTTP:
tirra-engine

# Same thing, but just deliver once and exit (no server):
tirra-engine --once

# Serve an already-delivered brief over HTTP:
tirra-serve

# Record a realized bid outcome so P(win) personalizes per agency:
tirra-engine --record-bid "Department of Veterans Affairs" 60000 1
```

- Brief URL (while serving): `http://127.0.0.1:8787/brief.json`
- Outputs land in `.tirra_delivery/` (JSON + Markdown + log) at the repo root
- Scheduled runs: `scripts/run_scheduled.sh serve` (brief refresh + deliver + serve),
  `scripts/run_scheduled.sh collect` (full 40+-source `daily_collection` DAG —
  slow, run once/day; see `deploy/systemd/` for production timer units)

---

## Project Structure

```
agent/
├── cli.py              # Entry point
├── config/             # Env-var config (TIRRA_* prefix)
├── core/               # Orchestrator pipeline
├── data/               # DataCache, DNS bypass
├── learning/           # RL, bandit arms
├── memory/             # Episodic + semantic + working memory
├── pipeline/           # DAG scheduler, SQLite store, operators
├── planner/            # Hierarchical task decomposition
├── quant/              # BOCPD, HMM, spectral, scoring
├── reasoning/          # LLM client (support role only)
└── tools/              # 60+ data tools (Layer 1)
tests/                  # Extensive edge-case and integration coverage
docs/
├── research/           # Research notes (one per feature)
├── specs/              # Implementation specs
├── memory/             # Checkpoints + project memory
└── adr/                # Architecture Decision Records
wiki/
├── pages/              # LLM-maintained synthesized knowledge pages
├── raw/                # Immutable future source ingests
├── index.md            # Generated wiki catalog
├── log.md              # Append-only wiki activity log
└── SCHEMA.md           # Wiki maintenance rules
tasks/
├── active/             # Current work
└── done/               # Completed tasks
```

---

## Development Workflow

This project follows a strict phased pipeline. No shortcuts.

```
Research → Specification → Implementation (one atomic step at a time)
```

1. **Research** — write `docs/research/<name>.md`. Read code, search OSS, no code changes.
2. **Spec** — write `docs/specs/<name>_spec.md`. Transform research into ordered atomic steps.
3. **Task** — create `tasks/active/<name>.md` with numbered, verifiable steps.
4. **Implement** — one step at a time. Each step: change → test → mark done → next.
5. **Review** — correctness, security, layer discipline, test coverage.
6. **Checkpoint** — write `docs/memory/chat_checkpoint_<date>.md`.

The repo also maintains a compiled wiki layer under `wiki/` for topic-centric synthesis across raw workflow artifacts. Rebuild and lint it with `tirra-wiki-catalog`.

Every step must be independently verifiable. If you can't describe a one-line test for it, break it further.

Use the latest checkpoint in `docs/memory/` plus `[[quant_training_ground]]` as the current architectural entry point.

---

## Make Targets

| Target | Description |
|--------|-------------|
| `make setup` | Full dev setup (install + pre-commit hooks) |
| `make test` | Run all tests |
| `make test-fast` | Run tests excluding slow/live markers |
| `make lint` | Run ruff linter |
| `make format` | Run ruff formatter |
| `make check` | Lint + test (CI equivalent) |
| `make clean` | Remove cache artifacts |
| `make help` | Show all targets |

---

## Cost Discipline

**$0 until proven edge.**

- Data: free public APIs (SEC EDGAR, GDELT, CFTC, Polymarket, ClinicalTrials.gov, OpenSky, NYISO, NASA FIRMS, …)
- Math: open source (numpy, scipy, statsmodels, hmmlearn, filterpy, pgmpy, cvxpy)
- LLM: Groq free tier or Ollama local
- Compute: local-first

Only spend money when the system has demonstrated alpha on backtests with statistical significance.
