---
name: awos-kernel
description: Guidelines for making changes to the AWOS kernel (scaffold/agent/)
license: MIT
---

# AWOS Kernel Change Protocol

## Kernel architecture (scaffold/agent/)
- `orchestrator.py` — main execution loop (Plan -> Execute -> Verify -> Learn)
- `planner.py` / `worker.py` / `verifier.py` — coding app pipeline
- `reward_store.py` — quality x speed / cost objective
- `ml_router.py` — LinUCB bandit for model selection
- `escalation_engine.py` — 5-tier model ladder
- `memory/vector_memory.py` — ChromaDB-backed semantic memory
- `memory/codebase_index.py` — semantic codebase indexing

## Kernel invariants (do NOT break)
1. The orchestrator loop must always be: Plan -> Execute -> Verify -> Learn
2. The reward function must always be: quality x speed / cost
3. Budget ledger must enforce hard caps — no bypass
4. Self-learning must be opt-in via `.awos/` — never auto-modify without consent
5. Model routing must fall back gracefully — never crash on provider failure
6. The CLI (`awos.py`) must remain the single entry point

## Before touching the kernel
1. Load skill `awos-preflight`
2. Document expected behavior change
3. Identify tests that verify the invariant
4. Run `pytest tests/ -v --tb=short -m "not integration"` before starting
5. Follow `awos-live-proof` for any orchestrator/CLI/safety changes
