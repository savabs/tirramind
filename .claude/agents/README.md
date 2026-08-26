# TirraMind agent team

21 specialists across three teams. Each encodes failure modes TirraMind has
**actually produced**, so a fresh session inherits hard-won knowledge instead of
rediscovering it.

---

## Org structure

```
                        OWNER
                          │
                          ▼
              ┌───────────────────────┐
              │  principal-architect  │   ← your direct contact
              └───────────┬───────────┘
                          │  consults
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
 quant-researcher  applied-mathematician  systems-architect
   (what edge)        (which method)      (what shape)
                          │
                          │  directs
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
  PRODUCT / ML TEAM                  REVENUE TEAM
  (8 specialists)                    (7 specialists)
```

**Talk to `principal-architect`** — or just run `/team <what you want>`.

It decomposes intent, verifies ground truth before planning on it, and routes
work to exactly one owner. The other two teams execute.

### "Ask the team" never means "run every agent"

Dispatch is **triaged, not fanned out**. Every team request must produce:

```
REQUEST:    <the ask in one line>
DISPATCH:   <agent> — <why>
EXCLUDED:   <agents not called> — <why not>
SEQUENCE:   parallel | <a> then <b> (because <dependency>)
```

- **Default 1–3 agents.** 4–6 needs a stated reason. **7+ needs your approval.**
- The **EXCLUDED** line is mandatory — naming who was *not* called is what
  forces real triage instead of reflexive fan-out.
- Dispatching **nobody** is a valid outcome. If a `grep` settles it, do that.

A full-roster audit once cost 1.6M tokens and 40 minutes to answer questions
three agents owned. Breadth is not thoroughness. Enforced by CLAUDE.md §12.

**A busy specialist can be cloned — carefully.** A specialist is a role file,
not an employee; running a second instance is allowed when new work in its
domain is provably disjoint from what the busy instance is touching (state the
split in the triage block, don't just assert it). Cap: 2 total instances of one
role (original + at most one clone), counted from `journal.jsonl` ground truth,
never from `git status`. Only the top-level dispatcher decides to clone — no
specialist lists `Agent`/`Workflow` in its `tools:`, so none can spawn its own
clone. Full policy: CLAUDE.md §12.

**Every agent has an exclusive domain.** Each file carries a `## Boundaries —
you do NOT own` section naming which specialist owns adjacent territory. If two
agents could both answer a question, that is a bug in the roster — fix the
boundaries rather than letting them both work it.

---

## Model policy

Opus is reserved for open-ended judgment. Most of this work is *verification
against a known standard*, which Sonnet does well and far cheaper.

| model | count | share | criterion |
|---|---:|---:|---|
| **opus** | 6 | 28% | open-ended judgment: novel design, theory, strategy, adversarial creativity |
| **sonnet** | 13 | 62% | applying a defined standard, tracing code, checklist verification |
| **haiku** | 2 | 10% | mechanical, well-specified, bounded checks |

**Opus (6):** the entire Direction team (`principal-architect`,
`quant-researcher`, `applied-mathematician`, `systems-architect`) — that team
exists *because* its work is open-ended — plus `product-strategist` (commercial
judgment) and `security-auditor` (attacking a composed system needs creativity,
not a checklist).

**Headcount ≠ usage.** 28% of agents are opus, but the Direction team is
consulted occasionally for strategy while execution agents run constantly. Real
opus *invocations* should land well under 20%.

**Why the rest are not opus:** `schema-sentinel` compares numbers against a
spec. `payments-auditor` and `training-engineer` work through defined
checklists. `silent-failure-hunter` matches four documented patterns.
`code-reviewer` is scoped to a diff plus routing. `layer-architect` applies the
written 7-layer standard. `quant-evaluator` executes a measurement designed by
`applied-mathematician`. All demanding but *bounded* — the standard is written
down in the agent file, which is exactly what Sonnet applies well.

**Haiku (2):** `test-integrity-auditor` (run named test files, sort failures
into four buckets) and `frontend-engineer` (check links, verify no `REPLACE_`
placeholders survive, check markup). Both carry an explicit instruction to
escalate rather than guess when a call turns subjective.

Raising a model is cheap if an agent underperforms — start low and promote on
evidence, not on the assumption that harder-sounding work needs a bigger model.

---

## Team 0 — Direction

Decides *what* to build and *why it should work*. Your interface to everything
else. Consulted before engineering effort is spent, not after.

| agent | owns exclusively |
|---|---|
| `principal-architect` | **your direct contact** — intent → plan, technical roadmap, routing work to a single owner |
| `quant-researcher` | finance/market theory: what edge could exist, mechanism, horizon, capacity, decay |
| `applied-mathematician` | method selection, assumptions, statistical validity, multiple-testing control |
| `systems-architect` | system-level design: data flow, storage, scale limits, what the architecture should become |

## Team 1 — Product / ML

The thing being built.

| agent | owns exclusively |
|---|---|
| `schema-sentinel` | registry ↔ DB ↔ checkpoint consistency, feature dimensions |
| `training-engineer` | the training run, losses, checkpoint artifacts, LESSONS.md gates |
| `quant-evaluator` | measurement: IC, decile spread, backtests, leakage audits |
| `pipeline-engineer` | `agent/pipeline/`: DAG structure, node config, timeouts, executor, chain runner |
| `market-data-engineer` | L1 financial/macro sources (~13k LOC, 22 tools) |
| `physical-data-engineer` | L1 physical/geospatial/network sources (~12k LOC, 15 tools) |
| `public-record-engineer` | L1 government/legal/health/social sources (~14.5k LOC, 19 tools) |
| `layer-architect` | module placement, import direction, boundary violations, orphaned code |
| `silent-failure-hunter` | swallowed errors / success-without-effect in **production** code |
| `test-integrity-auditor` | whether **tests** prove what they claim; test isolation |

## Team 2 — Production / Revenue

The business around it. Nothing to do with the ML product; everything to do with
whether a customer can pay and receive what they bought.

| agent | owns exclusively |
|---|---|
| `payments-auditor` | Paddle integration, webhook correctness, subscription state, refunds |
| `customer-lifecycle` | the **seams**: journey end-to-end, credential delivery, onboarding, support burden |
| `api-backend-engineer` | `brief_server.py` HTTP contract: routes, gating, status codes, rate limits |
| `frontend-engineer` | the storefront: markup, checkout UX, accessibility, links |
| `trust-and-compliance` | legal pages, MoR obligations, data-handling claims, redistribution licensing |
| `security-auditor` | adversarial review **across** component boundaries |
| `infra-operator` | `deploy/`, systemd, TLS, DNS, backups, secrets placement, uptime |

## Cross-team

| agent | owns exclusively |
|---|---|
| `product-strategist` | positioning, tier design, pricing amounts, truth of capability claims |
| `code-reviewer` | a specific **diff** + the **commit plan**; routes domain questions out |

---

## The boundary lines that were hardest to draw

These pairs look like overlaps and are not:

| pair | the split |
|---|---|
| `quant-researcher` / `quant-evaluator` | *generates* the hypothesis **vs** *tests* it. Nobody grades their own idea. |
| `applied-mathematician` / `quant-evaluator` | designs the measurement **vs** executes it and reports the number |
| `systems-architect` / `layer-architect` | what the architecture **should become** vs policing what it **is** |
| `systems-architect` / `infra-operator` | chooses the storage engine **vs** operates it |
| `principal-architect` / `product-strategist` | *technical* roadmap **vs** what we sell and for how much |
| the three L1 data engineers / `pipeline-engineer` | tool internals + required params **vs** node config, timeouts, dependencies |
| `market-` / `physical-` / `public-record-` engineers | disjoint file sets in `agent/tools/` — split by source domain, never overlapping |
| `public-record-engineer` / `trust-and-compliance` | making the fetch work **vs** whether we may legally resell it |
| `training-engineer` / `quant-evaluator` | is the run *valid* **vs** is the result *useful*. A run can pass every gate and have zero edge. |
| `silent-failure-hunter` / `test-integrity-auditor` | both chase "green but broken" — split by file tree: `agent/` **vs** `tests/` |
| `payments-auditor` / `customer-lifecycle` | the key is *minted correctly* **vs** the key *reaches a human* |
| `api-backend-engineer` / `infra-operator` | the HTTP **contract** vs **running** the process |
| `product-strategist` / `trust-and-compliance` | is the capability claim *true* **vs** is the legal promise *keepable* |
| `security-auditor` / everyone | each owner verifies their part is *correct*; security asks whether the *composition* can be beaten |
| `code-reviewer` / every specialist | reviewer owns the diff and commit plan, and **names** the specialist rather than adjudicating their domain |

---

## Suggested combinations

**Any open-ended question** → start at `principal-architect`. It verifies ground
truth, then routes.

**Deciding what to build next**
`quant-researcher` (is there a plausible mechanism?) →
`applied-mathematician` (can it be measured validly with the data we have?) →
`systems-architect` (what does it cost structurally?) →
`principal-architect` (sequence it, or kill it)

**Before a retrain**
`schema-sentinel` (dimension spec) → `training-engineer` (run with gates) →
`quant-evaluator` (measure, assume leakage first)

**Before going live with payments**
`payments-auditor` (money correctness) → `customer-lifecycle` (can they actually
use it?) → `trust-and-compliance` (are the promises keepable?) →
`security-auditor` (can it be beaten?)

**Adding a data source**
`layer-architect` (placement) → the L1 engineer for that domain (implement) →
`pipeline-engineer` (wire the node) → `silent-failure-hunter` (did rows land?)
→ `trust-and-compliance` if the licence permits redistribution at all

**Before committing**
`code-reviewer` → whichever specialist it routes you to

---

## Why these exist

Read `LESSONS.md` PART 1 and
`docs/research/intelligence_layer_reactivation.md`. Every agent traces to a
specific incident where the codebase looked healthy and was not:

> Green status, passing tests, and clean logs are **not** evidence.
> Row counts, reproductions, and measured numbers are.

The revenue team exists for the same reason. As of 2026-08-26 a paying customer
would be charged and **never receive their API key** — it is minted, the webhook
ack strips it, the email is captured, and nothing sends it. Every individual
component passes its own review. Only someone walking the whole journey finds it.

**Note:** agent definitions load at session start. Newly added agents need a
Claude Code restart before `subagent_type` will resolve them.
