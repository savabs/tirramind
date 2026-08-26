---
title: "Research: Scheduler tier's 'custom DAG submission' claim has zero backing code"
tags:
  - doc/research
  - topic/product
  - topic/api
  - status/active
date: 2026-08-27
---

# Research: Scheduler tier's "custom DAG submission" claim has zero backing code

> Filed by api-backend-engineer while reviewing `agent/brief_server.py`'s
> Scheduler tier surface. No implementation code for the flagged claim itself
> (see §3 for what — deliberately — was not built and why). Per
> `RulesForAI.md` §3: facts + design space, decision made, handoff specified.

## 1. What's real vs. what's claimed

`products/brief_subscription/pricing.html`'s Scheduler tier ($50/mo) markets
two distinct things:

1. "Read-only visibility into the DAG-based execution engine that runs our
   own pipeline in production — run history, status, and timing per DAG,"
   with `GET /api/v1/dag/runs?dag_name=daily_collection&limit=20`.
   **This is real.** `agent/brief_server.py`'s `_serve_dag_runs` (gated by
   `_SCHEDULER_TIERS`) reads `PipelineStore().get_runs()` against the actual
   `dag_runs` table the production pipeline writes to. Verified live:
   403 unauthorized / 200 with a valid Scheduler-tier key, correct data
   returned.
2. A separate line, tagged `BETA`: "Submitting your own custom DAGs for
   execution is in closed beta — contact us if you want in."
   **This is not real, at any level.** Confirmed this session:
   - No route anywhere (`brief_server.py` has no `POST /api/v1/dag/*` of
     any kind — the only DAG route is the read-only `GET /api/v1/dag/runs`
     above).
   - No DAG-definition parser or schema anywhere in the codebase (grepped
     for `submit`, `custom_dag`, `DAGParser`, `dag_submit` across `agent/` —
     nothing relevant).
   - No manual/human-in-the-loop process either — there's no internal
     tool, script, or admin path an operator could use to hand-run a
     customer-submitted DAG on request. "Contact us" would currently reach
     a mailbox with nothing behind it to fulfill the request.

This is a different failure mode than the Entity Graph mismatch
(`entity_graph_tier_mismatch.md`): that tier at least served *something*
real (a smaller/different dataset than advertised). This line describes a
capability that does not exist in any form — not a scoped-down version, not
a manual fallback.

## 2. Why I'm not building it, even narrowly

The instruction that produced this doc floated a narrower version worth
considering: "submit a JSON filter/parameter set against one of the existing
10 fixed DAGs, not an arbitrary custom DAG" (i.e., trigger/parameterize
execution of a known DAG, rather than accept an arbitrary customer-authored
DAG definition). I looked at this seriously rather than defaulting to "too
big" — here's why it's still not a same-pass change, even in that narrower
form:

- **The DAGs are not cheap, isolated, or fast.** `agent/pipeline/dags/*`
  (10 files: `daily_collection`, `whale_tracking`, `entity_scoring`,
  `feature_generation`, `gnn_inference`, `inference`, `rl_training`,
  `convergence_detection`, `adversarial_scan`, `world_model_update`) run via
  `DAGExecutor` (`agent/pipeline/executor.py`): topo-sort into layers, run
  each layer in a `ThreadPoolExecutor`, with retries. `daily_collection`
  alone fans out to CFTC, FINRA, NYISO (x2), GDELT, Polymarket, CT-log, and
  DNS lookups — external network calls, not instant, against **free-tier
  APIs the production pipeline itself depends on and could rate-limit
  itself out of** if a paying customer could trigger extra runs on demand.
- **`brief_server.py` is a single `ThreadingHTTPServer`.** My own agent
  doc's own scrutiny checklist (`api-backend-engineer.md` §4, "rate
  limiting") already flags that nothing today caps request *rate* — "one
  customer can saturate the box." Adding an endpoint whose entire job is to
  kick off a multi-minute, multi-network-call job multiplies that risk
  category rather than fitting within it. A blocking synchronous trigger
  ties up a request thread for the DAG's full runtime; a fire-and-forget
  async trigger needs a queue, dedup, and per-customer concurrency limits
  that don't exist anywhere in this codebase today.
- **It writes to the same `pipeline.db` the scheduled production runs use**
  (SQLite, `agent/pipeline/store.py`) — my own boundaries doc already names
  "SQLite locking is a real risk" under concurrent writes from the existing
  surface; customer-triggered writes competing with the cron-scheduled
  production DAG runs is a new instance of exactly that risk, not a smaller
  one.
- **This crosses out of my remit as scoped in
  `.claude/agents/api-backend-engineer.md`.** Rate limiting/concurrency
  under load is explicitly infra-operator's territory; a paying customer
  gaining a new way to make the production system perform live network
  I/O and shared-resource writes on their command is exactly the kind of
  "adversarial bypass / abuse surface" `security-auditor` owns review of.
  Shipping it unilaterally inside `brief_server.py` would be deciding both
  of those calls silently, which is the same mistake the original Entity
  Graph mismatch made (a feature shipped past the people who own its
  blast radius).

None of this rules out building it *ever* — it means it needs sign-off and
design from infra-operator (concurrency/isolation/rate-limiting) and
security-auditor (abuse surface) before it's safe to scope as "small," and
that didn't fit this pass.

## 3. Decision

**Recommendation to product-strategist (copy is your call, not mine):**
remove or replace the "Submitting your own custom DAGs for execution is in
closed beta" line. Unlike a feature that's merely incomplete, there is
*nothing* behind it — not even a manual process a "contact us" click could
route to. Two honest options, in descending order of how quickly this can
be corrected:

- **(a) Delete the line entirely.** The tier's primary claim (read-only run
  history) is real and fully stands on its own; the beta line is not load
  bearing for the $50/mo price and its removal doesn't gut the offer.
- **(b) Replace it with a true statement** if there's an actual product
  intent to build this — e.g., "Want to trigger one of our existing DAGs
  with custom parameters? Contact us — this is unbuilt today, we're
  scoping it" — but only if that's genuinely true; don't reword the same
  unbacked promise into softer language that still implies more exists
  than does.

**If a follow-up spec is wanted for the narrow version** (parameterized
trigger of one of the 10 fixed DAGs — not arbitrary custom DAGs), it needs
to decide, before any code is written:

1. **Concurrency model**: synchronous (blocks a request thread for the
   DAG's full runtime — unacceptable for `daily_collection`-scale DAGs) vs.
   asynchronous (needs a job queue, a way to poll status, and a decision on
   whether `dag_runs` gets a `trigger=customer_api` value distinct from
   `manual`/`scheduled`).
2. **Rate limits**: per-key concurrent-run cap and cooldown, informed by
   the free-tier API rate limits each DAG's tools are already subject to —
   this needs infra-operator input on what headroom actually exists.
3. **Which DAGs, if any, are even customer-safe to expose.** Read-only
   `whale_tracking`/`daily_collection` triggers are a different risk
   profile than `rl_training` or `world_model_update`, which mutate model
   state the whole system depends on — a customer-triggered run of those
   is a different class of problem than "give me fresher data."
4. **Abuse surface sign-off from security-auditor** — a customer-controlled
   trigger onto a system that performs outbound network calls and shared
   writes is new attack surface (resource exhaustion, at minimum) that
   didn't exist when every DAG trigger came from the internal scheduler.
5. **Parameter surface**: even "JSON filter/parameter set" needs a defined,
   validated schema per DAG (not "pass through whatever JSON you like" —
   that reintroduces the "arbitrary custom DAG" risk this option was
   supposed to avoid) — whoever owns the eventual spec decides this
   per-DAG, not as one generic passthrough.

## Related
- [[entity_graph_tier_mismatch]]
