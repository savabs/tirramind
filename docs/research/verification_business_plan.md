---
title: "Plan: sell verification, not discovery"
tags:
  - doc/research
  - topic/strategy
  - status/current
date: 2026-09-26
---

# Plan: sell verification, not discovery

## 1. The correction to the earlier scan

`ultraresearch_venture_scan_2026-09-06.md` evaluated **pattern discovery** as a
product and correctly found it weak. It never evaluated **pattern
verification** — that framing did not exist yet. Its *measurements* still hold
(3 genuine cross-domain joins of 561 source pairs; a 5-month data moat). Its
*conclusion* does not cover this, so it is not evidence against it.

The distinction the scan was missing:

| | discovery | verification |
|---|---|---|
| Is it scarce? | No — everyone claims it | **Yes** |
| Provable to a buyer before they pay? | Hard | **Yes, on their own pattern** |
| Needs a data moat? | Yes | **No** — the rigour is the product |
| Do we have it working today? | No | **Demonstrably yes** |

Anyone can find patterns; a computer finds infinite patterns. Almost nobody can
tell you which ones survive out of sample, and almost nobody *wants* to, because
the honest answer is usually "it doesn't".

## 2. What we already have

**`scripts/cftc_event_study.py`** — the engine, currently welded to CFTC:
causal (backward-only) z-scores, `block_bootstrap_ci` for serially-correlated
series, Benjamini–Hochberg correction, publication-lag handling (Tuesday as-of /
Friday release), pseudo-replication quantified.

**`docs/publications/cot_null_result.md`** (34 KB) — the credibility asset, and
it is stronger *because* it is negative: **0 of 51 hypotheses survived BH at
α=0.05**, method deliberately hostile to a positive result, and it states its own
weaknesses (pooled two-sided |z| cancels a signed effect; ~10% power). It then
replicates the finding against Sanders, Irwin & Merrin (2009) — 0 of 30 survive
there too.

That document is the sales pitch. It demonstrates the one thing the market
cannot buy elsewhere: someone who will kill a hypothesis properly, including
their own.

**What does not exist:** a generalised verification engine. The maths is real but
it only speaks CFTC.

## 3. The offer

> **"You have a pattern. We tell you whether it's real."**

A verification report on a stated hypothesis. Input: a claim of the form *when X
happens, Y moves, within N days*. Output: a verdict with

- forward-return event study on the claimed relationship
- **multiple-comparison correction** — the step that kills most claims and that
  almost no vendor applies
- block-bootstrap confidence intervals (serial correlation is the default in
  finance, not the exception)
- **explicit power analysis** — "no detectable effect at low power" is not "no
  effect", and conflating them is the most common error in this space
- publication-lag and point-in-time discipline, so the test could have been run
  at the time
- pseudo-replication count: 123 events across 116 distinct weeks is not 123
  independent observations

Fixed price, fixed turnaround, and the verdict is whatever it is. **We do not
sell positive results.** That is the product: a report a fund can show its risk
committee precisely because we had no stake in the answer.

## 4. Who buys, in priority order

**A. Alt-data vendors, proving signal to funds.** Strongest candidate. Every
vendor selling a dataset to a quant fund must show it carries signal, and they
cannot credibly mark their own homework. Independent verification is a budgeted
line item, recurring per dataset, and the buyer is easy to find (they all
publish "request a trial"). Their incentive is to look good, which is a tension
to price in — but a vendor confident enough to commission an *independent* test
gains more than one who self-reports.

**B. Funds and desks validating in-house strategies.** A researcher has a
hypothesis and an incentive to believe it. An outside test with pre-registered
methodology is cheap insurance against a bad allocation. Smaller shops without
a dedicated validation function are the wedge.

**C. Signal and newsletter sellers wanting credibility.** Lower budgets, faster
decisions, and a useful source of early volume and public track record.

## 5. The wedge: verify someone else's public claim, unasked

Before selling anything, build the track record in public.

Take a **publicly-stated claim** about a dataset — a vendor's marketing, a
paper, a widely-repeated market belief — and verify it rigorously with the
engine. Publish the result either way.

Why this is the right first move:
- **The marketing IS the product demo.** Nobody has to trust a description.
- It costs compute and nothing else.
- It creates a public artifact that gets discovered, which suits a low-outreach
  posture and does not depend on it.
- Each verification teaches us which *kinds* of pattern survive — which is how
  we eventually earn the right to do discovery, with evidence instead of a
  hypothesis.
- A negative result is not a failure here; it is the strongest possible
  advertisement for the service.

The CFTC null is already this, accidentally. It should be the first entry in a
series, not a one-off tombstone.

## 6. Four weeks

**Week 1 — generalise the engine.** Lift the CFTC specifics out of
`cftc_event_study.py` into a reusable module: given an event series and a return
series, produce the full verdict (BH, block bootstrap, power, pseudo-replication,
point-in-time check). Deliverable: the CFTC study reproduced *through the
generalised path*, matching the published numbers exactly. If it does not
reproduce, the generalisation is wrong.

**Week 2 — verify one public claim end to end.** Pick a claim, run it, publish.
Deliverable: a second public verification report, and a repeatable template.

**Week 3 — turn the report into an offer.** A page describing the method, the
two published reports as evidence, a fixed price, a fixed turnaround, and a
stated refusal to sell positive results. Deliverable: something a stranger can
read and buy.

**Week 4 — first contact.** The offer goes to category A and B. `docs/studio/`
already holds three target lists. Deliverable: messages sent — the step that has
never once happened.

## 7. What would kill this, stated in advance

- **Nobody pays for bad news.** The central risk. If every buyer only wants
  validation, the honest product has no market. *Test:* by end of week 4, does
  anyone engage with a report whose headline is "this does not hold"?
- **The method is commoditised.** BH correction and block bootstrap are
  textbook. The moat is rigour and reputation, not technique — and reputation
  takes time we may not have.
- **It is a services business, not a product.** It scales with hours, not users.
  Acceptable as a first revenue source and a way to learn the market; it is not
  the end state.
- **It competes with free.** A quant shop can do this in-house. The buyer is
  specifically someone who cannot, or who needs the *independence*.

## 8. What this means for TirraMind

TirraMind stays **research**, with a stop condition. The collector keeps running
because time-gated data only accrues while collecting, and that remains the one
genuinely uncopyable asset. The graph thesis gets its honest test.

But it is no longer the revenue plan, and it should stop absorbing the attention
that the sellable thing needs. It is the more interesting problem, which is
exactly why it has crowded out the boring one.

## Related

- [[cot_null_result]] — the credibility asset; a null result done properly
- [[cftc_forward_return_event_study]] — the engine, currently CFTC-specific
- [[ultraresearch_venture_scan_2026-09-06]] — evaluated discovery, not verification
- [[what_tirramind_is_for]] — why the graph premise stayed unproven
