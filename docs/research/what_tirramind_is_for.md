---
title: "What TirraMind is for"
tags:
  - doc/research
  - topic/strategy
  - status/current
date: 2026-09-23
status: framing — supersedes the positioning in README.md §"Read this first"
---

# What TirraMind is for

Written 2026-09-23, after the audit that found the model had never been a graph
network. Every number below was measured against the live database on that day,
not quoted from an earlier write-up. Where a previously published figure has
changed, both are given.

This document exists because the project has been carrying three different
goals under one name, and the work kept stalling at the seam between them.

---

## 1. The three goals that were being conflated

| | goal | status |
|---|---|---|
| **A** | *Does a cross-source entity graph carry predictive signal?* | research question — **still unanswered** |
| **B** | *Sell the infrastructure — "shovels, not alpha"* | product thesis — **depends on A more than it appears to** |
| **C** | *Produce cash flow* | outcome — **downstream of both** |

B has been doing the work of avoiding A. The reasoning was: if we sell the
pipes, we never have to prove the digging works. That escape does not hold —
see §5 — but the instinct behind it is sound and survives in a narrower form.

---

## 2. What was actually measured today

**Scale.**

```
6,110 entities · 384,285 observations · 17,581 links
34 sources writing · 64 collectors written
```

Half of Layer 1 is code that runs and produces nothing: **64 collectors exist,
34 have ever written a row.**

Three sources are 89% of all observations — `defi_flows` (162k), `gdelt` (92k),
`instrument_universe` (90k). The "80 free sources" story is three sources and a
long tail.

**The live database still spans 1920 → 2030.** The write guard added this week
stops *new* out-of-range rows; it never cleaned the existing ones. `trainer.py`
derives its train/val/test boundaries from `MIN`/`MAX(observed_at)`, so those
rows still move the evaluation window. This is unfinished work, not a
historical note.

---

## 3. The cross-source premise: re-measured, and it still fails

The README calls this "the finding that ended the project": of 561 possible
source pairs, only 3 share an entity across genuine domain boundaries.

That was measured **before** the country-code merge. The merge was expected to
change it — the alpha-2/alpha-3 bug had fragmented 279 country entities into
namespaces that could not join, which is exactly the failure mode that would
suppress cross-source overlap. Connectivity went from 0/215 to 214/215.

**Re-run today, post-merge:**

| | before | after merge |
|---|---:|---:|
| possible source pairs | 561 | 561 |
| pairs sharing ≥1 entity | 43 | **52** |
| same-namespace only | 40 | **49** |
| **genuine cross-domain** | **3** | **3** |

The merge added 9 joined pairs, all of them country↔country — a source joined
to another view of the same ISO code. **Genuine cross-domain joins did not
move: still 3 pairs, 4 entities** (`cert_transparency`↔`dns_monitor`,
`form144`↔`insider_filings`, `drug_regulatory`↔`gov_contracts`).

**Conclusion: the original premise is dead, and fixing the graph apparatus did
not revive it.** This is worth stating flatly because it was the most plausible
remaining hope — that cross-source synthesis had been suppressed by a bug
rather than absent from the data. It was not. Every identifier namespace is an
island because the *sources* do not share identifiers, and no amount of
engineering creates a join that the data does not contain.

---

## 4. What the graph actually is

17,581 links, grouped by what they connect:

| subgraph | links | share | sources |
|---|---:|---:|---|
| GDELT country↔country events | 9,487 | 54% | 1 |
| whale_alert wallet↔wallet / wallet↔instrument | 3,515 | 20% | 1 |
| topic↔instrument | 2,752 | 16% | 2 (1,657 from a repair script) |
| **person↔company** | **1,139** | **6%** | **2 — genuinely multi-source** |
| everything else (company/instrument↔country, etc.) | 688 | 4% | various |

So the "graph of links between things happening in the world" is, measured
honestly, **four single-source subgraphs that barely touch each other.** The
genuinely multi-source relation is 6% of edges.

This does not make the graph worthless. It makes it a *different object* than
the one the project was named for.

---

## 5. Why "we sell shovels" doesn't dodge the question

The framing is sound in general — infrastructure is a real business and you do
not need predictive edge to sell pipes. It fails here for two specific reasons.

**A shovel is worth money only if digging works.** If nobody can find signal in
this graph, a subscription to it is selling access to an untested hypothesis.
The published null already sits in the README; a paying customer will read it.

**The no-outreach constraint removes the usual escape.** The standing rule on
this project is no selling, no outreach, no discovery interviews, ever. A
product nobody is told about has to be *discovered*, and the only thing that
gets discovered is a public artifact so obviously useful that people arrive on
their own. That raises the bar on the artifact rather than lowering it.

What survives is the narrower version: **the collector is already a shovel that
works.** 34 sources, 27 needing no credential, normalised into one queryable
table. That is a real object with no predictive claim attached, and it is
honest to ship as-is. It is just not a subscription business on its own.

---

## 6. The one hypothesis that is still alive and testable

Strip away what was disproved and this is what remains:

> **Do GDELT geopolitical events on a country predict forward returns on the
> instruments linked to that country?**

It is a real hypothesis, it is much narrower than "cross-source synthesis", and
for the first time it is actually testable — the four defects that made every
prior run meaningless are fixed, so a retrain would genuinely exercise the HGT
layers and memory rather than a per-node MLP.

**But measure the surface first, because it is small:**

```
countries in the GDELT event graph        : 215
instruments with a country link           : 337
instruments with a country link AND prices:  86   ← the entire tradeable surface
instrument_daily rows                     : 77,014  (~865 days × 89 names)
```

**86 instruments.** That is the whole cross-section. The CSRC loss sorts into
deciles — that is 8 or 9 names per decile. Any IC computed on this will have
wide error bars, and a single sector concentration could produce the whole
result.

This number should be checked against what a decile-spread test needs *before*
the retrain, not after. If 86 is too thin to support the test, the honest move
is to widen the instrument universe first — which is a data problem with a
known fix, not another modelling cycle.

---

## 7. What we are expecting — stated as outcomes, not hopes

**Most likely (my estimate: ~60%).** The retrain runs, the graph layers finally
receive edges and events, and the result is indistinguishable from the per-node
baseline. The graph adds nothing on this data. *This is a good outcome* — it
retires a three-year hypothesis for the cost of a weekend, and it is the first
version of that conclusion that would actually be evidence.

**Possible (~30%).** A weak effect appears on the 86-name cross-section, too
small and too wide-errored to trade, but enough to justify widening the
universe and re-testing. The right response is more instruments, not more
model.

**Unlikely (~10%).** A robust effect survives the same hostile methodology the
CFTC study used — BH correction, block bootstrap, publication-lag honoured. If
this happens, B becomes a business with evidence behind it.

**Note the asymmetry:** two of the three outcomes end with the graph thesis
retired or narrowed. That is not pessimism, it is what the measurements in §3
and §4 imply. Framing the project around a 10% branch is how the last two
projects died.

---

## 8. What success means, and the stop condition

**Success for this project is a defensible answer, not a positive one.**

The single most valuable thing TirraMind can produce is a properly-powered,
honestly-reported test of whether alternative-data entity graphs predict
returns — including, and especially, if the answer is no. The existing 0/51
null is not that: it covered COT (1.5% of the corpus) at roughly 10% power,
against a model that was not using its graph layers. It is a null result about
a broken instrument.

**The stop condition must be written before the run, not after.** The specific
number — what IC, what decile spread, what survival rate under BH correction
would make us walk away — is an owner decision and is deliberately left blank
here. It is the one discipline that would have saved the previous two projects,
and it is worthless if chosen once the result is visible.

Until that number exists, the retrain is theatre.

---

## 9. What this means for the immediate work

In order:

1. **Set the stop condition.** Blocking, owner-only. Everything else is
   premature without it.
2. **Clean the 1920/2030 rows** from the live DB. The guard stops new ones; the
   old ones still move every split boundary. Backup first.
3. **Decide whether 86 instruments can carry the test.** If not, widen the
   universe before modelling.
4. **Then retrain from scratch** — not `--resume`; the country merge changed
   the node id space, so old weights are meaningless.
5. **Publish the result either way**, with the same hostility to a positive
   finding that the CFTC study used.

Related: [`full_audit_2026-09-23.md`](full_audit_2026-09-23.md),
[`graph_connectivity_failure.md`](graph_connectivity_failure.md),
[`cftc_forward_return_event_study.md`](cftc_forward_return_event_study.md),
[`../memory/checkpoint_2026-09-23_session_handoff.md`](../memory/checkpoint_2026-09-23_session_handoff.md).

---

## 10. Niche first — and the data names the niche

Added after the framing question: *is this too general, and should we go niche
then expand?* Yes, and the generality is in the **data strategy**, not the
pitch.

```
64 collectors written · 34 writing · ~100 entities each
```

Sixty-four sources at a hundred entities each gives sixty-four islands. Five
sources at five thousand entities each gives a graph. Every downstream symptom
in §3 and §4 follows from going wide where depth was required.

### The candidate that looked best, and why it isn't

The macro cluster joins **28 of 28 source pairs — 100% density**, the only
cluster where cross-source synthesis fully works. The number is a mirage:

```
United States   sources=8  obs=20,898
Japan           sources=4  obs=    869
Canada, Germany, China, France, Italy   sources=2
all others      sources=1
```

It is a **star around the United States**, not a dense graph — every macro
source carries US data, so every pair "joins" on one shared entity. Twenty
countries total. The ceiling is too low for a cross-sectional test.

### The real diagnosis

```
company entities : 1,066
  with a ticker  :    20
  with a CIK     :    16
```

**Companies are keyed by name strings, not registry identifiers.** This is why
the SEC-family cluster joins at 13% when it should join at nearly 100% — every
SEC filing type carries a CIK by construction. `form144` ↔ `insider_filings`
share exactly **one** person between two datasets keyed on the same registry.

That is an engineering defect, not a fact about the world, and it has been
read as a finding. It belongs with F-14/F-15 and the ghost-chains regression in
the recurring pattern of this codebase: right mechanism, wrong wiring,
reporting green throughout.

### Recommendation: US-listed public companies, keyed by CIK

The only candidate satisfying all three conditions:

| condition | why CIK passes |
|---|---|
| canonical shared identifier | joins by construction, not fuzzy name matching |
| free depth | SEC bulk data — Form 4, 144, 8-K, 13F, litigation — keyless, thousands of entities |
| bridges to labels | CIK → ticker → prices, where a testable hypothesis lives |

Crypto fails condition 1 — `whale_alert` (addresses), `defi_flows` (protocol
names) and `polymarket` (market ids) share **0 entities across 2,929**. Macro
fails condition 2, per above.

**Expansion rule, held strictly: a new source enters only if it resolves to
CIK.** That one rule is what would have prevented 64 collectors producing 3
joins. It makes expansion principled rather than accretive.

### The cheap test that could falsify this

Resolve the existing SEC-family collectors to CIK and re-measure that cluster's
join density. **If it does not move from 13% toward 80%+, the diagnosis is
wrong** and the problem is not identifier resolution. Days, not weeks, on data
already collected. Run it before committing to the niche.

This test now precedes the retrain in §9 — a graph of 1,066 name-keyed
companies is not worth training on, and if CIK resolution works, the object
being trained changes enough to invalidate any run started before it.

## Related

- [[full_audit_2026-09-23]] — the 8-lens audit this framing rests on
- [[graph_connectivity_failure]] — why the graph had no edges
- [[cftc_forward_return_event_study]] — the 0/51 null, and its power limits
- [[checkpoint_2026-09-23_session_handoff]] — working-tree state and open decisions
