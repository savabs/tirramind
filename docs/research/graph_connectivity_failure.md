---
title: "Why the cross-domain graph never predicted anything"
tags:
  - doc/research
  - topic/world-model
  - layer/world-model
date: 2026-09-23
status: active
---

# Why the cross-domain graph never predicted anything

**Finding: the world-events half of the graph and the tradeable half are two
disconnected components, joined by nothing. The GNN had no cross-domain path to
propagate along, because no cross-domain path exists.**

This is a connectivity and entity-resolution failure, not a market-efficiency
result. It is fixable.

---

## 1. What this does *not* say

`docs/publications/cot_null_result.md` reports 0 of 51 hypotheses surviving
Benjamini-Hochberg. That study tested **one** family: CFTC COT managed-money
positioning `|z| >= 2` level-extremity against 20-day forward returns on 19
futures contracts. Its own Section 7 states the honest claim — *"No detectable
effect, at approximately 10% power, on a specification the literature does not
endorse."*

`futures_positioning` is **5,488 of 375,657** observations — **1.5%** of the
corpus. The COT null killed the COT product. It never tested the entity graph,
and it is not evidence about the graph in either direction.

The graph's predictive value is, as of this document, **untested** — not refuted.

---

## 2. The graph looks healthy

`.tirra_pipeline/pipeline.db` (150 MB):

| table | rows |
| --- | --- |
| entities | 6,172 |
| entity_observations | 375,657 |
| entity_links | 17,581 |
| entity_alerts | 4,852 |
| convergence_clusters | 42 |
| depth_evaluations | **0** |
| rl_transitions / portfolio_weights / paper_trade_pnl | **0 / 0 / 0** |

17,581 links across 18 link types reads like a rich cross-domain graph. It is
not one.

---

## 3. The graph is two islands

Link topology, joined to entity types on both endpoints:

| type_a | link_type | type_b | n | avg_conf |
| --- | --- | --- | ---: | ---: |
| country | event_involves | country | 9,487 | 0.90 |
| topic | topic_relates_to_instrument | instrument | 2,752 | **0.64** |
| wallet | transacts_with | wallet | 2,131 | 1.00 |
| wallet | trades_instrument | instrument | 1,384 | 1.00 |
| person | works_for | company | 1,139 | 1.00 |
| company | market_authorized_in | country | 141 | 1.00 |
| instrument | produced_in | country | 119 | 0.87 |
| *(12 further types)* | | | < 110 each | |

**66% of all links (11,618) are within-domain**: country↔country (GDELT) and
wallet↔wallet (whale_alert). Neither touches an instrument.

### 3.1 The decisive measurement

```sql
-- GDELT countries: 215.  Of those, how many reach an instrument in one hop?
```

**215 GDELT countries. 0 reach an instrument.**

Not "few". Zero. The 9,487 `event_involves` links — 54% of the entire graph —
form an island with no edge to anything tradeable.

---

## 4. Root cause: two country-code conventions, never reconciled

The two islands are not semantically disconnected. They are disconnected because
**the same country is keyed under two different code systems and therefore
hashes to two different `entity_id`s.**

`entity_id_from_key` (`agent/pipeline/entity.py:144`) is
`sha256(f"{entity_type}:{key}")[:16]`. The key is the country *code*, so the
choice of code system silently decides identity.

| writer | file | key used | example |
| --- | --- | --- | --- |
| GDELT | `agent/tools/gdelt.py:750` | **CAMEO alpha-3** (`Actor1CountryCode`, col 7) | `USA`, `FRA`, `LBN` |
| instrument universe | `agent/pipeline/cross_entity.py:291` | **ISO/FIPS alpha-2** | `US`, `FR`, `LB` |

`country:USA` and `country:US` are different strings, so they are different
entities. Verified directly:

```
CAMEO alpha-3                        ISO alpha-2
  USA -> 65c0a83a115572ef  ALASKA      US -> f07b20bdd8b5f9bb  US
  FRA -> c2360f0ee92ac838  FRENCH      FR -> 930c34ae323eb4e4  FR
  CAN -> 0dc100fc22ae1b83  SASKATCHEWAN CA -> 33581f9bc52c5b46  CA
  JPN -> e7acc8dfc013d69e  TOYOTA      JP -> c8d97408e0b9f9d8  JP
  RUS -> 3cede9de52cf8304  RUSSIA      RU -> c9db6e2a9784c856  RU
```

**The United States exists twice.** The CAMEO-keyed record carries **385 links
and 18,980 observations**; the ISO-keyed record is the one the instruments
attach to. They have never been joined.

### 4.1 The canonical_names are also wrong (cosmetic, but it hid the bug)

`gdelt.py` sets `canonical_name = actor.get("name") or country`, where `name` is
GDELT's `Actor1Name` — the *actor*, not the country. The name stored is whichever
actor happened to arrive first:

| entity key (CAMEO-3) | stored canonical_name | actual country |
| --- | --- | --- |
| `USA` | ALASKA | United States |
| `CAN` | SASKATCHEWAN | Canada |
| `JPN` | **TOYOTA** | Japan |
| `FRA` | FRENCH | France |
| `IRL` | IRISH | Ireland |
| `AUS` | SYDNEY | Australia |

This is why the graph appears to contain cities, provinces, adjectives and a car
manufacturer as "countries". The **identity is correct** (all US actors do
collapse to one `country:USA` entity) — only the display name is junk. That
matters for the fix: **no link is mis-attributed, so an alpha-3 → alpha-2 merge
is safe and loses nothing.**

### 4.2 Scale of the split

| | ISO alpha-2 side | CAMEO alpha-3 side |
| --- | ---: | ---: |
| country entities | 63 | 216 |
| link endpoints | 512 | 18,974 |
| observations | 2,230 | **92,211** |

**92,211 `geopolitical_event` observations — 24.5% of the entire corpus — sit on
the CAMEO side with no path to any instrument.**

### 4.3 The mapping already exists in the database

`entity_aliases` holds **216 rows with `source='fips'`** — one per CAMEO-keyed
country — storing exactly the alpha-3 code needed to drive the merge:

```
fips  USA -> ALASKA        fips  LBN -> LEBANON
fips  FRA -> FRENCH        fips  CAN -> SASKATCHEWAN
fips  IRN -> IRAN          fips  JPN -> TOYOTA
```

So the migration is mechanical: for each country entity, read its alpha-3 alias,
map to ISO-3166-1 alpha-2, find-or-create that entity, repoint links and
observations, merge. No inference or fuzzy matching is required.

## 5. The one real bridge is semantic noise

With GDELT orphaned, the only substantial world→tradeable seam is
`topic_relates_to_instrument`: 2,752 links at **avg confidence 0.64**, the
lowest in the graph. **1,657 of them (60%) come from a source named
`repair_topic_links` at confidence 0.6.**

A random sample:

| topic | instrument | conf | source |
| --- | --- | ---: | --- |
| James Norton announced as next James Bond? | 10-Year T-Note Futures | 0.6 | repair_topic_links |
| Josh O'Connor announced as next James Bond? | 20+ Year Treasury ETF | 0.6 | repair_topic_links |
| Will bitcoin hit $1m before GTA VI? | **S&P 500 ETF** | 0.6 | repair_topic_links |
| Aziz Akhannouch out as Morocco PM? | 10-Year T-Note Futures | 0.7 | polymarket |
| Will James Talarico win the 2028 election? | S&P 500 ETF | 0.6 | repair_topic_links |

A James Bond casting announcement is linked to Treasury futures. A Bitcoin price
question is linked to the S&P 500 ETF and **not** to Bitcoin. These edges are
close to arbitrary — they carry a topic-to-instrument association that does not
encode any real-world relationship.

This is the repo's own documented pattern from
`docs/publications/thirteen_ways_a_pipeline_lies.md`: healthy-looking and full
of constants.

---

## 6. Concentration

Inbound link degree, instrument side:

| instrument | degree |
| --- | ---: |
| Bitcoin | **1,464** |
| VIX Short-Term Futures ETF | 424 |
| 20+ Year Treasury ETF | 411 |
| 10-Year T-Note Futures | 411 |
| S&P 500 ETF | 397 |
| Gold | 95 |
| *(87 further instruments)* | < 95 each |

Bitcoin holds 3.5x the next instrument, almost entirely via whale_alert
wallet→instrument edges. Across 93 instruments, a GNN trained on this graph
largely learns "Bitcoin, plus five index proxies".

---

## 7. The June IC numbers cannot be cited (F-12)

`.tirra_pipeline/ic_results.json` reports GNN-EmbNorm `mean_ic = 0.0468`,
`t = 2.21`, `n = 39`, with embedding-bypassing heads negative
(GNN-ValueHead −0.0398, GNN-ReturnHead −0.0339), and names
`GNN-ConcatReturnHead` as `primary_ic_strategy` with **no results entry at all**.

**Those numbers describe a feature space that no longer exists.** Verified by
schema audit:

| dimension | `gnn_model_phase50.pt` (the June eval) | current code + `gnn_model.pt` |
| --- | ---: | ---: |
| entity types | **9** (no `vessel`, `domain`, `maritime_area`) | 12 |
| observation types (`obs_type_head`) | **46** | 52 |
| `BASE_FEAT_DIM` | **14** | 15 |
| instrument feature dim | **23** (no microstructure, no options/rates/dividend) | 50 |

Loading phase50 under the current architecture skips `obs_type_head` on shape
mismatch and reports 6 missing keys. No code bridges 23 → 50 instrument dims
without discarding the learned `type_projections.instrument` weights anyway.

This is exactly the `F-12` checkpoint-archaeology failure in `LESSONS.md:461-533`.
**The June IC is not weak evidence of edge — it is not evidence about the current
model in either direction.** It must not be quoted as a baseline.

By contrast, `gnn_model.pt` (Aug 26) **is** valid against the current schema:
`Trainer.load_model` returns 0 missing, 0 unexpected, 0 skipped keys, and a live
`GraphBuilder(store).build()` reproduces its declared `in_channels` exactly. Its
TGN memory was sized for 5,628 nodes vs 6,172 live, which `HetTGNMemory.resize()`
zero-pads safely (`agent/models/gnn/het_tgn.py:245-260`) — so it is usable, just
not trained on the newest 544 entities.

**Consequence: there is currently no trustworthy IC number for this project.**
Not a good one, not a bad one. The graph's edge is unmeasured, and sections 3–6
explain why measuring it before reconnecting the graph would be measuring the
wrong object anyway.

## 8. Why this is good news

Every defect above is an engineering defect at the Layer 1 → Layer 3 boundary,
not a statement about markets:

- The world-events data was **collected correctly** (92,211 observations).
- It was **attached to the wrong entity records**.
- The model was therefore asked to find cross-domain structure in a graph that
  contained none.

The hypothesis "cross-domain links between world events and instruments carry
predictive information" has **not been tested yet**. It became testable only
after the graph is actually connected.

---

## 9. Ordered next steps (see spec)

1. **Canonicalize country entities to ISO-3166-1 alpha-2** at the writer
   boundary, with an alias table, and migrate the 216 full-name records.
   Expected effect: 9,487 GDELT links and 92,211 observations join the main
   component for the first time.
2. **Reject non-country geographies** (`GEORGETOWN`, `SASKATCHEWAN`) at write
   time; route them to a `region`/`city` type or drop them.
3. **Merge the intra-convention duplicates** (`US`, `LY`).
4. **Quarantine `repair_topic_links`.** Re-derive topic→instrument edges from
   an explicit, auditable rule, or drop the 1,657 synthetic edges. Measure the
   graph with and without them.
5. **Re-measure connectivity** — the acceptance test is "N of 215 GDELT
   countries reach an instrument", currently 0.
6. **Only then** run a leak-audited, multiple-testing-corrected IC evaluation
   (per `F-04`) on the reconnected graph, with `GNN-ConcatReturnHead` actually
   enabled per `F-02`.

Step 6 is the first honest test of the project's core thesis. Steps 1–5 are
prerequisites, and none of them involve a model change.

---

## 10. Prevention

This belongs in `LESSONS.md` as a new entry. Proposed rule:

> **Any Layer 1 writer that creates an entity must canonicalize through a
> shared resolver before writing.** A graph's link count is not evidence of
> connectivity. Before trusting any graph, measure whether the components you
> believe are joined actually reach each other:
> `SELECT count of source-domain entities with a path to target-domain entities`.
> A cross-domain model trained on a disconnected graph will report normal
> losses and near-zero true signal, indefinitely.

## Related

- [[graph_connectivity_repair_spec]] — the ordered fix
- [[graph_connectivity_repair]] — task tracking it
- [[cot_null_result]] — the null this document is careful not to overclaim
- [[thirteen_ways_a_pipeline_lies]] — the same pattern, at architecture scale
