---
title: "Thirteen Ways a Pipeline Lies"
subtitle: "Production failures from a data system that reported success the entire time"
date: 2026-08-29
tags:
  - doc/publication
  - topic/data-engineering
---

# Thirteen Ways a Pipeline Lies

I spent about two years building a data pipeline that collects free public
data — CFTC futures positioning, DeFi TVL, SEC insider filings, GDELT events,
bankruptcy dockets, AIS vessel tracks, government contracts — scores it for
anomalies, and sold a weekly digest off the result.

I am shutting the product down. Four independent measurements say the edge
isn't there, the data isn't proprietary, the price point is occupied, and the
cross-source graph I built the whole thesis on has exactly two joinable source
pairs out of 561. That's a separate write-up.

This one is about something more portable, and in my opinion more useful:
**the codebase evolved from "healthy-looking and empty" to "healthy-looking
and full of constants."**

The first state is easy to catch. Every monitoring check you'd naturally write
— did the job run, did it exit zero, did it write rows — catches an empty
table. The second state passes all of those checks. The job runs, exits zero,
writes rows, and the rows are garbage that is *shaped* like a result. Sorting
by a column that's secretly a Unix timestamp looks exactly like sorting by
anomaly score, right up until you print the column.

Here are thirteen concrete ways this system lied to me. All of them shipped.
Most of them shipped to paying customers. For each I've given the symptom, how
long it hid, the root cause, the fix, and the general rule — the rule is the
part worth your time, the war story is just evidence that I paid for it.

The title is a nod to the repo's own failure log, which runs F-01 through
F-13. The numbering below is its own; some of these are in that log, some were
found while writing this.

Two snapshots frame everything that follows. This is the same database three
days apart:

| table | 2026-08-26 | 2026-08-29 |
|---|---:|---:|
| `entity_observations` | 365,739 | 375,657 |
| `entity_alerts` | **0** | 4,852 |
| `convergence_clusters` | **0** | 42 |
| `beliefs` | **0** | 115 |
| `signals` | **0** | 92 |

On the 26th I could see the problem from across the room. On the 29th it looked
fixed. Ways 7 through 10 are what those 4,852 rows and 42 rows actually contain.

---

## 1. The DAG reported "completed" when the write threw

**Symptom.** `daily_collection` surfaced as healthy. Its run history, once I
looked directly at it, was: one successful run ever, on 2026-04-19, and
`pipeline_data` holding 194 rows total for the life of the project.

**How long it hid.** From April to late August — about four months. Two runs
were additionally stuck in status `running` forever, which is the one signal
that *did* look wrong, and which I misdiagnosed (see Way 3).

**Root cause.** In the executor's `_execute_layer`, the storage-exception path
emitted a `log.warning` and left the node result's status as `"completed"`.
A run whose writes all threw was byte-for-byte indistinguishable, in
`dag_runs`, from a run that succeeded.

**Fix** (`0b52ab7`). Storage exceptions set status to `failed`. A stale-run
reaper plus a heartbeat so a process that dies mid-flight stops claiming to be
`running`. And a distinct classification for missing-credential skips, so an
unconfigured source degrades loudly instead of returning an empty list that
looks like "nothing happened today."

Here is the run history now. Note that it got *worse*-looking, which is the
point:

```
2026-04-19  completed
2026-04-19  failed
2026-04-21  failed
2026-04-21  failed
2026-08-25  failed  x4
2026-08-27  failed  x4
```

One success, eleven failures. Before the fix, those eleven were reporting as
`completed`, except two that were stuck in `running` and had been since April.

Eight nodes now skip explicitly on missing FRED/EIA/NASA credentials; two fail
on real upstream outages (a 403 from the lobbying source, a dead PatentsView
endpoint). That matches ground truth instead of hiding it.

There is a second thing visible in that list: between 2026-04-21 and
2026-08-25 there are **no runs at all**. The cron schedules were declared in
the DAG definitions and never actually installed anywhere. So for four months
the pipeline was simultaneously not running, and reporting success when it did.

> **Rule.** An exception handler that logs and continues has converted a
> failure into a silence. Every `except: log.warning(...)` on a write path is
> a decision that the caller should not be told. Make that decision
> deliberately, and write down why, or don't make it.

---

## 2. "Rows written" was a count of nodes, not a count of rows

This one I found while writing this post. It is unfixed as of today, and I'm
including it because a list of failures I already solved is a less honest
document than one that includes the one I found on the way to publishing.

**Symptom.** The guard I added in Way 1 to catch silent success — a run that
completes, has store-eligible nodes, and writes nothing gets downgraded to
`failed` — does not fire for the scoring DAGs. `rl_training` has six
`completed` runs and `rl_transitions` has zero rows. `adversarial_scan` has
six `completed` runs and `adversarial_flags` has zero rows.

**Root cause.** `agent/pipeline/executor.py:215`:

```python
run.rows_written = sum(1 for nr in run.node_results.values() if nr.stored)
```

`nr.stored` is a `bool` — "this node's result envelope landed in the store."
So `rows_written` is a count of *nodes*, capped at the number of nodes in the
DAG. A DAG whose every operator dutifully persisted one summary envelope into
the generic `pipeline_data` table, while writing nothing to the domain table it
exists to populate, scores `rows_written = 5` and sails through. The guard's
other escape hatch is `eligible_store_nodes == 0`, and the two DAGs above take
one or the other — I have not pinned down which, and it doesn't much matter,
because both are the same mistake: the guard reasons about the executor's
bookkeeping rather than about the database.

I wrote a guard against fake success, named its variable after the thing I
wanted to measure, and populated it with a proxy that cannot go to zero in the
failure mode I was guarding against.

**Fix.** Not yet applied. The correct measure is per-destination-table row
deltas taken before and after the run, which is what an auditor would ask for
and what I should have written the first time.

> **Rule.** When you build a health check, ask what value the field would take
> during the exact incident you are defending against. If it can't reach the
> alerting value during that incident, you have built a check that is
> guaranteed to pass. And a field name is documentation: `rows_written` that
> holds a node count will be trusted by the next reader, who will be me.

---

## 3. A timeout that could not stop the work

**Symptom.** The nightly chain ran all nine DAGs, then refused to exit for 40+
minutes. The box — 1 vCPU, 1.9 GB — reached **20 MB available with swap at
2038 / 2047 MB**. Two nodes, `train_gnn` and `generate_features`, were both
marked `failed`, and memory kept climbing after they had "finished."

**How long it hid.** Months. It produced the same evidence as Way 1 — a run
stuck in `running` with no error recorded — so I fixed the reporting bug,
saw the signature persist, and had no second hypothesis.

**Root cause.** Two faults compounding.

The executor's own source said the thing out loud: a timeout marks the node
failed and moves on, *"since the operator keeps running in the background
because a thread cannot be forcibly stopped."* `future.result(timeout=...)`
times out the **wait**, not the **work**. A timed-out operator is still
running and still allocating. The timeout was bookkeeping, not cancellation.

And `TIRRA_PIPELINE_WORKERS` defaulted to 4, on one core. Four workers on a
single core buy no parallelism — only context-switching and roughly 4x peak
RSS. Nothing compared that default against `nproc`.

Net: a 30-minute `train_gnn` timeout leaves an orphaned thread eating RAM for
as long as the parent lives, and the parent cannot exit while it runs.

**Fix** (`30bb00f`, `50308b3`). Workers set to 1 on single-core hosts, with the
measurement recorded next to the setting. The executor now logs an explicit
`ORPHAN THREAD` line naming the node and how far past its timeout it still is,
and refuses to start new work in a pool a timeout has already degraded. The
nightly chain was narrowed to the light DAGs that actually feed the product.
And `MemoryMax=1200M` / `MemorySwapMax=1500M` on the systemd unit, so the OOM
killer takes the batch job rather than choosing the customer-facing API
service for me.

> **Rule.** A timeout that cannot cancel its work is not a timeout, it is a
> leak with a stopwatch attached. In any runtime without forcible thread
> cancellation — Python very much included — either give the operator a
> cooperative cancellation token it checks, or run it in a process you can
> actually kill. And never let a background job share an OOM domain with
> something a customer is waiting on.
>
> Corollary, which cost me more than the bug did: when a fix doesn't clear the
> symptom, the fix was not wrong — you had two causes. I had assumed one.

---

## 4. Duplicate ingestion manufactured "structural break this week"

**Symptom.** The weekly digest told paying subscribers there was a structural
break *this week* in five separate CFTC futures contracts. There were no
structural breaks. There was one collector running eight times in a day.

**Root cause.** Collection re-ingests the same CFTC report on every run with no
upsert. So `entity_observations` accumulates duplicate rows. Measured on the
live database, and you can re-run this yourself:

```sql
SELECT COUNT(*)                                        -- 5488
FROM entity_observations WHERE observation_type='futures_positioning';

SELECT COUNT(*) FROM (SELECT DISTINCT entity_id, observed_at
FROM entity_observations WHERE observation_type='futures_positioning');
                                                       -- 5046
```

442 duplicates of 5,488 rows, 8.1%. And they are not spread evenly — a single
day, 2026-08-18, carries 272 rows against a typical day's ~35, because
collection ran about eight times that day.

The consequence is the interesting part. The last ten rows for every
high-duplication contract, by insertion order:

```
2026-08-18, 2026-08-18, 2026-08-18, 2026-08-18, 2026-08-18,
2026-08-18, 2026-08-18, 2026-08-18, 2026-06-02, 2026-06-02
```

Eight consecutive "weekly" observations carrying the **same** timestamp. The
changepoint detector (BOCPD) sees a discontinuity where that block begins —
which is real, the series genuinely changes shape there — and the elapsed time
from that index to now, measured in the series' own index space, is 0.0 weeks.
"A structural break, zero weeks ago." Rendered to the customer as *this week*.

**Fix** (`1bba4d4`). The read path now dedupes on `(entity_id, observed_at)`,
keeping `max(rowid)` so the most recently written value for a timestamp wins.
Cotton went from 169 points to 159 and its "this week" changepoint became
`None`. Ondo TVL went from 3,570 points to 1,190 — roughly 3x duplication in
DeFi too. Every surviving changepoint is now 15 to 186 weeks old, which is the
honest answer: there are no fresh structural breaks in this data right now.

That is a read-side guard, and I'm labelling it as one. The ingestion path
still needs the upsert; without it the duplicate rows keep accumulating and
every consumer other than the digest still sees them.

> **Rule.** Idempotency is not a nice-to-have on an ingestion path, it is the
> difference between a time series and a log of your own cron schedule. And
> notice the shape of the failure: the *data* was fine, the *analysis* was
> fine, and the artefact of re-running the collector was laundered through a
> correct changepoint detector into a confident false claim. Anywhere you
> measure elapsed time by counting array positions, you have assumed the
> array's spacing is real.

---

## 5. A 50-point baseline where 1,096 points existed

This is the most expensive one, and it is the one I'd want a stranger to take
away.

**Symptom.** The digest was shipping 29 volatility "anomalies" per run. After
the fix it shipped 4. Twenty-five of twenty-nine were false positives, going
out to subscribers, every week.

**Root cause.** The scorer computed z-scores against
`instrument_volatility`, which holds roughly 4 months of history. The exact
same 89 instruments also had `instrument_daily` going back to April 2023,
under a superseded observation-type name that nobody had reconciled. Live
numbers:

```
instrument_volatility  89 entities   46-61 points    2026-04-18 .. 2026-08-26
instrument_daily       89 entities  752-1096 points  2023-04-18 .. 2026-06-09
```

A ~50-point baseline makes ordinary seasonal variance look extreme. Nothing
was broken. No exception was raised, no row was missing, no test failed. The
statistics were computed correctly on the wrong denominator, and 86% of the
output was noise.

**Fix** (`dd93bdc`). The scorable-source table now maps a *tuple* of
`(observation_type, fields)` pairs per source and unions across type names,
with the later entry winning on the overlap period. `realized_vol_20d` findings
went 29 → 4.

The same commit killed three other things for related reasons, which I mention
because "make the product smaller" was the correct move each time:
`polymarket` was removed (its per-entity series tops out at 15 points, a hard
structural ceiling below the 20-point minimum — it was reported as a working
source every single run, forever); `instrument_return.close` was cut because 28
of its 33 findings had a z-sign matching the trailing return direction, i.e.
"this asset has been trending," sold as anomaly detection. `sovereign_debt` was
*kept* despite contributing almost nothing, because 5 of 13 entities clear the
floor with real variance — it is genuinely quiet rather than structurally
incapable, and that distinction is the whole job.

> **Rule.** A statistical result has three inputs — the method, the data, and
> the *amount* of data — and only the first two get reviewed. Print `n`
> alongside every derived statistic, in the output the customer sees, and set
> a floor below which you emit nothing rather than something. Then go looking
> for whether a longer version of your series already exists in your own
> store under a different name. Mine did: three years of it, for the identical
> 89 instruments, in the identical table, under a superseded type name.

---

## 6. Field names that did not exist, discarding 175,275 observations per run

**Symptom.** Every edition of the digest was 100% CFTC content, despite five
sources being configured. I had assumed the other four were quiet.

**Root cause.** The scorable-source config declared field names that do not
appear in the stored JSON payloads, and the series builder silently skips any
field it cannot find. Three of five sources therefore built *zero* series,
with no error logged anywhere:

| source | declared fields | actual fields | rows used |
|---|---|---|---:|
| `instrument_volatility` | `volatility`, `value` | `realized_vol_20d`, `intraday_range` | 0 of 4,294 |
| `tvl_change` | `tvl_change`, `value` | `tvl_usd` | 0 of 162,251 |
| `market_probability` | `probability`, `value` | `yes_price`, `no_price`, `volume_24h` | 0 of 8,730 |

175,275 observations per run, read from disk, matched against nothing,
discarded in silence.

**Fix** (`85c9777`). Corrected field names. Findings went from 8 (100% CFTC) to
26 across three sources. The DeFi TVL series turned out to carry 3,123- and
3,570-point baselines with genuine changepoints — far deeper history than
CFTC's 169 weekly points. The best data in the system had been invisible
because of a string literal.

> **Rule.** A lookup that misses is an event, not a `None`. Any config that
> names a field in a schemaless payload must be validated against a live
> sample at startup and fail loudly on a miss — because "this source is quiet"
> and "this source is misconfigured" produce identical output, and only one of
> them is something you'd act on.

---

## 7. The anomaly ranking sorted by Unix timestamp

Now we are in "healthy-looking and full of constants" territory. The
`entity_alerts` table has 4,852 rows. It is the input to the ranking that
decides which entities are interesting. It is fully populated, non-null,
plausibly distributed, and completely determined by the clock.

**Symptom.** Nothing. That's the point. The table looks fine.

**Root cause.** `agent/fusion/surprise.py`, the temporal component:

```python
dt_actual = obs.get("observed_at", 0.0)
# We use the raw absolute error, z-scored later
temporal_s = abs(dt_pred_val - dt_actual)
```

`observed_at` is a raw Unix epoch, around 1.787e9. The model's predicted delta
is a small number near zero. So `temporal_surprise` is, to four significant
figures, **the observation's timestamp**. The comment promises z-scoring
"later." Later never arrives.

The composite is a fixed weighted sum of five components. Decomposing it over
all 4,852 rows against the code's own normalised weights:

| component | mean contribution | share of composite |
|---|---:|---:|
| `temporal` | 265,258,327.59 | **90.066%** |
| `neighborhood` | 29,252,180.70 | **9.932%** |
| `value` | 4,542.36 | 0.002% |
| `obs_type` | 6.85 | 0.000% |
| `memory` | 0.00 | 0.000% |

The reconstruction matches the stored `composite_surprise` mean to fifteen
significant figures, so this is arithmetic, not inference. And
`neighborhood_surprise` is itself a weighted mean of neighbours' composites —
so it, too, is 90% timestamp. The Spearman correlation between
`composite_surprise` and the two epoch-derived terms alone is **0.99993**.

Everything the score was supposed to measure — how surprising the observation's
*type* was, how far its *value* deviated, how much the entity's memory *drifted*
— together accounts for 0.002% of the ranking.

Three further constants in the same table, all across all 4,852 rows:

- `memory_drift`, `cusum_statistic`, `event_study_score` are **exactly 0.0** in
  every row. Three of nine columns are dead and nothing said so.
- `obs_type_surprise` sits at `23.025850929940457` in 4,808 of 4,852 rows —
  that is `-log(1e-10)`, the clamp floor. 99.1% of the time the model assigns
  the observed type a probability below the clamp, and the "surprise" is the
  clamp constant, not a measurement.
- `alert_time` has exactly **one** distinct value across all 4,852 rows.

**Fix.** None applied — this table is now excluded from everything customer-
facing, and the tier that would have consumed it is being retired.

> **Rule.** Any quantity that enters a weighted sum must be dimensionless
> before it gets there, and the place to enforce that is a unit test that
> asserts each component's magnitude range, not a comment saying it'll be
> normalised later. Then: for every composite score you ship, log the
> per-component contribution share on a real sample. If one term is 90% of the
> score, either that's intended and you should say so, or you have just found
> a bug that no amount of staring at the formula would have shown you — this
> formula is *correct*. It's the inputs that aren't.

---

## 8. Cosine similarity that could not return anything but 1.0

**Symptom.** `convergence_clusters` — the table representing "these entities
are surprising together, which is the actual product thesis" — has 42 rows.
Their `correlated_surprise_score` values run from **0.9976510** to
**0.9998583**. All 42. The score's own docstring says "1.0 for perfectly
aligned surprise patterns, 0.0 for orthogonal."

**Root cause.** The score is the mean pairwise cosine similarity of the
five-element surprise vectors from Way 7 — used raw, unnormalised. Every such
vector has the shape `[~23, ~1.7e9, ~small, ~1.4e8, 0.0]`. Its largest
component is between one and eight orders of magnitude bigger than every other
component, and it is a timestamp, so it agrees closely between any two rows
collected in the same era. Two such vectors point in almost the same direction
*no matter which entities they describe*.

Measured on 300 real alert vectors, 44,850 pairs:

```
pairwise cosine     min 0.7306   p1 0.9093   median 0.9890   max 1.0000
                    49.1% of all pairs exceed 0.99
```

Averaging over a cluster's pairs pulls it tighter still, which is how 42 out of
42 land in a band 0.0022 wide. The number is not measuring convergence. It is
measuring that these are all timestamps.

**Fix.** None; the table is excluded from the product and the tier is retiring.

> **Rule.** Cosine similarity on unnormalised, mixed-unit vectors is not a
> similarity measure, it is a measure of which component has the biggest units.
> Standardise per dimension before any geometric operation. And a diagnostic
> for free: if a score that is *defined* on [0, 1] never leaves a 0.002-wide
> band across your entire dataset, it is not a score. Assert the spread, not
> just the range — `0 <= x <= 1` passes on a constant.

---

## 9. Placeholders that outlived the promise to fill them

Same 42 rows, different mechanism, so it gets its own entry.

**Symptom.** `contributing_tools_json` is `[]` in all 42 rows.
`temporal_span_hours` is `0.0` in all 42 rows. Both are columns a reader would
use to sanity-check a cluster: which data sources produced it, and over what
window.

**Root cause.** `agent/fusion/convergence.py`:

```python
tools: tuple[str, ...] = ()          # filled by scorer
...
temporal_span_hours=0.0,             # same moment
```

The scorer does not fill it. `grep -n contributing_tools agent/fusion/entity_scorer.py`
returns nothing. The DAG faithfully persists the empty tuple.

`temporal_span_hours` is worse, because the class docstring twelve lines above
defines it: *"Hours between earliest and latest member alert_time."* That is a
subtraction over data the function already has in hand. It is a hardcoded
literal instead, with a comment asserting a fact — "same moment" — that nothing
measured.

Both survived from the initial implementation to production because a column
that is always empty raises no error and fails no test, and because writing the
comment felt like writing the ticket.

**Fix.** None; excluded with the rest of the table.

> **Rule.** `# filled in later` is an unassigned ticket living in a place with
> no ticket queue. If a field cannot be computed yet, make it nullable and
> leave it `NULL` — a `NULL` is a question, an empty list is an answer. Then
> add the cheap test that catches this entire family: for every column you
> persist, assert it takes more than one distinct value across a real run.
> That single test would have caught Ways 7, 8, and 9 on the day each landed.

---

## 10. A threshold that filtered nothing

**Symptom.** The convergence detector's first step is "filter to entities with
`composite_surprise` above a threshold." The default threshold is `2.0`.

The minimum `composite_surprise` in the table is **64,241,535**.

**Root cause.** The threshold was chosen when the composite was a z-score-like
quantity in single digits, which is what the design intended. Way 7's units bug
moved the composite eight orders of magnitude, and the threshold — a plain
default argument in a signature, correct at the time it was written — was never
revisited, because nothing connects the two. Every entity passes. The
"elevated entities" set is the set of all entities.

So the clustering that produced Way 8's numbers ran over the entire graph, not
over an anomalous subset. There is no selection step. There never was.

**Fix.** None; excluded.

> **Rule.** A constant threshold on a derived quantity is a silent coupling to
> that quantity's scale, and scale changes don't produce merge conflicts. Log
> the pass rate of every filter, every run — `filtered 4852 -> 4852 (100.0%)`
> is a line that would have ended this in one night. A filter that never
> rejects and a filter that always rejects are both bugs, and neither one
> throws.

---

## 11. Six tests that passed by never running the code

**Root cause.** `tests/test_entity_linking.py` has a fixture builder for GDELT
events, defaulting to `goldstein: float = 1.0`. In May, a Goldstein tension
gate was added to `GDELTTool._persist_entities`:

```python
if gs is None or gs >= _GOLDSTEIN_TENSION_THRESHOLD:   # -5.0
    continue
```

Goldstein scores run roughly −10 (conflict) to +10 (cooperation), and the gate
drops anything at or above −5.0 as routine diplomatic noise — by the source
file's own measured distribution, that is 89.9% of GDELT. Every fixture in the
class defaulted to `1.0`, squarely inside the discarded 89.9%. So every event
was dropped before reaching the link-creation code the class exists to test.

`TestGDELTEventInvolves` has ten tests, and the gate split them cleanly:

- **Four** assert a link *is* created (`test_normal_link_created`,
  `test_link_metadata_contains_event_info`, `test_dedup_same_country_pair`,
  `test_multiple_country_pairs`). These went **red** the day the gate landed.
- **Six** assert that *no* link is created — same country, missing country,
  whitespace-only country, no store attached. These stayed **green**, and were
  worthless. `assert len(links) == 0` is trivially true when nothing runs.
  Not one of them could have failed for any reason, and all six were serving
  as evidence that the gate logic and the country-comparison logic worked.

**How long it hid.** The gate landed 2026-05-12; this was found 2026-08-27.
107 days.

And here is the part I'm least proud of, because it's the part that made the
other 107 days possible: **the four red tests were red the whole time.** The
suite had 19 failures when I finally sat down with it. A suite with a standing
failure count is a suite nobody reads, and inside that unread failure count
were six tests quietly reporting success for a reason that had nothing to do
with the code under test.

**Fix** (`95674ff`). Fixture default changed to −7.0 so the code actually runs;
the class went green *and* meaningful. Suite went 19 failures to 10, and the
remaining 10 were real bugs fixed separately. The fix carries a seven-line
comment naming the commit that introduced the gate, because the next person to
see `goldstein=-7.0` will wonder why it's not zero.

> **Rule.** A test that cannot fail is worse than no test, because it consumes
> the attention a missing test would have attracted. Two cheap defences:
> mutation-test your critical paths (break the code on purpose, confirm the
> suite goes red), and treat coverage on *guard clauses* as a first-class
> metric — the gate that skipped everything was covered; the branch behind it
> was not. Whenever you add an early-`continue` to production code, grep the
> fixtures that feed it.
>
> And the meta-rule, which is the expensive one: **a non-zero standing failure
> count destroys the signal value of the entire suite.** Not just for the
> broken tests — for every test in it, including the six that were lying.

---

## 12. A test that asserted the bug

**Symptom.** Three DAGs failed or silently produced nothing.
`entity_scoring` crashed with `index 69 is out of bounds for dimension 1 with
size 69`. The other two threw `mat1 and mat2 shapes cannot be multiplied
(93x49 and 23x64)`. Six downstream tables held zero rows against 365,000
healthy observations.

**Root cause.** Three registries had drifted apart with nothing comparing them:

| | live DB | code constants | trained weights |
|---|---:|---:|---:|
| entity types | 12 | 11 | 12 |
| observation types | 38 present, 4 unknown to code | 48 | 48 |
| instrument feature dim | 49 | 49 | **23** |

Three failures fell out of that. `ENRICHMENT_DIM` was hardcoded to `55` —
correct only while the observation-type list had 46 entries; the writer indexes
`offset + 9 + ot_idx` over the *live* list, so at 48 types the block overflowed.
For one node type that overflow hit the end of the tensor and crashed. For
instrument nodes it ran into the price-feature block that follows: **silent
corruption instead of a crash, depending only on node type.**

Separately, `maritime_area` existed in the database but not in the code's
`ENTITY_TYPES`, so the feature builder fell back to `type_idx = 0` and one-hot
encoded every maritime area as a CFTC contract. A `log.warning` fired and the
run continued. It trained and scored as the wrong kind of entity for months.

And the part that belongs in this list: **a test asserted the buggy behaviour.**
`assert features[0, 0] == 1.0` — which is precisely what the wrong fallback
produces. The suite was green *over* the corruption, and had been used as
evidence that the corruption wasn't there.

**Fix** (`e974ce3`). `ENRICHMENT_DIM` is now derived
(`9 + len(OBSERVATION_TYPES)`), never written as a literal. An unknown entity
type gets an **all-zero** one-hot — claiming no identity rather than the wrong
one. A new `validate_schema_against_store()` raises before anything trains or
scores, listing every live type the code cannot encode. And checkpoint loading
now names the drift explicitly — `instrument: trained_weights=23
expected_by_model=49` — instead of logging "skipped N keys," which is how a
randomly-initialised layer had been passing for a harmless omission.

> **Rule.** Three rules, and I'd take all three anywhere:
> 1. Any dimension derived from a registry must be **computed**, never
>    hardcoded. A literal that equals `len(SOME_LIST)` today is a time bomb
>    with one registry edit left on its timer.
> 2. Never degrade an unknown categorical to index 0. Claiming no identity is
>    honest; claiming the *wrong* identity is corruption that trains cleanly
>    and evaluates cleanly and is invisible.
> 3. If a test asserts a fallback or default, verify the fallback is correct
>    before treating the green suite as evidence of anything. A test freezes
>    behaviour; it does not bless it.

---

## 13. Fifty-one hypotheses, zero survivors, and one that looked significant

The last lie is the one I'd been most at risk of telling other people.

**Symptom.** A forward-return event study on the product's own CFTC anomalies
produced a headline result: `mm_net_pct_oi` at `|z| >= 2`, 20 trading days,
n=123 events, **+2.18% edge over baseline**, 65.0% hit rate against a 50.4%
base rate, **p = 0.002**.

That is a marketing page. It is also noise.

**Root cause.** The study tested 9 fields x 3 horizons x 2 z-thresholds = 54
hypotheses, 51 of them with enough events to be testable. Under the null, the
expected number of uncorrected p < 0.05 results from 51 independent tests is
about 2.6. I got 7. Applying Benjamini-Hochberg at alpha = 0.05: **0 of 51
survive.** The best adjusted p-value is 0.102.

A single p = 0.002 pulled from a 51-cell grid is not evidence. It is the grid.

Two further problems, both of which cut against the positive:

**Pseudo-replication.** Even the best cell is not 123 independent
observations. Those 123 events fall on only **116 distinct weeks**, and the
busiest weeks have 5 of 5 eligible contracts firing simultaneously — the same
macro positioning shift hitting correlated commodities in the same week. The
effective n is meaningfully below the counted n, so even the uncorrected 0.002
overstates confidence before BH is applied at all.

**Attrition.** Of 21,294 (contract, field, week) combinations where a causal
z-score was computable, **7,029 (33.0%)** had to be dropped because no
instrument price existed within 7 days of the CFTC report's publication date.
One in three anomalies in the product's own history cannot be graded at all.

I built the method to be hostile to a positive result, which is the only way to
believe one:

- The null is the **unconditional** forward return over the same horizon pooled
  across all 19 tickers (n ≈ 1,578 per horizon), not zero. The 2023–2026
  commodity tape drifted up; measuring against zero would have handed me an
  "edge" that was just the drift.
- **Publication lag is honoured.** CFTC `observed_at` is the Tuesday *as-of*
  date (verified: every distinct timestamp in the DB falls on a Tuesday); the
  report goes public the following Friday. The study adds a 3-day lag and
  enters at the first close on or after publication. The naive version would
  have traded three days before the data existed. That constraint is where most
  of the 33% attrition comes from.
- The z-score function is a **direct copy** of the production one — expanding
  window, `hist = x[:-1]` — so the test scores what the product shipped, not an
  idealised version of it.
- No contract was selected or dropped on performance. All 19 price-linked
  contracts are used unconditionally.

**And the null replicates the literature.** Re-running BH on the published
p-values in Sanders, Irwin & Merrin (2009) — a positive result in the COT
literature — gives **0 of 30 survivors**, minimum adjusted p = 0.105, against
my 0.102. My negative result agrees with the published data. It disagrees with
the published *conclusion*, and the difference is entirely that the published
work does not correct for multiple testing. Neither does most of the rest of
this literature.

**The caveat I have to state, because a hostile reader would find it.** My
specification pools two-sided `|z|` and appends the forward return unsigned:

```python
if abs(z) >= zt:
    events[(field, zt, h)].append(log_ret)
```

A genuinely signed effect — high positioning predicts down, low predicts up —
cancels itself in that bucket by construction. Combined with roughly 10% power
at these sample sizes, the honest claim is **"no detectable effect at low power
on this specification,"** not "no effect exists." I am not going to overstate
my negative result to make the shutdown look more decisive than it is.

> **Rule.** Decide your hypothesis count *before* you look, and correct for it.
> If you tested a grid, report the grid — a result presented without its
> denominator is not a result. Then check whether your n is really n: events
> clustered in calendar time or across correlated instruments are not
> independent trials, and nothing in the p-value knows that. And when a
> negative result would be convenient for you, audit it as hard as you'd audit
> a positive one; mine has a specification flaw that I would very much have
> preferred not to find.

---

## What I'd actually keep

Reading these back, they sort into four families, and only one of them is
about being careless.

**It lied about whether it ran** (1, 2, 3). Failures converted to silence by
exception handlers; a health check measuring a proxy that cannot reach its own
alerting value during the incident it guards; a timeout that stops the waiting
rather than the working. Ordinary, and every system has them. These are the
cheap ones to find, because someone eventually notices the table is empty.

**It lied about what it had** (4, 5, 6, 12). Duplicates laundered through a
correct changepoint detector into a false market claim; a correct statistic on
a 22x-too-short baseline; 175,275 rows a run discarded on a string literal; a
registry drift that crashed for one node type and silently corrupted for
another. Shared signature: **nothing threw**. The data was wrong in a way that
has no exception type, so there was nothing to catch and nowhere to catch it.

**It lied about what it meant** (7, 8, 9, 10, 13). A ranking sorted by
wall-clock; a similarity that could not return anything but 1.0; placeholders
that outlived the comment promising to fill them; a filter with a 100% pass
rate; a p-value detached from its denominator. Every one of these tables is
populated, non-null, correctly typed, plausibly distributed, and would pass any
schema check you or I would write. The formulas are right. The units aren't.

**And the tests agreed with all of it** (11, 12, and Way 1's guard). Six tests
green because a gate skipped the code they existed to test. One test asserting
the corrupt fallback value, so the suite was green *over* the corruption and
was being cited as evidence against it. And a silent-success guard whose own
measure was a node count. Every one of those is a defence I built, that
reported the all-clear, on the specific failure it existed to catch.

The through-line: **every check I had asked "is there output?" and none asked
"does the output vary?"** A constant is the most convincing possible fake — it
has a type, a range, a mean, a plausible distribution once you have 4,852 of
them, and it is stable across runs, which reads as reliability.

If you take one thing from this, take the cheapest test in the list, from
Way 9. For every column you persist, on every real run, assert it takes more
than one distinct value, and log its distinct count and its spread. That single
check catches Ways 7, 8, 9, and 10 — four of the five worst things in this
post — on the day each of them lands, before any of it reaches a customer.

I did not have that check. That is why this is a post-mortem and not a
changelog.

---

## Reproducing any of this

Everything above is a query, a file and line, or a commit in the repository.
Nothing is anonymised and nothing is rounded in my favour.

| # | Evidence |
|---|---|
| 1 | `0b52ab7`; `SELECT dag_name, status, COUNT(*) FROM dag_runs GROUP BY 1,2` |
| 2 | `agent/pipeline/executor.py:215` and `:250`; unfixed |
| 3 | LESSONS.md F-13; `30bb00f`, `50308b3`; `agent/pipeline/executor.py:328-354`, `:471` |
| 4 | `1bba4d4`; duplicate-count query in the text above |
| 5 | `dd93bdc`; point counts per `observation_type` in `entity_observations` |
| 6 | `85c9777`; `4294 + 162251 + 8730 = 175275` |
| 7 | `agent/fusion/surprise.py:190-196`, weights at `:74-100`; decomposition over `entity_alerts` |
| 8 | `agent/fusion/convergence.py:184-199`; `SELECT MIN/MAX(correlated_surprise_score) FROM convergence_clusters` |
| 9 | `agent/fusion/convergence.py:167` and `:175` (docstring at `:46`); `grep -n contributing_tools agent/fusion/entity_scorer.py` returns nothing |
| 10 | `agent/fusion/convergence.py:97`, default `surprise_threshold=2.0`; `MIN(composite_surprise) = 64241535` |
| 11 | `95674ff`; `tests/test_entity_linking.py::TestGDELTEventInvolves` (line 611); `agent/tools/gdelt.py:50-53,716` |
| 12 | LESSONS.md F-12; `e974ce3` |
| 13 | `docs/research/cftc_forward_return_event_study.md`; `scripts/cftc_event_study.py` |

The failure log this is drawn from — F-01 through F-13, each with symptom, root
cause, fix, and prevention rule — is `LESSONS.md` in the repository. It was
maintained continuously, not reconstructed for this post, which is the only
reason I can date how long each of these hid.
