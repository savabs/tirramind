# Contributing

This project is in **maintenance and archive mode**. The subscription business
built on it was wound down after four independent measurements found no
predictive edge, no data moat, no joinable entity graph, and no viable price
point — see [the README](README.md#read-this-first-what-does-not-work). What
remains and is worth keeping is the collector, the negative result, and the
failure log.

Contributions are welcome in that spirit. Please read this before opening a PR,
because some categories of contribution will be declined on principle rather
than on quality.

---

## What is welcome

**New collectors.** The clearest way to add value. Requirements in
[docs/COLLECTOR.md §8](docs/COLLECTOR.md#8-adding-a-collector). In short: free
public source, subclass `Tool`, honest `success=False` on failure, source-derived
`observed_at`, register in the DAG.

**Fixes to existing collectors.** Parsers that broke when an upstream API
changed, sources that have gone stale, timestamp handling. `cftc_derived` has
not ingested in 118 days and is the most obviously dead thing here.

**Deduplication.** `entity_observations` has no unique constraint and 52.8% of
the reference database is duplicate rows
([§5.1](docs/COLLECTOR.md#51-observations-are-not-deduplicated)). A migration
adding `UNIQUE(entity_id, source_tool, observed_at, observation_type)` plus
`INSERT OR IGNORE` would be a genuine improvement. It needs a backfill story for
existing databases — please propose in an issue first.

**Documentation that removes a surprise.** If something cost you an hour to
figure out, that is a bug in the docs.

**Deleting things that do not work.** Removing or clearly quarantining the
degenerate layers is a contribution, not vandalism.

---

## What will be declined

**Claims of predictive performance without multiple-testing correction.** This
is the one hard rule. The project died of this specific error, twice: once in
our own work, and once in the published literature it was built on (re-running
Benjamini–Hochberg on Sanders, Irwin & Merrin (2009) leaves 0 of 30 results
standing). A PR asserting that some signal predicts returns must include:

1. A **pre-registered specification** — fields, horizons, thresholds fixed
   before you look at outcomes. State how many hypotheses you tested, including
   the ones you abandoned.
2. A **multiple-testing correction** over the full hypothesis count.
   Benjamini–Hochberg is fine. Reporting a min p-value is not.
3. A **non-trivial null.** Comparing against zero rather than the unconditional
   return manufactures significance.
4. **Publication lag honoured.** If the data is published Friday about Tuesday,
   you cannot trade it Tuesday.
5. **A power statement.** Our own null had roughly 10% power, which is why the
   honest claim is "no detectable effect on this specification", not "no effect
   exists". Hold yourself to the standard we failed to hold ourselves to
   originally.

Without these, the PR will be closed with a pointer to this section. It is not a
judgement on your idea — it is that we no longer have a way to tell a real
result from a lucky one, and neither do you.

**Reviving the model layers by tuning constants.** `agent/models/`,
`agent/fusion/`, `agent/learning/` and `agent/adversarial/` emit degenerate
output ([§7](docs/COLLECTOR.md#7-what-the-layers-above-the-collector-do)). A
convergence score that spans 0.9977–0.9999 is not a score that needs
recalibrating; it is a computation that is not measuring anything. Rewrites with
a test that would fail on constant output are welcome. Nudged thresholds are
not.

**Paid or credentialed data sources.** Free and public only. It is the one
structural property this collector actually has.

**Aspirational README edits.** Do not soften the honesty section. A reader who
discovers the limitations themselves distrusts everything else in the repo; a
reader who is told up front trusts the rest. That candour is the differentiator
against every abandoned alt-data repo with a hopeful README.

---

## Ground rules for changes

**Every number in a document must be checkable.** If you state a figure, include
the query or script that produces it, and run it. Documents here pair each claim
with its reproduction command; keep that pattern.

**Row counts and green runs are not evidence.** The single most expensive lesson
in this codebase: it went from *healthy-looking and empty* to *healthy-looking
and full of constants*. Every table had rows, every job reported success, and
none of the numbers meant anything. If your change makes something report
success, prove it also makes it correct.

**Read `LESSONS.md` before touching training, loss, checkpointing or eval
code.** F-01 through F-13 are real production failures with prevention rules.
F-04 (data leakage in IC evaluation), F-12 (schema drift silently invalidating
every checkpoint) and F-13 (an operator timeout that marks a node failed but
cannot stop the thread, so it leaks memory until the box dies) are the ones most
likely to bite you again.

**Respect the layer boundaries.** `agent/tools/` fetches data and does no
feature logic. `agent/quant/` is stateless math. The LLM layer
(`agent/reasoning/`) narrates and never decides. Code in the wrong layer is
invisible debt.

**No secrets, ever.** Keys go in `.env`, which is gitignored. Never commit a
credential, an internal hostname, an IP, or customer data. If you add a
key-gated source, document the env var in `.env.example` with an empty value and
a registration link — never a real key.

---

## Practical

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install httpx jsonschema defusedxml numpy yfinance   # collector only
pip install -e ".[dev]"                                  # plus tests and lint

pytest tests/ -m "not slow and not live"
ruff check . && ruff format .
```

Commits should be atomic and citable — `fix: pad history arrays on checkpoint
resume (F-03 prevention)`, not `updates and fixes`.

Contributions are accepted under the Apache License 2.0 (see
[LICENSE](LICENSE)). By submitting a PR you affirm you have the right to license
your contribution under those terms.
