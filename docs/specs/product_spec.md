---
title: "Product spec: the belief outflow"
tags:
  - doc/specs
  - topic/strategy
  - status/current
date: 2026-09-30
---

# Product spec: the belief outflow

Derived rather than guessed. Every decision below traces to one of four
analyses — the forces the form must resolve (Alexander), the system's purpose
and leverage point (Meadows), the job being hired (Christensen), and what would
falsify it (Fitzpatrick). Where the analysis contradicted an earlier decision of
ours, the contradiction is recorded, because two of them were wrong.

---

## 1. Purpose

> **Reduce the gap between what someone believes about their portfolio and what
> is true.**

One sentence, and it is the test for every feature. A thing that does not narrow
that gap does not belong here, however interesting it is.

## 2. The leverage point

The system's stocks and flows:

```
STOCK   holdings              visible — every broker app shows it
STOCK   BELIEFS               invisible, never audited
FLOW    beliefs form          fast: news, tips, a friend, intuition
FLOW    beliefs get tested    ZERO
```

**The belief stock has an inflow and no outflow.** Beliefs accumulate for
decades and compound into a portfolio, and nothing ever removes a wrong one.

Adding a feedback loop where none exists sits near the top of Meadows' leverage
hierarchy — far above parameters. Most of this market competes at the bottom of
it: another ratio, another chart, a prettier allocation pie. **This product is
the missing outflow**, which is why it is not another portfolio tracker.

Everything below is in service of that one loop.

## 3. The forces the form must resolve

| | force | resolved by |
|---|---|---|
| **F1** | I don't know what I actually own | exposure: holdings, factors, chains |
| **F2** | I don't know if my decisions are working | picking cost (demoted — see §5) |
| **F3** | I can't tell signal from noise | every claim carries n, power, correction |
| **F4** | I'm afraid of a shock I didn't see coming | **ghost chains** |
| **F5** | I don't want to be told what to do | **no advice, ever** |
| **F6** | I don't have time | answer in under ten seconds |
| **F7** | I don't trust tools that sell me things | **publish the graveyard** |

F5 and F7 are the two nobody designs for, and they carry more weight here than
any feature.

**F5** is why this is audit and not advice. It is also what keeps us outside SEBI
Investment Adviser registration: *"your 4% position contributes 19% of your
risk"* is arithmetic; *"you should trim it"* is regulated advice. The constraint
and the product are the same thing.

**F7 reframed the scoreboard.** We had been treating *"41,208 chains tested, 3
survived"* as a statistics exhibit. It is not — it is **trust infrastructure**.
Every free tool in this category is a funnel to a broker, so the only credible
signal is showing the questions you got wrong. Nobody publishes their graveyard.

## 4. The job being hired

| | job | verdict |
|---|---|---|
| **J1** | help me not get blindsided | **lead with this** — ongoing, emotional, recurring |
| **J2** | tell me if I'm good at this | **demote** — identity-threatening, drives churn |
| **J3** | give me something to check my ideas against | **build toward** — this is the ledger |

Nobody hires "ghost pattern discovery". They hire *"tell me what I'm exposed
to."* The engine is the same; the framing decides whether anyone wants it.

### Correction this forced

We had **picking cost as the hook** — *"picking 7 names instead of the index
cost you −14.2%."* It serves **J2**. A tool whose front page tells you that you
are bad at investing is a tool you do not return to. It stays in the product,
behind a click, because F2 is real; it is not the headline.

It was also already weaker than it looked: measured across every rolling
one-year window the same book beat its index **223 of 493 times (45%), median
gap −0.2%**. The −14.2% was one window.

## 5. What ships, in order

Each step is useful alone and adds to the same page rather than replacing it.

### Step 1 — What you're exposed to *(F1, F6; components already built)*

Paste holdings, get exposure. Direct, by factor, and the structural findings
that survived measurement:

- **risk ≠ weight** — `GOLDBEES 6.0% of the money, 0.5% of the variance;
  IDEA 3.0% and 6.8%`
- **the hidden same-bet** — `HDFCBANK + NIFTYBEES correlate 0.66 and are 22.5%
  of the book` — two line items, two asset classes, one position
- **factor exposure** — Nifty explains 64%, beta 0.79

Built and tested: `agent/portfolio/{holdings,concentration,overlap,beliefs}.py`.

Structure is **explanation, never headline** — it is the only role it earned.
Measured: *"12 positions, 3 bets"* occurred **0 times in 200** real books;
effective bets ranges 1.53–3.93 across 150 books, i.e. nearly the same number
for everyone.

### Step 2 — The exposures you never chose *(F4, J1 — the differentiator)*

Ghost chains that terminate at the user's holdings.

```
RELIANCE   → crude          → producer-country stress
TCS, INFY  → USD revenue    → USD/INR
```

Not "here are 3 chains in the world" but **"here is the chain that reaches your
18% position."** Same engine, different product: a chain in the abstract is a
curiosity, a chain through your holdings is urgent.

This is the only finding that passes the surprise test, and the reason is
structural: **the user chose the portfolio, so facts about its structure cannot
surprise them.** A chain is something they did not choose and cannot see.

### Step 3 — The scoreboard *(F3, F7)*

```
CHAINS TESTED        41,208
UNTESTABLE           38,904   (n < 30 events, or < 20 independent clusters)
TESTED                2,304
SURVIVED BH @ 0.05        3
MEDIAN POWER           6.1%
```

**The denominator is the product.** If three survive you have three findings; if
zero survive you have a better artifact — *"we tested 41,208 chains across 34
sources and none survived correction, and here is the power analysis showing
what we could have detected."*

This is the only component that gets **stronger when the answer is no**, which
is what makes it the trust layer rather than a liability. Every previous headline
in this project died when measured honestly; this one cannot die that way,
because the honesty *is* the headline.

### Step 4 — The belief ledger *(J3, and the actual leverage point)*

The user states what they believe. It gets tested, and re-tested as data
arrives. Every new hypothesis is corrected for **all the ones they already
tried** — which is the thing they structurally cannot do for themselves, because
it requires an honest record of their own failures.

This is the missing outflow from §2, the only component that compounds, and the
only one that earns a subscription rather than a one-off visit.

## 6. What we are not building

- **Advice, forecasts, allocation suggestions.** Violates F5; regulated; and it
  would require being right about the future, which trades away the entire
  liability profile that makes this viable.
- **A prediction product.** Measured: 3 genuine cross-domain joins out of 561
  source pairs. The data does not support it and fixing the engine did not
  change it.
- **Anything at the bottom of the leverage hierarchy** — more ratios, more
  charts. That is where the competition already is, and it is free.

## 7. What would falsify this

Three claims, each cheap, in increasing order of importance:

1. **People will paste holdings into a page.** Ship step 1 and count.
2. **Chain findings actually surprise them.** Show five people. Watch their
   faces. **This decides everything** — and it is the test we have skipped three
   times by reasoning about surprise instead of measuring it. Two headlines died
   only because we finally measured them.
3. **They come back.** Only step 4 creates a reason to.

If (2) fails, the thesis fails, and it costs an afternoon to find out.

## 8. The standing risk

Every honest measurement in this project has killed the headline it was meant to
support. That is not a flaw to engineer around — it is the finding, and it is
the reason the scoreboard has to be the trust layer rather than a footnote.

The counterweight to systems thinking is that it makes everything feel connected
and you can spend months building an elegant framework nobody buys. Two prior
projects died that way. **Ship step 1, show one person, listen.** Meadows
identifies the leverage point; only a user says whether it matters.

## Related

- [[verification_business_plan]] — the B2B framing this supersedes for consumer
- [[what_tirramind_is_for]] — why the prediction premise stayed unproven
- [[cot_null_result]] — the graveyard, done properly, and the credential
