---
title: "Plan 2026-09-06 — 'No naked numbers'"
tags:
  - doc/checkpoint
  - topic/architecture
  - status/current
---

# Plan 2026-09-06 — "No naked numbers"

Supersedes `revenue_plan_2026-05-08.md` and the interconnection-commercialisation
draft. Written after ~9 months, two withdrawn products, and one session that
killed six more directions with evidence.

## 1. What we have actually been trying to do

Strip away the domains (queues, options, briefs, entity graphs) and every
project here did the same thing:

> Take a number people act on, and find out how wrong it is — then say so.

- Brief product: 0 of 51 hypotheses survived BH. We published that and withdrew.
- Queue attrition: a nominal 80% interval covered 62.6%. We published that.
- Calibration: isotonic looked better and was worse than nothing. Sigmoid
  fixed it (slope 1.004). We measured it before believing it.
- Coverage test on equities: caught that the design would produce a false
  positive from mean-estimation noise, before running it.

The one skill this project demonstrably has is **measuring uncertainty
honestly, in finance, with code that is fast and correct.** That is the
business. Not any single domain.

## 2. The vision (does not change with time)

**Every number in finance ships with an honest, tested interval.**

Prices, greeks, VaR, forecasts, portfolio ranges — none of them should be
handed to a human without (a) an uncertainty band and (b) a record of how
often that band has been right. Winners need it to size up; losers need it to
survive. Bull or bear, 2026 or 2046, the need is identical. The models change;
the vision doesn't.

## 3. Why this fits the constraints

| Constraint | How it's met |
|---|---|
| Software/IP product, not data hoarding | A PyTorch library + a hosted validation service. Users bring their own data. We never redistribute a price row. |
| Moat = quality and power | Correctness (published golden tests vs QuantLib), speed (autograd greeks, GPU kernels), and honesty (every model ships with measured coverage). None of the competitors publish coverage. |
| Field accepts new founders | Developers install libraries from GitHub without caring who wrote them. `pip install` has no sales cycle. |
| No selling, no convincing | Distribution = PyPI, GitHub, one honest README, public scorecards. Support is the only human contact, and the owner is willing to do support. |
| Boring but powerful | Model validation is the most boring job in finance. It is also mandatory (SR 11-7 at banks) and unavoidable everywhere else. |
| Needed no matter the situation | Risk measurement is counter-cyclical-proof: demand rises in crashes and doesn't vanish in booms. |
| Interests: finance, maths, AI, low-level | Stochastic calculus, autograd, CUDA kernels, calibration theory, ML monitoring. All of it. |
| Adoption first, training later | Hosted scorecards accumulate *calibration records across models and regimes*. That is a dataset nobody has, and the only kind we should ever hoard: metadata about how wrong models are, not the market data itself. |

## 4. The product, concretely

Two parts of one thing. Ship in this order.

### Part A — `voltorch` (exists, published, 39 tests green)

Differentiable pricing and volatility surfaces in PyTorch. Heston, Bates,
Merton, VG via Fourier-COS; SABR, SVI; rough Bergomi; GBM/Heston SDEs.

Honest position: TorchQuant, fastvol, pfhedge, fast-vollib occupy this space.
We do not win on "first." We win on:
1. **Numerical correctness under stress** — the branch-cut fix, the SABR ATM
   0/0, Durrleman butterfly checks, total-variance interpolation. Publish a
   torture-test page where every competitor is run through the same cases.
2. **Autograd greeks with published error bounds** vs finite-difference and
   closed-form, per model, per regime.
3. **Every calibration returns an interval, not a point** — Hessian-based or
   bootstrap parameter uncertainty is the default return type. This is the
   feature no competitor has and the hook into Part B.

### Part B — the coverage engine (the actual IP; ~70% built in pieces)

A library and a hosted service that answers one question for any forecast
ledger: **how often was your interval right, and can your data even tell?**

What it does that scikit-learn / properscoring / Evidently / NannyML don't:
- Time-series-aware: overlapping horizons → **effective n**, block bootstrap
  CIs on every coverage number. (The pseudo-replication trap, built in.)
- Separates mean error from covariance error so a coverage failure is
  attributed correctly. (The false-positive trap, built in.)
- Calibration slope, Brier skill, bin coverage, PIT histograms — with
  nested-holdout hygiene so isotonic can't fake it.
- **Append-only, content-addressed ledgers** — a forecast, once logged, can't
  be edited. Scorecards are dated and tamper-evident. (The archive mechanism,
  reused.)
- Finance-native inputs: return intervals, VaR exceedances, option-implied
  vs realised, portfolio P10/P50/P90.

Pieces already in this repo: calibration code, permutation/dispersion tests,
content-addressed snapshot store, the 1,636-row prediction ledger, the
coverage-test protocol in scratchpad.

Known adjacent competitors to check for one hour before starting, not a
week: Evidently AI, NannyML, Arize (general ML monitoring, not finance,
not interval-coverage-with-effective-n). Metaculus/GJ (community forecasting,
not a tool). If one of them already does time-series-aware coverage with
block-bootstrap CIs for finance, Part B narrows to a plugin for it.

## 5. How it makes money (honest)

Self-serve only. Never a sales call.

1. **Public scorecards are free.** Post a ledger, get a dated public page:
   "Model X: 80% band covered 71% [64, 78], n_eff = 31." This is the adoption
   hook and the credential. It is also how we scored ourselves.
2. **Private ledgers + CI integration: paid.** $29–79/mo, Paddle checkout,
   key in email. Buyer: small quant shops, fintech devs, energy/betting
   desks, ML teams in finance who need SR 11-7-style evidence without a
   vendor. They already pay Portfolio Visualizer / Mezzi money for less.
3. **voltorch stays free and Apache-2.0.** It is the funnel and the reputation.
   A "pro" build (GPU kernels, exotics, signed wheels) is a possible year-2
   tier; do not build it in the first 3 months.

Arithmetic: $100k/yr ≈ 170 seats at $49/mo. That is not a 3-month outcome.
Realistic: 0–5 paying users at month 3, and the decision to continue is made
on **adoption signals** (below), not revenue.

## 6. The 3-month plan

Each phase has a gate. Passing a gate is the only way to start the next.

### Month 1 — make voltorch undeniable (weeks 1–4)
- [ ] PyPI publish (owner enters token; already blocked on this).
- [ ] Golden tests vs QuantLib for every model, committed, CI-run.
- [ ] Torture-test page: 20 pathological cases (deep OTM, T→0, ρ→±1, ATM
      SABR, negative rates) run through voltorch + TorchQuant + fastvol.
      Publish the table. Say where we lose.
- [ ] Calibration returns intervals by default (Hessian + bootstrap).
- [ ] One README, one 60-second example, zero marketing words.
- **Gate:** 100 GitHub stars or 500 PyPI downloads/week from strangers, or
  one unsolicited issue from someone with a real problem. Any one of three.
  If none by day 30: ship Part B anyway (it's the real IP), but assume the
  developer channel is slow, not dead.

### Month 2 — the coverage engine as a library (weeks 5–8)
- [ ] Extract from tirramind: calibration, dispersion test, ledger store,
      block bootstrap, effective-n. New package, one name, Apache-2.0.
- [ ] Run the corrected equity coverage test through it as the first public
      case study (the one in scratchpad, with the mean/cov separation).
      Publish the result *whatever it says* — "cannot resolve at 12m,
      n_eff = 25" is a valid and honest finding and a demonstration.
- [ ] Re-score the queue ledger through it. Apply the sigmoid fix as
      `logistic-v2`. Publish before/after coverage.
- [ ] Apply it to voltorch: implied vs realised vol coverage on SPY options
      (free EOD data, own use only, no redistribution).
- **Gate:** the library reproduces every published number in this project
  (4.30× SD, 62.6%, slope 1.004) as golden tests. If a port changes an
  answer, the public site is wrong and it must fail loudly.

### Month 3 — hosted scorecards (weeks 9–12)
- [ ] One Hetzner box (already own it — api.tirramind.com resolves, nothing
      listening). POST a ledger, GET a scorecard page. Public by default.
- [ ] Private ledgers behind a key. Paddle checkout — reuse
      `brief_subscription/` after fixing the four latent bugs listed in
      the previous plan (non-atomic save, default tier, refunds unhandled,
      unbounded Content-Length). These are now real, not theoretical.
- [ ] Our own three ledgers on it, dated, public: queue predictions,
      equity coverage, option vol coverage. We are user zero and the proof.
- **Gate:** one stranger posts a ledger. Not a payment — a ledger. That is
  the demand signal that costs the user something (their data) and costs us
  nothing to observe.

## 7. What we do not do

- No new domain until Part B has one external user.
- No market-data redistribution, ever. Users bring data.
- No sales calls, no discovery interviews, no cold outreach. Public work only.
- No teams/SSO/enterprise anything.
- No CI beyond "tests pass on push." (User's call, already made.)
- No re-deriving the six killed directions. They are dead.

## 8. Kill criteria (decided now, so they can't be moved later)

- Day 90: zero strangers have starred, downloaded, opened an issue, or
  posted a ledger → the developer channel does not work for this owner
  either, and the honest conclusion is that this person should join a team
  that has distribution rather than build a fourth solo product.
- Any point: a competitor ships time-series-aware coverage with effective-n
  and block-bootstrap CIs → Part B becomes a contribution to them, not a
  product.

## 9. Resume from here

- voltorch: `~/Projects/voltorch`, 39 passed 2 skipped, github.com/savabs/voltorch.
- Coverage-test corrected protocol: session scratchpad (copy into
  `docs/research/equity_coverage_protocol.md` first thing).
- Sigmoid fix verified, not applied: needs `logistic-v2` model id.
- Queue archive: 6/7 ISOs, both runners, self-committing. Touch nothing.
- Paddle: products still need archiving in the dashboard (owner only).
