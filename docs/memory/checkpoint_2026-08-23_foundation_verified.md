---
title: "Checkpoint 2026-08-23 — TirraMind Foundation + Cross-Domain Proof"
tags:
  - doc/checkpoint
  - phase/1
  - topic/signals
  - status/active
---

# Checkpoint: TirraMind Foundation + Cross-Domain Proof

**Date:** 2026-08-23

## Summary

TirraMind is now **proven running and firing live signals** — not theoretical.

| Milestone | Result |
|---|---|
| Python 3.12 venv + `[dev,quant,ml]` deps | ✅ |
| Core imports | ✅ |
| Fast test suite | ✅ **10,461 passed** / 59 env/key-gated fails / 14 skipped |
| Live signal families | ✅ `gov_contracts` (US awards), `cftc` (commodity positioning), `ais_vessel` (shipping) — all real free data |
| Cross-domain links | ✅ **12,271 entity links**; real contract-recipient companies linked via `awarded_by` / `operates_in` |
| Cross-domain entity footprint | ✅ US country seen across 5 macro domains (sovereign debt, energy, PMI, capital flows, CB balance) |
| Contract-recipient × other-domain overlap | ⚠️ **0 today** — the actual Tender Alpha gap to build |

## What changed

1. **`pyproject.toml`** — added `[tool.setuptools.packages.find]` so editable install works.
2. **`docs/signals_primer.md`** — plain-language inventory of all 5 signal families + live-fire proof.
3. **`docs/memory/project_metrics.md`** — updated canonical active metrics.
4. **`docs/cross_domain_signal_proof.md`** — concrete cross-domain proof (new).
5. **`docs/memory/checkpoint_2026-08-23_foundation_verified.md`** — this checkpoint.

## The honest bottom line

- **TirraMind runs.** 10,461 test passes in the fast suite; the signal machinery works.
- **Live signals work:** contracts (USASpending), CFTC positions, AIS shipping all return real data.
- **The cross-domain graph is real:** 12,271 typed entity links, and contract recipients (Sikorsky, Fluor, Georgia Tech, Lawrence Livermore, UC Regents, etc.) already have `awarded_by` / `operates_in` edges.
- **The gap is precisely the Tender Alpha edge:** contract-recipient companies do *not* yet overlap with shipping/insider/GDELT in any temporal window. That is the actual feature to build: connect `gov_contract` recipients → same company seen in `ais_vessel`, `insider_filings`, `gdelt`, `form144`.

## Next concrete step

1. Build the **contract-recipient → other-domain overlap** (the Tender Alpha moat): a small deterministic resolver that normalizes the `gov_contracts` recipient companies into canonical entities, then queries the same canonical company in other signal domains within a 24h–7d window.
2. Then productize the 1–3 signal families.

## Related
- [[signals_primer]]
- [[project_metrics]]
- [[revenue_plan_2026-05-08]]