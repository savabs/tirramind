---
title: "Research: Convergence Backtest NAPM Refresh"
tags:
  - doc/research
  - phase/7c
  - topic/convergence
---

# Research: Convergence Backtest NAPM Refresh

## Goal
Replace the fragile NAPM dependency in the convergence macro backtest with a maintained free macro proxy that is consistent with the repo's existing global-macro research and current tool surface.

## Current Architecture
- `agent/convergence/backtest.py` defines `FRED_SERIES`, a fixed mapping from historical macro series to convergence `signal_id` values.
- The macro backtest currently uses FRED-only historical series plus yfinance target assets.
- `NAPM` is mapped to `pmi.us.manufacturing` with `level_below_50` semantics.
- Existing repo research already documents that direct ISM/NAPM programmatic access is not a durable free source and that OECD CLI is the preferred global replacement for free leading-indicator coverage.

## Observations
- The backtest can still run without immediate failure because the rest of the signal set is sufficient, but keeping `NAPM` in the canonical series list creates a maintenance hazard.
- The replacement should preserve the statistical role of a forward-looking macro-momentum signal rather than just swapping names.
- The lowest-risk change is to keep the backtest FRED-native and replace `NAPM` with a maintained FRED series that behaves as a leading or near-leading growth proxy.
- Candidate FRED-native replacements already aligned with repo philosophy:
  - `USSLIND` — Leading Index for the United States, monthly, maintained, explicitly designed as a leading indicator.
  - `INDPRO` — Industrial production, monthly, broad activity proxy but more coincident than leading.
  - `PERMIT` already exists in the set, so it should not be reused as the direct replacement.
- Existing research relevant to this step:
  - `[[convergence_backtest]]` — backtest architecture and historical data constraints.
  - `[[7b-AA_global_pmi]]` — NAPM/ISM licensing constraint and OECD CLI alternative.

## Risks
- Changing the series alters the macro baseline and may shift detection counts or strategy metrics.
- If the replacement changes signal direction semantics, template matching may degrade silently unless tests pin behavior.
- Renaming the signal id would create unnecessary blast radius across templates and tests; prefer keeping `pmi.us.manufacturing` as the semantic slot unless there is a compelling reason to retaxonomize.

## Data Requirements
- Monthly historical series with long history and live, free availability.
- Signal must support a simple direction rule suitable for convergence evidence construction.
- Series must be available through the existing FRED fetch path to avoid widening scope.

## Math/Algorithm Survey
- The backtest uses directional evidence, not raw forecasting regression.
- For this slot we only need a scalar macro-momentum indicator with a monotone contraction interpretation.
- Preferred interpretation:
  - `USSLIND` below 0 implies below-trend / contraction pressure.
  - Direction rule: `level_below_0` mirrors the current threshold-based use of NAPM below 50.
- This keeps the backtest logic simple and avoids adding normalization or bespoke transforms.

## Step-Local References
- `[[convergence_backtest]]`
- `[[7b-AA_global_pmi]]`
- `agent/convergence/backtest.py`
- `tests/test_convergence_backtest.py`

---

## Related

- [[convergence_napm_refresh_spec|Spec: Convergence Napm Refresh]]
- [[convergence_detection]]
- [[convergence_backtest]]
- [[convergence_template_expansion]]
- [[convergence_signal_expansion]]
- [[convergence_backtest_fast_mode]]
- [[convergence_backtest_score_cache]]
- [[convergence_template_batch2]]
- [[convergence_audit_pre_worldmodel]]
