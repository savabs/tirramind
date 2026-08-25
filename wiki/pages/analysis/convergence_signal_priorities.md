---
title: Convergence Signal Priorities
tags:
  - doc/wiki
  - topic/convergence
  - layer/feature-engineering
type: analysis
summary: Ranked view of which next signal families add the most value to convergence before further schema expansion.
status: active
source_docs:
  - [[convergence_signal_expansion]]
  - [[7b-AI_internet_infrastructure]]
  - [[7b-AO_supply_chain_monitor]]
  - [[7b-AM_consumer_sentiment]]
  - [[convergence_detection]]
updated_on: 2026-04-05
---

# Convergence Signal Priorities

The next useful work is not broad schema inflation. It is converting a few
high-value, weakly represented evidence surfaces into real convergence signals.

## Main conclusion

Best next signal families:

1. Internet infrastructure outages and censorship
2. Power and grid stress
3. DeFi liquidity and rotation
4. Supply-chain price breadth
5. Consumer expectation divergence

Why this ordering:

- These surfaces add new causal information rather than duplicating current
  survey or macro fields.
- The top three are still underrepresented in convergence despite strong tool or
  research groundwork.
- They improve world-model evidence more than adding more metadata to emitted
  convergence events.

## Priority notes

### Internet infrastructure

- Highest orthogonality and strong geopolitical relevance.
- `internet_infrastructure` is still only a stub extractor in convergence.
- Best signal shapes: outage score, BGP visibility drop, censorship rate,
  messaging-block incidents, multi-country disruption breadth.

### Power and grid stress

- Public, operational, high-frequency physical data.
- Best signal shapes: demand-forecast gap, DA-RT spread, zone breadth, fuel-mix
  stress, peak-load anomaly.

### DeFi liquidity and rotation

- Adds 24/7 liquidity state not covered well elsewhere.
- Best signal shapes: stablecoin supply delta, chain TVL rotation, DEX panic
  ratio, protocol drawdown breadth.

### Supply-chain price breadth

- Already partly represented, but breadth and pass-through matter more than more
  raw sector rows.
- Best signal shapes: accelerating-sector breadth, import-domestic spread,
  PPI-CPI gap.

### Consumer expectation divergence

- Already partly represented.
- Best next additions are diffusion and divergence, not many extra line items.

## Explicit deprioritizations

- Food security: keep as a slow structural prior, not a near-term convergence
  focus.
- FCC / spectrum: free programmatic surface remains too weak to justify work.

## Recommended next build order

1. Replace the `internet_infrastructure` stub with real outage and censorship
   evidence extraction.
2. Add a real `power_grid` extractor.
3. Add a `defi_flows` extractor.
4. Expand `supply_chain_prices` breadth-style signals.
5. Expand consumer divergence signals selectively.

See also: [[pages/roadmap/current_phases]], [[pages/architecture/system_overview]]

---

## Related

- [[convergence_detection]]
- [[convergence_backtest]]
- [[convergence_template_expansion]]
- [[convergence_signal_expansion]]
- [[convergence_napm_refresh]]
- [[convergence_backtest_fast_mode]]
- [[convergence_backtest_score_cache]]
- [[convergence_template_batch2]]
- [[convergence_audit_pre_worldmodel]]
