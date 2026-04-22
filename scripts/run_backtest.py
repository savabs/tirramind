#!/usr/bin/env python3
"""Phase 37.10 — Run walk-forward backtest on real historical data.

Loads instrument returns from PipelineStore, runs baseline strategies
(EqualWeight, BuyAndHold SPY, BuyAndHold 60/40) through MultiAssetWalkForward,
and reports: Sharpe, max drawdown, win rate, total return.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.pipeline.store import PipelineStore
from agent.quant.walkforward_runner import (
    build_default_strategies,
    generate_attribution_report,
    load_instrument_returns,
    run_walkforward,
)


def main() -> None:
    db_path = Path(".tirra_pipeline/pipeline.db")
    if not db_path.exists():
        print(f"ERROR: {db_path} not found. Run daily_collection first.")
        sys.exit(1)

    store = PipelineStore(str(db_path))

    # Build entity_id → ticker map
    entities = store.query_all_entities()
    eid_to_ticker: dict[str, str] = {}
    ticker_to_eid: dict[str, str] = {}
    instrument_classes: dict[str, str] = {}
    for e in entities:
        if e["entity_type"] != "instrument":
            continue
        meta = e.get("metadata") or {}
        ticker = meta.get("ticker", e["entity_id"])
        eid = e["entity_id"]
        eid_to_ticker[eid] = ticker
        ticker_to_eid[ticker] = eid
        instrument_classes[eid] = meta.get("asset_class", "unknown")

    # Use entity_ids as the ticker list (since that's what's in observations)
    entity_ids = list(eid_to_ticker.keys())
    print(f"Loading returns for {len(entity_ids)} instruments...")

    dates, returns = load_instrument_returns(store, entity_ids)
    T, N = returns.shape
    print(f"Loaded: {T} dates × {N} instruments ({dates[0]} to {dates[-1]})")

    # Map entity_ids to human-readable tickers for display
    display_names = [eid_to_ticker.get(eid, eid) for eid in entity_ids]

    # Build strategies — need to reference SPY by entity_id
    from agent.quant.backtest import (
        BuyAndHoldBenchmarkStrategy,
        EqualWeightStrategy,
    )

    spy_eid = ticker_to_eid.get("SPY")
    agg_eid = ticker_to_eid.get("AGG")
    tlt_eid = ticker_to_eid.get("TLT")

    strategies = [EqualWeightStrategy()]
    if spy_eid:
        strategies.append(BuyAndHoldBenchmarkStrategy({spy_eid: 1.0}))
    bond_eid = agg_eid or tlt_eid
    if spy_eid and bond_eid:
        strategies.append(BuyAndHoldBenchmarkStrategy({spy_eid: 0.6, bond_eid: 0.4}))

    # Walk-forward config: monthly test windows, ~1yr training
    min_train = 252
    test_size = 21
    step_size = 21

    print(
        f"\nRunning walk-forward: min_train={min_train}, test={test_size}, step={step_size}"
    )
    print(f"Strategies: {[s.name for s in strategies]}")
    print()

    results = run_walkforward(
        returns=returns,
        instrument_names=entity_ids,
        instrument_classes=instrument_classes,
        strategies=strategies,
        min_train=min_train,
        test_size=test_size,
        step_size=step_size,
    )

    # Report
    print("=" * 72)
    print("WALK-FORWARD BACKTEST RESULTS")
    print("=" * 72)
    for name, result in results.items():
        m = result.aggregate_metrics
        # Map strategy name for display
        display = name
        if spy_eid and spy_eid in name:
            display = name.replace(spy_eid, "SPY")
        if bond_eid and bond_eid in name:
            display = display.replace(bond_eid, "AGG" if agg_eid else "TLT")

        print(f"\n{display}")
        print("-" * 40)
        print(f"  Folds:         {len(result.folds)}")
        print(
            f"  Total Return:  {m.get('total_return', 0):.4f} ({m.get('total_return', 0)*100:.2f}%)"
        )
        print(f"  Sharpe Ratio:  {m.get('sharpe', float('nan')):.3f}")
        print(
            f"  Max Drawdown:  {m.get('max_drawdown', 0):.4f} ({m.get('max_drawdown', 0)*100:.2f}%)"
        )
        print(f"  Win Rate:      {m.get('win_rate', 0):.3f}")
        print(f"  Volatility:    {m.get('volatility', 0):.4f}")

    # Attribution report
    instrument_regions: dict[str, str] = {}
    for eid in entity_ids:
        for e in entities:
            if e["entity_id"] == eid:
                meta = e.get("metadata") or {}
                instrument_regions[eid] = meta.get("region", "unknown")
                break

    reports = generate_attribution_report(
        results, instrument_classes, instrument_regions, top_n=5
    )

    print("\n" + "=" * 72)
    print("TOP 5 INSTRUMENTS BY CONTRIBUTION (per strategy)")
    print("=" * 72)
    for name, report in reports.items():
        display = name
        if spy_eid and spy_eid in name:
            display = name.replace(spy_eid, "SPY")
        if bond_eid and bond_eid in name:
            display = display.replace(bond_eid, "AGG" if agg_eid else "TLT")
        print(f"\n{display}:")
        for eid, contrib in report.top_instruments:
            ticker = eid_to_ticker.get(eid, eid)
            print(f"  {ticker:12s}  {contrib:+.4f}")

    print()


if __name__ == "__main__":
    main()
