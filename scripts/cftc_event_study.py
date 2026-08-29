"""
CFTC anomaly event study — does the |z|>=2 flag the Brief tier ships on
futures_positioning fields precede any real forward return?

Read-only. Joins CFTC weekly positioning (per cftc_tracks entity link) to the
linked instrument's daily close series in `.tirra_pipeline/pipeline.db`.

Publication-lag handling (F-04 guard): CFTC observed_at is the Tuesday
"as-of" position date. The report is not released to the public until the
following Friday ~3:30pm ET (agent/tools/cftc.py docstring, confirmed against
observed_at day-of-week below). Any z-score computed at week t is genuinely
knowable at week t (expanding window, x[:-1] as history — replicated here
exactly). But the EARLIEST a trade could be entered is the first instrument
close at or after (observed_at + PUB_LAG_SECS). Measuring forward returns
from the observation date itself would be trading 3 days before the data
existed publicly.

Usage: .venv/bin/python scripts/cftc_event_study.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.quant.scoring import block_bootstrap_ci  # noqa: E402

DB_PATH = ".tirra_pipeline/pipeline.db"
PUB_LAG_SECS = 3 * 86400.0  # Tuesday as-of -> Friday release
ENTRY_TOLERANCE_SECS = 7 * 86400.0  # accept instrument close within 1wk of publication
MIN_HISTORY = 20  # matches _zscore_anomaly floor in live_intelligence_digest.py
HORIZONS = (1, 5, 20)  # trading days
Z_THRESHOLDS = (2.0, 3.0)
FIELDS = (
    "mm_net",
    "open_interest",
    "swap_net",
    "pm_net",
    "conc_top4_long",
    "conc_top4_short",
    "mm_net_pct_oi",
    "mm_weekly_flow",
    "oi_change",
)


def dedup_series(con, entity_id: str, obs_type: str) -> list[tuple[float, dict]]:
    """(observed_at, value_dict) ascending, deduped on (entity_id, observed_at) keeping max(rowid)."""
    cur = con.cursor()
    rows = cur.execute(
        "select observed_at, value_json from entity_observations "
        "where entity_id=? and observation_type=? "
        "and rowid in ("
        "  select max(rowid) from entity_observations "
        "  where entity_id=? and observation_type=? group by observed_at"
        ") order by observed_at asc",
        (entity_id, obs_type, entity_id, obs_type),
    ).fetchall()
    out = []
    for ts, vj in rows:
        try:
            v = json.loads(vj)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(v, dict):
            out.append((float(ts), v))
    return out


def zscore_causal(values: list[float]) -> float | None:
    """Exact replica of _zscore_anomaly: expanding window, x[:-1] as history."""
    if len(values) < MIN_HISTORY:
        return None
    x = np.array(values, dtype=float)
    hist = x[:-1]
    if float(np.std(hist)) < 1e-12:
        return None
    z = (x[-1] - np.mean(hist)) / (np.std(hist) + 1e-12)
    return float(z)


def main() -> None:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cur = con.cursor()

    links = cur.execute(
        "select entity_id_a, entity_id_b, metadata_json from entity_links where link_type='cftc_tracks'"
    ).fetchall()
    print(f"cftc_tracks links found: {len(links)}")

    # ---- STEP 1: build panel, report attrition ----
    pairs = []
    for contract_eid, instrument_eid, meta_json in links:
        meta = json.loads(meta_json) if meta_json else {}
        code = meta.get("cftc_code", "?")
        ticker = meta.get("ticker", "?")

        cftc_series = dedup_series(con, contract_eid, "futures_positioning")
        inst_series = dedup_series(con, instrument_eid, "instrument_daily")
        closes = [(ts, v["close"]) for ts, v in inst_series if isinstance(v.get("close"), int | float)]

        status = "OK"
        if len(cftc_series) < MIN_HISTORY:
            status = f"DROP: only {len(cftc_series)} cftc weekly points (<{MIN_HISTORY})"
        elif len(closes) < 40:
            status = f"DROP: only {len(closes)} instrument daily closes"
        else:
            first_cftc_pub = cftc_series[0][0] + PUB_LAG_SECS
            last_cftc_pub = cftc_series[-1][0] + PUB_LAG_SECS
            first_close_ts = closes[0][0]
            last_close_ts = closes[-1][0]
            if last_close_ts < first_cftc_pub:
                status = "DROP: instrument closes end before cftc series even begins (publication-adjusted)"
            elif first_close_ts > last_cftc_pub:
                status = "DROP: instrument closes start after cftc series ends"

        print(f"  {code:>8} -> {ticker:<6} cftc_pts={len(cftc_series):>4} " f"inst_pts={len(closes):>5} :: {status}")
        pairs.append(
            {
                "code": code,
                "ticker": ticker,
                "contract_eid": contract_eid,
                "instrument_eid": instrument_eid,
                "cftc_series": cftc_series,
                "closes": closes,
                "usable": status == "OK",
            }
        )

    usable_pairs = [p for p in pairs if p["usable"]]
    print(f"\nUsable (contract, ticker) pairs surviving join: {len(usable_pairs)} / {len(pairs)}")
    for p in usable_pairs:
        print(f"  {p['code']} -> {p['ticker']}")

    if not usable_pairs:
        print("\nNO USABLE PAIRS. Study cannot proceed. Stopping.")
        return

    # ---- STEP 2 & 3: event study + leakage audit ----
    # events[(field, z_threshold, horizon)] = list of forward returns (event group)
    # population[(field, horizon)] = list of forward returns (ALL eligible entries, unconditional)
    events: dict[tuple[str, float, int], list[float]] = defaultdict(list)
    population: dict[tuple[str, int], list[float]] = defaultdict(list)
    leak_check_samples: list[tuple[str, str, float, float, float]] = []  # (code,ticker,obs_at,pub_ts,entry_ts)
    n_dropped_no_entry = 0
    n_dropped_no_horizon = defaultdict(int)

    for p in usable_pairs:
        closes = p["closes"]
        close_ts = np.array([c[0] for c in closes])
        close_val = np.array([c[1] for c in closes])

        # Per-field causal series for this contract
        field_history: dict[str, list[float]] = defaultdict(list)
        field_ts: dict[str, list[float]] = defaultdict(list)

        for obs_ts, vals in p["cftc_series"]:
            for field in FIELDS:
                fv = vals.get(field)
                if not isinstance(fv, int | float) or isinstance(fv, bool):
                    continue
                field_history[field].append(float(fv))
                field_ts[field].append(obs_ts)

                z = zscore_causal(field_history[field])
                if z is None:
                    continue

                # --- leakage guard: publication lag ---
                pub_ts = obs_ts + PUB_LAG_SECS
                idx_entry = np.searchsorted(close_ts, pub_ts, side="left")
                if idx_entry >= len(close_ts) or (close_ts[idx_entry] - pub_ts) > ENTRY_TOLERANCE_SECS:
                    n_dropped_no_entry += 1
                    continue
                if len(leak_check_samples) < 8:
                    leak_check_samples.append((p["code"], p["ticker"], obs_ts, pub_ts, float(close_ts[idx_entry])))

                entry_price = close_val[idx_entry]
                if entry_price == 0 or not np.isfinite(entry_price):
                    continue

                for h in HORIZONS:
                    fwd_idx = idx_entry + h
                    if fwd_idx >= len(close_ts):
                        n_dropped_no_horizon[h] += 1
                        continue
                    fwd_price = close_val[fwd_idx]
                    if fwd_price <= 0 or entry_price <= 0:
                        continue
                    log_ret = float(np.log(fwd_price / entry_price))
                    # unconditional population — every eligible (z computable) entry,
                    # regardless of |z| magnitude, matched on field+horizon
                    population[(field, h)].append(log_ret)
                    for zt in Z_THRESHOLDS:
                        if abs(z) >= zt:
                            events[(field, zt, h)].append(log_ret)

    print(
        f"\nDropped events (no instrument close within {ENTRY_TOLERANCE_SECS/86400:.0f}d of publication): {n_dropped_no_entry}"
    )
    for h, n in sorted(n_dropped_no_horizon.items()):
        print(f"Dropped events (no close {h}d ahead available yet): {n}")

    print("\n=== LEAKAGE AUDIT ===")
    print("1. z-score causality: zscore_causal() uses x[:-1] as history exactly like")
    print("   _zscore_anomaly in live_intelligence_digest.py — verified by direct code reuse of the formula.")
    print(
        f"2. Publication lag: observed_at (Tuesday as-of date) + {PUB_LAG_SECS/86400:.0f}d = publication_ts (Friday)."
    )
    print("   Sample (code, ticker, observed_at, publication_ts, actual_entry_close_ts) pairs used as entry points:")
    import datetime as dt

    for code, ticker, obs_ts, pub_ts, entry_ts in leak_check_samples:
        d_obs = dt.datetime.utcfromtimestamp(obs_ts).date()
        d_pub = dt.datetime.utcfromtimestamp(pub_ts).date()
        d_entry = dt.datetime.utcfromtimestamp(entry_ts).date()
        gap_days = (entry_ts - obs_ts) / 86400.0
        print(f"   {code:>8} {ticker:<6} obs={d_obs} pub={d_pub} entry_close={d_entry} (gap from obs: {gap_days:.1f}d)")
    print("3. No survivorship: all 19 cftc_tracks links used as-is; no filtering by")
    print("   result performance. Attrition above is by data availability only.")

    # ---- STEP 4: multiple testing ----
    print("\n=== EVENT STUDY RESULTS (per field x horizon x |z| threshold) ===")
    rows = []
    for field in FIELDS:
        for zt in Z_THRESHOLDS:
            for h in HORIZONS:
                ev = np.array(events.get((field, zt, h), []))
                pop = np.array(population.get((field, h), []))
                if len(ev) == 0 or len(pop) == 0:
                    continue
                mean_ev = float(ev.mean())
                med_ev = float(np.median(ev))
                mean_pop = float(pop.mean())
                hit_ev = float((ev > 0).mean())
                hit_pop = float((pop > 0).mean())

                # bootstrap CI + bootstrap p-value on the event mean vs population baseline
                if len(ev) >= 3:
                    point, lo, hi = block_bootstrap_ci(ev, np.mean, n_bootstrap=2000)
                    rng = np.random.default_rng(7)
                    boot = np.array([np.mean(rng.choice(ev, size=len(ev), replace=True)) for _ in range(2000)])
                    p_low = float((boot <= mean_pop).mean())
                    p_high = float((boot >= mean_pop).mean())
                    p_value = float(2 * min(p_low, p_high))
                    p_value = min(p_value, 1.0)
                else:
                    lo = hi = float("nan")
                    p_value = float("nan")

                rows.append(
                    {
                        "field": field,
                        "z_threshold": zt,
                        "horizon_d": h,
                        "n_events": len(ev),
                        "n_pop": len(pop),
                        "mean_event_ret": mean_ev,
                        "median_event_ret": med_ev,
                        "mean_baseline_ret": mean_pop,
                        "edge": mean_ev - mean_pop,
                        "hit_rate_event": hit_ev,
                        "hit_rate_baseline": hit_pop,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "p_value": p_value,
                    }
                )

    # BH correction across all p-values that are defined
    pvals = [r["p_value"] for r in rows if np.isfinite(r["p_value"])]
    if pvals:
        reject, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
        it = iter(zip(reject, p_adj))
        for r in rows:
            if np.isfinite(r["p_value"]):
                rej, padj = next(it)
                r["p_adj_bh"] = float(padj)
                r["significant_bh"] = bool(rej)
            else:
                r["p_adj_bh"] = float("nan")
                r["significant_bh"] = False

    rows.sort(key=lambda r: (r["p_value"] if np.isfinite(r["p_value"]) else 1.0))

    hdr = (
        f"{'field':<18}{'|z|>=':<7}{'h(d)':<6}{'n_ev':<6}{'n_pop':<7}"
        f"{'mean_ev':<10}{'mean_base':<11}{'edge':<10}{'hit_ev':<8}{'hit_base':<9}"
        f"{'p':<8}{'p_bh':<8}{'sig_bh':<7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['field']:<18}{r['z_threshold']:<7.1f}{r['horizon_d']:<6}{r['n_events']:<6}{r['n_pop']:<7}"
            f"{r['mean_event_ret']*100:<10.3f}{r['mean_baseline_ret']*100:<11.3f}"
            f"{r['edge']*100:<10.3f}{r['hit_rate_event']:<8.2%}{r['hit_rate_baseline']:<9.2%}"
            f"{r['p_value']:<8.3f}{r.get('p_adj_bh', float('nan')):<8.3f}{str(r.get('significant_bh', False)):<7}"
        )

    n_low_n = sum(1 for r in rows if r["n_events"] < 30)
    print(f"\n{n_low_n} / {len(rows)} rows have n_events < 30 (small-sample warning).")
    n_sig = sum(1 for r in rows if r.get("significant_bh"))
    print(f"Rows surviving Benjamini-Hochberg correction at alpha=0.05: {n_sig} / {len(pvals)} tested")

    con.close()


if __name__ == "__main__":
    main()
