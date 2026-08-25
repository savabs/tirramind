"""Training-time observation subsampling (modal balance).

Implements [[n1_outcome_input_train_doctrine]] §3: cap noisy modalities,
always keep raw intelligence source_tools at 100%.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

# N1+N4 raw sensors — never thin during subsample (floor at 100%).
RAW_SOURCE_TOOLS_FLOOR: frozenset[str] = frozenset(
    {
        "ais_vessel",
        "cftc",
        "cftc_derived",
        "capital_flows",
        "central_bank_balance",
        "comtrade",
        "energy_supply",
        "food_security",
        "gov_contracts",
        "insider_filings",
        "political_risk",
        "sanctions_monitor",
        "satellite_activity",
        "sovereign_debt",
        "transport_throughput",
        "weather_alerts",
        "whale_alert",
    }
)

_SUBSAMPLE_SEED = 42


def apply_training_obs_subsample(
    observations: list[dict[str, Any]],
    *,
    gdelt_subsample_frac: float = 1.0,
    defi_subsample_frac: float = 1.0,
    raw_source_floor: frozenset[str] | None = None,
    seed: int = _SUBSAMPLE_SEED,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Thin GDELT / DeFi TVL rows; keep floor tools and all other obs.

    Returns (sorted_observations, stats_dict).
    """
    floor = raw_source_floor if raw_source_floor is not None else RAW_SOURCE_TOOLS_FLOOR
    thin_gdelt = 0.0 < gdelt_subsample_frac < 1.0
    thin_defi = 0.0 < defi_subsample_frac < 1.0

    rng = random.Random(seed)
    kept: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()

    if not thin_gdelt and not thin_defi:
        stats["kept_total"] = len(observations)
        stats["dropped_total"] = 0
        stats["other_kept"] = len(observations)
        return observations, dict(stats)

    for o in observations:
        source = (o.get("source_tool") or "").strip()
        obs_type = (o.get("observation_type") or "").strip()

        if source in floor:
            kept.append(o)
            stats["floor_kept"] += 1
            continue

        if thin_gdelt and obs_type == "geopolitical_event":
            if rng.random() < gdelt_subsample_frac:
                kept.append(o)
                stats["gdelt_kept"] += 1
            else:
                stats["gdelt_dropped"] += 1
            continue

        if thin_defi and obs_type == "tvl_change":
            if rng.random() < defi_subsample_frac:
                kept.append(o)
                stats["defi_kept"] += 1
            else:
                stats["defi_dropped"] += 1
            continue

        kept.append(o)
        stats["other_kept"] += 1

    kept.sort(key=lambda x: float(x.get("observed_at", 0.0)))
    stats["kept_total"] = len(kept)
    stats["dropped_total"] = len(observations) - len(kept)
    return kept, dict(stats)


def training_mix_summary(
    observations: list[dict[str, Any]],
) -> list[tuple[str, int, float]]:
    """Return (observation_type, count, pct) sorted by count desc."""
    n = len(observations) or 1
    counts: Counter[str] = Counter()
    for o in observations:
        counts[o.get("observation_type") or "unknown"] += 1
    return [
        (k, v, 100.0 * v / n)
        for k, v in counts.most_common()
    ]
