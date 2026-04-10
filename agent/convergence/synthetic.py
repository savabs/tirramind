"""Synthetic scenario generation and validation for convergence detection.

Generates realistic multi-source evidence with known planted causal chains,
runs the full convergence detector pipeline, and measures detection accuracy
(precision, recall, F1, template accuracy, direction accuracy).

This module is the ground-truth validator for the convergence engine's
statistical machinery: FDR controls, clique detection, template matching,
and scoring.  It does not depend on live data or external APIs.

Usage::

    from agent.convergence.synthetic import generate_scenarios, run_synthetic_validation
    scenarios = generate_scenarios(n=100, seed=42)
    result = run_synthetic_validation(scenarios)
    print(f"F1: {result.f1:.3f}, template accuracy: {result.template_accuracy:.3f}")
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agent.convergence.detector import (
    ConvergenceDetector,
    ConvergenceDetectorConfig,
    DetectionResult,
)
from agent.convergence.evidence import Evidence
from agent.convergence.taxonomy import CATEGORIES, SignalMeta, SignalRegistry
from agent.convergence.templates import (
    TEMPLATE_LIBRARY,
    CausalTemplate,
    TemplateStep,
)
from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)

# Seconds per day.
_DAY = 86_400


# ── Scenario dataclasses ───────────────────────────────────────


@dataclass
class SyntheticScenario:
    """One test scenario with a known planted causal chain + decoys.

    Attributes
    ----------
    name : str
        Human-readable label (e.g. "supply_chain_disruption_001").
    planted_evidence : list[Evidence]
        Evidence items that embed a real causal chain with correlated
        anomalous values and correct temporal ordering.
    decoy_evidence : list[Evidence]
        Random, uncorrelated evidence items that should NOT trigger
        convergence detection.
    expected_template : str
        Template name the planted chain is designed to match.
    expected_direction : int
        Expected detection direction (+1 for stress, -1 for relief).
    expected_lead_signal : str
        Signal ID expected to be identified as the trigger.
    """

    name: str
    planted_evidence: list[Evidence]
    decoy_evidence: list[Evidence]
    expected_template: str
    expected_direction: int
    expected_lead_signal: str


@dataclass
class SyntheticValidationResult:
    """Aggregate result from running validation on many scenarios.

    Attributes
    ----------
    n_scenarios : int
        Number of scenarios tested.
    true_positives : int
        Planted chains correctly detected.
    false_negatives : int
        Planted chains missed.
    false_positives : int
        Decoy-only detections (no planted chain match).
    template_correct : int
        Of true positives, how many matched the expected template.
    direction_correct : int
        Of true positives, how many had the correct direction.
    lead_correct : int
        Of true positives, how many identified the correct lead signal.
    precision : float
        TP / (TP + FP).
    recall : float
        TP / (TP + FN).
    f1 : float
        Harmonic mean of precision and recall.
    template_accuracy : float
        template_correct / true_positives (or 0 if no TPs).
    direction_accuracy : float
        direction_correct / true_positives (or 0 if no TPs).
    details : list[dict[str, Any]]
        Per-scenario detail dicts.
    """

    n_scenarios: int
    true_positives: int
    false_negatives: int
    false_positives: int
    template_correct: int
    direction_correct: int
    lead_correct: int
    precision: float
    recall: float
    f1: float
    template_accuracy: float
    direction_accuracy: float
    details: list[dict[str, Any]] = field(default_factory=list)


# ── Signal ID generators ──────────────────────────────────────
#
# Each category has signal_id patterns that match the template regexes.
# These are concrete signal IDs that extractors would produce.

_SIGNAL_IDS_BY_CATEGORY: dict[str, list[str]] = {
    "physical_disruption": [
        "weather.us.severe_count",
        "earthquake.global.significant_count",
        "satellite.fire.global_count",
    ],
    "physical_flow": [
        "ais.baltic.vessel_count",
        "transport.us.truck_count",
        "energy_supply.us.petroleum_stocks",
    ],
    "regulatory_action": [
        "sanctions.global.recent_additions",
        "regulatory_gazette.us.significant_rules",
        "drug_regulatory.us.warning_letters",
    ],
    "monetary_policy": [
        "central_bank.fed.assets",
        "rate_monitor.fed.rate",
        "capital_flows.us.net_tic",
    ],
    "financial_stress": [
        "sovereign_debt.us.10y_yield",
        "bankruptcy.us.recent_count",
        "defi.tvl.total",
        "creditor.us.filings_count",
        "liquidity.us.stress_index",
    ],
    "positioning": [
        "cftc.crude_oil.mm_net_long",
        "finra.us.short_volume_ratio",
        "polymarket.whale.consensus",
        "insider.us.cluster_count",
        "form144.us.filings_count",
    ],
    "macro_momentum": [
        "pmi.us.manufacturing",
        "consumer_sentiment.us.umcsi",
        "treasury.us.receipts",
        "building_permits.us.total",
    ],
    "behavioral_intent": [
        "wikipedia.spike.max_z",
        "jobs.us.jolts_openings",
        "patent.us.filing_count",
        "lobbying.us.total_spend",
        "academic.arxiv.volume",
    ],
    "biological": [
        "disease.cdc.wastewater_detection",
        "food_security.fao.alert_count",
    ],
    "geopolitical": [
        "gdelt.global.event_count",
        "political_risk.acled.event_count",
        "migration.unhcr.displacement",
    ],
    "supply_chain": [
        "supply_chain.us.pressure_index",
        "interconnection.us.queue_mw",
        "gov_contract.us.award_count",
    ],
}


def _pick_signal_for_step(
    step: TemplateStep,
    rng: np.random.Generator,
) -> tuple[str, str]:
    """Pick a concrete (signal_id, category) that matches a TemplateStep.

    Returns
    -------
    (signal_id, category)
    """
    valid_cats = [c for c in step.category_pattern.split("|") if c in CATEGORIES]
    if not valid_cats:
        return "unknown.signal", "macro_momentum"

    # Try each category for a signal_id that matches the step regex
    rng.shuffle(valid_cats)  # type: ignore[arg-type]
    for cat in valid_cats:
        candidates = _SIGNAL_IDS_BY_CATEGORY.get(cat, [])
        matching = [
            sid for sid in candidates if re.search(step.signal_pattern, sid) is not None
        ]
        if matching:
            return rng.choice(matching), cat  # type: ignore[return-value]

    # Fallback: pick any signal from first valid category
    cat = valid_cats[0]
    candidates = _SIGNAL_IDS_BY_CATEGORY.get(cat, [])
    if candidates:
        return rng.choice(candidates), cat  # type: ignore[return-value]
    return f"synthetic.{cat}.fallback", cat


# ── Evidence generation ────────────────────────────────────────


def _generate_correlated_series(
    n_signals: int,
    n_points: int,
    correlation: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate n_signals correlated time series (n_signals × n_points).

    Uses a factor model: X_i = sqrt(ρ)*F + sqrt(1-ρ)*ε_i
    where F is the common factor and ε_i is idiosyncratic noise.
    This guarantees pairwise correlation ≈ ρ.
    """
    rho = max(0.0, min(1.0, correlation))
    factor = rng.standard_normal(n_points)
    series = np.empty((n_signals, n_points))
    for i in range(n_signals):
        noise = rng.standard_normal(n_points)
        series[i] = math.sqrt(rho) * factor + math.sqrt(1.0 - rho) * noise
    return series


def generate_planted_chain(
    template: CausalTemplate,
    start_ts: float,
    n_points: int = 60,
    correlation: float = 0.7,
    anomaly_magnitude: float = 3.0,
    seed: int | None = None,
) -> tuple[list[Evidence], str, int]:
    """Generate Evidence items that embed a causal chain matching *template*.

    **All signals share the same time window** so the alignment function can
    pair them.  The causal chain is encoded by staggering the *anomaly onset*
    within each series: signal 0 becomes anomalous first, signal 1 after
    ``within_days`` offset, etc.  This mirrors real convergence — different
    data sources start deviating at different times as the hidden cause
    propagates, but all exist in the same observation window.

    The planted signals are:
    - Correlated with each other (factor model with ρ = *correlation*)
    - Anomalous from staggered onset points onward
    - Assigned correct categories and signal_id patterns

    Parameters
    ----------
    template : CausalTemplate
        The causal chain to embed.
    start_ts : float
        Unix timestamp for the beginning of the scenario.
    n_points : int
        Total number of data points per signal.  Must be large enough
        to contain baseline + anomaly for all steps.
    correlation : float
        Target pairwise correlation between planted signals.
    anomaly_magnitude : float
        Z-score magnitude for anomalous observations.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    (evidence_list, lead_signal_id, expected_direction)
    """
    rng = np.random.default_rng(seed)
    n_steps = len(template.steps)

    # Pick concrete signal IDs for each step
    step_signals: list[tuple[str, str]] = []  # (signal_id, category)
    for step in template.steps:
        step_signals.append(_pick_signal_for_step(step, rng))

    lead_signal = step_signals[0][0]

    # Generate correlated baseline series (all share the same time grid)
    series = _generate_correlated_series(n_steps, n_points, correlation, rng)

    # Stagger anomaly onset: step 0 starts anomaly earliest, later steps later.
    # Map within_days offsets to index positions within the series.
    max_offset = max(s.within_days for s in template.steps)
    # Reserve the first portion as baseline, inject anomaly in the tail
    # Anomaly onset for step i = n_points - anomaly_window + offset_fraction
    anomaly_window = max(15, n_points // 3)
    for i, step in enumerate(template.steps):
        # Earlier steps get anomaly sooner; later steps get it later
        if max_offset > 0:
            frac = step.within_days / max_offset
        else:
            frac = 0.0
        onset = n_points - anomaly_window + int(frac * (anomaly_window // 2))
        onset = max(1, min(onset, n_points - 5))
        series[i, onset:] += anomaly_magnitude

    # Determine expected direction from first step that specifies one
    expected_direction = 1  # default stress
    for step in template.steps:
        if step.direction is not None:
            expected_direction = step.direction
            break

    # Build evidence items — ALL signals share the same timestamp grid
    evidence: list[Evidence] = []
    for i, (step, (sig_id, cat)) in enumerate(
        zip(template.steps, step_signals, strict=True)
    ):
        for j in range(n_points):
            ts = start_ts + j * _DAY
            val = float(series[i, j])
            direction = (
                step.direction if step.direction is not None else (1 if val > 0 else -1)
            )
            evidence.append(
                Evidence(
                    source=f"synthetic_{sig_id.split('.')[0]}",
                    signal_id=sig_id,
                    timestamp=ts,
                    value=val,
                    direction=direction,
                    confidence=0.8,
                    category=cat,
                    tags=("synthetic", template.name),
                    ttl=7 * _DAY,
                )
            )

    return evidence, lead_signal, expected_direction


def generate_decoy_signals(
    num_signals: int = 5,
    n_points: int = 60,
    start_ts: float = 0.0,
    seed: int | None = None,
) -> list[Evidence]:
    """Generate random, uncorrelated Evidence that should NOT trigger detection.

    Each decoy signal has independent noise, belongs to a random category,
    and has no anomalous values (z-scores within ±1.0).
    """
    rng = np.random.default_rng(seed)
    categories = sorted(CATEGORIES)
    evidence: list[Evidence] = []

    for i in range(num_signals):
        cat = categories[i % len(categories)]
        candidates = _SIGNAL_IDS_BY_CATEGORY.get(cat, [])
        if candidates:
            sig_id = f"decoy.{rng.choice(candidates).split('.')[-1]}.{i}"
        else:
            sig_id = f"decoy.{cat}.{i}"

        # Normal noise, small magnitude (z ≈ 0 to 1)
        values = rng.standard_normal(n_points) * 0.5

        for j in range(n_points):
            ts = start_ts + j * _DAY
            val = float(values[j])
            evidence.append(
                Evidence(
                    source=f"decoy_{i}",
                    signal_id=sig_id,
                    timestamp=ts,
                    value=val,
                    direction=0,
                    confidence=0.5,
                    category=cat,
                    tags=("decoy",),
                    ttl=7 * _DAY,
                )
            )

    return evidence


def generate_scenarios(
    n: int = 100,
    seed: int = 42,
    templates: list[CausalTemplate] | None = None,
) -> list[SyntheticScenario]:
    """Generate N synthetic test scenarios.

    Each scenario plants one causal chain from the template library
    amid 3-8 decoy signals.

    Parameters
    ----------
    n : int
        Number of scenarios to generate.
    seed : int
        Random seed for reproducibility.
    templates : list[CausalTemplate] | None
        Templates to sample from.  Defaults to TEMPLATE_LIBRARY.

    Returns
    -------
    list[SyntheticScenario]
    """
    rng = np.random.default_rng(seed)
    tmpl_pool = TEMPLATE_LIBRARY if templates is None else templates
    if not tmpl_pool:
        raise ValueError("No templates available for scenario generation.")

    scenarios: list[SyntheticScenario] = []
    base_ts = 1_600_000_000.0  # arbitrary epoch start (Sep 2020)

    for i in range(n):
        # Pick a random template
        tmpl = tmpl_pool[i % len(tmpl_pool)]
        start_ts = base_ts + i * 90 * _DAY  # spread scenarios apart

        # Generate planted chain
        planted, lead_sig, expected_dir = generate_planted_chain(
            template=tmpl,
            start_ts=start_ts,
            n_points=60,
            correlation=0.6 + rng.uniform(0, 0.3),  # 0.6–0.9
            anomaly_magnitude=2.5 + rng.uniform(0, 2.0),  # 2.5–4.5 z
            seed=int(rng.integers(0, 2**31)),
        )

        # Generate decoys
        n_decoys = int(rng.integers(3, 9))
        decoys = generate_decoy_signals(
            num_signals=n_decoys,
            n_points=60,
            start_ts=start_ts,
            seed=int(rng.integers(0, 2**31)),
        )

        scenarios.append(
            SyntheticScenario(
                name=f"{tmpl.name}_{i:03d}",
                planted_evidence=planted,
                decoy_evidence=decoys,
                expected_template=tmpl.name,
                expected_direction=expected_dir,
                expected_lead_signal=lead_sig,
            )
        )

    return scenarios


# ── Validation runner ──────────────────────────────────────────


def _build_registry_from_evidence(
    evidence: list[Evidence],
) -> SignalRegistry:
    """Build a SignalRegistry from observed Evidence items."""
    registry = SignalRegistry()
    seen: set[str] = set()
    for ev in evidence:
        if ev.signal_id not in seen:
            seen.add(ev.signal_id)
            registry.register(
                SignalMeta(
                    signal_id=ev.signal_id,
                    source=ev.source,
                    category=ev.category,
                    frequency="daily",
                    direction_semantics="synthetic",
                    default_ttl=ev.ttl,
                    min_observations=3,
                )
            )
    return registry


def _check_detection_matches_planted(
    results: list[DetectionResult],
    scenario: SyntheticScenario,
) -> dict[str, Any]:
    """Check whether any detection result matches the planted chain.

    Returns a detail dict with match info.
    """
    planted_sigs = {ev.signal_id for ev in scenario.planted_evidence}

    detail: dict[str, Any] = {
        "scenario": scenario.name,
        "expected_template": scenario.expected_template,
        "expected_direction": scenario.expected_direction,
        "n_detections": len(results),
        "detected": False,
        "template_match": False,
        "direction_match": False,
        "lead_match": False,
        "false_positives": 0,
    }

    for r in results:
        clique_sigs = set(r.clique.signals)
        # A detection "matches" if it shares ≥2 signals with the planted chain
        overlap = clique_sigs & planted_sigs
        if len(overlap) >= 2:
            detail["detected"] = True
            detail["overlap_count"] = len(overlap)
            detail["detected_event_type"] = r.event_type
            detail["detected_direction"] = r.direction

            if r.event_type == scenario.expected_template:
                detail["template_match"] = True
            if r.direction == scenario.expected_direction:
                detail["direction_match"] = True
            if r.lead_signal == scenario.expected_lead_signal:
                detail["lead_match"] = True
            break  # count first matching detection
        else:
            # Detection doesn't match planted chain → false positive
            detail["false_positives"] += 1

    if not detail["detected"]:
        # All detections (if any) are false positives
        detail["false_positives"] = len(results)

    return detail


def run_synthetic_validation(
    scenarios: list[SyntheticScenario],
    config: ConvergenceDetectorConfig | None = None,
    db_path: str = ":memory:",
) -> SyntheticValidationResult:
    """Run the full convergence detector on each scenario and score accuracy.

    Parameters
    ----------
    scenarios : list[SyntheticScenario]
        Scenarios with planted chains + decoys.
    config : ConvergenceDetectorConfig | None
        Detector configuration.  Uses relaxed defaults for synthetic
        validation if None.
    db_path : str
        SQLite path.  Default is in-memory for speed.

    Returns
    -------
    SyntheticValidationResult
    """
    if not scenarios:
        return SyntheticValidationResult(
            n_scenarios=0,
            true_positives=0,
            false_negatives=0,
            false_positives=0,
            template_correct=0,
            direction_correct=0,
            lead_correct=0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            template_accuracy=0.0,
            direction_accuracy=0.0,
        )

    # Relaxed config for synthetic (fewer constraints so planted chains fire)
    if config is None:
        config = ConvergenceDetectorConfig(
            z_threshold=1.5,
            p_threshold=0.10,
            fdr_q=0.10,
            min_clique_size=2,
            min_categories=2,
            min_persistence=1,
            corr_window=10,
            baseline_window=20,
            template_boost=0.5,
            lookback_days=365,
            max_pairs=500,
        )

    tp = 0
    fn = 0
    fp = 0
    template_ok = 0
    direction_ok = 0
    lead_ok = 0
    details: list[dict[str, Any]] = []

    for scenario in scenarios:
        # Combine planted + decoy evidence
        all_evidence = scenario.planted_evidence + scenario.decoy_evidence

        # Create fresh in-memory store for this scenario
        store = PipelineStore(":memory:")

        # Store evidence as pipeline data rows: group by source
        by_source: dict[str, list[Evidence]] = {}
        for ev in all_evidence:
            by_source.setdefault(ev.source, []).append(ev)

        # We need to store data in a format that extractors can re-extract.
        # Since these are synthetic, we store evidence directly and bypass
        # extractors by storing pre-extracted evidence grouped as tool outputs.
        # The detector's _load_evidence calls extract_evidence(tool_name, data)
        # for each stored row.  So we store data dicts that our extractors
        # understand — OR we inject evidence directly.
        #
        # The cleanest approach: monkey-patch _load_evidence on the detector
        # to return our pre-built evidence list directly.

        # Build registry
        registry = _build_registry_from_evidence(all_evidence)

        # Create detector
        detector = ConvergenceDetector(store, registry, config)

        # Monkey-patch _load_evidence to return our synthetic evidence
        detector._load_evidence = lambda s, u, _ev=all_evidence: _ev  # type: ignore[assignment]

        # Determine as_of: latest timestamp + 1 day
        max_ts = max(ev.timestamp for ev in all_evidence)
        as_of = max_ts + _DAY

        # Run detection
        try:
            results = detector.detect(as_of=as_of)
        except Exception:
            log.warning(
                "Detection failed for scenario %s", scenario.name, exc_info=True
            )
            results = []

        # Evaluate
        detail = _check_detection_matches_planted(results, scenario)
        details.append(detail)

        if detail["detected"]:
            tp += 1
            if detail["template_match"]:
                template_ok += 1
            if detail["direction_match"]:
                direction_ok += 1
            if detail["lead_match"]:
                lead_ok += 1
        else:
            fn += 1

        fp += detail["false_positives"]

    # Compute aggregate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    tmpl_acc = template_ok / tp if tp > 0 else 0.0
    dir_acc = direction_ok / tp if tp > 0 else 0.0

    return SyntheticValidationResult(
        n_scenarios=len(scenarios),
        true_positives=tp,
        false_negatives=fn,
        false_positives=fp,
        template_correct=template_ok,
        direction_correct=direction_ok,
        lead_correct=lead_ok,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        template_accuracy=round(tmpl_acc, 4),
        direction_accuracy=round(dir_acc, 4),
        details=details,
    )
