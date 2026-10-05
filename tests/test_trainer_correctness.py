"""Regression tests for the six confirmed trainer.py defects (audit 2026-09-23).

Every test in this file fails on the pre-fix behaviour:

    (A) count-based split over a future-polluted timestamp range
    (B) EWC computed, checkpointed, logged as active — never added to the loss
    (C) _loss_from_window omits the return loss, so Fisher ignores the return heads
    (D) F-01 collapse gate thresholds an unnormalised std and swallows its errors
    (E) subsampling applied to the feature stream only, and never to evaluate()
    (F) return upscale derived from registered-entity counts, not the window batch

References: docs/research/full_audit_2026-09-23.md (P0.6, P4.2–P4.5),
LESSONS.md F-01, F-03, F-15.
"""

from __future__ import annotations

import math
import time

import pytest
import torch

from agent.models.gnn.ewc import EWCState, compute_fisher
from agent.models.gnn.graph_builder import GraphBuilder
from agent.models.gnn.trainer import (
    SplitRangeError,
    Trainer,
    TrainerConfig,
    adjacent_window_pairs,
    collapse_detected,
    embedding_collapse_report,
    estimate_return_upscale,
    evaluate,
    split_observations_by_calendar,
)
from agent.pipeline.entity import entity_id_from_key
from agent.pipeline.store import PipelineStore

DAY = 86400.0
NOW = 1_758_600_000.0  # 2025-09-23, fixed so tests never depend on wall clock
YEAR_2030 = 1_895_000_000.0  # well beyond NOW
YEAR_1920 = -1_577_923_200.0  # 1920-01-01


def _obs(ts: float, *, entity_id: str = "e1", obs_type: str = "geopolitical_event", **value) -> dict:
    return {
        "entity_id": entity_id,
        "observed_at": float(ts),
        "observation_type": obs_type,
        "source_tool": "test",
        "value": dict(value),
    }


# ═══════════════════════════════════════════════════════════════
# (A) Calendar split
# ═══════════════════════════════════════════════════════════════


class TestCalendarSplit:
    def test_split_cuts_on_calendar_time_not_row_count(self):
        """A dense early burst must not drag the train cut back with it.

        900 rows live in the first 10% of the timeline and 100 rows are spread
        over the remaining 90%.  The old ``obs[:int(n*0.70)]`` cut lands inside
        the burst, so 'train' covers a tenth of the period it claims.
        """
        t0 = NOW - 1000 * DAY
        span = 1000 * DAY
        dense = [_obs(t0 + i * (0.1 * span / 900)) for i in range(900)]
        sparse = [_obs(t0 + 0.1 * span + i * (0.9 * span / 100)) for i in range(100)]
        obs = dense + sparse

        cfg = TrainerConfig(split_purge_gap_seconds=0.0)
        train, val, test = split_observations_by_calendar(obs, cfg, now=NOW)

        # The count-based split would have cut here — inside the burst.
        count_cut = sorted(o["observed_at"] for o in obs)[int(len(obs) * 0.70)]
        assert count_cut < t0 + 0.11 * span

        train_hi = max(o["observed_at"] for o in train)
        assert train_hi > t0 + 0.5 * span, "calendar cut must be a calendar cut"
        assert train_hi < t0 + 0.71 * span
        assert val and test

    def test_future_dated_rows_are_in_no_split(self):
        """912 future-dated rows made the whole test set live in 2026-2030."""
        obs = [_obs(NOW - (300 - i) * DAY) for i in range(300)]
        future = [_obs(YEAR_2030 + i * DAY, entity_id="future") for i in range(10)]
        cfg = TrainerConfig(split_purge_gap_seconds=0.0)

        train, val, test = split_observations_by_calendar(obs + future, cfg, now=NOW)

        for name, part in (("train", train), ("val", val), ("test", test)):
            assert part, f"{name} must not be empty"
            assert max(o["observed_at"] for o in part) <= NOW, f"{name} contains future rows"
        assert not any(o["entity_id"] == "future" for o in train + val + test)

    def test_ancient_rows_do_not_stretch_the_range(self):
        """One 1920 row would otherwise put the 70% calendar cut in 1997."""
        obs = [_obs(NOW - (300 - i) * DAY) for i in range(300)]
        ancient = [_obs(YEAR_1920, entity_id="ancient")]
        cfg = TrainerConfig(split_purge_gap_seconds=0.0)

        train, val, test = split_observations_by_calendar(obs + ancient, cfg, now=NOW)

        assert not any(o["entity_id"] == "ancient" for o in train + val + test)
        assert min(o["observed_at"] for o in train) > NOW - 301 * DAY
        assert len(train) > 100, "the real data must survive the trim"

    def test_shared_timestamp_cannot_straddle_a_cut(self):
        """227 DB rows share one timestamp exactly; they must land on one side."""
        t0 = NOW - 100 * DAY
        obs = [_obs(t0 + i * DAY) for i in range(100)]
        cut_ts = t0 + 70 * DAY
        obs += [_obs(cut_ts, entity_id=f"tie{i}") for i in range(50)]
        cfg = TrainerConfig(split_purge_gap_seconds=0.0)

        train, val, test = split_observations_by_calendar(obs, cfg, now=NOW)

        assert max(o["observed_at"] for o in train) < min(o["observed_at"] for o in val)
        assert max(o["observed_at"] for o in val) < min(o["observed_at"] for o in test)
        ties_train = sum(1 for o in train if o["entity_id"].startswith("tie"))
        ties_elsewhere = sum(1 for o in val + test if o["entity_id"].startswith("tie"))
        assert ties_train == 0 or ties_elsewhere == 0

    def test_purge_gap_separates_train_from_val(self):
        t0 = NOW - 400 * DAY
        obs = [_obs(t0 + i * DAY) for i in range(400)]
        cfg = TrainerConfig(forward_return_horizon=21)

        train, val, _test = split_observations_by_calendar(obs, cfg, now=NOW)

        gap = min(o["observed_at"] for o in val) - max(o["observed_at"] for o in train)
        assert gap >= 21 * DAY, f"purge gap {gap / DAY:.1f}d < forward horizon"

    def test_all_future_rows_raises_instead_of_returning_empty(self):
        cfg = TrainerConfig()
        with pytest.raises(SplitRangeError):
            split_observations_by_calendar(
                [_obs(YEAR_2030 + i * DAY) for i in range(10)],
                cfg,
                now=NOW,
            )


class TestWindowPairing:
    def test_pairs_spanning_a_gap_are_excluded(self):
        """A 4-year hole became one 'next window' pair with a log1p(1.3e8) target."""
        store = PipelineStore(db_path=":memory:")
        trainer = Trainer(store, TrainerConfig(window_size=DAY))
        t0 = NOW - 10 * DAY
        obs = [_obs(t0 + i * DAY) for i in range(5)]
        obs += [_obs(t0 + 4 * 365 * DAY + i * DAY) for i in range(5)]
        windows = trainer._make_windows(sorted(obs, key=lambda o: o["observed_at"]))

        pairs = adjacent_window_pairs(windows, DAY)

        assert len(windows) - 1 > len(pairs), "the gap-spanning pair must be dropped"
        for i in pairs:
            assert math.isclose(windows[i + 1][0] - windows[i][0], DAY, rel_tol=1e-9)


# ═══════════════════════════════════════════════════════════════
# (D) Collapse gate
# ═══════════════════════════════════════════════════════════════


class TestCollapseGate:
    @staticmethod
    def _rank_one(n: int = 64, dim: int = 64) -> torch.Tensor:
        """The live checkpoint's profile: huge std, effective rank ~1."""
        g = torch.Generator().manual_seed(0)
        direction = torch.randn(dim, generator=g)
        scale = torch.rand(n, generator=g) * 1e4 + 1.0
        return scale[:, None] * direction[None, :]

    def test_old_std_gate_passes_what_the_new_gate_catches(self):
        emb = self._rank_one()
        old_gate_fires = emb.std(dim=0).mean().item() < 0.05
        assert old_gate_fires is False, "fixture must reproduce the std the old gate passed"

        report = embedding_collapse_report(emb)
        assert report["collapse_detected"] is True
        assert report["eff_rank"] < 2.0
        assert report["emb_std"] > 100.0  # kept as a diagnostic only

    def test_full_rank_embeddings_are_not_flagged(self):
        g = torch.Generator().manual_seed(1)
        emb = torch.randn(64, 64, generator=g)
        assert collapse_detected(emb) is False

    def test_identical_rows_are_flagged(self):
        emb = torch.ones(64, 64) * 6403.0
        assert collapse_detected(emb) is True

    def test_degenerate_input_raises_instead_of_being_swallowed(self):
        with pytest.raises(ValueError):
            embedding_collapse_report(torch.ones(1, 64))
        with pytest.raises(ValueError):
            embedding_collapse_report(torch.ones(64))


# ═══════════════════════════════════════════════════════════════
# (F) Return upscale
# ═══════════════════════════════════════════════════════════════


class TestReturnUpscale:
    def test_upscale_uses_the_per_window_batch(self):
        """Old: 6,110 registered entities / 89 labelled = 68.65x.  True: ~2x."""
        windows = []
        for w in range(3):
            t = NOW + w * DAY
            nxt = [_obs(t, entity_id=f"other{i}", obs_type="geopolitical_event") for i in range(50)]
            nxt += [_obs(t, entity_id=f"inst{i}", obs_type="instrument_daily", log_return=0.01) for i in range(50)]
            windows.append((t, t + DAY, nxt))

        ratio, stats = estimate_return_upscale(windows, [0, 1], None, use_forward_returns=False)

        assert ratio == pytest.approx(2.0)
        assert stats["median_sup"] == 100
        assert stats["median_ret"] == 50
        assert ratio < 10.0, "must not reproduce the 68.65x entity-count ratio"

    def test_no_return_labels_gives_one_not_a_silent_large_number(self, caplog):
        windows = [
            (NOW, NOW + DAY, [_obs(NOW, entity_id="a")]),
            (NOW + DAY, NOW + 2 * DAY, [_obs(NOW + DAY, entity_id="b")]),
        ]
        with caplog.at_level("WARNING"):
            ratio, stats = estimate_return_upscale(windows, [0], None, use_forward_returns=False)
        assert ratio == 1.0
        assert stats["n_windows"] == 0
        assert any("no window pair carries a return label" in r.message for r in caplog.records)


# ═══════════════════════════════════════════════════════════════
# Store fixtures for the loop-level tests
# ═══════════════════════════════════════════════════════════════


def _build_store(
    *,
    n_instruments: int = 4,
    n_countries: int = 2,
    days: int = 30,
    per_day: int = 4,
    gdelt_per_day: int = 0,
) -> PipelineStore:
    """Small, recent, densely bucketed store: every day bucket is non-empty."""
    store = PipelineStore(db_path=":memory:")
    rng = torch.Generator().manual_seed(7)
    inst_ids = []
    for i in range(n_instruments):
        eid = entity_id_from_key("instrument", f"INST{i}")
        store.register_entity("instrument", f"INST{i}", eid)
        inst_ids.append(eid)
    country_ids = []
    for i in range(n_countries):
        eid = entity_id_from_key("country", f"C{i}")
        store.register_entity("country", f"C{i}", eid)
        country_ids.append(eid)
    for cid in country_ids:
        store.link_entities(inst_ids[0], cid, "headquartered_in", "test", confidence=0.9)

    t0 = NOW - days * DAY
    for d in range(days):
        for k in range(per_day):
            ts = t0 + d * DAY + k * (DAY / (per_day + 1))
            for j, eid in enumerate(inst_ids):
                lr = float(torch.randn(1, generator=rng).item()) * 0.02
                store.store_entity_observation(
                    entity_id=eid,
                    source_tool="test",
                    observed_at=ts + j,
                    observation_type="instrument_daily",
                    value={"log_return": lr, "close": 100.0 + j},
                )
            for j, eid in enumerate(country_ids):
                store.store_entity_observation(
                    entity_id=eid,
                    source_tool="test",
                    observed_at=ts + 10 + j,
                    observation_type="sanction_event",
                    value={"amount": 1.0},
                )
        for g in range(gdelt_per_day):
            store.store_entity_observation(
                entity_id=country_ids[g % len(country_ids)],
                source_tool="gdelt",
                observed_at=t0 + d * DAY + 100 + g,
                observation_type="geopolitical_event",
                value={"goldstein": 1.0, "num_mentions": 3},
            )
    return store


def _small_cfg(**kw) -> TrainerConfig:
    base = dict(
        hidden_dim=16,
        memory_dim=16,
        message_dim=16,
        num_heads=2,
        num_layers=1,
        epochs=1,
        window_size=DAY,
        split_purge_gap_seconds=0.0,
    )
    base.update(kw)
    return TrainerConfig(**base)


# ═══════════════════════════════════════════════════════════════
# (C) Fisher covers the return heads
# ═══════════════════════════════════════════════════════════════


class TestLossFromWindowCoversReturnHeads:
    def test_return_head_receives_gradient(self):
        store = _build_store(days=12)
        trainer = Trainer(store, _small_cfg())
        model = trainer.build_model()
        obs = trainer._graph_builder.prefetch_observations()
        windows = trainer._make_windows(obs)
        assert len(windows) >= 3
        i = adjacent_window_pairs(windows, DAY)[-1]
        data, id_map, _ = trainer._graph_builder.build(until=windows[i][1])

        loss = trainer._loss_from_window(data, id_map, windows[i][2], windows[i + 1][2])
        model.zero_grad()
        loss.backward()

        touched = {
            n
            for n, p in model.named_parameters()
            if "return" in n and p.grad is not None and float(p.grad.abs().sum()) > 0.0
        }
        assert touched, "no return-head parameter received gradient from _loss_from_window"

    def test_fisher_diagonal_is_nonzero_for_return_params(self):
        store = _build_store(days=12)
        trainer = Trainer(store, _small_cfg())
        model = trainer.build_model()
        obs = trainer._graph_builder.prefetch_observations()
        windows = trainer._make_windows(obs)
        i = adjacent_window_pairs(windows, DAY)[-1]
        data, id_map, _ = trainer._graph_builder.build(until=windows[i][1])

        fisher = compute_fisher(
            model,
            lambda: trainer._loss_from_window(data, id_map, windows[i][2], windows[i + 1][2]),
            n_samples=1,
        )

        return_fisher = {n: float(t.sum()) for n, t in fisher.items() if "return" in n}
        assert return_fisher, "model exposes no return parameters"
        assert any(v > 0.0 for v in return_fisher.values()), (
            "every return-head Fisher diagonal is 0 — EWC would protect everything except the ranking signal"
        )


# ═══════════════════════════════════════════════════════════════
# (B) EWC enters the training loss
# ═══════════════════════════════════════════════════════════════


class TestEWCInTrainingLoss:
    def test_ewc_penalty_is_added_to_the_loss(self):
        store = _build_store(days=14)
        trainer = Trainer(store, _small_cfg(epochs=1))
        model = trainer.build_model()
        trainer._ewc_state = EWCState(
            fisher={n: torch.ones_like(p) for n, p in model.named_parameters()},
            anchor={n: torch.zeros_like(p) for n, p in model.named_parameters()},
            lambda_=0.01,
        )

        history = trainer.train()

        assert "ewc" in history, "EWC never reached the loss, so it was never recorded"
        assert history["ewc"][-1] > 0.0
        assert len(history["ewc"]) == len(history["total"])  # F-03 alignment

    def test_ewc_can_be_switched_off_explicitly(self):
        store = _build_store(days=14)
        trainer = Trainer(store, _small_cfg(epochs=1, apply_ewc_in_training=False))
        model = trainer.build_model()
        trainer._ewc_state = EWCState(
            fisher={n: torch.ones_like(p) for n, p in model.named_parameters()},
            anchor={n: torch.zeros_like(p) for n, p in model.named_parameters()},
            lambda_=0.01,
        )

        history = trainer.train()

        assert history["ewc"][-1] == 0.0


# ═══════════════════════════════════════════════════════════════
# (E) Subsample parity
# ═══════════════════════════════════════════════════════════════


class TestSubsampleParity:
    def test_training_windows_come_from_the_subsampled_stream(self, monkeypatch):
        store = _build_store(days=14, gdelt_per_day=40)
        cfg = _small_cfg(epochs=1, gdelt_subsample_frac=0.05)
        trainer = Trainer(store, cfg)
        trainer.build_model()

        captured: dict = {}
        original = Trainer._split_observations

        def spy(self, observations=None):
            captured["fed"] = observations
            return original(self, observations=observations)

        monkeypatch.setattr(Trainer, "_split_observations", spy)
        trainer.train()

        fed = captured["fed"]
        assert fed is not None, "the split was not derived from the subsampled stream"
        fed_ids = {id(o) for o in fed}
        train_obs, _, _ = trainer._split_observations(observations=fed)
        windows = trainer._make_windows(train_obs)
        for _s, _e, w_obs in windows:
            for o in w_obs:
                assert id(o) in fed_ids, "supervision window holds an obs the model never saw"

        n_gdelt_db = sum(1 for o in store.query_all_observations() if o["observation_type"] == "geopolitical_event")
        n_gdelt_fed = sum(1 for o in fed if o["observation_type"] == "geopolitical_event")
        assert n_gdelt_db > 100
        assert n_gdelt_fed < 0.5 * n_gdelt_db, "the supervision stream was not thinned"

    def test_evaluate_sees_the_same_thinned_distribution(self, monkeypatch):
        store = _build_store(days=14, gdelt_per_day=40)
        cfg = _small_cfg(epochs=1, gdelt_subsample_frac=0.05)
        trainer = Trainer(store, cfg)
        model = trainer.build_model()

        fed_sizes: list[int] = []
        original = GraphBuilder.build_from_cached

        def spy(self, id_map, links, *, observations=None, **kw):
            fed_sizes.append(len(observations or []))
            return original(self, id_map, links, observations=observations, **kw)

        monkeypatch.setattr(GraphBuilder, "build_from_cached", spy)
        evaluate(model, store, cfg, split="val")

        assert fed_sizes, "evaluate() built no snapshots"
        n_all = len(store.query_all_observations())
        n_gdelt = sum(1 for o in store.query_all_observations() if o["observation_type"] == "geopolitical_event")
        assert n_gdelt > 100
        # Without the fix evaluate() fed the full unsubsampled table.
        assert max(fed_sizes) < n_all - 0.5 * n_gdelt


# ═══════════════════════════════════════════════════════════════
# End-to-end sanity: the split used by train() is bounded
# ═══════════════════════════════════════════════════════════════


def test_train_split_never_ends_in_the_future(monkeypatch):
    """The live DB still holds 912 future-dated rows; the split must survive them.

    They cannot be written any more (``store`` now rejects them), so they are
    injected at the read boundary exactly as a legacy row would come back.
    """
    store = _build_store(days=20)
    legacy_rows = store.query_all_observations()
    poisoned = legacy_rows + [
        _obs(YEAR_2030 + i * DAY, entity_id=entity_id_from_key("instrument", "INST0"), obs_type="instrument_daily")
        for i in range(5)
    ]
    monkeypatch.setattr(store, "query_all_observations", lambda *a, **k: list(poisoned))

    trainer = Trainer(store, _small_cfg())
    train, _val, test = trainer._split_observations()

    assert test, "test split must not be empty"
    assert max(o["observed_at"] for o in test) <= time.time()
    assert max(o["observed_at"] for o in train) < min(o["observed_at"] for o in test)
