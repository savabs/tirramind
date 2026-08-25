#!/usr/bin/env python3
"""Phase B.3 — Ridge ranker on frozen GNN embeddings (Stage2-RidgeRanker).

Compares Stage2 vs Momentum-Rank on canonical 21d forward-return labels.

Usage:
    python scripts/stage2_ranker_eval.py \\
        --checkpoint .tirra_pipeline/gnn_model_phase50.pt \\
        --weights-from-epoch .tirra_pipeline/checkpoints/phase50/epoch_090.pt \\
        --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from agent.models.gnn.graph_builder import GraphBuilder  # noqa: E402
from agent.models.gnn.trainer import Trainer  # noqa: E402
from agent.pipeline.store import PipelineStore  # noqa: E402
from agent.quant.forward_returns import (  # noqa: E402
    build_forward_return_lookup,
    forward_return_vector_for_date,
)
from honest_baseline_audit import (  # noqa: E402
    MomentumRankStrategy,
    _instrument_price_feature_matrix,
    _warm_ic_caches,
)
from phase40_gnn_backtest import (  # noqa: E402
    IC_EXIT_MEAN,
    IC_EXIT_TSTAT,
    MIN_TRAIN,
    STEP_SIZE,
    TEMPERATURE,
    TEST_SIZE,
    _instrument_embedding_matrix,
    _load_instrument_returns_fast,
    _print_ic_report,
    _softmax,
)

log = logging.getLogger("stage2_ranker_eval")


class Stage2RidgeRankerStrategy:
    """Ridge on frozen GNN embeddings (+ optional momentum feature)."""

    def __init__(
        self,
        *,
        trainer: Any | None,
        embedding_dir: Path | None,
        dates: list[str],
        returns: np.ndarray,
        prefetched: list[dict],
        id_map: Any,
        links: list[dict],
        fwd_lookup: dict[tuple[str, int], float],
        instrument_names: list[str],
        temperature: float = TEMPERATURE,
        ridge_alpha: float = 1.0,
        min_train: int = MIN_TRAIN,
        add_momentum: bool = True,
        graph_builder: GraphBuilder | None = None,
    ) -> None:
        self._trainer = trainer
        self._embedding_dir = embedding_dir
        self._manifest: dict | None = None
        if embedding_dir is not None:
            mp = embedding_dir / "manifest.json"
            if mp.exists():
                self._manifest = json.loads(mp.read_text())
        self._dates = dates
        self._returns = returns
        self._obs = prefetched
        self._obs_ts = [o["observed_at"] for o in prefetched]
        self._id_map = id_map
        self._links = links
        self._fwd_lookup = fwd_lookup
        self._names = instrument_names
        self._temperature = temperature
        self._ridge_alpha = ridge_alpha
        self._min_train = min_train
        self._add_momentum = add_momentum
        self._gb = graph_builder
        self._cache: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        suffix = "+mom" if self._add_momentum else ""
        return f"Stage2-RidgeRanker{suffix}"

    def _embedding_at(self, fold_date: str) -> np.ndarray:
        if self._embedding_dir is not None:
            path = self._embedding_dir / f"{fold_date}.npy"
            if path.exists():
                return np.load(path)
        if self._trainer is None:
            raise RuntimeError(f"No embedding for {fold_date}")
        return _instrument_embedding_matrix(
            self._trainer,
            fold_date,
            self._names,
            self._id_map,
            self._links,
            self._obs,
            self._obs_ts,
        )

    def _feature_row(self, fold_date: str, i: int, emb: np.ndarray) -> np.ndarray | None:
        if not np.all(np.isfinite(emb[i])):
            return None
        feats = emb[i]
        if self._add_momentum and self._gb is not None:
            price = _instrument_price_feature_matrix(
                self._gb,
                fold_date,
                self._names,
                self._id_map,
                self._links,
                self._obs,
                self._obs_ts,
                model=getattr(self._trainer, "model", None) if self._trainer else None,
            )
            if np.isfinite(price[i, 0]):
                feats = np.concatenate([emb[i], [float(price[i, 0])]])
        return feats

    def generate_weights(
        self,
        train_returns: np.ndarray,
        test_length: int,
        instrument_names: list[str],
        *,
        train_extra: dict | None = None,
        test_extra: dict | None = None,
    ) -> np.ndarray:
        train_len = len(train_returns)
        fold_date = self._dates[train_len]
        if fold_date not in self._cache:
            self._cache[fold_date] = self._compute_weights(train_len, fold_date, instrument_names)
        return np.tile(self._cache[fold_date], (test_length, 1))

    def _compute_weights(
        self, train_len: int, fold_date: str, instrument_names: list[str]
    ) -> np.ndarray:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        N = len(instrument_names)
        X_rows: list[np.ndarray] = []
        y_rows: list[float] = []
        train_start = self._min_train
        for split in range(train_start, train_len - TEST_SIZE, STEP_SIZE):
            emb = self._embedding_at(self._dates[split])
            y = forward_return_vector_for_date(
                self._fwd_lookup, instrument_names, self._dates[split]
            )
            for i in range(N):
                row = self._feature_row(self._dates[split], i, emb)
                if row is not None and np.isfinite(y[i]):
                    X_rows.append(row)
                    y_rows.append(float(y[i]))
        if len(X_rows) < N * 2:
            log.warning(
                "Stage2 fold %s: only %d train samples — equal weights",
                fold_date,
                len(X_rows),
            )
            return np.ones(N) / N

        X = np.stack(X_rows)
        y = np.array(y_rows, dtype=np.float64)
        scaler = StandardScaler()
        ridge = Ridge(alpha=self._ridge_alpha, fit_intercept=True)
        ridge.fit(scaler.fit_transform(X), y)

        test_emb = self._embedding_at(fold_date)
        scores = np.zeros(N, dtype=np.float64)
        for i in range(N):
            row = self._feature_row(fold_date, i, test_emb)
            if row is not None:
                scores[i] = float(ridge.predict(scaler.transform(row.reshape(1, -1)))[0])
        if not np.any(np.isfinite(scores)):
            return np.ones(N) / N
        return _softmax(scores, self._temperature)

    def compute_fold_ics(self, dates: list[str], returns: np.ndarray) -> np.ndarray:
        from scipy.stats import spearmanr

        fold_ics: list[float] = []
        split = self._min_train
        while split + TEST_SIZE <= len(dates):
            fold_date = dates[split]
            if fold_date not in self._cache:
                split += STEP_SIZE
                continue
            w = self._cache[fold_date]
            y = forward_return_vector_for_date(
                self._fwd_lookup, self._names, fold_date
            )
            valid = np.isfinite(w) & np.isfinite(y)
            if valid.sum() >= 5:
                ic, _ = spearmanr(w[valid], y[valid])
                if np.isfinite(ic):
                    fold_ics.append(float(ic))
            split += STEP_SIZE
        return np.array(fold_ics, dtype=np.float64)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description="Phase B Stage-2 ranker eval")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--weights-from-epoch", type=Path, default=None)
    ap.add_argument("--embedding-dir", type=Path, default=None)
    ap.add_argument("--db-path", type=Path, default=Path(".tirra_pipeline/pipeline.db"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(".tirra_pipeline/stage2_ranker_eval.json"),
    )
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--no-momentum", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    store = PipelineStore(str(args.db_path))
    gb = GraphBuilder(store)
    prefetched = gb.prefetch_observations()
    id_map, _, links = gb.prepare_static()
    fwd_lookup = build_forward_return_lookup(prefetched)

    entities = store.query_all_entities()
    entity_ids = [e["entity_id"] for e in entities if e["entity_type"] == "instrument"]
    dates, returns = _load_instrument_returns_fast(str(args.db_path), entity_ids)

    min_train = MIN_TRAIN
    if args.smoke:
        min_train = 126
        if len(dates) > 400:
            dates = dates[-400:]
            returns = returns[-400:]

    trainer = None
    if args.embedding_dir is None:
        if args.weights_from_epoch is not None:
            trainer = Trainer.load_model_with_epoch_weights(
                args.checkpoint, args.weights_from_epoch, store
            )
        else:
            trainer = Trainer.load_model(args.checkpoint, store)

    stage2 = Stage2RidgeRankerStrategy(
        trainer=trainer,
        embedding_dir=args.embedding_dir,
        dates=dates,
        returns=returns,
        prefetched=prefetched,
        id_map=id_map,
        links=links,
        fwd_lookup=fwd_lookup,
        instrument_names=entity_ids,
        ridge_alpha=args.ridge_alpha,
        min_train=min_train,
        add_momentum=not args.no_momentum,
        graph_builder=gb,
    )
    momentum = MomentumRankStrategy(
        gb,
        dates,
        prefetched,
        id_map,
        links,
        min_train=min_train,
        fwd_lookup=fwd_lookup,
    )

    print("\n" + "=" * 60)
    print("PHASE B — STAGE 2 RIDGE RANKER EVAL")
    print("=" * 60)

    _warm_ic_caches([stage2, momentum], dates, returns, entity_ids, min_train=min_train)

    ic_results: dict[str, dict[str, float | int | list[float]]] = {}
    for strat in (stage2, momentum):
        ics = strat.compute_fold_ics(dates, returns)
        n = len(ics)
        mean_ic = float(ics.mean()) if n else 0.0
        std_ic = float(ics.std()) if n > 1 else 0.0
        t_stat = float(mean_ic / (std_ic / np.sqrt(n))) if n > 1 and std_ic > 1e-12 else 0.0
        icir = float(mean_ic / std_ic) if std_ic > 1e-12 else 0.0
        ic_results[strat.name] = {
            "fold_ics": ics.tolist(),
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "icir": icir,
            "t_stat": t_stat,
            "n_folds": n,
            "passed": bool(mean_ic > IC_EXIT_MEAN and t_stat > IC_EXIT_TSTAT),
        }

    _print_ic_report({k: v for k, v in ic_results.items()})  # type: ignore[arg-type]

    primary = ic_results[stage2.name]
    gate_pass = bool(primary.get("passed"))
    print(
        f"\n★ PRIMARY GATE ({stage2.name}): "
        f"{'PASS' if gate_pass else 'FAIL'} — "
        f"IC={primary['mean_ic']:+.4f}, t={primary['t_stat']:+.2f}, n={primary['n_folds']}"
    )

    mom = ic_results[momentum.name]
    delta_ic = float(primary["mean_ic"]) - float(mom["mean_ic"])
    print(
        f"  vs Momentum-Rank: ΔIC={delta_ic:+.4f} "
        f"(Stage2 {'beats' if delta_ic > 0 else 'lags'} momentum)"
    )

    payload = {
        "phase": "B_stage2_ranker",
        "checkpoint": str(args.checkpoint),
        "weights_from_epoch": str(args.weights_from_epoch) if args.weights_from_epoch else None,
        "embedding_dir": str(args.embedding_dir) if args.embedding_dir else None,
        "smoke": args.smoke,
        "strategies": ic_results,
        "primary_strategy": stage2.name,
        "primary_gate": {
            "strategy": stage2.name,
            "mean_ic": primary["mean_ic"],
            "t_stat": primary["t_stat"],
            "n_folds": primary["n_folds"],
            "passed": gate_pass,
        },
        "vs_momentum_delta_ic": delta_ic,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nResults → {args.out}")


if __name__ == "__main__":
    main()
