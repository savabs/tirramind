#!/usr/bin/env python3
"""Phase 40 — Retrain HetTGN on real pipeline observations.

Usage:
    python scripts/retrain_gnn.py [--db-path PATH] [--epochs N] [--lr FLOAT]
                                  [--auto-tune] [--model-out PATH] [--backup]

Trains the HetTGN self-supervised model on all observations stored in the
PipelineStore. Reports per-epoch loss curves, then evaluates on val/test
splits (obs_type accuracy, time_delta MAE).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from agent.models.gnn.graph_builder import ENTITY_TYPES, OBSERVATION_TYPES
from agent.models.gnn.trainer import Trainer, TrainerConfig, evaluate
from agent.pipeline.store import PipelineStore

console = Console()


def _print_data_summary(store: PipelineStore) -> dict[str, int]:
    """Print a summary of the pipeline data and return counts."""
    entities = store.query_all_entities()
    observations = store.query_all_observations()
    links = store.query_all_entity_links()

    # Entity type counts
    type_counts: dict[str, int] = {}
    for e in entities:
        t = e["entity_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    # Observation type counts
    obs_counts: dict[str, int] = {}
    for o in observations:
        ot = o.get("observation_type", "unknown")
        obs_counts[ot] = obs_counts.get(ot, 0) + 1

    # Link type counts
    link_counts: dict[str, int] = {}
    for lk in links:
        lt = lk.get("link_type", "unknown")
        link_counts[lt] = link_counts.get(lt, 0) + 1

    # Date range
    timestamps = [
        o.get("observed_at", 0.0) for o in observations if o.get("observed_at", 0.0) > 0
    ]
    from datetime import datetime, timezone

    t_min = min(timestamps) if timestamps else 0
    t_max = max(timestamps) if timestamps else 0

    console.print("\n[bold cyan]═══ Data Summary ═══[/]")

    tbl = Table(title="Entities")
    tbl.add_column("Type", style="green")
    tbl.add_column("Count", justify="right")
    for t in sorted(type_counts):
        tbl.add_row(t, str(type_counts[t]))
    tbl.add_row("[bold]Total[/]", f"[bold]{sum(type_counts.values())}[/]")
    console.print(tbl)

    tbl = Table(title="Observations")
    tbl.add_column("Type", style="green")
    tbl.add_column("Count", justify="right")
    for ot in sorted(obs_counts, key=lambda x: -obs_counts[x]):
        tbl.add_row(ot, str(obs_counts[ot]))
    tbl.add_row("[bold]Total[/]", f"[bold]{sum(obs_counts.values())}[/]")
    console.print(tbl)

    tbl = Table(title="Entity Links")
    tbl.add_column("Type", style="green")
    tbl.add_column("Count", justify="right")
    for lt in sorted(link_counts, key=lambda x: -link_counts[x]):
        tbl.add_row(lt, str(link_counts[lt]))
    tbl.add_row("[bold]Total[/]", f"[bold]{sum(link_counts.values())}[/]")
    console.print(tbl)

    if timestamps:
        console.print(
            f"\n[dim]Date range: "
            f"{datetime.fromtimestamp(t_min, tz=timezone.utc).isoformat()} → "
            f"{datetime.fromtimestamp(t_max, tz=timezone.utc).isoformat()}[/]"
        )

    return {
        "entities": len(entities),
        "observations": len(observations),
        "links": len(links),
    }


def _print_model_summary(trainer: Trainer) -> None:
    """Print model architecture summary."""
    model = trainer.model
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    console.print("\n[bold cyan]═══ Model Summary ═══[/]")
    console.print(f"  Parameters: {total_params:,} total, {trainable:,} trainable")
    console.print(f"  Node types: {model.node_types}")
    console.print(f"  Memory nodes: {model.memory.num_nodes}")
    console.print(f"  Hidden dim: {model.hidden_dim}")


def _print_eval_results(metrics: dict[str, float], split: str) -> None:
    """Print evaluation metrics."""
    console.print(f"\n[bold cyan]═══ Evaluation ({split}) ═══[/]")
    console.print(
        f"  obs_type top-1 accuracy: {metrics['obs_type_acc_top1']:.4f} ({metrics['obs_type_acc_top1']*100:.1f}%)"
    )
    console.print(
        f"  obs_type top-5 accuracy: {metrics['obs_type_acc_top5']:.4f} ({metrics['obs_type_acc_top5']*100:.1f}%)"
    )
    console.print(
        f"  time_delta MAE: {metrics['time_delta_mae']:.1f} seconds ({metrics['time_delta_mae']/3600:.1f} hours)"
    )
    console.print(f"  num_predictions: {metrics['num_predictions']}")

    # Random baseline for obs_type
    random_top1 = 1.0 / max(len(OBSERVATION_TYPES), 1)
    random_top5 = min(5, len(OBSERVATION_TYPES)) / max(len(OBSERVATION_TYPES), 1)
    console.print(
        f"  [dim]Random baseline — top-1: {random_top1:.4f}, top-5: {random_top5:.4f}[/]"
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Retrain HetTGN on real pipeline observations"
    )
    parser.add_argument(
        "--db-path",
        default=".tirra_pipeline/pipeline.db",
        help="Path to PipelineStore SQLite DB (default: .tirra_pipeline/pipeline.db)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs (default: 20)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--auto-tune",
        action="store_true",
        help="Enable Kendall et al. 2018 uncertainty-weighted loss",
    )
    parser.add_argument(
        "--model-out",
        default=".tirra_pipeline/gnn_model.pt",
        help="Output path for trained model checkpoint",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Backup existing model before overwriting",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension (default: 64)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of HGT layers (default: 2)",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=86400.0,
        help="Temporal window size in seconds (default: 86400 = 1 day)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip val/test evaluation after training",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Only use observations after this date (ISO format, e.g. 2023-01-01). "
        "Useful for skipping sparse early data.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to use: 'cuda' or 'cpu'. Defaults to 'cuda' if available, else 'cpu'.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=".tirra_pipeline/checkpoints",
        help="Directory to save per-epoch checkpoints (default: .tirra_pipeline/checkpoints).",
    )
    parser.add_argument(
        "--resume",
        type=int,
        default=0,
        metavar="EPOCH",
        help="Resume training from this epoch number (loads epoch_NNN.pt from --checkpoint-dir).",
    )
    parser.add_argument(
        "--gdelt-frac",
        type=float,
        default=0.05,
        metavar="FRAC",
        help="Fraction of geopolitical_event obs to keep (0.0–1.0, default: 0.05). "
        "GDELT is 92%% of the DB; subsampling prevents OOM during snapshot pre-building.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=200,
        metavar="N",
        help="Cap training windows to last N time windows (default: 200). "
        "Limits peak RAM to O(N * graph_size). Set 0 to use all windows.",
    )
    parser.add_argument(
        "--max-ram-gb",
        type=float,
        default=8.0,
        metavar="GB",
        help="Hard virtual memory cap in GB (default: 8.0). "
        "Process raises MemoryError cleanly instead of hanging the laptop.",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        console.print(f"[red]ERROR: {db_path} not found.[/]")
        sys.exit(1)

    # ── RAM watchdog (prevents laptop hang on OOM) ────────────────────
    # Spawns a background thread that checks actual RSS every 10 s.
    # If RSS exceeds --max-ram-gb, logs the breach and exits cleanly
    # (SIGTERM → Python exit) instead of letting the OS swap the laptop
    # to a crawl. Uses psutil RSS (physical pages), NOT virtual address
    # space — RLIMIT_AS would also cap PyTorch's mmap'd library pages and
    # kills the process on import.
    if args.max_ram_gb > 0:
        try:
            import psutil as _psutil
            import threading as _threading
            import os as _os

            _ram_limit_bytes = int(args.max_ram_gb * 1024**3)
            _pid = _os.getpid()

            def _ram_watchdog() -> None:
                _proc = _psutil.Process(_pid)
                while True:
                    _threading.Event().wait(10)  # check every 10 s
                    try:
                        _rss = _proc.memory_info().rss
                    except _psutil.NoSuchProcess:
                        return
                    if _rss > _ram_limit_bytes:
                        console.print(
                            f"\n[bold red]RAM watchdog: RSS {_rss / 1e9:.1f} GB "
                            f"> limit {args.max_ram_gb:.1f} GB — exiting cleanly.[/]"
                        )
                        _os.kill(_pid, 15)  # SIGTERM → graceful exit
                        return

            _wd = _threading.Thread(
                target=_ram_watchdog, daemon=True, name="ram-watchdog"
            )
            _wd.start()
            console.print(
                f"  [dim]RAM watchdog: exit if RSS > {args.max_ram_gb:.1f} GB[/]"
            )
        except ImportError:
            console.print(
                "  [yellow]Warning: psutil not installed — RAM watchdog disabled.[/]"
            )

    model_out = Path(args.model_out)

    # Backup existing model if requested
    if args.backup and model_out.exists():
        backup_path = model_out.with_name(
            model_out.stem + "_pre_phase40" + model_out.suffix
        )
        shutil.copy2(model_out, backup_path)
        console.print(f"[yellow]Backed up existing model → {backup_path}[/]")

    store = PipelineStore(db_path=str(db_path))

    try:
        # ── Data Summary ──
        counts = _print_data_summary(store)
        if counts["observations"] == 0:
            console.print(
                "[red]ERROR: No observations in store. Nothing to train on.[/]"
            )
            sys.exit(1)

        # Parse --since into a timestamp
        obs_since = None
        if args.since:
            from datetime import datetime, timezone

            try:
                dt = datetime.fromisoformat(args.since)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                obs_since = dt.timestamp()
                console.print(
                    f"[yellow]Filtering observations since {args.since} (ts={obs_since:.0f})[/]"
                )
            except ValueError:
                console.print(f"[red]ERROR: Invalid --since date: {args.since}[/]")
                sys.exit(1)

        # ── Resolve device ──
        import torch as _torch

        if args.device is not None:
            device = args.device
        else:
            device = "cuda" if _torch.cuda.is_available() else "cpu"
        console.print(f"  device={device}")

        # ── Build Config ──
        config = TrainerConfig(
            hidden_dim=args.hidden_dim,
            memory_dim=args.hidden_dim,
            message_dim=args.hidden_dim,
            num_heads=2,
            num_layers=args.num_layers,
            learning_rate=args.lr,
            epochs=args.epochs,
            window_size=args.window_size,
            auto_tune_loss_weights=args.auto_tune,
            obs_since=obs_since,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            resume_from_epoch=args.resume,
            gdelt_subsample_frac=args.gdelt_frac,
            max_windows=args.max_windows,
        )

        console.print(f"\n[bold cyan]═══ Training Config ═══[/]")
        console.print(f"  epochs={config.epochs}, lr={config.learning_rate}")
        console.print(
            f"  hidden={config.hidden_dim}, layers={config.num_layers}, heads={config.num_heads}"
        )
        console.print(
            f"  window={config.window_size/3600:.1f}h, auto_tune={config.auto_tune_loss_weights}"
        )
        console.print(
            f"  Loss weights: obs={config.obs_type_weight}, dt={config.time_delta_weight}, "
            f"contr={config.contrastive_weight}, val={config.value_weight}, ret={config.return_weight}"
        )
        console.print(
            f"  GDELT subsample: {config.gdelt_subsample_frac:.3f} "
            f"(~{int(901704 * config.gdelt_subsample_frac):,} GDELT rows kept)"
        )
        console.print(
            f"  Max windows: {config.max_windows if config.max_windows > 0 else 'unlimited'} "
            f"(most recent training windows used)"
        )

        # ── Build Model ──
        trainer = Trainer(store, config)
        console.print("\n[dim]Building model from store contents...[/]")
        trainer.build_model()
        _print_model_summary(trainer)

        # ── Train ──
        console.print(f"\n[bold green]Training for {config.epochs} epochs...[/]")
        t0 = time.time()
        history = trainer.train()
        elapsed = time.time() - t0

        console.print(f"\n[bold green]Training complete![/] ({elapsed:.1f}s)")

        # Print loss curve
        tbl = Table(title="Loss Curve")
        tbl.add_column("Epoch", justify="right")
        tbl.add_column("Total", justify="right")
        tbl.add_column("obs_type", justify="right")
        tbl.add_column("time_delta", justify="right")
        tbl.add_column("contrastive", justify="right")
        tbl.add_column("value", justify="right")
        tbl.add_column("return", justify="right")
        for i in range(len(history["total"])):
            tbl.add_row(
                str(i + 1),
                f"{history['total'][i]:.4f}",
                f"{history['obs_type'][i]:.4f}",
                f"{history['time_delta'][i]:.4f}",
                f"{history['contrastive'][i]:.4f}",
                f"{history['value'][i]:.4f}",
                f"{history['return'][i]:.4f}" if "return" in history else "—",
            )
        console.print(tbl)

        if config.auto_tune_loss_weights:
            eff = trainer.effective_loss_weights()
            console.print(
                f"  Learned loss weights: obs={eff['obs_type']:.3f}, "
                f"dt={eff['time_delta']:.3f}, contr={eff['contrastive']:.3f}, "
                f"val={eff['value']:.3f}, ret={eff.get('return', 0):.3f}"
            )

        # ── Evaluate ──
        if not args.skip_eval:
            console.print("\n[dim]Evaluating on validation split...[/]")
            val_metrics = evaluate(trainer.model, store, config, split="val")
            _print_eval_results(val_metrics, "val")

            console.print("\n[dim]Evaluating on test split...[/]")
            test_metrics = evaluate(trainer.model, store, config, split="test")
            _print_eval_results(test_metrics, "test")

        # ── Save ──
        model_out.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(model_out))
        console.print(f"\n[green]Model saved → {model_out}[/]")

        # Final summary
        console.print(f"\n[bold cyan]═══ Summary ═══[/]")
        console.print(f"  Training: {config.epochs} epochs in {elapsed:.1f}s")
        console.print(
            f"  Final loss: {history['total'][-1]:.4f}"
            if history["total"]
            else "  No training done"
        )
        if not args.skip_eval:
            console.print(
                f"  Val top-1: {val_metrics['obs_type_acc_top1']:.4f}, "
                f"top-5: {val_metrics['obs_type_acc_top5']:.4f}"
            )
            console.print(
                f"  Test top-1: {test_metrics['obs_type_acc_top1']:.4f}, "
                f"top-5: {test_metrics['obs_type_acc_top5']:.4f}"
            )

    finally:
        store.close()

    sys.exit(0)


if __name__ == "__main__":
    main()
