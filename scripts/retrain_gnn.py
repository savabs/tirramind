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
        "--listnet",
        action="store_true",
        help="Use ListNet ranking loss for return head instead of Huber (Phase 41b).",
    )
    parser.add_argument(
        "--return-log-var-max",
        type=float,
        default=0.0,
        help=(
            "Upper clamp for the return component log-variance under --auto-tune. "
            "Default 0.0 ensures the return weight stays >= exp(-0)=1.0 so "
            "auto-tune cannot silence the return head (Phase 41b fix)."
        ),
    )
    parser.add_argument(
        "--return-weight",
        type=float,
        default=1.0,
        help="Weight for the instrument return auxiliary loss (default: 1.0). "
        "Increase (e.g. 2.0–4.0) when dt_loss dominates the return head signal. "
        "Has no effect when --auto-tune is active (auto-tune learns its own weights).",
    )
    parser.add_argument(
        "--listnet-temperature",
        type=float,
        default=1.0,
        metavar="TAU",
        help="ListNet softmax temperature tau (default: 1.0). "
        "Lower = sharper target distribution → stronger IC gradient. "
        "Try 0.5 when return loss is flat despite balanced losses.",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        metavar="FILE",
        help="JSON file produced by auto_improve.py with recommended flag overrides. "
        "Applied AFTER argparse so it can override any flag without re-specifying all args. "
        "Keys must match CLI flag names with '--' stripped and '-' replaced by '_' "
        "(e.g. 'return_weight', 'lr', 'gdelt_frac'). "
        "Special key 'resume_epoch' maps to --resume.",
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
        "--num-heads",
        type=int,
        default=2,
        help="Number of HGT attention heads (default: 2). Must be a divisor of --hidden-dim.",
    )
    parser.add_argument(
        "--direction-loss",
        action="store_true",
        help="Add binary cross-entropy direction loss (sign prediction) alongside return loss. "
        "Adds cfg.direction_loss_weight * BCE(sign(pred), sign(target)) to total loss.",
    )
    parser.add_argument(
        "--direction-loss-weight",
        type=float,
        default=0.3,
        help="Weight for direction BCE loss when --direction-loss is active (default: 0.3).",
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
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        metavar="PROJECT",
        help="Weights & Biases project name. When set, streams per-epoch losses "
        "and effective loss weights to wandb in real time. Requires 'wandb' "
        "package (pip install wandb). If omitted, W&B logging is disabled.",
    )
    parser.add_argument(
        "--wandb-run",
        type=str,
        default=None,
        metavar="RUN_NAME",
        help="W&B run display name (e.g. 'h-a-epoch31-40'). Auto-generated by "
        "W&B if omitted.",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default=None,
        metavar="TAG1,TAG2",
        help="Comma-separated list of W&B tags for the run (e.g. 'h-a,phase43').",
    )
    parser.add_argument(
        "--auto-improve",
        action="store_true",
        help="After training completes, run auto_improve.py --no-watch to check for "
        "IC stagnation and write a trigger file + triage report if detected. "
        "Requires scripts/auto_improve.py and scripts/auto_research.py.",
    )
    args = parser.parse_args()

    # ── Apply agent-recommended config overrides from --config-file ──────────
    # auto_improve.py writes next_config.json with flag_overrides. Loading it
    # here means the agent can recommend "--return-weight 2.0" and the next
    # run picks it up without the user having to re-type all flags.
    if args.config_file is not None:
        _cf = Path(args.config_file)
        if not _cf.exists():
            console.print(f"[red]ERROR: --config-file {_cf} not found.[/]")
            sys.exit(1)
        import json as _json_cfg  # noqa: PLC0415

        _overrides = _json_cfg.loads(_cf.read_text())
        _flag_map = _overrides.get("flag_overrides", {})
        _remove = set(_overrides.get("remove_flags", []))
        for _flag, _val in _flag_map.items():
            # '--return-weight' → 'return_weight'; '--lr' → 'lr'
            _attr = _flag.lstrip("-").replace("-", "_")
            if hasattr(args, _attr):
                setattr(args, _attr, _val)
                console.print(f"  [cyan][config-file] override: {_flag}={_val}[/]")
        for _flag in _remove:
            _attr = _flag.lstrip("-").replace("-", "_")
            if hasattr(args, _attr):
                setattr(args, _attr, False)  # store_true flags only
                console.print(f"  [cyan][config-file] disabled: {_flag}[/]")
        if "resume_epoch" in _overrides:
            args.resume = int(_overrides["resume_epoch"])
            console.print(f"  [cyan][config-file] resume_epoch={args.resume}[/]")
        console.print(
            f"  [cyan][config-file] applied overrides from {_cf.name} (pattern: {_overrides.get('pattern', '?')})[/]"
        )

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
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            learning_rate=args.lr,
            epochs=args.epochs,
            window_size=args.window_size,
            auto_tune_loss_weights=args.auto_tune,
            use_listnet_return_loss=args.listnet,
            return_log_var_max=args.return_log_var_max,
            obs_since=obs_since,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            resume_from_epoch=args.resume,
            gdelt_subsample_frac=args.gdelt_frac,
            max_windows=args.max_windows,
            use_direction_loss=args.direction_loss,
            direction_loss_weight=args.direction_loss_weight,
            return_weight=args.return_weight,
            listnet_temperature=args.listnet_temperature,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run,
            wandb_tags=(
                [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
                if args.wandb_tags
                else None
            ),
        )

        # ── Write run_config.json (agent reads this to know what was tried) ──
        if args.checkpoint_dir:
            import json as _json_rc  # noqa: PLC0415
            import time as _time_rc  # noqa: PLC0415

            _rc_path = Path(args.checkpoint_dir) / "run_config.json"
            _rc_path.parent.mkdir(parents=True, exist_ok=True)
            _rc_path.write_text(
                _json_rc.dumps(
                    {
                        "started": _time_rc.strftime("%Y-%m-%dT%H:%M:%S"),
                        "resume_from_epoch": config.resume_from_epoch,
                        "config": {
                            "lr": config.learning_rate,
                            "return_weight": float(config.return_weight),
                            "gdelt_frac": float(config.gdelt_subsample_frac),
                            "listnet_temp": float(config.listnet_temperature),
                            "auto_tune": bool(config.auto_tune_loss_weights),
                            "use_listnet": bool(config.use_listnet_return_loss),
                            "epochs": config.epochs,
                            "hidden_dim": config.hidden_dim,
                            "direction_loss": bool(config.use_direction_loss),
                        },
                    },
                    indent=2,
                )
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

        # ── Save immediately after training (before any display that could crash) ──
        model_out.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(model_out))
        console.print(f"\n[green]Model saved → {model_out}[/]")

        # Print loss curve
        tbl = Table(title="Loss Curve")
        tbl.add_column("Epoch", justify="right")
        tbl.add_column("Total", justify="right")
        tbl.add_column("obs_type", justify="right")
        tbl.add_column("time_delta", justify="right")
        tbl.add_column("contrastive", justify="right")
        tbl.add_column("value", justify="right")
        tbl.add_column("return", justify="right")
        _ret_hist = history.get("return", [])
        for i in range(len(history["total"])):
            tbl.add_row(
                str(i + 1),
                f"{history['total'][i]:.4f}",
                f"{history['obs_type'][i]:.4f}",
                f"{history['time_delta'][i]:.4f}",
                f"{history['contrastive'][i]:.4f}",
                f"{history['value'][i]:.4f}",
                f"{_ret_hist[i]:.4f}" if i < len(_ret_hist) else "—",
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

    # ── Auto-improve: post-training stagnation check ──────────────────────────
    if args.auto_improve and args.checkpoint_dir:
        console.print("\n[cyan]Running auto-improve stagnation check...[/]")
        import subprocess as _sp  # noqa: PLC0415

        _ai_script = Path(__file__).parent / "auto_improve.py"
        _result = _sp.run(
            [
                sys.executable,
                str(_ai_script),
                "--checkpoint-dir",
                args.checkpoint_dir,
                "--no-watch",
            ],
            capture_output=False,
        )
        if _result.returncode == 2:
            console.print(
                "[yellow]Auto-improve: stagnation detected — trigger + triage written to knowledge/[/]"
            )
            console.print(
                "[yellow]  → Invoke 'apply training fix' in Copilot chat to apply the patch.[/]"
            )
        elif _result.returncode == 0:
            console.print(
                "[green]Auto-improve: return loss improving — no action needed.[/]"
            )

    sys.exit(0)


if __name__ == "__main__":
    main()
