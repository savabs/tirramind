#!/usr/bin/env python3
"""Push V65/V66/V67 diagnostic smokes in one batch (sequential Kaggle pushes).

Kaggle runs one kernel version at a time; this script uploads code once, then
pushes three configs back-to-back. Save logs per version from the Kaggle UI or
provide them manually — only the latest lands in kaggle_logs_latest.txt.

Usage:
    python scripts/kaggle_batch_v65.py              # push all three
    python scripts/kaggle_batch_v65.py --dry-run    # print configs only
    python scripts/kaggle_batch_v65.py --from 66    # skip V65, push V66+V67
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from kaggle_launch import (  # noqa: E402
    CANONICAL_KERNEL_VERSION,
    upload_code_dataset,
    push_kernel,
    write_state,
    KERNEL_SLUG,
    KAGGLE_URL,
)
from kaggle_loop import (  # noqa: E402
    apply_config,
    bump_canonical_version,
    patch_notebook_config,
    require_kernel_version_sync,
)

BATCH_DIR = ROOT / ".tirra_pipeline" / "v65_batch"
EPOCHS = 10

_BASE: dict = {
    "phase": "50l",
    "resume_epoch": 0,
    "listnet_temperature": 0.1,
    "return_weight": 3.0,
    "obs_type_weight": 1.0,
    "time_delta_weight": 1.0,
    "value_weight": 1.0,
    "contrastive_weight": 1.0,
    "hidden_dim": 128,
    "num_layers": 2,
    "num_heads": 4,
    "epochs": EPOCHS,
    "window_size_h": 168,
    "gdelt_frac": 0.05,
    "defi_frac": 1.0,
    "n1_doctrine": False,
    "max_windows": 200,
    "vicreg_weight": 0.0,
    "use_contranorm": False,
    "use_log_loss": False,
    "eval_smoke": True,
    "run_full_backtest": False,
    "skip_eval": False,
    "skip_retrain_split_eval": True,
    "post_train_eval": True,
    "auto_tune": True,
    "return_log_var_max": "-0.693",
    "contrastive_log_var_min": "0.0",
    "direction_loss": True,
    "residual_returns": True,
    "raw_bypass_head": False,
    "freeze_backbone": False,
    "use_csrc_loss": True,
    "csrc_temperature": 0.1,
    "csrc_n_deciles": 5,
    "use_concat_head": True,
    "return_pred_clamp": 50.0,
    "use_concat_batchnorm": True,
    "use_pcgrad": False,
    "time_delta_nan_fix": True,
    "xsnorm_price_feats": True,
    "obs_type_ce_clamp": 20.0,
}

VARIANTS: list[dict] = [
    {
        **_BASE,
        "kernel_version": 65,
        "fix": "v65a_clamp_batchnorm_diag",
    },
    {
        **_BASE,
        "kernel_version": 66,
        "fix": "v66b_no_csrc_vicreg",
        "use_csrc_loss": False,
        "contrastive_weight": 0.0,
        "vicreg_weight": 0.1,
    },
    {
        **_BASE,
        "kernel_version": 67,
        "fix": "v67c_pcgrad",
        "use_pcgrad": True,
    },
]


def _save_manifest(entries: list[dict]) -> Path:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    path = BATCH_DIR / "manifest.json"
    path.write_text(json.dumps(entries, indent=2))
    return path


def push_variant(cfg: dict, *, upload: bool) -> dict:
    ver = int(cfg["kernel_version"])
    bump_canonical_version(ver)
    fp = patch_notebook_config(cfg)
    require_kernel_version_sync()
    if upload:
        upload_code_dataset()
    used_gpu = push_kernel(EPOCHS, ver, enable_gpu=True, fallback_cpu=True)
    entry = {
        "kernel_version": ver,
        "fix": cfg.get("fix"),
        "fingerprint": fp,
        "pushed_at": datetime.now(timezone.utc).isoformat(),
        "config": deepcopy(cfg),
        "log_hint": f".tirra_pipeline/kaggle_logs_v{ver}.txt",
        "enable_gpu": used_gpu,
    }
    write_state(
        {
            "kernel_slug": KERNEL_SLUG,
            "kernel_version": ver,
            "kaggle_ui_version": ver,
            "epochs": EPOCHS,
            "fingerprint": fp,
            "fix": cfg.get("fix"),
            "config": cfg,
            "kernel_status": "RUNNING",
            "batch": "v65_diagnostic",
            "enable_gpu": used_gpu,
        }
    )
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch push V65–V67 smokes")
    parser.add_argument(
        "--from",
        dest="from_version",
        type=int,
        default=65,
        help="Start at this kernel version (default 65)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configs without pushing",
    )
    args = parser.parse_args()

    to_push = [v for v in VARIANTS if v["kernel_version"] >= args.from_version]
    if not to_push:
        raise SystemExit(f"No variants with kernel_version >= {args.from_version}")

    print(f"Batch: {len(to_push)} variant(s) from V{args.from_version}")
    for v in to_push:
        print(f"  V{v['kernel_version']}: {v['fix']}")

    if args.dry_run:
        for v in to_push:
            print(json.dumps(v, indent=2))
        return

    if CANONICAL_KERNEL_VERSION >= to_push[0]["kernel_version"]:
        print(
            f"Note: repo canonical is V{CANONICAL_KERNEL_VERSION}; "
            f"batch will bump per variant."
        )

    manifest: list[dict] = []
    for i, cfg in enumerate(to_push):
        print(f"\n{'='*60}\nPushing V{cfg['kernel_version']} ({cfg['fix']})\n{'='*60}")
        entry = push_variant(cfg, upload=(i == 0))
        manifest.append(entry)
        print(f"  fingerprint={entry['fingerprint']}")
        print(f"  URL: {KAGGLE_URL}")

    path = _save_manifest(manifest)
    print(f"\n✓ Batch push complete. Manifest: {path}")
    print(
        "Kaggle queues versions — fetch logs per version from UI or:\n"
        "  kaggle kernels logs deeperisbetter/tirramind-phase50 > .tirra_pipeline/kaggle_logs_vNN.txt"
    )


if __name__ == "__main__":
    main()
