#!/usr/bin/env python3
"""Push V68–V71 final pre-architecture smokes (eval-aligned + training variants).

All runs use the V65 training base (clamp50 + LayerNorm + CSRC) unless noted.
Eval uses GNN-ConcatReturnHead as primary gate (training-aligned scoring).

GPU quota errors auto-fallback to CPU via kaggle_launch.push_kernel().

Usage:
    python scripts/kaggle_batch_v68.py
    python scripts/kaggle_batch_v68.py --from 70
    python scripts/kaggle_batch_v68.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from kaggle_launch import (  # noqa: E402
    CANONICAL_KERNEL_VERSION,
    KERNEL_SLUG,
    KAGGLE_URL,
    push_kernel,
    upload_code_dataset,
    write_state,
)
from kaggle_loop import (  # noqa: E402
    bump_canonical_version,
    patch_notebook_config,
    require_kernel_version_sync,
)

BATCH_DIR = ROOT / ".tirra_pipeline" / "v68_batch"
EPOCHS = 10
PUSH_DELAY_SEC = 3

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
    "primary_ic_strategy": "GNN-ConcatReturnHead",
    "time_delta_nan_fix": True,
    "xsnorm_price_feats": True,
    "obs_type_ce_clamp": 20.0,
}

VARIANTS: list[dict] = [
    {
        **_BASE,
        "kernel_version": 68,
        "fix": "v68a_eval_concat_gate",
    },
    {
        **_BASE,
        "kernel_version": 69,
        "fix": "v68b_listnet_tau05",
        "listnet_temperature": 0.5,
    },
    {
        **_BASE,
        "kernel_version": 70,
        "fix": "v68c_pcgrad_safe",
        "use_pcgrad": True,
    },
    {
        **_BASE,
        "kernel_version": 71,
        "fix": "v68d_tau05_pcgrad",
        "listnet_temperature": 0.5,
        "use_pcgrad": True,
    },
]


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
        "enable_gpu": used_gpu,
        "log_hint": f".tirra_pipeline/kaggle_logs_v{ver}.txt",
    }
    write_state(
        {
            "kernel_slug": KERNEL_SLUG,
            "kernel_version": ver,
            "epochs": EPOCHS,
            "fingerprint": fp,
            "fix": cfg.get("fix"),
            "config": cfg,
            "kernel_status": "RUNNING",
            "enable_gpu": used_gpu,
            "batch": "v68_final_smokes",
        }
    )
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch push V68–V71 smokes")
    parser.add_argument("--from", dest="from_version", type=int, default=68)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    to_push = [v for v in VARIANTS if v["kernel_version"] >= args.from_version]
    if not to_push:
        raise SystemExit(f"No variants with kernel_version >= {args.from_version}")

    print(f"V68 batch: {len(to_push)} variant(s) from V{args.from_version}")
    for v in to_push:
        print(f"  V{v['kernel_version']}: {v['fix']}")

    if args.dry_run:
        for v in to_push:
            print(json.dumps(v, indent=2))
        return

    if CANONICAL_KERNEL_VERSION >= to_push[0]["kernel_version"]:
        print(f"Note: repo canonical V{CANONICAL_KERNEL_VERSION}; batch bumps per variant.")

    manifest: list[dict] = []
    for i, cfg in enumerate(to_push):
        if i > 0:
            time.sleep(PUSH_DELAY_SEC)
        print(f"\n{'='*60}\nPushing V{cfg['kernel_version']} ({cfg['fix']})\n{'='*60}")
        entry = push_variant(cfg, upload=(i == 0))
        manifest.append(entry)
        accel = "GPU" if entry["enable_gpu"] else "CPU"
        print(f"  fingerprint={entry['fingerprint']}  accel={accel}")
        print(f"  URL: {KAGGLE_URL}")

    path = BATCH_DIR / "manifest.json"
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    print(f"\n✓ Batch complete. Manifest: {path}")
    print("Save logs per version from Kaggle UI when each run finishes.")


if __name__ == "__main__":
    main()
