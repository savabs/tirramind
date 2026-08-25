#!/usr/bin/env python3
"""Generate or refresh **draft** chain briefs from ghost archive alert JSON.

Publish-ready briefs are edited by hand in ghost_archive/briefs/.

Usage:
    python scripts/generate_brief.py --all
    python scripts/generate_brief.py ghost_archive/alerts/2026-06-09_MP-1_EIA_REGIME_CFTC_001.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.quant.ghost_brief import DRAFT_BRIEFS_DIR, write_brief_for_alert  # noqa: E402

ARCHIVE_DIR = Path("ghost_archive/alerts")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate chain briefs from alert JSON")
    parser.add_argument("alert", nargs="?", help="Path to alert JSON")
    parser.add_argument("--all", action="store_true", help="Process all alerts in archive")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing briefs")
    args = parser.parse_args()

    paths: list[Path] = []
    if args.all:
        paths = sorted(ARCHIVE_DIR.glob("*.json"))
    elif args.alert:
        paths = [Path(args.alert)]
    else:
        parser.error("Provide alert path or --all")

    written = 0
    for path in paths:
        alert = json.loads(path.read_text(encoding="utf-8"))
        out = write_brief_for_alert(
            alert, briefs_dir=DRAFT_BRIEFS_DIR, overwrite=args.overwrite, draft=True
        )
        print(f"  → {out}")
        written += 1

    print(f"\n{written} draft(s) in {DRAFT_BRIEFS_DIR}/ — edit into ghost_archive/briefs/ to publish")


if __name__ == "__main__":
    main()
