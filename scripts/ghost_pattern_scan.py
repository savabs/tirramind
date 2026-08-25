#!/usr/bin/env python3
"""Scan MP-1 ghost chain templates against pipeline.db and write archive alerts.

Usage:
    python scripts/ghost_pattern_scan.py
    python scripts/ghost_pattern_scan.py --dry-run
    python scripts/ghost_pattern_scan.py --template templates/ghost_chains/mp1/ais_eia_cftc.yaml
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.quant.ghost_brief import DRAFT_BRIEFS_DIR, write_brief_for_alert  # noqa: E402
from agent.quant.ghost_chains import (  # noqa: E402
    list_mp1_templates,
    load_chain_template,
    scan_templates,
)

DB_PATH = Path(".tirra_pipeline/pipeline.db")
ARCHIVE_DIR = Path("ghost_archive/alerts")


def _next_alert_seq(day: str, micro_playground: str) -> int:
    """Return next 001-style sequence for alerts issued on `day`."""
    prefix = f"{day}_{micro_playground}_"
    seqs = []
    for path in ARCHIVE_DIR.glob(f"{prefix}*.json"):
        suffix = path.stem.split("_")[-1]
        if suffix.isdigit():
            seqs.append(int(suffix))
    return max(seqs, default=0) + 1


def _template_for_id(tmpl_id: str, templates: list[Path]):
    for path in templates:
        tmpl = load_chain_template(path)
        if tmpl.id == tmpl_id:
            return tmpl
    fallback = Path(f"templates/ghost_chains/mp1/{tmpl_id}.yaml")
    if fallback.exists():
        return load_chain_template(fallback)
    return None


def _existing_templates_today(day: str) -> set[str]:
    templates: set[str] = set()
    for path in ARCHIVE_DIR.glob(f"{day}_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        tmpl = data.get("chain_template")
        if tmpl:
            templates.add(str(tmpl))
    return templates


def main() -> None:
    parser = argparse.ArgumentParser(description="Ghost pattern chain scanner (MP-1)")
    parser.add_argument("--db-path", default=str(DB_PATH))
    parser.add_argument("--template", action="append", help="Single template YAML (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="Print matches, do not write files")
    parser.add_argument("--as-of", default=None, help="ISO datetime for scan (default: now UTC)")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}")
        print("Run daily_collection or backfill scripts first.")
        sys.exit(1)

    if args.template:
        templates = [Path(p) for p in args.template]
    else:
        templates = list_mp1_templates()

    if not templates:
        print("ERROR: No templates found under templates/ghost_chains/mp1/")
        sys.exit(1)

    as_of = (
        datetime.fromisoformat(args.as_of.replace("Z", "+00:00"))
        if args.as_of
        else datetime.now(timezone.utc)
    )

    con = sqlite3.connect(str(db_path))
    print(f"Ghost Pattern Scan — {len(templates)} template(s), as_of={as_of.isoformat()}")
    print("=" * 60)

    alerts = scan_templates(con, templates, as_of)
    con.close()

    day = as_of.strftime("%Y-%m-%d")
    already = _existing_templates_today(day)
    written = 0

    if not alerts:
        print("No chain matches above threshold.")
        for path in templates:
            tmpl = load_chain_template(path)
            print(f"  — {tmpl.id}: no match (check data density / thresholds)")
        sys.exit(0)

    for alert in alerts:
        tmpl_id = alert["chain_template"]
        if tmpl_id in already:
            print(f"SKIP: {tmpl_id} already has an alert for {day}")
            continue

        print(f"MATCH: {tmpl_id} score={alert['chain_score']}")
        for node in alert["nodes"]:
            print(f"    {node['entity']}: z={node['z']} @ {node['observed_at']}")

        seq = _next_alert_seq(day, alert["micro_playground"])
        alert["alert_id"] = (
            f"{day}_{alert['micro_playground']}_{tmpl_id.upper()}_{seq:03d}"
        )

        if args.dry_run:
            print(json.dumps(alert, indent=2))
            continue

        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        out = ARCHIVE_DIR / f"{alert['alert_id']}.json"
        out.write_text(json.dumps(alert, indent=2) + "\n", encoding="utf-8")
        tmpl = _template_for_id(tmpl_id, templates)
        brief_path = write_brief_for_alert(
            alert, briefs_dir=DRAFT_BRIEFS_DIR, template=tmpl, draft=True
        )
        already.add(tmpl_id)
        written += 1
        print(f"  → wrote {out}")
        print(f"  → draft {brief_path}")

    if written:
        print(f"\n{written} alert(s) written to {ARCHIVE_DIR}/")
    else:
        print("\nNo new alerts written (matches skipped or dry-run).")


if __name__ == "__main__":
    main()
