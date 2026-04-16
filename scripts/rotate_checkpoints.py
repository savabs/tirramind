#!/usr/bin/env python3
"""Rotate old checkpoint files to keep the memory directory manageable.

Keeps the N most recent checkpoints. Archives older ones by extracting
title + date + first content section into a yearly archive file.

Usage:
    python scripts/rotate_checkpoints.py [--keep 15] [--dry-run]
"""

import argparse
import datetime
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEMORY_DIR = ROOT / "docs" / "memory"

CHECKPOINT_PATTERN = re.compile(r"^chat_checkpoint_(\d{4}-\d{2}-\d{2})")


def _parse_date(filename: str) -> datetime.date | None:
    """Extract date from checkpoint filename."""
    m = CHECKPOINT_PATTERN.search(filename)
    if m:
        try:
            return datetime.date.fromisoformat(m.group(1))
        except ValueError:
            return None
    return None


def _extract_summary(filepath: Path, max_lines: int = 15) -> str:
    """Extract the title and first few content lines for archiving."""
    lines = filepath.read_text().splitlines()
    summary_lines = []
    in_frontmatter = False
    content_started = False

    for line in lines:
        if line.strip() == "---":
            if not in_frontmatter:
                in_frontmatter = True
                continue
            else:
                in_frontmatter = False
                continue
        if in_frontmatter:
            if line.startswith("title:"):
                summary_lines.append(line.split(":", 1)[1].strip().strip('"'))
            continue
        if not content_started and line.startswith("# "):
            content_started = True
        if content_started:
            summary_lines.append(line)
            if len(summary_lines) >= max_lines:
                break

    return "\n".join(summary_lines) if summary_lines else filepath.stem


def rotate(keep: int = 15, dry_run: bool = False) -> dict:
    """Rotate checkpoints. Returns stats dict."""
    if not MEMORY_DIR.exists():
        return {"kept": 0, "archived": 0, "skipped": 0}

    # Find all checkpoint files
    checkpoints: list[tuple[datetime.date, Path]] = []
    for f in MEMORY_DIR.glob("chat_checkpoint_*.md"):
        dt = _parse_date(f.name)
        if dt:
            checkpoints.append((dt, f))

    # Sort by date descending (newest first), then by name for same-day ordering
    checkpoints.sort(key=lambda x: (x[0], x[1].name), reverse=True)

    to_keep = checkpoints[:keep]
    to_archive = checkpoints[keep:]

    if not to_archive:
        return {"kept": len(to_keep), "archived": 0, "skipped": 0}

    # Group archives by year
    by_year: dict[int, list[tuple[datetime.date, Path]]] = {}
    for dt, fp in to_archive:
        by_year.setdefault(dt.year, []).append((dt, fp))

    archived_count = 0
    for year, items in sorted(by_year.items()):
        archive_path = MEMORY_DIR / f"checkpoint_archive_{year}.md"

        if not dry_run:
            # Append to archive file
            with open(archive_path, "a") as af:
                if archive_path.stat().st_size == 0 if archive_path.exists() else True:
                    af.write(
                        f'---\ntitle: "Checkpoint Archive {year}"\ntags:\n  - doc/memory\n---\n\n# Checkpoint Archive {year}\n\n'
                    )

                for dt, fp in sorted(items, key=lambda x: x[0]):
                    summary = _extract_summary(fp)
                    af.write(f"\n## {dt.isoformat()} — {fp.stem}\n\n{summary}\n\n---\n")
                    fp.unlink()
                    archived_count += 1
        else:
            archived_count += len(items)
            for dt, fp in items:
                print(f"  [DRY RUN] Would archive: {fp.name}")

    return {"kept": len(to_keep), "archived": archived_count, "skipped": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Rotate old checkpoint files.")
    parser.add_argument(
        "--keep",
        type=int,
        default=15,
        help="Number of recent checkpoints to keep (default: 15)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without deleting.",
    )
    args = parser.parse_args()

    stats = rotate(keep=args.keep, dry_run=args.dry_run)
    print(f"Checkpoints: {stats['kept']} kept, {stats['archived']} archived")


if __name__ == "__main__":
    main()
