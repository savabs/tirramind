#!/usr/bin/env python3
"""Extract recurring patterns and lessons from completed tasks and checkpoints.

Scans tasks/done/ and recent checkpoints for learnings, risks realized,
and recurring themes. Groups by topic tag.

Usage:
    python scripts/extract_patterns.py [--limit 20]
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DONE_DIR = ROOT / "tasks" / "done"
MEMORY_DIR = ROOT / "docs" / "memory"

# Sections that typically contain learnings
LEARNING_HEADERS = re.compile(
    r"^##\s+(What We Learned|Lessons|Risks|Observations|Key Insights|What Changed|Retrospective)",
    re.IGNORECASE | re.MULTILINE,
)

TAG_PATTERN = re.compile(r"^\s*-\s+(topic/\S+|layer/\S+)", re.MULTILINE)


def _extract_tags(content: str) -> list[str]:
    """Extract topic and layer tags from frontmatter."""
    return TAG_PATTERN.findall(content)


def _extract_learning_sections(content: str) -> list[str]:
    """Extract content under learning-related headers."""
    sections = []
    lines = content.splitlines()
    capturing = False
    current: list[str] = []

    for line in lines:
        if LEARNING_HEADERS.match(line):
            if current:
                sections.append("\n".join(current))
                current = []
            capturing = True
            current.append(line)
        elif capturing:
            # Stop at next ## header
            if line.startswith("## ") and not LEARNING_HEADERS.match(line):
                sections.append("\n".join(current))
                current = []
                capturing = False
            else:
                current.append(line)

    if current:
        sections.append("\n".join(current))

    return sections


def extract_patterns(limit: int = 20) -> str:
    """Scan completed tasks and checkpoints for patterns."""
    findings: dict[str, list[str]] = defaultdict(list)

    # Scan completed tasks
    if DONE_DIR.exists():
        for f in sorted(DONE_DIR.glob("*.md")):
            if f.name == ".gitkeep":
                continue
            content = f.read_text()
            tags = _extract_tags(content)
            sections = _extract_learning_sections(content)
            if sections:
                key = tags[0] if tags else "uncategorized"
                for s in sections:
                    findings[key].append(f"**{f.stem}**:\n{s}")

    # Scan recent checkpoints (last `limit`)
    checkpoints = sorted(MEMORY_DIR.glob("chat_checkpoint_*.md"), reverse=True)[:limit]
    for f in checkpoints:
        content = f.read_text()
        tags = _extract_tags(content)
        sections = _extract_learning_sections(content)
        if sections:
            key = tags[0] if tags else "checkpoints"
            for s in sections:
                findings[key].append(f"**{f.stem}**:\n{s}")

    if not findings:
        return "No completed tasks or checkpoints with learning sections found."

    # Format output
    output_lines = ["# Extracted Patterns\n"]
    for topic in sorted(findings.keys()):
        output_lines.append(f"\n## {topic}\n")
        for entry in findings[topic]:
            output_lines.append(entry)
            output_lines.append("")

    return "\n".join(output_lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract patterns from completed work."
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="Max recent checkpoints to scan."
    )
    args = parser.parse_args()

    result = extract_patterns(limit=args.limit)
    print(result)


if __name__ == "__main__":
    main()
