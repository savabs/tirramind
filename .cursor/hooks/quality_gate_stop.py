#!/usr/bin/env python3
"""Cursor stop hook: nudge agent when quality gate checks would fail."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tirramind_context import collect_quality_nudges

MAX_LOOPS = 2


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        print("{}")
        return

    status = payload.get("status", "")
    loop_count = int(payload.get("loop_count", 0) or 0)

    if status != "completed" or loop_count >= MAX_LOOPS:
        print("{}")
        return

    nudges = collect_quality_nudges()
    if not nudges:
        print("{}")
        return

    followup = (
        "Quality gate nudge — before ending this session, resolve the following:\n"
        + "\n".join(nudges)
        + "\n\nRun `make quality-gate-fast` to verify. "
        "If work is intentionally incomplete, say so and write a checkpoint."
    )
    print(json.dumps({"followup_message": followup}))


if __name__ == "__main__":
    main()
