#!/usr/bin/env python3
"""Cursor sessionStart hook: inject active tasks and latest checkpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tirramind_context import build_session_context


def main() -> None:
    try:
        json.load(sys.stdin)
    except json.JSONDecodeError:
        print("{}")
        return

    context = build_session_context()
    print(json.dumps({"additional_context": context}))


if __name__ == "__main__":
    main()
