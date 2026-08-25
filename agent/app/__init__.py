"""Application entrypoints for the Intelligence engine (console scripts).

Provides thin, importable wrappers so `pyproject.toml` can register stable
commands (`tirra-engine`, `tirra-serve`, `tirra-brief`) without depending on
`scripts/` internals.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _at_repo_root() -> None:
    """Anchor all state to the repo root so paths are predictable from any cwd."""
    os.chdir(_ROOT)


def engine() -> int:
    """Run the unified engine (collect → brief → deliver). Default = just run it."""
    _at_repo_root()
    from scripts.tirra_engine import main
    return main()


def serve() -> int:
    """Run the brief HTTP server."""
    import argparse

    _at_repo_root()
    from agent.brief_server import serve as _serve

    parser = argparse.ArgumentParser(description="Serve the delivered Intelligence Brief")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--out", type=str, default=".tirra_delivery")
    args = parser.parse_args()
    _serve(out_dir=args.out, port=args.port, host=args.host)
    return 0


def brief() -> int:
    """Build + deliver one Intelligence Brief."""
    _at_repo_root()
    from scripts.intelligence_brief import main as _brief_main
    return _brief_main()


__all__ = ["engine", "serve", "brief"]
