#!/usr/bin/env python3
"""Bootstrap and sync local MCP config from .cursor/mcp.json.example + .env keys."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / ".cursor" / "mcp.json.example"
CURSOR_MCP = ROOT / ".cursor" / "mcp.json"
VSCODE_MCP = ROOT / ".vscode" / "mcp.json"
ENV_FILE = ROOT / ".env"

# .env key → (mcp server id, mcp env var name)
_ENV_TO_MCP: list[tuple[str, str, str]] = [
    ("TAVILY_API_KEY", "tavily", "TAVILY_API_KEY"),
    ("CONTEXT7_API_KEY", "context7", "CONTEXT7_API_KEY"),
    ("WOLFRAM_APP_ID", "wolfram-alpha", "WOLFRAM_APP_ID"),
    ("WOLFRAM_ALPHA_APP_ID", "wolfram-alpha", "WOLFRAM_APP_ID"),
]

_PLACEHOLDER_PREFIXES = ("YOUR_",)


def _load_env() -> dict[str, str]:
    env_vars: dict[str, str] = {}
    if not ENV_FILE.exists():
        return env_vars
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env_vars[key.strip().upper()] = value.strip().strip('"').strip("'")
    return env_vars


def _load_example() -> dict:
    if not EXAMPLE.exists():
        raise FileNotFoundError(f"Missing template: {EXAMPLE}")
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def _load_or_template() -> dict:
    if CURSOR_MCP.exists():
        return json.loads(CURSOR_MCP.read_text(encoding="utf-8"))
    return _load_example()


def _ensure_server(config: dict, server_id: str, template: dict) -> None:
    servers = config.setdefault("mcpServers", {})
    if server_id not in servers and server_id in template.get("mcpServers", {}):
        servers[server_id] = json.loads(json.dumps(template["mcpServers"][server_id]))


def _sync_env_keys(config: dict, env_vars: dict[str, str]) -> list[str]:
    """Inject .env values into mcp.json env blocks. Returns list of synced fields."""
    synced: list[str] = []
    servers = config.setdefault("mcpServers", {})

    for env_key, server_id, mcp_env_key in _ENV_TO_MCP:
        value = env_vars.get(env_key.upper())
        if not value:
            continue
        server = servers.setdefault(server_id, {})
        server_env = server.setdefault("env", {})
        if server_env.get(mcp_env_key) != value:
            server_env[mcp_env_key] = value
            synced.append(f"{server_id}.{mcp_env_key}")

    # Drop placeholder API keys that cause "invalid key" errors at runtime
    for server_id, server in list(servers.items()):
        env_block = server.get("env")
        if not isinstance(env_block, dict):
            continue
        for key, val in list(env_block.items()):
            if isinstance(val, str) and val.startswith(_PLACEHOLDER_PREFIXES):
                if key == "CONTEXT7_API_KEY" and not env_vars.get("CONTEXT7_API_KEY"):
                    del env_block[key]
                    synced.append(f"{server_id}.{key} (removed placeholder)")

    return synced


def _write_json(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    try:
        template = _load_example()
        config = _load_or_template()
        env_vars = _load_env()

        # Ensure all template servers exist (e.g. wolfram-alpha added after initial setup)
        for server_id in template.get("mcpServers", {}):
            _ensure_server(config, server_id, template)

        created = not CURSOR_MCP.exists()
        synced = _sync_env_keys(config, env_vars)
        _write_json(CURSOR_MCP, config)

        _write_json(VSCODE_MCP, config)

        if created:
            print(f"Created {CURSOR_MCP.relative_to(ROOT)}")
        else:
            print(f"Updated {CURSOR_MCP.relative_to(ROOT)}")

        if synced:
            print("Synced from .env:")
            for item in synced:
                print(f"  - {item}")
        else:
            print("No .env keys synced (add TAVILY_API_KEY / WOLFRAM_APP_ID to .env)")

        print("\nServers configured:")
        for name in config.get("mcpServers", {}):
            print(f"  - {name}")

        print("\nNext: restart Cursor → Settings → Tools & MCP → verify green status")
        print("Git MCP: pip install mcp-server-git")
        return 0
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"setup_mcp: ERROR — {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
