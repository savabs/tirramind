---
title: "Research: cc-cache-fix (Rangizingo/cc-cache-fix)"
tags:
  - doc/research
---

# Research: cc-cache-fix (Rangizingo/cc-cache-fix)

**Source:** https://github.com/Rangizingo/cc-cache-fix/tree/main
**Stars:** ~410 | **Forks:** ~132 | **Language:** Python 86%, Shell 10%, PowerShell 4%
**Contributors:** Rangizingo (Peter Blanco), Claude (co-authored)
**License:** No license file found (unlicensed / all rights reserved by default)
**Date reviewed:** 2025-04-02

---

## What It Is

A patch + test toolkit that fixes known caching bugs in **Claude Code** (Anthropic's CLI coding agent, `@anthropic-ai/claude-code` npm package). It monkey-patches the minified `cli.js` file to restore proper prompt-caching behavior.

The repo creates a separate `claude-patched` wrapper command, leaving stock `claude` untouched.

## Problems It Fixes

Three patches are applied to `cli.js`:

1. **db8 attachment filter** — Persists `deferred_tools_delta` and `mcp_instructions_delta` attachments in the session JSONL so the cache prefix is reconstructed correctly on resume. Without this, resuming a session invalidates the cache.

2. **Fingerprint meta skip** — Ensures the first-message hash used in the attribution header ignores injected meta messages, keeping the cache key stable across turns.

3. **Force 1-hour cache TTL** — Bypasses subscription/feature-flag check so all cache markers use 1-hour TTL instead of the default 5 minutes. This dramatically improves cache hit rates.

## How It Works

- `patch.py` / `patches/apply-patches.py`: Finds Claude Code's `cli.js`, backs it up, applies regex-based patches to the minified JS source.
- `test_cache.py`: Tests whether resume cache is working (measures read ratio; healthy = ~65-70%).
- `usage_audit.py`: Audits real session cache efficiency.
- `smoke_check.sh`: End-to-end installer + test + summary script.
- Installer scripts for Linux, macOS, Windows.

## License / Reuse Constraints

**No license file exists in the repository.** Under GitHub's ToS and copyright law, this means the code is all-rights-reserved by default. We cannot copy, port, or redistribute the code. We can only reference it conceptually.

## Relevance to TirraMind

### Direct applicability: **None**

- TirraMind is a quant information-arbitrage system. It does not depend on, embed, or extend Claude Code's CLI. There is no `cli.js`, `@anthropic-ai/claude-code`, or npm dependency anywhere in the repository.
- TirraMind's LLM integration (`agent/reasoning/llm_client.py`) uses the OpenAI-compatible API (Groq / Ollama) — a completely different interface from Claude Code's CLI.
- The cache bugs this repo fixes are specific to Anthropic's prompt-caching implementation inside the Claude Code CLI tool, not the Anthropic API in general.

### Indirect / developer-experience applicability: **Low / situational**

- If a TirraMind developer personally uses Claude Code as their IDE agent (e.g., via Copilot or the `claude` CLI), this patch toolkit could improve their personal development experience by fixing cache regressions and reducing API costs.
- This is a developer-tooling concern, not a TirraMind architecture concern. It would never be a dependency of the project.

## Conclusion

**Not useful for TirraMind's codebase.** The repo patches bugs in Anthropic's Claude Code CLI tool. TirraMind doesn't use Claude Code as a dependency — it has its own LLM client using OpenAI-compatible APIs. The repo is only relevant if you personally use `claude` CLI for coding and want better caching behavior during your development sessions.

## Related

- [[project_memory]]
