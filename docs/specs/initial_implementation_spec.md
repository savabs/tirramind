---
title: "Spec: initial_implementation"
tags:
  - doc/spec
---

# Spec: initial_implementation

## Goal

Get TirraMind agent running end-to-end: environment setup, memory persistence wired, config validation, and semantic memory populated during runs. After this, `python -m agent "search for X"` should work.

## Files Affected

| File | Action |
|------|--------|
| `agent/core/orchestrator.py` | Modify — wire memory persistence paths, populate semantic memory, improve tool arg inference |
| `agent/config/settings.py` | Modify — add startup validation method |
| `agent/cli.py` | Modify — add dotenv loading + config validation on startup |
| `.env.example` | Create — document required env vars |
| `.gitignore` | Create — exclude .env, __pycache__, .tirra_memory |
| `pyproject.toml` | Modify — add python-dotenv dependency |

## Implementation Steps

### 1. Add `.gitignore`
Standard Python gitignore + `.env` + `.tirra_memory/`

### 2. Add `.env.example`
Document all `TIRRA_*` variables with placeholder values.

### 3. Add `python-dotenv` to dependencies
In `pyproject.toml`, add `python-dotenv>=1.0` to dependencies.

### 4. Wire dotenv loading in `agent/cli.py`
Load `.env` at startup before `AgentConfig.from_env()`.

### 5. Add config validation in `agent/config/settings.py`
Add `AgentConfig.validate()` method that checks API key presence for OpenAI provider.

### 6. Call validation in `agent/cli.py`
Call `config.validate()` before running and print helpful error if misconfigured.

### 7. Wire memory persistence in `agent/core/orchestrator.py`
Pass `config.memory_dir` to `EpisodicMemory` and `SemanticMemory` constructors with appropriate file paths.

### 8. Populate semantic memory during execution
After successful tool executions and research, store key findings as `Fact` objects in semantic memory.

### 9. Improve `_infer_tool_args` context
Include goal, recent episode history, and previous task results in the prompt to the LLM.

## Edge Cases

- Missing `.env` file → should work (env vars can be set directly)
- Ollama provider → no API key required (should skip validation)
- Memory directory doesn't exist → should be auto-created
- Empty semantic memory on first run → already handled (returns "(no facts stored)")

## Testing Plan

1. `python -m agent --help` → should print usage
2. `python -m agent "list files in the current directory"` → should run the full pipeline with `list_directory` tool and return results
3. After run, `.tirra_memory/` should contain episodic.jsonl and semantic.jsonl files

---

## Related

- [[initial_implementation|Research: Initial Implementation]]
