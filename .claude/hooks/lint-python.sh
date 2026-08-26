#!/usr/bin/env bash
# PostToolUse — lint the Python file that was just edited.
#
# Deliberately `ruff check` (report) rather than `ruff format` (rewrite).
# Auto-reformatting on every edit would rewrite untouched lines in a repo that
# currently carries ~93 uncommitted files, burying real changes in style churn
# and making review harder. Reporting surfaces genuine problems without that
# cost. Run `.venv/bin/ruff format <file>` explicitly when you actually want it.
#
# Never blocks — a lint finding is information, not a reason to stop.
set -uo pipefail

INPUT=$(cat)
FILE_PATH=$(printf '%s' "$INPUT" | jq -r '.tool_input.file_path // .tool_input.path // empty')

[[ "$FILE_PATH" != *.py ]] && exit 0
[[ -f "$FILE_PATH" ]] || exit 0

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUFF="$PROJECT_DIR/.venv/bin/ruff"
[[ -x "$RUFF" ]] || RUFF="$(command -v ruff 2>/dev/null || true)"
[[ -x "$RUFF" ]] || exit 0   # not installed — stay silent rather than nag

OUT=$("$RUFF" check --quiet "$FILE_PATH" 2>&1 || true)

if [[ -n "$OUT" ]]; then
  # Cap the output — a long lint dump would drown the actual work.
  TRIMMED=$(printf '%s' "$OUT" | head -20)
  jq -n --arg msg "ruff check flagged $(basename "$FILE_PATH"):"$'\n'"$TRIMMED" '{
    hookSpecificOutput: {
      hookEventName: "PostToolUse",
      systemMessage: $msg
    }
  }'
fi

exit 0
