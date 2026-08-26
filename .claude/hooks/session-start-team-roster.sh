#!/usr/bin/env bash
# SessionStart hook — re-affirms the agent team's existence and dispatch rules
# at the start of every session, as a second, independent anchor alongside
# CLAUDE.md's own auto-injection.
#
# Why this exists as well as CLAUDE.md §12: CLAUDE.md is already unconditionally
# injected every session, but the owner asked for a belt-and-suspenders
# mechanism that doesn't rely on a single injection path. This hook rebuilds
# the roster live from .claude/agents/*.md frontmatter (never a hardcoded
# list, so it can't silently go stale) and restates the triage/cloning rules
# in the SessionStart context, independent of how/whether CLAUDE.md loaded.
set -uo pipefail

AGENTS_DIR="${CLAUDE_PROJECT_DIR}/.claude/agents"
[[ -d "$AGENTS_DIR" ]] || exit 0

ROSTER=""
for f in "$AGENTS_DIR"/*.md; do
  [[ "$(basename "$f")" == "README.md" ]] && continue
  name=$(grep -m1 '^name:' "$f" | sed 's/^name:[[:space:]]*//')
  desc=$(grep -m1 '^description:' "$f" | sed 's/^description:[[:space:]]*//')
  [[ -z "$name" ]] && continue
  ROSTER+="- ${name}: ${desc}"$'\n'
done

COUNT=$(printf '%s' "$ROSTER" | grep -c '^-' || true)

CONTEXT=$(cat <<EOF
TirraMind agent team — ${COUNT} specialists exist right now in .claude/agents/,
each with an exclusive domain (see each file's "## Boundaries" section):

${ROSTER}
Dispatch rules (CLAUDE.md §12, full detail there):
- "Ask the team" never means "run every agent" — triage first via
  principal-architect or /team. Produce REQUEST/DISPATCH/EXCLUDED/SEQUENCE.
- Default 1-3 agents. 4-6 needs a stated reason. 7+ needs the owner's
  explicit approval.
- A busy specialist may be cloned into DISJOINT scope only, capped at 2
  total instances of one role (original + at most one clone), counted from
  journal.jsonl ground truth — never git status. Only the top-level
  dispatcher clones; no specialist has Agent/Workflow tool access.
- Use Workflow for real parallel multi-agent dispatch; the Agent tool for a
  single specialist. Custom subagent_type may need a session restart to
  register — if dispatch by type fails, spawn general-purpose and instruct
  it to read its own role file first.
EOF
)

jq -n --arg ctx "$CONTEXT" '{
  hookSpecificOutput: {
    hookEventName: "SessionStart",
    additionalContext: $ctx
  }
}'
exit 0
