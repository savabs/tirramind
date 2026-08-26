#!/usr/bin/env bash
# PreToolUse guard — blocks writes that this repo has actually been burned by.
#
# Reads the hook payload on stdin, denies via exit 2 + permissionDecision.
# Silent pass-through (exit 0) for everything else.
#
# Why each rule exists is documented inline — a rule whose reason is forgotten
# gets deleted by the next person who finds it inconvenient.
set -uo pipefail

INPUT=$(cat)
FILE_PATH=$(printf '%s' "$INPUT" | jq -r '.tool_input.file_path // .tool_input.path // empty')

[[ -z "$FILE_PATH" ]] && exit 0

deny() {
  jq -n --arg reason "$1" '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: $reason
    }
  }'
  exit 2
}

# ── 1. .env files ─────────────────────────────────────────────────────────
# Holds LIVE credentials: Paddle API key + webhook secret, Anthropic, GitHub,
# HuggingFace, WandB, Kaggle. An agent has no business rewriting that file —
# a bad edit can corrupt or leak all of them at once. The human edits it.
case "$(basename "$FILE_PATH")" in
  .env|.env.*|*.env)
    # .env.example / .env.production.example are templates with no secrets.
    if [[ "$FILE_PATH" != *.example ]]; then
      deny ".env files hold live credentials and are human-edit-only. Ask the user to make this change themselves (they can run: !\$EDITOR .env). Templates ending in .example are editable."
    fi
    ;;
esac

# ── 2. Existing model checkpoints ─────────────────────────────────────────
# CLAUDE.md §5: "Checkpoints are immutable once created."
# On 2026-08-26 a diagnostic run of the gnn_inference DAG overwrote
# .tirra_pipeline/gnn_model.pt in place and destroyed the May 25 checkpoint.
# The rule was written down, was read, and was still broken — because prose is
# advisory and a hook is enforcement.
#
# NOTE: only blocks OVERWRITING an existing file. Creating a NEW checkpoint is
# exactly what checkpoint_store.save_versioned() does and must stay allowed.
if [[ "$FILE_PATH" == *.pt || "$FILE_PATH" == *.pth || "$FILE_PATH" == *.ckpt ]]; then
  if [[ -e "$FILE_PATH" ]]; then
    deny "Checkpoint $(basename "$FILE_PATH") already exists and is immutable (CLAUDE.md §5). Use agent/models/gnn/checkpoint_store.py: save_versioned() to write a NEW versioned artifact, or archive_checkpoint() to retire this one. Never write through an existing checkpoint — that is how a file ended up with in_channels=49 metadata against 23-wide weights."
  fi
fi

# ── 3. The live pipeline database ─────────────────────────────────────────
# 138 MB holding 5,628 entities / 365k observations / 16,870 typed links
# accumulated over months. Most upstream APIs only serve a recent window, so it
# is NOT reproducible. Written by the pipeline, never by an editor.
if [[ "$FILE_PATH" == *.tirra_pipeline/*.db || "$FILE_PATH" == *pipeline.db ]]; then
  deny "pipeline.db is non-reproducible accumulated state (months of observations most APIs will not serve again). It is written by the pipeline, never edited directly. Use PipelineStore, or scripts/run_chain.py."
fi

exit 0
