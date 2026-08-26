#!/usr/bin/env bash
# PostToolUse warning — fires when a change touches the registries whose drift
# silently invalidated every model checkpoint for months (LESSONS.md F-12).
#
# Warns, never blocks: these edits are often correct and necessary. The point is
# that the consequence (every existing checkpoint is now invalid) must be stated
# out loud at the moment of the edit, not discovered weeks later as an opaque
# torch shape error.
set -uo pipefail

INPUT=$(cat)
FILE_PATH=$(printf '%s' "$INPUT" | jq -r '.tool_input.file_path // .tool_input.path // empty')

[[ -z "$FILE_PATH" ]] && exit 0

CHANGED=$(printf '%s' "$INPUT" | jq -r '
  [.tool_input.new_string?, .tool_input.new_str?, .tool_input.content?]
  | map(select(. != null)) | join("\n")
')
[[ -z "$CHANGED" ]] && exit 0

warn() {
  jq -n --arg msg "$1" --arg ctx "$2" '{
    hookSpecificOutput: {
      hookEventName: "PostToolUse",
      systemMessage: $msg,
      additionalContext: $ctx
    }
  }'
  exit 0
}

# ── The registries ────────────────────────────────────────────────────────
if [[ "$FILE_PATH" == *graph_builder.py ]]; then

  if grep -qE '(ENTITY_TYPES|OBSERVATION_TYPES)' <<<"$CHANGED"; then
    warn \
"SCHEMA REGISTRY EDITED — every existing checkpoint is now invalid.

One-hot position derives from list index, so inserting a type shifts every later
index. This is LESSONS.md F-12: instrument features drifted 14 -> 23 -> 49 with
no retrain, and 'maritime_area' trained as 'cftc_contract' for months." \
"Required follow-up:
1. Both lists MUST stay alphabetically sorted (insertions must be reviewable).
2. Run: validate_schema_against_store(store) — confirm DB and code agree.
3. Confirm every derived dim still computes: ENRICHMENT_DIM == 9 + len(OBSERVATION_TYPES),
   BASE_FEAT_DIM == len(ENTITY_TYPES) + 3.
4. A retrain is now REQUIRED. Registry edit and retrain are the same change.
5. Update the count assertions in tests/ with a dated comment explaining why.
Delegate verification to the schema-sentinel agent."
  fi

  if grep -qE '(ENRICHMENT_DIM|BASE_FEAT_DIM|PRICE_FEAT_DIM|MICROSTRUCTURE_DIM|M15_QUANT_DIM)' <<<"$CHANGED"; then
    warn \
"FEATURE DIMENSION EDITED — check it is DERIVED, not a literal.

ENRICHMENT_DIM was hardcoded to 55 (correct only at 46 obs types). When the
registry grew to 48, the obs_type_dist writer at 'offset + 9 + ot_idx' ran past
the block: with BASE_FEAT_DIM=14 the tensor was 69 wide and ot_idx=46 addressed
index 69 — crashing entity_scoring, and SILENTLY corrupting the price-feature
block for instrument nodes." \
"A dimension derived from a registry must be computed, never written as a number.
Correct: ENRICHMENT_DIM = _ENRICHMENT_SCALAR_DIM + len(OBSERVATION_TYPES)
Wrong:   ENRICHMENT_DIM = 61
Any literal here is a time bomb that detonates one registry edit later."
  fi
fi

# ── Checkpoint loading ────────────────────────────────────────────────────
if [[ "$FILE_PATH" == *trainer.py ]] && grep -qE 'load_state_dict|strict=False|in_channels' <<<"$CHANGED"; then
  warn \
"CHECKPOINT LOADING EDITED — a skipped key is a randomly-initialised layer." \
"load_state_dict(strict=False) silently skipping a mismatched key leaves that
layer at random init. That is how a 23-wide instrument projection survived
against 49-wide features, surfacing much later as an opaque torch shape error
naming no entity type. Any skip must name the layer and both widths."
fi

exit 0
