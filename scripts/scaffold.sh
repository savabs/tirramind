#!/usr/bin/env bash
# scaffold.sh — Create the file structure for a new TirraMind feature
#
# Usage: ./scripts/scaffold.sh <feature_name> [layer_number]
#
# Example:
#   ./scripts/scaffold.sh wikipedia_pageviews 1
#   ./scripts/scaffold.sh vpin_calculation 2

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <feature_name> [layer_number]"
    echo ""
    echo "Layers:"
    echo "  1 = Surveillance Surface (agent/tools/)"
    echo "  2 = Feature Engineering  (agent/quant/)"
    echo "  3 = World Model          (agent/models/)"
    echo "  4 = Signal Fusion        (agent/fusion/)"
    echo "  5 = RL Policy            (agent/learning/)"
    echo "  6 = Adversarial          (agent/adversarial/)"
    echo "  7 = LLM Support          (agent/reasoning/)"
    exit 1
fi

FEATURE="$1"
LAYER="${2:-0}"
DATE=$(date +%Y-%m-%d)

# Directories
mkdir -p docs/research docs/specs docs/memory tasks/active

# Research doc
RESEARCH="docs/research/${FEATURE}.md"
if [[ ! -f "$RESEARCH" ]]; then
    cat > "$RESEARCH" << EOF
# Feature: ${FEATURE}

## Current Architecture
- (relevant modules, patterns, dependencies)

## Observations
- (what exists, what's missing, what connects to what)

## Risks
- (edge cases, breaking changes, security concerns)

## Data Requirements
- (what data series/sources are needed, what's available, what's missing)

## Math/Algorithm Survey
- (what algorithms apply, what libraries exist vs. build from scratch, complexity)
EOF
    echo "Created: $RESEARCH"
else
    echo "Exists:  $RESEARCH"
fi

# Spec doc
SPEC="docs/specs/${FEATURE}_spec.md"
if [[ ! -f "$SPEC" ]]; then
    cat > "$SPEC" << EOF
# Spec: ${FEATURE}

## Goal
What the feature must accomplish.

## Layer
$(case $LAYER in
    1) echo "Layer 1: Surveillance Surface (agent/tools/)" ;;
    2) echo "Layer 2: Feature Engineering (agent/quant/)" ;;
    3) echo "Layer 3: World Model (agent/models/)" ;;
    4) echo "Layer 4: Signal Fusion (agent/fusion/)" ;;
    5) echo "Layer 5: RL Policy (agent/learning/)" ;;
    6) echo "Layer 6: Adversarial (agent/adversarial/)" ;;
    7) echo "Layer 7: LLM Support (agent/reasoning/)" ;;
    *) echo "TBD" ;;
esac)

## Files Affected
- (list files to create or modify)

## Implementation Steps
- [ ] 1.1: (first atomic step)
- [ ] 1.2: (second atomic step)

## Edge Cases
- (possible failure scenarios)

## Testing Plan
- (how the feature should be validated)
EOF
    echo "Created: $SPEC"
else
    echo "Exists:  $SPEC"
fi

# Task file
TASK="tasks/active/${FEATURE}.md"
if [[ ! -f "$TASK" ]]; then
    cat > "$TASK" << EOF
# Task: ${FEATURE}

Status: active
Created: ${DATE}
Research: docs/research/${FEATURE}.md
Spec: docs/specs/${FEATURE}_spec.md

## Steps
- [ ] 0.1: Research — fill in docs/research/${FEATURE}.md
- [ ] 0.2: Spec — fill in docs/specs/${FEATURE}_spec.md
- [ ] 0.3: Implementation — (add numbered steps after spec is written)
EOF
    echo "Created: $TASK"
else
    echo "Exists:  $TASK"
fi

echo ""
echo "=== Feature scaffolded: ${FEATURE} ==="
echo ""
echo "Next steps:"
echo "  1. Open a new Copilot chat"
echo "  2. Type: @quant-researcher Read #file:${TASK} and fill in the research doc"
echo "  3. Then: Read #file:${RESEARCH} and write the spec at ${SPEC}"
echo "  4. Then switch to premium model and type: #next-step"
echo "  5. Or for the full automated pipeline: #full-pipeline ${FEATURE}"
echo ""
