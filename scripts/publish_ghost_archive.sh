#!/usr/bin/env bash
# Publish ghost_archive/ to a public GitHub repo (set GH_USER + GH_ARCHIVE_REPO in .env)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.env"
fi

REPO_NAME="${GH_ARCHIVE_REPO:-tirramind-ghost-archive}"
GH_USER="${GH_USER:-dry-clean}"
# Dedicated ghost-pattern account token (falls back to GITHUB_TOKEN)
GITHUB_TOKEN="${DRY_CLEAN_GITHUB_TOKEN:-${GITHUB_TOKEN:-}}"
if [[ -z "$GITHUB_TOKEN" ]]; then
  echo "ERROR: Set DRY_CLEAN_GITHUB_TOKEN (or GITHUB_TOKEN) in .env"
  exit 1
fi

WORK="/tmp/${REPO_NAME}-publish"
REMOTE="https://x-access-token:${GITHUB_TOKEN}@github.com/${GH_USER}/${REPO_NAME}.git"

echo "==> Staging publish tree at $WORK"
rm -rf "$WORK"
mkdir -p "$WORK"/{alerts,briefs,schema}

cp "$ROOT/ghost_archive/scorecard.md" "$WORK/"
cp "$ROOT/ghost_archive/schema/"*.json "$WORK/schema/"
cp "$ROOT/ghost_archive/alerts/"*.json "$WORK/alerts/" 2>/dev/null || true
cp "$ROOT/ghost_archive/briefs/"*_CHAIN_BRIEF_*.md "$WORK/briefs/" 2>/dev/null || true

cat > "$WORK/README.md" <<'EOF'
# TirraMind Ghost Pattern Archive

Public scorecard for **cross-domain commodity chain alerts** — not single-indicator takes.

Each alert links **≥2 data domains** (e.g. EIA inventories + CFTC positioning) with public sources and pre-registered outcomes.

## Micro-playground MP-1 — Atlantic Energy

| Instrument readout | WTI (`CL=F`), Brent, Natural Gas |
|--------------------|----------------------------------|
| Sensors | EIA petroleum, CFTC COT, GDELT, AIS |

## Layout

```
alerts/     # Machine-readable alert JSON (issued before resolution)
briefs/     # Human chain briefs (publish quality)
scorecard.md
schema/     # JSON Schema
```

## Latest scorecard

See [scorecard.md](./scorecard.md).

## Chain briefs

| # | Brief | Outcome |
|---|-------|---------|
| 1 | [US Crude Draw Meets NG Spec Liquidation](./briefs/2026-06-09_MP-1_CHAIN_BRIEF_001.md) | CL=F +4.19% (2 sessions) |
| 2 | [Russia Stress, WTI Specs Not Crowded](./briefs/2026-06-09_MP-1_CHAIN_BRIEF_002.md) | CL=F -4.15% (5 sessions) |
| 3 | [Tight Inventories, WTI Not Stretched](./briefs/2026-06-09_MP-1_CHAIN_BRIEF_003.md) | CL=F -4.15% (5 sessions) |

## Method

- Alerts archived **before** forward return is measured (no retrofitted calls).
- Z-scores from 52-week rolling history on public data (EIA, CFTC, GDELT).
- Evaluation window: **2–5 trading sessions** on the readout instrument.

## Disclaimer

Not investment advice. Scenario flags for hedgers and researchers — not buy/sell signals.

Cross-domain commodity chain alerts — MP-1 Atlantic Energy.
EOF

cat > "$WORK/.gitignore" <<'EOF'
briefs/draft/
.DS_Store
EOF

echo "==> Ensure GitHub repo exists: ${GH_USER}/${REPO_NAME}"
status=$(curl -s -o /dev/null -w "%{http_code}" \
  -H "Authorization: token ${GITHUB_TOKEN}" \
  "https://api.github.com/repos/${GH_USER}/${REPO_NAME}")
if [[ "$status" == "404" ]]; then
  curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    https://api.github.com/user/repos \
    -d "{\"name\":\"${REPO_NAME}\",\"description\":\"Public ghost pattern chain alerts — MP-1 Atlantic Energy\",\"private\":false}" \
    | grep -q '"full_name"' && echo "    Created repo." || { echo "ERROR: could not create repo"; exit 1; }
elif [[ "$status" == "200" ]]; then
  echo "    Repo already exists."
else
  echo "ERROR: GitHub API status $status"; exit 1
fi

echo "==> Git commit + push"
cd "$WORK"
git init -q
git branch -M main
git add -A
git config user.email "ghost-archive@tirramind.local"
git config user.name "TirraMind Ghost Archive"
git commit -q -m "$(cat <<EOF
Launch MP-1 ghost pattern archive — 3 chain briefs + scorecard.

Public alerts with pre-registered outcomes for Atlantic energy cross-domain chains.
EOF
)"
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE"
git push -u origin main --force

echo ""
echo "Published: https://github.com/${GH_USER}/${REPO_NAME}"
