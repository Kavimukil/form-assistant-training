#!/bin/bash
set -euo pipefail

# Always run from repository root (works if invoked from scripts/).
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

[[ -d .venv ]] || { echo "ERROR: .venv missing. Run: bash scripts/bootstrap_macos.sh" >&2; exit 2; }
# shellcheck disable=SC1091
source .venv/bin/activate
[[ -f generated/phase0-v1/train.sft.jsonl ]] || { echo "ERROR: compiled SFT dataset missing. Pass G5/G6 first." >&2; exit 3; }

# Smoke already passed in this workspace; set SKIP_SMOKE=1 to jump straight to full SFT.
if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  bash scripts/smoke_train_macos.sh
  echo "50-step MPS smoke gate passed. Starting full Phase-0 SFT."
else
  echo "SKIP_SMOKE=1 — skipping 50-step smoke; starting full Phase-0 SFT."
  bash scripts/preflight_mps.sh
fi

python -m form_sft.training.train --config config/train-full-sft.yaml --device mps
printf '\nPHASE0_FULL_SFT: started/finished via train_phase0_macos.sh\n'
