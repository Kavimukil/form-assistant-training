#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

[[ -d .venv ]] || { echo "ERROR: .venv missing. Run: bash scripts/bootstrap_macos.sh" >&2; exit 2; }
# shellcheck disable=SC1091
source .venv/bin/activate
[[ -f generated/phase0-v1/train.sft.jsonl ]] || { echo "ERROR: missing generated/phase0-v1/train.sft.jsonl. Pass dataset/compiler gates first." >&2; exit 3; }
[[ -f generated/phase0-v1/validation.sft.jsonl ]] || { echo "ERROR: missing validation.sft.jsonl." >&2; exit 3; }
bash scripts/preflight_mps.sh
python -m form_sft.training.train --config config/train-full-sft.yaml --device mps --smoke-steps 50
printf '\nPHASE0_50_STEP_SMOKE: PASS\n'
