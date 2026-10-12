#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
[[ -f generated/phase0-v1/train.sft.jsonl ]] || { echo 'Missing compiled SFT dataset. Pass G5/G6 first.'; exit 3; }
python -m form_sft.training.train --config config/train-full-sft.yaml --smoke-steps 50
echo 'Smoke training passed. Starting full SFT.'
python -m form_sft.training.train --config config/train-full-sft.yaml
