#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
[[ -x .venv/bin/python ]] || { echo "ERROR: .venv missing; run ./scripts/bootstrap_wsl.sh" >&2; exit 1; }
case "$ROOT" in /mnt/[a-zA-Z]/*) echo "ERROR: train from WSL Linux filesystem, not $ROOT" >&2; exit 1 ;; esac
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi not visible inside WSL2." >&2; exit 1; }
echo '=== nvidia-smi ==='; nvidia-smi
source .venv/bin/activate
python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch inside WSL2")
p = torch.cuda.get_device_properties(0)
gib = p.total_memory / 1024**3
print("gpu", torch.cuda.get_device_name(0))
print("vram_gib", round(gib, 2))
print("bf16_supported", torch.cuda.is_bf16_supported())
if gib < 20:
    print("WARNING: <20 GiB VRAM; Phase-0 defaults may need adjustment")
PY
[[ -f generated/phase0-v1/train.jsonl ]] || echo "WARNING: Phase-0 dataset not present; sync it from Mac after G5."
echo 'WSL2 CUDA preflight PASS'
