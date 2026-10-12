#!/usr/bin/env bash
set -euo pipefail

# Run INSIDE Ubuntu/WSL2. Creates a Linux-native venv; never reuse/copy a macOS venv.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: bootstrap_wsl.sh must run on Linux/WSL2." >&2; exit 1
fi
if ! grep -qiE '(microsoft|wsl)' /proc/version 2>/dev/null; then
  echo "WARNING: Linux detected, but WSL marker was not found. Continuing."
fi
case "$ROOT" in
  /mnt/[a-zA-Z]/*) echo "ERROR: repo is under $ROOT. Move it to the WSL Linux filesystem (e.g. ~/functiongemma-phase0), not /mnt/c/..." >&2; exit 1 ;;
esac

if ! command -v python3 >/dev/null; then
  echo "ERROR: python3 missing. Install Python 3.11+ and python3-venv in Ubuntu." >&2; exit 1
fi
python3 - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11+ required; found {sys.version.split()[0]}")
print("python", sys.version.split()[0])
PY

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

if [[ "${SKIP_TORCH_INSTALL:-0}" == "1" ]]; then
  grep -v '^torch' requirements.txt > /tmp/functiongemma-phase0-requirements.txt
  pip install -r /tmp/functiongemma-phase0-requirements.txt
else
  pip install -r requirements.txt
fi
pip install -r requirements-dev.txt
pip install -e .

mkdir -p "$HOME/.cache/huggingface" outputs checkpoints artifacts/gates
python -m form_sft.cli doctor

cat <<TXT
WSL bootstrap complete.
Repo:     $ROOT
Venv:     $ROOT/.venv
HF cache: $HOME/.cache/huggingface

Next:
  source .venv/bin/activate
  nvidia-smi
  ./scripts/preflight_wsl_cuda.sh
  # authenticate to Hugging Face if required
  ./scripts/train_phase0.sh
TXT
