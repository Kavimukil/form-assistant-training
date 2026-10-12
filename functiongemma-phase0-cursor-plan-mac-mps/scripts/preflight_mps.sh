#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

[[ "$(uname -s)" == "Darwin" ]] || { echo "ERROR: MPS preflight is for macOS." >&2; exit 2; }
[[ -d .venv ]] || { echo "ERROR: .venv missing. Run: bash scripts/bootstrap_macos.sh" >&2; exit 3; }
# shellcheck disable=SC1091
source .venv/bin/activate

python - <<'PY'
import platform
import torch

print("platform", platform.platform())
print("machine", platform.machine())
print("torch", torch.__version__)
print("mps_built", torch.backends.mps.is_built())
print("mps_available", torch.backends.mps.is_available())
print("cuda_available", torch.cuda.is_available())

if not torch.backends.mps.is_built():
    raise SystemExit("FAIL: installed PyTorch was not built with MPS support")
if not torch.backends.mps.is_available():
    raise SystemExit("FAIL: MPS is not available on this Mac/PyTorch environment")

# Force a real Metal allocation/op rather than trusting feature flags alone.
x = torch.arange(1024, dtype=torch.float32, device="mps")
y = (x * 2).sum()
torch.mps.synchronize()
assert y.item() == 1047552.0
print("mps_tensor_smoke", "PASS")
print("PHASE0_MPS_PREFLIGHT", "PASS")
PY
