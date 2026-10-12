#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
nvidia-smi
python - <<'PY'
import torch
from huggingface_hub import model_info
assert torch.cuda.is_available(), 'CUDA not available to PyTorch'
p=torch.cuda.get_device_properties(0)
print(f'GPU: {p.name}; VRAM={p.total_memory/1024**3:.2f} GiB; bf16={torch.cuda.is_bf16_supported()}')
if p.total_memory < 20*1024**3: print('WARNING: <20 GiB VRAM; reduce batch size if needed')
try:
    info=model_info('google/functiongemma-270m-it')
    print('HF model access: OK', info.id)
except Exception as e:
    raise SystemExit('Cannot access gated FunctionGemma model. Accept license and run huggingface-cli login.\n'+str(e))
PY
