# FunctionGemma Phase-0 Form Extraction

Cursor-ready scaffold for the frozen 25k Phase-0 experiment.

## Canonical Phase-0 host

**Apple Silicon Mac = development + dataset generation/audit + FunctionGemma training on MPS.**

WSL2/CUDA scripts remain optional for later phases/sweeps; they are not required for Phase-0. Never copy `.venv` between macOS and Linux.

## 1. Bootstrap macOS

```bash
./scripts/bootstrap_macos.sh
source .venv/bin/activate
./scripts/preflight_mps.sh
```

The preflight requires a real MPS tensor operation to pass. It does not silently enable CPU fallback.

## 2. Build and audit Phase-0

```bash
pytest -q
python -m form_sft.cli build --spec config/phase0-v1.yaml
python -m form_sft.cli audit --spec config/phase0-v1.yaml
```

Do not train until dataset Gate G5 and compiler Gate G6 pass and `generated/phase0-v1/train.sft.jsonl` plus `validation.sft.jsonl` exist.

## 3. Mandatory 50-step MPS smoke gate

```bash
./scripts/smoke_train_macos.sh
```

Equivalent direct command:

```bash
python -m form_sft.training.train \
  --config config/train-full-sft.yaml \
  --device mps \
  --smoke-steps 50
```

The smoke run writes to a separate `-smoke` output directory and does not save normal training checkpoints.

## 4. Full Phase-0 SFT

Recommended guarded command:

```bash
./scripts/train_phase0_macos.sh
```

It reruns the 50-step smoke gate and starts full SFT only if the smoke test passes.

Direct full-training command:

```bash
python -m form_sft.training.train \
  --config config/train-full-sft.yaml \
  --device mps
```

## Portable device selection

The Python trainer supports:

```text
--device auto   CUDA -> MPS -> CPU
--device mps    require Apple MPS
--device cuda   require CUDA
--device cpu    explicit CPU fallback
```

For the conservative Phase-0 baseline, MPS uses float32 model weights and does not enable CUDA-oriented BF16/FP16 Trainer flags. Optimize precision only after the baseline is correct and reproducible.

## Optional WSL2/CUDA path

The existing rsync/bootstrap/pull-back scripts are retained under `scripts/` for later experiments. They are not part of the Phase-0 critical path.

## Mac-only operator guide

For exact Phase-0 commands, expected artifacts, and MPS troubleshooting, see [`MAC_PHASE0_README.md`](MAC_PHASE0_README.md).
