# Phase-0 on Apple Silicon Mac

This is the shortest operator guide for the Phase-0 FunctionGemma experiment.

**Canonical host:** Apple Silicon Mac (M4 Pro / 48 GB-class unified memory) using PyTorch MPS.

> `build`, `audit`, and `compile` are gated implementation stubs in the initial scaffold. Until Cursor completes the corresponding gates, their intentional non-zero exit is not an MPS failure. Follow `IMPLEMENTATION_PLAN.md` and do not train until G5/G6 pass.

## 0. Unpack and enter the repo

```bash
unzip functiongemma-phase0-cursor-plan-mac-mps.zip
cd <unpacked-directory>
```

Run all commands below from the repository root.

## 1. Bootstrap the Mac environment

```bash
./scripts/bootstrap_macos.sh
source .venv/bin/activate
```

Expected final output:

```text
macOS Phase-0 environment ready.
Next: source .venv/bin/activate && ./scripts/preflight_mps.sh
```

The `.venv` is local to macOS. Never copy it to Linux/WSL.

## 2. Verify MPS

```bash
./scripts/preflight_mps.sh
```

Expected tail:

```text
mps_built True
mps_available True
mps_tensor_smoke PASS
PHASE0_MPS_PREFLIGHT PASS
```

The check performs a real Metal tensor operation. Do not enable CPU fallback merely to make this gate pass.

## 3. Run repository tests

```bash
pytest -q
```

Expected: all implemented tests pass with no failures.

## 4. Implement and pass dataset gates G0-G6

Useful commands as Cursor fills the stubs:

```bash
python -m form_sft.cli doctor
python -m form_sft.cli plan --spec config/phase0-v1.yaml
python -m form_sft.cli build --spec config/phase0-v1.yaml
python -m form_sft.cli audit --spec config/phase0-v1.yaml
python -m form_sft.cli compile --spec config/phase0-v1.yaml
```

The frozen Phase-0 build must ultimately produce:

```text
generated/phase0-v1/train.jsonl              20,000 rows
generated/phase0-v1/validation.jsonl          2,500 rows
generated/phase0-v1/test.jsonl                2,500 rows
generated/phase0-v1/train.sft.jsonl          20,000 rows
generated/phase0-v1/validation.sft.jsonl      2,500 rows
generated/phase0-v1/audit.json
generated/phase0-v1/manifest.json
artifacts/gates/...                           G0-G6 PASS artifacts
```

`audit.json` must report exact quotas, zero structural/semantic errors, zero prohibited leakage, and zero duplicate IDs. **Do not continue if G5 or G6 fails.**

## 5. Hugging Face model access

The trainer must be able to read `google/functiongemma-270m-it`. If the model download reports authorization/gating errors, authenticate with the Hugging Face CLI/account that has accepted the model terms, then rerun the smoke test.

## 6. Mandatory 50-step MPS smoke test

After `train.sft.jsonl` and `validation.sft.jsonl` exist:

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

Expected startup includes:

```text
Phase-0 device=mps ...
```

Expected successful tail:

```text
PHASE0_50_STEP_SMOKE: PASS
```

Smoke output uses:

```text
artifacts/checkpoints/phase0-full-sft-smoke/
```

Normal model checkpoints are disabled for the smoke run.

## 7. Full Phase-0 training

Recommended guarded command:

```bash
./scripts/train_phase0_macos.sh
```

It reruns the 50-step smoke gate and starts full SFT only after it passes.

Direct command, only when deliberately bypassing the wrapper:

```bash
python -m form_sft.training.train \
  --config config/train-full-sft.yaml \
  --device mps
```

Expected final model directory:

```text
artifacts/checkpoints/phase0-full-sft/
```

It should contain the saved model/tokenizer plus Trainer checkpoint/state artifacts according to the configured epoch save strategy.

## 8. Phase-0 evaluation

After training, run the implemented evaluation/gate commands from `IMPLEMENTATION_PLAN.md` against the frozen 2,500-row test set. Never regenerate or tune against the test set after training begins.

Key outputs to preserve are the final checkpoint, evaluation metrics, per-partition breakdowns, false-mutation rate, and G7-G10 gate artifacts.

---

## MPS troubleshooting

### `mps_available False`

Check:

```bash
uname -m
python -c 'import torch; print(torch.__version__, torch.backends.mps.is_built(), torch.backends.mps.is_available())'
```

Expected architecture is `arm64`. Recreate `.venv` with a native Apple-Silicon Python if the environment was created under Rosetta/x86_64, then rerun `./scripts/bootstrap_macos.sh`.

### `installed PyTorch was not built with MPS support`

Remove the environment and bootstrap again with native arm64 Python:

```bash
rm -rf .venv
./scripts/bootstrap_macos.sh
source .venv/bin/activate
./scripts/preflight_mps.sh
```

### Unsupported MPS operation / `NotImplementedError`

Do **not** permanently set `PYTORCH_ENABLE_MPS_FALLBACK=1` for the benchmark. CPU fallback can hide unsupported operations and invalidate timing comparisons.

For diagnosis only:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
python -m form_sft.training.train \
  --config config/train-full-sft.yaml \
  --device mps \
  --smoke-steps 2
```

If that succeeds, capture the unsupported operation and fix/replace it before the official 50-step gate.

### Process killed / excessive memory pressure

Phase-0 starts conservatively. If macOS memory pressure becomes problematic, reduce `per_device_train_batch_size` in `config/train-full-sft.yaml` and increase `gradient_accumulation_steps` to preserve the effective batch size. Record any config change as a new experiment configuration rather than silently changing the frozen baseline.

Current baseline effective train batch size is:

```text
8 per-device × 4 gradient accumulation = 32
```

### Training appears to run on CPU

The trainer must print:

```text
Phase-0 device=mps
```

Stop the run if it prints `device=cpu`. Then rerun `./scripts/preflight_mps.sh`.

### NaN / unstable loss

The Phase-0 MPS baseline intentionally uses the portable FP32 path. Do not introduce FP16/BF16, quantization, FlashAttention, or CUDA-specific optimizations while diagnosing baseline correctness. First verify the compiled SFT rows and try the 50-step smoke test with the frozen configuration.

### Hugging Face 401/403 or gated-model error

This is model-access/authentication, not MPS. Authenticate the local Hugging Face environment with an account that has access to FunctionGemma, then retry. Do not replace the Phase-0 model ID to make the gate pass.

### Smoke test fails because dataset files are missing

This is expected until G5/G6 are implemented and passed. Required files are:

```text
generated/phase0-v1/train.sft.jsonl
generated/phase0-v1/validation.sft.jsonl
```

Do not create dummy files just to reach training.

---

## Device portability

The Python trainer remains portable:

```text
--device auto   CUDA -> MPS -> CPU
--device mps    require Apple MPS
--device cuda   require NVIDIA CUDA
--device cpu    explicit CPU
```

For official Phase-0 runs on the Mac, use `--device mps` so an unavailable MPS backend fails closed instead of silently selecting CPU.
