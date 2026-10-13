# Train FunctionGemma on NVIDIA CUDA (WSL2)

This run continues from the universal-full-v7 checkpoint for 200 steps on `universal-full-v9`. It does not use the v8 weights. The validation and test files are the frozen `universal-full-v1` benchmark. Run it inside **Ubuntu on WSL2**, with the repo on the Linux filesystem (`~/...`), not under `/mnt/c`. Training from a Windows drive is slow and `scripts/bootstrap_wsl.sh` refuses that path.

The Windows NVIDIA driver is what WSL uses. Do not install a separate Linux NVIDIA driver inside Ubuntu.

## 1. Check the GPU from Ubuntu

Open Ubuntu (not PowerShell):

```bash
nvidia-smi
```

You should see the GPU name, driver version, and VRAM. If `nvidia-smi` is missing, update the NVIDIA driver on Windows, then run `wsl --shutdown` from PowerShell and open Ubuntu again.

## 2. Install Ubuntu packages

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git git-lfs build-essential
git lfs install
python3 --version
```

Python 3.11 or newer is required.

## 3. Clone the repo onto the Linux disk

```bash
cd ~
git clone https://github.com/Kavimukil/form-assistant-training.git
cd form-assistant-training
git lfs pull
cd functiongemma-phase0-cursor-plan-mac-mps
pwd
```

`pwd` must look like `/home/<you>/form-assistant-training/functiongemma-phase0-cursor-plan-mac-mps`. A path starting with `/mnt/c` or `/mnt/d` is the wrong disk.

## 4. Copy the 25k dataset onto that disk

`generated/universal-full-v9/` is built only after the v7 and v8 test predictions are copied back. The build stops if either file is missing. The v7 checkpoint must already be on the training machine at `artifacts/checkpoints/universal-full-v7/`. Do not point training at `artifacts/checkpoints/universal-full-v8`.

Copy the prediction files into this checkout:

```bash
mkdir -p artifacts/eval/universal-full-v7-test artifacts/eval/universal-full-v8-test
```

The files are `artifacts/eval/universal-full-v7-test/universal-full-v7-test_predictions.jsonl` and `artifacts/eval/universal-full-v8-test/universal-full-v8-test_predictions.jsonl`. Then, in `functiongemma-phase0-cursor-plan-mac-mps`:

```bash
python scripts/build_v9.py
```

That writes 18,000 replayed v7 training rows plus 1,000 string-copy drills and 1,000 choice-copy drills. Alias speech is included only when the v7 choice misses are mostly labels or aliases. Validation and test stay the frozen v1 files. Copy the result to the Linux checkout:

```bash
rsync -az --info=progress2 \
  generated/universal-full-v9/ \
  USER@HOST:~/form-assistant-training/functiongemma-phase0-cursor-plan-mac-mps/generated/universal-full-v9/
```

Replace `USER@HOST` with the Ubuntu SSH target. If Ubuntu is this same WSL instance and the files are already under the Linux path above, skip the copy.

On Ubuntu, confirm the three SFT files are real JSONL:

```bash
wc -l generated/universal-full-v9/train.sft.jsonl \
  generated/universal-full-v9/validation.sft.jsonl \
  generated/universal-full-v9/test.sft.jsonl
head -c 80 generated/universal-full-v9/train.sft.jsonl; echo
```

The counts must be 20000, 2500, and 2500. The first line must start with `{`. A line that starts with `version https://git-lfs.github.com` is a pointer, not data.

## 5. Create the Linux virtualenv

```bash
./scripts/bootstrap_wsl.sh
source .venv/bin/activate
```

This creates `.venv` on Linux, installs `requirements.txt` and `requirements-dev.txt`, and installs the `form_sft` package. Do not copy a macOS `.venv` into WSL.

Check that PyTorch can see CUDA:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

`torch.cuda.is_available()` must print `True`, and `torch.version.cuda` must not be `None`.

If the wheel is CPU-only, install a CUDA build that matches the driver from [pytorch.org](https://pytorch.org/get-started/locally/), then install the rest of the project without replacing that torch:

```bash
# example only — use the command from the PyTorch site for your driver
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
SKIP_TORCH_INSTALL=1 ./scripts/bootstrap_wsl.sh
source .venv/bin/activate
```

## 6. Confirm the v7 checkpoint

This run loads weights from `artifacts/checkpoints/universal-full-v7`. It does not download `google/functiongemma-270m-it`.

```bash
test -f artifacts/checkpoints/universal-full-v7/config.json \
  && ls artifacts/checkpoints/universal-full-v7
```

The directory must contain `config.json`, tokenizer files, and a weight file such as `model.safetensors`. If those files are missing, copy the finished v7 checkpoint onto this disk before the smoke test. A Mac smoke from the base model is not this run.

## 7. Run the CUDA preflight

```bash
./scripts/preflight_wsl_cuda.sh
```

This fails unless all of the following are true:

- `nvidia-smi` works inside WSL
- the repo is not under `/mnt/<drive>`
- `.venv` exists
- `torch.cuda.is_available()` is true

It prints the GPU name, VRAM, and whether BF16 is supported. Under 20 GiB of VRAM it prints a warning. The training config uses batch size 8 and gradient accumulation 4 (effective batch 32). If you run out of memory, lower `per_device_train_batch_size` in a copy of `config/train-universal-v9.yaml` and raise `gradient_accumulation_steps` so the effective batch stays 32. Leave the original config unchanged.

`--device cuda` refuses to train if CUDA is missing. `--device auto` prefers CUDA, then Apple MPS, then CPU. On WSL, pass `--device cuda`.

## 8. What this run trains

Start from the finished v7 weights at `artifacts/checkpoints/universal-full-v7`. Do not resume that run's optimizer. Do not start from v8 or from `google/functiongemma-270m-it`.

`config/train-universal-v9.yaml` is the training config. It reads:

- `generated/universal-full-v9/train.sft.jsonl` (20,000 rows: 18,000 from v7 plus 2,000 copy drills)
- `generated/universal-full-v9/validation.sft.jsonl` (2,500 rows, frozen from `universal-full-v1`)

The loss is `assistant_only`. The run is 200 steps at learning rate `5e-6`, not a full epoch at `2e-5`. Seed `20261012`, max sequence length 1152.

`generated/universal-full-v9/test.sft.jsonl` is the frozen test. Do not train on it, and do not run it until validation passes the stop gate in step 11.

The `eval_loss` printed at step 200 is the language-model loss. It is not the slot score in step 11.

## 9. Smoke test (50 steps)

Do this before the full epoch. The smoke run writes `artifacts/checkpoints/universal-full-v9-smoke/` and does not keep mid-run checkpoints. It still saves a final snapshot and `smoke_report.json`.

```bash
python -m form_sft.training.train \
  --config config/train-universal-v9.yaml \
  --device cuda \
  --smoke-steps 50
```

A passing smoke run prints a finite loss and `checkpoint_reload=PASS`. Stop if the loss is NaN, if the v7 checkpoint is missing, or if the process is killed for out-of-memory.

## 10. Full training

200 steps from the v7 weights. Do not resume the smoke checkpoint and do not resume the v8 checkpoint. The new checkpoint goes to `artifacts/checkpoints/universal-full-v9/`:

```bash
python -m form_sft.training.train \
  --config config/train-universal-v9.yaml \
  --device cuda
```

On CUDA, the trainer uses BF16 when the GPU supports it, otherwise FP16. Watch progress with `nvidia-smi` in a second Ubuntu terminal.

A passing run prints a finite loss and `checkpoint_reload=PASS`. The checkpoint directory must contain `config.json` and a weight file such as `model.safetensors`.

## 11. Evaluate and screenshot the comparison

Run validation first. Stop there, and do not run the test, unless every validation bar below holds. Then run the frozen test once. Use `evaluate-json`, `--device cuda`, `--batch-size 8`, and the frozen 2,500-row file. `n` must be 2500. Step-200 `eval_loss` is not part of this comparison.

Screenshot the terminal after each command, with the command and the JSON summary both visible. Copy the metrics and predictions files back as well. Do not retype the numbers.

Validation:

```bash
python -m form_sft.cli evaluate-json \
  --file generated/universal-full-v9/validation.sft.jsonl \
  --model-id artifacts/checkpoints/universal-full-v9 \
  --device cuda \
  --tag universal-full-v9-val \
  --batch-size 8
```

This writes `artifacts/eval/universal-full-v9-val/universal-full-v9-val_metrics.json` and `universal-full-v9-val_predictions.jsonl`.

Stop if validation `no_change_recall` is not 1.0, `false_mutation_rate` is not 0.0, `field` is below 0.9500, `enum` is below 0.90, or `exact_call` is below 0.8956. Do not run the test in that case.

Frozen test, once, only after validation passes:

```bash
python -m form_sft.cli evaluate-json \
  --file generated/universal-full-v9/test.sft.jsonl \
  --model-id artifacts/checkpoints/universal-full-v9 \
  --device cuda \
  --tag universal-full-v9-test \
  --batch-size 8
```

This writes `artifacts/eval/universal-full-v9-test/universal-full-v9-test_metrics.json` and `universal-full-v9-test_predictions.jsonl`. Leave the checkpoint unchanged after the test run.

Read a v9 number against the same row. v7 is the baseline. v8 is the rejected run and is not a weight source.

| Printed field | v7 validation | v7 test | v8 validation | v8 test |
| --- | ---: | ---: | ---: | ---: |
| `n` | 2500 | 2500 | 2500 | 2500 |
| `exact_call` | 0.9056 | 0.8412 | 0.7868 | 0.6724 |
| `op` | 0.9608 | 0.9712 | 0.7944 | 0.7608 |
| `field` | 0.9608 | 0.9712 | 0.7944 | 0.7608 |
| `value` | 0.9360 | 0.8133 | 0.9252 | 0.7997 |
| `enum` | 0.9407 | 0.6864 | 0.9044 | 0.6883 |
| `no_change_precision` | 1.0 | 1.0 | 0.0 | 0.0 |
| `no_change_recall` | 1.0 | 1.0 | 0.0 | 0.0 |
| `false_mutation_rate` | 0.0 | 0.0 | 1.0 | 0.9492 |

The test screenshot is an improvement when `exact_call` is at least 0.88, `value` is at least 0.88, `enum` is at least 0.90, and `field` is at least 0.95, while `false_mutation_rate` stays 0.0 and `no_change_recall` stays 1.0. These files do not sign `go_no_go.json`. Do not start a second stage automatically if the bars are missed.

## 12. Where files land

| Path | What it is |
| --- | --- |
| `generated/universal-full-v9/train.sft.jsonl` | The only training file (20,000 rows) |
| `generated/universal-full-v9/validation.sft.jsonl` | Frozen universal-full-v1 validation file (2,500 rows) |
| `generated/universal-full-v9/test.sft.jsonl` | Frozen universal-full-v1 test file (2,500 rows). Evaluate once, after validation passes |
| `artifacts/checkpoints/universal-full-v7/` | Weights this run starts from |
| `artifacts/checkpoints/universal-full-v9-smoke/` | 50-step smoke snapshot |
| `artifacts/checkpoints/universal-full-v9/` | The 200-step checkpoint |
| `artifacts/eval/universal-full-v9-val/` | Validation metrics and per-row predictions |
| `artifacts/eval/universal-full-v9-test/` | One-shot test metrics and per-row predictions |
| `~/.cache/huggingface/` | Downloaded base model and tokenizer |

Checkpoint directories are gitignored. Leave them on the WSL disk. Bring back the eval directories, not the weight files, unless you explicitly want the checkpoint.
