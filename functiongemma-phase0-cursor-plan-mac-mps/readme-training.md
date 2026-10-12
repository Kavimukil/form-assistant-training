# Train FunctionGemma on NVIDIA CUDA (WSL2)

This run continues from the universal-full-v6 checkpoint for one epoch on `universal-full-v7`. The validation and test files are the frozen `universal-full-v1` benchmark. Run it inside **Ubuntu on WSL2**, with the repo on the Linux filesystem (`~/...`), not under `/mnt/c`. Training from a Windows drive is slow and `scripts/bootstrap_wsl.sh` refuses that path.

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

`generated/universal-full-v7/` is built locally and is not in the clone. Copy it from the machine that generated it into the Linux checkout. The v6 checkpoint must already be on that machine at `artifacts/checkpoints/universal-full-v6/`. From the build machine, in `functiongemma-phase0-cursor-plan-mac-mps`:

```bash
rsync -az --info=progress2 \
  generated/universal-full-v7/ \
  USER@HOST:~/form-assistant-training/functiongemma-phase0-cursor-plan-mac-mps/generated/universal-full-v7/
```

Replace `USER@HOST` with the Ubuntu SSH target. If Ubuntu is this same WSL instance and the files are already under the Linux path above, skip the copy.

On Ubuntu, confirm the three SFT files are real JSONL:

```bash
wc -l generated/universal-full-v7/train.sft.jsonl \
  generated/universal-full-v7/validation.sft.jsonl \
  generated/universal-full-v7/test.sft.jsonl
head -c 80 generated/universal-full-v7/train.sft.jsonl; echo
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

## 6. Confirm the v6 checkpoint

This run loads weights from `artifacts/checkpoints/universal-full-v6`. It does not download `google/functiongemma-270m-it`.

```bash
test -f artifacts/checkpoints/universal-full-v6/config.json \
  && ls artifacts/checkpoints/universal-full-v6
```

The directory must contain `config.json`, tokenizer files, and a weight file such as `model.safetensors`. If those files are missing, copy the finished v6 checkpoint onto this disk before the smoke test. A Mac smoke from the base model is not this run.

## 7. Run the CUDA preflight

```bash
./scripts/preflight_wsl_cuda.sh
```

This fails unless all of the following are true:

- `nvidia-smi` works inside WSL
- the repo is not under `/mnt/<drive>`
- `.venv` exists
- `torch.cuda.is_available()` is true

It prints the GPU name, VRAM, and whether BF16 is supported. Under 20 GiB of VRAM it prints a warning. The training config uses batch size 8 and gradient accumulation 4 (effective batch 32). If you run out of memory, lower `per_device_train_batch_size` in a copy of `config/train-universal-v7.yaml` and raise `gradient_accumulation_steps` so the effective batch stays 32. Leave the original config unchanged.

`--device cuda` refuses to train if CUDA is missing. `--device auto` prefers CUDA, then Apple MPS, then CPU. On WSL, pass `--device cuda`.

## 8. What this run trains

Start from the finished v6 weights at `artifacts/checkpoints/universal-full-v6`. Do not resume that run's optimizer. Do not start again from `google/functiongemma-270m-it`.

`config/train-universal-v7.yaml` is the training config. It reads:

- `generated/universal-full-v7/train.sft.jsonl` (20,000 rows)
- `generated/universal-full-v7/validation.sft.jsonl` (2,500 rows, frozen from `universal-full-v1`)

The loss is `assistant_only`. One epoch, seed `20261012`, max sequence length 1152. v6 trained at 1024, and 1,510 of its 20,000 training rows were longer than that, so the option ids at the end of the prompt were cut off. 1152 covers the longest v7 row (1,084 tokens).

`generated/universal-full-v7/test.sft.jsonl` (2,500 rows) is the frozen `universal-full-v1` test file. Do not train on it. The v7 training split keeps the v6 wording, speaks field aliases on one quarter of set rows and option aliases on half of them, cuts no-change from 3,000 to 1,500, and raises option-order rows from 1,200 to 2,200. The last 2,200 single-field training rows are enum so those permutations have a base. It does not use the validation or test sentences.

The `eval_loss` printed for the epoch is the language-model loss. It is not the slot score in step 11.

## 9. Smoke test (50 steps)

Do this before the full epoch. The smoke run writes `artifacts/checkpoints/universal-full-v7-smoke/` and does not keep mid-run checkpoints. It still saves a final snapshot and `smoke_report.json`.

```bash
python -m form_sft.training.train \
  --config config/train-universal-v7.yaml \
  --device cuda \
  --smoke-steps 50
```

A passing smoke run prints a finite loss and `checkpoint_reload=PASS`. Stop if the loss is NaN, if the v6 checkpoint is missing, or if the process is killed for out-of-memory.

## 10. Full training

One epoch from the v6 weights. Do not resume the smoke checkpoint. The new checkpoint goes to `artifacts/checkpoints/universal-full-v7/`:

```bash
python -m form_sft.training.train \
  --config config/train-universal-v7.yaml \
  --device cuda
```

On CUDA, the trainer uses BF16 when the GPU supports it, otherwise FP16. Watch progress with `nvidia-smi` in a second Ubuntu terminal.

A passing run prints a finite loss and `checkpoint_reload=PASS`. The checkpoint directory must contain `config.json` and a weight file such as `model.safetensors`.

## 11. Evaluate and screenshot the comparison

Run validation first, then the frozen test once. Use the same command shape as the universal-full-v1 screenshots: `evaluate-json`, `--device cuda`, `--batch-size 8`, and the frozen 2,500-row file. The printed JSON has the same keys, so the two screenshots compare directly. `n` must be 2500. Epoch `eval_loss` is not part of this comparison.

Screenshot the terminal after each command, with the command and the JSON summary both visible. Copy the metrics and predictions files back as well. Do not retype the numbers.

Validation:

```bash
python -m form_sft.cli evaluate-json \
  --file generated/universal-full-v7/validation.sft.jsonl \
  --model-id artifacts/checkpoints/universal-full-v7 \
  --device cuda \
  --tag universal-full-v7-val \
  --batch-size 8
```

This writes `artifacts/eval/universal-full-v7-val/universal-full-v7-val_metrics.json` and `universal-full-v7-val_predictions.jsonl`.

Frozen test, once:

```bash
python -m form_sft.cli evaluate-json \
  --file generated/universal-full-v7/test.sft.jsonl \
  --model-id artifacts/checkpoints/universal-full-v7 \
  --device cuda \
  --tag universal-full-v7-test \
  --batch-size 8
```

This writes `artifacts/eval/universal-full-v7-test/universal-full-v7-test_metrics.json` and `universal-full-v7-test_predictions.jsonl`. Leave the checkpoint unchanged after the test run.

The previous checkpoint, `artifacts/checkpoints/universal-full-v6`, scored these same frozen files as follows. Read a v7 number against the same row.

| Printed field | v6 validation | v6 test |
| --- | ---: | ---: |
| `n` | 2500 | 2500 |
| `exact_call` | 0.904 | 0.7992 |
| `op` | 0.9192 | 0.9192 |
| `field` | 0.9192 | 0.9192 |
| `value` | 0.8871 | 0.7637 |
| `enum` | 0.9007 | 0.7973 |
| `no_change_precision` | 1.0 | 1.0 |
| `no_change_recall` | 1.0 | 1.0 |
| `false_mutation_rate` | 0.0 | 0.0 |

The test screenshot is an improvement when `exact_call` is at least 0.88, `value` is at least 0.88, and `enum` is at least 0.90, while `false_mutation_rate` stays 0.0 and `no_change_recall` stays 1.0. Validation `enum` must stay at least 0.90, and validation `exact_call` must not fall by more than 0.01. These files do not sign `generated/universal-full-v7/go_no_go.json`.

## 12. Where files land

| Path | What it is |
| --- | --- |
| `generated/universal-full-v7/train.sft.jsonl` | The only training file (20,000 rows) |
| `generated/universal-full-v7/validation.sft.jsonl` | Frozen universal-full-v1 validation file (2,500 rows) |
| `generated/universal-full-v7/test.sft.jsonl` | Frozen universal-full-v1 test file (2,500 rows). Evaluate once, after validation |
| `artifacts/checkpoints/universal-full-v6/` | Weights this run starts from |
| `artifacts/checkpoints/universal-full-v7-smoke/` | 50-step smoke snapshot |
| `artifacts/checkpoints/universal-full-v7/` | The one-epoch checkpoint |
| `artifacts/eval/universal-full-v7-val/` | Validation metrics and per-row predictions |
| `artifacts/eval/universal-full-v7-test/` | One-shot test metrics and per-row predictions |
| `~/.cache/huggingface/` | Downloaded base model and tokenizer |

Checkpoint directories are gitignored. Leave them on the WSL disk. Bring back the eval directories, not the weight files, unless you explicitly want the checkpoint.
