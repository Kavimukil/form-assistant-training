# Train FunctionGemma on NVIDIA CUDA (WSL2)

This run trains the unseen-schema corpus `universal-full-v6`. The validation and test files are the frozen `universal-full-v1` benchmark. Run it inside **Ubuntu on WSL2**, with the repo on the Linux filesystem (`~/...`), not under `/mnt/c`. Training from a Windows drive is slow and `scripts/bootstrap_wsl.sh` refuses that path.

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

`generated/universal-full-v6/` is built locally and is not in the clone. Copy it from the machine that generated it into the Linux checkout. From that machine, in `functiongemma-phase0-cursor-plan-mac-mps`:

```bash
rsync -az --info=progress2 \
  generated/universal-full-v6/ \
  USER@HOST:~/form-assistant-training/functiongemma-phase0-cursor-plan-mac-mps/generated/universal-full-v6/
```

Replace `USER@HOST` with the Ubuntu SSH target. If Ubuntu is this same WSL instance and the files are already under the Linux path above, skip the copy.

On Ubuntu, confirm the three SFT files are real JSONL:

```bash
wc -l generated/universal-full-v6/train.sft.jsonl \
  generated/universal-full-v6/validation.sft.jsonl \
  generated/universal-full-v6/test.sft.jsonl
head -c 80 generated/universal-full-v6/train.sft.jsonl; echo
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

## 6. Log in to Hugging Face

The trainer downloads `google/functiongemma-270m-it`. Accept the model terms on Hugging Face, then:

```bash
hf auth login
```

`huggingface-cli login` works on older `huggingface_hub` releases.

## 7. Run the CUDA preflight

```bash
./scripts/preflight_wsl_cuda.sh
```

This fails unless all of the following are true:

- `nvidia-smi` works inside WSL
- the repo is not under `/mnt/<drive>`
- `.venv` exists
- `torch.cuda.is_available()` is true

It prints the GPU name, VRAM, and whether BF16 is supported. Under 20 GiB of VRAM it prints a warning. The training config uses batch size 8 and gradient accumulation 4 (effective batch 32). If you run out of memory, lower `per_device_train_batch_size` in a copy of `config/train-universal-v6.yaml` and raise `gradient_accumulation_steps` so the effective batch stays 32. Leave the original config unchanged.

`--device cuda` refuses to train if CUDA is missing. `--device auto` prefers CUDA, then Apple MPS, then CPU. On WSL, pass `--device cuda`.

## 8. What this run trains

Train once from `google/functiongemma-270m-it`. Do not resume an older checkpoint.

`config/train-universal-v6.yaml` is the training config. It reads:

- `generated/universal-full-v6/train.sft.jsonl` (20,000 rows)
- `generated/universal-full-v6/validation.sft.jsonl` (2,500 rows, frozen from `universal-full-v1`)

The loss is `assistant_only`: prompt tokens are masked and the supervised text is the `{"function_calls":[...]}` target. Three epochs, seed `20261012`, max sequence length 1024.

`generated/universal-full-v6/test.sft.jsonl` (2,500 rows) is the frozen `universal-full-v1` test file. Do not train on it. Do not train on `quarantine.jsonl`. The v6 training split adds no-change paraphrases, value-before-field wording, and 1,200 option-order rows. It does not use the validation or test sentences.

The `eval_loss` printed each epoch is the language-model loss on the validation file. It is not the slot score in step 11.

## 9. Smoke test (50 steps)

Do this before the full run. The smoke run writes `artifacts/checkpoints/universal-full-v6-smoke/` and does not keep mid-run checkpoints. It still saves a final snapshot and `smoke_report.json`.

```bash
python -m form_sft.training.train \
  --config config/train-universal-v6.yaml \
  --device cuda \
  --smoke-steps 50
```

A passing smoke run prints a finite loss and `checkpoint_reload=PASS`. Stop if the loss is NaN, if the process exits because the installed TRL cannot do `assistant_only` loss, or if it is killed for out-of-memory.

## 10. Full training

One SFT run, 3 epochs, from `google/functiongemma-270m-it`. Do not resume the smoke checkpoint. The new checkpoint goes to `artifacts/checkpoints/universal-full-v6/`:

```bash
python -m form_sft.training.train \
  --config config/train-universal-v6.yaml \
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
  --file generated/universal-full-v6/validation.sft.jsonl \
  --model-id artifacts/checkpoints/universal-full-v6 \
  --device cuda \
  --tag universal-full-v6-val \
  --batch-size 8
```

This writes `artifacts/eval/universal-full-v6-val/universal-full-v6-val_metrics.json` and `universal-full-v6-val_predictions.jsonl`.

Frozen test, once:

```bash
python -m form_sft.cli evaluate-json \
  --file generated/universal-full-v6/test.sft.jsonl \
  --model-id artifacts/checkpoints/universal-full-v6 \
  --device cuda \
  --tag universal-full-v6-test \
  --batch-size 8
```

This writes `artifacts/eval/universal-full-v6-test/universal-full-v6-test_metrics.json` and `universal-full-v6-test_predictions.jsonl`. Leave the checkpoint unchanged after the test run.

The previous checkpoint, `artifacts/checkpoints/universal-full-v1`, scored these frozen files as follows. Read a v6 number against the same row.

| Printed field | v1 validation | v1 test |
| --- | ---: | ---: |
| `n` | 2500 | 2500 |
| `exact_call` | 0.7032 | 0.4452 |
| `op` | 0.798 | 0.652 |
| `field` | 0.798 | 0.652 |
| `value` | 0.7164 | 0.5315 |
| `enum` | 0.9197 | 0.3469 |
| `no_change_precision` | 1.0 | 0.0 |
| `no_change_recall` | 0.6283 | 0.0 |
| `false_mutation_rate` | 0.3717 | 1.0 |

Improvement on the test screenshot is a higher `exact_call`, a higher `enum`, a higher `no_change_recall`, and a lower `false_mutation_rate`. The test false-mutation rate was 1.0, so that number has to move before the run is an improvement. These files do not sign `generated/universal-full-v6/go_no_go.json`.

## 12. Where files land

| Path | What it is |
| --- | --- |
| `generated/universal-full-v6/train.sft.jsonl` | The only training file (20,000 rows) |
| `generated/universal-full-v6/validation.sft.jsonl` | Frozen universal-full-v1 validation file (2,500 rows) |
| `generated/universal-full-v6/test.sft.jsonl` | Frozen universal-full-v1 test file (2,500 rows). Evaluate once, after validation |
| `artifacts/checkpoints/universal-full-v6-smoke/` | 50-step smoke snapshot |
| `artifacts/checkpoints/universal-full-v6/` | The one full SFT checkpoint |
| `artifacts/eval/universal-full-v6-val/` | Validation metrics and per-row predictions |
| `artifacts/eval/universal-full-v6-test/` | One-shot test metrics and per-row predictions |
| `~/.cache/huggingface/` | Downloaded base model and tokenizer |

Checkpoint directories are gitignored. Leave them on the WSL disk. Bring back the eval directories, not the weight files, unless you explicitly want the checkpoint.
