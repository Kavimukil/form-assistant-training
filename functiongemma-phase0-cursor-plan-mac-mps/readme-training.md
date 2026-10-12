# Train FunctionGemma on NVIDIA CUDA (WSL2)

This run trains the unseen-schema corpus `universal-full-v1`. Run it inside **Ubuntu on WSL2**, with the repo on the Linux filesystem (`~/...`), not under `/mnt/c`. Training from a Windows drive is slow and `scripts/bootstrap_wsl.sh` refuses that path.

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

`generated/universal-full-v1/` is built locally and is not in the clone. Copy it from the machine that generated it into the Linux checkout. From that machine, in `functiongemma-phase0-cursor-plan-mac-mps`:

```bash
rsync -az --info=progress2 \
  generated/universal-full-v1/ \
  USER@HOST:~/form-assistant-training/functiongemma-phase0-cursor-plan-mac-mps/generated/universal-full-v1/
```

Replace `USER@HOST` with the Ubuntu SSH target. If Ubuntu is this same WSL instance and the files are already under the Linux path above, skip the copy.

On Ubuntu, confirm the three SFT files are real JSONL:

```bash
wc -l generated/universal-full-v1/train.sft.jsonl \
  generated/universal-full-v1/validation.sft.jsonl \
  generated/universal-full-v1/test.sft.jsonl
head -c 80 generated/universal-full-v1/train.sft.jsonl; echo
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

It prints the GPU name, VRAM, and whether BF16 is supported. Under 20 GiB of VRAM it prints a warning. The training config uses batch size 8 and gradient accumulation 4 (effective batch 32). If you run out of memory, lower `per_device_train_batch_size` in a copy of `config/train-universal-full.yaml` and raise `gradient_accumulation_steps` so the effective batch stays 32. Leave the original config unchanged.

`--device cuda` refuses to train if CUDA is missing. `--device auto` prefers CUDA, then Apple MPS, then CPU. On WSL, pass `--device cuda`.

## 8. What this run trains

Train once from `google/functiongemma-270m-it`. Do not resume an older checkpoint.

`config/train-universal-full.yaml` is the training config. It reads:

- `generated/universal-full-v1/train.sft.jsonl` (20,000 rows)
- `generated/universal-full-v1/validation.sft.jsonl` (2,500 rows)

The loss is `assistant_only`: prompt tokens are masked and the supervised text is the `{"function_calls":[...]}` target. Three epochs, seed `20261012`, max sequence length 1024.

`generated/universal-full-v1/test.sft.jsonl` (2,500 rows) is the frozen test file. Do not train on it. Do not train on `quarantine.jsonl`.

The `eval_loss` printed each epoch is the language-model loss on the validation file. It is not the slot score in step 11.

## 9. Smoke test (50 steps)

Do this before the full run. The smoke run writes `artifacts/checkpoints/universal-full-v1-smoke/` and does not keep mid-run checkpoints. It still saves a final snapshot and `smoke_report.json`.

```bash
python -m form_sft.training.train \
  --config config/train-universal-full.yaml \
  --device cuda \
  --smoke-steps 50
```

A passing smoke run prints a finite loss and `checkpoint_reload=PASS`. Stop if the loss is NaN, if the process exits because the installed TRL cannot do `assistant_only` loss, or if it is killed for out-of-memory.

## 10. Full training

One SFT run, 3 epochs, from `google/functiongemma-270m-it`. Do not resume the smoke checkpoint. The new checkpoint goes to `artifacts/checkpoints/universal-full-v1/`:

```bash
python -m form_sft.training.train \
  --config config/train-universal-full.yaml \
  --device cuda
```

On CUDA, the trainer uses BF16 when the GPU supports it, otherwise FP16. Watch progress with `nvidia-smi` in a second Ubuntu terminal.

A passing run prints a finite loss and `checkpoint_reload=PASS`. The checkpoint directory must contain `config.json` and a weight file such as `model.safetensors`.

## 11. Evaluate the checkpoint

Run validation before the frozen test. This command scores greedy `function_calls` JSON. It writes:

- `artifacts/eval/universal-full-v1-val/universal-full-v1-val_metrics.json`
- `artifacts/eval/universal-full-v1-val/universal-full-v1-val_predictions.jsonl`

```bash
python -m form_sft.cli evaluate-json \
  --file generated/universal-full-v1/validation.sft.jsonl \
  --model-id artifacts/checkpoints/universal-full-v1 \
  --device cuda \
  --tag universal-full-v1-val \
  --batch-size 8
```

The process prints a short JSON summary: row count, exact call, operation, field, value, enum, no-change precision and recall, and false-mutation rate. That printout is enough to decide whether validation looks right. Copy the two files above back to the machine where you review results. Do not retype the numbers.

If validation looks right, run the frozen test set once:

```bash
python -m form_sft.cli evaluate-json \
  --file generated/universal-full-v1/test.sft.jsonl \
  --model-id artifacts/checkpoints/universal-full-v1 \
  --device cuda \
  --tag universal-full-v1-test \
  --batch-size 8
```

That writes `artifacts/eval/universal-full-v1-test/universal-full-v1-test_metrics.json` and `universal-full-v1-test_predictions.jsonl`. Copy those two files back as well. Leave the checkpoint as it is after the test run. Converting it to GGUF is outside this repo.

These files do not sign `generated/universal-full-v1/go_no_go.json`. That decision still needs the unseen-schema suite comparison.

## 12. Where files land

| Path | What it is |
| --- | --- |
| `generated/universal-full-v1/train.sft.jsonl` | The only training file (20,000 rows) |
| `generated/universal-full-v1/validation.sft.jsonl` | The only validation file (2,500 rows) |
| `generated/universal-full-v1/test.sft.jsonl` | Frozen test file (2,500 rows). Evaluate once, after validation |
| `artifacts/checkpoints/universal-full-v1-smoke/` | 50-step smoke snapshot |
| `artifacts/checkpoints/universal-full-v1/` | The one full SFT checkpoint |
| `artifacts/eval/universal-full-v1-val/` | Validation metrics and per-row predictions |
| `artifacts/eval/universal-full-v1-test/` | One-shot test metrics and per-row predictions |
| `~/.cache/huggingface/` | Downloaded base model and tokenizer |

Checkpoint directories are gitignored. Leave them on the WSL disk. Bring back the eval directories, not the weight files, unless you explicitly want the checkpoint.
