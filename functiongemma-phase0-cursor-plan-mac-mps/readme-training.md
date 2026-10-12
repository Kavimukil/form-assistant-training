# Train FunctionGemma on NVIDIA CUDA (WSL2)

Run this inside **Ubuntu on WSL2**, with the repo on the Linux filesystem (`~/...`), not under `/mnt/c`. Training from a Windows drive is slow and `scripts/bootstrap_wsl.sh` refuses that path.

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

`git lfs pull` is required. `generated/json-calls-v3/train.sft.jsonl` is stored in Git LFS (about 91 MB). Without LFS you get a pointer file and training fails.

Confirm the dataset is real text, not a pointer:

```bash
head -c 80 generated/json-calls-v3/train.sft.jsonl; echo
```

The line should start with `{`. A line that starts with `version https://git-lfs.github.com` means LFS did not download the file. Run `git lfs pull` again.

## 4. Create the Linux virtualenv

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

## 5. Log in to Hugging Face

The trainer downloads `google/functiongemma-270m-it`. Accept the model terms on Hugging Face, then:

```bash
hf auth login
```

`huggingface-cli login` works on older `huggingface_hub` releases.

## 6. Run the CUDA preflight

```bash
./scripts/preflight_wsl_cuda.sh
```

This fails unless all of the following are true:

- `nvidia-smi` works inside WSL
- the repo is not under `/mnt/<drive>`
- `.venv` exists
- `torch.cuda.is_available()` is true

It prints the GPU name, VRAM, and whether BF16 is supported. Under 20 GiB of VRAM it prints a warning. The default config uses batch size 8 and gradient accumulation 4 (effective batch 32). If you run out of memory, lower `per_device_train_batch_size` in the YAML you are using and raise `gradient_accumulation_steps` so the effective batch stays 32. Keep that change in a copy of the config so the baseline file stays unchanged.

`--device cuda` refuses to train if CUDA is missing. `--device auto` prefers CUDA, then Apple MPS, then CPU. On WSL, pass `--device cuda`.

## 7. One training set

Train once from `google/functiongemma-270m-it`. Do not resume an older checkpoint.

`config/train-json-calls.yaml` is the only training config for this run. It reads:

- `generated/json-calls-v3/train.sft.jsonl` (38,379 rows)
- `generated/json-calls-v3/validation.sft.jsonl` (4,714 rows)

v3 contains every json-calls v2 example. The added rows teach spoken aliases, spaced words against underscored option ids, a rejected option in the same sentence, light disfluency, and true no-change on those same fields. The test file `generated/json-calls-v3/test.sft.jsonl` is the same 2,500-row file as v1 and v2. Do not train on it.

## 8. Smoke test (50 steps)

Do this before the full run. The smoke run writes `artifacts/checkpoints/json-calls-sft-v3-smoke/` and does not keep mid-run checkpoints. It still saves a final snapshot and `smoke_report.json`.

```bash
python -m form_sft.training.train \
  --config config/train-json-calls.yaml \
  --device cuda \
  --smoke-steps 50
```

A passing smoke run prints a finite loss and `checkpoint_reload=PASS`. Stop if the loss is NaN or the process is killed for out-of-memory.

## 9. Full training

One SFT run, 3 epochs. The checkpoint goes to `artifacts/checkpoints/json-calls-sft-v3/`:

```bash
python -m form_sft.training.train \
  --config config/train-json-calls.yaml \
  --device cuda
```

On CUDA, the trainer uses BF16 when the GPU supports it, otherwise FP16. Watch progress with `nvidia-smi` in a second Ubuntu terminal.

## 10. Evaluate the checkpoint

Validation first:

```bash
python -m form_sft.cli evaluate-json \
  --file generated/json-calls-v3/validation.sft.jsonl \
  --model-id artifacts/checkpoints/json-calls-sft-v3 \
  --device cuda \
  --tag json-calls-v3-val \
  --batch-size 8
```

If validation looks right, run the frozen test set once:

```bash
python -m form_sft.cli evaluate-json \
  --file generated/json-calls-v3/test.sft.jsonl \
  --model-id artifacts/checkpoints/json-calls-sft-v3 \
  --device cuda \
  --tag json-calls-v3-test \
  --batch-size 8
```

Metrics and predictions are written under `artifacts/eval/<tag>/`. Leave the checkpoint as it is after the test run. Converting it to GGUF is outside this repo.

## 11. Where files land

| Path | What it is |
| --- | --- |
| `generated/json-calls-v3/train.sft.jsonl` | The only training file |
| `generated/json-calls-v3/validation.sft.jsonl` | The only validation file |
| `generated/json-calls-v3/test.sft.jsonl` | Frozen 2,500-row test file |
| `artifacts/checkpoints/json-calls-sft-v3-smoke/` | 50-step smoke snapshot |
| `artifacts/checkpoints/json-calls-sft-v3/` | The one full SFT checkpoint |
| `~/.cache/huggingface/` | Downloaded base model and tokenizer |

Checkpoint directories are gitignored. Leave them on the WSL disk.
