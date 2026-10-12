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

`git lfs pull` is required. `generated/json-calls-v1/train.sft.jsonl` is stored in Git LFS (about 74 MB). Without LFS you get a pointer file and training fails.

Confirm the dataset is real text, not a pointer:

```bash
head -c 80 generated/json-calls-v1/train.sft.jsonl; echo
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

## 7. Smoke test (50 steps)

Do this before a full run. The smoke run writes `artifacts/checkpoints/<name>-smoke/` and does not keep mid-run checkpoints. It still saves a final snapshot and `smoke_report.json`.

JSON-call corpus (the dataset in this repo):

```bash
python -m form_sft.training.train \
  --config config/train-json-calls.yaml \
  --device cuda \
  --smoke-steps 50
```

Phase-0 form corpus, after `generated/phase0-v1/train.sft.jsonl` and `validation.sft.jsonl` exist:

```bash
python -m form_sft.training.train \
  --config config/train-full-sft.yaml \
  --device cuda \
  --smoke-steps 50
```

A passing smoke run prints a finite loss and `checkpoint_reload=PASS`. Stop if the loss is NaN or the process is killed for out-of-memory.

## 8. Full training

JSON-call SFT (3 epochs). Checkpoints go to `artifacts/checkpoints/json-calls-sft/`:

```bash
python -m form_sft.training.train \
  --config config/train-json-calls.yaml \
  --device cuda
```

Phase-0 full SFT. Checkpoints go to `artifacts/checkpoints/phase0-full-sft/`:

```bash
python -m form_sft.training.train \
  --config config/train-full-sft.yaml \
  --device cuda
```

`./scripts/train_phase0.sh` runs the Phase-0 smoke test and then the full Phase-0 job. It uses `--device auto`, so run `./scripts/preflight_wsl_cuda.sh` first.

On CUDA, the trainer uses BF16 when the GPU supports it, otherwise FP16. Watch progress with `nvidia-smi` in a second Ubuntu terminal.

## 9. Evaluate the checkpoint

JSON-call validation set:

```bash
python -m form_sft.cli evaluate-json \
  --file generated/json-calls-v1/validation.sft.jsonl \
  --model-id artifacts/checkpoints/json-calls-sft \
  --device cuda \
  --tag json-calls-val \
  --batch-size 8
```

Phase-0 validation set (`generated/phase0-v1/validation.jsonl`):

```bash
python -m form_sft.cli evaluate \
  --split validation \
  --model-id artifacts/checkpoints/phase0-full-sft \
  --device cuda \
  --tag phase0-val \
  --batch-size 8
```

Metrics and predictions are written under `artifacts/eval/<tag>/`.

## 10. Where files land

| Path | What it is |
| --- | --- |
| `artifacts/checkpoints/json-calls-sft-smoke/` | JSON-call 50-step smoke snapshot |
| `artifacts/checkpoints/json-calls-sft/` | JSON-call full SFT |
| `artifacts/checkpoints/phase0-full-sft-smoke/` | Phase-0 50-step smoke snapshot |
| `artifacts/checkpoints/phase0-full-sft/` | Phase-0 full SFT |
| `~/.cache/huggingface/` | Downloaded base model and tokenizer |

These checkpoint directories are gitignored. Leave them on the WSL disk.
