# Cursor / Codex execution prompt
Implement this repository strictly according to IMPLEMENTATION_PLAN.md and config/phase0-v1.yaml. Work one gated phase at a time starting at G0. Preserve all future-phase stubs. Never relax quotas, leakage rules, semantic contracts, or GO/NO-GO thresholds to make a gate pass. After each phase run tests/audits and write artifacts/gates/G<N>.json containing status, commands, counts, failures, and git diff summary. Stop immediately on a failed gate and explain the invariant. Semantic truth must be generated before utterance realization. Teacher models may paraphrase only; they may never generate labels. Split/family assignment must happen before realization. Do not hand-code FunctionGemma control tokens: use the tokenizer chat template/tools interface. Do not begin full GPU training until G0-G8 smoke requirements pass.

## Mac -> WSL2 boundary
Treat macOS as the development/dataset-build host and Ubuntu under WSL2 as the CUDA training host. Preserve `.rsync-filter` and the sync/bootstrap/pull scripts. Never propose copying `.venv` across hosts. Never make `/mnt/c/...` the training working directory. Training outputs/checkpoints belong to WSL until explicitly pulled back. Keep lightweight result pull and checkpoint pull as separate operations.


## Phase-0 execution override: Apple Silicon Mac

For Phase-0, macOS/Apple Silicon MPS is the canonical training host. Run `scripts/bootstrap_macos.sh`, `scripts/preflight_mps.sh`, then the mandatory `scripts/smoke_train_macos.sh` 50-step gate before `scripts/train_phase0_macos.sh`. WSL2/CUDA is optional later infrastructure. Training code must remain portable through `--device auto|mps|cuda|cpu`; do not reintroduce a CUDA-only requirement.
