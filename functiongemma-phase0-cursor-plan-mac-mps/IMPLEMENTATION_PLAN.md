# Phase-0 Cursor Implementation Plan

## Frozen objective
Build, validate, fine-tune, and benchmark FunctionGemma 270M on a 25,000-row controlled corpus. Runtime contracts are frozen for this phase: single operation only; `set_string`, `set_integer`, `set_number`, `set_boolean`, `set_enum`, `set_date`, `no_change`; ACTIVE_FIELD and REACHABLE_FIELDS specialization.

## Non-goals
No multi-operation extraction, relative-date arithmetic, workflow reasoning, production quantization, multilingual expansion, or model-driven reachability.

## Repository rule
Every phase keeps stubs for later phases. A stub may raise `NotImplementedError` until its entry gate is passed. Do not delete interfaces to make tests pass.

## Phase 0 — bootstrap
Tasks:
1. Create package/CLI/config loader.
2. Pin Phase-0 spec and deterministic seed.
3. Add unit-test skeletons and audit report model.
4. Add local and CUDA-host setup scripts.

Exit Gate G0:
- `python -m form_sft.cli doctor` succeeds.
- config parses.
- tests import all modules.
- no generated corpus exists yet.

## Phase 1 — contracts + validators
Implement:
- SemanticContext dataclasses/models.
- operation contract.
- structural JSON Schema validation.
- semantic validator: field IDs, op/type mapping, enum membership, bounds, duplicate IDs.
- runtime specializer with ACTIVE_FIELD and REACHABLE_FIELDS.

Tests:
- valid operation per primitive.
- unknown field rejected.
- wrong op rejected.
- enum hallucination rejected.
- duplicate field/option rejected.
- bounds checked.
- active must be reachable.

Exit Gate G1:
- contract tests 100% pass.
- malformed context fails closed.
- specialized schemas expose only legal fields/options.

## Phase 2 — quota planner
Implement deterministic slot allocation BEFORE utterance realization.
Each slot receives:
- split
- primitive
- specialization mode
- reachable candidate bucket
- correction flag/subtype
- ASR flag/subtype
- realization source
- field identity class
- difficulty
- holdout family IDs

Hard requirements:
- exact quotas; tolerance = 0.
- correction=5000; ASR=3750; overlap=750.
- train/val/test=20000/2500/2500.
- active/reachable=18750/6250.
- family assignment occurs before realization.

Exit Gate G2:
- planner emits exactly 25,000 unique slot IDs.
- all quota equations reconcile exactly.
- same seed => byte-identical plan.
- different seed changes assignments but not quotas.

## Phase 3 — semantic truth generators
Implement generators for integer, number, boolean, enum, string, date, no_change.
Generate canonical target first, then context.
Rules:
- never ask teacher model for labels.
- opaque field/option IDs must be supported.
- explicit out-of-range extraction remains extraction; business validity stays separate where constrained decoding does not make target impossible.
- no_change includes other-person plausible values.

Exit Gate G3:
- 10k dry-run semantic samples: 0 structural/semantic errors.
- each primitive family has coverage tests.
- target can be regenerated deterministically from sample seed.

## Phase 4 — realization + transforms
Implement deterministic templates first.
Then correction transforms, fillers, conservative synthetic ASR transforms.
Teacher generator remains an adapter stub unless credentials/provider are configured.
Teacher may only paraphrase a known semantic target.

Exit Gate G4:
- semantic target is invariant across transforms.
- exact correction/ASR quotas maintained.
- no transform changes field/option/value semantics.
- teacher rows are rejected if target preservation cannot be proven.

## Phase 5 — split isolation + 25k build
Implement family registry and leakage checker.
Test partitions: IID 400; unseen fields 400; unseen templates 400; unseen numeric compositions 300; unseen enums 300; unseen corrections 250; ASR robustness 200; hard negatives 250.

Build outputs:
- generated/phase0-v1/train.jsonl
- validation.jsonl
- test.jsonl
- benchmark/*.jsonl
- manifest.json
- audit.json

Exit Gate G5 — DATASET GO:
- exactly 25,000 rows.
- exact quota match for every hard quota.
- zero duplicate IDs.
- zero structural errors.
- zero semantic errors.
- zero prohibited leakage.
- build is reproducible from seed.
- manual review: >=200 stratified rows accepted.

NO-GO: do not train if any G5 condition fails.

## Phase 6 — FunctionGemma compiler
Compile canonical rows to FunctionGemma conversations/tools using the tokenizer's installed chat template. Do not hand-code FunctionGemma control tokens.
Required checks:
- render `messages + tools` with `tokenizer.apply_chat_template`.
- verify assistant tool call survives round-trip inspection.
- measure token-length distribution.
- reject/truncate by explicit policy, never silently.

Exit Gate G6:
- 100 sampled rows render successfully.
- rendered targets are exact canonical operations.
- no custom/guessed FunctionGemma special tokens.
- p99 sequence length reported.

## Phase 7 — baseline before training
Run base FunctionGemma against frozen validation/test-format samples using greedy decoding.
Store predictions and metrics. This is A0.

Exit Gate G7:
- evaluator reports exact call, op, field, value, enum, no_change P/R, FMR.
- metrics broken down by primitive/difficulty/mode/noise/holdout.
- predictions are persisted.

## Phase 8 — remote CUDA full SFT
Default: full SFT of 270M model on one CUDA GPU. Start conservative; 24GB VRAM is ample for this model but actual sequence length and optimizer state still require preflight measurement.
Suggested initial config:
- bf16 when supported, otherwise fp16.
- gradient checkpointing off initially; enable only if needed.
- batch size 8, grad accumulation 4 (effective 32) as starting point.
- 3 epochs maximum for first run.
- LR 2e-5 starting hypothesis.
- eval/save each epoch.
- best checkpoint by validation exact-tool-call metric if callback implemented; otherwise validation loss + post-eval.
- greedy generation for benchmark.

Do not treat these hyperparameters as frozen dataset parameters; they are experiment config.

Exit Gate G8:
- CUDA visible and >=20 GiB VRAM recommended.
- Hugging Face model access verified.
- 50-step smoke train completes with finite loss.
- checkpoint reloads.
- full run only after smoke pass.

## Phase 9 — evaluation / GO-NO-GO
Evaluate frozen 2,500 test rows and compare to A0 and Qwen baseline if available.
Primary: exact tool call accuracy and false mutation rate.
Report all frozen thresholds from YAML.

Exit Gate G9:
GO only if safety/FMR and target capability thresholds pass. If not, diagnose by slice before adding data.

## Phase 10 — ablation ladder
A0 base; A1 primitives; A2 linguistic variation; A3 negatives; A4 corrections; A5 synthetic ASR; A6 teacher paraphrases; A7 real ASR; A8 reachable-field training.
Do not change frozen test set between ablations.

## Cursor execution instruction
Work one phase at a time. At each gate: run tests/audit, write `artifacts/gates/G<N>.json`, stop on failure, and report the failing invariant. Never auto-relax a quota or holdout to progress.

---

## Host execution contract — Mac -> WSL2 Ubuntu/CUDA

This contract is part of the Phase-0 implementation plan.

### Mac owns
- source edits and Cursor/Codex work;
- unit tests that do not require CUDA;
- Phase-0 quota planning, corpus generation, audit and leakage checks;
- frozen YAML/config and benchmark definitions.

### WSL2 owns
- a Linux-native `.venv` created by `scripts/bootstrap_wsl.sh`;
- Hugging Face/model caches under the Linux user's home directory;
- CUDA preflight, A0 GPU baseline, smoke SFT and full SFT;
- training `outputs/` and `checkpoints/` until explicitly pulled back.

### Sync rules
- `.rsync-filter` is authoritative for Mac -> WSL2 tree sync.
- Never sync `.venv`, Python bytecode/caches, IDE state, W&B state, outputs or checkpoints from Mac.
- Remote project path must be inside the WSL Linux filesystem, not `/mnt/c`, `/mnt/d`, etc.
- `sync_to_wsl.sh` is incremental and canonical; `package_for_remote.sh` is fallback only.
- `sync_dataset_to_wsl.sh` may be used after G5 when only generated Phase-0 artifacts changed.

### Pull-back rules
- `pull_results_from_wsl.sh` retrieves lightweight outputs/gates while excluding model-weight blobs.
- `pull_checkpoints_from_wsl.sh` is the only normal path for checkpoint retrieval and uses resumable rsync.

### WSL/CUDA Gate G8A
Before the existing G8 smoke-training gate:
- repository path is not under `/mnt/<drive>/`;
- WSL/Linux `.venv` exists and `doctor` passes;
- `nvidia-smi` is visible inside WSL2;
- `torch.cuda.is_available()` is true;
- detected GPU/VRAM/BF16 capability is logged;
- Phase-0 training JSONL exists on WSL after sync.

NO-GO: do not start smoke training if G8A fails.


## Phase-0 execution override: Apple Silicon Mac

For Phase-0, macOS/Apple Silicon MPS is the canonical training host. Run `scripts/bootstrap_macos.sh`, `scripts/preflight_mps.sh`, then the mandatory `scripts/smoke_train_macos.sh` 50-step gate before `scripts/train_phase0_macos.sh`. WSL2/CUDA is optional later infrastructure. Training code must remain portable through `--device auto|mps|cuda|cpu`; do not reintroduce a CUDA-only requirement.
