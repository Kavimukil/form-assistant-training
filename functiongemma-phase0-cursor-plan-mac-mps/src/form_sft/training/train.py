from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from form_sft.training.compiler import load_tokenizer
from form_sft.training.device import select_device


def load_training_tokenizer(model_id: str):
    """Same loader as compile and eval, unwrapped for TRL's processing class."""
    loaded = load_tokenizer(model_id)
    tokenizer = getattr(loaded, "tokenizer", loaded)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train-full-sft.yaml")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--train-file", default=None, help="override train SFT jsonl path")
    parser.add_argument("--validation-file", default=None, help="override validation SFT jsonl path")
    parser.add_argument("--output-dir", default=None, help="override checkpoint output_dir")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    plan = select_device(args.device)
    model_id = cfg["model_id"]
    output_dir = args.output_dir or cfg["output_dir"]
    if args.smoke_steps and not args.output_dir:
        output_dir = f"{output_dir}-smoke"

    print(f"Phase-0 device={plan.device} dtype={plan.torch_dtype} bf16={plan.bf16} fp16={plan.fp16}")

    tokenizer = load_training_tokenizer(model_id)
    # Prefer dtype= for transformers>=5; keep torch_dtype kw for older callers via from_pretrained.
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=plan.torch_dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=plan.torch_dtype)
    model.to(plan.device)

    train_file = args.train_file or cfg.get("train_file") or "generated/phase0-v1/train.sft.jsonl"
    validation_file = args.validation_file or cfg.get("validation_file") or "generated/phase0-v1/validation.sft.jsonl"
    print(f"Phase-0 train_file={train_file}")
    print(f"Phase-0 validation_file={validation_file}")
    print(f"Phase-0 output_dir={output_dir}")
    ds = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "validation": validation_file,
        },
    )
    loss_policy = cfg.get("loss_policy", "full_sequence")
    # This TRL build applies assistant_only_loss only to conversational rows.
    # The SFT files are plain text, so assistant-only loss uses the prompt prefix
    # already stored on each row and trains on the completion after it.
    if loss_policy == "assistant_only":
        def _with_completion(row: dict) -> dict:
            prompt = row["prompt"]
            text = row["text"]
            if not text.startswith(prompt):
                raise ValueError(f"{row.get('id')}: prompt is not a prefix of text")
            row["completion"] = text[len(prompt) :]
            return row

        for split in ds:
            if "prompt" not in ds[split].column_names:
                raise SystemExit(f"{split} rows have no prompt column for assistant_only loss")
            ds[split] = ds[split].map(_with_completion)
    else:
        for split in ds:
            if "prompt" in ds[split].column_names:
                ds[split] = ds[split].remove_columns("prompt")

    # transformers>=5 removed warmup_ratio; derive warmup_steps from ratio × planned steps.
    per_device = int(cfg["per_device_train_batch_size"])
    grad_acc = int(cfg["gradient_accumulation_steps"])
    effective_batch = max(1, per_device * grad_acc)
    if args.smoke_steps:
        total_steps = args.smoke_steps
    else:
        steps_per_epoch = max(1, (len(ds["train"]) + effective_batch - 1) // effective_batch)
        total_steps = steps_per_epoch * int(cfg["num_train_epochs"])
    warmup_steps = max(0, int(float(cfg.get("warmup_ratio", 0.0)) * total_steps))

    sft_extra = {}
    if loss_policy == "assistant_only":
        fields = getattr(SFTConfig, "__dataclass_fields__", {})
        if "completion_only_loss" not in fields:
            raise SystemExit("assistant_only loss policy is not supported by the installed TRL")
        sft_extra["completion_only_loss"] = True
    elif loss_policy != "full_sequence":
        raise SystemExit(f"unknown loss_policy {loss_policy}")

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=per_device,
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=grad_acc,
        weight_decay=cfg["weight_decay"],
        warmup_steps=warmup_steps,
        logging_steps=min(cfg["logging_steps"], args.smoke_steps) if args.smoke_steps else cfg["logging_steps"],
        # Smoke: no mid-run checkpoints; final reloadable snapshot is saved after train().
        save_strategy="no" if args.smoke_steps else cfg["save_strategy"],
        eval_strategy="no" if args.smoke_steps else cfg["eval_strategy"],
        bf16=plan.bf16,
        fp16=plan.fp16,
        use_cpu=(plan.device == "cpu"),
        report_to="none",
        max_steps=args.smoke_steps if args.smoke_steps else -1,
        seed=cfg.get("seed", 20261006),
        dataset_text_field="text",
        max_length=int(cfg.get("max_seq_length", 1024)),
        packing=bool(cfg.get("packing", False)),
        **sft_extra,
    )

    # Compiler creates the `text` column using FunctionGemma's tokenizer/chat template.
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=None if args.smoke_steps else ds["validation"],
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    loss = getattr(train_result, "training_loss", None)
    if loss is not None and (loss != loss or loss == float("inf")):  # NaN/Inf
        raise SystemExit(f"SMOKE/TRAIN FAILED: non-finite loss={loss}")

    # Always persist a reloadable snapshot (smoke and full).
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    log_history = []
    if getattr(trainer, "state", None) is not None:
        log_history = list(getattr(trainer.state, "log_history", []) or [])
    step_losses = [h["loss"] for h in log_history if isinstance(h, dict) and "loss" in h]
    finite = all(isinstance(x, (int, float)) and x == x and x != float("inf") for x in step_losses) if step_losses else (
        loss is not None and loss == loss and loss != float("inf")
    )
    report = {
        "device": plan.device,
        "smoke_steps": args.smoke_steps or None,
        "output_dir": output_dir,
        "train_loss": loss,
        "step_losses": step_losses,
        "finite_loss": bool(finite),
        "log_history": log_history,
        "loss_policy": loss_policy,
        "checkpoint_reload": False,
    }

    # G8 reload check: weights must load back from the output directory.
    try:
        reloaded = AutoModelForCausalLM.from_pretrained(output_dir, dtype=plan.torch_dtype)
    except TypeError:
        reloaded = AutoModelForCausalLM.from_pretrained(output_dir, torch_dtype=plan.torch_dtype)
    _ = next(reloaded.parameters()).shape
    report["checkpoint_reload"] = True
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "smoke_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Also keep a trainer_state-like stub for gate tooling that looks for log_history.
    (Path(output_dir) / "trainer_state.json").write_text(
        json.dumps({"log_history": log_history, "train_loss": loss}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Phase-0 checkpoint_reload=PASS dir={output_dir} loss={loss}")
    if args.smoke_steps:
        print(f"PHASE0_50_STEP_SMOKE_INTERNAL: steps={args.smoke_steps} loss={loss}")


if __name__ == "__main__":
    # MPS fallback is opt-in via environment if an unsupported op appears; do not
    # silently enable it because CPU fallbacks can hide performance problems.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    main()
