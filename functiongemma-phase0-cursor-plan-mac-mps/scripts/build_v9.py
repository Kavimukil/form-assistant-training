"""Materialize universal-full-v9 only after the v7/v8 prediction gate passes."""
from __future__ import annotations

import json
from pathlib import Path

from form_sft.evaluation.value_misses import compare_predictions
from form_sft.training.compiler import load_tokenizer
from form_sft.training.length_gate import require_fit
from form_sft.universal.pipeline import serialize_dataset
from form_sft.universal.v9_replay import write_v9_dataset

ROOT = Path(__file__).resolve().parents[1]
V7_PREDICTIONS = ROOT / "artifacts/eval/universal-full-v7-test/universal-full-v7-test_predictions.jsonl"
V8_PREDICTIONS = ROOT / "artifacts/eval/universal-full-v8-test/universal-full-v8-test_predictions.jsonl"
DATASET = ROOT / "generated/universal-full-v1/test.jsonl"
V7_TRAIN = ROOT / "generated/universal-full-v7/train.jsonl"
FROZEN = ROOT / "generated/universal-full-v1"
OUT = ROOT / "generated/universal-full-v9"
AUDIT = ROOT / "artifacts/audit/universal-v9-gate.json"


def main() -> None:
    comparison = compare_predictions(V7_PREDICTIONS, V8_PREDICTIONS, DATASET)
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not comparison.get("proceed"):
        raise SystemExit(f"v9 build gate stopped materialization; see {AUDIT}")
    summary = write_v9_dataset(V7_TRAIN, FROZEN, OUT, comparison)
    serial = serialize_dataset(OUT)
    report = require_fit(OUT / "train.sft.jsonl", load_tokenizer("google/functiongemma-270m-it"), 1152)
    print(json.dumps({"dataset": summary, "serial_passed": serial["passed"], "length": report}, indent=2))


if __name__ == "__main__":
    main()
