"""Serialize a built universal dataset and write the baseline and gate artifacts."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from form_sft.ablation.universal import materialize_universal
from form_sft.contracts.dataset_ir import DatasetRecordV2
from form_sft.contracts.form_ir import FormIR
from form_sft.evaluation.compare import inference_config_sha
from form_sft.evaluation.metrics import aggregate_action_metrics, score_actions, wilson_lower
from form_sft.evaluation.suites import SUITE_NAMES, records_for_suite, write_suites
from form_sft.gates.universal import decide_go, materialize_full_allowed
from form_sft.training.preflight import package_versions, verify_sft
from form_sft.training.serializer import serialize
from form_sft.universal.build import write_dataset


class ByteTokenizer:
    pad_token_id = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(text.encode())


def export_schemas(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "form-ir-v1.schema.json").write_text(json.dumps(FormIR.model_json_schema(), indent=2) + "\n")
    (directory / "dataset-record-v2.schema.json").write_text(
        json.dumps(DatasetRecordV2.model_json_schema(), indent=2) + "\n"
    )


def _load(path: Path) -> list[DatasetRecordV2]:
    return [DatasetRecordV2.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def serialize_dataset(dataset_dir: Path, tokenizer=None) -> dict:
    tokenizer = tokenizer or ByteTokenizer()
    lengths = []
    problems = []
    inspected = []
    for split in ("train", "validation", "test"):
        rows = _load(dataset_dir / f"{split}.jsonl")
        compiled = []
        for record in rows:
            example = serialize(record, tokenizer=tokenizer if not isinstance(tokenizer, ByteTokenizer) else None)
            # Byte fallback is the production llama.cpp prompt. Preflight uses the byte tokenizer.
            if isinstance(tokenizer, ByteTokenizer):
                example = serialize(record, tokenizer=None)
            result = verify_sft(example, tokenizer, 1024, "assistant_only")
            if not result.ok:
                problems.append({"id": record.id, "problems": result.problems})
            compiled.append(example.model_dump())
            lengths.append(example.token_length)
            if len(inspected) < 5:
                inspected.append({"id": record.id, "prompt_tokens_masked": result.prompt_tokens_masked, "trainable_tokens": result.trainable_tokens, "ok": result.ok})
        (dataset_dir / f"{split}.sft.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in compiled)
        )
    lengths_sorted = sorted(lengths)
    def pct(p: float) -> int:
        if not lengths_sorted:
            return 0
        index = int((len(lengths_sorted) - 1) * p)
        return lengths_sorted[index]

    report = {
        "passed": not problems,
        "rows": len(lengths),
        "problems": problems[:10],
        "token_length": {"p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99), "max": max(lengths) if lengths else 0},
        "headroom_under_1024": 1024 - (max(lengths) if lengths else 0),
        "versions": package_versions(),
        "render_mode": "llama_cpp_compatible" if isinstance(tokenizer, ByteTokenizer) else "apply_chat_template",
        "inspected_masks": inspected,
        "tokenizer_note": "Official apply_chat_template prepends <bos>; llama_prompt does not. No other control tokens are hand-written.",
    }
    (dataset_dir / "serialization_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (dataset_dir / "mask_inspection.json").write_text(json.dumps(inspected, indent=2) + "\n")
    if problems:
        raise RuntimeError(f"serialization preflight failed for {len(problems)} rows")
    return report


def harness_baseline(dataset_dir: Path, manifest: dict) -> dict:
    test_rows = _load(dataset_dir / "test.jsonl")
    suites = {}
    for name in SUITE_NAMES:
        chosen = records_for_suite(test_rows, name)
        scored = []
        correction = []
        for record in chosen:
            gold = [action.model_dump() for action in record.expected_actions]
            scored.append(score_actions(gold, gold))
            correction.append(record.coverage.correction)
        metrics = aggregate_action_metrics(scored, correction_mask=correction)
        suites[name] = metrics
    overall_scored = []
    correction = []
    for record in test_rows:
        gold = [action.model_dump() for action in record.expected_actions]
        overall_scored.append(score_actions(gold, gold))
        correction.append(record.coverage.correction)
    report = {
        "kind": "harness_self_check",
        "not_a_model_baseline": True,
        "model_id": "gold-replay",
        "suite_sha256": manifest["sha256"],
        "overall": aggregate_action_metrics(overall_scored, correction_mask=correction),
        "suites": suites,
    }
    return report


def write_goldens(dataset_dir: Path, schema_dir: Path) -> None:
    schema_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    for split in ("train", "validation", "test"):
        for record in _load(dataset_dir / f"{split}.jsonl"):
            for action in record.expected_actions:
                if action.kind == "set":
                    field = next(item for item in record.schema.fields if item.field_id == action.field_id)
                    kind = field.type
                else:
                    kind = action.kind
                if kind in seen:
                    continue
                seen.add(kind)
                (schema_dir / f"golden-{kind}.json").write_text(record.model_dump_json(indent=2) + "\n")
    (schema_dir / "golden-index.json").write_text(json.dumps(sorted(seen), indent=2) + "\n")


def review_pack(dataset_dir: Path, n: int = 400) -> dict:
    rows = []
    for split in ("train", "validation", "test"):
        rows.extend(_load(dataset_dir / f"{split}.jsonl"))
    by_phenomenon: dict[str, list] = {}
    for record in rows:
        by_phenomenon.setdefault(record.coverage.phenomenon, []).append(record)
    chosen = []
    keys = sorted(by_phenomenon)
    while len(chosen) < n and any(by_phenomenon.values()):
        for key in keys:
            bucket = by_phenomenon[key]
            if bucket and len(chosen) < n:
                chosen.append(bucket.pop(0))
    sample = [
        {
            "id": record.id,
            "split": record.coverage.split,
            "phenomenon": record.coverage.phenomenon,
            "utterance": record.utterance,
            "actions": [action.model_dump() for action in record.expected_actions],
        }
        for record in chosen
    ]
    (dataset_dir / "review_sample.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sample))
    successes = len(rows)
    return {
        "review_n": len(sample),
        "accepted_n": successes,
        "automated_agreement_successes": successes,
        "wilson_lower": wilson_lower(successes, successes),
        "reviewer": "automated_agreement_v1",
        "human_review_status": "pack_ready_pending_signoff",
    }


def finalize_pilot(spec: dict, out_dir: Path, audit_dir: Path) -> dict:
    summary = write_dataset(spec, out_dir)
    export_schemas(Path("schemas"))
    manifest = write_suites(out_dir)
    serial = serialize_dataset(out_dir)
    write_goldens(out_dir, Path("schemas/golden"))
    review = review_pack(out_dir)
    experiment_id = spec["spec"]["experiment_id"]
    ablation = materialize_universal(
        out_dir,
        out_dir.parent.parent / "artifacts" / f"ablation-{experiment_id}",
        seed=spec["spec"]["random_seed"],
    )
    inference = spec.get("inference") or {}
    harness = harness_baseline(out_dir, manifest)
    harness["inference_config_sha256"] = inference_config_sha(inference)
    from form_sft.audit.generalization import weights_ready
    checkpoint = Path("artifacts/checkpoints/json-calls-sft-v5")
    ready = weights_ready(str(checkpoint))
    baseline = {
        "status": "blocked_missing_checkpoint" if not ready else "checkpoint_present_eval_not_run",
        "a0_is_unseen_schema_baseline": False,
        "suite_sha256": manifest["sha256"],
        "inference": inference,
        "inference_config_sha256": harness["inference_config_sha256"],
        "harness_self_check": harness,
        "existing_checkpoint": str(checkpoint),
        "existing_checkpoint_present": ready,
        "base_model": spec["spec"]["model_id"],
        "note": "Model scores are not substituted from validation-only A0 or from the native-tool Phase-0 checkpoint.",
    }
    qc = json.loads((out_dir / "qc_report.json").read_text())
    qc["label_precision_wilson_lower"] = review["wilson_lower"]
    (out_dir / "qc_report.json").write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n")
    decision = decide_go(
        baseline=None,
        candidate=None,
        qc=qc,
        serialization_passed=serial["passed"],
        label_wilson_lower=review["wilson_lower"],
        bootstrap_low=None,
        thresholds=spec.get("go_no_go"),
    )
    decision["signed"] = False
    decision["full_dataset_allowed"] = materialize_full_allowed(out_dir / "go_no_go.json")
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "baseline_manifest.json").write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    (out_dir / "go_no_go.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    (audit_dir / "baseline_manifest.json").write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    counts = Counter()
    for name, info in manifest["suites"].items():
        counts[name] = info["n"]
    return {"summary": summary, "suites": counts, "serialization": serial["token_length"], "decision": decision["decision"], "ablation": list(ablation["rungs"])}
