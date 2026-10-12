"""Audit the existing corpora and freeze a reproducible unseen-schema baseline manifest.

Validation-only A0 metrics are recorded as historical context. They are not an
unseen-schema baseline.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from form_sft.adapters.calls import canonical_calls
from form_sft.adapters.legacy import from_legacy_row
from form_sft.contracts.models import CanonicalRow
from form_sft.dataset.leakage import check_leakage
from form_sft.evaluation.metrics import wilson_lower
from form_sft.generation.catalog import snake
from form_sft.training.json_prompt import calls_from_row
from form_sft.validation.record import validate_record
from form_sft.validation.semantic import validate_row

PHASE0_COUNTS = {"train": 20000, "validation": 2500, "test": 2500}


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 4096:
        return False
    head = path.read_text(encoding="utf-8", errors="replace")[:80]
    return head.startswith("version https://git-lfs.github.com")


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def audit_corpora(canonical_paths: list[Path], sft_paths: list[Path], predictions: list[Path]) -> dict:
    inventory = {"files": [], "counts": {}, "fingerprints": {}, "field_ids": {}, "problems": []}
    label_rows = []
    rows_for_leak = []
    migration_errors = 0
    migration_checked = 0
    for path in canonical_paths:
        item = {"path": str(path), "exists": path.exists(), "lfs_pointer": _pointer(path), "sha256": _sha256(path)}
        if not path.exists() or item["lfs_pointer"]:
            item["status"] = "missing"
            inventory["problems"].append(f"missing or LFS pointer: {path}")
            inventory["files"].append(item)
            continue
        count = 0
        fingerprints = set()
        field_ids = set()
        for raw in _iter_jsonl(path):
            count += 1
            row = CanonicalRow.model_validate(raw)
            fingerprints.add(row.semantic_context.fingerprint)
            for field in row.semantic_context.fields:
                field_ids.add(field.field_id)
                if row.metadata.get("field_identity") == "semantic" and field.field_id != snake(field.label):
                    label_rows.append({"id": row.id, "field_id": field.field_id, "label": field.label, "code": "LABEL_ID_MISMATCH"})
            if validate_row(row):
                label_rows.append({"id": row.id, "code": "SEMANTIC_INVALID"})
            if "test" in path.name or path.parent.name == "benchmark":
                rows_for_leak.append(raw)
            if migration_checked < 25000 and path.name in {"train.jsonl", "validation.jsonl", "test.jsonl"}:
                converted = from_legacy_row(row)
                if validate_record(converted):
                    migration_errors += 1
                calls = calls_from_row(row)
                adapted = canonical_calls(converted, "production_json")
                if calls != adapted:
                    inventory["problems"].append(f"call mismatch {row.id}")
                migration_checked += 1
        item["rows"] = count
        item["status"] = "ok"
        inventory["files"].append(item)
        inventory["counts"][str(path)] = count
        inventory["fingerprints"][str(path)] = len(fingerprints)
        inventory["field_ids"][str(path)] = len(field_ids)
    for path in sft_paths:
        item = {"path": str(path), "exists": path.exists(), "lfs_pointer": _pointer(path), "sha256": _sha256(path)}
        if not path.exists() or item["lfs_pointer"]:
            item["status"] = "missing"
            inventory["problems"].append(f"missing or LFS pointer: {path}")
        else:
            item["rows"] = sum(1 for _ in _iter_jsonl(path))
            item["status"] = "ok"
        inventory["files"].append(item)
    prediction_summary = []
    for path in predictions:
        if not path.exists():
            prediction_summary.append({"path": str(path), "status": "missing"})
            continue
        metrics = json.loads(path.read_text()) if path.suffix == ".json" else {"rows": sum(1 for _ in _iter_jsonl(path))}
        prediction_summary.append({"path": str(path), "status": "ok", "n": metrics.get("n") or metrics.get("overall", {}).get("n")})
    leaks = check_leakage(rows_for_leak) if rows_for_leak else []
    return {
        "inventory": inventory,
        "label_errors": label_rows[:50],
        "label_error_count": len(label_rows),
        "migration_checked": migration_checked,
        "migration_errors": migration_errors,
        "prediction_files": prediction_summary,
        "leakage_on_supplied_rows": len(leaks),
    }


def stratified_review(path: Path, n: int = 200) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    if not path.exists() or _pointer(path):
        return []
    for raw in _iter_jsonl(path):
        buckets[raw["metadata"].get("primitive") or raw["target"]["name"]].append(
            {"id": raw["id"], "primitive": raw["metadata"].get("primitive"), "partition": raw["metadata"].get("test_partition"), "utterance": raw["utterance"], "target": raw["target"]}
        )
    chosen = []
    keys = sorted(buckets)
    while len(chosen) < n and any(buckets.values()):
        for key in keys:
            if buckets[key] and len(chosen) < n:
                chosen.append(buckets[key].pop(0))
    return chosen


def train_inference_diff() -> dict:
    return {
        "categories": [
            {"code": "ENUM_NAME", "detail": "Canonical set_enum/option_id becomes production set_choice/value."},
            {"code": "NO_CHANGE", "detail": "Canonical no_change tool becomes an empty function_calls list."},
            {"code": "USER_CONTEXT", "detail": "Native prompts contain the utterance only; production prompts add Context and Tools."},
            {"code": "TOOL_SCHEMA", "detail": "specialize() is per field, tools_for_row merges fields, tool_line is text."},
            {"code": "EXTRA_OPS", "detail": "Production fixtures add clear_field and set_multi_choice, which Phase-0 OperationName does not contain."},
            {"code": "TOKENIZER", "detail": "Compile and eval load AutoProcessor; train.py must use the same loader."},
            {"code": "A0_SCOPE", "detail": "A0 metrics are validation-only and are not an unseen-schema baseline."},
        ]
    }


def historical_reference(repo: Path) -> dict:
    a0 = repo / "artifacts/eval/A0/A0_metrics.json"
    a1 = repo / "artifacts/eval/A1-test/A1-test_metrics.json"
    g9 = repo / "artifacts/gates/G9.json"
    out = {"a0_is_unseen_schema_baseline": False, "a0_scope": "validation_only"}
    if a0.exists():
        metrics = json.loads(a0.read_text())
        out["a0"] = {"exact_call": metrics["overall"]["exact_call"], "false_mutation_rate": metrics["overall"]["false_mutation_rate"], "split_path": metrics.get("split_path")}
    if a1.exists():
        metrics = json.loads(a1.read_text())
        holdout = metrics.get("by", {}).get("holdout", {})
        out["phase0_checkpoint_frozen_test"] = {
            "exact_call": metrics["overall"]["exact_call"],
            "false_mutation_rate": metrics["overall"]["false_mutation_rate"],
            "unseen_fields": holdout.get("unseen_fields", {}).get("exact_call"),
            "unseen_enums": holdout.get("unseen_enums", {}).get("exact_call"),
            "unseen_numeric_compositions": holdout.get("unseen_numeric_compositions", {}).get("exact_call"),
            "interface": "native_tool_calls",
            "note": "Do not compare these numbers with production JSON scores.",
        }
    if g9.exists():
        out["g9_decision"] = json.loads(g9.read_text()).get("decision")
    return out


def write_audit(repo: Path, out_dir: Path) -> dict:
    canonical = [repo / "generated/phase0-v1" / name for name in ("train.jsonl", "validation.jsonl", "test.jsonl")]
    sft = [
        repo / "generated/phase0-v1/train.sft.jsonl",
        repo / "generated/json-calls-v5/train.sft.jsonl",
        repo / "generated/json-calls-v5/test.sft.jsonl",
    ]
    predictions = [
        repo / "artifacts/eval/A0/A0_metrics.json",
        repo / "artifacts/eval/A1-test/A1-test_metrics.json",
    ]
    report = audit_corpora(canonical, sft, predictions)
    review = stratified_review(repo / "generated/phase0-v1/test.jsonl", 200)
    counts = {}
    for split, expected in PHASE0_COUNTS.items():
        path = repo / "generated/phase0-v1" / f"{split}.jsonl"
        actual = sum(1 for _ in _iter_jsonl(path)) if path.exists() and not _pointer(path) else None
        counts[split] = {"expected": expected, "actual": actual, "ok": actual == expected}
    leakage = {"phase0_audit": None}
    audit_path = repo / "generated/phase0-v1/audit.json"
    if audit_path.exists():
        previous = json.loads(audit_path.read_text())
        leakage = {
            "prohibited_leakage": previous.get("prohibited_leakage"),
            "duplicate_content": previous.get("duplicate_content"),
            "structural_errors": previous.get("structural_errors"),
            "semantic_errors": previous.get("semantic_errors"),
        }
    payload = {
        "counts": counts,
        "inventory": report["inventory"],
        "migration": {"checked": report["migration_checked"], "errors": report["migration_errors"]},
        "label_error_count": report["label_error_count"],
        "historical": historical_reference(repo),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inventory.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (out_dir / "label_review_sample.json").write_text(json.dumps({"n": len(review), "rows": review}, indent=2) + "\n")
    (out_dir / "leakage.json").write_text(json.dumps(leakage, indent=2, sort_keys=True) + "\n")
    (out_dir / "train_inference_diff.json").write_text(json.dumps(train_inference_diff(), indent=2) + "\n")
    successes = len(review)
    (out_dir / "baseline_manifest.json").write_text(
        json.dumps(
            {
                "status": "awaiting_suite_hash",
                "a0_is_unseen_schema_baseline": False,
                "historical": payload["historical"],
                "review_n": len(review),
                "review_wilson_lower": wilson_lower(successes, len(review)) if review else 0.0,
                "checkpoints": {
                    "base": "google/functiongemma-270m-it",
                    "existing": "artifacts/checkpoints/json-calls-sft-v5",
                    "existing_present": (repo / "artifacts/checkpoints/json-calls-sft-v5").exists(),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return payload


def phase0_count_ok(repo: Path) -> bool:
    for split, expected in PHASE0_COUNTS.items():
        path = repo / "generated/phase0-v1" / f"{split}.jsonl"
        if not path.exists() or _pointer(path):
            return False
        if sum(1 for _ in _iter_jsonl(path)) != expected:
            return False
    return True


def weights_ready(model_id: str) -> bool:
    path = Path(model_id)
    if not path.exists():
        return False
    if path.is_file():
        return True
    names = {item.name for item in path.iterdir()} if path.is_dir() else set()
    return "config.json" in names or any(name.endswith((".safetensors", ".gguf", ".bin")) for name in names)
