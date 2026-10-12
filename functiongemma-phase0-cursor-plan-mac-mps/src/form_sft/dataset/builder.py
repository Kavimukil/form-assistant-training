"""G5 orchestration: plan -> semantic truth -> realization -> validate -> leakage -> quotas -> write.

Fails closed: nothing is written unless every check passes.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from form_sft.contracts.models import CanonicalRow, Operation, SemanticContext
from form_sft.dataset.leakage import check_leakage, dedupe_key
from form_sft.dataset.quota_planner import Slot, plan_slots, plan_to_jsonl
from form_sft.dataset.quotas import check_quotas, failed
from form_sft.generation.generator import generate_semantic_truth, realize_row
from form_sft.validation.semantic import validate_row
from form_sft.validation.structural import validate_structural

GENERATOR_VERSION = "phase0-generator-1.0"
MAX_ATTEMPTS = 60
ATTEMPT_STRIDE = 1_000_003


class BuildError(RuntimeError):
    pass


def row_to_json(row: CanonicalRow) -> str:
    return json.dumps(row.model_dump(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def make_row(slot: Slot, seed: int, attempt: int = 0) -> tuple[CanonicalRow, dict]:
    eff = seed + attempt * ATTEMPT_STRIDE
    sample = generate_semantic_truth(slot, eff)                 # G3: truth first
    target0, ctx0 = copy.deepcopy(sample["target"]), copy.deepcopy(sample["context"])
    text = realize_row(sample, slot, eff)                       # G4: then surface
    if sample["target"] != target0 or sample["context"] != ctx0:
        raise BuildError(f"{slot.id}: realization mutated semantic truth")
    real = sample["realization"]
    md = {
        "experiment_id": slot.id.rsplit("-", 2)[0],
        "slot_id": slot.id, "seed": seed, "attempt": attempt,
        "split": slot.split, "test_partition": slot.test_partition,
        "primitive": slot.primitive, "specialization_mode": slot.specialization_mode,
        "reachable_bucket": slot.reachable_bucket, "candidate_count": sample["n_candidates"],
        "correction": slot.correction, "correction_subtype": slot.correction_subtype,
        "asr_noise": slot.asr_noise, "asr_subtype": slot.asr_subtype, "asr_applied": real.get("asr_applied"),
        "source": slot.source, "field_identity": slot.field_identity, "difficulty": slot.difficulty,
        "families": dict(slot.families),
        "template_id": real["template_id"], "template_variant": real["template_variant"],
        "value_form": real["value_form"], "correction_template": real.get("correction_template"),
        "nc_kind": sample["nc_kind"], "subject_field_id": sample["subject_field_id"],
        "subject_type": sample["subject_type"], "domain": sample["domain"],
        "teacher_mode": real.get("teacher_mode"),
    }
    row = CanonicalRow(id=slot.id, metadata=md, semantic_context=SemanticContext(**sample["context"]),
                       utterance=text, target=Operation(**sample["target"]))
    errs = validate_structural(row) + validate_row(row)
    if errs:
        raise BuildError(f"{slot.id}: validation errors {errs}")
    return row, sample


def build_rows(spec: dict, slots: list[Slot] | None = None) -> list[CanonicalRow]:
    seed = spec["spec"]["random_seed"]
    slots = slots if slots is not None else plan_slots(spec)
    seen: dict[str, str] = {}
    rows: list[CanonicalRow] = []
    for slot in slots:
        for attempt in range(MAX_ATTEMPTS):
            row, _ = make_row(slot, seed, attempt)
            key = dedupe_key(row.model_dump())
            if key not in seen:
                seen[key] = slot.id
                rows.append(row)
                break
        else:
            raise BuildError(f"{slot.id}: could not produce a unique row in {MAX_ATTEMPTS} attempts")
    return rows


def records_from_rows(rows: list[CanonicalRow]) -> list[dict]:
    return [r.metadata for r in rows]


def _write(path: Path, lines: list[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(l + "\n" for l in lines).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def build(spec: dict, spec_path: str | Path | None = None, out_root: str | Path = "generated") -> dict:
    exp = spec["spec"]["experiment_id"]
    out = Path(out_root) / exp
    slots = plan_slots(spec)
    rows = build_rows(spec, slots)

    dumped = [r.model_dump() for r in rows]
    checks = check_quotas(records_from_rows(rows), spec)
    bad = failed(checks)
    if bad:
        raise BuildError("quota mismatch: " + "; ".join(f"{c['name']} {c['expected']}!={c['actual']}" for c in bad))
    leaks = check_leakage(dumped)
    if leaks:
        raise BuildError(f"prohibited leakage: {len(leaks)} e.g. {leaks[:3]}")

    files: dict[str, str] = {}
    for split, name in (("train", "train.jsonl"), ("validation", "validation.jsonl"), ("test", "test.jsonl")):
        files[name] = _write(out / name, [row_to_json(r) for r in rows if r.metadata["split"] == split])
    for part in spec["test_partitions"]:
        files[f"benchmark/{part}.jsonl"] = _write(
            out / "benchmark" / f"{part}.jsonl",
            [row_to_json(r) for r in rows if r.metadata["split"] == "test" and r.metadata["test_partition"] == part])
    files["plan.jsonl"] = _write(out / "plan.jsonl", plan_to_jsonl(slots).splitlines())

    spec_bytes = Path(spec_path).read_bytes() if spec_path else json.dumps(spec, sort_keys=True).encode()
    manifest = {
        "experiment_id": exp, "spec_version": spec["spec"]["version"], "random_seed": spec["spec"]["random_seed"],
        "model_id": spec["spec"]["model_id"], "generator_version": GENERATOR_VERSION,
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "total_rows": len(rows),
        "split_counts": {s: sum(1 for r in rows if r.metadata["split"] == s) for s in spec["splits"]},
        "test_partition_counts": {p: sum(1 for r in rows if r.metadata["test_partition"] == p and r.metadata["split"] == "test")
                                  for p in spec["test_partitions"]},
        "teacher": {"mode": "deterministic_paraphrase_no_credentials", "external_calls": 0,
                    "labels_from_teacher": False},
        "quota_checks": {"total": len(checks), "failed": 0},
        "files": dict(sorted(files.items())),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
