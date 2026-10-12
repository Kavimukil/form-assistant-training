"""Corpus audit: re-reads generated JSONL and re-proves every G5 invariant from the files alone."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from form_sft.contracts.models import CanonicalRow
from form_sft.dataset.leakage import check_leakage, duplicate_content_count
from form_sft.dataset.quotas import check_quotas, failed
from form_sft.validation.semantic import validate_row
from form_sft.validation.structural import validate_structural

PRIM_FOR_OP = {"set_string": "string", "set_integer": "integer", "set_number": "number", "set_boolean": "boolean",
               "set_enum": "enum", "set_date": "date", "no_change": "no_change"}


def _read(path: Path) -> list[str]:
    return path.read_text().splitlines() if path.exists() else []


def audit_dir(spec: dict, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    problems: list[str] = []
    split_files = {"train": "train.jsonl", "validation": "validation.jsonl", "test": "test.jsonl"}
    rows: list[CanonicalRow] = []
    parse_errors = 0
    file_counts: dict[str, int] = {}
    for split, name in split_files.items():
        lines = _read(out / name)
        file_counts[name] = len(lines)
        if len(lines) != spec["splits"][split]:
            problems.append(f"{name}: {len(lines)} rows, expected {spec['splits'][split]}")
        for ln in lines:
            try:
                r = CanonicalRow.model_validate_json(ln)
            except Exception:  # noqa: BLE001 - any parse failure is a structural error
                parse_errors += 1
                continue
            if r.metadata.get("split") != split:
                problems.append(f"{r.id}: split metadata {r.metadata.get('split')} != file {split}")
            rows.append(r)

    structural = sum(len(validate_structural(r)) for r in rows) + parse_errors
    semantic = 0
    consistency = 0
    samples: list[dict] = []
    for r in rows:
        s_errs = validate_row(r)
        semantic += len(s_errs)
        md = r.metadata
        c_errs = []
        if PRIM_FOR_OP[r.target.name] != md["primitive"]:
            c_errs.append("C_PRIMITIVE_MISMATCH")
        n = len(r.semantic_context.reachable_field_ids)
        b = md["reachable_bucket"]
        if md["specialization_mode"] == "reachable_fields":
            if not ((b in ("2", "3", "4", "5") and n == int(b)) or (b == "6_to_8" and 6 <= n <= 8)):
                c_errs.append("C_BUCKET_MISMATCH")
        elif b is not None or r.semantic_context.active_field_id is None:
            c_errs.append("C_ACTIVE_MODE_MISMATCH")
        consistency += len(c_errs)
        if (s_errs or c_errs) and len(samples) < 10:
            samples.append({"id": r.id, "errors": s_errs + c_errs})

    dumped = [r.model_dump() for r in rows]
    ids = [r.id for r in rows]
    dup_ids = len(ids) - len(set(ids))
    quota_checks = check_quotas([r.metadata for r in rows], spec)
    leaks = check_leakage(dumped)

    # benchmark files must equal the test partitions exactly
    bench_ok = True
    for part, n in spec["test_partitions"].items():
        lines = _read(out / "benchmark" / f"{part}.jsonl")
        expect = sorted(r.id for r in rows if r.metadata["split"] == "test" and r.metadata["test_partition"] == part)
        got = sorted(json.loads(l)["id"] for l in lines)
        if len(lines) != n or got != expect:
            bench_ok = False
            problems.append(f"benchmark/{part}.jsonl mismatch ({len(lines)} vs {n})")

    # manifest integrity
    manifest_ok = False
    mpath = out / "manifest.json"
    if mpath.exists():
        manifest = json.loads(mpath.read_text())
        manifest_ok = True
        for name, sha in manifest.get("files", {}).items():
            p = out / name
            if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != sha:
                manifest_ok = False
                problems.append(f"manifest hash mismatch: {name}")
    else:
        problems.append("manifest.json missing")

    failing = failed(quota_checks)
    constraints = spec["constraints"]
    passed = (
        not problems and not failing and len(rows) == spec["spec"]["total_rows"]
        and dup_ids == constraints["duplicate_ids"] and structural == constraints["structural_errors"]
        and semantic == constraints["semantic_errors"] and consistency == 0
        and len(leaks) == constraints["prohibited_leakage"] and bench_ok and manifest_ok
    )
    report = {
        "experiment_id": spec["spec"]["experiment_id"],
        "passed": passed,
        "total_rows": len(rows),
        "file_counts": file_counts,
        "duplicate_ids": dup_ids,
        "duplicate_content": duplicate_content_count(dumped),
        "structural_errors": structural,
        "semantic_errors": semantic,
        "consistency_errors": consistency,
        "prohibited_leakage": len(leaks),
        "leakage_samples": leaks[:10],
        "benchmark_files_ok": bench_ok,
        "manifest_ok": manifest_ok,
        "quota_tolerance": constraints["exact_quota_tolerance"],
        "quota_checks": quota_checks,
        "quota_checks_total": len(quota_checks),
        "quota_checks_failed": len(failing),
        "failed_quota_checks": failing,
        "problems": problems,
        "error_samples": samples,
        "distribution": {
            "test_partition": dict(Counter(r.metadata["test_partition"] for r in rows if r.metadata["split"] == "test")),
            "nc_kind": dict(Counter(r.metadata["nc_kind"] for r in rows if r.metadata["primitive"] == "no_change")),
            "source": dict(Counter(r.metadata["source"] for r in rows)),
        },
    }
    return report


def write_audit(report: dict, out_dir: str | Path) -> Path:
    p = Path(out_dir) / "audit.json"
    p.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return p
