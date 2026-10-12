"""Schema-family-disjoint evaluation suites."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from form_sft.contracts.dataset_ir import DatasetRecordV2

SUITE_NAMES = (
    "schema_family_disjoint",
    "unseen_label",
    "opaque_id",
    "option_permutation",
    "counterfactual_context",
    "asr",
    "code_mixed",
    "correction",
    "no_change",
)


def load_records(path: Path) -> list[DatasetRecordV2]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(DatasetRecordV2.model_validate_json(line))
    return rows


def records_for_suite(records: list[DatasetRecordV2], suite: str) -> list[DatasetRecordV2]:
    return [record for record in records if suite in record.coverage.suites]


def write_suites(dataset_dir: str | Path, out_dir: str | Path | None = None) -> dict:
    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir) if out_dir else dataset_dir / "suites"
    out_dir.mkdir(parents=True, exist_ok=True)
    test_rows = load_records(dataset_dir / "test.jsonl")
    manifest = {"suites": {}, "dataset_dir": str(dataset_dir)}
    for name in SUITE_NAMES:
        chosen = records_for_suite(test_rows, name)
        path = out_dir / f"{name}.jsonl"
        payload = "".join(
            json.dumps(record.model_dump(), sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in chosen
        )
        path.write_text(payload)
        manifest["suites"][name] = {
            "path": str(path),
            "n": len(chosen),
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        }
    blob = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["sha256"] = hashlib.sha256(blob).hexdigest()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def suite_leak_report(dataset_dir: str | Path) -> dict:
    dataset_dir = Path(dataset_dir)
    rows = []
    for split in ("train", "validation", "test"):
        rows.extend(load_records(dataset_dir / f"{split}.jsonl"))
    families: dict[str, set[str]] = {}
    groups: dict[str, set[str]] = {}
    templates: dict[str, set[str]] = {}
    for record in rows:
        families.setdefault(record.schema.schema_family_id, set()).add(record.coverage.split)
        templates.setdefault(record.schema.template_lineage_id, set()).add(record.coverage.split)
        if record.counterfactual_group_id:
            groups.setdefault(record.counterfactual_group_id, set()).add(record.coverage.split)
    def crossed(table: dict[str, set[str]]) -> list[str]:
        return [key for key, splits in table.items() if len(splits) > 1]

    return {
        "schema_families": crossed(families),
        "template_lineages": crossed(templates),
        "counterfactual_groups": crossed(groups),
    }
