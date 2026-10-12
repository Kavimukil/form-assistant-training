"""Count-matched universal ablation ladder.

Cumulative rungs nest. Matched rungs draw the same number of training rows so
a later phenomenon is not credited for simply adding data.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from form_sft.contracts.dataset_ir import DatasetRecordV2

RUNGS = (
    ("U0", "legacy-like single-field control", ("single_set",)),
    ("U1", "schema and state diversity", ("single_set", "clear")),
    ("U2", "opaque ids and option permutations", ("single_set", "clear", "option_permutation")),
    ("U3", "counterfactuals and negatives", ("single_set", "clear", "option_permutation", "counterfactual", "no_change")),
    ("U4", "multi-field corrections and noise", ("single_set", "clear", "option_permutation", "counterfactual", "no_change", "multi_field", "correction", "asr", "code_mixed")),
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def select_ids(rows: list[dict], phenomena: tuple[str, ...]) -> list[str]:
    allowed = set(phenomena)
    return [row["id"] for row in rows if row["coverage"]["phenomenon"] in allowed]


def materialize_universal(
    dataset_dir: str | Path,
    out_root: str | Path,
    *,
    seed: int = 20261012,
) -> dict:
    dataset_dir = Path(dataset_dir)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    test_path = dataset_dir / "test.jsonl"
    frozen = _sha(test_path)
    train = _load(dataset_dir / "train.jsonl")
    validation = _load(dataset_dir / "validation.jsonl")
    cumulative_ids: list[set[str]] = []
    matched_n = None
    report = {"frozen_test_sha256": frozen, "frozen_test_path": str(test_path), "rungs": {}}
    for tag, title, phenomena in RUNGS:
        train_ids = select_ids(train, phenomena)
        val_ids = select_ids(validation, phenomena)
        if matched_n is None:
            matched_n = len(train_ids)
        if cumulative_ids and not set(cumulative_ids[-1]) <= set(train_ids):
            raise RuntimeError(f"{tag} is not a superset of the previous rung")
        cumulative_ids.append(set(train_ids))
        rng = random.Random(f"{seed}|{tag}")
        matched = train_ids if len(train_ids) <= matched_n else rng.sample(train_ids, matched_n)
        rung_dir = out_root / tag
        rung_dir.mkdir(parents=True, exist_ok=True)
        (rung_dir / "train_ids.json").write_text(json.dumps(train_ids, indent=2) + "\n")
        (rung_dir / "matched_train_ids.json").write_text(json.dumps(sorted(matched), indent=2) + "\n")
        manifest = {
            "tag": tag,
            "title": title,
            "phenomena": list(phenomena),
            "train_n": len(train_ids),
            "validation_n": len(val_ids),
            "matched_train_n": len(matched),
            "frozen_test_sha256": frozen,
            "nested_in_previous": True,
        }
        (rung_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        report["rungs"][tag] = manifest
    (out_root / "ladder_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def assert_record(row: dict) -> DatasetRecordV2:
    return DatasetRecordV2.model_validate(row)
