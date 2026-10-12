"""Phase-10 cumulative ablation ladder over the frozen Phase-0 corpus.

Rungs (each adds phenomena; frozen test set is never rewritten):

- A0: base model (no SFT)
- A1: clean active-field primitive extraction (deterministic templates)
- A2: + linguistic variation (curated_manual)
- A3: + negatives (no_change)
- A4: + corrections
- A5: + synthetic ASR
- A6: + teacher paraphrases
- A7: + real ASR seeds
- A8: + reachable-field training (= full train set)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

ABLATION_ORDER = ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8")

ABLATION_TITLES = {
    "A0": "base (no SFT)",
    "A1": "primitives (clean active-field, deterministic)",
    "A2": "linguistic variation (+ curated_manual)",
    "A3": "negatives (+ no_change)",
    "A4": "corrections",
    "A5": "synthetic ASR",
    "A6": "teacher paraphrases",
    "A7": "real ASR seeds",
    "A8": "reachable-field training (full)",
}


def rung_level(tag: str) -> int:
    if tag not in ABLATION_ORDER:
        raise ValueError(f"unknown ablation tag {tag!r}; expected one of {ABLATION_ORDER}")
    return ABLATION_ORDER.index(tag)


def include_row(metadata: dict[str, Any], level: int) -> bool:
    """Return True if a canonical row belongs in cumulative rung ``level`` (1..8)."""
    if level <= 0:
        return False
    if level >= 8:
        return True

    mode = metadata.get("specialization_mode")
    src = metadata.get("source")
    prim = metadata.get("primitive")
    corr = bool(metadata.get("correction"))
    asr = bool(metadata.get("asr_noise"))

    if mode != "active_field":
        return False

    allowed_sources = {"deterministic_template"}
    if level >= 2:
        allowed_sources.add("curated_manual")
    if level >= 6:
        allowed_sources.add("teacher_generated")
    if level >= 7:
        allowed_sources.add("realistic_asr_seed")
    if src not in allowed_sources:
        return False

    if prim == "no_change" and level < 3:
        return False
    if corr and level < 4:
        return False
    if asr and level < 5:
        return False
    return True


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_canonical_meta(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[row["id"]] = row["metadata"]
    return out


def _filter_sft(
    sft_path: Path,
    meta_by_id: dict[str, dict[str, Any]],
    level: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    kept: list[dict[str, Any]] = []
    ids: list[str] = []
    with sft_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            md = meta_by_id.get(row["id"])
            if md is None:
                raise KeyError(f"SFT id {row['id']} missing from canonical metadata")
            if include_row(md, level):
                kept.append(row)
                ids.append(row["id"])
    return kept, ids


def materialize_ladder(
    *,
    corpus_dir: str | Path = "generated/phase0-v1",
    out_root: str | Path = "artifacts/ablation",
) -> dict[str, Any]:
    """Write cumulative train/validation SFT subsets + manifests for A1–A8."""
    corpus_dir = Path(corpus_dir)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    test_path = corpus_dir / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(test_path)
    frozen_test_sha = _sha256_file(test_path)

    train_meta = _load_canonical_meta(corpus_dir / "train.jsonl")
    val_meta = _load_canonical_meta(corpus_dir / "validation.jsonl")
    train_sft = corpus_dir / "train.sft.jsonl"
    val_sft = corpus_dir / "validation.sft.jsonl"

    rungs: dict[str, Any] = {}
    prev_train_ids: set[str] | None = None
    for tag in ABLATION_ORDER:
        level = rung_level(tag)
        rung_dir = out_root / tag
        rung_dir.mkdir(parents=True, exist_ok=True)

        if tag == "A0":
            manifest = {
                "tag": tag,
                "title": ABLATION_TITLES[tag],
                "level": level,
                "train": None,
                "validation": None,
                "train_n": 0,
                "validation_n": 0,
                "nested_in_next": True,
                "frozen_test_sha256": frozen_test_sha,
                "frozen_test_path": str(test_path),
            }
            (rung_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            rungs[tag] = manifest
            continue

        train_rows, train_ids = _filter_sft(train_sft, train_meta, level)
        val_rows, val_ids = _filter_sft(val_sft, val_meta, level)
        train_out = rung_dir / "train.sft.jsonl"
        val_out = rung_dir / "validation.sft.jsonl"
        with train_out.open("w", encoding="utf-8") as f:
            for row in train_rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        with val_out.open("w", encoding="utf-8") as f:
            for row in val_rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

        train_id_set = set(train_ids)
        nested_ok = True if prev_train_ids is None else prev_train_ids <= train_id_set
        if not nested_ok:
            raise RuntimeError(f"ablation nesting violated at {tag}")
        prev_train_ids = train_id_set

        ids_path = rung_dir / "train_ids.json"
        ids_path.write_text(json.dumps(train_ids, indent=2) + "\n", encoding="utf-8")

        manifest = {
            "tag": tag,
            "title": ABLATION_TITLES[tag],
            "level": level,
            "train": str(train_out),
            "validation": str(val_out),
            "train_ids": str(ids_path),
            "train_n": len(train_rows),
            "validation_n": len(val_rows),
            "train_sha256": _sha256_file(train_out),
            "validation_sha256": _sha256_file(val_out),
            "nested_in_previous": nested_ok,
            "frozen_test_sha256": frozen_test_sha,
            "frozen_test_path": str(test_path),
        }
        (rung_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        rungs[tag] = manifest

    report = {
        "corpus_dir": str(corpus_dir),
        "out_root": str(out_root),
        "frozen_test_path": str(test_path),
        "frozen_test_sha256": frozen_test_sha,
        "frozen_test_n": sum(1 for line in test_path.open(encoding="utf-8") if line.strip()),
        "titles": ABLATION_TITLES,
        "rungs": {
            tag: {
                "train_n": rungs[tag]["train_n"],
                "validation_n": rungs[tag]["validation_n"],
                "title": rungs[tag]["title"],
            }
            for tag in ABLATION_ORDER
        },
        "note": "Evaluate every rung on the same frozen test.jsonl; never regenerate test.",
    }
    (out_root / "ladder_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def summarize_evals(
    eval_root: str | Path = "artifacts/eval",
    tags: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Collect overall metrics for ablation eval tags if present."""
    eval_root = Path(eval_root)
    tags = list(tags or [f"{t}-test" if t != "A0" else "A0-test" for t in ABLATION_ORDER])
    # Also accept plain A0 validation metrics as fallback key.
    out: dict[str, Any] = {}
    for tag in tags:
        path = eval_root / tag / f"{tag}_metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        overall = metrics.get("overall", {})
        out[tag] = {
            "exact_call": overall.get("exact_call"),
            "false_mutation_rate": overall.get("false_mutation_rate"),
            "op": overall.get("op"),
            "field": overall.get("field"),
            "n": overall.get("n"),
            "model_id": metrics.get("model_id"),
            "split_path": metrics.get("split_path"),
            "path": str(path),
        }
    return out
