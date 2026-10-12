"""Compile phase0 rows and the synthetic slice into json-calls-v1 SFT files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from form_sft.contracts.models import CanonicalRow
from form_sft.training.compiler import (
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_MODEL_ID,
    LENGTH_POLICY_REJECT,
    CompileError,
    _count_tokens,
    _iter_jsonl,
    _percentile,
    load_tokenizer,
)
from form_sft.training.json_examples import fixture_utterances, held_out_fixtures, synthetic_examples
from form_sft.training.json_prompt import (
    assistant_span,
    calls_from_row,
    context_from_row,
    dumps_calls,
    prompt_span,
    render_training_text,
    tool_line,
    user_block,
)


def _sft_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    metadata = metadata or {}
    return {
        "split": metadata.get("split"),
        "primitive": metadata.get("primitive"),
        "specialization_mode": metadata.get("specialization_mode"),
        "source": metadata.get("source"),
        "skill": metadata.get("skill") or "",
    }


def _compile_example(tokenizer, example: dict[str, Any], *, max_seq_length: int) -> dict[str, Any]:
    user = user_block(example["utterance"], example["context"], example["tools"])
    assistant = dumps_calls(example["target"]["function_calls"])
    text = render_training_text(tokenizer, user, assistant)
    if assistant_span(text) != assistant:
        raise CompileError(f"{example['id']}: rendered assistant JSON does not match the target")
    n_tok = _count_tokens(tokenizer, text)
    if n_tok > max_seq_length:
        raise CompileError(f"{example['id']}: token_length {n_tok} > max_seq_length {max_seq_length}")
    return {
        "id": example["id"],
        "text": text,
        "prompt": prompt_span(text),
        "target": example["target"],
        "token_length": n_tok,
        "truncated": False,
        "metadata": _sft_metadata(example.get("metadata")),
    }


def _compile_canonical(raw: dict, tokenizer, *, max_seq_length: int) -> dict[str, Any]:
    row = CanonicalRow.model_validate(raw)
    context = context_from_row(row)
    example = {
        "id": row.id,
        "utterance": row.utterance,
        "context": context,
        "tools": tool_line(context["fields"]),
        "target": {"function_calls": calls_from_row(row)},
        "metadata": _sft_metadata(
            {
                "split": row.metadata.get("split"),
                "primitive": row.metadata.get("primitive"),
                "specialization_mode": row.metadata.get("specialization_mode"),
                "source": row.metadata.get("source"),
            }
        ),
    }
    return _compile_example(tokenizer, example, max_seq_length=max_seq_length)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def build_json_corpus(
    corpus_dir: Path | str,
    out_dir: Path | str,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    length_policy: str = LENGTH_POLICY_REJECT,
    tokenizer=None,
) -> dict[str, Any]:
    if length_policy != LENGTH_POLICY_REJECT:
        raise CompileError("json-calls compile only supports length policy reject")
    corpus_dir = Path(corpus_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if tokenizer is None:
        tokenizer = load_tokenizer(model_id)

    banned = fixture_utterances()
    report: dict[str, Any] = {
        "model_id": model_id,
        "max_seq_length": max_seq_length,
        "format": "function_calls_json",
        "splits": {},
        "passed": True,
        "problems": [],
    }
    compiled_by_split: dict[str, list[dict]] = {split: [] for split in ("train", "validation", "test")}
    for split in compiled_by_split:
        src = corpus_dir / f"{split}.jsonl"
        if not src.exists():
            report["passed"] = False
            report["problems"].append(f"missing {src}")
            continue
        for raw in _iter_jsonl(src):
            if raw.get("utterance") in banned:
                raise CompileError(f"{raw.get('id')}: utterance matches a held-out fixture")
            compiled_by_split[split].append(_compile_canonical(raw, tokenizer, max_seq_length=max_seq_length))

    for split in ("train", "validation"):
        for example in synthetic_examples(split):
            compiled_by_split[split].append(_compile_example(tokenizer, example, max_seq_length=max_seq_length))

    fixtures = [_compile_example(tokenizer, example, max_seq_length=max_seq_length) for example in held_out_fixtures()]
    fixture_ids = {row["id"] for row in fixtures}
    train_ids = {row["id"] for row in compiled_by_split["train"]}
    if fixture_ids & train_ids:
        raise CompileError("held-out fixture ids leaked into train")

    for split, rows in compiled_by_split.items():
        path = out_dir / f"{split}.sft.jsonl"
        n = _write_jsonl(path, rows)
        lengths = sorted(row["token_length"] for row in rows)
        report["splits"][split] = {
            "compiled_rows": n,
            "output": str(path),
            "p50": _percentile(lengths, 50),
            "p95": _percentile(lengths, 95),
            "p99": _percentile(lengths, 99),
            "max": max(lengths) if lengths else 0,
            "synthetic_rows": sum(row["metadata"].get("source") == "synthetic_json" for row in rows),
        }
    fixture_path = out_dir / "fixtures.jsonl"
    report["fixtures"] = {"compiled_rows": _write_jsonl(fixture_path, fixtures), "output": str(fixture_path)}
    report["held_out_ids"] = sorted(fixture_ids)
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise CompileError("; ".join(report["problems"]) or "json compile failed")
    return report
