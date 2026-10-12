"""Reject a training row whose tools text or completion is cut off at max_seq_length."""
from __future__ import annotations

import json
from pathlib import Path


class LengthGateError(RuntimeError):
    pass


def _encode(tokenizer, text: str) -> list[int]:
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    return list(tok.encode(text, add_special_tokens=False))


def _decode(tokenizer, ids: list[int]) -> str:
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    return tok.decode(ids)


def percentile(lengths: list[int], fraction: float) -> int:
    if not lengths:
        return 0
    ordered = sorted(lengths)
    index = int((len(ordered) - 1) * fraction)
    return ordered[index]


def row_problems(text: str, tokenizer, max_length: int) -> list[str]:
    ids = _encode(tokenizer, text)
    if len(ids) <= max_length:
        return []
    kept = _decode(tokenizer, ids[:max_length])
    problems = []
    if "Tools:" not in kept:
        problems.append("tools_truncated")
    marker = "<start_of_turn>model\n"
    start = text.rfind(marker)
    completion = text[start + len(marker) :] if start >= 0 else ""
    if completion and completion not in kept:
        problems.append("completion_truncated")
    if not problems:
        problems.append("truncated")
    return problems


def audit_sft(path: str | Path, tokenizer, max_length: int) -> dict:
    lengths = []
    problems = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = row["text"]
        lengths.append(len(_encode(tokenizer, text)))
        found = row_problems(text, tokenizer, max_length)
        if found:
            problems.append({"id": row.get("id"), "problems": found, "tokens": lengths[-1]})
            if len(problems) >= 20:
                break
    return {
        "rows": len(lengths),
        "max_length": max_length,
        "p50": percentile(lengths, 0.50),
        "p95": percentile(lengths, 0.95),
        "p99": percentile(lengths, 0.99),
        "max": max(lengths) if lengths else 0,
        "over_max": sum(1 for item in lengths if item > max_length),
        "problems": problems,
        "passed": not problems and all(item <= max_length for item in lengths),
    }


def require_fit(path: str | Path, tokenizer, max_length: int) -> dict:
    report = audit_sft(path, tokenizer, max_length)
    if not report["passed"]:
        sample = report["problems"][:3]
        raise LengthGateError(
            f"{path} has {report['over_max']} rows over {max_length}; sample={sample}"
        )
    return report
