"""Group inexact set_choice predictions into field, option-id, and parse failures."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def _calls(payload: dict | None) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    calls = payload.get("function_calls")
    return list(calls) if isinstance(calls, list) else []


def _choice_calls(payload: dict | None) -> list[dict]:
    return [call for call in _calls(payload) if call.get("name") == "set_choice"]


def _parsed(pred: dict | None) -> bool:
    return isinstance(pred, dict) and isinstance(pred.get("function_calls"), list)


def set_choice_bucket(row: dict) -> str | None:
    """Bucket one inexact row whose gold contains set_choice.

    Exact rows and rows with no set_choice call return None. A row whose
    set_choice calls match, but some other call does not, is other_inexact.
    """
    gold_choices = _choice_calls(row.get("gold"))
    if not gold_choices or row.get("exact_call"):
        return None
    pred = row.get("pred")
    if not _parsed(pred):
        return "parse_failure"
    pred_by_field: dict[str, list] = {}
    for call in _choice_calls(pred):
        args = call.get("arguments") or {}
        pred_by_field.setdefault(args.get("field_id"), []).append(args.get("value"))
    wrong_field = False
    wrong_option = False
    for call in gold_choices:
        args = call.get("arguments") or {}
        field_id = args.get("field_id")
        if field_id not in pred_by_field:
            wrong_field = True
        elif args.get("value") not in pred_by_field[field_id]:
            wrong_option = True
    if wrong_field:
        return "wrong_field"
    if wrong_option:
        return "right_field_wrong_option_id"
    return "other_inexact"


def _option_miss(row: dict) -> dict:
    gold = []
    pred = []
    for call in _choice_calls(row.get("gold")):
        args = call.get("arguments") or {}
        gold.append({"field_id": args.get("field_id"), "option_id": args.get("value")})
    for call in _choice_calls(row.get("pred")):
        args = call.get("arguments") or {}
        pred.append({"field_id": args.get("field_id"), "option_id": args.get("value")})
    return {"id": row.get("id"), "gold": gold, "pred": pred}


def slice_set_choice(path: str | Path, *, example_limit: int = 8) -> dict:
    counts = Counter()
    examples: dict[str, list] = {
        "wrong_field": [],
        "right_field_wrong_option_id": [],
        "parse_failure": [],
    }
    set_choice_rows = 0
    exact = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not _choice_calls(row.get("gold")):
            continue
        set_choice_rows += 1
        bucket = set_choice_bucket(row)
        if bucket is None:
            exact += 1
            continue
        counts[bucket] += 1
        if bucket in examples and len(examples[bucket]) < example_limit:
            examples[bucket].append(_option_miss(row))
    inexact = sum(counts.values())
    return {
        "file": str(path),
        "set_choice_rows": set_choice_rows,
        "exact": exact,
        "inexact": inexact,
        "wrong_field": counts["wrong_field"],
        "right_field_wrong_option_id": counts["right_field_wrong_option_id"],
        "parse_failure": counts["parse_failure"],
        "other_inexact": counts["other_inexact"],
        "examples": examples,
    }
