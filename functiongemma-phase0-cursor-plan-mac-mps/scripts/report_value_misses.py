"""Slice inexact eval rows and measure real FunctionGemma token lengths."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from form_sft.evaluation.set_choice_slice import slice_set_choice
from form_sft.evaluation.value_misses import classify_files
from form_sft.training.compiler import load_tokenizer
from form_sft.training.length_gate import _encode, percentile


def _calls(payload: dict | None) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    calls = payload.get("function_calls")
    return list(calls) if isinstance(calls, list) else []


def _bucket(row: dict) -> str | None:
    if row.get("exact_call"):
        return None
    gold = _calls(row.get("gold"))
    pred = row.get("pred")
    parse_ok = isinstance(pred, dict) and isinstance(pred.get("function_calls"), list)
    if not parse_ok:
        return "parse_failure"
    got = _calls(pred)
    if len(gold) == 1 and len(got) == 1:
        g_args = gold[0].get("arguments") or {}
        p_args = got[0].get("arguments") or {}
        if gold[0].get("name") != got[0].get("name") or g_args.get("field_id") != p_args.get("field_id"):
            return "wrong_field"
        return "right_field_wrong_value"
    return "wrong_field"


def slice_predictions(path: Path) -> dict:
    by_name: dict[str, Counter] = {}
    totals = Counter()
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        n += 1
        bucket = _bucket(row)
        if bucket is None:
            continue
        totals[bucket] += 1
        names = [call.get("name") or "empty" for call in _calls(row.get("gold"))] or ["empty"]
        for name in names:
            by_name.setdefault(name, Counter())
            by_name[name][bucket] += 1
    value_errors = totals["right_field_wrong_value"]
    enum_value_errors = by_name.get("set_choice", Counter())["right_field_wrong_value"]
    enum_majority = value_errors > 0 and enum_value_errors * 2 >= value_errors
    return {
        "rows": n,
        "inexact": dict(totals),
        "by_call": {name: dict(counts) for name, counts in sorted(by_name.items())},
        "enum_is_majority_of_value_errors": enum_majority,
        "alias_share": "option_alias_on_half_of_train_sets" if enum_majority or value_errors == 0 else "match_the_dominant_call",
    }


def length_report(paths: list[Path], tokenizer) -> dict:
    report = {}
    for path in paths:
        lengths = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            lengths.append(len(_encode(tokenizer, json.loads(line)["text"])))
        report[path.name] = {
            "rows": len(lengths),
            "p50": percentile(lengths, 0.50),
            "p95": percentile(lengths, 0.95),
            "p99": percentile(lengths, 0.99),
            "max": max(lengths) if lengths else 0,
            "over_1024": sum(1 for item in lengths if item > 1024),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None, help="frozen jsonl joined by id")
    parser.add_argument("--sft", type=Path, nargs="*", default=[])
    parser.add_argument("--model-id", default="google/functiongemma-270m-it")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload: dict = {"model_id": args.model_id}
    if args.predictions:
        if not args.predictions.exists():
            raise SystemExit(f"predictions file not found: {args.predictions}")
        payload["predictions"] = slice_predictions(args.predictions)
        payload["set_choice"] = slice_set_choice(args.predictions)
        if args.dataset:
            payload["value_misses"] = classify_files(args.predictions, args.dataset)
    else:
        payload["predictions"] = {
            "status": "absent",
            "alias_share": "option_alias_on_half_of_train_sets",
            "note": "v6 test enum 0.7973 against field 0.9192, so option aliases are the default drill.",
        }
    if args.sft:
        payload["lengths"] = length_report(args.sft, load_tokenizer(args.model_id))
        maxima = [item["max"] for item in payload["lengths"].values()]
        payload["max_seq_length"] = max(maxima) if maxima else 1024
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
