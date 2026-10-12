"""G7/G9: greedy FunctionGemma evaluation with exact tool-call + sliced metrics."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm import tqdm

from form_sft.contracts.models import CanonicalRow
from form_sft.training.compiler import (
    DEVELOPER_PROMPT,
    DEFAULT_MODEL_ID,
    parse_tool_calls_from_rendered,
    targets_equal,
    tools_for_row,
)
from form_sft.training.json_prompt import parse_function_calls, score_function_calls
from form_sft.training.device import select_device


MUTATING_OPS = {
    "set_string",
    "set_integer",
    "set_number",
    "set_boolean",
    "set_enum",
    "set_date",
}


@dataclass
class PredRow:
    id: str
    gold: dict
    pred: dict | None
    raw: str
    exact_call: bool
    slices: dict[str, Any]


def prompt_messages(row: CanonicalRow) -> list[dict]:
    return [
        {"role": "developer", "content": DEVELOPER_PROMPT},
        {"role": "user", "content": row.utterance},
    ]


def render_prompt(tokenizer, row: CanonicalRow) -> str:
    mode = row.metadata.get("specialization_mode")
    tools = tools_for_row(row.semantic_context, mode)
    return tokenizer.apply_chat_template(
        prompt_messages(row),
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )


def _gold_dict(row: CanonicalRow) -> dict:
    return {"name": row.target.name, "arguments": dict(row.target.arguments)}


def score_prediction(gold: dict, pred: dict | None) -> dict[str, bool]:
    exact = bool(pred) and targets_equal(pred, gold)
    op_ok = bool(pred) and pred["name"] == gold["name"]
    g_args = gold.get("arguments") or {}
    p_args = (pred or {}).get("arguments") or {}
    field_ok = op_ok and (
        gold["name"] == "no_change"
        or g_args.get("field_id") == p_args.get("field_id")
    )
    value_ok = False
    enum_ok = False
    if exact:
        value_ok = enum_ok = True
    elif field_ok and gold["name"] == "set_enum":
        enum_ok = g_args.get("option_id") == p_args.get("option_id")
        value_ok = enum_ok
    elif field_ok and gold["name"] in MUTATING_OPS - {"set_enum"}:
        gv, pv = g_args.get("value"), p_args.get("value")
        if isinstance(gv, (int, float)) and isinstance(pv, (int, float)):
            value_ok = float(gv) == float(pv)
        else:
            value_ok = gv == pv
    nc_tp = gold["name"] == "no_change" and bool(pred) and pred["name"] == "no_change"
    nc_fp = gold["name"] != "no_change" and bool(pred) and pred["name"] == "no_change"
    nc_fn = gold["name"] == "no_change" and (not pred or pred["name"] != "no_change")
    # False mutation: gold says no_change but model emitted a mutating op
    false_mutation = gold["name"] == "no_change" and bool(pred) and pred["name"] in MUTATING_OPS
    return {
        "exact_call": exact,
        "op": op_ok,
        "field": field_ok,
        "value": value_ok if gold["name"] != "no_change" else nc_tp,
        "enum": enum_ok if gold["name"] == "set_enum" else (exact if gold["name"] != "set_enum" else False),
        "no_change_tp": nc_tp,
        "no_change_fp": nc_fp,
        "no_change_fn": nc_fn,
        "false_mutation": false_mutation,
        "parse_ok": pred is not None,
    }


def _safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d else 0.0


def aggregate_metrics(rows: list[PredRow]) -> dict[str, Any]:
    n = len(rows)
    flags = [r.slices["flags"] for r in rows]
    exact = sum(f["exact_call"] for f in flags)
    op = sum(f["op"] for f in flags)
    field = sum(f["field"] for f in flags)
    value_rows = [f for f, r in zip(flags, rows) if r.gold["name"] != "no_change"]
    enum_rows = [f for f, r in zip(flags, rows) if r.gold["name"] == "set_enum"]
    nc_gold = [f for f, r in zip(flags, rows) if r.gold["name"] == "no_change"]
    nc_pred = [f for f in flags if f["no_change_tp"] or f["no_change_fp"]]
    tp = sum(f["no_change_tp"] for f in flags)
    fp = sum(f["no_change_fp"] for f in flags)
    fn = sum(f["no_change_fn"] for f in flags)
    fm = sum(f["false_mutation"] for f in flags)
    return {
        "n": n,
        "exact_call": _safe_div(exact, n),
        "exact_call_count": exact,
        "op": _safe_div(op, n),
        "field": _safe_div(field, n),
        "value": _safe_div(sum(f["value"] for f in value_rows), len(value_rows)),
        "value_n": len(value_rows),
        "enum": _safe_div(sum(f["enum"] for f in enum_rows), len(enum_rows)),
        "enum_n": len(enum_rows),
        "no_change_precision": _safe_div(tp, tp + fp),
        "no_change_recall": _safe_div(tp, tp + fn),
        "no_change_f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "no_change_support": len(nc_gold),
        "false_mutation_rate": _safe_div(fm, len(nc_gold)),
        "false_mutation_count": fm,
        "parse_ok": _safe_div(sum(f["parse_ok"] for f in flags), n),
    }


def _slice_key(row: CanonicalRow, name: str) -> str:
    md = row.metadata
    if name == "primitive":
        return str(md.get("primitive"))
    if name == "difficulty":
        return str(md.get("difficulty"))
    if name == "mode":
        return str(md.get("specialization_mode"))
    if name == "noise":
        if md.get("correction") and md.get("asr_noise"):
            return "correction_and_asr"
        if md.get("correction"):
            return "correction_only"
        if md.get("asr_noise"):
            return "asr_only"
        return "clean"
    if name == "holdout":
        return str(md.get("test_partition") or md.get("split"))
    if name == "reachable_bucket":
        b = md.get("reachable_bucket")
        return str(b) if b is not None else "none"
    return "all"


def sliced_metrics(pred_rows: list[PredRow], canonical: dict[str, CanonicalRow]) -> dict[str, dict[str, Any]]:
    dims = ("primitive", "difficulty", "mode", "noise", "holdout", "reachable_bucket")
    out: dict[str, dict[str, Any]] = {}
    for dim in dims:
        buckets: dict[str, list[PredRow]] = defaultdict(list)
        for pr in pred_rows:
            buckets[_slice_key(canonical[pr.id], dim)].append(pr)
        out[dim] = {k: aggregate_metrics(v) for k, v in sorted(buckets.items())}
    return out


def _predrows_from_saved(saved: Iterable[dict], canonical: dict[str, CanonicalRow]) -> list[PredRow]:
    out: list[PredRow] = []
    for p in saved:
        rid = p["id"]
        row = canonical[rid]
        flags = p.get("flags") or score_prediction(p["gold"], p.get("pred"))
        out.append(
            PredRow(
                id=rid,
                gold=p["gold"],
                pred=p.get("pred"),
                raw=p.get("raw", ""),
                exact_call=bool(p.get("exact_call", flags.get("exact_call"))),
                slices={
                    "flags": flags,
                    "primitive": row.metadata.get("primitive"),
                    "difficulty": row.metadata.get("difficulty"),
                    "mode": row.metadata.get("specialization_mode"),
                    "noise": _slice_key(row, "noise"),
                    "holdout": _slice_key(row, "holdout"),
                    "reachable_bucket": _slice_key(row, "reachable_bucket"),
                },
            )
        )
    return out


def evaluate_go_no_go(
    pred_rows: list[PredRow],
    canonical: dict[str, CanonicalRow],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Apply frozen YAML go_no_go thresholds to test predictions.

    Returns a report with per-threshold checks. ``stretch`` FMR is informational only.
    """
    by_id = {r.id: r for r in pred_rows}

    def subset(pred: Any) -> list[PredRow]:
        return [by_id[i] for i, row in canonical.items() if i in by_id and pred(row)]

    checks: list[dict[str, Any]] = []

    def add(name: str, actual: float, threshold: float, *, higher_is_better: bool, n: int) -> None:
        ok = (actual >= threshold) if higher_is_better else (actual <= threshold)
        checks.append(
            {
                "name": name,
                "ok": bool(ok),
                "actual": actual,
                "threshold": threshold,
                "n": n,
                "higher_is_better": higher_is_better,
            }
        )

    overall = aggregate_metrics(pred_rows)
    safety = thresholds.get("safety") or {}
    if "false_mutation_rate_max" in safety:
        add(
            "safety.false_mutation_rate",
            overall["false_mutation_rate"],
            float(safety["false_mutation_rate_max"]),
            higher_is_better=False,
            n=overall["no_change_support"],
        )

    af_thr = thresholds.get("active_field") or {}
    af_rows = subset(lambda r: r.metadata.get("specialization_mode") == "active_field")
    af_m = aggregate_metrics(af_rows) if af_rows else aggregate_metrics([])

    if "primitive_extraction" in af_thr:
        mut = [p for p in af_rows if canonical[p.id].metadata.get("primitive") != "no_change"]
        m = aggregate_metrics(mut) if mut else {"exact_call": 0.0, "n": 0}
        add("active_field.primitive_extraction", m["exact_call"], float(af_thr["primitive_extraction"]),
            higher_is_better=True, n=m["n"])

    for prim in ("integer", "number", "boolean", "enum"):
        if prim not in af_thr:
            continue
        rows = [p for p in af_rows if canonical[p.id].metadata.get("primitive") == prim]
        m = aggregate_metrics(rows) if rows else {"exact_call": 0.0, "n": 0}
        add(f"active_field.{prim}", m["exact_call"], float(af_thr[prim]), higher_is_better=True, n=m["n"])

    if "correction" in af_thr:
        rows = [p for p in af_rows if canonical[p.id].metadata.get("correction")]
        m = aggregate_metrics(rows) if rows else {"exact_call": 0.0, "n": 0}
        add("active_field.correction", m["exact_call"], float(af_thr["correction"]),
            higher_is_better=True, n=m["n"])

    if "synthetic_asr" in af_thr:
        rows = [p for p in af_rows if canonical[p.id].metadata.get("asr_noise")]
        m = aggregate_metrics(rows) if rows else {"exact_call": 0.0, "n": 0}
        add("active_field.synthetic_asr", m["exact_call"], float(af_thr["synthetic_asr"]),
            higher_is_better=True, n=m["n"])

    if "no_change_precision" in af_thr:
        add("active_field.no_change_precision", af_m["no_change_precision"],
            float(af_thr["no_change_precision"]), higher_is_better=True, n=af_m["no_change_support"])
    if "no_change_recall" in af_thr:
        add("active_field.no_change_recall", af_m["no_change_recall"],
            float(af_thr["no_change_recall"]), higher_is_better=True, n=af_m["no_change_support"])

    rf_thr = thresholds.get("reachable_fields") or {}
    bucket_map = {"candidates_2": "2", "candidates_3": "3", "candidates_5": "5"}
    for key, bucket in bucket_map.items():
        if key not in rf_thr:
            continue
        rows = subset(
            lambda r, b=bucket: r.metadata.get("specialization_mode") == "reachable_fields"
            and str(r.metadata.get("reachable_bucket")) == b
        )
        m = aggregate_metrics(rows) if rows else {"exact_call": 0.0, "n": 0}
        add(f"reachable_fields.{key}", m["exact_call"], float(rf_thr[key]),
            higher_is_better=True, n=m["n"])

    stretch = safety.get("stretch")
    stretch_info = None
    if stretch is not None:
        stretch_info = {
            "actual": overall["false_mutation_rate"],
            "threshold": float(stretch),
            "met": overall["false_mutation_rate"] <= float(stretch),
            "n": overall["no_change_support"],
        }

    hard = [c for c in checks]
    return {
        "go": all(c["ok"] for c in hard),
        "checks": checks,
        "overall": overall,
        "active_field": af_m,
        "fmr_stretch": stretch_info,
    }


def load_canonical_jsonl(path: Path) -> list[CanonicalRow]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(CanonicalRow.model_validate(json.loads(line)))
    return rows


@torch.inference_mode()
def generate_greedy(
    model,
    tokenizer,
    prompts: list[str],
    *,
    device: str,
    max_new_tokens: int = 128,
) -> list[str]:
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # Left padding keeps prompt as a contiguous suffix before generation for decoder-only models.
    prev_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "left"
    try:
        enc = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        decoded = []
        for i, seq in enumerate(out):
            plen = int(prompt_lens[i])
            # With left padding, non-pad prompt tokens occupy the right side of the prompt region.
            # generate() returns prompt+new; cut by original prompt token count from the right of the padded prompt.
            input_len = enc["input_ids"].shape[1]
            gen_ids = seq[input_len:]
            decoded.append(tok.decode(gen_ids, skip_special_tokens=False))
            _ = plen
        return decoded
    finally:
        tok.padding_side = prev_side


def evaluate_rows(
    rows: list[CanonicalRow],
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "auto",
    batch_size: int = 4,
    max_new_tokens: int = 128,
    limit: int | None = None,
    show_progress: bool = True,
) -> tuple[list[PredRow], dict[str, Any]]:
    plan = select_device(device)
    from transformers import AutoModelForCausalLM, AutoProcessor

    tokenizer = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=plan.torch_dtype)
    model.to(plan.device)
    model.eval()

    use_rows = rows[:limit] if limit else rows
    prompts = [render_prompt(tokenizer, r) for r in use_rows]
    preds: list[PredRow] = []
    iterator = range(0, len(use_rows), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=f"eval[{plan.device}]", unit="batch")

    for start in iterator:
        h = use_rows[start : start + batch_size]
        batch_prompts = prompts[start : start + batch_size]
        gens = generate_greedy(
            model, tokenizer, batch_prompts, device=plan.device, max_new_tokens=max_new_tokens
        )
        for row, raw in zip(h, gens):
            parsed_list = parse_tool_calls_from_rendered(raw)
            pred = parsed_list[0] if parsed_list else None
            gold = _gold_dict(row)
            flags = score_prediction(gold, pred)
            preds.append(
                PredRow(
                    id=row.id,
                    gold=gold,
                    pred=pred,
                    raw=raw,
                    exact_call=flags["exact_call"],
                    slices={
                        "flags": flags,
                        "primitive": row.metadata.get("primitive"),
                        "difficulty": row.metadata.get("difficulty"),
                        "mode": row.metadata.get("specialization_mode"),
                        "noise": _slice_key(row, "noise"),
                        "holdout": _slice_key(row, "holdout"),
                        "reachable_bucket": _slice_key(row, "reachable_bucket"),
                    },
                )
            )

    by_id = {r.id: r for r in use_rows}
    metrics = {
        "overall": aggregate_metrics(preds),
        "by": sliced_metrics(preds, by_id),
        "model_id": model_id,
        "device": plan.device,
        "decoding": "greedy",
        "n": len(preds),
    }
    return preds, metrics


def evaluate(
    split_path: str | Path,
    *,
    out_dir: str | Path,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "auto",
    batch_size: int = 4,
    max_new_tokens: int = 128,
    limit: int | None = None,
    tag: str = "A0",
) -> dict[str, Any]:
    """Run baseline/eval on a canonical JSONL split; persist predictions + metrics."""
    split_path = Path(split_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_canonical_jsonl(split_path)
    preds, metrics = evaluate_rows(
        rows,
        model_id=model_id,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        limit=limit,
    )
    pred_path = out_dir / f"{tag}_predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(
                json.dumps(
                    {
                        "id": p.id,
                        "gold": p.gold,
                        "pred": p.pred,
                        "raw": p.raw,
                        "exact_call": p.exact_call,
                        "slices": {k: v for k, v in p.slices.items() if k != "flags"},
                        "flags": p.slices["flags"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    metrics_path = out_dir / f"{tag}_metrics.json"
    payload = {
        **metrics,
        "tag": tag,
        "split_path": str(split_path),
        "predictions_path": str(pred_path),
    }
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _empty_calls(gold: dict) -> bool:
    return not (gold.get("function_calls") or [])


def _has_choice(gold: dict) -> bool:
    return any(call.get("name") == "set_choice" for call in (gold.get("function_calls") or []))


def aggregate_json_metrics(rows: list[PredRow]) -> dict[str, Any]:
    """Aggregate score_function_calls flags. Empty function_calls is the no-change class."""
    n = len(rows)
    flags = [r.slices["flags"] for r in rows]
    value_rows = [f for f, r in zip(flags, rows) if not _empty_calls(r.gold)]
    enum_rows = [f for f, r in zip(flags, rows) if _has_choice(r.gold)]
    nc_gold = [f for f, r in zip(flags, rows) if _empty_calls(r.gold)]
    tp = sum(f["no_change_tp"] for f in flags)
    fp = sum(f["no_change_fp"] for f in flags)
    fn = sum(f["no_change_fn"] for f in flags)
    fm = sum(f["false_mutation"] for f in flags)
    return {
        "n": n,
        "exact_call": _safe_div(sum(f["exact_call"] for f in flags), n),
        "exact_call_count": sum(f["exact_call"] for f in flags),
        "op": _safe_div(sum(f["op"] for f in flags), n),
        "field": _safe_div(sum(f["field"] for f in flags), n),
        "value": _safe_div(sum(f["value"] for f in value_rows), len(value_rows)),
        "value_n": len(value_rows),
        "enum": _safe_div(sum(f["enum"] for f in enum_rows), len(enum_rows)),
        "enum_n": len(enum_rows),
        "no_change_precision": _safe_div(tp, tp + fp),
        "no_change_recall": _safe_div(tp, tp + fn),
        "no_change_f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "no_change_support": len(nc_gold),
        "false_mutation_rate": _safe_div(fm, len(nc_gold)),
        "false_mutation_count": fm,
        "parse_ok": _safe_div(sum(f["parse_ok"] for f in flags), n),
    }


def load_json_sft(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def evaluate_json(
    split_path: str | Path,
    *,
    out_dir: str | Path,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "auto",
    batch_size: int = 4,
    max_new_tokens: int = 256,
    limit: int | None = None,
    tag: str = "json-calls",
) -> dict[str, Any]:
    """Greedy eval of prompts whose gold target is {"function_calls": [...]}."""
    split_path = Path(split_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_json_sft(split_path)
    if limit:
        rows = rows[:limit]
    plan = select_device(device)
    from transformers import AutoModelForCausalLM, AutoProcessor

    tokenizer = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=plan.torch_dtype)
    model.to(plan.device)
    model.eval()

    prompts = [row["prompt"] for row in rows]
    preds: list[PredRow] = []
    iterator = range(0, len(rows), batch_size)
    iterator = tqdm(iterator, desc=f"eval-json[{plan.device}]", unit="batch")
    for start in iterator:
        batch = rows[start : start + batch_size]
        gens = generate_greedy(
            model,
            tokenizer,
            prompts[start : start + batch_size],
            device=plan.device,
            max_new_tokens=max_new_tokens,
        )
        for row, raw in zip(batch, gens):
            pred = parse_function_calls(raw)
            flags = score_function_calls(row["target"], pred)
            preds.append(
                PredRow(
                    id=row["id"],
                    gold=row["target"],
                    pred=pred,
                    raw=raw,
                    exact_call=flags["exact_call"],
                    slices={"flags": flags, **(row.get("metadata") or {})},
                )
            )

    metrics = {
        "overall": aggregate_json_metrics(preds),
        "model_id": model_id,
        "device": plan.device,
        "decoding": "greedy",
        "format": "function_calls_json",
        "n": len(preds),
    }
    pred_path = out_dir / f"{tag}_predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as handle:
        for pred in preds:
            handle.write(
                json.dumps(
                    {
                        "id": pred.id,
                        "gold": pred.gold,
                        "pred": pred.pred,
                        "raw": pred.raw,
                        "exact_call": pred.exact_call,
                        "flags": pred.slices["flags"],
                        "metadata": {k: v for k, v in pred.slices.items() if k != "flags"},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    payload = {**metrics, "tag": tag, "split_path": str(split_path), "predictions_path": str(pred_path)}
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
