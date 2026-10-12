"""Action-set metrics for universal form filling."""
from __future__ import annotations

import random
from collections import Counter
from typing import Any


def _norm(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return tuple(_norm(item) for item in value)
    return value


def action_key(action: dict) -> tuple:
    return (action.get("kind"), action.get("field_id"), _norm(action.get("value")))


def score_actions(gold: list[dict], pred: list[dict] | None) -> dict[str, Any]:
    parse_ok = pred is not None
    predicted = pred or []
    gold_keys = Counter(action_key(action) for action in gold)
    pred_keys = Counter(action_key(action) for action in predicted) if parse_ok else Counter()
    exact = parse_ok and gold_keys == pred_keys
    gold_mut = [action for action in gold if action.get("kind") != "no_change"]
    pred_mut = [action for action in predicted if action.get("kind") != "no_change"] if parse_ok else []
    gold_fields = Counter(action.get("field_id") for action in gold_mut)
    pred_fields = Counter(action.get("field_id") for action in pred_mut)
    field_hits = sum((gold_fields & pred_fields).values())
    field_den = sum(gold_fields.values())
    value_hits = sum((gold_keys & pred_keys).values()) if gold_mut else 0
    if not gold_mut:
        field_acc = 1.0 if parse_ok and not pred_mut else 0.0
        value_acc = field_acc
    else:
        field_acc = field_hits / field_den
        value_acc = value_hits / len(gold_mut)
    gold_abstain = not gold_mut
    pred_abstain = parse_ok and not pred_mut
    return {
        "exact_action_set": exact,
        "field": field_acc == 1.0 if gold_mut else field_acc == 1.0,
        "field_recall": field_acc,
        "value": value_acc == 1.0 if gold_mut else value_acc == 1.0,
        "value_recall": value_acc,
        "false_mutation": gold_abstain and bool(pred_mut),
        "gold_abstain": gold_abstain,
        "pred_abstain": pred_abstain,
        "abstention_tp": gold_abstain and pred_abstain,
        "abstention_fp": (not gold_abstain) and pred_abstain,
        "abstention_fn": gold_abstain and not pred_abstain,
        "parse_ok": parse_ok,
        "correction": exact,
    }


def _div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def aggregate_action_metrics(scored: list[dict], *, correction_mask: list[bool] | None = None) -> dict[str, Any]:
    n = len(scored)
    gold_mut = [row for row in scored if not row["gold_abstain"]]
    gold_abs = [row for row in scored if row["gold_abstain"]]
    tp = sum(row["abstention_tp"] for row in scored)
    fp = sum(row["abstention_fp"] for row in scored)
    fn = sum(row["abstention_fn"] for row in scored)
    correction_rows = []
    if correction_mask is not None:
        correction_rows = [row for row, flag in zip(scored, correction_mask) if flag]
    return {
        "n": n,
        "exact_action_set": _div(sum(row["exact_action_set"] for row in scored), n),
        "field": _div(sum(row["field_recall"] for row in gold_mut), len(gold_mut)),
        "value": _div(sum(row["value_recall"] for row in gold_mut), len(gold_mut)),
        "correction": _div(sum(row["exact_action_set"] for row in correction_rows), len(correction_rows)),
        "correction_n": len(correction_rows),
        "false_mutation_rate": _div(sum(row["false_mutation"] for row in scored), len(gold_abs)),
        "false_mutation_count": sum(row["false_mutation"] for row in scored),
        "abstention_precision": _div(tp, tp + fp),
        "abstention_recall": _div(tp, tp + fn),
        "abstention_support": len(gold_abs),
        "parse_ok": _div(sum(row["parse_ok"] for row in scored), n),
    }


def paired_bootstrap_delta(
    baseline: list[bool],
    candidate: list[bool],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired bootstrap requires equal non-empty series")
    rng = random.Random(seed)
    n = len(baseline)
    deltas = []
    for _ in range(n_resamples):
        indexes = [rng.randrange(n) for _ in range(n)]
        base = sum(baseline[index] for index in indexes) / n
        cand = sum(candidate[index] for index in indexes) / n
        deltas.append(cand - base)
    deltas.sort()
    low_at = int(0.025 * (n_resamples - 1))
    high_at = int(0.975 * (n_resamples - 1))
    return {
        "mean": sum(deltas) / len(deltas),
        "low": deltas[low_at],
        "high": deltas[high_at],
        "n": float(n),
    }


def wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    z2 = z * z
    denom = 1 + z2 / total
    centre = p + z2 / (2 * total)
    margin = z * ((p * (1 - p) / total + z2 / (4 * total * total)) ** 0.5)
    return (centre - margin) / denom
