"""Classify inexact set_choice and set_string calls against the frozen schema."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def _calls(payload: dict | None) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    calls = payload.get("function_calls")
    return list(calls) if isinstance(calls, list) else []


def _parsed(pred: dict | None) -> bool:
    return isinstance(pred, dict) and isinstance(pred.get("function_calls"), list)


def _fields_by_id(record: dict) -> dict[str, dict]:
    return {field["field_id"]: field for field in record["schema"]["fields"]}


def _syllables(record: dict) -> set[str]:
    found = set()
    for field in record["schema"]["fields"]:
        found.add(field.get("label"))
        found.update(field.get("aliases") or [])
        for option in field.get("options") or []:
            found.add(option.get("label"))
            found.update(option.get("aliases") or [])
            found.add(option.get("id"))
    return {item for item in found if isinstance(item, str)}


def _choice_class(pred_value, field: dict, gold_value) -> str:
    option = next((item for item in field.get("options") or [] if item.get("id") == gold_value), None)
    if option is not None and pred_value == option.get("label"):
        return "spoken_label"
    if option is not None and pred_value in (option.get("aliases") or []):
        return "option_alias"
    other_ids = {item.get("id") for item in field.get("options") or [] if item.get("id") != gold_value}
    if pred_value in other_ids:
        return "other_option_id"
    if isinstance(pred_value, str) and pred_value.startswith("opt"):
        return "malformed_id"
    return "other"


def _string_class(pred_value, field_id: str, record: dict, gold_value) -> str:
    values = (record.get("state") or {}).get("values") or {}
    current = values.get(field_id)
    if pred_value == current and pred_value != gold_value:
        return "current_state"
    others = [value for fid, value in values.items() if fid != field_id]
    if pred_value in others:
        return "other_field_value"
    if (
        isinstance(pred_value, str)
        and isinstance(gold_value, str)
        and pred_value
        and pred_value != gold_value
        and (pred_value in gold_value or gold_value in pred_value)
    ):
        return "truncated_substring"
    if pred_value in _syllables(record):
        return "other_syllable"
    return "other"


def classify_joined(prediction: dict, record: dict) -> list[dict]:
    """One entry per inexact set_choice or set_string gold call."""
    gold_calls = _calls(prediction.get("gold"))
    pred = prediction.get("pred")
    parsed = _parsed(pred)
    pred_calls = _calls(pred) if parsed else []
    fields = _fields_by_id(record)
    misses = []
    for call in gold_calls:
        name = call.get("name")
        if name not in {"set_choice", "set_string"}:
            continue
        args = call.get("arguments") or {}
        field_id = args.get("field_id")
        gold_value = args.get("value")
        field = fields.get(field_id) or {}
        if not parsed:
            kind = "parse_failure"
        else:
            matched = [
                item
                for item in pred_calls
                if item.get("name") == name and (item.get("arguments") or {}).get("field_id") == field_id
            ]
            if not matched:
                kind = "wrong_field"
            else:
                pred_value = (matched[0].get("arguments") or {}).get("value")
                if pred_value == gold_value:
                    continue
                if name == "set_choice":
                    kind = _choice_class(pred_value, field, gold_value)
                else:
                    kind = _string_class(pred_value, field_id, record, gold_value)
        misses.append({"id": prediction.get("id"), "name": name, "field_id": field_id, "class": kind})
    return misses


def _majority_ok(counts: Counter, wanted: set[str]) -> bool:
    total = sum(counts.values())
    if total == 0:
        return False
    return sum(counts[name] for name in wanted) * 2 > total


def classify_files(predictions_path: str | Path, dataset_path: str | Path) -> dict:
    records = {}
    for line in Path(dataset_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            records[row["id"]] = row
    choice = Counter()
    string = Counter()
    missing_ids = 0
    for line in Path(predictions_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        prediction = json.loads(line)
        record = records.get(prediction.get("id"))
        if record is None:
            missing_ids += 1
            continue
        for miss in classify_joined(prediction, record):
            bucket = choice if miss["name"] == "set_choice" else string
            bucket[miss["class"]] += 1
    choice_ok = _majority_ok(choice, {"spoken_label", "option_alias"})
    string_ok = _majority_ok(string, {"current_state", "other_field_value", "other_syllable"})
    return {
        "set_choice": dict(choice),
        "set_string": dict(string),
        "missing_ids": missing_ids,
        "choice_majority_is_label_or_alias": choice_ok,
        "string_majority_is_other_syllable": string_ok,
        "proceed": choice_ok and string_ok,
    }


def _example(prediction: dict, miss: dict) -> dict:
    gold_value = None
    pred_value = None
    for call in _calls(prediction.get("gold")):
        args = call.get("arguments") or {}
        if call.get("name") == miss["name"] and args.get("field_id") == miss["field_id"]:
            gold_value = args.get("value")
    for call in _calls(prediction.get("pred")):
        args = call.get("arguments") or {}
        if call.get("name") == miss["name"] and args.get("field_id") == miss["field_id"]:
            pred_value = args.get("value")
    return {
        "id": miss["id"],
        "name": miss["name"],
        "field_id": miss["field_id"],
        "class": miss["class"],
        "gold_value": gold_value,
        "pred_value": pred_value,
    }


def _classify_path(path: Path, records: dict[str, dict], *, example_limit: int = 8) -> dict:
    choice: Counter = Counter()
    string: Counter = Counter()
    examples: dict[str, list] = {}
    missing_ids = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        prediction = json.loads(line)
        record = records.get(prediction.get("id"))
        if record is None:
            missing_ids += 1
            continue
        for miss in classify_joined(prediction, record):
            bucket = choice if miss["name"] == "set_choice" else string
            bucket[miss["class"]] += 1
            key = f"{miss['name']}:{miss['class']}"
            examples.setdefault(key, [])
            if len(examples[key]) < example_limit:
                examples[key].append(_example(prediction, miss))
    return {
        "set_choice": dict(choice),
        "set_string": dict(string),
        "missing_ids": missing_ids,
        "examples": examples,
    }


def compare_predictions(v7_path: str | Path, v8_path: str | Path, dataset_path: str | Path) -> dict:
    """Compare per-call misses. Examples contain values, never utterances."""
    v7_path, v8_path, dataset_path = Path(v7_path), Path(v8_path), Path(dataset_path)
    missing = [str(path) for path in (v7_path, v8_path, dataset_path) if not path.exists()]
    if missing:
        return {"predictions_present": False, "missing": missing, "proceed": False}
    records = {}
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            records[row["id"]] = row
    v7 = _classify_path(v7_path, records)
    v8 = _classify_path(v8_path, records)
    choice = Counter(v7["set_choice"])
    value_misses = sum(
        count for name, count in choice.items() if name not in {"wrong_field", "parse_failure"}
    )
    label_or_alias = choice["spoken_label"] + choice["option_alias"]
    use_alias = value_misses > 0 and label_or_alias * 2 > value_misses
    return {
        "predictions_present": True,
        "v7": v7,
        "v8": v8,
        "use_option_alias": use_alias,
        "proceed": True,
    }
