"""Production prompt: instruction + utterance + context + tool signatures.

The supervised target is one JSON object, {"function_calls":[...]}.
This path does not pass tools into the chat template, so FunctionGemma's
function-declaration tokens are never injected.

DatasetRecordV2 serialization enters through ``training.serializer``. The
helpers below stay the production prompt implementation.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

from form_sft.contracts.models import CanonicalRow, SemanticField
from form_sft.specialization.runtime import SpecializationMode, select_fields

DEVELOPER_INSTRUCTION = (
    "You extract form field values from English user utterances.\n"
    "Use only the listed tools. Emit exactly one JSON object and nothing else.\n"
    'Schema: {"function_calls":[{"name":"<tool>","arguments":{...}}]}\n'
    'If nothing should change, emit {"function_calls":[]}.\n'
    "Do not invent fields, options, or prose. Do not wrap tools in markdown.\n"
    "Form context: <below>\n"
    "Tools: <below>"
)

_MODEL_TURN = "<start_of_turn>model\n"
_END_TURN = "<end_of_turn>"


def dumps_calls(calls: list[dict[str, Any]]) -> str:
    return json.dumps({"function_calls": calls}, ensure_ascii=False, separators=(",", ":"))


def set_call(name: str, field_id: str, value: Any) -> dict[str, Any]:
    return {"name": name, "arguments": {"field_id": field_id, "value": value}}


def clear_call(field_id: str) -> dict[str, Any]:
    return {"name": "clear_field", "arguments": {"field_id": field_id}}


def _option_ids(field: dict[str, Any]) -> list[str]:
    return [str(opt["id"]) for opt in field.get("options") or []]


def tool_signature(field: dict[str, Any]) -> str:
    fid = field["id"]
    kind = field["type"]
    if kind == "integer":
        setter = f"set_integer({fid}: integer)"
    elif kind == "number":
        setter = f"set_number({fid}: number)"
    elif kind == "boolean":
        setter = f"set_boolean({fid}: boolean)"
    elif kind == "string":
        setter = f"set_string({fid}: string)"
    elif kind == "date":
        setter = f"set_date({fid}: date YYYY-MM-DD)"
    elif kind == "enum":
        setter = f"set_choice({fid} enum {'|'.join(_option_ids(field))})"
    elif kind == "multi":
        setter = f"set_multi_choice({fid}: unique array of {'|'.join(_option_ids(field))})"
    else:
        raise ValueError(f"unsupported field type {kind!r}")
    return f"{setter}, clear_field({fid})"


def tool_line(fields: list[dict[str, Any]]) -> str:
    return ", ".join(tool_signature(field) for field in fields)


def context_json(context: dict[str, Any]) -> str:
    return json.dumps(context, ensure_ascii=False, separators=(",", ":"))


def user_block(utterance: str, context: dict[str, Any], tools: str) -> str:
    return f"User: {utterance.strip()}\nContext: {context_json(context)}\nTools: {tools}"


def field_to_context(field: SemanticField) -> dict[str, Any]:
    item: dict[str, Any] = {"id": field.field_id, "type": field.type, "label": field.label}
    if field.type == "enum" and field.options:
        options = []
        for opt in field.options:
            rendered: dict[str, Any] = {"id": opt.id, "label": opt.label}
            if opt.aliases:
                rendered["aliases"] = list(opt.aliases)
            options.append(rendered)
        item["options"] = options
    constraints: dict[str, Any] = {}
    if field.minimum is not None:
        constraints["min"] = field.minimum
    if field.maximum is not None:
        constraints["max"] = field.maximum
    if field.min_length is not None:
        constraints["min_length"] = field.min_length
    if field.max_length is not None:
        constraints["max_length"] = field.max_length
    if constraints:
        item["constraints"] = constraints
    return item


def context_from_row(row: CanonicalRow) -> dict[str, Any]:
    mode = SpecializationMode(row.metadata["specialization_mode"])
    fields = select_fields(row.semantic_context, mode)
    return {
        "locale": row.semantic_context.locale,
        "active": row.semantic_context.active_field_id,
        "fields": [field_to_context(field) for field in fields],
    }


def calls_from_row(row: CanonicalRow) -> list[dict[str, Any]]:
    target = row.target
    if target.name == "no_change":
        return []
    args = target.arguments
    if target.name == "set_enum":
        return [set_call("set_choice", str(args["field_id"]), args["option_id"])]
    return [set_call(target.name, str(args["field_id"]), args["value"])]


def messages_for_training(user: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "developer", "content": DEVELOPER_INSTRUCTION},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def messages_for_prompt(user: str) -> list[dict[str, str]]:
    return [
        {"role": "developer", "content": DEVELOPER_INSTRUCTION},
        {"role": "user", "content": user},
    ]


def render_training_text(tokenizer, user: str, assistant: str) -> str:
    text = tokenizer.apply_chat_template(
        messages_for_training(user, assistant),
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty chat-template render")
    return text


def render_prompt_text(tokenizer, user: str) -> str:
    text = tokenizer.apply_chat_template(
        messages_for_prompt(user),
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(text, str) or _MODEL_TURN not in text:
        raise ValueError("prompt render missing model turn")
    return text


def llama_prompt(user: str) -> str:
    """Prompt for llama.cpp, which prepends <bos> itself."""
    return (
        "<start_of_turn>developer\n"
        f"{DEVELOPER_INSTRUCTION}<end_of_turn>\n"
        "<start_of_turn>user\n"
        f"{user.strip()}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def assistant_span(text: str) -> str:
    idx = text.rfind(_MODEL_TURN)
    if idx < 0:
        raise ValueError("model turn missing")
    rest = text[idx + len(_MODEL_TURN) :]
    end = rest.find(_END_TURN)
    if end < 0:
        raise ValueError("model turn is not closed")
    return rest[:end]


def prompt_span(text: str) -> str:
    idx = text.rfind(_MODEL_TURN)
    if idx < 0:
        raise ValueError("model turn missing")
    return text[: idx + len(_MODEL_TURN)]


def parse_function_calls(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("function_calls"), list):
        return None
    return {"function_calls": obj["function_calls"]}


def _norm_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return value
    if isinstance(value, list):
        return tuple(sorted((_norm_value(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True, default=str)))
    return value


def call_key(call: dict[str, Any]) -> tuple:
    args = call.get("arguments") or {}
    value = _norm_value(args["value"]) if "value" in args else None
    return (call.get("name"), args.get("field_id"), value)


def calls_equal(gold: list[dict[str, Any]], pred: list[dict[str, Any]]) -> bool:
    return Counter(call_key(call) for call in gold) == Counter(call_key(call) for call in pred)


def score_function_calls(gold: dict[str, Any], pred: dict[str, Any] | None) -> dict[str, bool]:
    """Score a function_calls object. An empty gold list is the no-change case."""
    g_calls = list(gold.get("function_calls") or [])
    parse_ok = pred is not None and isinstance(pred.get("function_calls"), list)
    p_calls = list(pred.get("function_calls") or []) if parse_ok else []
    empty_gold = len(g_calls) == 0
    empty_pred = parse_ok and len(p_calls) == 0
    exact = parse_ok and calls_equal(g_calls, p_calls)
    op_ok = field_ok = value_ok = False
    if exact:
        op_ok = field_ok = value_ok = True
    elif len(g_calls) == 1 and len(p_calls) == 1:
        g0, p0 = g_calls[0], p_calls[0]
        op_ok = g0.get("name") == p0.get("name")
        field_ok = op_ok and (g0.get("arguments") or {}).get("field_id") == (p0.get("arguments") or {}).get("field_id")
        if field_ok and g0.get("name") == "clear_field":
            value_ok = True
        elif field_ok:
            value_ok = _norm_value((g0.get("arguments") or {}).get("value")) == _norm_value(
                (p0.get("arguments") or {}).get("value")
            )
    false_mutation = empty_gold and parse_ok and not empty_pred
    choice_gold = [call for call in g_calls if call.get("name") == "set_choice"]
    if choice_gold:
        pred_keys = Counter(call_key(call) for call in p_calls) if parse_ok else Counter()
        enum_ok = all(pred_keys[call_key(call)] > 0 for call in choice_gold)
    else:
        enum_ok = exact
    return {
        "exact_call": exact,
        "op": op_ok,
        "field": field_ok,
        "value": value_ok if not empty_gold else empty_pred,
        "enum": enum_ok,
        "no_change_tp": empty_gold and empty_pred,
        "no_change_fp": (not empty_gold) and empty_pred,
        "no_change_fn": empty_gold and not empty_pred,
        "false_mutation": false_mutation,
        "parse_ok": parse_ok,
    }
