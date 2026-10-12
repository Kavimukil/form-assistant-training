"""Map between CanonicalAction and the production or native tool contracts."""
from __future__ import annotations

from typing import Any

from form_sft.contracts.dataset_ir import CanonicalAction, DatasetRecordV2
from form_sft.contracts.models import Operation
from form_sft.training.json_prompt import clear_call, set_call

_SETTER = {
    "string": "set_string",
    "integer": "set_integer",
    "number": "set_number",
    "boolean": "set_boolean",
    "date": "set_date",
    "enum": "set_choice",
    "multi": "set_multi_choice",
}


class CallAdapterError(ValueError):
    pass


def canonical_calls(record: DatasetRecordV2, tool_contract: str = "production_json") -> list[dict[str, Any]]:
    if tool_contract not in {"production_json", "native_tools"}:
        raise CallAdapterError(f"unknown tool contract {tool_contract}")
    actions = record.expected_actions
    if len(actions) == 1 and actions[0].kind == "no_change":
        if tool_contract == "native_tools":
            return [{"name": "no_change", "arguments": {}}]
        return []
    by_id = {field.field_id: field for field in record.schema.fields}
    calls = []
    for action in actions:
        if action.kind == "no_change":
            raise CallAdapterError("no_change mixed with mutations")
        field = by_id[action.field_id]
        if action.kind == "clear":
            if tool_contract == "native_tools":
                raise CallAdapterError("native contract has no clear_field")
            calls.append(clear_call(field.field_id))
            continue
        if tool_contract == "native_tools":
            if field.type == "enum":
                calls.append({"name": "set_enum", "arguments": {"field_id": field.field_id, "option_id": action.value}})
            elif field.type == "multi":
                raise CallAdapterError("native contract has no set_multi_choice")
            else:
                calls.append({"name": _SETTER[field.type], "arguments": {"field_id": field.field_id, "value": action.value}})
        else:
            calls.append(set_call(_SETTER[field.type], field.field_id, action.value))
    return calls


def calls_to_actions(calls: list[dict[str, Any]]) -> list[CanonicalAction]:
    if not calls:
        return [CanonicalAction(kind="no_change")]
    actions = []
    for call in calls:
        name = call.get("name")
        args = call.get("arguments") or {}
        if name == "no_change":
            actions.append(CanonicalAction(kind="no_change"))
        elif name == "clear_field":
            actions.append(CanonicalAction(kind="clear", field_id=args.get("field_id")))
        elif name == "set_enum":
            actions.append(CanonicalAction(kind="set", field_id=args.get("field_id"), value=args.get("option_id")))
        else:
            actions.append(CanonicalAction(kind="set", field_id=args.get("field_id"), value=args.get("value")))
    return actions


def action_to_operation(action: CanonicalAction, record: DatasetRecordV2) -> Operation:
    if action.kind == "no_change":
        return Operation(name="no_change", arguments={})
    if action.kind != "set":
        raise CallAdapterError("legacy operation set does not include this action")
    field = {item.field_id: item for item in record.schema.fields}[action.field_id]
    if field.type == "enum":
        return Operation(name="set_enum", arguments={"field_id": action.field_id, "option_id": action.value})
    if field.type == "multi":
        raise CallAdapterError("legacy operation set does not include multi")
    name = {
        "string": "set_string",
        "integer": "set_integer",
        "number": "set_number",
        "boolean": "set_boolean",
        "date": "set_date",
    }[field.type]
    return Operation(name=name, arguments={"field_id": action.field_id, "value": action.value})
