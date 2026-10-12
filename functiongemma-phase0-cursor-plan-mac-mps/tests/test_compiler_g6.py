"""G6 unit tests that do not require downloading the gated FunctionGemma weights."""
from __future__ import annotations

from form_sft.contracts.models import (
    CanonicalRow,
    EnumOption,
    Operation,
    SemanticContext,
    SemanticField,
)
from form_sft.training.compiler import (
    messages_for_row,
    parse_tool_calls_from_rendered,
    targets_equal,
    tools_for_row,
)


def _row(op: Operation, mode="active_field") -> CanonicalRow:
    ctx = SemanticContext(
        revision=1,
        fingerprint="sha256:test",
        locale="en-IN",
        active_field_id="age",
        reachable_field_ids=["age", "plan"],
        fields=[
            SemanticField(field_id="age", label="Age", type="integer", minimum=0, maximum=120),
            SemanticField(
                field_id="plan",
                label="Plan",
                type="enum",
                options=[EnumOption(id="basic", label="Basic"), EnumOption(id="premium", label="Premium")],
            ),
        ],
    )
    return CanonicalRow(
        id="t1",
        metadata={"specialization_mode": mode, "split": "train", "primitive": "integer", "source": "deterministic_template"},
        semantic_context=ctx,
        utterance="twenty nine",
        target=op,
    )


def test_tools_include_legal_ops_and_no_change():
    row = _row(Operation(name="set_integer", arguments={"field_id": "age", "value": 29}))
    tools = tools_for_row(row.semantic_context, "active_field")
    names = {t["function"]["name"] for t in tools}
    assert names == {"set_integer", "no_change"}
    si = next(t for t in tools if t["function"]["name"] == "set_integer")
    assert si["function"]["parameters"]["properties"]["field_id"]["enum"] == ["age"]


def test_reachable_tools_expose_both_fields():
    row = _row(Operation(name="set_enum", arguments={"field_id": "plan", "option_id": "basic"}), mode="reachable_fields")
    tools = tools_for_row(row.semantic_context, "reachable_fields")
    names = {t["function"]["name"] for t in tools}
    assert names == {"set_integer", "set_enum", "no_change"}


def test_messages_end_in_assistant_tool_call():
    row = _row(Operation(name="set_integer", arguments={"field_id": "age", "value": 29}))
    msgs = messages_for_row(row)
    assert msgs[0]["role"] == "developer"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["tool_calls"][0]["function"]["name"] == "set_integer"


def test_parse_tool_calls_roundtrip_shape():
    text = (
        "<start_of_turn>model\n"
        "<start_function_call>call:set_integer{field_id:<escape>age<escape>,value:29}<end_function_call>"
        "<end_of_turn>"
    )
    parsed = parse_tool_calls_from_rendered(text)
    assert parsed == [{"name": "set_integer", "arguments": {"field_id": "age", "value": 29}}]
    assert targets_equal(parsed[0], {"name": "set_integer", "arguments": {"field_id": "age", "value": 29}})


def test_parse_no_change_empty_args():
    text = "<start_function_call>call:no_change{}<end_function_call>"
    parsed = parse_tool_calls_from_rendered(text)
    assert parsed == [{"name": "no_change", "arguments": {}}]
