"""Serialize DatasetRecordV2 with the production JSON tool-call contract.

Native FunctionGemma tool declarations stay a secondary adapter. New corpora
do not append the hand-built rows in ``json_examples``.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from form_sft.adapters.calls import CallAdapterError, calls_to_actions, canonical_calls
from form_sft.bridges.form_state import production_context
from form_sft.contracts.dataset_ir import CanonicalAction, DatasetRecordV2
from form_sft.training.json_prompt import (
    assistant_span,
    dumps_calls,
    llama_prompt,
    prompt_span,
    render_training_text,
    tool_line,
    user_block,
)


class SFTExample(BaseModel):
    id: str
    text: str
    prompt: str
    target: dict[str, Any]
    token_length: int
    truncated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    contract: str
    render_mode: str


class SerializeError(RuntimeError):
    pass


def deserialize_target(text: str) -> list[CanonicalAction]:
    from form_sft.training.json_prompt import parse_function_calls

    parsed = parse_function_calls(text)
    if parsed is None:
        raise SerializeError("target is not a function_calls object")
    return calls_to_actions(parsed["function_calls"])


def _fallback_text(user: str, assistant: str) -> str:
    prompt = llama_prompt(user)
    return f"{prompt}{assistant}<end_of_turn>\n"


def serialize(record: DatasetRecordV2, contract: str = "production_json", tokenizer=None) -> SFTExample:
    if contract == "native_tools":
        return _serialize_native(record, tokenizer)
    if contract != "production_json":
        raise SerializeError(f"unknown contract {contract}")
    try:
        calls = canonical_calls(record, "production_json")
    except CallAdapterError as exc:
        raise SerializeError(str(exc)) from exc
    context = production_context(record)
    tools = tool_line(context["fields"])
    user = user_block(record.utterance, context, tools)
    assistant = dumps_calls(calls)
    if tokenizer is None:
        text = _fallback_text(user, assistant)
        prompt = llama_prompt(user)
        render_mode = "llama_cpp_compatible"
        token_length = len(text.encode())
    else:
        text = render_training_text(tokenizer, user, assistant)
        prompt = prompt_span(text)
        if assistant_span(text) != assistant:
            raise SerializeError(f"{record.id}: assistant span drifted")
        token_length = len(_encode(tokenizer, text))
        render_mode = "apply_chat_template"
    return SFTExample(
        id=record.id,
        text=text,
        prompt=prompt,
        target={"function_calls": calls},
        token_length=token_length,
        truncated=False,
        metadata={
            "split": record.coverage.split,
            "phenomenon": record.coverage.phenomenon,
            "specialization_mode": record.coverage.specialization_mode,
            "suites": list(record.coverage.suites),
        },
        contract=contract,
        render_mode=render_mode,
    )


def _encode(tokenizer, text: str) -> list[int]:
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    return list(tok.encode(text, add_special_tokens=False))


def _serialize_native(record: DatasetRecordV2, tokenizer) -> SFTExample:
    from form_sft.training.compiler import messages_for_row, tools_for_row

    calls = canonical_calls(record, "native_tools")
    if len(calls) != 1:
        raise SerializeError("native adapter currently round-trips one tool call")
    # Reuse the installed compiler helpers by presenting a one-call legacy row.
    from form_sft.adapters.legacy import LegacyAdapterError, to_legacy_row

    try:
        legacy = to_legacy_row(record)
    except LegacyAdapterError as exc:
        raise SerializeError(str(exc)) from exc
    tools = tools_for_row(legacy.semantic_context, legacy.metadata["specialization_mode"])
    messages = messages_for_row(legacy)
    payload = {"messages": messages, "tools": tools}
    text = __import__("json").dumps(payload, sort_keys=True)
    if tokenizer is not None:
        from form_sft.training.compiler import compile_example

        text = compile_example(tokenizer, messages, tools)
    return SFTExample(
        id=record.id,
        text=text,
        prompt=text,
        target={"function_calls": calls},
        token_length=len(text.encode()),
        metadata={"split": record.coverage.split},
        contract="native_tools",
        render_mode="native_tools",
    )
