"""Tokenizer, truncation, and loss-mask checks shared by compile and train."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from form_sft.training.json_prompt import assistant_span, prompt_span
from form_sft.training.serializer import SFTExample, deserialize_target


class PreflightResult(BaseModel):
    ok: bool
    problems: list[str] = Field(default_factory=list)
    token_length: int = 0
    prompt_tokens_masked: int = 0
    trainable_tokens: int = 0
    render_mode: str = ""


def _unwrap(tokenizer):
    return getattr(tokenizer, "tokenizer", tokenizer)


def encode(tokenizer, text: str) -> list[int]:
    tok = _unwrap(tokenizer)
    return list(tok.encode(text, add_special_tokens=False))


def mask_labels(input_ids: list[int], prompt_len: int, pad_id: int | None, *, loss_policy: str) -> list[int]:
    labels = []
    for index, token in enumerate(input_ids):
        masked = token == pad_id or (loss_policy == "assistant_only" and index < prompt_len)
        labels.append(-100 if masked else token)
    return labels


def _norm_value(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def verify_sft(example: SFTExample, tokenizer, max_length: int, loss_policy: str = "assistant_only") -> PreflightResult:
    problems: list[str] = []
    ids = encode(tokenizer, example.text)
    if example.render_mode == "apply_chat_template" and len(ids) > max_length:
        problems.append(f"token_length {len(ids)} > {max_length}")
    elif example.render_mode != "apply_chat_template" and example.token_length > max_length * 8:
        problems.append(f"byte_length {example.token_length} is far above the sequence budget")
    try:
        if example.contract == "production_json":
            span = assistant_span(example.text)
            actions = deserialize_target(span)
        else:
            from form_sft.adapters.calls import calls_to_actions

            actions = calls_to_actions(example.target["function_calls"])
        from form_sft.adapters.calls import calls_to_actions

        gold = calls_to_actions(example.target["function_calls"])
        got = [(item.kind, item.field_id, _norm_value(item.value)) for item in actions]
        expected = [(item.kind, item.field_id, _norm_value(item.value)) for item in gold]
        if got != expected:
            problems.append("round_trip")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"round_trip:{type(exc).__name__}")
    ids = encode(tokenizer, example.text)
    prompt_ids = encode(tokenizer, example.prompt)
    if ids[: len(prompt_ids)] != prompt_ids:
        problems.append("prompt_prefix")
    pad_id = getattr(_unwrap(tokenizer), "pad_token_id", None)
    labels = mask_labels(ids, len(prompt_ids), pad_id, loss_policy=loss_policy)
    if loss_policy == "assistant_only":
        if any(label != -100 for label in labels[: len(prompt_ids)]):
            problems.append("prompt_not_masked")
        if not labels[len(prompt_ids) :] or any(label == -100 for label in labels[len(prompt_ids) :]):
            problems.append("assistant_not_trainable")
        if example.render_mode == "apply_chat_template":
            span = assistant_span(example.text)
            if not span:
                problems.append("missing_assistant_span")
            if prompt_span(example.text) != example.prompt:
                problems.append("prompt_span")
    return PreflightResult(
        ok=not problems,
        problems=problems,
        token_length=len(ids),
        prompt_tokens_masked=sum(1 for label in labels if label == -100),
        trainable_tokens=sum(1 for label in labels if label != -100),
        render_mode=example.render_mode,
    )


def package_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {}
    for name in ("transformers", "trl", "torch"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            versions[name] = f"unavailable:{type(exc).__name__}"
    return versions
