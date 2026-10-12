"""Quarantine utterances that fit more than one reachable action set."""
from __future__ import annotations

import re

from form_sft.contracts.dataset_ir import DatasetRecordV2


def _norm(text: str) -> str:
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9. ]", " ", text)
    return " ".join(text.split())


def _has(utterance: str, piece: str) -> bool:
    return f" {_norm(piece)} " in f" {_norm(utterance)} "


def classify_ambiguity(record: DatasetRecordV2) -> str:
    """Return ``accepted`` or ``quarantine``. Deterministic."""
    if record.coverage.ambiguity == "quarantine":
        return "quarantine"
    by_id = {field.field_id: field for field in record.schema.fields}
    utterance = record.utterance
    if any(action.kind == "no_change" for action in record.expected_actions):
        return "accepted"
    mentioned = []
    for fid in record.state.reachable_field_ids:
        field = by_id[fid]
        named = _has(utterance, field.label) or any(_has(utterance, alias) for alias in field.aliases)
        option_hits = [opt for opt in field.options if _has(utterance, opt.label) or any(_has(utterance, alias) for alias in opt.aliases)]
        if named or option_hits:
            mentioned.append((field, named, option_hits))
    if len(mentioned) > len([action for action in record.expected_actions if action.kind != "no_change"]) and not any(
        named for _, named, _ in mentioned
    ):
        return "quarantine"
    # Two reachable fields share an option surface and the utterance does not name a field.
    surfaces: dict[str, list[str]] = {}
    for fid in record.state.reachable_field_ids:
        field = by_id[fid]
        for opt in field.options:
            if _has(utterance, opt.label):
                surfaces.setdefault(_norm(opt.label), []).append(field.field_id)
    named_any = False
    for fid in record.state.reachable_field_ids:
        field = by_id[fid]
        if _has(utterance, field.label) or any(_has(utterance, alias) for alias in field.aliases):
            named_any = True
    if any(len(set(owners)) > 1 for owners in surfaces.values()) and not named_any:
        return "quarantine"
    # A correction that ends on a different value than the action is contradictory.
    if record.coverage.correction:
        for action in record.expected_actions:
            if action.kind != "set":
                continue
            field = by_id[action.field_id]
            if field.type == "integer":
                nums = [int(item) for item in re.findall(r"\d+", utterance)]
                if nums and nums[-1] != action.value:
                    return "quarantine"
    return "accepted"
