"""Check that an utterance supports the action set and only that action set."""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from form_sft.contracts.dataset_ir import DatasetRecordV2


class AgreementResult(BaseModel):
    ok: bool
    reasons: list[str]


def _norm(text: str) -> str:
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9. ]", " ", text)
    return " ".join(text.split())


def _has(utterance: str, piece: str) -> bool:
    return f" {_norm(piece)} " in f" {_norm(utterance)} "


def _surface(field, value: Any) -> str:
    if field.type == "boolean":
        return "yes" if value else "no"
    if field.type == "enum":
        for opt in field.options:
            if opt.id == value:
                return opt.label
        return str(value)
    if field.type == "multi":
        labels = []
        by_id = {opt.id: opt.label for opt in field.options}
        for item in value:
            labels.append(by_id.get(item, str(item)))
        return " and ".join(labels)
    return str(value)


def semantic_agreement(record: DatasetRecordV2, *, old_surfaces: list[str] | None = None) -> AgreementResult:
    reasons: list[str] = []
    by_id = {field.field_id: field for field in record.schema.fields}
    utterance = record.utterance
    actions = record.expected_actions
    if len(actions) == 1 and actions[0].kind == "no_change":
        for fid in record.state.reachable_field_ids:
            field = by_id[fid]
            if _has(utterance, field.label):
                reasons.append(f"no_change mentions {field.field_id}")
            for opt in field.options:
                if _has(utterance, opt.label):
                    reasons.append(f"no_change mentions option {opt.id}")
        return AgreementResult(ok=not reasons, reasons=reasons)

    for action in actions:
        field = by_id[action.field_id]
        if action.kind == "clear":
            if not any(_has(utterance, cue) for cue in ("blank out", "erase", "wipe", "clear", "remove")):
                reasons.append(f"clear cue missing for {field.field_id}")
            if not (_has(utterance, field.label) or any(_has(utterance, alias) for alias in field.aliases)):
                reasons.append(f"clear field missing for {field.field_id}")
            continue
        surface = _surface(field, action.value)
        label_mentioned = _has(utterance, field.label) or any(_has(utterance, alias) for alias in field.aliases)
        if not _has(utterance, surface):
            reasons.append(f"value surface missing for {field.field_id}")
        if not label_mentioned:
            # An option label is enough when it identifies one reachable field.
            owners = []
            for other in record.schema.fields:
                if other.field_id not in record.state.reachable_field_ids:
                    continue
                if any(_has(utterance, opt.label) and opt.id == action.value for opt in other.options):
                    owners.append(other.field_id)
            if owners != [field.field_id]:
                reasons.append(f"field not identified for {field.field_id}")
    if record.coverage.correction:
        surfaces = [_surface(by_id[action.field_id], action.value) for action in actions if action.kind == "set"]
        if old_surfaces:
            for old in old_surfaces:
                if not _has(utterance, old):
                    reasons.append("correction omits the replaced value")
        norm_u = _norm(utterance)
        for surface in surfaces:
            if norm_u.rfind(_norm(surface)) < 0:
                reasons.append("correction final value missing")
    return AgreementResult(ok=not reasons, reasons=reasons)
