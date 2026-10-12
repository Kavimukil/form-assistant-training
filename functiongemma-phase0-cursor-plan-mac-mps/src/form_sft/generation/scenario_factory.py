"""Sample a valid form state and action set, then realize an utterance.

Actions and state are chosen before any utterance text exists.
"""
from __future__ import annotations

import copy
import random
from datetime import date, timedelta
from typing import Any

from pydantic import BaseModel, Field

from form_sft.contracts.dataset_ir import CanonicalAction
from form_sft.contracts.form_ir import FormIR, FormState

_FRAMES = {
    "train": {
        "single": "assign {field} as {value}",
        "clear": "blank out {field}",
        "correction": "assign {field} as {old} but now {new}",
        "no_change": "forecast board stays quiet",
        "pair": "{field} as {value}",
        "joiner": " plus ",
        "code": "haan",
        "filler": "uh",
        "choose": "choose {value}",
    },
    "validation": {
        "single": "record {field} equals {value}",
        "clear": "wipe {field}",
        "correction": "record {field} equals {old} rather {new}",
        "no_change": "calendar slate holds steady",
        "pair": "{field} equals {value}",
        "joiner": " along with ",
        "code": "bhau",
        "filler": "umm",
        "choose": "pick {value}",
    },
    "test": {
        "single": "place {value} into {field}",
        "clear": "erase {field}",
        "correction": "replace {field} from {old} into {new}",
        "no_change": "schedule slate remains idle",
        "pair": "{value} into {field}",
        "joiner": " together with ",
        "code": "anna",
        "filler": "eh",
        "choose": "select {value}",
    },
}


class ScenarioTruth(BaseModel):
    form: FormIR
    state: FormState
    actions: list[CanonicalAction]
    phenomenon: str
    split: str
    specialization_mode: str
    template_id: str
    template_lineage_id: str
    old_surfaces: list[str] = Field(default_factory=list)
    asr: bool = False
    code_mixed: bool = False
    correction: bool = False
    counterfactual: bool = False
    option_permutation: bool = False
    ambiguity: str = "accepted"
    field_types: list[str] = Field(default_factory=list)


def _rng(seed: int, *parts: object) -> random.Random:
    return random.Random("|".join(map(str, (seed, *parts))))


def surface_for(field, value: Any) -> str:
    if field.type == "boolean":
        return "yes" if value else "no"
    if field.type == "enum":
        for opt in field.options:
            if opt.id == value:
                return opt.label
        return str(value)
    if field.type == "multi":
        labels = {opt.id: opt.label for opt in field.options}
        return " and ".join(labels[item] for item in value)
    return str(value)


def _sample_value(field, rng: random.Random, avoid: Any = None) -> Any:
    for _ in range(20):
        if field.type == "integer":
            value = rng.randint(int(field.minimum), int(field.maximum))
        elif field.type == "number":
            lo, hi = float(field.minimum), float(field.maximum)
            value = round(lo + (hi - lo) * rng.random(), 2)
            if float(value).is_integer():
                value = float(value)
        elif field.type == "boolean":
            value = rng.choice([True, False])
        elif field.type == "date":
            start = date.fromisoformat(str(field.minimum))
            end = date.fromisoformat(str(field.maximum))
            value = (start + timedelta(days=rng.randrange((end - start).days + 1))).isoformat()
        elif field.type == "string":
            from form_sft.generation.schema_factory import syllable

            value = syllable(rng.randrange(20_000))
            if field.max_length and len(value) > field.max_length:
                value = value[: field.max_length]
        elif field.type == "enum":
            value = rng.choice(field.options).id
        elif field.type == "multi":
            picked = rng.sample(list(field.options), k=2)
            value = [opt.id for opt in picked]
        else:
            raise ValueError(field.type)
        if value != avoid:
            return value
    return value


def _field_by_type(form: FormIR, kind: str):
    return next(field for field in form.fields if field.type == kind)


def _targets(form: FormIR, request: dict, rng: random.Random) -> list:
    phenomenon = request["phenomenon"]
    if phenomenon == "single_set" or phenomenon in {"asr", "code_mixed", "correction", "option_permutation"}:
        kind = request.get("type_name") or ("integer", "number", "boolean", "enum", "string", "date")[request["index"] % 6]
        return [_field_by_type(form, kind)]
    if phenomenon == "clear":
        return [_field_by_type(form, "string")]
    if phenomenon == "no_change":
        return [_field_by_type(form, "string")]
    if phenomenon == "counterfactual":
        return [_field_by_type(form, "enum")]
    if phenomenon == "multi_field":
        if request["index"] % 2:
            kinds = ["integer", "string", "multi"]
        else:
            kinds = ["integer", "enum"]
        return [_field_by_type(form, kind) for kind in kinds]
    if phenomenon == "ambiguity":
        return [_field_by_type(form, "enum"), _field_by_type(form, "multi")]
    raise ValueError(phenomenon)


def _apply_shared_label(form: FormIR, label: str) -> FormIR:
    updated = form.model_copy(deep=True)
    enum_field = _field_by_type(updated, "enum")
    enum_field.options[0].label = label
    return updated


def generate_scenario(form: FormIR, state_seed: int, coverage_request: dict) -> ScenarioTruth:
    request = dict(coverage_request)
    rng = _rng(state_seed, request.get("index", 0), "state")
    if request.get("shared_label"):
        form = _apply_shared_label(form, request["shared_label"])
    if request["phenomenon"] == "ambiguity":
        form = _apply_shared_label(form, request.get("shared_label") or "sharedtoken")
        multi = _field_by_type(form, "multi")
        enum_field = _field_by_type(form, "enum")
        multi.options[0].label = enum_field.options[0].label
    targets = _targets(form, request, rng)
    actions: list[CanonicalAction] = []
    old_surfaces: list[str] = []
    phenomenon = request["phenomenon"]
    if phenomenon == "no_change":
        actions = [CanonicalAction(kind="no_change")]
    elif phenomenon == "clear":
        actions = [CanonicalAction(kind="clear", field_id=targets[0].field_id)]
    elif phenomenon == "ambiguity":
        enum_field = targets[0]
        actions = [CanonicalAction(kind="set", field_id=enum_field.field_id, value=enum_field.options[0].id)]
    else:
        for field in targets:
            value = field.options[0].id if request.get("shared_label") and field.type == "enum" else _sample_value(field, rng)
            if phenomenon == "correction":
                old = _sample_value(field, rng, avoid=value)
                old_surfaces.append(surface_for(field, old))
            actions.append(CanonicalAction(kind="set", field_id=field.field_id, value=value))

    mode = request.get("mode")
    if mode is None:
        mode = "reachable_fields" if len([item for item in actions if item.kind != "no_change"]) != 1 else "active_field"
    target_ids = [action.field_id for action in actions if action.field_id]
    others = [field.field_id for field in form.fields if field.field_id not in target_ids]
    if mode == "active_field":
        reachable = target_ids + others[:2]
        active = target_ids[0] if target_ids else form.fields[0].field_id
    else:
        reachable = [field.field_id for field in form.fields]
        active = None
    values = {}
    for field in form.fields:
        if field.field_id not in reachable:
            continue
        action = next((item for item in actions if item.field_id == field.field_id and item.kind == "set"), None)
        avoid = action.value if action else None
        if field.type == "boolean" and any(other.type == "date" and other.field_id in reachable for other in form.fields):
            values[field.field_id] = True if action is None or action.value is False else False
            if action and action.value is True:
                values[field.field_id] = False
            continue
        values[field.field_id] = _sample_value(field, rng, avoid=avoid)
        if action and action.kind == "clear":
            values[field.field_id] = _sample_value(field, rng)
    state = FormState(
        revision=1 + (request.get("index", 0) % 7),
        active_field_id=active,
        values=values,
        reachable_field_ids=reachable,
        visible_field_ids=list(reachable),
    )
    from form_sft.validation.record import hidden_field_ids

    hidden = hidden_field_ids(form, state)
    state.reachable_field_ids = [fid for fid in state.reachable_field_ids if fid not in hidden]
    state.visible_field_ids = list(state.reachable_field_ids)
    if state.active_field_id in hidden:
        state.active_field_id = state.reachable_field_ids[0]
    return ScenarioTruth(
        form=form,
        state=state,
        actions=actions,
        phenomenon=phenomenon,
        split=request["split"],
        specialization_mode=mode,
        template_id=f"{request['split']}.{phenomenon}.v1",
        template_lineage_id=form.template_lineage_id,
        old_surfaces=old_surfaces,
        asr=bool(request.get("asr") or phenomenon == "asr"),
        code_mixed=bool(request.get("code_mixed") or phenomenon == "code_mixed"),
        correction=phenomenon == "correction",
        counterfactual=phenomenon == "counterfactual",
        option_permutation=bool(request.get("option_permutation")),
        ambiguity="quarantine" if phenomenon == "ambiguity" else "accepted",
        field_types=[field.type for field in targets],
    )


def realize_scenario(truth: ScenarioTruth, surface_seed: int) -> str:
    """Render an utterance. The truth object is not modified."""
    del surface_seed
    frames = _FRAMES[truth.split]
    by_id = {field.field_id: field for field in truth.form.fields}
    if truth.phenomenon == "no_change":
        text = frames["no_change"]
    elif truth.phenomenon == "clear":
        field = by_id[truth.actions[0].field_id]
        text = frames["clear"].format(field=field.label)
    elif truth.phenomenon == "ambiguity":
        field = by_id[truth.actions[0].field_id]
        text = surface_for(field, truth.actions[0].value)
    elif truth.phenomenon == "counterfactual":
        field = by_id[truth.actions[0].field_id]
        text = frames["choose"].format(value=surface_for(field, truth.actions[0].value))
    elif truth.correction:
        action = truth.actions[0]
        field = by_id[action.field_id]
        text = frames["correction"].format(
            field=field.label,
            old=truth.old_surfaces[0],
            new=surface_for(field, action.value),
        )
    elif truth.phenomenon == "multi_field":
        parts = []
        for action in truth.actions:
            field = by_id[action.field_id]
            parts.append(frames["pair"].format(field=field.label, value=surface_for(field, action.value)))
        text = frames["joiner"].join(parts)
    else:
        action = truth.actions[0]
        field = by_id[action.field_id]
        text = frames["single"].format(field=field.label, value=surface_for(field, action.value))
    if truth.code_mixed:
        text = f"{frames['code']} {text}"
    if truth.asr:
        text = text.replace(" ", f" {frames['filler']} ", 1)
    return " ".join(text.split())


def clone_truth(truth: ScenarioTruth) -> ScenarioTruth:
    return truth.model_copy(deep=True)


def reverse_enum_options(form: FormIR) -> FormIR:
    updated = copy.deepcopy(form)
    for field in updated.fields:
        if field.type == "enum":
            field.options = list(reversed(field.options))
    updated.schema_id = form.schema_id + "9"
    return updated
