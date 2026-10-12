"""Replay canonical actions against a form and validate dataset records."""
from __future__ import annotations

import copy
from datetime import date
from typing import Any

from pydantic import BaseModel

from form_sft.contracts.dataset_ir import VALIDATOR_VERSION, DatasetRecordV2
from form_sft.contracts.form_ir import FormIR, FormState, schema_fingerprint

_REQUIRED_PROVENANCE = (
    "generator",
    "generator_version",
    "template_id",
    "template_lineage_id",
    "parent_schema_id",
    "parent_scenario_id",
    "validator_version",
)


class ValidationIssue(BaseModel):
    code: str
    detail: str = ""


class ActionError(ValueError):
    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}")


def _issue(code: str, detail: str = "") -> ValidationIssue:
    return ValidationIssue(code=code, detail=detail)


def dependency_cycle(form: FormIR) -> bool:
    graph: dict[str, list[str]] = {}
    for rule in form.dependencies:
        graph.setdefault(rule.when_field_id, []).append(rule.then_field_id)
    visiting: set[str] = set()
    visited: set[str] = set()

    def walk(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for nxt in graph.get(node, []):
            if walk(nxt):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(walk(node) for node in list(graph))


def _type_ok(field, value: Any) -> bool:
    if field.type == "string":
        return isinstance(value, str)
    if field.type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if field.type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if field.type == "boolean":
        return isinstance(value, bool)
    if field.type == "enum":
        return isinstance(value, str)
    if field.type == "date":
        return isinstance(value, str)
    if field.type == "multi":
        return isinstance(value, list) and all(isinstance(item, str) for item in value)
    return False


def _constraint_error(field, value: Any) -> str | None:
    if not _type_ok(field, value):
        return "V_VALUE_TYPE"
    if field.type == "enum" and value not in {opt.id for opt in field.options}:
        return "V_UNKNOWN_OPTION"
    if field.type == "multi":
        ids = [opt.id for opt in field.options]
        if len(value) != len(set(value)) or any(item not in ids for item in value):
            return "V_UNKNOWN_OPTION"
    if field.type in {"integer", "number"}:
        if field.minimum is not None and value < field.minimum:
            return "V_VALUE_CONSTRAINT"
        if field.maximum is not None and value > field.maximum:
            return "V_VALUE_CONSTRAINT"
    if field.type == "string":
        if field.min_length is not None and len(value) < field.min_length:
            return "V_VALUE_CONSTRAINT"
        if field.max_length is not None and len(value) > field.max_length:
            return "V_VALUE_CONSTRAINT"
    if field.type == "date":
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            return "V_VALUE_TYPE"
        if field.minimum and parsed < date.fromisoformat(str(field.minimum)):
            return "V_VALUE_CONSTRAINT"
        if field.maximum and parsed > date.fromisoformat(str(field.maximum)):
            return "V_VALUE_CONSTRAINT"
    return None


def hidden_field_ids(form: FormIR, state: FormState) -> set[str]:
    hidden: set[str] = set()
    for rule in form.dependencies:
        if rule.effect != "hide":
            continue
        current = state.values.get(rule.when_field_id)
        if current == rule.when_value:
            hidden.add(rule.then_field_id)
    return hidden


def apply_actions(form: FormIR, state: FormState, actions: list) -> FormState:
    """Return the post-action state. The input state is not mutated."""
    by_id = {field.field_id: field for field in form.fields}
    post = state.model_copy(deep=True)
    kinds = [action.kind for action in actions]
    if not actions:
        raise ActionError("V_NO_ACTIONS")
    if "no_change" in kinds:
        if kinds != ["no_change"]:
            raise ActionError("V_NO_CHANGE_MIXED")
        return post
    for action in actions:
        fid = action.field_id
        if fid not in by_id:
            raise ActionError("V_ACTION_UNKNOWN_FIELD", str(fid))
        if fid not in post.reachable_field_ids or fid in hidden_field_ids(form, post):
            raise ActionError("V_ACTION_NOT_REACHABLE", str(fid))
        if action.kind == "clear":
            if action.value is not None:
                raise ActionError("V_CLEAR_VALUE", fid)
            post.values[fid] = None
            continue
        if action.kind != "set":
            raise ActionError("V_ACTION_KIND", action.kind)
        if action.value is None and by_id[fid].type != "string":
            raise ActionError("V_SET_MISSING_VALUE", fid)
        problem = _constraint_error(by_id[fid], action.value)
        if problem:
            raise ActionError(problem, fid)
        post.values[fid] = copy.deepcopy(action.value)
    post.revision = state.revision + 1
    hidden = hidden_field_ids(form, post)
    post.reachable_field_ids = [fid for fid in post.reachable_field_ids if fid not in hidden]
    post.visible_field_ids = [fid for fid in post.visible_field_ids if fid not in hidden]
    if post.active_field_id in hidden:
        post.active_field_id = None
    return post


def validate_record(record: DatasetRecordV2) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    form, state = record.schema, record.state
    ids = [field.field_id for field in form.fields]
    if len(ids) != len(set(ids)):
        issues.append(_issue("V_DUPLICATE_FIELD_ID"))
    by_id = {field.field_id: field for field in form.fields}
    for field in form.fields:
        opt_ids = [opt.id for opt in field.options]
        if len(opt_ids) != len(set(opt_ids)):
            issues.append(_issue("V_DUPLICATE_OPTION_ID", field.field_id))
        if field.type in {"enum", "multi"} and not field.options:
            issues.append(_issue("V_MISSING_OPTIONS", field.field_id))
    known = set(ids)
    for rule in form.dependencies:
        if rule.when_field_id not in known or rule.then_field_id not in known:
            issues.append(_issue("V_DEPENDENCY_UNKNOWN_FIELD", rule.rule_id))
        if rule.when_field_id == rule.then_field_id:
            issues.append(_issue("V_DEPENDENCY_CYCLE", rule.rule_id))
    if dependency_cycle(form):
        issues.append(_issue("V_DEPENDENCY_CYCLE"))
    if state.active_field_id is not None and state.active_field_id not in state.reachable_field_ids:
        issues.append(_issue("V_ACTIVE_NOT_REACHABLE", str(state.active_field_id)))
    for fid in list(state.reachable_field_ids) + list(state.visible_field_ids) + list(state.values):
        if fid not in known:
            issues.append(_issue("V_STATE_UNKNOWN_FIELD", fid))
    if record.coverage.specialization_mode == "active_field" and not state.active_field_id:
        issues.append(_issue("V_ACTIVE_NOT_REACHABLE", "missing"))
    if not record.utterance or not record.utterance.strip():
        issues.append(_issue("V_EMPTY_UTTERANCE"))
    for name in _REQUIRED_PROVENANCE:
        value = getattr(record.provenance, name)
        if value is None or value == "":
            issues.append(_issue("V_PROVENANCE", name))
    if record.provenance.validator_version != VALIDATOR_VERSION and record.provenance.legacy_metadata is None:
        issues.append(_issue("V_PROVENANCE", "validator_version"))
    expected_fp = schema_fingerprint(form, state)
    if record.schema_fingerprint != expected_fp:
        issues.append(_issue("V_FINGERPRINT_MISMATCH", record.schema_fingerprint))
    if record.provenance.legacy_metadata is None and record.semantic_context.fingerprint != expected_fp:
        issues.append(_issue("V_FINGERPRINT_MISMATCH", "semantic_context"))
    if record.semantic_context.revision != state.revision:
        issues.append(_issue("V_STALE_REVISION", str(state.revision)))
    try:
        apply_actions(form, state, record.expected_actions)
    except ActionError as exc:
        issues.append(_issue(exc.code, str(exc)))
    mutating = [action for action in record.expected_actions if action.kind != "no_change"]
    if record.coverage.action_arity != len(mutating):
        issues.append(_issue("V_ARITY", str(record.coverage.action_arity)))
    return issues
