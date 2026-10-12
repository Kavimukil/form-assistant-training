"""Map FormIR plus FormState onto the existing SemanticContext and production prompt context."""
from __future__ import annotations

from typing import Any

from form_sft.contracts.form_ir import FormIR, FormState, schema_fingerprint
from form_sft.contracts.models import CanonicalRow, EnumOption, Operation, SemanticContext, SemanticField
from form_sft.training.json_prompt import context_from_row

_LEGACY_TYPES = {"string", "integer", "number", "boolean", "enum", "date"}


class FormStateBridge:
    @staticmethod
    def to_semantic_context(form: FormIR, state: FormState, mode: str) -> SemanticContext:
        if mode not in {"active_field", "reachable_fields"}:
            raise ValueError(f"unknown specialization mode {mode}")
        by_id = {field.field_id: field for field in form.fields}
        semantic_fields: list[SemanticField] = []
        for fid in state.reachable_field_ids:
            field = by_id[fid]
            if field.type not in _LEGACY_TYPES:
                continue
            options = [EnumOption(id=opt.id, label=opt.label, aliases=list(opt.aliases)) for opt in field.options]
            semantic_fields.append(
                SemanticField(
                    field_id=field.field_id,
                    label=field.label,
                    type=field.type,  # type: ignore[arg-type]
                    description=field.description,
                    minimum=field.minimum,
                    maximum=field.maximum,
                    min_length=field.min_length,
                    max_length=field.max_length,
                    options=options,
                )
            )
        return SemanticContext(
            revision=state.revision,
            fingerprint=schema_fingerprint(form, state),
            locale=form.locale,
            active_field_id=state.active_field_id,
            reachable_field_ids=[fid for fid in state.reachable_field_ids if by_id[fid].type in _LEGACY_TYPES],
            fields=semantic_fields,
        )


def field_dict(field) -> dict[str, Any]:
    """Production field object. Keys match ``field_to_context`` for legacy types."""
    item: dict[str, Any] = {"id": field.field_id, "type": field.type, "label": field.label}
    if field.type in {"enum", "multi"} and field.options:
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


def selected_field_ids(form: FormIR, state: FormState, mode: str) -> list[str]:
    if mode == "active_field":
        if not state.active_field_id:
            raise ValueError("active_field_id required")
        return [state.active_field_id]
    return list(state.reachable_field_ids)


def production_context(record) -> dict[str, Any]:
    """Context block embedded in the production user turn.

    For records whose fields all fit SemanticContext, this equals
    ``context_from_row`` so converted Phase-0 rows keep their prompt shape.
    """
    mode = record.coverage.specialization_mode
    by_id = {field.field_id: field for field in record.schema.fields}
    ids = selected_field_ids(record.schema, record.state, mode)
    legacy_only = all(by_id[fid].type in _LEGACY_TYPES for fid in ids)
    if legacy_only and record.semantic_context is not None:
        row = CanonicalRow(
            id=record.id,
            metadata={"specialization_mode": mode},
            semantic_context=record.semantic_context,
            utterance=record.utterance or "x",
            target=Operation(name="no_change", arguments={}),
        )
        produced = context_from_row(row)
        return produced
    return {
        "locale": record.schema.locale,
        "active": record.state.active_field_id,
        "fields": [field_dict(by_id[fid]) for fid in ids],
    }
