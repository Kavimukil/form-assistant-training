"""Model-independent form schema and state.

Field ids, option ids, constraints, and dependency structure are the identity
of a form. Labels and aliases are surface text and are excluded from
fingerprints.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

FieldKind = Literal["string", "integer", "number", "boolean", "enum", "date", "multi"]
DependencyEffect = Literal["require", "hide", "set_default"]


class OptionSpec(BaseModel):
    id: str
    label: str
    aliases: list[str] = Field(default_factory=list)


class FieldSpec(BaseModel):
    field_id: str
    label: str
    type: FieldKind
    description: str | None = None
    aliases: list[str] = Field(default_factory=list)
    minimum: int | float | str | None = None
    maximum: int | float | str | None = None
    min_length: int | None = None
    max_length: int | None = None
    options: list[OptionSpec] = Field(default_factory=list)
    ontology_lineage: str = ""


class DependencyRule(BaseModel):
    rule_id: str
    when_field_id: str
    when_value: Any = None
    then_field_id: str
    effect: DependencyEffect
    default_value: Any = None


class FormIR(BaseModel):
    schema_id: str
    schema_family_id: str
    template_lineage_id: str
    ontology_lineage_id: str
    locale: str = "en-IN"
    fields: list[FieldSpec]
    dependencies: list[DependencyRule] = Field(default_factory=list)


class FormState(BaseModel):
    revision: int
    active_field_id: str | None = None
    values: dict[str, Any] = Field(default_factory=dict)
    reachable_field_ids: list[str]
    visible_field_ids: list[str]


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def structural_projection(form: FormIR, state: FormState) -> dict:
    """Schema plus visible state, with labels and descriptions removed."""
    by_id = {field.field_id: field for field in form.fields}
    fields = []
    for fid in state.reachable_field_ids:
        field = by_id[fid]
        fields.append(
            {
                "field_id": field.field_id,
                "type": field.type,
                "minimum": field.minimum,
                "maximum": field.maximum,
                "min_length": field.min_length,
                "max_length": field.max_length,
                "option_ids": [opt.id for opt in field.options],
                "ontology_lineage": field.ontology_lineage,
            }
        )
    return {
        "schema_id": form.schema_id,
        "schema_family_id": form.schema_family_id,
        "ontology_lineage_id": form.ontology_lineage_id,
        "locale": form.locale,
        "active_field_id": state.active_field_id,
        "reachable_field_ids": list(state.reachable_field_ids),
        "visible_field_ids": list(state.visible_field_ids),
        "values": {fid: state.values.get(fid) for fid in state.reachable_field_ids},
        "fields": fields,
        "dependencies": [rule.model_dump() for rule in form.dependencies],
    }


def schema_fingerprint(form: FormIR, state: FormState) -> str:
    body = _canonical(structural_projection(form, state)).encode()
    return "sha256:" + hashlib.sha256(body).hexdigest()
