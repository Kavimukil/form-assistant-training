"""Lossless conversion between Phase-0 CanonicalRow and DatasetRecordV2.

``to_legacy_row`` accepts only a single set or no_change action whose field type
exists in the Phase-0 operation vocabulary.
"""
from __future__ import annotations

from form_sft.adapters.calls import CallAdapterError, action_to_operation
from form_sft.bridges.form_state import FormStateBridge
from form_sft.contracts.dataset_ir import (
    VALIDATOR_VERSION,
    CanonicalAction,
    CoverageTags,
    DatasetRecordV2,
    Provenance,
)
from form_sft.contracts.form_ir import FieldSpec, FormIR, FormState, OptionSpec, schema_fingerprint
from form_sft.contracts.models import CanonicalRow, Operation


class LegacyAdapterError(ValueError):
    pass


def _action_from_operation(op: Operation) -> CanonicalAction:
    if op.name == "no_change":
        return CanonicalAction(kind="no_change")
    args = op.arguments
    if op.name == "set_enum":
        return CanonicalAction(kind="set", field_id=args["field_id"], value=args["option_id"])
    return CanonicalAction(kind="set", field_id=args["field_id"], value=args["value"])


def from_legacy_row(row: CanonicalRow) -> DatasetRecordV2:
    fields = []
    for field in row.semantic_context.fields:
        fields.append(
            FieldSpec(
                field_id=field.field_id,
                label=field.label,
                type=field.type,
                description=field.description,
                minimum=field.minimum,
                maximum=field.maximum,
                min_length=field.min_length,
                max_length=field.max_length,
                options=[OptionSpec(id=opt.id, label=opt.label, aliases=list(opt.aliases)) for opt in field.options],
                ontology_lineage=str(row.metadata.get("domain") or "legacy"),
            )
        )
    family = str(row.metadata.get("domain") or row.semantic_context.fingerprint)
    form = FormIR(
        schema_id=row.semantic_context.fingerprint,
        schema_family_id=family,
        template_lineage_id=str(row.metadata.get("template_id") or "legacy"),
        ontology_lineage_id=family,
        locale=row.semantic_context.locale,
        fields=fields,
        dependencies=[],
    )
    visible = [field.field_id for field in row.semantic_context.fields]
    state = FormState(
        revision=row.semantic_context.revision,
        active_field_id=row.semantic_context.active_field_id,
        values={},
        reachable_field_ids=list(row.semantic_context.reachable_field_ids),
        visible_field_ids=visible,
    )
    action = _action_from_operation(row.target)
    mutating = 0 if action.kind == "no_change" else 1
    coverage = CoverageTags(
        split=str(row.metadata.get("split") or "train"),
        phenomenon=str(row.metadata.get("primitive") or action.kind),
        action_arity=mutating,
        specialization_mode=row.metadata.get("specialization_mode") or "reachable_fields",
        suites=["legacy_phase0"],
        opaque_ids=row.metadata.get("field_identity") == "opaque",
        correction=bool(row.metadata.get("correction")),
        asr=bool(row.metadata.get("asr_noise")),
        field_types=[row.metadata.get("primitive")] if row.metadata.get("primitive") else [],
    )
    provenance = Provenance(
        generator="legacy_phase0",
        generator_version="phase0-generator-1.0",
        seed=int(row.metadata.get("seed") or 0),
        family_seed=0,
        state_seed=0,
        surface_seed=int(row.metadata.get("attempt") or 0),
        template_id=str(row.metadata.get("template_id") or "legacy"),
        template_lineage_id=str(row.metadata.get("template_id") or "legacy"),
        transforms=[],
        parent_schema_id=form.schema_id,
        parent_scenario_id=row.id,
        validator_version=VALIDATOR_VERSION,
        source=str(row.metadata.get("source") or "legacy"),
        legacy_metadata=dict(row.metadata),
        legacy_fingerprint=row.semantic_context.fingerprint,
    )
    return DatasetRecordV2(
        id=row.id,
        schema=form,
        state=state,
        semantic_context=row.semantic_context,
        utterance=row.utterance,
        expected_actions=[action],
        provenance=provenance,
        coverage=coverage,
        schema_fingerprint=schema_fingerprint(form, state),
    )


def to_legacy_row(record: DatasetRecordV2) -> CanonicalRow:
    if record.provenance.legacy_metadata is None:
        raise LegacyAdapterError("record was not converted from a legacy row")
    if len(record.expected_actions) != 1:
        raise LegacyAdapterError("legacy rows contain one operation")
    try:
        target = action_to_operation(record.expected_actions[0], record)
    except CallAdapterError as exc:
        raise LegacyAdapterError(str(exc)) from exc
    return CanonicalRow(
        id=record.id,
        metadata=dict(record.provenance.legacy_metadata),
        semantic_context=record.semantic_context,
        utterance=record.utterance,
        target=target,
    )


def rebuilt_context(record: DatasetRecordV2):
    """Rebuild a SemanticContext from the converted form. Legacy fingerprints stay on the stored context."""
    return FormStateBridge.to_semantic_context(record.schema, record.state, record.coverage.specialization_mode)
