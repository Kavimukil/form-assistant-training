"""Structural (JSON-Schema) validation of canonical rows. Fail closed.

Complements ``validation.semantic.validate_row``: the semantic validator checks
meaning (field IDs, op/type, enum membership, bounds); this module checks that the
row is well-formed and that its target validates against the *specialized* runtime
schema for the row's declared specialization mode (exactly the schema a constrained
decoder would see).
"""
from __future__ import annotations

from jsonschema import Draft202012Validator, FormatChecker

from form_sft.contracts.models import CanonicalRow
from form_sft.specialization.runtime import SpecializationMode, specialize

REQUIRED_METADATA = (
    "experiment_id", "split", "primitive", "specialization_mode", "reachable_bucket", "correction",
    "correction_subtype", "asr_noise", "asr_subtype", "source", "field_identity", "difficulty",
    "test_partition", "families",
)
_FORMATS = FormatChecker()
_SENTINELS = "\u27e6\u27e7\u27ea\u27eb"


def validate_structural(row: CanonicalRow) -> list[str]:
    errors: list[str] = []
    if not row.id or not isinstance(row.id, str):
        errors.append("S_MISSING_ID")
    if not row.utterance or not row.utterance.strip():
        errors.append("S_EMPTY_UTTERANCE")
    elif any(c in row.utterance for c in _SENTINELS):
        errors.append("S_SENTINEL_LEAK")
    for k in REQUIRED_METADATA:
        if k not in row.metadata:
            errors.append(f"S_MISSING_METADATA:{k}")
    if errors:
        return errors

    ctx = row.semantic_context
    if len(set(ctx.reachable_field_ids)) != len(ctx.reachable_field_ids):
        errors.append("S_DUPLICATE_REACHABLE_ID")
    mode_name = row.metadata["specialization_mode"]
    try:
        mode = SpecializationMode(mode_name)
    except ValueError:
        return errors + ["S_UNKNOWN_SPECIALIZATION_MODE"]
    try:
        schema = specialize(ctx, mode)
    except ValueError:
        return errors + ["S_CONTEXT_INVALID"]
    if mode is SpecializationMode.ACTIVE_FIELD and ctx.active_field_id != row.metadata.get("subject_field_id", ctx.active_field_id):
        errors.append("S_ACTIVE_FIELD_MISMATCH")
    validator = Draft202012Validator(schema, format_checker=_FORMATS)
    target = row.target.model_dump()
    if not validator.is_valid(target):
        errors.append("S_TARGET_SCHEMA_VIOLATION")
    if row.target.name == "no_change" and row.target.arguments:
        errors.append("S_NO_CHANGE_ARGUMENTS")
    return errors
