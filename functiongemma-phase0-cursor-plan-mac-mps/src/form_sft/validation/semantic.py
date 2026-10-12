"""Reference semantic validator. Keep model-independent and fail closed."""
from __future__ import annotations
from datetime import date
from form_sft.contracts.models import CanonicalRow

OP_FOR_TYPE = {"string":"set_string","integer":"set_integer","number":"set_number","boolean":"set_boolean","enum":"set_enum","date":"set_date"}

def validate_row(row: CanonicalRow) -> list[str]:
    errors: list[str] = []
    fields = row.semantic_context.fields
    ids = [f.field_id for f in fields]
    if len(ids) != len(set(ids)): errors.append("E_DUPLICATE_FIELD_ID")
    by_id = {f.field_id:f for f in fields}
    for f in fields:
        if f.type == "enum":
            opts=[o.id for o in f.options]
            if len(opts)!=len(set(opts)): errors.append("E_DUPLICATE_OPTION_ID")
    if row.target.name == "no_change":
        if row.target.arguments: errors.append("E_NO_CHANGE_ARGUMENTS")
        return errors
    fid=row.target.arguments.get("field_id")
    if fid not in by_id:
        errors.append("E_UNKNOWN_FIELD_ID"); return errors
    f=by_id[fid]
    if OP_FOR_TYPE[f.type] != row.target.name:
        errors.append("E_OPERATION_TYPE_MISMATCH"); return errors
    if f.type == "enum":
        if row.target.arguments.get("option_id") not in {o.id for o in f.options}: errors.append("E_UNKNOWN_OPTION_ID")
    if f.type in {"integer","number"}:
        v=row.target.arguments.get("value")
        if f.minimum is not None and v < f.minimum: errors.append("E_VALUE_BELOW_MINIMUM")
        if f.maximum is not None and v > f.maximum: errors.append("E_VALUE_ABOVE_MAXIMUM")
    if f.type == "date":
        try:
            v=date.fromisoformat(row.target.arguments["value"])
            if f.minimum and v < date.fromisoformat(str(f.minimum)): errors.append("E_DATE_BEFORE_MINIMUM")
            if f.maximum and v > date.fromisoformat(str(f.maximum)): errors.append("E_DATE_AFTER_MAXIMUM")
        except (ValueError, KeyError): errors.append("E_INVALID_DATE")
    return errors
