"""G1: contract + validator tests (one per exit criterion)."""
import pytest
from form_sft.contracts.models import CanonicalRow, EnumOption, Operation, SemanticContext, SemanticField
from form_sft.specialization.runtime import SpecializationMode, select_fields, specialize
from form_sft.validation.semantic import validate_row


def ctx(**kw):
    base = dict(revision=1, fingerprint="sha256:t", active_field_id="age", reachable_field_ids=["age", "plan", "name"], fields=[
        SemanticField(field_id="age", label="Age", type="integer", minimum=0, maximum=120),
        SemanticField(field_id="plan", label="Plan", type="enum", options=[EnumOption(id="basic", label="Basic"), EnumOption(id="premium", label="Premium")]),
        SemanticField(field_id="name", label="Name", type="string"),
        SemanticField(field_id="ok", label="Ok", type="boolean"),
        SemanticField(field_id="dob", label="DOB", type="date", minimum="2000-01-01", maximum="2030-01-01"),
        SemanticField(field_id="w", label="W", type="number", minimum=1.0, maximum=9.5),
    ])
    base.update(kw)
    return SemanticContext(**base)


def row(name, args, c=None):
    return CanonicalRow(id="x", metadata={}, semantic_context=c or ctx(), utterance="u", target=Operation(name=name, arguments=args))


@pytest.mark.parametrize("name,args", [
    ("set_integer", {"field_id": "age", "value": 5}), ("set_number", {"field_id": "w", "value": 2.5}),
    ("set_boolean", {"field_id": "ok", "value": True}), ("set_enum", {"field_id": "plan", "option_id": "basic"}),
    ("set_string", {"field_id": "name", "value": "A"}), ("set_date", {"field_id": "dob", "value": "2024-02-29"}),
    ("no_change", {})])
def test_valid_operation_per_primitive(name, args):
    assert validate_row(row(name, args)) == []


def test_unknown_field():
    assert "E_UNKNOWN_FIELD_ID" in validate_row(row("set_integer", {"field_id": "zzz", "value": 1}))

def test_wrong_op():
    assert "E_OPERATION_TYPE_MISMATCH" in validate_row(row("set_string", {"field_id": "age", "value": "x"}))

def test_enum_hallucination():
    assert "E_UNKNOWN_OPTION_ID" in validate_row(row("set_enum", {"field_id": "plan", "option_id": "gold"}))

def test_duplicate_field_and_option():
    c = ctx(fields=[SemanticField(field_id="a", label="A", type="integer"), SemanticField(field_id="a", label="A", type="integer")], reachable_field_ids=["a"], active_field_id="a")
    assert "E_DUPLICATE_FIELD_ID" in validate_row(row("no_change", {}, c))
    c = ctx(fields=[SemanticField(field_id="a", label="A", type="enum", options=[EnumOption(id="o", label="1"), EnumOption(id="o", label="2")])], reachable_field_ids=["a"], active_field_id="a")
    assert "E_DUPLICATE_OPTION_ID" in validate_row(row("no_change", {}, c))

def test_bounds():
    assert "E_VALUE_ABOVE_MAXIMUM" in validate_row(row("set_integer", {"field_id": "age", "value": 121}))
    assert "E_VALUE_BELOW_MINIMUM" in validate_row(row("set_number", {"field_id": "w", "value": 0.5}))
    assert "E_DATE_AFTER_MAXIMUM" in validate_row(row("set_date", {"field_id": "dob", "value": "2031-01-01"}))
    assert "E_INVALID_DATE" in validate_row(row("set_date", {"field_id": "dob", "value": "2024-13-01"}))

def test_active_must_be_reachable():
    with pytest.raises(ValueError):
        specialize(ctx(active_field_id="ok"), SpecializationMode.ACTIVE_FIELD)

def test_unknown_reachable_and_duplicate_ids_fail_closed():
    with pytest.raises(ValueError):
        select_fields(ctx(reachable_field_ids=["age", "nope"]), SpecializationMode.REACHABLE_FIELDS)
    dup = ctx(fields=[SemanticField(field_id="a", label="A", type="integer")] * 2, reachable_field_ids=["a"], active_field_id="a")
    with pytest.raises(ValueError):
        select_fields(dup, SpecializationMode.ACTIVE_FIELD)

def test_specialized_schema_exposes_only_legal_options():
    s = specialize(ctx(), SpecializationMode.REACHABLE_FIELDS)
    names = [b["properties"]["arguments"].get("properties", {}).get("field_id", {}).get("const") for b in s["oneOf"]]
    assert names == ["age", "plan", "name", None]
    enum_branch = s["oneOf"][1]["properties"]["arguments"]["properties"]["option_id"]["enum"]
    assert enum_branch == ["basic", "premium"]
