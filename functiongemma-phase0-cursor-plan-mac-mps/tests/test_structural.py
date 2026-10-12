from form_sft.contracts.models import CanonicalRow, Operation
from form_sft.validation.structural import validate_structural
from tests.test_validators_g1 import ctx

MD = dict(experiment_id="e", split="train", primitive="integer", specialization_mode="active_field", reachable_bucket=None,
          correction=False, correction_subtype=None, asr_noise=False, asr_subtype=None, source="deterministic_template",
          field_identity="semantic", difficulty="D1", test_partition="train", families={}, subject_field_id="age")


def mk(name, args, utt="twenty", md=None, c=None):
    return CanonicalRow(id="r", metadata=md if md is not None else dict(MD), semantic_context=c or ctx(), utterance=utt, target=Operation(name=name, arguments=args))


def test_valid_row_passes():
    assert validate_structural(mk("set_integer", {"field_id": "age", "value": 20})) == []

def test_extra_argument_rejected():
    assert "S_TARGET_SCHEMA_VIOLATION" in validate_structural(mk("set_integer", {"field_id": "age", "value": 20, "x": 1}))

def test_wrong_value_type_rejected():
    assert "S_TARGET_SCHEMA_VIOLATION" in validate_structural(mk("set_integer", {"field_id": "age", "value": "20"}))

def test_non_active_field_rejected_in_active_mode():
    assert "S_TARGET_SCHEMA_VIOLATION" in validate_structural(mk("set_string", {"field_id": "name", "value": "A"}))

def test_no_change_with_args_rejected():
    assert validate_structural(mk("no_change", {"field_id": "age"}))

def test_missing_metadata_and_empty_utterance():
    md = dict(MD); md.pop("difficulty")
    assert any(e.startswith("S_MISSING_METADATA") for e in validate_structural(mk("no_change", {}, md=md)))
    assert "S_EMPTY_UTTERANCE" in validate_structural(mk("no_change", {}, utt=" "))

def test_sentinel_leak_rejected():
    assert "S_SENTINEL_LEAK" in validate_structural(mk("no_change", {}, utt="a \u27e6b\u27e7"))

def test_bad_date_format_rejected():
    md = dict(MD, specialization_mode="reachable_fields")
    r = mk("set_date", {"field_id": "dob", "value": "2024-13-45"}, md=md)
    assert "S_TARGET_SCHEMA_VIOLATION" in validate_structural(r)
