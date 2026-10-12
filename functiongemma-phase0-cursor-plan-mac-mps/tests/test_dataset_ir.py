from form_sft.adapters.calls import canonical_calls
from form_sft.universal.build import build_records
from tests.support_universal import tiny_spec


def test_multi_action_order_invariant_and_unknown_field():
    accepted, _ = build_records(tiny_spec())
    multi = next(row for row in accepted if row.coverage.phenomenon == "multi_field")
    calls = canonical_calls(multi, "production_json")
    assert len(calls) == multi.coverage.action_arity
    flipped = multi.model_copy(deep=True)
    flipped.expected_actions = list(reversed(multi.expected_actions))
    assert sorted(canonical_calls(flipped, "production_json"), key=str) == sorted(calls, key=str)
    broken = multi.model_copy(deep=True)
    broken.expected_actions[0].field_id = "missing"
    from form_sft.validation.record import validate_record

    assert any(issue.code == "V_ACTION_UNKNOWN_FIELD" for issue in validate_record(broken))
