from form_sft.dataset.similarity import find_cross_split_leaks, jaccard, shingles
from form_sft.universal.build import build_records
from form_sft.validation.ambiguity import classify_ambiguity
from form_sft.validation.record import apply_actions
from tests.support_universal import tiny_spec


def test_qc_quarantine_replay_and_family_isolation():
    accepted, quarantine = build_records(tiny_spec())
    assert quarantine
    assert all(classify_ambiguity(row) == "quarantine" for row in quarantine)
    assert find_cross_split_leaks(accepted, threshold=0.9) == []
    sample = next(row for row in accepted if row.coverage.phenomenon == "clear")
    post = apply_actions(sample.schema, sample.state, sample.expected_actions)
    assert post.values[sample.expected_actions[0].field_id] is None
    assert post.revision == sample.state.revision + 1
    correction = next(row for row in accepted if row.coverage.correction)
    assert classify_ambiguity(correction) == "accepted"
    leaked = accepted[0].model_copy(deep=True)
    leaked.coverage.split = "test" if leaked.coverage.split != "test" else "train"
    assert any(item["code"] == "L_SCHEMA_FAMILY" for item in find_cross_split_leaks(accepted + [leaked]))


def test_option_permutation_keeps_actions_and_changes_order():
    accepted, _ = build_records(tiny_spec())
    perm = next(row for row in accepted if row.coverage.option_permutation)
    base = next(
        row
        for row in accepted
        if row.coverage.phenomenon == "single_set" and row.utterance == perm.utterance and row.schema.schema_family_id == perm.schema.schema_family_id
    )
    assert base.expected_actions == perm.expected_actions
    base_ids = [opt.id for field in base.schema.fields if field.type == "enum" for opt in field.options]
    perm_ids = [opt.id for field in perm.schema.fields if field.type == "enum" for opt in field.options]
    assert base_ids == list(reversed(perm_ids))
    assert jaccard(shingles(base.utterance), shingles(perm.utterance)) == 1.0


def test_contradictory_correction_is_quarantined():
    accepted, _ = build_records(tiny_spec())
    row = next(item for item in accepted if item.coverage.correction).model_copy(deep=True)
    row.utterance = row.utterance + " but now 4"
    row.expected_actions[0].value = 9
    assert classify_ambiguity(row) == "quarantine"
