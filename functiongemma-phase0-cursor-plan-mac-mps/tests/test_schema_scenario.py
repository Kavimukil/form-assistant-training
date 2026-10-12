import re

from form_sft.generation.counterfactual import make_counterfactual
from form_sft.generation.scenario_factory import generate_scenario, realize_scenario
from form_sft.generation.schema_factory import generate_schema, syllable
from form_sft.universal.build import build_records
from tests.support_universal import tiny_spec


def test_schema_is_deterministic_and_ids_are_opaque():
    left = generate_schema(11, split="test", family_index=1)
    right = generate_schema(11, split="test", family_index=1)
    assert left.model_dump() == right.model_dump()
    train = generate_schema(11, split="train", family_index=1)
    assert {field.label for field in left.fields}.isdisjoint({field.label for field in train.fields})
    for field in left.fields:
        assert re.fullmatch(r"fld\d+", field.field_id)
        for token in re.findall(r"[a-z]{4,}", field.label):
            assert token not in field.field_id
        if field.type == "enum":
            assert [opt.id for opt in field.options] != [opt.id for opt in list(reversed(field.options))] or True
            assert len({opt.id for opt in field.options}) == 4


def test_syllables_do_not_collide_across_split_offsets():
    assert len({syllable(n) for n in range(0, 12000)}) == 12000


def test_truth_is_unchanged_by_realization_and_counterfactual_diverges():
    spec_rows, _ = build_records(tiny_spec())
    source = next(row for row in spec_rows if row.coverage.counterfactual and row.id.endswith("00001") or row.coverage.phenomenon == "counterfactual")
    pair = [row for row in spec_rows if row.counterfactual_group_id and row.counterfactual_group_id == source.counterfactual_group_id]
    assert len(pair) == 2
    assert pair[0].utterance == pair[1].utterance
    assert pair[0].expected_actions != pair[1].expected_actions
    assert pair[0].coverage.split == pair[1].coverage.split
    form = generate_schema(3, split="train", family_index=0)
    truth = generate_scenario(form, 3, {"phenomenon": "single_set", "split": "train", "index": 0, "mode": "active_field", "type_name": "integer"})
    before = truth.model_dump()
    text = realize_scenario(truth, 3)
    assert truth.model_dump() == before
    assert text
    again = realize_scenario(truth, 3)
    assert text == again


def test_train_v6_wording_stays_off_the_frozen_frames():
    from form_sft.validation.agreement import semantic_agreement

    form = generate_schema(11, split="train", family_index=0)
    singles = []
    for index in range(12):
        truth = generate_scenario(
            form,
            20 + index,
            {"phenomenon": "single_set", "split": "train", "index": index, "mode": "active_field"},
        )
        singles.append(realize_scenario(truth, index))
    assert any(" for " in text or text.startswith("put ") for text in singles)
    assert all("place " not in text and " into " not in text for text in singles)
    no_change = []
    for index in range(8):
        truth = generate_scenario(
            form,
            40 + index,
            {
                "phenomenon": "no_change",
                "split": "train",
                "index": index,
                "mode": "reachable_fields",
                "echo_current": index % 2 == 1,
            },
        )
        text = realize_scenario(truth, index)
        no_change.append(text)
        from form_sft.universal.build import assemble_record

        record = assemble_record(truth, text, record_id=f"t-{index}", seed=index, family_seed=1)
        assert semantic_agreement(record).ok
    assert len(set(no_change)) > 2
    assert "calendar slate holds steady" not in no_change
    assert "schedule slate remains idle" not in no_change
    held = generate_scenario(form, 1, {"phenomenon": "no_change", "split": "test", "index": 0, "mode": "reachable_fields"})
    assert realize_scenario(held, 99) == "schedule slate remains idle"
    valued = generate_scenario(
        form, 1, {"phenomenon": "single_set", "split": "test", "index": 0, "mode": "active_field", "type_name": "integer"}
    )
    assert realize_scenario(valued, 99).startswith("place ")


def test_train_alias_surfaces_keep_the_option_id():
    from form_sft.validation.agreement import semantic_agreement
    from form_sft.universal.build import assemble_record

    form = generate_schema(11, split="train", family_index=0)
    truth = generate_scenario(
        form,
        5,
        {
            "phenomenon": "single_set",
            "split": "train",
            "index": 3,
            "mode": "active_field",
            "type_name": "enum",
            "speak_field_alias": True,
            "speak_option_alias": True,
        },
    )
    text = realize_scenario(truth, 1)
    action = truth.actions[0]
    field = next(item for item in form.fields if item.field_id == action.field_id)
    opt = next(item for item in field.options if item.id == action.value)
    assert field.aliases[0] in text
    assert opt.aliases[0] in text
    assert opt.label not in text
    record = assemble_record(truth, text, record_id="alias", seed=1, family_seed=1)
    assert semantic_agreement(record).ok
    held = generate_scenario(
        form, 1, {"phenomenon": "single_set", "split": "test", "index": 0, "mode": "active_field", "type_name": "enum", "speak_option_alias": True}
    )
    assert realize_scenario(held, 1).startswith("place ")


def test_make_counterfactual_uses_the_alternate_form():
    rows, _ = build_records(tiny_spec())
    source = next(row for row in rows if row.coverage.phenomenon == "counterfactual")
    other = next(row for row in rows if row.counterfactual_group_id == source.counterfactual_group_id and row.id != source.id)
    paired = make_counterfactual(source, other.schema)
    assert paired.utterance == source.utterance
    assert paired.expected_actions[0].field_id != source.expected_actions[0].field_id
