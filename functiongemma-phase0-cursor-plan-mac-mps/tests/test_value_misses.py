import json
from pathlib import Path

from form_sft.evaluation.value_misses import classify_files, classify_joined
from form_sft.universal.build import train_single_set_type


def _field(field_id, kind, options=None):
    return {
        "field_id": field_id,
        "type": kind,
        "label": f"label-{field_id}",
        "aliases": [f"alias-{field_id}"],
        "options": options or [],
    }


def _record():
    return {
        "id": "row-1",
        "schema": {
            "fields": [
                _field(
                    "enum-field",
                    "enum",
                    [
                        {"id": "opt-a", "label": "red", "aliases": ["scarlet"]},
                        {"id": "opt-b", "label": "blue", "aliases": ["navy"]},
                    ],
                ),
                _field("string-field", "string"),
                _field("other-field", "string"),
            ]
        },
        "state": {"values": {"string-field": "kept", "other-field": "nearby", "enum-field": "opt-b"}},
    }


def _pred(gold, pred):
    return {"id": "row-1", "gold": {"function_calls": gold}, "pred": {"function_calls": pred}}


def test_choice_classes_cover_label_alias_and_other_id():
    record = _record()
    gold = [{"name": "set_choice", "arguments": {"field_id": "enum-field", "value": "opt-a"}}]
    label = classify_joined(_pred(gold, [{"name": "set_choice", "arguments": {"field_id": "enum-field", "value": "red"}}]), record)
    alias = classify_joined(_pred(gold, [{"name": "set_choice", "arguments": {"field_id": "enum-field", "value": "scarlet"}}]), record)
    other = classify_joined(_pred(gold, [{"name": "set_choice", "arguments": {"field_id": "enum-field", "value": "opt-b"}}]), record)
    wrong = classify_joined(_pred(gold, [{"name": "set_choice", "arguments": {"field_id": "other", "value": "opt-a"}}]), record)
    assert [item[0]["class"] for item in (label, alias, other, wrong)] == [
        "spoken_label",
        "option_alias",
        "other_option_id",
        "wrong_field",
    ]


def test_string_classes_and_file_gate(tmp_path: Path):
    record = _record()
    gold = [{"name": "set_string", "arguments": {"field_id": "string-field", "value": "spoken"}}]
    current = classify_joined(
        _pred(gold, [{"name": "set_string", "arguments": {"field_id": "string-field", "value": "kept"}}]),
        record,
    )
    other_field = classify_joined(
        _pred(gold, [{"name": "set_string", "arguments": {"field_id": "string-field", "value": "nearby"}}]),
        record,
    )
    syllable = classify_joined(
        _pred(gold, [{"name": "set_string", "arguments": {"field_id": "string-field", "value": "red"}}]),
        record,
    )
    assert [item[0]["class"] for item in (current, other_field, syllable)] == [
        "current_state",
        "other_field_value",
        "other_syllable",
    ]
    predictions = tmp_path / "predictions.jsonl"
    dataset = tmp_path / "test.jsonl"
    dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")
    rows = [
        _pred(gold, [{"name": "set_string", "arguments": {"field_id": "string-field", "value": "kept"}}]),
        _pred(
            [{"name": "set_choice", "arguments": {"field_id": "enum-field", "value": "opt-a"}}],
            [{"name": "set_choice", "arguments": {"field_id": "enum-field", "value": "red"}}],
        ),
    ]
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = classify_files(predictions, dataset)
    assert report["proceed"] is True
    assert report["set_string"]["current_state"] == 1
    assert report["set_choice"]["spoken_label"] == 1


def test_string_force_sits_before_the_enum_tail():
    assert train_single_set_type(8899, 8900, 1200, 1000) == "enum"
    assert train_single_set_type(7700, 8900, 1200, 1000) == "enum"
    assert train_single_set_type(7699, 8900, 1200, 1000) == "string"
    assert train_single_set_type(6700, 8900, 1200, 1000) == "string"
    assert train_single_set_type(6699, 8900, 1200, 1000) is None
