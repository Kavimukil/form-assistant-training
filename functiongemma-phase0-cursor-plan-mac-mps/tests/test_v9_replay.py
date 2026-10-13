import json
from pathlib import Path

import pytest

from form_sft.universal.build import UniversalBuildError, write_v9_dataset
from form_sft.universal.v9_replay import generate_copy_drills, select_replay_ids


def _row(row_id, phenomenon, field_type="string"):
    return {"id": row_id, "coverage": {"phenomenon": phenomenon, "field_types": [field_type]}}


def test_replay_keeps_every_protected_row():
    rows = [_row(f"nc-{i}", "no_change") for i in range(3)]
    rows += [_row(f"perm-{i}", "option_permutation", "enum") for i in range(2)]
    rows += [_row("cf-0", "counterfactual", "enum")]
    rows += [_row(f"set-{i}", "single_set", "integer" if i % 2 else "string") for i in range(6)]
    chosen = set(select_replay_ids(rows, keep=8))
    assert len(chosen) == 8
    assert {row["id"] for row in rows if row["coverage"]["phenomenon"] != "single_set"} <= chosen


def test_prediction_gate_refuses_to_materialize(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps(_row("a", "no_change")) + "\n", encoding="utf-8")
    with pytest.raises(UniversalBuildError, match="prediction gate"):
        write_v9_dataset(train, tmp_path, tmp_path / "out", {"predictions_present": False, "proceed": False}, keep=1)


def test_copy_drills_keep_the_target_id_and_spoken_string():
    forbidden = set()
    seen = set()
    strings = generate_copy_drills(
        seed=11, kind="string", count=2, use_option_alias=False, seen=seen, forbidden=forbidden, family_count=2
    )
    choices = generate_copy_drills(
        seed=11, kind="enum", count=2, use_option_alias=True, seen=seen, forbidden=forbidden, family_count=2
    )
    for record in strings:
        value = record.expected_actions[0].value
        assert value in record.utterance
    for record in choices:
        action = record.expected_actions[0]
        field = next(item for item in record.schema.fields if item.field_id == action.field_id)
        option = next(item for item in field.options if item.id == action.value)
        assert action.value.startswith("opt")
        assert option.label in record.utterance or option.aliases[0] in record.utterance
        assert action.value not in record.utterance
