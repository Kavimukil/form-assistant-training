import json
from pathlib import Path

from form_sft.evaluation.set_choice_slice import set_choice_bucket, slice_set_choice


def _row(exact, gold_calls, pred):
    return {"id": "r", "exact_call": exact, "gold": {"function_calls": gold_calls}, "pred": pred}


def _choice(field_id, option_id):
    return {"name": "set_choice", "arguments": {"field_id": field_id, "value": option_id}}


def test_buckets_cover_the_three_miss_types():
    assert set_choice_bucket(_row(True, [_choice("f", "a")], {"function_calls": [_choice("f", "a")]})) is None
    assert set_choice_bucket(_row(False, [_choice("f", "a")], None)) == "parse_failure"
    assert set_choice_bucket(_row(False, [_choice("f", "a")], {"function_calls": [_choice("other", "a")]})) == "wrong_field"
    assert (
        set_choice_bucket(_row(False, [_choice("f", "a")], {"function_calls": [_choice("f", "b")]}))
        == "right_field_wrong_option_id"
    )


def test_slice_counts_only_inexact_set_choice_rows(tmp_path: Path):
    path = tmp_path / "predictions.jsonl"
    rows = [
        _row(True, [_choice("f", "a")], {"function_calls": [_choice("f", "a")]}),
        _row(False, [_choice("f", "a")], {"function_calls": [_choice("f", "b")]}),
        _row(False, [_choice("f", "a")], "not json"),
        _row(False, [{"name": "set_string", "arguments": {"field_id": "s", "value": "x"}}], None),
    ]
    rows[0]["id"] = "exact"
    rows[1]["id"] = "option"
    rows[2]["id"] = "parse"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = slice_set_choice(path)
    assert report["set_choice_rows"] == 3
    assert report["exact"] == 1
    assert report["right_field_wrong_option_id"] == 1
    assert report["parse_failure"] == 1
    assert report["wrong_field"] == 0
    assert report["examples"]["right_field_wrong_option_id"][0]["id"] == "option"
