import json
from pathlib import Path

from form_sft.adapters.legacy import from_legacy_row, to_legacy_row
from form_sft.contracts.models import CanonicalRow
from form_sft.training.json_prompt import calls_from_row
from form_sft.adapters.calls import canonical_calls
from form_sft.validation.record import validate_record


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_roundtrip_and_call_shape():
    path = ROOT / "generated/phase0-v1/test.jsonl"
    raw = json.loads(path.read_text().splitlines()[0])
    enum = None
    for line in path.read_text().splitlines():
        item = json.loads(line)
        if item["target"]["name"] == "set_enum":
            enum = item
            break
        if item["target"]["name"] == "no_change" and enum is None:
            enum = item
    row = CanonicalRow.model_validate(raw)
    converted = from_legacy_row(row)
    assert validate_record(converted) == []
    assert to_legacy_row(converted).model_dump() == row.model_dump()
    assert canonical_calls(converted, "production_json") == calls_from_row(row)
    if enum and enum["target"]["name"] == "set_enum":
        enum_row = from_legacy_row(CanonicalRow.model_validate(enum))
        calls = canonical_calls(enum_row, "production_json")
        assert calls[0]["name"] == "set_choice"
        assert calls[0]["arguments"]["value"] == enum["target"]["arguments"]["option_id"]
