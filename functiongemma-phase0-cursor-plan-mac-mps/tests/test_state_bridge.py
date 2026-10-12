from form_sft.bridges.form_state import production_context
from form_sft.training.json_prompt import context_from_row
from form_sft.universal.build import build_records
from tests.support_universal import tiny_spec


def test_production_context_matches_legacy_helper_for_scalar_fields():
    accepted, _ = build_records(tiny_spec())
    row = next(record for record in accepted if record.coverage.phenomenon == "single_set" and record.coverage.field_types == ["integer"])
    produced = production_context(row)
    from form_sft.contracts.models import CanonicalRow, Operation

    legacy = CanonicalRow(
        id=row.id,
        metadata={"specialization_mode": row.coverage.specialization_mode},
        semantic_context=row.semantic_context,
        utterance=row.utterance,
        target=Operation(name="no_change", arguments={}),
    )
    assert produced == context_from_row(legacy)
    assert "label" in produced["fields"][0]
    assert "description" not in produced["fields"][0]
