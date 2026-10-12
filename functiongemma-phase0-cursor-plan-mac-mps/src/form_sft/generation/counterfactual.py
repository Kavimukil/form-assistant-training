"""Counterfactual schema-context pairs that share an utterance and diverge in the action."""
from __future__ import annotations

from form_sft.bridges.form_state import FormStateBridge
from form_sft.contracts.dataset_ir import VALIDATOR_VERSION, CanonicalAction, CoverageTags, DatasetRecordV2, Provenance
from form_sft.contracts.form_ir import FormIR, FormState, schema_fingerprint
from form_sft.generation.scenario_factory import ScenarioTruth, realize_scenario, surface_for


def _option_by_label(form: FormIR, label: str):
    for field in form.fields:
        for opt in field.options:
            if opt.label == label:
                return field, opt
    raise ValueError(f"alternate form has no option labeled {label}")


def make_counterfactual(record: DatasetRecordV2, alternate_form: FormIR) -> DatasetRecordV2:
    """Keep the utterance and retarget it using an option label that exists on ``alternate_form``."""
    source_action = next(action for action in record.expected_actions if action.kind == "set")
    source_field = next(field for field in record.schema.fields if field.field_id == source_action.field_id)
    label = surface_for(source_field, source_action.value)
    field, option = _option_by_label(alternate_form, label)
    state = FormState(
        revision=record.state.revision,
        active_field_id=None,
        values={item.field_id: None for item in alternate_form.fields},
        reachable_field_ids=[item.field_id for item in alternate_form.fields],
        visible_field_ids=[item.field_id for item in alternate_form.fields],
    )
    # Fill legal placeholder values so replay accepts the pre-state, then set the action.
    from form_sft.generation.scenario_factory import _sample_value, _rng

    rng = _rng(record.provenance.state_seed, "cf", alternate_form.schema_id)
    for item in alternate_form.fields:
        if item.type == "boolean":
            state.values[item.field_id] = True
        elif item.field_id == field.field_id:
            state.values[item.field_id] = _sample_value(item, rng, avoid=option.id)
        else:
            state.values[item.field_id] = _sample_value(item, rng)
    action = CanonicalAction(kind="set", field_id=field.field_id, value=option.id)
    group = record.counterfactual_group_id or f"cf-{record.id}"
    coverage = record.coverage.model_copy(
        deep=True,
        update={
            "phenomenon": "counterfactual",
            "action_arity": 1,
            "specialization_mode": "reachable_fields",
            "counterfactual": True,
            "field_types": [field.type],
            "ambiguity": "accepted",
        },
    )
    provenance = record.provenance.model_copy(
        deep=True,
        update={
            "parent_schema_id": alternate_form.schema_id,
            "parent_scenario_id": record.provenance.parent_scenario_id + "-cf",
            "transforms": list(record.provenance.transforms) + ["counterfactual_context"],
            "validator_version": VALIDATOR_VERSION,
        },
    )
    context = FormStateBridge.to_semantic_context(alternate_form, state, "reachable_fields")
    paired = DatasetRecordV2(
        id=record.id + "-cf",
        schema=alternate_form,
        state=state,
        semantic_context=context,
        utterance=record.utterance,
        expected_actions=[action],
        provenance=provenance,
        coverage=coverage,
        schema_fingerprint=schema_fingerprint(alternate_form, state),
        counterfactual_group_id=group,
    )
    return paired


def counterfactual_truth(base: ScenarioTruth, alternate_form: FormIR, group_id: str) -> tuple[str, ScenarioTruth]:
    """Build the paired truth after the shared utterance is fixed."""
    utterance = realize_scenario(base, base.form.schema_id.__hash__())
    return utterance, base.model_copy(update={"counterfactual": True}), group_id
