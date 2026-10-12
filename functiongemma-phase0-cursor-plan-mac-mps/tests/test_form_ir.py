from form_sft.contracts.form_ir import DependencyRule, FieldSpec, FormIR, FormState, OptionSpec, schema_fingerprint
from form_sft.validation.record import dependency_cycle, validate_record


def _form():
    fields = [
        FieldSpec(field_id="fld0000000001", label="bano keli", type="integer", minimum=0, maximum=10, ontology_lineage="ont"),
        FieldSpec(
            field_id="fld0000000002",
            label="ceno meli",
            type="enum",
            ontology_lineage="ont",
            options=[OptionSpec(id="opt00000001", label="pira"), OptionSpec(id="opt00000002", label="sora")],
        ),
    ]
    return FormIR(
        schema_id="sch0000000001",
        schema_family_id="fam-test-000",
        template_lineage_id="tpl-test",
        ontology_lineage_id="ont-test-000",
        fields=fields,
    )


def test_fingerprint_ignores_labels():
    form = _form()
    state = FormState(revision=1, reachable_field_ids=["fld0000000001"], visible_field_ids=["fld0000000001"], values={"fld0000000001": 3})
    renamed = form.model_copy(deep=True)
    renamed.fields[0].label = "completely different label"
    assert schema_fingerprint(form, state) == schema_fingerprint(renamed, state)


def test_duplicate_option_and_cycle_codes():
    form = _form()
    form.fields[1].options.append(OptionSpec(id="opt00000001", label="extra"))
    form.dependencies = [
        DependencyRule(rule_id="a", when_field_id="fld0000000001", when_value=1, then_field_id="fld0000000002", effect="require"),
        DependencyRule(rule_id="b", when_field_id="fld0000000002", when_value="opt00000001", then_field_id="fld0000000001", effect="require"),
    ]
    assert dependency_cycle(form)
    state = FormState(revision=1, active_field_id="missing", reachable_field_ids=["fld0000000001"], visible_field_ids=["fld0000000001"])
    from form_sft.contracts.dataset_ir import VALIDATOR_VERSION, CanonicalAction, CoverageTags, DatasetRecordV2, Provenance
    from form_sft.contracts.models import SemanticContext

    record = DatasetRecordV2(
        id="r",
        schema=form,
        state=state,
        semantic_context=SemanticContext(revision=1, fingerprint=schema_fingerprint(form, state), reachable_field_ids=["fld0000000001"], fields=[]),
        utterance="assign bano keli as 3",
        expected_actions=[CanonicalAction(kind="set", field_id="fld0000000001", value=3)],
        provenance=Provenance(
            generator="t",
            generator_version="1",
            seed=1,
            family_seed=1,
            state_seed=1,
            surface_seed=1,
            template_id="t",
            template_lineage_id="tpl-test",
            parent_schema_id=form.schema_id,
            parent_scenario_id="r",
            validator_version=VALIDATOR_VERSION,
        ),
        coverage=CoverageTags(split="test", phenomenon="single_set", action_arity=1, specialization_mode="reachable_fields"),
        schema_fingerprint=schema_fingerprint(form, state),
    )
    codes = {issue.code for issue in validate_record(record)}
    assert "V_DUPLICATE_OPTION_ID" in codes
    assert "V_DEPENDENCY_CYCLE" in codes
    assert "V_ACTIVE_NOT_REACHABLE" in codes
