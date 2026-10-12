from form_sft.contracts.models import *
from form_sft.specialization.runtime import *
from form_sft.validation.semantic import validate_row

def ctx():
    return SemanticContext(revision=1,fingerprint="sha256:test",locale="en-IN",active_field_id="age",reachable_field_ids=["age","plan"],fields=[SemanticField(field_id="age",label="Age",type="integer",minimum=0,maximum=120),SemanticField(field_id="plan",label="Plan",type="enum",options=[EnumOption(id="basic",label="Basic"),EnumOption(id="premium",label="Premium")])])

def test_active_specialization():
    s=specialize(ctx(),SpecializationMode.ACTIVE_FIELD)
    assert len(s["oneOf"])==2
    assert s["oneOf"][0]["properties"]["arguments"]["properties"]["field_id"]["const"]=="age"

def test_valid_integer():
    r=CanonicalRow(id="x",metadata={},semantic_context=ctx(),utterance="twenty nine",target=Operation(name="set_integer",arguments={"field_id":"age","value":29}))
    assert validate_row(r)==[]

def test_enum_hallucination():
    r=CanonicalRow(id="x",metadata={},semantic_context=ctx(),utterance="gold",target=Operation(name="set_enum",arguments={"field_id":"plan","option_id":"gold"}))
    assert "E_UNKNOWN_OPTION_ID" in validate_row(r)
