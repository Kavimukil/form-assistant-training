from enum import Enum
from form_sft.contracts.models import SemanticContext, SemanticField

class SpecializationMode(str, Enum):
    ACTIVE_FIELD="active_field"
    REACHABLE_FIELDS="reachable_fields"

def select_fields(ctx: SemanticContext, mode: SpecializationMode) -> list[SemanticField]:
    by_id={f.field_id:f for f in ctx.fields}
    if len(by_id)!=len(ctx.fields): raise ValueError("duplicate field IDs")
    for fid in ctx.reachable_field_ids:
        if fid not in by_id: raise ValueError(f"unknown reachable field: {fid}")
    if ctx.active_field_id is not None and ctx.active_field_id not in ctx.reachable_field_ids:
        raise ValueError("active field must be reachable")
    if mode is SpecializationMode.ACTIVE_FIELD:
        if not ctx.active_field_id: raise ValueError("active_field_id required")
        return [by_id[ctx.active_field_id]]
    return [by_id[x] for x in ctx.reachable_field_ids]

def specialize(ctx: SemanticContext, mode: SpecializationMode) -> dict:
    branches=[]
    for f in select_fields(ctx, mode):
        if f.type == "enum":
            args={"type":"object","additionalProperties":False,"required":["field_id","option_id"],"properties":{"field_id":{"const":f.field_id},"option_id":{"type":"string","enum":[o.id for o in f.options]}}}
            op="set_enum"
        else:
            op={"string":"set_string","integer":"set_integer","number":"set_number","boolean":"set_boolean","date":"set_date"}[f.type]
            value={"type":{"string":"string","integer":"integer","number":"number","boolean":"boolean","date":"string"}[f.type]}
            if f.type=="date": value["format"]="date"
            if f.type in {"integer","number"}:
                if f.minimum is not None: value["minimum"]=f.minimum
                if f.maximum is not None: value["maximum"]=f.maximum
            args={"type":"object","additionalProperties":False,"required":["field_id","value"],"properties":{"field_id":{"const":f.field_id},"value":value}}
        branches.append({"type":"object","additionalProperties":False,"required":["name","arguments"],"properties":{"name":{"const":op},"arguments":args}})
    branches.append({"type":"object","additionalProperties":False,"required":["name","arguments"],"properties":{"name":{"const":"no_change"},"arguments":{"type":"object","additionalProperties":False,"maxProperties":0}}})
    return {"$schema":"https://json-schema.org/draft/2020-12/schema","oneOf":branches}
