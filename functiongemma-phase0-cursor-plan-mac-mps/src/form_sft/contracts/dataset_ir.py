"""Canonical dataset record shared by generation, QC, and serialization."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from form_sft.contracts.form_ir import FormIR, FormState
from form_sft.contracts.models import SemanticContext

ActionKind = Literal["set", "clear", "no_change"]
AmbiguityStatus = Literal["accepted", "quarantine"]


class CanonicalAction(BaseModel):
    kind: ActionKind
    field_id: str | None = None
    value: Any = None


class Provenance(BaseModel):
    generator: str
    generator_version: str
    seed: int
    family_seed: int
    state_seed: int
    surface_seed: int
    template_id: str
    template_lineage_id: str
    transforms: list[str] = Field(default_factory=list)
    parent_schema_id: str
    parent_scenario_id: str
    validator_version: str
    quarantine_reason: str | None = None
    source: str = "deterministic_template"
    legacy_metadata: dict[str, Any] | None = None
    legacy_fingerprint: str | None = None


class CoverageTags(BaseModel):
    split: str
    phenomenon: str
    action_arity: int
    specialization_mode: Literal["active_field", "reachable_fields"]
    suites: list[str] = Field(default_factory=list)
    opaque_ids: bool = True
    correction: bool = False
    asr: bool = False
    code_mixed: bool = False
    counterfactual: bool = False
    option_permutation: bool = False
    ambiguity: AmbiguityStatus = "accepted"
    field_types: list[str] = Field(default_factory=list)


class DatasetRecordV2(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    id: str
    schema: FormIR
    state: FormState
    semantic_context: SemanticContext
    utterance: str
    expected_actions: list[CanonicalAction]
    provenance: Provenance
    coverage: CoverageTags
    schema_fingerprint: str
    counterfactual_group_id: str | None = None


VALIDATOR_VERSION = "record-validator-1.0"
GENERATOR_VERSION = "universal-generator-1.0"
