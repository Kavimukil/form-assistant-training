from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field

FieldType = Literal["string", "integer", "number", "boolean", "enum", "date"]
OperationName = Literal["set_string", "set_integer", "set_number", "set_boolean", "set_enum", "set_date", "no_change"]

class EnumOption(BaseModel):
    id: str
    label: str
    aliases: list[str] = Field(default_factory=list)

class SemanticField(BaseModel):
    field_id: str
    label: str
    type: FieldType
    description: str | None = None
    minimum: int | float | str | None = None
    maximum: int | float | str | None = None
    min_length: int | None = None
    max_length: int | None = None
    options: list[EnumOption] = Field(default_factory=list)

class SemanticContext(BaseModel):
    revision: int
    fingerprint: str
    locale: str = "en-IN"
    active_field_id: str | None = None
    reachable_field_ids: list[str]
    fields: list[SemanticField]

class Operation(BaseModel):
    name: OperationName
    arguments: dict[str, Any]

class CanonicalRow(BaseModel):
    id: str
    metadata: dict[str, Any]
    semantic_context: SemanticContext
    utterance: str
    target: Operation
