"""Synthetic FormIR schemas with opaque ids and split-disjoint labels."""
from __future__ import annotations

import hashlib

from form_sft.contracts.form_ir import DependencyRule, FieldSpec, FormIR, OptionSpec

_CONS = "bcdfghjklmnprstvwxyz"
_VOWELS = "aeiou"
_KINDS = ("integer", "number", "boolean", "enum", "string", "date", "multi")


def syllable(n: int) -> str:
    return (
        _CONS[n % 20]
        + _VOWELS[(n // 20) % 5]
        + _CONS[(n // 100) % 20]
        + _VOWELS[(n // 2000) % 5]
        + _CONS[(n // 10000) % 20]
    )


def _digits(seed: int, *parts: str, width: int = 10) -> str:
    raw = hashlib.sha256(":".join((str(seed), *parts)).encode()).hexdigest()
    return f"{int(raw, 16) % (10 ** width):0{width}d}"


def label_base(split: str, family_index: int) -> int:
    offset = {"train": 0, "validation": 20_000, "test": 40_000}[split]
    return offset + family_index * 120


def _bounds(kind: str, family_index: int) -> tuple:
    bucket = family_index % 3
    if kind == "integer":
        return [(0, 12), (10, 99), (100, 999)][bucket]
    if kind == "number":
        return [(0.0, 10.0), (1.0, 50.0), (10.0, 250.0)][bucket]
    if kind == "date":
        return [("2024-01-01", "2026-12-31"), ("2026-01-01", "2028-06-30"), ("2027-01-01", "2029-12-31")][bucket]
    if kind == "string":
        return (3, [12, 16, 24][bucket])
    return (None, None)


def generate_schema(family_seed: int, *, split: str, family_index: int, family_id: str | None = None) -> FormIR:
    base = label_base(split, family_index)
    family = family_id or f"fam-{split}-{family_index:03d}"
    ontology = f"ont-{split}-{family_index:03d}"
    template = f"tpl-{split}"
    schema_id = "sch" + _digits(family_seed, split, str(family_index), "schema")
    fields: list[FieldSpec] = []
    for role, kind in enumerate(_KINDS):
        label = f"{syllable(base + role)} {syllable(base + role + 30)}"
        alias = syllable(base + 60 + role)
        field_id = "fld" + _digits(family_seed, split, str(family_index), str(role))
        options: list[OptionSpec] = []
        minimum = maximum = None
        min_length = max_length = None
        if kind in {"enum", "multi"}:
            for opt_i in range(4):
                options.append(
                    OptionSpec(
                        id="opt" + _digits(family_seed, split, str(family_index), str(role), str(opt_i), width=8),
                        label=syllable(base + 20 + role * 4 + opt_i),
                        aliases=[syllable(base + 70 + role * 4 + opt_i)],
                    )
                )
        elif kind in {"integer", "number", "date"}:
            minimum, maximum = _bounds(kind, family_index)
        elif kind == "string":
            min_length, max_length = _bounds(kind, family_index)
        fields.append(
            FieldSpec(
                field_id=field_id,
                label=label,
                type=kind,
                description=f"Entry for {label}",
                aliases=[alias],
                minimum=minimum,
                maximum=maximum,
                min_length=min_length,
                max_length=max_length,
                options=options,
                ontology_lineage=ontology,
            )
        )
    flag = next(field for field in fields if field.type == "boolean")
    dated = next(field for field in fields if field.type == "date")
    dependencies = [
        DependencyRule(
            rule_id=f"dep-{family}-hide-date",
            when_field_id=flag.field_id,
            when_value=False,
            then_field_id=dated.field_id,
            effect="hide",
        )
    ]
    return FormIR(
        schema_id=schema_id,
        schema_family_id=family,
        template_lineage_id=template,
        ontology_lineage_id=ontology,
        locale="en-IN",
        fields=fields,
        dependencies=dependencies,
    )
