"""Holdout family registry (G5 split isolation).

A *family* is a generalisation axis. Every slot is assigned one family id per axis
BEFORE any utterance is realised. Holdout family ids may only ever be used by test
rows whose ``test_partition`` is the matching ``unseen_*`` partition.

axis        general ids                         holdout id(s)                  partition
field       dom:<general domain>                dom:agri, dom:logistics        unseen_fields
template    tpl0..tpl4                          tpl5                           unseen_templates
numeric     numf_general | none                 numf_holdout                   unseen_numeric_compositions
enum        enum_general | none                 enum_holdout                   unseen_enums
correction  corr_v0, corr_v1 | none             corr_v2                        unseen_corrections
"""
from __future__ import annotations

from form_sft.generation.catalog import (
    ENUM_HOLDOUT_DEFS, GENERAL_DOMAINS, HOLDOUT_DOMAIN_DEFS, HOLDOUT_DOMAINS, all_field_ids,
)

AXES = ("field", "template", "numeric", "enum", "correction")

PARTITION_AXIS = {
    "unseen_fields": "field",
    "unseen_templates": "template",
    "unseen_numeric_compositions": "numeric",
    "unseen_enums": "enum",
    "unseen_corrections": "correction",
}
AXIS_PARTITION = {v: k for k, v in PARTITION_AXIS.items()}

TEMPLATE_VARIANTS = 6
HOLDOUT_TEMPLATE_VARIANT = 5
CORRECTION_VARIANTS = 3
HOLDOUT_CORRECTION_VARIANT = 2

HOLDOUT_IDS: dict[str, frozenset[str]] = {
    "field": frozenset(f"dom:{d}" for d in HOLDOUT_DOMAINS),
    "template": frozenset({f"tpl{HOLDOUT_TEMPLATE_VARIANT}"}),
    "numeric": frozenset({"numf_holdout"}),
    "enum": frozenset({"enum_holdout"}),
    "correction": frozenset({f"corr_v{HOLDOUT_CORRECTION_VARIANT}"}),
}

GENERAL_IDS: dict[str, tuple[str, ...]] = {
    "field": tuple(f"dom:{d}" for d in GENERAL_DOMAINS),
    "template": tuple(f"tpl{i}" for i in range(HOLDOUT_TEMPLATE_VARIANT)),
    "numeric": ("numf_general",),
    "enum": ("enum_general",),
    "correction": tuple(f"corr_v{i}" for i in range(HOLDOUT_CORRECTION_VARIANT)),
}

# Concrete field-ID registries (enumerable because IDs are pure functions of defs).
HOLDOUT_FIELD_IDS_BY_PARTITION: dict[str, frozenset[str]] = {
    "unseen_fields": frozenset(all_field_ids(HOLDOUT_DOMAIN_DEFS)),
    "unseen_enums": frozenset(all_field_ids(ENUM_HOLDOUT_DEFS)),
}


def is_holdout(axis: str, family_id: str) -> bool:
    return family_id in HOLDOUT_IDS[axis]


def holdout_axes_used(families: dict[str, str]) -> set[str]:
    return {a for a in AXES if families.get(a) in HOLDOUT_IDS[a]}


def variant_index(family_id: str) -> int:
    """tpl3 -> 3, corr_v1 -> 1."""
    return int("".join(c for c in family_id.split("_")[-1] if c.isdigit()))
