"""Assign whole schema families to one split before any utterance is realized."""
from __future__ import annotations

from pydantic import BaseModel, Field


class FamilyGroup(BaseModel):
    family_id: str
    template_lineage_id: str
    ontology_lineage_id: str
    split: str | None = None
    counterfactual_group_ids: list[str] = Field(default_factory=list)
    record_ids: list[str] = Field(default_factory=list)


class SplitManifest(BaseModel):
    assignments: dict[str, str]
    family_counts: dict[str, int]
    record_counts: dict[str, int]
    leaks: list[dict] = Field(default_factory=list)


class SplitError(RuntimeError):
    pass


def split_by_family(groups: list[FamilyGroup], quotas: dict[str, int], seed: int = 0) -> SplitManifest:
    """Keep each family, template lineage, and ontology lineage inside one split.

    ``seed`` is accepted so callers can record the plan seed. Assignment uses the
    split already stored on each group; this function does not move a family
    after it has been planned.
    """
    del seed
    if any(group.split is None for group in groups):
        raise SplitError("every family must be assigned before realization")
    assignments = {group.family_id: group.split for group in groups}
    if len(assignments) != len(groups):
        raise SplitError("duplicate family id")
    leaks: list[dict] = []
    lineage_split: dict[str, str] = {}
    for group in groups:
        for lineage in (group.template_lineage_id, group.ontology_lineage_id):
            previous = lineage_split.get(lineage)
            if previous is not None and previous != group.split:
                leaks.append({"code": "L_LINEAGE", "id": lineage, "detail": [previous, group.split]})
            lineage_split[lineage] = group.split or ""
    family_counts: dict[str, int] = {name: 0 for name in quotas}
    record_counts: dict[str, int] = {name: 0 for name in quotas}
    for group in groups:
        family_counts[group.split] = family_counts.get(group.split, 0) + 1
        record_counts[group.split] = record_counts.get(group.split, 0) + len(group.record_ids)
    if leaks:
        raise SplitError(f"lineage crosses splits: {leaks[:3]}")
    return SplitManifest(
        assignments=assignments,
        family_counts=family_counts,
        record_counts=record_counts,
        leaks=[],
    )
