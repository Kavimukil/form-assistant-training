"""Normalized duplicates and token-shingle near-duplicates."""
from __future__ import annotations

import re
from collections import defaultdict

from form_sft.contracts.dataset_ir import DatasetRecordV2


def normalize_utterance(text: str) -> str:
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return " ".join(text.split())


def shingles(text: str, n: int = 3) -> set[str]:
    tokens = normalize_utterance(text).split()
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def action_signature(record: DatasetRecordV2) -> str:
    parts = []
    for action in record.expected_actions:
        parts.append(f"{action.kind}|{action.field_id}|{action.value}")
    return "||".join(sorted(parts))


def exact_key(record: DatasetRecordV2) -> str:
    return "\n".join(
        [
            normalize_utterance(record.utterance),
            action_signature(record),
            record.schema_fingerprint,
        ]
    )


def find_duplicate_groups(records: list[DatasetRecordV2]) -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        groups[exact_key(record)].append(record.id)
    return [ids for ids in groups.values() if len(ids) > 1]


def find_cross_split_leaks(
    records: list[DatasetRecordV2],
    *,
    threshold: float = 0.9,
) -> list[dict]:
    """Leak when utterance families, schema families, or template lineages cross splits."""
    leaks: list[dict] = []
    by_split: dict[str, list[DatasetRecordV2]] = defaultdict(list)
    for record in records:
        by_split[record.coverage.split].append(record)

    family_splits: dict[str, set[str]] = defaultdict(set)
    template_splits: dict[str, set[str]] = defaultdict(set)
    ontology_splits: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        split = record.coverage.split
        family_splits[record.schema.schema_family_id].add(split)
        template_splits[record.schema.template_lineage_id].add(split)
        ontology_splits[record.schema.ontology_lineage_id].add(split)
        if record.counterfactual_group_id:
            group_splits[record.counterfactual_group_id].add(split)
    for family, splits in family_splits.items():
        if len(splits) > 1:
            leaks.append({"code": "L_SCHEMA_FAMILY", "id": family, "detail": sorted(splits)})
    for lineage, splits in template_splits.items():
        if len(splits) > 1:
            leaks.append({"code": "L_TEMPLATE_LINEAGE", "id": lineage, "detail": sorted(splits)})
    for lineage, splits in ontology_splits.items():
        if len(splits) > 1:
            leaks.append({"code": "L_ONTOLOGY_LINEAGE", "id": lineage, "detail": sorted(splits)})
    for group, splits in group_splits.items():
        if len(splits) > 1:
            leaks.append({"code": "L_COUNTERFACTUAL_GROUP", "id": group, "detail": sorted(splits)})

    split_names = [name for name in ("train", "validation", "test") if name in by_split]
    for i, left_name in enumerate(split_names):
        for right_name in split_names[i + 1 :]:
            right = by_split[right_name]
            right_shingles = [(row, shingles(row.utterance)) for row in right]
            for row in by_split[left_name]:
                left_set = shingles(row.utterance)
                for other, other_set in right_shingles:
                    score = jaccard(left_set, other_set)
                    if score >= threshold:
                        leaks.append(
                            {
                                "code": "L_NEAR_DUPLICATE",
                                "id": row.id,
                                "detail": {"other": other.id, "jaccard": round(score, 4)},
                            }
                        )
    return leaks
