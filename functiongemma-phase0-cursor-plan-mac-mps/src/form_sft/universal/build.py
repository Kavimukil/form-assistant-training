"""Truth-first universal dataset builder.

Schema families are assigned to splits before scenarios or utterances exist.
Accepted rows are written only after validation, agreement, and leakage checks.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from form_sft.bridges.form_state import FormStateBridge
from form_sft.contracts.dataset_ir import (
    GENERATOR_VERSION,
    VALIDATOR_VERSION,
    CoverageTags,
    DatasetRecordV2,
    Provenance,
)
from form_sft.contracts.form_ir import schema_fingerprint
from form_sft.dataset.family_splitter import FamilyGroup, split_by_family
from form_sft.dataset.similarity import exact_key, find_cross_split_leaks, find_duplicate_groups
from form_sft.generation.counterfactual import make_counterfactual
from form_sft.generation.scenario_factory import (
    ScenarioTruth,
    _apply_shared_label,
    generate_scenario,
    realize_scenario,
    reverse_enum_options,
)
from form_sft.generation.schema_factory import generate_schema
from form_sft.validation.agreement import semantic_agreement
from form_sft.validation.ambiguity import classify_ambiguity
from form_sft.validation.record import validate_record

DIRECT = ("single_set", "multi_field", "correction", "clear", "no_change", "asr", "code_mixed")


class UniversalBuildError(RuntimeError):
    pass


def _dump(record: DatasetRecordV2) -> str:
    return json.dumps(record.model_dump(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def suites_for(split: str, phenomenon: str) -> list[str]:
    suites = ["opaque_id"]
    if split == "test":
        suites.extend(["schema_family_disjoint", "unseen_label"])
    named = {
        "option_permutation": "option_permutation",
        "counterfactual": "counterfactual_context",
        "asr": "asr",
        "code_mixed": "code_mixed",
        "correction": "correction",
        "no_change": "no_change",
    }
    if phenomenon in named:
        suites.append(named[phenomenon])
    return suites


def assemble_record(
    truth: ScenarioTruth,
    utterance: str,
    *,
    record_id: str,
    seed: int,
    family_seed: int,
    group_id: str | None = None,
) -> DatasetRecordV2:
    mode = truth.specialization_mode
    before = truth.model_dump()
    context = FormStateBridge.to_semantic_context(truth.form, truth.state, mode)
    if truth.model_dump() != before:
        raise UniversalBuildError("realization mutated scenario truth")
    mutating = [action for action in truth.actions if action.kind != "no_change"]
    coverage = CoverageTags(
        split=truth.split,
        phenomenon=truth.phenomenon,
        action_arity=len(mutating),
        specialization_mode=mode,
        suites=suites_for(truth.split, truth.phenomenon),
        opaque_ids=True,
        correction=truth.correction,
        asr=truth.asr,
        code_mixed=truth.code_mixed,
        counterfactual=truth.counterfactual,
        option_permutation=truth.option_permutation,
        ambiguity=truth.ambiguity,  # type: ignore[arg-type]
        field_types=list(truth.field_types),
    )
    provenance = Provenance(
        generator="universal",
        generator_version=GENERATOR_VERSION,
        seed=seed,
        family_seed=family_seed,
        state_seed=seed,
        surface_seed=seed,
        template_id=truth.template_id,
        template_lineage_id=truth.template_lineage_id,
        transforms=[name for name, flag in (("asr", truth.asr), ("code_mixed", truth.code_mixed), ("correction", truth.correction)) if flag],
        parent_schema_id=truth.form.schema_id,
        parent_scenario_id=record_id,
        validator_version=VALIDATOR_VERSION,
        quarantine_reason="ambiguous option surface" if truth.ambiguity == "quarantine" else None,
    )
    return DatasetRecordV2(
        id=record_id,
        schema=truth.form,
        state=truth.state,
        semantic_context=context,
        utterance=utterance,
        expected_actions=list(truth.actions),
        provenance=provenance,
        coverage=coverage,
        schema_fingerprint=schema_fingerprint(truth.form, truth.state),
        counterfactual_group_id=group_id,
    )


def _check_quotas(spec: dict) -> None:
    for split, expected in spec["splits"].items():
        actual = sum(counts[split] for counts in spec["coverage"].values())
        if actual != expected:
            raise UniversalBuildError(f"{split} coverage {actual} != split size {expected}")


def train_single_set_type(index: int, count: int, perm: int, string_force: int) -> str | None:
    """Force the tail of train single-set rows so permutation and string copy have bases."""
    if index >= count - perm:
        return "enum"
    if string_force and index >= count - perm - string_force:
        return "string"
    return None


def _request(phenomenon: str, split: str, index: int) -> dict:
    request = {
        "phenomenon": phenomenon,
        "split": split,
        "index": index,
        "asr": phenomenon == "asr",
        "code_mixed": phenomenon == "code_mixed",
    }
    if phenomenon == "correction":
        request["type_name"] = "integer"
        request["mode"] = "active_field"
    elif phenomenon in {"no_change", "multi_field", "counterfactual", "ambiguity"}:
        request["mode"] = "reachable_fields"
        if phenomenon == "no_change" and split == "train" and index % 2:
            request["echo_current"] = True
    elif phenomenon == "clear":
        request["mode"] = "active_field"
    else:
        request["mode"] = "active_field" if index % 2 == 0 else "reachable_fields"
    return request


def _permute(record: DatasetRecordV2, new_id: str) -> DatasetRecordV2:
    form = reverse_enum_options(record.schema)
    state = record.state.model_copy(deep=True)
    mode = record.coverage.specialization_mode
    coverage = record.coverage.model_copy(
        deep=True,
        update={
            "phenomenon": "option_permutation",
            "option_permutation": True,
            "suites": suites_for(record.coverage.split, "option_permutation"),
        },
    )
    provenance = record.provenance.model_copy(
        deep=True,
        update={
            "parent_schema_id": form.schema_id,
            "parent_scenario_id": new_id,
            "transforms": list(record.provenance.transforms) + ["option_permutation"],
        },
    )
    return DatasetRecordV2(
        id=new_id,
        schema=form,
        state=state,
        semantic_context=FormStateBridge.to_semantic_context(form, state, mode),
        utterance=record.utterance,
        expected_actions=list(record.expected_actions),
        provenance=provenance,
        coverage=coverage,
        schema_fingerprint=schema_fingerprint(form, state),
        counterfactual_group_id=record.counterfactual_group_id,
    )


def build_records(spec: dict) -> tuple[list[DatasetRecordV2], list[DatasetRecordV2]]:
    _check_quotas(spec)
    seed = int(spec["spec"]["random_seed"])
    exp = spec["spec"]["experiment_id"]
    forms = {
        split: [generate_schema(seed, split=split, family_index=index) for index in range(count)]
        for split, count in spec["families"].items()
    }
    accepted: list[DatasetRecordV2] = []
    quarantine: list[DatasetRecordV2] = []
    seen: set[str] = set()
    serial = 0

    def add(truth: ScenarioTruth, *, group_id: str | None = None, bucket: str = "accepted") -> DatasetRecordV2:
        nonlocal serial
        serial += 1
        record_id = f"{exp}-{truth.split}-{serial:05d}"
        utterance = realize_scenario(truth, seed + serial)
        record = assemble_record(
            truth, utterance, record_id=record_id, seed=seed + serial, family_seed=seed, group_id=group_id
        )
        issues = validate_record(record)
        if issues:
            raise UniversalBuildError(f"{record.id} invalid: {issues[:4]}")
        verdict = classify_ambiguity(record)
        if verdict != record.coverage.ambiguity:
            raise UniversalBuildError(f"{record.id} ambiguity {verdict} != {record.coverage.ambiguity}")
        agreement = semantic_agreement(record, old_surfaces=truth.old_surfaces)
        if bucket == "accepted" and not agreement.ok:
            raise UniversalBuildError(f"{record.id} agreement: {agreement.reasons}")
        key = exact_key(record)
        if bucket == "accepted" and key in seen:
            serial -= 1
            raise UniversalBuildError("duplicate")
        if bucket == "accepted":
            seen.add(key)
        (accepted if bucket == "accepted" else quarantine).append(record)
        return record

    for phenomenon in DIRECT:
        for split, count in spec["coverage"][phenomenon].items():
            family_count = len(forms[split])
            for index in range(count):
                placed = False
                for attempt in range(48):
                    request = _request(phenomenon, split, index)
                    if phenomenon == "single_set" and split == "train":
                        perm = int(spec["coverage"]["option_permutation"]["train"])
                        string_force = int(spec.get("forced_string_train") or 0)
                        forced = train_single_set_type(index, count, perm, string_force)
                        if forced:
                            request["type_name"] = forced
                    if split == "train" and phenomenon not in {"no_change", "clear", "ambiguity"}:
                        if index % 4 == 1:
                            request["speak_field_alias"] = True
                    truth = generate_scenario(
                        forms[split][index % family_count],
                        seed + index + len(phenomenon) * 1009 + attempt * 1_000_003,
                        request,
                    )
                    try:
                        add(truth)
                    except UniversalBuildError as exc:
                        if str(exc) != "duplicate":
                            raise
                        continue
                    placed = True
                    break
                if not placed:
                    raise UniversalBuildError(f"could not place a unique {split} {phenomenon} row at index {index}")

    for split, count in spec["coverage"]["option_permutation"].items():
        bases = [
            row
            for row in accepted
            if row.coverage.split == split and row.coverage.phenomenon == "single_set" and "enum" in row.coverage.field_types
        ]
        placed = 0
        for base in bases:
            if placed == count:
                break
            serial += 1
            permuted = _permute(base, f"{exp}-{split}-{serial:05d}")
            issues = validate_record(permuted)
            if issues:
                raise UniversalBuildError(f"{permuted.id} invalid permutation: {issues[:3]}")
            key = exact_key(permuted)
            if key in seen:
                serial -= 1
                continue
            seen.add(key)
            accepted.append(permuted)
            placed += 1
        if placed < count:
            raise UniversalBuildError(f"{split} placed {placed} unique permutations, need {count}")

    for split, count in spec["coverage"]["counterfactual"].items():
        family_count = len(forms[split])
        if family_count < 2:
            raise UniversalBuildError(f"{split} needs two families for counterfactuals")
        for pair in range(count // 2):
            shared = f"hue{split[0]}{pair}"
            group = f"cf-{split}-{pair:04d}"
            request = _request("counterfactual", split, pair)
            request["shared_label"] = shared
            placed = False
            for attempt in range(48):
                form_a = _apply_shared_label(forms[split][(pair + attempt) % family_count], shared)
                form_b = _apply_shared_label(forms[split][(pair + attempt + 1) % family_count], shared)
                try:
                    source = add(
                        generate_scenario(form_a, seed + 5000 + pair + attempt * 1_000_003, request),
                        group_id=group,
                    )
                except UniversalBuildError as exc:
                    if str(exc) != "duplicate":
                        raise
                    continue
                paired = make_counterfactual(source, form_b)
                serial += 1
                paired.id = f"{exp}-{split}-{serial:05d}"
                paired.counterfactual_group_id = group
                paired.provenance.parent_scenario_id = paired.id
                paired.coverage.suites = suites_for(split, "counterfactual")
                issues = validate_record(paired)
                if issues:
                    raise UniversalBuildError(f"{paired.id} invalid counterfactual: {issues[:4]}")
                if paired.utterance != source.utterance or paired.expected_actions == source.expected_actions:
                    raise UniversalBuildError("counterfactual pair did not diverge only in context")
                key = exact_key(paired)
                if key in seen:
                    serial -= 1
                    accepted.pop()
                    seen.discard(exact_key(source))
                    continue
                seen.add(key)
                accepted.append(paired)
                placed = True
                break
            if not placed:
                raise UniversalBuildError(f"could not place a unique {split} counterfactual pair {pair}")

    for index in range(int(spec.get("ambiguity_quarantine") or 0)):
        truth = generate_scenario(forms["train"][0], seed + 9000 + index, _request("ambiguity", "train", index))
        add(truth, bucket="quarantine")
    return accepted, quarantine


def _coverage_matrix(records: list[DatasetRecordV2]) -> dict:
    matrix: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        matrix[record.coverage.split][record.coverage.phenomenon] += 1
    return {split: dict(counts) for split, counts in matrix.items()}


def unseen_report(records: list[DatasetRecordV2]) -> dict:
    train_ids = {field.field_id for row in records if row.coverage.split == "train" for field in row.schema.fields}
    targets = [
        action.field_id
        for row in records
        if row.coverage.split == "test"
        for action in row.expected_actions
        if action.field_id
    ]
    unseen = [fid for fid in targets if fid not in train_ids]
    train_families = {row.schema.schema_family_id for row in records if row.coverage.split == "train"}
    test_families = {row.schema.schema_family_id for row in records if row.coverage.split == "test"}
    return {
        "test_target_ids": len(targets),
        "unseen_target_ids": len(unseen),
        "unseen_target_ratio": (len(unseen) / len(targets)) if targets else 1.0,
        "test_families_in_train": sorted(test_families & train_families),
    }


def write_dataset(spec: dict, out_dir: str | Path, *, near_duplicate_threshold: float = 0.9) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    accepted, quarantine = build_records(spec)
    duplicates = find_duplicate_groups(accepted)
    leaks = find_cross_split_leaks(accepted, threshold=near_duplicate_threshold)
    if duplicates or leaks:
        report = {"passed": False, "duplicates": duplicates, "leaks": leaks[:20]}
        (out / "qc_report.json").write_text(json.dumps(report, indent=2) + "\n")
        raise UniversalBuildError(f"qc failed duplicates={len(duplicates)} leaks={len(leaks)}")
    by_family: dict[str, list[str]] = defaultdict(list)
    sample_for: dict[str, DatasetRecordV2] = {}
    for record in accepted:
        by_family[record.schema.schema_family_id].append(record.id)
        sample_for[record.schema.schema_family_id] = record
    groups = []
    for family, ids in by_family.items():
        sample = sample_for[family]
        groups.append(
            FamilyGroup(
                family_id=family,
                template_lineage_id=sample.schema.template_lineage_id,
                ontology_lineage_id=sample.schema.ontology_lineage_id,
                split=sample.coverage.split,
                counterfactual_group_ids=sorted(
                    {
                        row.counterfactual_group_id
                        for row in accepted
                        if row.schema.schema_family_id == family and row.counterfactual_group_id
                    }
                ),
                record_ids=ids,
            )
        )
    split_manifest = split_by_family(groups, spec["splits"], seed=int(spec["spec"]["random_seed"]))
    unseen = unseen_report(accepted)
    if unseen["unseen_target_ratio"] < 0.8 or unseen["test_families_in_train"]:
        raise UniversalBuildError(f"family holdout failed: {unseen}")
    for split in spec["splits"]:
        rows = [record for record in accepted if record.coverage.split == split]
        (out / f"{split}.jsonl").write_text("".join(_dump(record) + "\n" for record in rows))
    (out / "quarantine.jsonl").write_text("".join(_dump(record) + "\n" for record in quarantine))
    (out / "rejected.jsonl").write_text("")
    seen: set[str] = set()
    registry = []
    for record in accepted:
        if record.schema.schema_id in seen:
            continue
        seen.add(record.schema.schema_id)
        registry.append(record.schema.model_dump())
    (out / "schema_registry.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in registry))
    (out / "scenario_plan.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "id": record.id,
                    "split": record.coverage.split,
                    "phenomenon": record.coverage.phenomenon,
                    "schema_family_id": record.schema.schema_family_id,
                    "template_id": record.provenance.template_id,
                },
                sort_keys=True,
            )
            + "\n"
            for record in accepted
        )
    )
    matrix = _coverage_matrix(accepted)
    summary = {
        "experiment_id": spec["spec"]["experiment_id"],
        "generator_version": GENERATOR_VERSION,
        "accepted": len(accepted),
        "quarantine": len(quarantine),
        "splits": {split: sum(1 for row in accepted if row.coverage.split == split) for split in spec["splits"]},
        "unseen": unseen,
        "passed": True,
    }
    (out / "coverage_matrix.json").write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    (out / "family_split.json").write_text(split_manifest.model_dump_json(indent=2) + "\n")
    (out / "lineage.json").write_text(
        json.dumps(
            {
                "families": [
                    {
                        "id": group.family_id,
                        "split": group.split,
                        "template": group.template_lineage_id,
                        "ontology": group.ontology_lineage_id,
                    }
                    for group in groups
                ]
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (out / "qc_report.json").write_text(
        json.dumps(
            {
                "passed": True,
                "accepted": len(accepted),
                "quarantine": len(quarantine),
                "duplicates": 0,
                "leaks": 0,
                "structural_errors": 0,
                "unseen": unseen,
                "coverage": matrix,
                "threshold": near_duplicate_threshold,
                "label_precision": 1.0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (out / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
