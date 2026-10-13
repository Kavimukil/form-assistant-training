"""Replay-heavy v9 train split. Does not change v7 or v8 generation."""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

from form_sft.dataset.similarity import exact_key, normalize_utterance
from form_sft.generation.scenario_factory import generate_scenario, realize_scenario, reverse_enum_options
from form_sft.generation.schema_factory import generate_schema
from form_sft.universal.build import UniversalBuildError, _dump, _request, assemble_record
from form_sft.validation.agreement import semantic_agreement
from form_sft.validation.record import validate_record

PROTECTED = {"no_change", "option_permutation", "counterfactual"}
FROZEN_SFT_HASHES = {
    "validation.sft.jsonl": "ad7e4e3048c47ff4f2b5f38d4f91c639d41017be",
    "test.sft.jsonl": "0a76c40407ec79ad8dc258442e083f0f27cf6ddd",
}


def _stratum(row: dict) -> tuple:
    coverage = row["coverage"]
    return coverage["phenomenon"], tuple(coverage.get("field_types") or [])


def select_replay_ids(rows: list[dict], *, keep: int = 18000) -> list[str]:
    """Keep every protected row, then fill the rest by phenomenon and field type."""
    protected = [row for row in rows if row["coverage"]["phenomenon"] in PROTECTED]
    if len(protected) > keep:
        raise UniversalBuildError(f"protected rows {len(protected)} exceed {keep}")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row["coverage"]["phenomenon"] not in PROTECTED:
            groups[_stratum(row)].append(row)
    need = keep - len(protected)
    keys = sorted(groups)
    total = sum(len(groups[key]) for key in keys)
    if need > total:
        raise UniversalBuildError(f"need {need} replay rows, have {total}")
    raw = [len(groups[key]) * need / total for key in keys]
    counts = [int(value) for value in raw]
    leftover = need - sum(counts)
    order = sorted(range(len(keys)), key=lambda index: (-(raw[index] - counts[index]), keys[index]))
    for index in order[:leftover]:
        counts[index] += 1
    chosen = list(protected)
    for key, count in zip(keys, counts):
        pool = sorted(groups[key], key=lambda row: row["id"])
        chosen.extend(pool[:count])
    if len(chosen) != keep:
        raise UniversalBuildError(f"replay selected {len(chosen)}, need {keep}")
    return [row["id"] for row in chosen]


def _place_drill(form, request, *, seed, serial, seen, forbidden):
    for attempt in range(48):
        truth = generate_scenario(form, seed + attempt * 1_000_003 + serial, request)
        if request["type_name"] == "enum":
            field_id = truth.actions[0].field_id
            field = next(item for item in truth.form.fields if item.field_id == field_id)
            wanted = field.options[serial % len(field.options)].id
            if truth.actions[0].value != wanted:
                continue
        utterance = realize_scenario(truth, seed + serial + attempt)
        if normalize_utterance(utterance) in forbidden:
            continue
        record = assemble_record(
            truth,
            utterance,
            record_id=f"universal-full-v9-drill-{serial:05d}",
            seed=seed + serial,
            family_seed=seed,
        )
        issues = validate_record(record)
        if issues:
            continue
        agreement = semantic_agreement(record, old_surfaces=truth.old_surfaces)
        if not agreement.ok:
            continue
        key = exact_key(record)
        if key in seen:
            continue
        seen.add(key)
        return record
    raise UniversalBuildError(f"could not place v9 {request['type_name']} drill {serial}")


def generate_copy_drills(
    *,
    seed: int,
    kind: str,
    count: int,
    use_option_alias: bool,
    seen: set[str],
    forbidden: set[str],
    family_count: int = 80,
) -> list:
    forms = [generate_schema(seed, split="train", family_index=index) for index in range(family_count)]
    records = []
    for serial in range(count):
        form = forms[serial % family_count]
        if kind == "enum" and serial % 2:
            form = reverse_enum_options(form)
        request = _request("single_set", "train", serial)
        request["type_name"] = "string" if kind == "string" else "enum"
        if serial % 2:
            request["speak_field_alias"] = True
        if kind == "enum" and use_option_alias and serial % 2 == 0:
            request["speak_option_alias"] = True
        records.append(
            _place_drill(form, request, seed=seed + (0 if kind == "string" else 50_000), serial=serial, seen=seen, forbidden=forbidden)
        )
    return records


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_v9_dataset(
    v7_train: str | Path,
    frozen_dir: str | Path,
    out_dir: str | Path,
    comparison: dict,
    *,
    keep: int = 18000,
    string_count: int = 1000,
    choice_count: int = 1000,
    family_count: int = 80,
    seed: int = 20261012,
    expected_hashes: dict[str, str] | None = None,
) -> dict:
    if not comparison.get("predictions_present") or not comparison.get("proceed"):
        raise UniversalBuildError("v9 prediction gate did not pass; refusing to materialize")
    v7_train = Path(v7_train)
    frozen_dir = Path(frozen_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = {}
    rows = []
    for line in v7_train.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        lines[row["id"]] = line
        rows.append(row)
    chosen = select_replay_ids(rows, keep=keep)
    protected_ids = {row["id"] for row in rows if row["coverage"]["phenomenon"] in PROTECTED}
    if not protected_ids <= set(chosen):
        raise UniversalBuildError("replay dropped a protected no-change, permutation, or counterfactual row")
    forbidden = set()
    for name in ("validation.jsonl", "test.jsonl"):
        for line in (frozen_dir / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                forbidden.add(normalize_utterance(json.loads(line)["utterance"]))
    seen = set()
    drills = generate_copy_drills(
        seed=seed,
        kind="string",
        count=string_count,
        use_option_alias=False,
        seen=seen,
        forbidden=forbidden,
        family_count=family_count,
    )
    drills.extend(
        generate_copy_drills(
            seed=seed,
            kind="enum",
            count=choice_count,
            use_option_alias=bool(comparison.get("use_option_alias")),
            seen=seen,
            forbidden=forbidden,
            family_count=family_count,
        )
    )
    body = "".join(lines[row_id] + "\n" for row_id in chosen)
    body += "".join(_dump(record) + "\n" for record in drills)
    (out / "train.jsonl").write_text(body, encoding="utf-8")
    hashes = expected_hashes if expected_hashes is not None else FROZEN_SFT_HASHES
    for name, expected in hashes.items():
        source = frozen_dir / name
        actual = _sha1(source)
        if actual != expected:
            raise UniversalBuildError(f"{name} hash {actual} != {expected}")
        shutil.copyfile(source, out / name)
        json_name = name.replace(".sft", "")
        shutil.copyfile(frozen_dir / json_name, out / json_name)
    return {
        "train_rows": keep + string_count + choice_count,
        "replay_rows": keep,
        "string_drills": string_count,
        "choice_drills": choice_count,
        "use_option_alias": bool(comparison.get("use_option_alias")),
    }
