"""Prohibited-leakage checker (G5). Operates on built rows (dict form).

Prohibited:
  L_FAMILY_ID        a holdout family id on a row outside its matching test partition
  L_PARTITION_FAMILY an unseen_* test row that does NOT carry its holdout family
  L_FIELD_ID         holdout field IDs in any context outside their partition
  L_TARGET_FIELD     unseen_fields/unseen_enums row whose subject field is not a holdout field
  L_TEMPLATE         holdout template variant used outside unseen_templates (checked on content)
  L_NUMERIC_FORM     holdout numeric composition surface outside unseen_numeric_compositions
  L_CORRECTION       holdout correction template outside unseen_corrections
  L_DUP_CROSS_SPLIT  identical (utterance, target, context signature) in different splits
  L_DUP_ID           duplicate row id
"""
from __future__ import annotations

import json
from collections import defaultdict

from form_sft.dataset import families as fam
from form_sft.generation.realize import norm

HOLDOUT_NUMERIC_FORMS = {"hundred_composite", "fraction_words"}


def dedupe_key(row: dict) -> str:
    ctx = row["semantic_context"]
    return json.dumps([
        norm(row["utterance"]), row["target"], row["metadata"]["specialization_mode"],
        ctx["active_field_id"], sorted(ctx["reachable_field_ids"]),
    ], sort_keys=True, separators=(",", ":"))


def check_leakage(rows: list[dict]) -> list[dict]:
    v: list[dict] = []

    def add(code, row, detail=""):
        v.append({"code": code, "id": row["id"], "detail": detail})

    seen_ids: set[str] = set()
    by_key: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        md = row["metadata"]
        part = md["test_partition"] if md["split"] == "test" else None
        fams = md["families"]

        if row["id"] in seen_ids:
            add("L_DUP_ID", row)
        seen_ids.add(row["id"])
        by_key[dedupe_key(row)].append((row["id"], md["split"]))

        for axis in fam.AXES:
            fid = fams.get(axis)
            if fid in fam.HOLDOUT_IDS[axis] and part != fam.AXIS_PARTITION[axis]:
                add("L_FAMILY_ID", row, f"{axis}={fid} in partition {md['test_partition']}")
        if part in fam.PARTITION_AXIS:
            ax = fam.PARTITION_AXIS[part]
            if fams.get(ax) not in fam.HOLDOUT_IDS[ax]:
                add("L_PARTITION_FAMILY", row, f"{part} lacks holdout {ax}")

        ctx_ids = {f["field_id"] for f in row["semantic_context"]["fields"]}
        for p, ids in fam.HOLDOUT_FIELD_IDS_BY_PARTITION.items():
            if ctx_ids & ids and part != p:
                add("L_FIELD_ID", row, f"holdout fields of {p} present in {md['test_partition']}")
            if part == p and md.get("subject_field_id") not in ids:
                add("L_TARGET_FIELD", row, f"{p} subject {md.get('subject_field_id')} not held out")

        if md.get("template_variant") == fam.HOLDOUT_TEMPLATE_VARIANT and part != "unseen_templates":
            add("L_TEMPLATE", row, md.get("template_id", ""))
        if part == "unseen_templates" and md.get("template_variant") != fam.HOLDOUT_TEMPLATE_VARIANT:
            add("L_PARTITION_FAMILY", row, "unseen_templates row without holdout template content")
        if md.get("value_form") in HOLDOUT_NUMERIC_FORMS and part != "unseen_numeric_compositions":
            add("L_NUMERIC_FORM", row, md["value_form"])
        if part == "unseen_numeric_compositions" and md.get("value_form") not in HOLDOUT_NUMERIC_FORMS:
            add("L_PARTITION_FAMILY", row, "unseen_numeric row without holdout composition")
        ct = md.get("correction_template") or ""
        if ct.endswith(f".v{fam.HOLDOUT_CORRECTION_VARIANT}") and part != "unseen_corrections":
            add("L_CORRECTION", row, ct)
        if part == "unseen_corrections" and not ct.endswith(f".v{fam.HOLDOUT_CORRECTION_VARIANT}"):
            add("L_PARTITION_FAMILY", row, "unseen_corrections row without holdout correction content")

    for key, members in by_key.items():
        if len({s for _, s in members}) > 1:
            for rid, _ in members:
                v.append({"code": "L_DUP_CROSS_SPLIT", "id": rid, "detail": key[:120]})
    return v


def duplicate_content_count(rows: list[dict]) -> int:
    c: dict[str, int] = defaultdict(int)
    for r in rows:
        c[dedupe_key(r)] += 1
    return sum(n - 1 for n in c.values() if n > 1)
