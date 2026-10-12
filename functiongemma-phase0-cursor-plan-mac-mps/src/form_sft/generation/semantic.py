"""G3: semantic truth generation. Canonical Operation first, then SemanticContext.

Deterministic from (seed, slot.id). No utterance text is produced here.
"""
from __future__ import annotations

import hashlib
import json
import random

from form_sft.dataset import families as fam
from form_sft.dataset.quota_planner import Slot
from form_sft.generation import catalog as cat
from form_sft.generation import lexicon as lex
from form_sft.generation.values import gen_value

TYPES = cat.FIELD_TYPES


def slot_rng(seed: int, slot_id: str, purpose: str) -> random.Random:
    return random.Random(f"{seed}|{slot_id}|{purpose}")


def _target_candidates(slot: Slot, ftype: str) -> list[cat.FieldDef]:
    domain = slot.families["field"].split(":", 1)[1]
    defs = [d for d in cat.FIELD_DEFS if d.domain == domain and d.type == ftype]
    if slot.families["enum"] == "enum_holdout":
        defs = [d for d in defs if d.holdout_enum]
    elif domain in cat.GENERAL_DOMAINS:
        defs = [d for d in defs if not d.holdout_enum]
    if slot.families["numeric"] == "numf_holdout":
        if ftype == "integer":
            defs = [d for d in defs if d.maximum >= 1100]
        else:
            defs = [d for d in defs if d.maximum - d.minimum >= 2]
    if not defs:
        raise ValueError(f"no field definition for slot {slot.id} ({ftype}, {domain})")
    return defs


def _context_pool(slot: Slot) -> tuple[cat.FieldDef, ...]:
    if slot.families["field"] in fam.HOLDOUT_IDS["field"]:
        return cat.HOLDOUT_DOMAIN_DEFS
    return cat.GENERAL_DEFS


def _pick_nc_kind(slot: Slot, rng: random.Random) -> str:
    if slot.correction:
        return "unresolved"
    w = lex.NC_ADVERSARIAL_WEIGHTS if slot.test_partition == "hard_negatives_adversarial" else lex.NC_KIND_WEIGHTS
    kinds = sorted(w)
    return rng.choices(kinds, weights=[w[k] for k in kinds])[0]


def candidate_count(slot: Slot, rng: random.Random) -> int:
    if slot.specialization_mode == "active_field":
        return rng.choice([1, 1, 2, 3, 4])
    b = slot.reachable_bucket
    if b == "6_to_8":
        return rng.randint(6, 8)
    return int(b)


def generate_semantic_truth(slot: Slot, seed: int) -> dict:
    rng = slot_rng(seed, slot.id, "semantic")
    is_nc = slot.primitive == "no_change"
    holdout_numeric = slot.families["numeric"] == "numf_holdout"

    ftype = rng.choice(TYPES) if is_nc else slot.primitive
    tdef = rng.choice(_target_candidates(slot, ftype))
    nc_kind = _pick_nc_kind(slot, rng) if is_nc else None

    n_reach = candidate_count(slot, rng)
    n_extra = rng.choice([0, 1, 1, 2])
    if nc_kind == "nonreachable_field":
        n_extra = max(n_extra, 1)

    # distractor definitions (never duplicate the target; labels are unique per def)
    pool = [d for d in _context_pool(slot) if d.key != tdef.key]
    same = [d for d in pool if d.domain == tdef.domain]
    other = [d for d in pool if d.domain != tdef.domain]
    rng.shuffle(same)
    rng.shuffle(other)
    need = (n_reach - 1) + n_extra
    k_same = min(len(same), rng.randint(0, need))
    chosen = same[:k_same] + other[: need - k_same]
    if len(chosen) < need:  # pool exhaustion (cannot happen with catalog sizes)
        chosen += same[k_same:][: need - len(chosen)]
    if len(chosen) != need:
        raise ValueError("insufficient distractor fields")
    reach_defs = [tdef] + chosen[: n_reach - 1]
    extra_defs = chosen[n_reach - 1:]
    all_defs = reach_defs + extra_defs
    rng.shuffle(all_defs)

    identity = slot.field_identity
    fields = [cat.build_field(d, identity) for d in all_defs]
    by_def = {d.key: f for d, f in zip(all_defs, fields)}
    subject = by_def[tdef.key]
    reachable_ids = [by_def[d.key]["field_id"] for d in all_defs if d in reach_defs]

    if slot.specialization_mode == "active_field":
        active_id = subject["field_id"]
    else:
        active_id = rng.choice(reachable_ids) if rng.random() < 0.5 else None

    # ---- canonical target FIRST --------------------------------------------
    value = None
    old_values: list = []
    other_field = None
    if not is_nc:
        value = gen_value(tdef, subject, rng, holdout_numeric)
        if tdef.type == "enum":
            target = {"name": "set_enum", "arguments": {"field_id": subject["field_id"], "option_id": value}}
        else:
            op = {"string": "set_string", "integer": "set_integer", "number": "set_number",
                  "boolean": "set_boolean", "date": "set_date"}[tdef.type]
            target = {"name": op, "arguments": {"field_id": subject["field_id"], "value": value}}
        if slot.correction:
            n_old = 2 if slot.correction_subtype == "multiple_corrections" else 1
            chain, prev = [], value
            for _ in range(n_old):
                prev = gen_value(tdef, subject, rng, holdout_numeric, avoid=(prev,))
                chain.append(prev)
            old_values = list(reversed(chain))  # oldest first
    else:
        target = {"name": "no_change", "arguments": {}}
        if nc_kind in lex.NC_NEEDS_VALUE or nc_kind == "unresolved":
            vdef = tdef
            vfield = subject
            if nc_kind == "nonreachable_field":
                odef = extra_defs[0]
                vdef, vfield = odef, by_def[odef.key]
                other_field = {"field_id": vfield["field_id"], "label": vfield["label"], "type": vfield["type"]}
            value = gen_value(vdef, vfield, rng, False)
            if nc_kind == "unresolved":
                old_values = [value]

    ctx_fields = fields
    fingerprint = "sha256:" + hashlib.sha256(
        json.dumps(ctx_fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    context = {
        "revision": rng.randint(1, 40),
        "fingerprint": fingerprint,
        "locale": "en-IN",
        "active_field_id": active_id,
        "reachable_field_ids": reachable_ids,
        "fields": ctx_fields,
    }
    value_field_id = None
    if is_nc and value is not None:
        value_field_id = (other_field or {"field_id": subject["field_id"]})["field_id"]
    return {
        "slot_id": slot.id,
        "mode": slot.specialization_mode,
        "target": target,
        "context": context,
        "subject_field_id": subject["field_id"],
        "subject_label": subject["label"],
        "subject_type": tdef.type,
        "subject_def": tdef.key,
        "domain": tdef.domain,
        "value": value,                 # raw python value (target value, or distractor value for nc)
        "value_field_id": value_field_id if is_nc else subject["field_id"],
        "old_values": old_values,       # oldest first (correction chains)
        "nc_kind": nc_kind,
        "other_field": other_field,
        "n_candidates": len(reachable_ids),
        "n_extra_fields": len(extra_defs),
    }
