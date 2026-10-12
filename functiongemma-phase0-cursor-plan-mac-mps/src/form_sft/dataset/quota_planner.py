"""G2: deterministic exact-quota allocator. Runs BEFORE any utterance realization.

Stages (each stage asserts its own marginals; a final full audit runs at the end):
  1 split x primitive              (exact, from YAML nested counts)
  2 correction x ASR intersection  (per-split overlap, then subtypes, couplings)
  3 specialization + reachable bucket
  4 field_identity, realization_source, difficulty
  5 test partitions                (constrained, exact)
  6 holdout family ids             (before realization)
Same seed => byte-identical plan.  Different seed => different assignment, same quotas.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass

from form_sft.dataset import families as fam
from form_sft.dataset.quotas import SPLITS, check_quotas, failed

SPLIT_ABBR = {"train": "tr", "validation": "va", "test": "te"}
NUMERIC = ("integer", "number")
BUCKET_RANK = {None: 0.0, "2": 0.5, "3": 1.0, "4": 1.5, "5": 2.0, "6_to_8": 2.5}


class QuotaError(RuntimeError):
    pass


@dataclass(frozen=True)
class Slot:
    id: str
    split: str
    primitive: str
    specialization_mode: str
    correction: bool
    asr_noise: bool
    source: str
    field_identity: str
    difficulty: str
    families: dict[str, str]
    reachable_bucket: str | None = None  # None | "2" | "3" | "4" | "5" | "6_to_8"
    correction_subtype: str | None = None
    asr_subtype: str | None = None
    test_partition: str | None = None  # test: one of 8 partitions; else "train"/"validation"

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------ apportioning --

def apportion(row_totals: list[int], col_totals: list[int]) -> list[list[int]]:
    """Deterministic integer matrix with exact row and column sums (largest remainder)."""
    total = sum(row_totals)
    if total != sum(col_totals):
        raise QuotaError(f"marginals disagree: {total} != {sum(col_totals)}")
    m = [[(r * c) // total for c in col_totals] for r in row_totals]
    rem = [[(r * c) % total for c in col_totals] for r in row_totals]
    rdef = [r - sum(row) for r, row in zip(row_totals, m)]
    cdef = [c - sum(m[i][j] for i in range(len(row_totals))) for j, c in enumerate(col_totals)]
    while any(rdef):
        best = None
        for i, rd in enumerate(rdef):
            if rd <= 0:
                continue
            for j, cd in enumerate(cdef):
                if cd > 0 and (best is None or rem[i][j] > rem[best[0]][best[1]]):
                    best = (i, j)
        if best is None:
            raise QuotaError("apportion failed to reconcile deficits")
        i, j = best
        m[i][j] += 1
        rdef[i] -= 1
        cdef[j] -= 1
        rem[i][j] = -1
    return m


def _pick(rng: random.Random, pool: list[int], k: int, what: str) -> list[int]:
    if k > len(pool):
        raise QuotaError(f"cannot pick {k} rows for {what}: only {len(pool)} eligible")
    chosen = rng.sample(sorted(pool), k)
    taken = set(chosen)
    pool[:] = [x for x in pool if x not in taken]
    return chosen


def _rng(seed: int, *parts) -> random.Random:
    return random.Random("|".join([str(seed), *map(str, parts)]))


# ---------------------------------------------------------------- planner --

def plan_slots(spec: dict) -> list[Slot]:
    seed = spec["spec"]["random_seed"]
    exp = spec["spec"]["experiment_id"]
    splits = list(spec["splits"])
    if tuple(splits) != SPLITS:
        raise QuotaError(f"unexpected split order {splits}")
    sizes = [spec["splits"][s] for s in splits]

    prim_names = list(spec["primitives"])
    corr_spec = spec["phenomena"]["correction"]
    asr_spec = spec["phenomena"]["asr_noise"]
    corr_sub_names = list(corr_spec["subtypes"])
    asr_sub_names = list(asr_spec["subtypes"])
    bucket_names = [str(k) for k in spec["specialization"]["reachable_fields"]["candidate_counts"]]
    bucket_totals = list(spec["specialization"]["reachable_fields"]["candidate_counts"].values())

    # ---- global matrices (split x category), all exact --------------------
    corr_split = [corr_spec["split"][s] for s in splits]
    asr_split = [asr_spec["split"][s] for s in splits]
    corr_mat = apportion(corr_split, [corr_spec["subtypes"][k] for k in corr_sub_names])
    asr_mat = apportion(asr_split, [asr_spec["subtypes"][k] for k in asr_sub_names])
    reach_split = [spec["specialization"]["reachable_fields"]["split"][s] for s in splits]
    bucket_mat = apportion(reach_split, bucket_totals)
    ident_names = list(spec["field_identity"])
    ident_mat = apportion(sizes, [spec["field_identity"][k] for k in ident_names])
    src_names = list(spec["realization_source"])
    src_mat = apportion(sizes, [spec["realization_source"][k] for k in src_names])
    diff_names = list(spec["difficulty"])
    diff_mat = apportion(sizes, [spec["difficulty"][k] for k in diff_names])

    # overlap per split: proportional to min(corr, asr)  (e.g. 600/75/75)
    overlap_total = spec["intersection"]["correction_and_asr"]
    weights = [min(c, a) for c, a in zip(corr_split, asr_split)]
    overlap = _split_total(overlap_total, weights)

    plan_rows: dict[str, list[dict]] = {}
    for si, split in enumerate(splits):
        n = sizes[si]
        rng = _rng(seed, "stage1", split)

        # -- stage 1: split x primitive
        prims: list[str] = []
        for p in prim_names:
            prims += [p] * spec["primitives"][p]["split"][split]
        if len(prims) != n:
            raise QuotaError(f"{split}: primitive counts {len(prims)} != split size {n}")
        rng.shuffle(prims)
        rows = [dict(i=i, split=split, primitive=p, correction=False, correction_subtype=None, asr_noise=False,
                     asr_subtype=None, specialization_mode="active_field", reachable_bucket=None, source=None,
                     field_identity=None, difficulty=None, test_partition=split) for i, p in enumerate(prims)]
        plan_rows[split] = rows
        idx_all = list(range(n))

        # -- stage 2: correction x ASR
        rng = _rng(seed, "stage2", split)
        c_total, a_total, both = corr_split[si], asr_split[si], overlap[si]
        corr_counts = dict(zip(corr_sub_names, corr_mat[si]))
        asr_counts = dict(zip(asr_sub_names, asr_mat[si]))

        nc_pool = [i for i in idx_all if rows[i]["primitive"] == "no_change"]
        non_nc_pool = [i for i in idx_all if rows[i]["primitive"] != "no_change"]
        unresolved = _pick(rng, nc_pool, corr_counts["unresolved_to_no_change"], "unresolved_to_no_change")
        for i in unresolved:
            rows[i].update(correction=True, correction_subtype="unresolved_to_no_change")
        rest_sub = [k for k in corr_sub_names if k != "unresolved_to_no_change"]
        rest_list: list[str] = []
        for k in rest_sub:
            rest_list += [k] * corr_counts[k]
        rng.shuffle(rest_list)
        corr_rows = _pick(rng, non_nc_pool, len(rest_list), "other correction subtypes")
        for i, k in zip(corr_rows, rest_list):
            rows[i].update(correction=True, correction_subtype=k)
        if sum(1 for r in rows if r["correction"]) != c_total:
            raise QuotaError(f"{split}: correction count mismatch")

        corr_idx = [i for i in idx_all if rows[i]["correction"]]
        noncorr_idx = [i for i in idx_all if not rows[i]["correction"]]
        k_nc = asr_counts["number_specific_confusion"]
        both_nc = (k_nc * both) // a_total if a_total else 0
        asr_only_nc = k_nc - both_nc
        corr_numeric = [i for i in corr_idx if rows[i]["primitive"] in NUMERIC]
        noncorr_numeric = [i for i in noncorr_idx if rows[i]["primitive"] in NUMERIC]
        asr_set: dict[int, str | None] = {}
        for i in _pick(rng, corr_numeric, both_nc, "number_confusion (both)"):
            asr_set[i] = "number_specific_confusion"
        for i in _pick(rng, noncorr_numeric, asr_only_nc, "number_confusion (asr only)"):
            asr_set[i] = "number_specific_confusion"
        corr_left = [i for i in corr_idx if i not in asr_set]
        noncorr_left = [i for i in noncorr_idx if i not in asr_set]
        for i in _pick(rng, corr_left, both - both_nc, "asr overlap"):
            asr_set[i] = None
        for i in _pick(rng, noncorr_left, (a_total - both) - asr_only_nc, "asr only"):
            asr_set[i] = None
        rest_asr: list[str] = []
        for k in asr_sub_names:
            if k != "number_specific_confusion":
                rest_asr += [k] * asr_counts[k]
        rng.shuffle(rest_asr)
        free = sorted(i for i, v in asr_set.items() if v is None)
        if len(free) != len(rest_asr):
            raise QuotaError(f"{split}: asr subtype/row mismatch {len(free)} vs {len(rest_asr)}")
        for i, k in zip(free, rest_asr):
            asr_set[i] = k
        for i, k in asr_set.items():
            rows[i].update(asr_noise=True, asr_subtype=k)

        # -- stage 3: specialization + reachable bucket
        rng = _rng(seed, "stage3", split)
        r_total = reach_split[si]
        reach_rows = rng.sample(idx_all, r_total)
        bl: list[str] = []
        for b, cnt in zip(bucket_names, bucket_mat[si]):
            bl += [b] * cnt
        rng.shuffle(bl)
        for i, b in zip(reach_rows, bl):
            rows[i].update(specialization_mode="reachable_fields", reachable_bucket=b)

        # -- stage 4: identity / source / difficulty
        rng = _rng(seed, "stage4", split)
        idl: list[str] = []
        for k, c in zip(ident_names, ident_mat[si]):
            idl += [k] * c
        rng.shuffle(idl)
        for r, k in zip(rows, idl):
            r["field_identity"] = k

        counts = dict(zip(src_names, src_mat[si]))
        asr_idx = sorted(i for i in idx_all if rows[i]["asr_noise"])
        seed_rows = _pick(rng, asr_idx, counts["realistic_asr_seed"], "realistic_asr_seed")
        for i in seed_rows:
            rows[i]["source"] = "realistic_asr_seed"
        left = [i for i in idx_all if rows[i]["source"] is None]
        sl: list[str] = []
        for k in src_names:
            if k != "realistic_asr_seed":
                sl += [k] * counts[k]
        rng.shuffle(sl)
        for i, k in zip(left, sl):
            rows[i]["source"] = k

        scored = sorted(
            idx_all,
            key=lambda i: (2.0 * rows[i]["correction"] + 1.0 * rows[i]["asr_noise"]
                           + BUCKET_RANK[rows[i]["reachable_bucket"]] + rng.random() * 1.5, i))
        dl: list[str] = []
        for k, c in zip(diff_names, diff_mat[si]):
            dl += [k] * c
        for i, k in zip(scored, dl):
            rows[i]["difficulty"] = k

        # -- stage 5: test partitions
        if split == "test":
            _assign_test_partitions(rows, spec["test_partitions"], _rng(seed, "stage5"))

    # ---- stage 6: families (before realization), then freeze ---------------
    slots: list[Slot] = []
    for split in splits:
        for r in plan_rows[split]:
            sid = f"{exp}-{SPLIT_ABBR[split]}-{r['i']:05d}"
            rng = _rng(seed, "stage6", sid)
            families = _assign_families(r, rng)
            slots.append(Slot(
                id=sid, split=split, primitive=r["primitive"], specialization_mode=r["specialization_mode"],
                correction=r["correction"], asr_noise=r["asr_noise"], source=r["source"],
                field_identity=r["field_identity"], difficulty=r["difficulty"], families=families,
                reachable_bucket=r["reachable_bucket"], correction_subtype=r["correction_subtype"],
                asr_subtype=r["asr_subtype"],
                test_partition=(r["test_partition"] if split == "test" else
                                ("train" if split == "train" else "validation"))))

    errs = audit_plan(slots, spec)
    if errs:
        raise QuotaError("plan failed audit: " + "; ".join(f"{e['name']} exp={e['expected']} got={e['actual']}" for e in errs))
    return slots


def _split_total(total: int, weights: list[int]) -> list[int]:
    """Largest-remainder split of ``total`` proportional to ``weights``."""
    s = sum(weights)
    base = [(total * w) // s for w in weights]
    rem = [(total * w) % s for w in weights]
    for i in sorted(range(len(weights)), key=lambda i: (-rem[i], i))[: total - sum(base)]:
        base[i] += 1
    return base


def _assign_test_partitions(rows: list[dict], quotas: dict, rng: random.Random) -> None:
    idx = list(range(len(rows)))
    pool = list(idx)
    rng.shuffle(pool)

    def take(k, pred, name):
        elig = [i for i in pool if pred(rows[i])]
        if len(elig) < k:
            raise QuotaError(f"test partition {name}: need {k}, only {len(elig)} eligible")
        chosen = elig[:k]
        s = set(chosen)
        pool[:] = [i for i in pool if i not in s]
        for i in chosen:
            rows[i]["test_partition"] = name

    # most constrained first
    take(quotas["unseen_corrections"], lambda r: r["correction"], "unseen_corrections")
    take(quotas["asr_robustness"], lambda r: r["asr_noise"], "asr_robustness")
    take(quotas["hard_negatives_adversarial"], lambda r: r["primitive"] == "no_change", "hard_negatives_adversarial")
    take(quotas["unseen_enums"], lambda r: r["primitive"] == "enum", "unseen_enums")
    take(quotas["unseen_numeric_compositions"], lambda r: r["primitive"] in NUMERIC, "unseen_numeric_compositions")
    for name in ("unseen_fields", "unseen_templates", "iid_control"):
        take(quotas[name], lambda r: True, name)
    if pool:
        raise QuotaError(f"{len(pool)} test rows left without partition")


def _assign_families(r: dict, rng: random.Random) -> dict[str, str]:
    part = r["test_partition"] if r["split"] == "test" else None
    F = fam
    families: dict[str, str] = {}
    families["field"] = (rng.choice(sorted(F.HOLDOUT_IDS["field"])) if part == "unseen_fields"
                         else rng.choice(F.GENERAL_IDS["field"]))
    families["template"] = (sorted(F.HOLDOUT_IDS["template"])[0] if part == "unseen_templates"
                            else rng.choice(F.GENERAL_IDS["template"]))
    if r["primitive"] in NUMERIC:
        families["numeric"] = "numf_holdout" if part == "unseen_numeric_compositions" else "numf_general"
    else:
        families["numeric"] = "none"
    if r["primitive"] == "enum":
        families["enum"] = "enum_holdout" if part == "unseen_enums" else "enum_general"
    else:
        families["enum"] = "none"
    if r["correction"]:
        families["correction"] = (sorted(F.HOLDOUT_IDS["correction"])[0] if part == "unseen_corrections"
                                  else rng.choice(F.GENERAL_IDS["correction"]))
    else:
        families["correction"] = "none"
    return families


# ------------------------------------------------------------------ audit --

def audit_plan(slots: list[Slot], spec: dict) -> list[dict]:
    """Return failing checks (empty list == exact reconciliation)."""
    errs = failed(check_quotas([s.to_dict() for s in slots], spec))
    ids = [s.id for s in slots]
    if len(set(ids)) != len(ids):
        errs.append({"name": "unique_slot_ids", "expected": len(ids), "actual": len(set(ids)), "ok": False})
    # family isolation at plan level
    for s in slots:
        for axis in fam.AXES:
            fid = s.families.get(axis)
            if fid in fam.HOLDOUT_IDS[axis] and s.test_partition != fam.AXIS_PARTITION[axis]:
                errs.append({"name": f"holdout_family_leak:{axis}", "expected": 0, "actual": s.id, "ok": False})
                break
        if s.test_partition in fam.PARTITION_AXIS and s.families[fam.PARTITION_AXIS[s.test_partition]] not in fam.HOLDOUT_IDS[fam.PARTITION_AXIS[s.test_partition]]:
            errs.append({"name": "partition_missing_holdout_family", "expected": s.test_partition, "actual": s.id, "ok": False})
    return errs


def plan_to_jsonl(slots: list[Slot]) -> str:
    return "".join(json.dumps(s.to_dict(), sort_keys=True, separators=(",", ":")) + "\n" for s in slots)
