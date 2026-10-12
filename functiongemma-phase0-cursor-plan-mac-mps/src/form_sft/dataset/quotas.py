"""Exact-quota checker shared by the planner (G2) and the corpus audit (G5).

Works on flat records (dict) with keys:
  split, primitive, specialization_mode, reachable_bucket, correction,
  correction_subtype, asr_noise, asr_subtype, source, field_identity,
  difficulty, test_partition
Tolerance is always 0.
"""
from __future__ import annotations

from collections import Counter

SPLITS = ("train", "validation", "test")


def _norm_key(k) -> str:
    return str(k)


def check_quotas(records: list[dict], spec: dict) -> list[dict]:
    """Return a list of check dicts {name, expected, actual, ok}."""
    checks: list[dict] = []

    def add(name, expected, actual):
        checks.append({"name": name, "expected": expected, "actual": actual, "ok": expected == actual})

    def cnt(pred) -> int:
        return sum(1 for r in records if pred(r))

    add("total_rows", spec["spec"]["total_rows"], len(records))
    for s, n in spec["splits"].items():
        add(f"split:{s}", n, cnt(lambda r, s=s: r["split"] == s))

    for p, d in spec["primitives"].items():
        add(f"primitive:{p}", d["total"], cnt(lambda r, p=p: r["primitive"] == p))
        for s, n in d["split"].items():
            add(f"primitive:{p}:{s}", n, cnt(lambda r, p=p, s=s: r["primitive"] == p and r["split"] == s))

    for mode, d in spec["specialization"].items():
        add(f"specialization:{mode}", d["total"], cnt(lambda r, m=mode: r["specialization_mode"] == m))
        for s, n in d["split"].items():
            add(f"specialization:{mode}:{s}", n,
                cnt(lambda r, m=mode, s=s: r["specialization_mode"] == m and r["split"] == s))
    for b, n in spec["specialization"]["reachable_fields"]["candidate_counts"].items():
        add(f"reachable_bucket:{_norm_key(b)}", n, cnt(lambda r, b=b: r["reachable_bucket"] == _norm_key(b)))
    add("reachable_bucket:none_for_active", spec["specialization"]["active_field"]["total"],
        cnt(lambda r: r["reachable_bucket"] is None))
    add("reachable_bucket:only_reachable_mode", 0,
        cnt(lambda r: (r["reachable_bucket"] is None) != (r["specialization_mode"] == "active_field")))

    corr = spec["phenomena"]["correction"]
    add("correction:total", corr["total"], cnt(lambda r: r["correction"]))
    for s, n in corr["split"].items():
        add(f"correction:{s}", n, cnt(lambda r, s=s: r["correction"] and r["split"] == s))
    for st, n in corr["subtypes"].items():
        add(f"correction_subtype:{st}", n, cnt(lambda r, st=st: r["correction_subtype"] == st))
    add("correction_subtype:null_iff_clean", 0,
        cnt(lambda r: bool(r["correction"]) != (r["correction_subtype"] is not None)))

    asr = spec["phenomena"]["asr_noise"]
    add("asr:total", asr["total"], cnt(lambda r: r["asr_noise"]))
    for s, n in asr["split"].items():
        add(f"asr:{s}", n, cnt(lambda r, s=s: r["asr_noise"] and r["split"] == s))
    for st, n in asr["subtypes"].items():
        add(f"asr_subtype:{st}", n, cnt(lambda r, st=st: r["asr_subtype"] == st))
    add("asr_subtype:null_iff_clean", 0, cnt(lambda r: bool(r["asr_noise"]) != (r["asr_subtype"] is not None)))

    inter = spec["intersection"]
    add("intersection:clean", inter["clean"], cnt(lambda r: not r["correction"] and not r["asr_noise"]))
    add("intersection:correction_only", inter["correction_only"], cnt(lambda r: r["correction"] and not r["asr_noise"]))
    add("intersection:asr_only", inter["asr_only"], cnt(lambda r: r["asr_noise"] and not r["correction"]))
    add("intersection:correction_and_asr", inter["correction_and_asr"], cnt(lambda r: r["asr_noise"] and r["correction"]))
    add("intersection:overlap_constraint", spec["constraints"]["correction_asr_overlap"],
        cnt(lambda r: r["asr_noise"] and r["correction"]))

    for k, n in spec["realization_source"].items():
        add(f"realization_source:{k}", n, cnt(lambda r, k=k: r["source"] == k))
    for k, n in spec["field_identity"].items():
        add(f"field_identity:{k}", n, cnt(lambda r, k=k: r["field_identity"] == k))
    for k, n in spec["difficulty"].items():
        add(f"difficulty:{k}", n, cnt(lambda r, k=k: r["difficulty"] == k))

    tp = Counter(r["test_partition"] for r in records if r["split"] == "test")
    for k, n in spec["test_partitions"].items():
        add(f"test_partition:{k}", n, tp.get(k, 0))
    add("test_partition:no_unknown", 0, sum(v for k, v in tp.items() if k not in spec["test_partitions"]))
    add("test_partition:absent_outside_test", 0,
        cnt(lambda r: r["split"] != "test" and r["test_partition"] in spec["test_partitions"]))

    # structural couplings (planner invariants, not YAML quotas)
    add("coupling:unresolved_is_no_change", 0,
        cnt(lambda r: (r["correction_subtype"] == "unresolved_to_no_change") != (r["correction"] and r["primitive"] == "no_change")))
    add("coupling:number_confusion_numeric", 0,
        cnt(lambda r: r["asr_subtype"] == "number_specific_confusion" and r["primitive"] not in ("integer", "number")))
    add("coupling:asr_seed_is_asr", 0, cnt(lambda r: r["source"] == "realistic_asr_seed" and not r["asr_noise"]))
    return checks


def failed(checks: list[dict]) -> list[dict]:
    return [c for c in checks if not c["ok"]]
