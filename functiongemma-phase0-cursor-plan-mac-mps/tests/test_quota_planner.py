import json
from collections import Counter

from form_sft.dataset import families as fam
from form_sft.dataset.quota_planner import QuotaError, Slot, apportion, audit_plan, plan_slots, plan_to_jsonl
from form_sft.dataset.quotas import check_quotas, failed


def test_25000_unique_slots(slots):
    assert len(slots) == 25000
    assert len({s.id for s in slots}) == 25000
    assert all(isinstance(s, Slot) for s in slots)


def test_all_quotas_exact(slots, spec):
    checks = check_quotas([s.to_dict() for s in slots], spec)
    assert len(checks) > 100
    assert failed(checks) == []
    assert audit_plan(slots, spec) == []


def test_global_equations(slots):
    c = Counter((s.correction, s.asr_noise) for s in slots)
    assert c[(False, False)] == 17000 and c[(True, False)] == 4250 and c[(False, True)] == 3000 and c[(True, True)] == 750
    both = Counter(s.split for s in slots if s.correction and s.asr_noise)
    assert both == {"train": 600, "validation": 75, "test": 75}
    assert sum(s.specialization_mode == "reachable_fields" for s in slots) == 6250
    assert Counter(s.reachable_bucket for s in slots if s.reachable_bucket) == {"2": 3125, "3": 1875, "4": 625, "5": 375, "6_to_8": 250}


def test_same_seed_byte_identical(spec, slots):
    assert plan_to_jsonl(plan_slots(spec)) == plan_to_jsonl(slots)


def test_different_seed_changes_assignment_not_quotas(spec, slots):
    s2 = json.loads(json.dumps(spec)); s2["spec"]["random_seed"] += 1
    other = plan_slots(s2)
    assert plan_to_jsonl(other) != plan_to_jsonl(slots)
    assert failed(check_quotas([s.to_dict() for s in other], s2)) == []


def test_test_partitions_and_families_assigned_before_realization(slots, spec):
    tp = Counter(s.test_partition for s in slots if s.split == "test")
    assert dict(tp) == spec["test_partitions"]
    for s in slots:
        assert set(s.families) == set(fam.AXES)
        for axis in fam.AXES:
            held = s.families[axis] in fam.HOLDOUT_IDS[axis]
            assert held == (s.test_partition == fam.AXIS_PARTITION[axis])


def test_partition_eligibility(slots):
    for s in slots:
        if s.test_partition == "hard_negatives_adversarial": assert s.primitive == "no_change"
        if s.test_partition == "unseen_enums": assert s.primitive == "enum"
        if s.test_partition == "unseen_numeric_compositions": assert s.primitive in ("integer", "number")
        if s.test_partition == "unseen_corrections": assert s.correction
        if s.test_partition == "asr_robustness": assert s.asr_noise


def test_apportion_exact_marginals():
    m = apportion([5000, 625, 625], [3125, 1875, 625, 375, 250])
    assert [sum(r) for r in m] == [5000, 625, 625]
    assert [sum(c) for c in zip(*m)] == [3125, 1875, 625, 375, 250]


def test_apportion_rejects_inconsistent_marginals():
    import pytest
    with pytest.raises(QuotaError):
        apportion([1, 2], [5])
