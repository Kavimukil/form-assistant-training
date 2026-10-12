import copy
import json
import random

import pytest

from form_sft.dataset import families as fam
from form_sft.dataset.builder import BuildError, make_row, row_to_json
from form_sft.dataset.leakage import check_leakage
from form_sft.dataset.quota_planner import plan_slots
from form_sft.generation import catalog as cat

SEED = 20261006


@pytest.fixture(scope="module")
def sample_rows(slots):
    chosen = random.Random(5).sample(slots, 700) + [s for s in slots if s.split == "test" and s.test_partition.startswith("unseen")][:200]
    seen, rows = set(), []
    for s in chosen:
        if s.id in seen:
            continue
        seen.add(s.id)
        rows.append(make_row(s, SEED)[0].model_dump())
    return rows


def test_no_leakage_in_sample(sample_rows):
    assert check_leakage(sample_rows) == []


def test_holdout_fields_only_in_unseen_fields(sample_rows):
    ids = fam.HOLDOUT_FIELD_IDS_BY_PARTITION["unseen_fields"]
    for r in sample_rows:
        used = {f["field_id"] for f in r["semantic_context"]["fields"]} & ids
        assert bool(used) == (r["metadata"]["test_partition"] == "unseen_fields")


def test_unseen_enums_use_holdout_enum_fields(sample_rows):
    ids = fam.HOLDOUT_FIELD_IDS_BY_PARTITION["unseen_enums"]
    rows = [r for r in sample_rows if r["metadata"]["test_partition"] == "unseen_enums"]
    assert rows and all(r["metadata"]["subject_field_id"] in ids for r in rows)


def test_leak_detected_family_id(sample_rows):
    r = copy.deepcopy(next(x for x in sample_rows if x["metadata"]["split"] == "train"))
    r["metadata"]["families"]["template"] = "tpl5"
    assert any(v["code"] == "L_FAMILY_ID" for v in check_leakage([r]))


def test_leak_detected_holdout_field_in_train(sample_rows):
    r = copy.deepcopy(next(x for x in sample_rows if x["metadata"]["split"] == "train"))
    hid = sorted(fam.HOLDOUT_FIELD_IDS_BY_PARTITION["unseen_fields"])[0]
    r["semantic_context"]["fields"][0]["field_id"] = hid
    assert any(v["code"] == "L_FIELD_ID" for v in check_leakage([r]))


def test_leak_detected_cross_split_duplicate(sample_rows):
    a = copy.deepcopy(next(x for x in sample_rows if x["metadata"]["split"] == "train"))
    b = copy.deepcopy(a); b["id"] = "other"; b["metadata"]["split"] = "test"; b["metadata"]["test_partition"] = "iid_control"
    assert any(v["code"] == "L_DUP_CROSS_SPLIT" for v in check_leakage([a, b]))


def test_leak_detected_duplicate_id(sample_rows):
    a = sample_rows[0]
    assert any(v["code"] == "L_DUP_ID" for v in check_leakage([a, copy.deepcopy(a)]))


def test_holdout_template_content_leak(sample_rows):
    r = copy.deepcopy(next(x for x in sample_rows if x["metadata"]["split"] == "train"))
    r["metadata"]["template_variant"] = 5
    assert any(v["code"] == "L_TEMPLATE" for v in check_leakage([r]))


def test_make_row_deterministic_and_attempt_changes(slots):
    s = slots[321]
    assert row_to_json(make_row(s, SEED)[0]) == row_to_json(make_row(s, SEED)[0])
    assert row_to_json(make_row(s, SEED, 1)[0]) != row_to_json(make_row(s, SEED, 0)[0])


def test_build_fails_closed_on_inconsistent_spec(spec, tmp_path):
    from form_sft.dataset.builder import build
    from form_sft.dataset.quota_planner import QuotaError
    broken = json.loads(json.dumps(spec))
    broken["primitives"]["integer"]["total"] += 1  # quota the plan cannot satisfy -> nothing written
    with pytest.raises(QuotaError):
        build(broken, out_root=tmp_path)
    assert not list(tmp_path.rglob("*.jsonl"))


def test_build_fails_closed_on_leakage(spec, tmp_path, monkeypatch, slots):
    import form_sft.dataset.builder as b
    monkeypatch.setattr(b, "plan_slots", lambda sp: slots)
    monkeypatch.setattr(b, "build_rows", lambda sp, sl=None: [])
    monkeypatch.setattr(b, "check_quotas", lambda recs, sp: [])
    monkeypatch.setattr(b, "check_leakage", lambda rows: [{"code": "L_FAMILY_ID", "id": "x", "detail": ""}])
    with pytest.raises(BuildError):
        b.build(spec, out_root=tmp_path)
    assert not list(tmp_path.rglob("*.jsonl"))
