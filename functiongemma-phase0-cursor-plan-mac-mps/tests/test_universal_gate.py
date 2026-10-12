import json
from pathlib import Path

from form_sft.ablation.universal import materialize_universal
from form_sft.audit.generalization import PHASE0_COUNTS, train_inference_diff
from form_sft.gates.universal import decide_go, materialize_full_allowed
from form_sft.universal.build import write_dataset
from tests.support_universal import tiny_spec


ROOT = Path(__file__).resolve().parents[1]


def test_phase0_counts_and_mismatch_categories():
    for split, expected in PHASE0_COUNTS.items():
        path = ROOT / "generated/phase0-v1" / f"{split}.jsonl"
        assert sum(1 for line in path.read_text().splitlines() if line.strip()) == expected
    codes = {item["code"] for item in train_inference_diff()["categories"]}
    assert "A0_SCOPE" in codes
    assert "ENUM_NAME" in codes


def test_gate_boundaries_and_ablation_counts(tmp_path):
    spec = tiny_spec()
    write_dataset(spec, tmp_path)
    report = materialize_universal(tmp_path, tmp_path / "ablation", seed=1)
    matched = {item["matched_train_n"] for item in report["rungs"].values()}
    assert len(matched) == 1
    assert report["rungs"]["U0"]["train_n"] <= report["rungs"]["U4"]["train_n"]
    qc = {"passed": True}
    suites = {name: {"exact_action_set": 0.5} for name in ("unseen_label", "opaque_id", "option_permutation", "counterfactual_context", "no_change")}
    baseline = {"false_mutation_rate": 0.0, "suites": suites}
    candidate = {"false_mutation_rate": 0.0, "macro_exact_delta": 0.04, "suites": {name: {"exact_action_set": 0.55} for name in suites}}
    go = decide_go(baseline=baseline, candidate=candidate, qc=qc, serialization_passed=True, label_wilson_lower=0.995, bootstrap_low=0.01)
    assert go["go"]
    unsafe = decide_go(baseline=baseline, candidate={**candidate, "false_mutation_rate": 0.02}, qc=qc, serialization_passed=True, label_wilson_lower=0.995, bootstrap_low=0.01)
    assert not unsafe["go"]
    missing = decide_go(baseline=None, candidate=None, qc=qc, serialization_passed=True, label_wilson_lower=0.995, bootstrap_low=None)
    assert missing["decision"] == "NO-GO"
    decision = tmp_path / "go.json"
    decision.write_text(json.dumps({"go": False, "signed": False}))
    assert materialize_full_allowed(decision) is False
    decision.write_text(json.dumps({"go": True, "signed": True}))
    assert materialize_full_allowed(decision) is True
