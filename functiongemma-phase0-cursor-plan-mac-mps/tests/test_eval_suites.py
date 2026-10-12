from form_sft.evaluation.compare import compare_checkpoints, inference_config_sha
from form_sft.evaluation.metrics import aggregate_action_metrics, paired_bootstrap_delta, score_actions
from form_sft.evaluation.suites import write_suites
from form_sft.universal.build import write_dataset
from tests.support_universal import tiny_spec


def test_action_set_metrics_are_order_invariant():
    gold = [
        {"kind": "set", "field_id": "a", "value": 1},
        {"kind": "clear", "field_id": "b", "value": None},
    ]
    pred = [
        {"kind": "clear", "field_id": "b", "value": None},
        {"kind": "set", "field_id": "a", "value": 1},
    ]
    scored = score_actions(gold, pred)
    assert scored["exact_action_set"]
    partial = score_actions(gold, [{"kind": "set", "field_id": "a", "value": 2}])
    assert partial["field_recall"] == 0.5
    assert not partial["value"]
    empty = score_actions([{"kind": "no_change", "field_id": None, "value": None}], [{"kind": "no_change", "field_id": None, "value": None}])
    assert empty["exact_action_set"] and not empty["false_mutation"]
    mutated = score_actions([{"kind": "no_change", "field_id": None, "value": None}], gold)
    assert mutated["false_mutation"]
    assert score_actions(gold, None)["parse_ok"] is False
    metrics = aggregate_action_metrics([scored, empty, mutated], correction_mask=[True, False, False])
    assert metrics["correction"] == 1
    delta = paired_bootstrap_delta([True, False, True], [True, True, True], n_resamples=50, seed=1)
    assert delta["low"] <= delta["mean"] <= delta["high"]
    again = paired_bootstrap_delta([True, False, True], [True, True, True], n_resamples=50, seed=1)
    assert delta == again


def test_suites_and_compare_reject_hash_mismatch(tmp_path):
    spec = tiny_spec()
    write_dataset(spec, tmp_path)
    manifest = write_suites(tmp_path)
    assert manifest["suites"]["opaque_id"]["n"] == spec["splits"]["test"]
    assert manifest["suites"]["counterfactual_context"]["n"] == 2
    inference = spec["inference"]
    sha = inference_config_sha(inference)
    existing = {"suite_sha256": manifest["sha256"], "inference_config_sha256": sha, "overall": {"exact_action_set": 0.4}, "suites": {"opaque_id": {"exact_action_set": 0.4, "false_mutation_rate": 0.2, "n": 4}}, "model_id": "base"}
    candidate = {"suite_sha256": manifest["sha256"], "inference_config_sha256": sha, "overall": {"exact_action_set": 0.5}, "suites": {"opaque_id": {"exact_action_set": 0.5, "false_mutation_rate": 0.0, "n": 4}}, "model_id": "new", "bootstrap": {"low": 0.01}}
    report = compare_checkpoints(existing, candidate, manifest, inference)
    assert abs(report["overall_delta"] - 0.1) < 1e-9
    candidate["suite_sha256"] = "deadbeef"
    try:
        compare_checkpoints(existing, candidate, manifest, inference)
        assert False, "mismatch should fail"
    except ValueError:
        pass
