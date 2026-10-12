from form_sft.contracts.models import CanonicalRow, Operation, SemanticContext, SemanticField
from form_sft.evaluation.evaluate import PredRow, evaluate_go_no_go, score_prediction


def _row(rid: str, *, primitive: str, mode: str, target: dict, correction=False, asr=False, bucket=None):
    fields = [SemanticField(field_id="age", label="Age", type="integer")]
    ctx = SemanticContext(
        revision=1,
        fingerprint="sha256:t",
        active_field_id="age" if mode == "active_field" else None,
        reachable_field_ids=["age"],
        fields=fields,
    )
    md = {
        "experiment_id": "e",
        "split": "test",
        "primitive": primitive,
        "specialization_mode": mode,
        "reachable_bucket": bucket,
        "correction": correction,
        "asr_noise": asr,
        "difficulty": "D1",
        "test_partition": "iid_control",
        "subject_field_id": "age",
    }
    return CanonicalRow(
        id=rid,
        metadata=md,
        semantic_context=ctx,
        utterance="x",
        target=Operation(**target),
    )


def _pred(row: CanonicalRow, pred: dict | None) -> PredRow:
    gold = {"name": row.target.name, "arguments": dict(row.target.arguments)}
    flags = score_prediction(gold, pred)
    return PredRow(row.id, gold, pred, "", flags["exact_call"], {"flags": flags})


def test_go_no_go_passes_thresholds():
    rows = [
        _row("1", primitive="integer", mode="active_field",
             target={"name": "set_integer", "arguments": {"field_id": "age", "value": 1}}),
        _row("2", primitive="no_change", mode="active_field",
             target={"name": "no_change", "arguments": {}}),
        _row("3", primitive="integer", mode="reachable_fields", bucket="2",
             target={"name": "set_integer", "arguments": {"field_id": "age", "value": 2}}),
    ]
    canon = {r.id: r for r in rows}
    preds = [
        _pred(rows[0], {"name": "set_integer", "arguments": {"field_id": "age", "value": 1}}),
        _pred(rows[1], {"name": "no_change", "arguments": {}}),
        _pred(rows[2], {"name": "set_integer", "arguments": {"field_id": "age", "value": 2}}),
    ]
    thr = {
        "active_field": {
            "primitive_extraction": 0.99,
            "integer": 0.99,
            "no_change_precision": 0.99,
            "no_change_recall": 0.97,
        },
        "reachable_fields": {"candidates_2": 0.98},
        "safety": {"false_mutation_rate_max": 0.005, "stretch": 0.001},
    }
    report = evaluate_go_no_go(preds, canon, thr)
    assert report["go"]
    assert report["fmr_stretch"]["met"]


def test_go_no_go_fails_fmr():
    rows = [
        _row("1", primitive="no_change", mode="active_field",
             target={"name": "no_change", "arguments": {}}),
    ]
    canon = {r.id: r for r in rows}
    preds = [
        _pred(rows[0], {"name": "set_integer", "arguments": {"field_id": "age", "value": 1}}),
    ]
    thr = {"safety": {"false_mutation_rate_max": 0.005}, "active_field": {}, "reachable_fields": {}}
    report = evaluate_go_no_go(preds, canon, thr)
    assert not report["go"]
    assert any(c["name"] == "safety.false_mutation_rate" and not c["ok"] for c in report["checks"])
