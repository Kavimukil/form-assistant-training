from form_sft.evaluation.evaluate import aggregate_metrics, score_prediction, PredRow


def test_exact_and_fmr_scoring():
    gold_nc = {"name": "no_change", "arguments": {}}
    assert score_prediction(gold_nc, {"name": "no_change", "arguments": {}})["exact_call"]
    assert score_prediction(gold_nc, {"name": "set_integer", "arguments": {"field_id": "a", "value": 1}})[
        "false_mutation"
    ]
    gold = {"name": "set_integer", "arguments": {"field_id": "age", "value": 29}}
    assert score_prediction(gold, {"name": "set_integer", "arguments": {"field_id": "age", "value": 29}})["exact_call"]
    assert score_prediction(gold, {"name": "set_integer", "arguments": {"field_id": "age", "value": 30}})["field"]
    assert not score_prediction(gold, {"name": "set_integer", "arguments": {"field_id": "age", "value": 30}})["value"]


def test_aggregate_no_change_pr():
    rows = [
        PredRow("1", {"name": "no_change", "arguments": {}}, {"name": "no_change", "arguments": {}}, "", True,
                {"flags": score_prediction({"name": "no_change", "arguments": {}}, {"name": "no_change", "arguments": {}})}),
        PredRow("2", {"name": "no_change", "arguments": {}}, {"name": "set_boolean", "arguments": {"field_id": "x", "value": True}}, "", False,
                {"flags": score_prediction({"name": "no_change", "arguments": {}}, {"name": "set_boolean", "arguments": {"field_id": "x", "value": True}})}),
        PredRow("3", {"name": "set_integer", "arguments": {"field_id": "a", "value": 1}}, {"name": "no_change", "arguments": {}}, "", False,
                {"flags": score_prediction({"name": "set_integer", "arguments": {"field_id": "a", "value": 1}}, {"name": "no_change", "arguments": {}})}),
    ]
    m = aggregate_metrics(rows)
    assert m["n"] == 3
    assert m["false_mutation_rate"] == 0.5
    assert m["no_change_precision"] == 0.5  # tp=1 fp=1
    assert m["no_change_recall"] == 0.5     # tp=1 fn=1
