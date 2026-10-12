import json
from pathlib import Path

from form_sft.audit.generalization import stratified_review, train_inference_diff


def test_review_sample_and_truncated_mismatch(tmp_path):
    path = tmp_path / "test.jsonl"
    rows = []
    for index, primitive in enumerate(["integer", "enum", "no_change"]):
        rows.append(
            {
                "id": f"r{index}",
                "metadata": {"primitive": primitive, "test_partition": "unseen_fields"},
                "utterance": f"utterance {index}",
                "target": {"name": "set_integer", "arguments": {}},
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    sample = stratified_review(path, n=3)
    assert {row["primitive"] for row in sample} == {"integer", "enum", "no_change"}
    assert any(item["code"] == "TOKENIZER" for item in train_inference_diff()["categories"])
    pointer = tmp_path / "pointer.jsonl"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\n")
    assert stratified_review(pointer, n=5) == []
