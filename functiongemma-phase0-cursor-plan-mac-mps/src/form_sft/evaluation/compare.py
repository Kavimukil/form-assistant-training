"""Compare two checkpoints only when the suite and inference settings match."""
from __future__ import annotations

import hashlib
import json
from typing import Any


class CompareError(ValueError):
    pass


def _sha(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def inference_config_sha(config: dict) -> str:
    return _sha(config)


def compare_checkpoints(
    existing: dict[str, Any],
    candidate: dict[str, Any],
    manifest: dict[str, Any],
    inference_config: dict[str, Any],
) -> dict[str, Any]:
    expected_suite = manifest.get("sha256")
    expected_infer = inference_config_sha(inference_config)
    for name, report in (("existing", existing), ("candidate", candidate)):
        if report.get("suite_sha256") != expected_suite:
            raise CompareError(f"{name} suite hash does not match the manifest")
        if report.get("inference_config_sha256") != expected_infer:
            raise CompareError(f"{name} inference config does not match")
    suite_rows = []
    for suite, existing_metrics in (existing.get("suites") or {}).items():
        candidate_metrics = (candidate.get("suites") or {}).get(suite)
        if candidate_metrics is None:
            raise CompareError(f"candidate missing suite {suite}")
        suite_rows.append(
            {
                "suite": suite,
                "existing_exact_action_set": existing_metrics.get("exact_action_set"),
                "candidate_exact_action_set": candidate_metrics.get("exact_action_set"),
                "delta": candidate_metrics.get("exact_action_set", 0) - existing_metrics.get("exact_action_set", 0),
                "existing_false_mutation_rate": existing_metrics.get("false_mutation_rate"),
                "candidate_false_mutation_rate": candidate_metrics.get("false_mutation_rate"),
                "n": candidate_metrics.get("n"),
            }
        )
    return {
        "suite_sha256": expected_suite,
        "inference_config_sha256": expected_infer,
        "existing_model": existing.get("model_id"),
        "candidate_model": candidate.get("model_id"),
        "overall_delta": candidate.get("overall", {}).get("exact_action_set", 0)
        - existing.get("overall", {}).get("exact_action_set", 0),
        "suites": suite_rows,
        "bootstrap": candidate.get("bootstrap"),
    }
