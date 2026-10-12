"""GO/NO-GO for universal unseen-schema training.

Thresholds are product limits plus the requirement that a candidate beat the
matched baseline. Missing candidate metrics are a NO-GO, and a NO-GO does not
authorize the 25,000-row dataset.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CRITICAL_SUITES = (
    "unseen_label",
    "opaque_id",
    "option_permutation",
    "counterfactual_context",
    "no_change",
)


def decide_go(
    *,
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    qc: dict[str, Any],
    serialization_passed: bool,
    label_wilson_lower: float,
    bootstrap_low: float | None,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    limits = {
        "false_mutation_rate_max": 0.005,
        "false_mutation_regression_max": 0.001,
        "min_macro_delta": 0.03,
        "max_suite_regression": 0.02,
        "min_label_wilson": 0.99,
    }
    if thresholds:
        limits.update(thresholds)
    checks = []

    def add(name: str, ok: bool, detail: Any) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add("qc_passed", bool(qc.get("passed")), qc.get("passed"))
    add("serialization_passed", serialization_passed, serialization_passed)
    add("label_precision_wilson", label_wilson_lower >= limits["min_label_wilson"], label_wilson_lower)
    add("candidate_metrics_present", candidate is not None and baseline is not None, bool(candidate and baseline))
    if baseline and candidate:
        base_fmr = float(baseline.get("false_mutation_rate", 1))
        cand_fmr = float(candidate.get("false_mutation_rate", 1))
        add("false_mutation_cap", cand_fmr <= limits["false_mutation_rate_max"], cand_fmr)
        add(
            "false_mutation_regression",
            cand_fmr <= base_fmr + limits["false_mutation_regression_max"],
            {"baseline": base_fmr, "candidate": cand_fmr},
        )
        macro_delta = float(candidate.get("macro_exact_delta", -1))
        add("macro_delta", macro_delta >= limits["min_macro_delta"], macro_delta)
        add("bootstrap_low_positive", bootstrap_low is not None and bootstrap_low > 0, bootstrap_low)
        regressions = []
        for suite in CRITICAL_SUITES:
            before = float((baseline.get("suites") or {}).get(suite, {}).get("exact_action_set", 0))
            after = float((candidate.get("suites") or {}).get(suite, {}).get("exact_action_set", 0))
            if after < before - limits["max_suite_regression"]:
                regressions.append({"suite": suite, "before": before, "after": after})
        add("critical_suites", not regressions, regressions)
    go = all(item["ok"] for item in checks)
    return {"go": go, "decision": "GO" if go else "NO-GO", "checks": checks, "thresholds": limits}


def materialize_full_allowed(decision_path: str | Path) -> bool:
    path = Path(decision_path)
    if not path.exists():
        return False
    payload = json.loads(path.read_text())
    return bool(payload.get("go")) and payload.get("signed") is True
