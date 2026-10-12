"""Shared tiny spec for universal dataset tests."""
from __future__ import annotations

from copy import deepcopy

TINY = {
    "spec": {
        "experiment_id": "universal-tiny",
        "random_seed": 7,
        "model_id": "google/functiongemma-270m-it",
    },
    "splits": {"train": 24, "validation": 10, "test": 24},
    "families": {"train": 2, "validation": 2, "test": 2},
    "ambiguity_quarantine": 2,
    "coverage": {
        "single_set": {"train": 12, "validation": 6, "test": 12},
        "multi_field": {"train": 2, "validation": 0, "test": 2},
        "correction": {"train": 2, "validation": 0, "test": 2},
        "clear": {"train": 2, "validation": 0, "test": 0},
        "no_change": {"train": 2, "validation": 2, "test": 2},
        "asr": {"train": 0, "validation": 0, "test": 2},
        "code_mixed": {"train": 2, "validation": 0, "test": 0},
        "option_permutation": {"train": 0, "validation": 0, "test": 2},
        "counterfactual": {"train": 2, "validation": 2, "test": 2},
    },
    "go_no_go": {
        "false_mutation_rate_max": 0.005,
        "false_mutation_regression_max": 0.001,
        "min_macro_delta": 0.03,
        "max_suite_regression": 0.02,
        "min_label_wilson": 0.99,
    },
    "inference": {"contract": "production_json", "decoding": "greedy", "max_seq_length": 1024},
}


def tiny_spec() -> dict:
    spec = deepcopy(TINY)
    # code_mixed train 2 replaces nothing; recount train.
    # 12+2+2+2+2+0+2+0+2 = 24. test 12+2+2+0+2+2+0+2+2 = 24. val 6+0+0+0+2+0+0+0+2 = 10.
    return spec
