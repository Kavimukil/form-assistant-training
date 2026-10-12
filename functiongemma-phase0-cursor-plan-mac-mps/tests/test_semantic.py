import random

import pytest

from form_sft.contracts.models import CanonicalRow, Operation, SemanticContext
from form_sft.generation import catalog as cat
from form_sft.generation.semantic import generate_semantic_truth
from form_sft.validation.semantic import validate_row
from form_sft.validation.structural import validate_structural

SEED = 20261006


def row_for(sample, slot):
    md = {k: getattr(slot, k) for k in ("split", "primitive", "specialization_mode", "reachable_bucket", "correction", "correction_subtype",
                                          "asr_noise", "asr_subtype", "source", "field_identity", "difficulty", "test_partition", "families")}
    md.update(experiment_id="t", subject_field_id=sample["subject_field_id"])
    return CanonicalRow(id=slot.id, metadata=md, semantic_context=SemanticContext(**sample["context"]), utterance="x", target=Operation(**sample["target"]))


def test_catalog_labels_and_ids_unique():
    assert len({d.label for d in cat.FIELD_DEFS}) == len(cat.FIELD_DEFS)
    for ident in cat.IDENTITIES:
        assert len({cat.field_id(d, ident) for d in cat.FIELD_DEFS}) == len(cat.FIELD_DEFS)
    assert {cat.field_id(d, "opaque")[:4] for d in cat.FIELD_DEFS} == {"fld_"}


def test_2000_samples_valid_and_deterministic(slots):
    for s in random.Random(0).sample(slots, 2000):
        a = generate_semantic_truth(s, SEED)
        assert a == generate_semantic_truth(s, SEED)
        r = row_for(a, s)
        assert validate_structural(r) == [] and validate_row(r) == []
        assert (a["target"]["name"] == "no_change") == (s.primitive == "no_change")
        assert len(a["context"]["reachable_field_ids"]) == a["n_candidates"]


def test_every_primitive_and_mode_covered(slots):
    seen = set()
    for s in slots[::7]:
        a = generate_semantic_truth(s, SEED)
        seen.add((s.primitive, s.specialization_mode, a["target"]["name"]))
    prims = {p for p, _, _ in seen}
    assert prims == {"integer", "number", "boolean", "enum", "string", "date", "no_change"}
    assert {m for _, m, _ in seen} == {"active_field", "reachable_fields"}


def test_opaque_ids_supported(slots):
    s = next(x for x in slots if x.field_identity == "opaque" and x.primitive == "enum")
    a = generate_semantic_truth(s, SEED)
    assert a["target"]["arguments"]["field_id"].startswith("fld_")
    assert a["target"]["arguments"]["option_id"].startswith("opt_")


def test_no_change_includes_other_person_and_unresolved(slots):
    kinds = {generate_semantic_truth(s, SEED)["nc_kind"] for s in slots if s.primitive == "no_change"}
    assert {"other_person", "unresolved", "nonreachable_field"} <= kinds


def test_different_seed_changes_truth(slots):
    s = slots[100]
    assert generate_semantic_truth(s, SEED) != generate_semantic_truth(s, SEED + 1)


def test_correction_chain_distinct(slots):
    for s in [x for x in slots if x.correction and x.primitive != "no_change"][:300]:
        a = generate_semantic_truth(s, SEED)
        chain = a["old_values"] + [a["value"]]
        assert all(x != y for x, y in zip(chain, chain[1:]))
        assert len(a["old_values"]) == (2 if s.correction_subtype == "multiple_corrections" else 1)
