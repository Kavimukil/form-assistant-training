import random

import pytest

from form_sft.dataset.builder import make_row
from form_sft.generation.generator import generate_semantic_truth, realize_row
from form_sft.generation.realize import ASRTransformError, apply_asr_noise, finalize_text, norm, tokenize, val, prot, render_tokens
from form_sft.generation.teacher import TeacherAdapter, TeacherRejected
from form_sft.generation.values import fraction_words, hundred_composite, int_words, numeric_forms

SEED = 20261006


def test_number_words():
    assert int_words(29) == "twenty nine" and int_words(29, True) == "twenty-nine"
    assert int_words(105) == "one hundred and five"
    assert int_words(12500) == "twelve thousand five hundred"
    assert hundred_composite(1250) == "twelve hundred and fifty"
    assert fraction_words(72.5) == "seventy two and a half"
    assert numeric_forms(72.5, "number")["words"] == "seventy two point five"


def test_target_invariant_across_realization(slots):
    for s in random.Random(1).sample(slots, 600):
        sample = generate_semantic_truth(s, SEED)
        before = (sample["target"], sample["context"])
        import copy
        snap = copy.deepcopy(before)
        realize_row(sample, s, SEED)
        assert (sample["target"], sample["context"]) == snap


def test_every_asr_and_correction_subtype_is_effective(slots):
    seen_asr, seen_corr = set(), set()
    for s in slots:
        if s.asr_subtype in seen_asr and s.correction_subtype in seen_corr:
            continue
        if not (s.asr_noise or s.correction):
            continue
        row, sample = make_row(s, SEED)
        if s.asr_noise:
            seen_asr.add(s.asr_subtype)
            assert row.metadata["asr_applied"]
        if s.correction:
            seen_corr.add(s.correction_subtype)
            assert row.metadata["correction_template"].startswith(s.correction_subtype)
    assert len(seen_asr) == 8 and len(seen_corr) == 7


def test_asr_never_touches_protected_spans(slots):
    for s in [x for x in slots if x.asr_noise][:400]:
        sample = generate_semantic_truth(s, SEED)
        row_text = realize_row(sample, s, SEED)
        real = sample["realization"]
        if sample["nc_kind"] is None:
            assert any(f.lower() in norm(row_text) or norm(f) in norm(row_text) for f in real["value_forms"])


def test_teacher_rejects_unproven_target():
    with pytest.raises(TeacherRejected):
        TeacherAdapter.require_preserved(False, "x")


def test_teacher_provider_stub_preserved(monkeypatch):
    monkeypatch.setenv("FORM_SFT_TEACHER_PROVIDER", "some-provider")
    with pytest.raises(NotImplementedError):
        TeacherAdapter()


def test_tokenize_roundtrip():
    t = "Set " + prot("age") + " to " + val("29") + "."
    assert render_tokens(tokenize(t)) == t
    assert finalize_text(t) == "Set age to 29."


def test_holdout_numeric_forms_present(slots):
    forms = set()
    for s in [x for x in slots if x.test_partition == "unseen_numeric_compositions"][:80]:
        row, _ = make_row(s, SEED)
        forms.add(row.metadata["value_form"])
    assert forms == {"hundred_composite", "fraction_words"}
