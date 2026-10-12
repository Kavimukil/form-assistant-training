from form_sft.ablation.ladder import ABLATION_ORDER, include_row, rung_level


def test_rung_levels():
    assert rung_level("A0") == 0
    assert rung_level("A1") == 1
    assert rung_level("A8") == 8


def test_include_cumulative_nesting():
    base = {
        "specialization_mode": "active_field",
        "source": "deterministic_template",
        "primitive": "integer",
        "correction": False,
        "asr_noise": False,
    }
    assert include_row(base, 1)
    assert all(include_row(base, L) for L in range(1, 9))

    curated = {**base, "source": "curated_manual"}
    assert not include_row(curated, 1)
    assert include_row(curated, 2)

    nc = {**base, "primitive": "no_change"}
    assert not include_row(nc, 2)
    assert include_row(nc, 3)

    corr = {**base, "correction": True}
    assert not include_row(corr, 3)
    assert include_row(corr, 4)

    asr = {**base, "asr_noise": True}
    assert not include_row(asr, 4)
    assert include_row(asr, 5)

    teacher = {**base, "source": "teacher_generated"}
    assert not include_row(teacher, 5)
    assert include_row(teacher, 6)

    real = {**base, "source": "realistic_asr_seed", "asr_noise": True}
    assert not include_row(real, 6)
    assert include_row(real, 7)

    reachable = {**base, "specialization_mode": "reachable_fields"}
    assert not include_row(reachable, 7)
    assert include_row(reachable, 8)


def test_ablation_order_complete():
    assert ABLATION_ORDER == ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8")
