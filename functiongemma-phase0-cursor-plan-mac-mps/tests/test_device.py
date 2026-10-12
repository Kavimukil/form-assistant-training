from form_sft.training.device import select_device


def test_cpu_device_is_always_available():
    plan = select_device("cpu")
    assert plan.device == "cpu"
    assert plan.bf16 is False
    assert plan.fp16 is False
