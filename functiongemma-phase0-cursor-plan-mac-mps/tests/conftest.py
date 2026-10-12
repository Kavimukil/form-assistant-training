import pytest, yaml

@pytest.fixture(scope="session")
def spec():
    with open("config/phase0-v1.yaml") as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="session")
def slots(spec):
    from form_sft.dataset.quota_planner import plan_slots
    return plan_slots(spec)
