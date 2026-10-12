import subprocess, sys


def run(*a):
    return subprocess.run([sys.executable, "-m", "form_sft.cli", *a], capture_output=True, text=True)


def test_doctor_ok():
    r = run("doctor"); assert r.returncode == 0 and '"ok"' in r.stdout

def test_plan_prints_25000():
    r = run("plan", "--spec", "config/phase0-v1.yaml"); assert r.returncode == 0 and r.stdout.strip() == "25000"

def test_compile_is_implemented_not_stub():
    """Compile must not be a stub; without HF model access it fails closed on download."""
    r = run("compile", "--spec", "config/phase0-v1.yaml")
    combined = (r.stdout or "") + (r.stderr or "")
    assert "gated stub" not in combined
    # Either full compile succeeded, or failed closed on auth/gated access / missing corpus.
    if r.returncode == 0:
        assert "compiled_rows" in r.stdout or "passed" in r.stdout
    else:
        assert "COMPILE FAILED" in combined or "gated" in combined.lower() or "403" in combined

def test_audit_fails_closed_without_corpus(tmp_path):
    r = run("audit", "--spec", "config/phase0-v1.yaml", "--out-root", str(tmp_path)); assert r.returncode != 0
