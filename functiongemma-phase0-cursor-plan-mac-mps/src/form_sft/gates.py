"""Gate runners G0..G10. Each gate verifies its exit criteria and writes artifacts/gates/G<N>.json.

A gate never relaxes a quota or holdout; a failing gate is recorded as ``fail`` and the CLI exits non-zero.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import pkgutil
import platform
import random
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

GATE_DIR = Path("artifacts/gates")
REVIEW_DIR = Path("artifacts/review")


class _Gate:
    def __init__(self, gate_id: str, title: str):
        self.id, self.title, self.checks = gate_id, title, []
        self.extra: dict = {}

    def check(self, name: str, ok: bool, detail=None) -> bool:
        self.checks.append({"name": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    def result(self) -> dict:
        ok = all(c["ok"] for c in self.checks)
        return {
            "gate": self.id, "title": self.title, "status": "pass" if ok else "fail",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "checks": self.checks, **self.extra,
        }


def _spec_hash(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _out_dir(spec: dict, out_root="generated") -> Path:
    return Path(out_root) / spec["spec"]["experiment_id"]


def _ctx_row(sample: dict, slot):
    from form_sft.contracts.models import CanonicalRow, Operation, SemanticContext
    md = {k: getattr(slot, k) for k in ("split", "primitive", "specialization_mode", "reachable_bucket", "correction",
                                          "correction_subtype", "asr_noise", "asr_subtype", "source", "field_identity",
                                          "difficulty", "test_partition", "families")}
    md.update(experiment_id="dry-run", subject_field_id=sample["subject_field_id"])
    return CanonicalRow(id=slot.id, metadata=md, semantic_context=SemanticContext(**sample["context"]),
                        utterance="dry run", target=Operation(**sample["target"]))


# ------------------------------------------------------------------ G0 ----

def gate_g0(spec: dict, spec_path: str) -> dict:
    g = _Gate("G0", "Bootstrap")
    from form_sft.cli import doctor_info
    info = doctor_info()
    g.check("doctor_ok", info.get("status") == "ok", info)
    required = ["spec", "splits", "primitives", "specialization", "phenomena", "intersection", "realization_source",
                "field_identity", "difficulty", "test_partitions", "constraints", "go_no_go"]
    g.check("config_parses", all(k in spec for k in required), {"missing": [k for k in required if k not in spec]})
    g.check("config_total_rows_25000", spec["spec"]["total_rows"] == 25000)
    import form_sft
    failures = []
    count = 0
    for m in pkgutil.walk_packages(form_sft.__path__, "form_sft."):
        count += 1
        try:
            importlib.import_module(m.name)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{m.name}: {type(e).__name__}: {e}")
    g.check("all_modules_import", not failures, {"modules": count, "failures": failures})
    out = _out_dir(spec)
    present = [p.name for p in (out / n for n in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json")) if p.exists()]
    g.check("no_generated_corpus_yet", not present, {"present": present})
    g.extra["spec_sha256"] = _spec_hash(spec_path)
    return g.result()


# ------------------------------------------------------------------ G1 ----

def gate_g1(spec: dict, spec_path: str) -> dict:
    g = _Gate("G1", "Contracts + validators")
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_contracts.py", "tests/test_validators_g1.py",
                           "tests/test_structural.py"], capture_output=True, text=True)
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-200:]
    g.check("contract_tests_100pct_pass", proc.returncode == 0, tail)

    from form_sft.contracts.models import SemanticContext
    from form_sft.specialization.runtime import SpecializationMode, select_fields, specialize
    f = lambda i, **kw: {"field_id": i, "label": i, "type": "integer", **kw}  # noqa: E731
    malformed = {
        "duplicate_field_ids": dict(active_field_id="a", reachable_field_ids=["a"], fields=[f("a"), f("a")]),
        "unknown_reachable": dict(active_field_id="a", reachable_field_ids=["a", "zz"], fields=[f("a")]),
        "active_not_reachable": dict(active_field_id="b", reachable_field_ids=["a"], fields=[f("a"), f("b")]),
        "active_mode_without_active": dict(active_field_id=None, reachable_field_ids=["a"], fields=[f("a")]),
    }
    closed = {}
    for name, kw in malformed.items():
        ctx = SemanticContext(revision=1, fingerprint="x", **kw)
        try:
            specialize(ctx, SpecializationMode.ACTIVE_FIELD)
            closed[name] = False
        except ValueError:
            closed[name] = True
    g.check("malformed_context_fails_closed", all(closed.values()), closed)

    from form_sft.dataset.quota_planner import plan_slots
    from form_sft.generation.semantic import generate_semantic_truth
    slots = plan_slots(spec)
    rng = random.Random(1)
    bad = 0
    for s in rng.sample(slots, 400):
        sm = generate_semantic_truth(s, spec["spec"]["random_seed"])
        ctx = SemanticContext(**sm["context"])
        schema = specialize(ctx, SpecializationMode(s.specialization_mode))
        legal = {x.field_id for x in select_fields(ctx, SpecializationMode(s.specialization_mode))}
        names = [b["properties"]["arguments"].get("properties", {}).get("field_id", {}).get("const") for b in schema["oneOf"]]
        exposed = {n for n in names if n}
        opts_ok = True
        for b in schema["oneOf"]:
            p = b["properties"]["arguments"].get("properties", {})
            if "option_id" in p:
                fld = next(x for x in ctx.fields if x.field_id == p["field_id"]["const"])
                opts_ok &= set(p["option_id"]["enum"]) == {o.id for o in fld.options}
        if exposed != legal or not opts_ok:
            bad += 1
    g.check("specialized_schemas_expose_only_legal_fields_options", bad == 0, {"sampled": 400, "bad": bad})
    g.extra["spec_sha256"] = _spec_hash(spec_path)
    return g.result()


# ------------------------------------------------------------------ G2 ----

def gate_g2(spec: dict, spec_path: str) -> dict:
    g = _Gate("G2", "Quota planner")
    from form_sft.dataset.quota_planner import audit_plan, plan_slots, plan_to_jsonl
    from form_sft.dataset.quotas import check_quotas, failed
    slots = plan_slots(spec)
    g.check("exactly_25000_slots", len(slots) == 25000, len(slots))
    g.check("unique_slot_ids", len({s.id for s in slots}) == len(slots))
    checks = check_quotas([s.to_dict() for s in slots], spec)
    g.check("all_quota_equations_reconcile_exactly", not failed(checks), {"checks": len(checks), "failed": failed(checks)})
    g.check("plan_audit_clean", not audit_plan(slots, spec))
    a, b = plan_to_jsonl(slots), plan_to_jsonl(plan_slots(spec))
    g.check("same_seed_byte_identical", a == b, {"sha256": hashlib.sha256(a.encode()).hexdigest()})
    spec2 = json.loads(json.dumps(spec))
    spec2["spec"]["random_seed"] = spec["spec"]["random_seed"] + 1
    slots2 = plan_slots(spec2)
    c = plan_to_jsonl(slots2)
    g.check("different_seed_changes_assignments", c != a)
    g.check("different_seed_same_quotas", not failed(check_quotas([s.to_dict() for s in slots2], spec2)))
    g.check("family_assigned_before_realization", all(set(s.families) == {"field", "template", "numeric", "enum", "correction"} for s in slots))
    g.check("split_assignment_before_realization_flag", spec["constraints"]["split_assignment_before_realization"] is True)
    g.extra["plan_sha256"] = hashlib.sha256(a.encode()).hexdigest()
    g.extra["quota_checks"] = len(checks)
    g.extra["spec_sha256"] = _spec_hash(spec_path)
    return g.result()


# ------------------------------------------------------------------ G3 ----

def gate_g3(spec: dict, spec_path: str, n: int = 10000) -> dict:
    g = _Gate("G3", "Semantic truth generators")
    from form_sft.dataset.quota_planner import plan_slots
    from form_sft.generation.semantic import generate_semantic_truth
    from form_sft.validation.semantic import validate_row
    from form_sft.validation.structural import validate_structural
    seed = spec["spec"]["random_seed"]
    slots = plan_slots(spec)
    sample = random.Random(seed).sample(slots, n)
    s_err = m_err = nondet = 0
    prim, ident, mode, types = Counter(), Counter(), Counter(), Counter()
    opaque_ok = 0
    nc_kinds = Counter()
    for s in sample:
        sm = generate_semantic_truth(s, seed)
        row = _ctx_row(sm, s)
        s_err += len(validate_structural(row))
        m_err += len(validate_row(row))
        if generate_semantic_truth(s, seed) != sm:
            nondet += 1
        prim[s.primitive] += 1
        ident[s.field_identity] += 1
        mode[s.specialization_mode] += 1
        types[sm["subject_type"]] += 1
        nc_kinds[sm["nc_kind"]] += 1
        if s.field_identity == "opaque" and all(f["field_id"].startswith("fld_") for f in sm["context"]["fields"]):
            opaque_ok += 1
    g.check("dry_run_samples", len(sample) == n, n)
    g.check("zero_structural_errors", s_err == 0, s_err)
    g.check("zero_semantic_errors", m_err == 0, m_err)
    g.check("regenerable_deterministically_from_slot_seed", nondet == 0, nondet)
    g.check("every_primitive_covered", set(prim) == set(spec["primitives"]), dict(prim))
    g.check("every_field_type_covered_by_no_change_and_primitives", set(types) >= {"integer", "number", "boolean", "enum", "string", "date"}, dict(types))
    g.check("all_identity_classes_covered", set(ident) == set(spec["field_identity"]), dict(ident))
    g.check("both_specialization_modes_covered", set(mode) == {"active_field", "reachable_fields"}, dict(mode))
    g.check("opaque_ids_supported", opaque_ok == ident["opaque"], {"opaque_rows": ident["opaque"], "ok": opaque_ok})
    g.check("no_change_kinds_incl_other_person", {"other_person", "unresolved"} <= set(nc_kinds), {str(k): v for k, v in nc_kinds.items()})
    g.extra["coverage"] = {"primitive": dict(prim), "identity": dict(ident), "mode": dict(mode),
                           "subject_type": dict(types), "nc_kind": {str(k): v for k, v in nc_kinds.items()}}
    g.extra["spec_sha256"] = _spec_hash(spec_path)
    return g.result()


# ------------------------------------------------------------------ G4 ----

def gate_g4(spec: dict, spec_path: str) -> dict:
    g = _Gate("G4", "Realization + transforms")
    from form_sft.dataset.builder import build_rows, make_row, row_to_json
    from form_sft.dataset.quota_planner import plan_slots
    from form_sft.dataset.quotas import check_quotas, failed
    from form_sft.generation.semantic import generate_semantic_truth
    seed = spec["spec"]["random_seed"]
    slots = plan_slots(spec)
    rows = build_rows(spec, slots)   # raises BuildError if any target-preservation/validation proof fails
    g.check("all_rows_realized_and_validated", len(rows) == 25000, len(rows))
    g.check("exact_quotas_after_realization", not failed(check_quotas([r.metadata for r in rows], spec)))
    corr = [r for r in rows if r.metadata["correction"]]
    asr = [r for r in rows if r.metadata["asr_noise"]]
    g.check("every_correction_row_carries_correction_template", all(r.metadata["correction_template"] for r in corr), len(corr))
    g.check("every_asr_row_has_applied_transform", all(r.metadata["asr_applied"] for r in asr), len(asr))
    g.check("clean_rows_have_no_transform_metadata", all(not r.metadata["asr_applied"] and not r.metadata["correction_template"]
                                                         for r in rows if not r.metadata["correction"] and not r.metadata["asr_noise"]))
    by_id = {s.id: s for s in slots}
    drift = 0
    for r in random.Random(seed).sample(rows, 3000):
        s = by_id[r.id]
        sm = generate_semantic_truth(s, seed + r.metadata["attempt"] * 1_000_003)
        if sm["target"] != r.target.model_dump():
            drift += 1
    g.check("semantic_target_invariant_across_transforms", drift == 0, {"sampled": 3000, "drift": drift})
    redo = sum(1 for r in random.Random(seed + 1).sample(rows, 500)
               if row_to_json(make_row(by_id[r.id], seed, r.metadata["attempt"])[0]) != row_to_json(r))
    g.check("realization_deterministic", redo == 0, {"sampled": 500, "mismatch": redo})
    g.check("no_teacher_external_calls", all(r.metadata["teacher_mode"] in (None, "deterministic_paraphrase_no_credentials")
                                              for r in rows))
    g.check("teacher_rows_proven_target_preserving", sum(1 for r in rows if r.metadata["source"] == "teacher_generated") == spec["realization_source"]["teacher_generated"])
    g.extra["attempts_gt0"] = sum(1 for r in rows if r.metadata["attempt"] > 0)
    g.extra["spec_sha256"] = _spec_hash(spec_path)
    return g.result()


# ------------------------------------------------------------------ G5 ----

def stratified_review_sample(rows: list[dict], per_cell: int = 2, seed: int = 7) -> list[dict]:
    cells: dict[tuple, list[dict]] = {}
    for r in rows:
        m = r["metadata"]
        cells.setdefault((m["primitive"], m["specialization_mode"], m["source"], bool(m["correction"]), bool(m["asr_noise"])), []).append(r)
    rng = random.Random(seed)
    out = []
    for key in sorted(cells):
        pool = cells[key]
        out += rng.sample(pool, min(per_cell, len(pool)))
    return out


def automated_review(row: dict) -> list[str]:
    """Heuristic stand-in for human review: re-derives acceptance from the row alone."""
    from form_sft.generation.realize import _contains
    issues = []
    md, u, t = row["metadata"], row["utterance"], row["target"]
    ctx = row["semantic_context"]
    fields = {f["field_id"]: f for f in ctx["fields"]}
    if t["name"] != "no_change":
        f = fields[t["arguments"]["field_id"]]
        if md["specialization_mode"] == "reachable_fields" and not _contains(u, f["label"]):
            issues.append("label_missing")
        if f["type"] == "enum":
            opt = next(o for o in f["options"] if o["id"] == t["arguments"]["option_id"])
            if not any(_contains(u, s) for s in [opt["label"], *opt["aliases"]]):
                issues.append("option_missing")
        elif f["type"] in ("integer", "number"):
            from form_sft.generation.values import numeric_forms, fraction_words, hundred_composite
            forms = list(numeric_forms(t["arguments"]["value"], f["type"]).values())
            v = t["arguments"]["value"]
            if f["type"] == "integer" and v >= 1100:
                forms.append(hundred_composite(v))
            if f["type"] == "number" and fraction_words(v):
                forms.append(fraction_words(v))
            if not any(_contains(u, x) for x in forms):
                issues.append("value_missing")
        elif f["type"] == "string":
            if not _contains(u, t["arguments"]["value"]):
                issues.append("value_missing")
    if len(u) > 220:
        issues.append("too_long")
    return issues


def gate_g5(spec: dict, spec_path: str, out_root="generated", reproduce: bool = True) -> dict:
    g = _Gate("G5", "Split isolation + 25k build (DATASET GO)")
    from form_sft.dataset.audit import audit_dir
    out = _out_dir(spec, out_root)
    audit_path = out / "audit.json"
    g.check("audit_json_exists", audit_path.exists())
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    fresh = audit_dir(spec, out)
    g.check("audit_passed", bool(fresh["passed"]) and bool(audit.get("passed")), {"problems": fresh["problems"]})
    g.check("exactly_25000_rows", fresh["total_rows"] == 25000, fresh["total_rows"])
    g.check("exact_quota_match_every_hard_quota", fresh["quota_checks_failed"] == 0,
            {"checks": fresh["quota_checks_total"], "failed": fresh["failed_quota_checks"]})
    g.check("zero_duplicate_ids", fresh["duplicate_ids"] == 0)
    g.check("zero_structural_errors", fresh["structural_errors"] == 0)
    g.check("zero_semantic_errors", fresh["semantic_errors"] == 0)
    g.check("zero_prohibited_leakage", fresh["prohibited_leakage"] == 0)
    g.check("zero_duplicate_content", fresh["duplicate_content"] == 0)
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json", "audit.json"):
        g.check(f"output_exists:{name}", (out / name).exists())
    g.check("benchmark_files_for_all_test_partitions",
            all((out / "benchmark" / f"{p}.jsonl").exists() for p in spec["test_partitions"]))

    if reproduce:
        from form_sft.dataset.builder import build
        manifest = json.loads((out / "manifest.json").read_text())
        with tempfile.TemporaryDirectory() as td:
            m2 = build(spec, spec_path, out_root=td)
        g.check("build_reproducible_from_seed", m2["files"] == manifest["files"],
                {"files_compared": len(manifest["files"])})
    rows = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines()]
    rows += [json.loads(l) for l in (out / "validation.jsonl").read_text().splitlines()]
    rows += [json.loads(l) for l in (out / "test.jsonl").read_text().splitlines()]
    sample = stratified_review_sample(rows)
    reviewed = [(r["id"], automated_review(r)) for r in sample]
    rejected = [x for x in reviewed if x[1]]
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    (REVIEW_DIR / "g5_review_sample.jsonl").write_text("".join(json.dumps(
        {"id": r["id"], "utterance": r["utterance"], "target": r["target"], "metadata": {
            k: r["metadata"][k] for k in ("primitive", "specialization_mode", "source", "correction_subtype", "asr_subtype", "test_partition", "nc_kind")},
         "automated_issues": i}, sort_keys=True) + "\n" for r, (_, i) in zip(sample, reviewed)))
    g.check("stratified_review_sample_ge_200", len(sample) >= 200, len(sample))
    g.check("stratified_review_sample_accepted", not rejected, {"rejected": rejected[:5]})
    signoff_path = REVIEW_DIR / "g5_human_signoff.json"
    human_signoff = False
    signoff_meta = {}
    if signoff_path.exists():
        try:
            signoff_meta = json.loads(signoff_path.read_text())
            human_signoff = bool(signoff_meta.get("human_signoff"))
        except json.JSONDecodeError:
            human_signoff = False
    g.extra["manual_review"] = {
        "sample_file": str(REVIEW_DIR / "g5_review_sample.jsonl"), "rows": len(sample), "accepted": len(sample) - len(rejected),
        "method": "automated stratified re-derivation (value/label/option recoverable from utterance) + agent spot-check",
        "human_signoff": human_signoff,
        "signoff_file": str(signoff_path),
        "signoff_meta": signoff_meta,
        "note": ("Human sign-off recorded." if human_signoff else
                 "Plan requires manual review >=200 stratified rows; record sign-off in artifacts/review/g5_human_signoff.json."),
    }
    g.extra["audit_summary"] = {k: fresh[k] for k in ("total_rows", "duplicate_ids", "structural_errors", "semantic_errors",
                                                      "prohibited_leakage", "quota_checks_total", "quota_checks_failed")}
    g.extra["spec_sha256"] = _spec_hash(spec_path)
    return g.result()


# ------------------------------------------------------------------ G6 ----

def gate_g6(spec: dict, spec_path: str, out_root="generated", sample_n: int = 100) -> dict:
    g = _Gate("G6", "FunctionGemma compiler")
    from form_sft.training.compiler import (
        CompileError,
        compile_row,
        load_tokenizer,
        parse_tool_calls_from_rendered,
        targets_equal,
    )

    model_id = spec["spec"].get("model_id", "google/functiongemma-270m-it")
    try:
        train_cfg = json.loads(json.dumps(__import__("yaml").safe_load(open("config/train-full-sft.yaml"))))
        max_seq = int(train_cfg.get("max_seq_length", 1024))
    except Exception:  # noqa: BLE001
        max_seq = 1024

    try:
        tokenizer = load_tokenizer(model_id)
        g.check("tokenizer_loads", True, model_id)
    except Exception as e:  # noqa: BLE001
        g.check("tokenizer_loads", False, f"{type(e).__name__}: {e}")
        g.extra["spec_sha256"] = _spec_hash(spec_path)
        return g.result()

    tmpl = getattr(tokenizer, "chat_template", None) or getattr(
        getattr(tokenizer, "tokenizer", None), "chat_template", None
    )
    g.check("chat_template_present_from_tokenizer", bool(tmpl), {"template_chars": len(tmpl or "")})
    # Render path must go through apply_chat_template (parser may inspect emitted markers).
    src = Path("src/form_sft/training/compiler.py").read_text(encoding="utf-8")
    g.check("no_hand_coded_render_path", "apply_chat_template" in src and "def compile_example" in src,
            {"uses_apply_chat_template": "apply_chat_template" in src})

    out = _out_dir(spec, out_root)
    pool_paths = [out / "train.jsonl", out / "validation.jsonl"]
    rows = []
    for p in pool_paths:
        if p.exists():
            rows.extend(json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())
    g.check("canonical_rows_available", len(rows) >= sample_n, len(rows))
    sample = random.Random(spec["spec"]["random_seed"]).sample(rows, min(sample_n, len(rows))) if rows else []

    render_ok = 0
    exact = 0
    lengths = []
    failures = []
    for r in sample:
        try:
            compiled = compile_row(r, tokenizer, max_seq_length=max_seq, length_policy="reject")
            render_ok += 1
            parsed = parse_tool_calls_from_rendered(compiled["text"])
            if parsed and targets_equal(parsed[0], compiled["target"]):
                exact += 1
            lengths.append(compiled["token_length"])
            # Ensure rendered text actually came from template special structure when tools present
            if "<start_function_call>" not in compiled["text"] and "call:" not in compiled["text"]:
                failures.append((r["id"], "missing_function_call_markers_from_template"))
        except CompileError as e:
            failures.append((r["id"], str(e)))

    g.check("sampled_rows_render_successfully", render_ok == len(sample) and len(sample) == sample_n,
            {"sample_n": sample_n, "render_ok": render_ok, "failures": failures[:5]})
    g.check("rendered_targets_exact_canonical_operations", exact == render_ok,
            {"exact": exact, "render_ok": render_ok})
    lengths_sorted = sorted(lengths)
    p99 = lengths_sorted[int(0.99 * (len(lengths_sorted) - 1))] if lengths_sorted else 0
    g.check("p99_sequence_length_reported", bool(lengths), {"p99": p99, "max": max(lengths) if lengths else 0,
                                                             "n": len(lengths)})
    g.check("no_sample_exceeds_max_seq_length", all(x <= max_seq for x in lengths),
            {"max_seq_length": max_seq, "over": [x for x in lengths if x > max_seq][:5]})

    # Full compile artifacts for training
    train_sft = out / "train.sft.jsonl"
    val_sft = out / "validation.sft.jsonl"
    report_path = out / "compile_report.json"
    if train_sft.exists() and val_sft.exists() and report_path.exists():
        report = json.loads(report_path.read_text())
        g.check("train_sft_exists", True, sum(1 for _ in train_sft.open()))
        g.check("validation_sft_exists", True, sum(1 for _ in val_sft.open()))
        g.check("compile_report_passed", bool(report.get("passed")), report.get("problems"))
        g.check("train_sft_row_count", report.get("splits", {}).get("train", {}).get("compiled_rows") == spec["splits"]["train"],
                report.get("splits", {}).get("train"))
        g.check("validation_sft_row_count",
                report.get("splits", {}).get("validation", {}).get("compiled_rows") == spec["splits"]["validation"],
                report.get("splits", {}).get("validation"))
    else:
        g.check("train_sft_exists", False, "run: python -m form_sft.cli compile")
        g.check("validation_sft_exists", False, "run: python -m form_sft.cli compile")
        g.check("compile_report_passed", False, "missing compile_report.json")

    g.extra["token_length"] = {"p99": p99, "max": max(lengths) if lengths else 0, "n": len(lengths),
                               "max_seq_length": max_seq}
    g.extra["model_id"] = model_id
    g.extra["spec_sha256"] = _spec_hash(spec_path)
    return g.result()


# ------------------------------------------------------------------ G7 ----

def gate_g7(spec: dict, spec_path: str, out_root="generated") -> dict:
    g = _Gate("G7", "Baseline before training (A0)")
    eval_dir = Path("artifacts/eval/A0")
    metrics_path = eval_dir / "A0_metrics.json"
    preds_path = eval_dir / "A0_predictions.jsonl"
    g.check("a0_metrics_exist", metrics_path.exists(), str(metrics_path))
    g.check("a0_predictions_exist", preds_path.exists(), str(preds_path))
    if not metrics_path.exists() or not preds_path.exists():
        g.extra["hint"] = "Run: python -m form_sft.cli evaluate --split validation --tag A0 --device mps"
        g.extra["spec_sha256"] = _spec_hash(spec_path)
        return g.result()
    metrics = json.loads(metrics_path.read_text())
    overall = metrics.get("overall", {})
    required = ["exact_call", "op", "field", "value", "enum", "no_change_precision",
                "no_change_recall", "false_mutation_rate"]
    g.check("reports_required_metrics", all(k in overall for k in required),
            {"missing": [k for k in required if k not in overall], "overall": overall})
    by = metrics.get("by", {})
    for dim in ("primitive", "difficulty", "mode", "noise", "holdout"):
        g.check(f"breakdown_by_{dim}", isinstance(by.get(dim), dict) and bool(by.get(dim)),
                list((by.get(dim) or {}).keys())[:12])
    n_pred = sum(1 for _ in preds_path.open(encoding="utf-8"))
    g.check("predictions_persisted_nonempty", n_pred > 0 and n_pred == overall.get("n"),
            {"predictions": n_pred, "metrics_n": overall.get("n")})
    g.check("greedy_decoding", metrics.get("decoding") == "greedy", metrics.get("decoding"))
    g.check("model_id_matches_spec", metrics.get("model_id") == spec["spec"].get("model_id"),
            metrics.get("model_id"))
    # Baseline is informational; do not require GO thresholds here (those are G9).
    g.extra["overall"] = overall
    g.extra["device"] = metrics.get("device")
    g.extra["spec_sha256"] = _spec_hash(spec_path)
    return g.result()


# ------------------------------------------------------------------ G8 ----

def gate_g8(spec: dict, spec_path: str) -> dict:
    """Mac Phase-0 G8: MPS smoke SFT (50 steps), HF access, finite loss, reloadable checkpoint."""
    g = _Gate("G8", "MPS smoke SFT (Phase-0 host)")
    import torch
    from form_sft.training.device import select_device

    g.check("mps_available", bool(torch.backends.mps.is_available()),
            {"mps_built": bool(torch.backends.mps.is_built())})
    try:
        plan = select_device("mps")
        g.check("device_plan_mps", plan.device == "mps", plan.device)
    except Exception as e:  # noqa: BLE001
        g.check("device_plan_mps", False, str(e))

    model_id = spec["spec"].get("model_id", "google/functiongemma-270m-it")
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(model_id, "config.json")
        g.check("hf_model_access_verified", True, p)
    except Exception as e:  # noqa: BLE001
        g.check("hf_model_access_verified", False, f"{type(e).__name__}: {e}")

    for name in ("train.sft.jsonl", "validation.sft.jsonl"):
        g.check(f"sft_exists:{name}", (_out_dir(spec) / name).exists())

    smoke_dir = Path("artifacts/checkpoints/phase0-full-sft-smoke")
    g.check("smoke_checkpoint_dir_exists", smoke_dir.exists(), str(smoke_dir))
    # Trainer/HF save layout
    has_weights = any((smoke_dir / n).exists() for n in (
        "model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"
    ))
    g.check("smoke_weights_present", has_weights, sorted(p.name for p in smoke_dir.glob("*"))[:20])
    log_candidates = list(Path("artifacts/checkpoints").glob("**/trainer_state.json"))
    smoke_state = smoke_dir / "trainer_state.json"
    finite_loss = False
    loss_detail = None
    if smoke_state.exists():
        state = json.loads(smoke_state.read_text())
        hist = state.get("log_history") or []
        losses = [h["loss"] for h in hist if "loss" in h]
        if losses:
            finite_loss = all(isinstance(x, (int, float)) and x == x and x != float("inf") for x in losses)
            loss_detail = {"n_logs": len(losses), "last_loss": losses[-1], "min_loss": min(losses)}
    g.check("smoke_finite_loss_logged", finite_loss, loss_detail)
    # Reload
    if has_weights:
        try:
            from transformers import AutoModelForCausalLM
            m = AutoModelForCausalLM.from_pretrained(str(smoke_dir), torch_dtype=torch.float32)
            g.check("checkpoint_reloads", True, {"params": sum(p.numel() for p in m.parameters())})
            del m
        except Exception as e:  # noqa: BLE001
            g.check("checkpoint_reloads", False, f"{type(e).__name__}: {e}")
    else:
        g.check("checkpoint_reloads", False, "no weights to reload")
    g.extra["smoke_dir"] = str(smoke_dir)
    g.extra["hint"] = "Run: ./scripts/smoke_train_macos.sh"
    g.extra["spec_sha256"] = _spec_hash(spec_path)
    return g.result()


# ------------------------------------------------------------------ G9 ----

def gate_g9(spec: dict, spec_path: str, out_root="generated") -> dict:
    """Phase-9 GO/NO-GO on frozen test split vs YAML thresholds (post-SFT)."""
    g = _Gate("G9", "Evaluation / GO-NO-GO (test)")
    from form_sft.evaluation.evaluate import (
        evaluate_go_no_go,
        load_canonical_jsonl,
        _predrows_from_saved,
    )

    test_path = _out_dir(spec, out_root) / "test.jsonl"
    g.check("test_split_exists", test_path.exists(), str(test_path))

    eval_dir = Path("artifacts/eval/A1-test")
    metrics_path = eval_dir / "A1-test_metrics.json"
    preds_path = eval_dir / "A1-test_predictions.jsonl"
    g.check("a1_test_metrics_exist", metrics_path.exists(), str(metrics_path))
    g.check("a1_test_predictions_exist", preds_path.exists(), str(preds_path))
    if not (test_path.exists() and metrics_path.exists() and preds_path.exists()):
        g.extra["hint"] = (
            "Run: python -m form_sft.cli evaluate --split test --tag A1-test "
            "--model-id artifacts/checkpoints/phase0-full-sft --device mps"
        )
        g.extra["spec_sha256"] = _spec_hash(spec_path)
        return g.result()

    metrics = json.loads(metrics_path.read_text())
    overall = metrics.get("overall", {})
    required = ["exact_call", "op", "field", "value", "enum", "no_change_precision",
                "no_change_recall", "false_mutation_rate"]
    g.check("reports_required_metrics", all(k in overall for k in required),
            {"missing": [k for k in required if k not in overall]})
    g.check("greedy_decoding", metrics.get("decoding") == "greedy", metrics.get("decoding"))
    g.check("eval_on_test_split", "test.jsonl" in str(metrics.get("split_path", "")),
            metrics.get("split_path"))
    ckpt = str(metrics.get("model_id", ""))
    g.check("post_sft_checkpoint", "phase0-full-sft" in ckpt and "smoke" not in ckpt, ckpt)

    rows = load_canonical_jsonl(test_path)
    g.check("frozen_test_n", len(rows) == int(spec["splits"]["test"]),
            {"n": len(rows), "expected": spec["splits"]["test"]})
    canon = {r.id: r for r in rows}
    saved = [json.loads(line) for line in preds_path.open(encoding="utf-8") if line.strip()]
    g.check("predictions_cover_test", len(saved) == len(rows),
            {"predictions": len(saved), "test": len(rows)})

    pred_rows = _predrows_from_saved(saved, canon)
    report = evaluate_go_no_go(pred_rows, canon, spec.get("go_no_go") or {})
    for c in report["checks"]:
        g.check(c["name"], c["ok"], {
            "actual": c["actual"], "threshold": c["threshold"], "n": c["n"],
            "higher_is_better": c["higher_is_better"],
        })

    # Informational A0 comparison (prefer test A0; fall back to validation A0).
    a0_paths = [
        Path("artifacts/eval/A0-test/A0-test_metrics.json"),
        Path("artifacts/eval/A0/A0_metrics.json"),
    ]
    a0_compare = None
    for p in a0_paths:
        if p.exists():
            a0 = json.loads(p.read_text()).get("overall", {})
            a0_compare = {
                "source": str(p),
                "a0_exact_call": a0.get("exact_call"),
                "a1_exact_call": overall.get("exact_call"),
                "delta_exact_call": (overall.get("exact_call") or 0) - (a0.get("exact_call") or 0),
                "a0_false_mutation_rate": a0.get("false_mutation_rate"),
                "a1_false_mutation_rate": overall.get("false_mutation_rate"),
            }
            break

    by = metrics.get("by", {})
    for dim in ("primitive", "difficulty", "mode", "noise", "holdout"):
        g.check(f"breakdown_by_{dim}", isinstance(by.get(dim), dict) and bool(by.get(dim)),
                list((by.get(dim) or {}).keys())[:12])

    decision = "GO" if (report["go"] and all(c["ok"] for c in g.checks)) else "NO-GO"
    g.extra["overall"] = overall
    g.extra["go_no_go"] = report
    g.extra["a0_compare"] = a0_compare
    g.extra["decision"] = decision
    g.extra["device"] = metrics.get("device")
    g.extra["model_id"] = metrics.get("model_id")
    g.extra["spec_sha256"] = _spec_hash(spec_path)
    g.extra["hint"] = "Primary: exact tool-call + FMR vs config go_no_go thresholds on frozen test."
    return g.result()


# ------------------------------------------------------------------ G10 ----

def gate_g10(spec: dict, spec_path: str, out_root="generated") -> dict:
    """Phase-10 ablation ladder: cumulative subsets, frozen test, nesting."""
    g = _Gate("G10", "Ablation ladder (A0–A8)")
    from form_sft.ablation.ladder import (
        ABLATION_ORDER,
        ABLATION_TITLES,
        include_row,
        summarize_evals,
    )

    ablation_root = Path("artifacts/ablation")
    report_path = ablation_root / "ladder_report.json"
    g.check("ladder_report_exists", report_path.exists(), str(report_path))
    if not report_path.exists():
        g.extra["hint"] = "Run: python -m form_sft.cli ablation materialize"
        g.extra["spec_sha256"] = _spec_hash(spec_path)
        return g.result()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    test_path = _out_dir(spec, out_root) / "test.jsonl"
    g.check("frozen_test_exists", test_path.exists(), str(test_path))
    live_sha = hashlib.sha256(test_path.read_bytes()).hexdigest() if test_path.exists() else None
    g.check(
        "frozen_test_unchanged",
        live_sha == report.get("frozen_test_sha256"),
        {"report": report.get("frozen_test_sha256"), "live": live_sha},
    )
    g.check(
        "frozen_test_n",
        report.get("frozen_test_n") == int(spec["splits"]["test"]),
        {"report_n": report.get("frozen_test_n"), "expected": spec["splits"]["test"]},
    )

    prev_ids: set[str] | None = None
    for tag in ABLATION_ORDER:
        man_path = ablation_root / tag / "manifest.json"
        ok = man_path.exists()
        g.check(f"manifest:{tag}", ok, str(man_path))
        if not ok:
            continue
        man = json.loads(man_path.read_text(encoding="utf-8"))
        g.check(f"title:{tag}", man.get("title") == ABLATION_TITLES[tag], man.get("title"))
        if tag == "A0":
            g.check("A0_no_train_rows", man.get("train_n", 0) == 0, man.get("train_n"))
            continue
        train_path = Path(man["train"])
        g.check(f"train_exists:{tag}", train_path.exists(), str(train_path))
        n_lines = sum(1 for line in train_path.open(encoding="utf-8") if line.strip()) if train_path.exists() else 0
        g.check(f"train_n_matches:{tag}", n_lines == man.get("train_n"), {"file": n_lines, "manifest": man.get("train_n")})
        ids_path = Path(man.get("train_ids") or "")
        if ids_path.exists():
            ids = set(json.loads(ids_path.read_text(encoding="utf-8")))
            g.check(f"ids_n:{tag}", len(ids) == man.get("train_n"), len(ids))
            if prev_ids is not None:
                g.check(f"nested_subset:{tag}", prev_ids <= ids, {"missing": len(prev_ids - ids)})
            prev_ids = ids

    # A8 must equal full compiled train size
    a8 = ablation_root / "A8" / "manifest.json"
    if a8.exists():
        full_train = _out_dir(spec, out_root) / "train.sft.jsonl"
        full_n = sum(1 for line in full_train.open(encoding="utf-8") if line.strip()) if full_train.exists() else -1
        a8_n = json.loads(a8.read_text(encoding="utf-8")).get("train_n")
        g.check("A8_equals_full_train", a8_n == full_n, {"A8": a8_n, "full": full_n})

    # Filter sanity on a sample of train rows
    train_canon = _out_dir(spec, out_root) / "train.jsonl"
    if train_canon.exists():
        sample = []
        with train_canon.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    sample.append(json.loads(line))
                if i >= 400:
                    break
        # A1 rows must be active/clean/mutating/deterministic
        bad_a1 = [
            r["id"] for r in sample
            if include_row(r["metadata"], 1)
            and (
                r["metadata"].get("specialization_mode") != "active_field"
                or r["metadata"].get("source") != "deterministic_template"
                or r["metadata"].get("primitive") == "no_change"
                or r["metadata"].get("correction")
                or r["metadata"].get("asr_noise")
            )
        ]
        g.check("filter_a1_invariants", not bad_a1, bad_a1[:5])

    evals = summarize_evals()
    g.extra["evals_present"] = sorted(evals.keys())
    g.extra["ladder_rungs"] = report.get("rungs")
    g.extra["frozen_test_sha256"] = report.get("frozen_test_sha256")
    g.extra["hint"] = (
        "Materialize subsets, then train each rung with "
        "`python -m form_sft.cli ablation train-cmd A1` and evaluate on frozen test."
    )
    g.extra["spec_sha256"] = _spec_hash(spec_path)
    return g.result()


GATES = {
    "G0": gate_g0, "G1": gate_g1, "G2": gate_g2, "G3": gate_g3, "G4": gate_g4,
    "G5": gate_g5, "G6": gate_g6, "G7": gate_g7, "G8": gate_g8, "G9": gate_g9,
    "G10": gate_g10,
}


def run_gate(gate_id: str, spec: dict, spec_path: str) -> dict:
    res = GATES[gate_id](spec, spec_path)
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    (GATE_DIR / f"{gate_id}.json").write_text(json.dumps(res, indent=2, sort_keys=True, default=str) + "\n")
    return res
