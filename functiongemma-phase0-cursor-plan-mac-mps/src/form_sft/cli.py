from __future__ import annotations
import argparse, json, platform, sys, yaml

def load_spec(path):
    with open(path) as f: return yaml.safe_load(f)

def doctor_info() -> dict:
    return {"python": sys.version, "platform": platform.platform(), "status": "ok"}

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    for name in ("plan","build","audit","compile","compile-json","evaluate","evaluate-json","gate"):
        x=sub.add_parser(name); x.add_argument("--spec", default="config/phase0-v1.yaml")
        if name in ("plan","build","audit","compile","compile-json","evaluate"):
            x.add_argument("--out-root", default="generated", help="parent dir of <experiment_id>/ outputs")
        if name == "plan":
            x.add_argument("--write", action="store_true", help="also write the plan artifact (plan.jsonl)")
        if name == "compile":
            x.add_argument("--model-id", default=None, help="override spec/model tokenizer id")
            x.add_argument("--max-seq-length", type=int, default=None)
            x.add_argument("--length-policy", choices=["reject", "truncate"], default="reject")
            x.add_argument("--include-test", action="store_true", help="also compile test.jsonl")
        if name == "compile-json":
            x.add_argument("--model-id", default=None, help="tokenizer id or local checkpoint")
            x.add_argument("--max-seq-length", type=int, default=None)
            x.add_argument("--out-dir", default="generated/json-calls-v1")
        if name in ("evaluate", "evaluate-json"):
            x.add_argument("--split", default="validation", choices=["train","validation","test"])
            x.add_argument("--tag", default="A0", help="artifact tag, e.g. A0 for base baseline")
            x.add_argument("--device", default="auto", choices=["auto","mps","cuda","cpu"])
            x.add_argument("--batch-size", type=int, default=4)
            x.add_argument("--limit", type=int, default=0, help="optional cap for smoke/debug eval")
            x.add_argument("--model-id", default=None)
            if name == "evaluate-json":
                x.add_argument("--file", default="generated/json-calls-v1/fixtures.jsonl")
        if name == "gate":
            x.add_argument("gate_id", choices=["G0","G1","G2","G3","G4","G5","G6","G7","G8","G9","G10"])
    ab = sub.add_parser("ablation", help="Phase-10 cumulative ablation ladder")
    ab_sub = ab.add_subparsers(dest="ablation_cmd", required=True)
    ab_mat = ab_sub.add_parser("materialize", help="Write A1–A8 cumulative SFT subsets")
    ab_mat.add_argument("--spec", default="config/phase0-v1.yaml")
    ab_mat.add_argument("--out-root", default="generated")
    ab_mat.add_argument("--ablation-root", default="artifacts/ablation")
    ab_sum = ab_sub.add_parser("summary", help="Summarize materialized rungs + available evals")
    ab_sum.add_argument("--ablation-root", default="artifacts/ablation")
    ab_train = ab_sub.add_parser("train-cmd", help="Print train command for a rung (A1–A8)")
    ab_train.add_argument("tag", choices=["A1","A2","A3","A4","A5","A6","A7","A8"])
    ab_train.add_argument("--device", default="mps", choices=["auto","mps","cuda","cpu"])
    ab_train.add_argument("--smoke-steps", type=int, default=0)
    ab_train.add_argument("--ablation-root", default="artifacts/ablation")
    sub.add_parser("audit-generalization").add_argument("--out", default="artifacts/audit/universal-v1")
    bu = sub.add_parser("build-universal")
    bu.add_argument("--spec", default="config/universal-pilot-v1.yaml")
    bu.add_argument("--out-root", default="generated")
    eb = sub.add_parser("evaluate-baseline")
    eb.add_argument("--suite-manifest", default="generated/universal-pilot-v1/suites/manifest.json")
    eb.add_argument("--model-id", default="google/functiongemma-270m-it")
    eb.add_argument("--out", default="artifacts/audit/universal-v1")
    a=p.parse_args()
    if a.cmd=="doctor":
        print(json.dumps(doctor_info(), indent=2)); return
    if a.cmd == "audit-generalization":
        from pathlib import Path
        from form_sft.audit.generalization import write_audit
        report = write_audit(Path("."), Path(a.out))
        print(json.dumps({"counts": report["counts"], "migration_errors": report["migration"]["errors"], "label_error_count": report["label_error_count"]}, indent=2))
        return
    if a.cmd == "build-universal":
        from pathlib import Path
        from form_sft.universal.pipeline import export_schemas, finalize_pilot
        spec = load_spec(a.spec)
        out = Path(a.out_root) / spec["spec"]["experiment_id"]
        export_schemas(Path("schemas"))
        summary = finalize_pilot(spec, out, Path("artifacts/audit") / spec["spec"]["experiment_id"])
        print(json.dumps(summary, indent=2, default=str))
        return
    if a.cmd == "evaluate-baseline":
        from pathlib import Path
        manifest_path = Path(a.suite_manifest)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        destination = out / "baseline_manifest.json"
        payload = json.loads(destination.read_text()) if destination.exists() else {}
        payload.update({"model_id": a.model_id, "suite_manifest": str(manifest_path)})
        if manifest_path.exists():
            payload["suite_sha256"] = json.loads(manifest_path.read_text()).get("sha256")
        from form_sft.audit.generalization import weights_ready
        ready = weights_ready(a.model_id)
        payload["weights_present"] = ready
        payload["existing_checkpoint_present"] = ready
        payload["a0_is_unseen_schema_baseline"] = False
        if not ready:
            payload["status"] = "blocked_model_not_local"
            payload["reason"] = "Refusing to download weights or to substitute validation-only A0 metrics."
        else:
            payload["status"] = "weights_present_not_scored_in_this_run"
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": payload["status"], "weights_present": payload["weights_present"], "suite_sha256": payload.get("suite_sha256")}, indent=2))
        return
    if a.cmd == "ablation":
        from pathlib import Path
        from form_sft.ablation.ladder import ABLATION_TITLES, materialize_ladder, summarize_evals
        if a.ablation_cmd == "materialize":
            spec = load_spec(a.spec)
            corpus = Path(a.out_root) / spec["spec"]["experiment_id"]
            report = materialize_ladder(corpus_dir=corpus, out_root=a.ablation_root)
            print(json.dumps(report, indent=2)); return
        if a.ablation_cmd == "summary":
            root = Path(a.ablation_root)
            report_path = root / "ladder_report.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else None
            evals = summarize_evals()
            # also pick up A1-test style tags from full-sft and ablation checkpoints
            print(json.dumps({"ladder_report": report, "evals": evals}, indent=2)); return
        if a.ablation_cmd == "train-cmd":
            root = Path(a.ablation_root) / a.tag
            man = json.loads((root / "manifest.json").read_text())
            out_dir = f"artifacts/checkpoints/ablation-{a.tag}"
            cmd = [
                "python -m form_sft.training.train",
                "--config config/train-full-sft.yaml",
                f"--device {a.device}",
                f"--train-file {man['train']}",
                f"--validation-file {man['validation']}",
                f"--output-dir {out_dir}",
            ]
            if a.smoke_steps:
                cmd.append(f"--smoke-steps {a.smoke_steps}")
            print(f"# {a.tag}: {ABLATION_TITLES[a.tag]} (train_n={man['train_n']})")
            print(" ".join(cmd)); return
        raise SystemExit(f"unknown ablation command {a.ablation_cmd}")
    spec=load_spec(a.spec)
    if a.cmd=="plan":
        from form_sft.dataset.quota_planner import plan_slots, plan_to_jsonl
        slots=plan_slots(spec)
        if a.write:
            from pathlib import Path
            out=Path(a.out_root)/spec["spec"]["experiment_id"]; out.mkdir(parents=True, exist_ok=True)
            (out/"plan.jsonl").write_text(plan_to_jsonl(slots))
        print(len(slots)); return
    if a.cmd=="build":
        from form_sft.dataset.builder import build, BuildError
        try:
            manifest=build(spec, a.spec, a.out_root)
        except BuildError as e:
            raise SystemExit(f"BUILD FAILED (closed): {e}")
        print(json.dumps({"total_rows":manifest["total_rows"],"split_counts":manifest["split_counts"],
                          "test_partition_counts":manifest["test_partition_counts"]}, indent=2)); return
    if a.cmd=="audit":
        from pathlib import Path
        from form_sft.dataset.audit import audit_dir, write_audit
        out=Path(a.out_root)/spec["spec"]["experiment_id"]
        report=audit_dir(spec, out); write_audit(report, out)
        print(json.dumps({k:report[k] for k in ("passed","total_rows","file_counts","duplicate_ids","structural_errors",
              "semantic_errors","consistency_errors","prohibited_leakage","quota_checks_total","quota_checks_failed","problems")}, indent=2))
        if not report["passed"]: raise SystemExit(1)
        return
    if a.cmd=="compile":
        from pathlib import Path
        from form_sft.training.compiler import CompileError, compile_corpus
        out=Path(a.out_root)/spec["spec"]["experiment_id"]
        model_id=a.model_id or spec["spec"].get("model_id") or "google/functiongemma-270m-it"
        # Prefer train config max_seq_length when present
        max_len=a.max_seq_length
        if max_len is None:
            try:
                train_cfg=load_spec("config/train-full-sft.yaml")
                max_len=int(train_cfg.get("max_seq_length", 1024))
            except OSError:
                max_len=1024
        splits=("train","validation","test") if a.include_test else ("train","validation")
        try:
            report=compile_corpus(out, model_id=model_id, max_seq_length=max_len,
                                  length_policy=a.length_policy, splits=splits)
        except Exception as e:
            # Surface gated-repo / auth failures clearly
            raise SystemExit(f"COMPILE FAILED (closed): {e}") from e
        print(json.dumps({"passed":report["passed"],"model_id":report["model_id"],
                          "token_length":report["token_length"],"splits":{
                              k:{"compiled_rows":v["compiled_rows"],"rejected_rows":v["rejected_rows"],
                                 "p99":v["p99"],"max":v["max"]} for k,v in report["splits"].items()}
                          }, indent=2))
        if not report["passed"]: raise SystemExit(1)
        return
    if a.cmd=="compile-json":
        from pathlib import Path
        from form_sft.training.json_corpus import build_json_corpus
        corpus=Path(a.out_root)/spec["spec"]["experiment_id"]
        model_id=a.model_id or spec["spec"].get("model_id") or "google/functiongemma-270m-it"
        max_len=a.max_seq_length
        if max_len is None:
            try:
                max_len=int(load_spec("config/train-json-calls.yaml").get("max_seq_length", 1024))
            except OSError:
                max_len=1024
        try:
            report=build_json_corpus(corpus, a.out_dir, model_id=model_id, max_seq_length=max_len)
        except Exception as e:
            raise SystemExit(f"COMPILE-JSON FAILED: {e}") from e
        print(json.dumps({"passed":report["passed"],"splits":{
            k:{"compiled_rows":v["compiled_rows"],"synthetic_rows":v["synthetic_rows"],"max":v["max"]}
            for k,v in report["splits"].items()},
            "fixtures":report["fixtures"]}, indent=2))
        return
    if a.cmd=="evaluate-json":
        from pathlib import Path
        from form_sft.evaluation.evaluate import evaluate_json
        model_id=a.model_id or spec["spec"].get("model_id") or "google/functiongemma-270m-it"
        try:
            report=evaluate_json(
                a.file,
                out_dir=Path("artifacts/eval")/a.tag,
                model_id=model_id,
                device=a.device,
                batch_size=a.batch_size,
                limit=a.limit or None,
                tag=a.tag,
            )
        except Exception as e:
            raise SystemExit(f"EVALUATE-JSON FAILED: {e}") from e
        o=report["overall"]
        print(json.dumps({
            "tag": report["tag"], "n": o["n"], "device": report["device"],
            "exact_call": o["exact_call"], "op": o["op"], "field": o["field"],
            "value": o["value"], "enum": o["enum"],
            "no_change_precision": o["no_change_precision"],
            "no_change_recall": o["no_change_recall"],
            "false_mutation_rate": o["false_mutation_rate"],
            "predictions_path": report["predictions_path"],
        }, indent=2))
        return
    if a.cmd=="evaluate":
        from pathlib import Path
        from form_sft.evaluation.evaluate import evaluate
        out=Path(a.out_root)/spec["spec"]["experiment_id"]
        split_path=out/f"{a.split}.jsonl"
        eval_out=Path("artifacts/eval")/a.tag
        model_id=a.model_id or spec["spec"].get("model_id") or "google/functiongemma-270m-it"
        try:
            report=evaluate(
                split_path,
                out_dir=eval_out,
                model_id=model_id,
                device=a.device,
                batch_size=a.batch_size,
                limit=a.limit or None,
                tag=a.tag,
            )
        except Exception as e:
            raise SystemExit(f"EVALUATE FAILED: {e}") from e
        o=report["overall"]
        print(json.dumps({
            "tag": report["tag"], "n": o["n"], "device": report["device"],
            "exact_call": o["exact_call"], "op": o["op"], "field": o["field"],
            "value": o["value"], "enum": o["enum"],
            "no_change_precision": o["no_change_precision"],
            "no_change_recall": o["no_change_recall"],
            "false_mutation_rate": o["false_mutation_rate"],
            "predictions_path": report["predictions_path"],
        }, indent=2))
        return
    if a.cmd=="gate":
        from form_sft.gates import run_gate
        res=run_gate(a.gate_id, spec, a.spec)
        print(json.dumps({"gate":res["gate"],"status":res["status"],
                          "failed":[c["name"] for c in res["checks"] if not c["ok"]]}, indent=2))
        if res["status"]!="pass": raise SystemExit(1)
        return
    raise SystemExit(f"{a.cmd} is a gated stub; see IMPLEMENTATION_PLAN.md")
if __name__=="__main__": main()
