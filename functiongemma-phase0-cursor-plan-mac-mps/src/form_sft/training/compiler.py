"""G6: compile canonical rows using tokenizer chat template; never hand-code control tokens."""
from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from form_sft.contracts.models import CanonicalRow, Operation, SemanticContext
from form_sft.specialization.runtime import SpecializationMode, select_fields
from form_sft.validation.semantic import OP_FOR_TYPE

DEVELOPER_PROMPT = "You are a model that can do function calling with the following functions"
DEFAULT_MODEL_ID = "google/functiongemma-270m-it"
DEFAULT_MAX_SEQ_LENGTH = 1024

# Length policy (explicit; never silent):
# - reject: drop the row and record it in compile_rejects.jsonl; fail if any train/val rejected
# - truncate: keep prefix tokens up to max_seq_length and set metadata.truncated=true (not default)
LENGTH_POLICY_REJECT = "reject"
LENGTH_POLICY_TRUNCATE = "truncate"

# FunctionGemma's installed chat template always opens <start_function_response> after the
# last assistant tool_calls turn (it expects a subsequent tool message). For Phase-0 SFT the
# supervised target is the function call only, matching model generation demos that stop at
# <end_function_call>. Explicit policy: strip a *trailing incomplete* open-response marker
# emitted by the template when no tool result follows. Never invent or rewrite other tokens.
STRIP_TRAILING_OPEN_FUNCTION_RESPONSE = True
_TRAILING_OPEN_FUNCTION_RESPONSE = "<start_function_response>"

_CALL_RE = re.compile(
    r"<start_function_call>\s*call:([A-Za-z0-9_]+)\{(.*?)\}\s*<end_function_call>",
    re.DOTALL,
)
_ARG_RE = re.compile(r"([A-Za-z0-9_]+):(?:<escape>(.*?)<escape>|([^,}]*))", re.DOTALL)


class CompileError(RuntimeError):
    pass


@dataclass(frozen=True)
class CompileStats:
    n_ok: int
    n_rejected: int
    token_lengths: list[int]
    p50: float
    p95: float
    p99: float
    max_len: int
    model_id: str
    max_seq_length: int
    length_policy: str


def compile_example(tokenizer, messages: list[dict], tools: list[dict]) -> str:
    """Render messages+tools with the installed FunctionGemma chat template only."""
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )


def load_tokenizer(model_id: str = DEFAULT_MODEL_ID):
    """Load the official FunctionGemma tokenizer/processor. Never invent special tokens."""
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id)
        # AutoProcessor exposes the same apply_chat_template as the tokenizer.
        if not getattr(processor, "chat_template", None) and hasattr(processor, "tokenizer"):
            if not processor.tokenizer.chat_template:
                raise CompileError(f"{model_id}: chat_template missing on tokenizer/processor")
        return processor
    except Exception:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        if not tok.chat_template:
            raise CompileError(f"{model_id}: chat_template missing on tokenizer")
        return tok


def _json_type(field_type: str) -> str:
    return {
        "string": "string",
        "integer": "integer",
        "number": "number",
        "boolean": "boolean",
        "date": "string",
        "enum": "string",
    }[field_type]


def tools_for_row(ctx: SemanticContext, mode: str) -> list[dict]:
    """Build OpenAI-style tool schemas from the specialized legal field set.

    Tool names are the canonical operation names (set_integer, no_change, ...).
    Field constraints are merged when multiple fields share an operation.
    """
    fields = select_fields(ctx, SpecializationMode(mode))
    by_op: dict[str, dict[str, Any]] = {}
    for f in fields:
        op = OP_FOR_TYPE[f.type]
        entry = by_op.setdefault(
            op,
            {
                "field_ids": [],
                "option_ids": set(),
                "has_value": f.type != "enum",
                "value_type": _json_type(f.type) if f.type != "enum" else None,
                "minimums": [],
                "maximums": [],
                "date_format": f.type == "date",
                "descriptions": [],
            },
        )
        entry["field_ids"].append(f.field_id)
        entry["descriptions"].append(f.label)
        if f.type == "enum":
            entry["option_ids"].update(o.id for o in f.options)
        if f.type in {"integer", "number"}:
            if f.minimum is not None:
                entry["minimums"].append(f.minimum)
            if f.maximum is not None:
                entry["maximums"].append(f.maximum)

    tools: list[dict] = []
    for op, meta in by_op.items():
        props: dict[str, Any] = {
            "field_id": {"type": "string", "enum": sorted(set(meta["field_ids"]))},
        }
        required = ["field_id"]
        if op == "set_enum":
            props["option_id"] = {"type": "string", "enum": sorted(meta["option_ids"])}
            required.append("option_id")
        else:
            value: dict[str, Any] = {"type": meta["value_type"]}
            if meta["date_format"]:
                value["format"] = "date"
            if meta["minimums"]:
                value["minimum"] = min(meta["minimums"])
            if meta["maximums"]:
                value["maximum"] = max(meta["maximums"])
            props["value"] = value
            required.append("value")
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": op,
                    "description": f"Set form field via {op} ({'; '.join(meta['descriptions'][:3])})",
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            }
        )
    tools.append(
        {
            "type": "function",
            "function": {
                "name": "no_change",
                "description": "Do not mutate any form field.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        }
    )
    tools.sort(key=lambda t: t["function"]["name"])
    return tools


def messages_for_row(row: CanonicalRow) -> list[dict]:
    """Conversation ending in the canonical assistant tool call (target)."""
    target = row.target
    fn: dict[str, Any] = {"name": target.name, "arguments": dict(target.arguments)}
    return [
        {"role": "developer", "content": DEVELOPER_PROMPT},
        {"role": "user", "content": row.utterance},
        {
            "role": "assistant",
            "tool_calls": [{"type": "function", "function": fn}],
        },
    ]


def _cast_arg(raw: str) -> Any:
    s = raw.strip()
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        if re.fullmatch(r"-?\d+\.\d+", s):
            return float(s)
    except ValueError:
        pass
    return s


def parse_tool_calls_from_rendered(text: str) -> list[dict[str, Any]]:
    """Inspect rendered assistant tool calls produced by the chat template."""
    out = []
    for name, args in _CALL_RE.findall(text):
        arguments: dict[str, Any] = {}
        for key, esc, bare in _ARG_RE.findall(args):
            arguments[key] = _cast_arg(esc if esc != "" else bare)
        out.append({"name": name, "arguments": arguments})
    return out


def _normalize_target(op: Operation | dict) -> dict:
    if isinstance(op, Operation):
        d = op.model_dump()
    else:
        d = {"name": op["name"], "arguments": dict(op.get("arguments") or {})}
    # Canonical JSON equality for floats/ints
    args = {}
    for k, v in d["arguments"].items():
        if isinstance(v, float) and v.is_integer():
            # keep float if original was float-like number field; compare via json roundtrip later
            args[k] = v
        else:
            args[k] = v
    return {"name": d["name"], "arguments": args}


def targets_equal(a: dict, b: dict) -> bool:
    if a["name"] != b["name"]:
        return False
    if set(a["arguments"]) != set(b["arguments"]):
        return False
    for k in a["arguments"]:
        av, bv = a["arguments"][k], b["arguments"][k]
        if av == bv:
            continue
        # numeric tolerance for 29 vs 29.0
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)) and float(av) == float(bv):
            continue
        return False
    return True


def _encode_ids(tokenizer, text: str) -> list[int]:
    """Return input_ids for text; works for AutoTokenizer and AutoProcessor."""
    tok = getattr(tokenizer, "tokenizer", None)
    enc = tok if tok is not None else tokenizer
    # Prefer encode when available (stable across processors).
    if hasattr(enc, "encode"):
        ids = enc.encode(text, add_special_tokens=False)
        return list(ids)
    out = enc(text, add_special_tokens=False)
    if isinstance(out, dict):
        return list(out["input_ids"])
    if hasattr(out, "input_ids"):
        ids = out.input_ids
        return list(ids[0] if hasattr(ids, "__getitem__") and not isinstance(ids[0], int) else ids)
    raise CompileError(f"cannot encode text with {type(enc).__name__}")


def _count_tokens(tokenizer, text: str) -> int:
    return len(_encode_ids(tokenizer, text))


def _percentile(sorted_vals: list[int], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def compile_row(
    row: CanonicalRow | dict,
    tokenizer,
    *,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    length_policy: str = LENGTH_POLICY_REJECT,
) -> dict:
    """Compile one canonical row to an SFT example with `text` column.

    Returns dict with keys: id, text, target, token_length, messages, tools, metadata.
    Raises CompileError on template/round-trip failure or reject-policy length overflow.
    """
    if isinstance(row, dict):
        row = CanonicalRow.model_validate(row)
    mode = row.metadata.get("specialization_mode")
    if mode not in {"active_field", "reachable_fields"}:
        raise CompileError(f"{row.id}: missing specialization_mode in metadata")

    tools = tools_for_row(row.semantic_context, mode)
    messages = messages_for_row(row)
    try:
        text = compile_example(tokenizer, messages, tools)
    except Exception as e:  # noqa: BLE001
        raise CompileError(f"{row.id}: apply_chat_template failed: {type(e).__name__}: {e}") from e

    if not isinstance(text, str) or not text.strip():
        raise CompileError(f"{row.id}: empty render")

    stripped_open_response = False
    if STRIP_TRAILING_OPEN_FUNCTION_RESPONSE and text.endswith(_TRAILING_OPEN_FUNCTION_RESPONSE):
        text = text[: -len(_TRAILING_OPEN_FUNCTION_RESPONSE)]
        stripped_open_response = True
        if not text.rstrip().endswith("<end_function_call>"):
            raise CompileError(
                f"{row.id}: after stripping open function_response, text does not end at end_function_call"
            )

    # Round-trip: assistant tool call must survive template rendering.
    parsed = parse_tool_calls_from_rendered(text)
    if not parsed:
        raise CompileError(f"{row.id}: no tool call found in rendered text")
    if len(parsed) != 1:
        raise CompileError(f"{row.id}: expected exactly one tool call, got {len(parsed)}")
    expected = _normalize_target(row.target)
    if not targets_equal(parsed[0], expected):
        raise CompileError(
            f"{row.id}: round-trip mismatch expected={expected} parsed={parsed[0]}"
        )

    n_tok = _count_tokens(tokenizer, text)
    truncated = False
    if n_tok > max_seq_length:
        if length_policy == LENGTH_POLICY_REJECT:
            raise CompileError(
                f"{row.id}: token_length {n_tok} > max_seq_length {max_seq_length} (policy=reject)"
            )
        if length_policy != LENGTH_POLICY_TRUNCATE:
            raise CompileError(f"{row.id}: unknown length_policy={length_policy!r}")
        tok = getattr(tokenizer, "tokenizer", tokenizer)
        ids = _encode_ids(tokenizer, text)[:max_seq_length]
        text = tok.decode(ids, skip_special_tokens=False)
        n_tok = len(ids)
        truncated = True
        # Re-check tool call still present after truncate
        parsed2 = parse_tool_calls_from_rendered(text)
        if not parsed2 or not targets_equal(parsed2[0], expected):
            raise CompileError(f"{row.id}: truncation destroyed tool-call target")

    return {
        "id": row.id,
        "text": text,
        "target": expected,
        "token_length": n_tok,
        "truncated": truncated,
        "stripped_trailing_open_function_response": stripped_open_response,
        "messages": messages,
        "tools": tools,
        "metadata": {
            "split": row.metadata.get("split"),
            "primitive": row.metadata.get("primitive"),
            "specialization_mode": mode,
            "source": row.metadata.get("source"),
        },
    }


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def compile_split_file(
    src: Path,
    dst: Path,
    tokenizer,
    *,
    max_seq_length: int,
    length_policy: str,
    reject_path: Path | None = None,
) -> CompileStats:
    lengths: list[int] = []
    ok_rows: list[dict] = []
    rejects: list[dict] = []
    for raw in _iter_jsonl(src):
        try:
            compiled = compile_row(
                raw, tokenizer, max_seq_length=max_seq_length, length_policy=length_policy
            )
            lengths.append(compiled["token_length"])
            ok_rows.append(
                {
                    "id": compiled["id"],
                    "text": compiled["text"],
                    "target": compiled["target"],
                    "token_length": compiled["token_length"],
                    "truncated": compiled["truncated"],
                    "stripped_trailing_open_function_response": compiled[
                        "stripped_trailing_open_function_response"
                    ],
                    "metadata": compiled["metadata"],
                }
            )
        except CompileError as e:
            rejects.append({"id": raw.get("id"), "error": str(e)})
    _write_jsonl(dst, ok_rows)
    if reject_path is not None:
        _write_jsonl(reject_path, rejects)
    lengths_sorted = sorted(lengths)
    return CompileStats(
        n_ok=len(ok_rows),
        n_rejected=len(rejects),
        token_lengths=lengths,
        p50=_percentile(lengths_sorted, 50),
        p95=_percentile(lengths_sorted, 95),
        p99=_percentile(lengths_sorted, 99),
        max_len=max(lengths) if lengths else 0,
        model_id=getattr(tokenizer, "name_or_path", DEFAULT_MODEL_ID),
        max_seq_length=max_seq_length,
        length_policy=length_policy,
    )


def compile_corpus(
    out_dir: Path | str,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    length_policy: str = LENGTH_POLICY_REJECT,
    splits: tuple[str, ...] = ("train", "validation"),
    tokenizer=None,
) -> dict:
    """Compile train/validation (and optional test) into *.sft.jsonl next to the corpus."""
    out_dir = Path(out_dir)
    if tokenizer is None:
        tokenizer = load_tokenizer(model_id)

    # Sanity: template must be present (installed), not hand-authored here.
    tmpl = getattr(tokenizer, "chat_template", None) or getattr(
        getattr(tokenizer, "tokenizer", None), "chat_template", None
    )
    if not tmpl:
        raise CompileError("tokenizer chat_template is empty; refusing to invent tokens")

    report: dict[str, Any] = {
        "model_id": model_id,
        "max_seq_length": max_seq_length,
        "length_policy": length_policy,
        "developer_prompt": DEVELOPER_PROMPT,
        "strip_trailing_open_function_response": STRIP_TRAILING_OPEN_FUNCTION_RESPONSE,
        "strip_policy_note": (
            "Official chat template opens <start_function_response> after the last "
            "assistant tool_calls turn. Phase-0 SFT strips that trailing incomplete "
            "marker so supervised targets end at <end_function_call>."
        ),
        "splits": {},
        "passed": True,
        "problems": [],
    }
    all_lengths: list[int] = []
    for split in splits:
        src = out_dir / f"{split}.jsonl"
        if not src.exists():
            report["passed"] = False
            report["problems"].append(f"missing {src}")
            continue
        dst = out_dir / f"{split}.sft.jsonl"
        rejects = out_dir / f"{split}.sft.rejects.jsonl"
        stats = compile_split_file(
            src,
            dst,
            tokenizer,
            max_seq_length=max_seq_length,
            length_policy=length_policy,
            reject_path=rejects,
        )
        all_lengths.extend(stats.token_lengths)
        report["splits"][split] = {
            "source_rows": stats.n_ok + stats.n_rejected,
            "compiled_rows": stats.n_ok,
            "rejected_rows": stats.n_rejected,
            "output": str(dst),
            "rejects": str(rejects),
            "p50": stats.p50,
            "p95": stats.p95,
            "p99": stats.p99,
            "max": stats.max_len,
        }
        if stats.n_rejected and length_policy == LENGTH_POLICY_REJECT:
            report["passed"] = False
            report["problems"].append(
                f"{split}: {stats.n_rejected} rows rejected under length/round-trip policy"
            )
        # For train/val used in training, require full coverage under reject policy
        if split in {"train", "validation"} and stats.n_ok != stats.n_ok + stats.n_rejected - stats.n_rejected:
            pass
        if split in {"train", "validation"} and stats.n_rejected > 0 and length_policy == LENGTH_POLICY_REJECT:
            report["passed"] = False

    lengths_sorted = sorted(all_lengths)
    report["token_length"] = {
        "n": len(all_lengths),
        "p50": _percentile(lengths_sorted, 50),
        "p95": _percentile(lengths_sorted, 95),
        "p99": _percentile(lengths_sorted, 99),
        "max": max(all_lengths) if all_lengths else 0,
        "mean": statistics.fmean(all_lengths) if all_lengths else 0.0,
    }
    (out_dir / "compile_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not report["passed"]:
        raise CompileError("; ".join(report["problems"]) or "compile failed")
    return report
