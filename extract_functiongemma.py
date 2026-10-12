#!/usr/bin/env python3
"""Extract form fields with function_gemma as one JSON function_calls object.

The prompt is the production instruction plus the utterance, form context, and
tool signatures. The model emits {"function_calls":[...]} and nothing else.

Usage:
    python extract_functiongemma.py \\
        --utterance "Age thirty two" \\
        --context '{"locale":"en","active":"age","fields":[{"id":"age","type":"integer","label":"Age"}]}'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "functiongemma-phase0-cursor-plan-mac-mps" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from form_sft.training.json_prompt import llama_prompt, parse_function_calls, tool_line, user_block

DEFAULT_MODEL = ROOT / "function_gemma_270M.gguf"


def load_model(path: str):
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise SystemExit(
            "llama-cpp-python is not installed.\n"
            "  pip install llama-cpp-python"
        ) from exc
    return Llama(model_path=path, n_ctx=2048, n_gpu_layers=-1, verbose=False)


def extract(llm, utterance: str, context: dict, tools: str) -> dict:
    prompt = llama_prompt(user_block(utterance, context, tools))
    result = llm(
        prompt,
        max_tokens=256,
        temperature=0.0,
        stop=["<end_of_turn>", "<eos>"],
    )
    text = result["choices"][0]["text"]
    parsed = parse_function_calls(text)
    if parsed is None:
        raise SystemExit(f"Model did not return function_calls JSON:\n{text}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utterance", required=True, help="English user utterance")
    parser.add_argument("--context", required=True, help="Form context JSON")
    parser.add_argument(
        "--tools",
        default=None,
        help="Tool signature line. Derived from context fields when omitted.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to the FunctionGemma GGUF (default: function_gemma_270M.gguf)",
    )
    args = parser.parse_args()
    try:
        context = json.loads(args.context)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Context is not JSON: {exc}") from exc
    if not isinstance(context, dict) or not isinstance(context.get("fields"), list):
        raise SystemExit('Context must be an object with a "fields" array.')
    tools = args.tools if args.tools else tool_line(context["fields"])
    if not Path(args.model).is_file():
        raise SystemExit(f"GGUF not found: {args.model}")
    llm = load_model(args.model)
    parsed = extract(llm, args.utterance, context, tools)
    json.dump(parsed, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
