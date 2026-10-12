from form_sft.training.json_prompt import llama_prompt
from form_sft.training.preflight import verify_sft
from form_sft.training.serializer import deserialize_target, serialize
from form_sft.universal.build import build_records
from form_sft.universal.pipeline import ByteTokenizer
from tests.support_universal import tiny_spec


def test_round_trip_masking_and_gguf_prefix():
    rows, _ = build_records(tiny_spec())
    row = next(item for item in rows if item.coverage.phenomenon == "multi_field")
    example = serialize(row)
    assert example.render_mode == "llama_cpp_compatible"
    assert example.text.startswith(example.prompt)
    assert example.prompt == llama_prompt(example.prompt.split("<start_of_turn>user\n", 1)[1].split("<end_of_turn>", 1)[0]) or example.prompt.startswith("<start_of_turn>developer")
    assert "<bos>" not in example.prompt
    result = verify_sft(example, ByteTokenizer(), 1024, "assistant_only")
    assert result.ok, result.problems
    assert result.trainable_tokens > 0
    assert deserialize_target(example.text[example.text.rfind("<start_of_turn>model\n") + len("<start_of_turn>model\n") : example.text.rfind("<end_of_turn>")])
    actions = deserialize_target(example.text[example.text.rfind("{") :].split("<end_of_turn>")[0] if False else example.text[example.text.rfind("<start_of_turn>model\n"):])
    assert {action.field_id for action in actions} == {action.field_id for action in row.expected_actions}


class PrefixTokenizer:
    """Chat template that differs from the GGUF prompt only by a BOS prefix."""

    pad_token_id = None
    bos = "<bos>"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(text.encode())

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        del tokenize, add_generation_prompt
        user = messages[1]["content"]
        assistant = messages[2]["content"]
        return self.bos + llama_prompt(user) + assistant + "<end_of_turn>\n"


def test_hf_template_diff_is_bos_only():
    rows, _ = build_records(tiny_spec())
    row = rows[0]
    plain = serialize(row)
    hf = serialize(row, tokenizer=PrefixTokenizer())
    assert hf.text == "<bos>" + plain.text
    assert verify_sft(hf, PrefixTokenizer(), 10000, "assistant_only").ok
