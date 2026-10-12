from form_sft.contracts.models import CanonicalRow
from form_sft.evaluation.evaluate import aggregate_json_metrics, PredRow
from form_sft.training.json_examples import fixture_utterances, held_out_fixtures, synthetic_examples, v4_choice_rows, v5_choice_rows
from form_sft.training.json_prompt import (
    calls_from_row,
    context_from_row,
    llama_prompt,
    parse_function_calls,
    score_function_calls,
    tool_line,
    user_block,
)


def test_enum_maps_to_set_choice_and_no_change_is_empty():
    enum_row = CanonicalRow.model_validate(
        {
            "id": "e",
            "metadata": {"specialization_mode": "active_field"},
            "semantic_context": {
                "revision": 1,
                "fingerprint": "fp",
                "locale": "en-IN",
                "active_field_id": "promotion",
                "reachable_field_ids": ["promotion"],
                "fields": [
                    {
                        "field_id": "promotion",
                        "label": "Promotion",
                        "type": "enum",
                        "options": [
                            {"id": "fall", "label": "fall", "aliases": ["autumn"]},
                            {"id": "spring", "label": "spring"},
                        ],
                    }
                ],
            },
            "utterance": "autumn",
            "target": {"name": "set_enum", "arguments": {"field_id": "promotion", "option_id": "fall"}},
        }
    )
    assert calls_from_row(enum_row) == [
        {"name": "set_choice", "arguments": {"field_id": "promotion", "value": "fall"}}
    ]
    ctx = context_from_row(enum_row)
    assert ctx["active"] == "promotion"
    assert "fingerprint" not in ctx
    assert tool_line(ctx["fields"]) == "set_choice(promotion enum fall|spring), clear_field(promotion)"

    empty = CanonicalRow.model_validate(
        {
            **enum_row.model_dump(),
            "target": {"name": "no_change", "arguments": {}},
        }
    )
    assert calls_from_row(empty) == []


def test_fixture_prompt_and_held_out_from_synthetic():
    fixtures = {row["id"]: row for row in held_out_fixtures()}
    age = fixtures["integer_age"]
    user = user_block(age["utterance"], age["context"], age["tools"])
    assert user.startswith("User: Age thirty two\nContext: ")
    assert '"active":"age"' in user
    assert age["tools"] == "set_integer(age: integer), clear_field(age)"
    assert age["target"]["function_calls"][0]["arguments"]["value"] == 32
    assert fixtures["constraint_age"]["target"]["function_calls"] == []
    assert fixtures["choice_alias"]["target"]["function_calls"][0]["arguments"]["value"] == "fall"
    assert fixtures["clear_promotion"]["target"]["function_calls"][0]["name"] == "clear_field"
    email = fixtures["email_spoken"]
    assert email["utterance"] == "kavi at gmail dot com"
    assert email["target"]["function_calls"] == [
        {"name": "set_string", "arguments": {"field_id": "email", "value": "kavi@gmail.com"}}
    ]
    assert email["tools"] == "set_string(email: string), clear_field(email)"
    train = synthetic_examples("train")
    values = [
        call["arguments"]["value"]
        for row in train
        for call in row["target"]["function_calls"]
        if "value" in call["arguments"]
    ]
    assert "kavi.r@gmail.com" in values
    assert 2500 in values
    assert 72.5 in values
    assert 12.5 in values
    banned = fixture_utterances()
    for split in ("train", "validation"):
        for row in synthetic_examples(split):
            assert row["utterance"] not in banned
            assert row["id"] not in fixtures
    _assert_holdout_skill_slice()
    prompt = llama_prompt(user)
    assert "<start_function_declaration>" not in prompt
    assert prompt.endswith("<start_of_turn>model\n")
    assert not prompt.startswith("<bos>")


def _assert_holdout_skill_slice():
    from collections import Counter

    from form_sft.training.json_examples import _SKILL_COUNTS, _banned_holdout_ids, _frozen_test_utterances

    banned_ids = _banned_holdout_ids()
    test_utterances = _frozen_test_utterances()
    by_split = {}
    for split in ("train", "validation"):
        rows = [row for row in synthetic_examples(split) if row["metadata"].get("skill")]
        counts = Counter(row["metadata"]["skill"] for row in rows)
        assert dict(counts) == _SKILL_COUNTS[split]
        by_split[split] = rows
        for row in rows:
            assert row["utterance"] not in test_utterances
            for field in row["context"]["fields"]:
                assert field["id"] not in banned_ids
                for opt in field.get("options") or []:
                    assert opt["id"] not in banned_ids
                    if row["metadata"]["skill"] not in (
                        "enum_surface",
                        "hedged_commit",
                        "hedged_abstain",
                        "restart_correction",
                    ):
                        assert opt["id"] != opt["label"]
    train_utt = {row["utterance"] for row in by_split["train"]}
    val_utt = {row["utterance"] for row in by_split["validation"]}
    assert train_utt.isdisjoint(val_utt)
    option_ids = {"train": set(), "validation": set()}
    for split, rows in by_split.items():
        for row in rows:
            if row["metadata"]["skill"] not in ("enum_option", "correction"):
                continue
            for field in row["context"]["fields"]:
                for opt in field.get("options") or []:
                    assert opt["label"].lower() not in opt["id"].lower()
                    option_ids[split].add(opt["id"])
    assert option_ids["train"].isdisjoint(option_ids["validation"])
    for left in option_ids["train"]:
        for right in option_ids["validation"]:
            assert left not in right and right not in left
    hundred = next(row for row in by_split["train"] if row["metadata"]["skill"] == "hundred_composite")
    assert hundred["target"]["function_calls"][0]["arguments"]["value"] == 1100
    assert "eleven hundred" in hundred["utterance"]
    fraction = next(
        row
        for row in by_split["train"]
        if row["metadata"]["skill"] == "fraction_words"
        and row["target"]["function_calls"][0]["arguments"]["value"] == 40.25
    )
    assert "forty and a quarter" in fraction["utterance"]
    bare = next(
        row
        for row in by_split["train"]
        if row["metadata"]["skill"] == "round_hundred"
        and row["target"]["function_calls"][0]["arguments"]["value"] == 1200
    )
    assert "twelve hundred" in bare["utterance"]
    assert " and " not in bare["utterance"]
    quarter = next(
        row
        for row in by_split["train"]
        if row["metadata"]["skill"] == "small_fraction"
        and row["target"]["function_calls"][0]["arguments"]["value"] == 4.75
    )
    assert "four and three quarters" in quarter["utterance"]
    bare_yes = next(
        row
        for row in by_split["train"]
        if row["metadata"]["skill"] == "bare_boolean"
        and row["target"]["function_calls"][0]["arguments"]["value"] is True
        and row["context"]["active"]
    )
    assert bare_yes["target"]["function_calls"][0]["name"] == "set_boolean"
    cleared = next(row for row in by_split["train"] if row["metadata"]["skill"] == "bare_clear")
    assert cleared["target"]["function_calls"][0]["name"] == "clear_field"
    restart = next(
        row
        for row in by_split["train"]
        if row["metadata"]["skill"] == "restart_correction" and row["metadata"]["primitive"] == "integer"
    )
    assert restart["target"]["function_calls"][0]["arguments"]["value"] < 30
    surface = next(
        row
        for row in by_split["train"]
        if row["metadata"]["skill"] == "enum_surface" and row["utterance"] == "brisk"
    )
    choice = surface["target"]["function_calls"][0]["arguments"]["value"]
    assert choice == "brisk"
    assert any(opt["id"] == opt["label"] == "brisk" for field in surface["context"]["fields"] for opt in field.get("options") or [])
    alias = next(row for row in by_split["train"] if row["metadata"]["skill"] == "enum_surface" and row["utterance"] == "it is veg")
    mapped = next(
        row
        for row in by_split["train"]
        if row["metadata"]["skill"] == "choice_map" and row["utterance"] == "go with indigo"
    )
    assert mapped["target"]["function_calls"][0]["arguments"]["value"] == "indigo_wash"
    assert mapped["target"]["function_calls"][0]["arguments"]["value"] != "indigo"
    abstain = next(row for row in by_split["train"] if row["metadata"]["skill"] == "choice_abstain")
    assert abstain["target"]["function_calls"] == []
    alias_value = alias["target"]["function_calls"][0]["arguments"]["value"]
    alias_field = next(field for field in alias["context"]["fields"] if field["label"] == "plate choice")
    assert alias_value == next(opt["id"] for opt in alias_field["options"] if opt["label"] == "vegetarian")
    assert alias_value != "veg"
    active_int = next(row for row in by_split["train"] if row["metadata"]["skill"] == "active_integer")
    assert active_int["context"]["active"] == active_int["target"]["function_calls"][0]["arguments"]["field_id"]
    assert any(field["label"] == "trip budget" for field in active_int["context"]["fields"])
    remainder = next(row for row in by_split["train"] if row["metadata"]["skill"] == "hundred_remainder")
    assert " and " in remainder["utterance"]
    assert remainder["target"]["function_calls"][0]["arguments"]["value"] % 100 != 0


def test_v4_choice_rows_are_unique_and_do_not_leak_test_ids():
    from form_sft.training.json_examples import _V4_COUNTS, _banned_holdout_ids, _v4_blocked_ids

    blocked = _v4_blocked_ids()
    assert _banned_holdout_ids() <= blocked
    by_split = {}
    for split, (n_commit, n_abstain) in _V4_COUNTS.items():
        rows = v4_choice_rows(split)
        by_split[split] = rows
        assert len(rows) == n_commit + n_abstain
        utterances = [row["utterance"] for row in rows]
        assert len(utterances) == len({item.casefold() for item in utterances})
        commits = [row for row in rows if row["metadata"]["skill"] == "choice_prod"]
        abstains = [row for row in rows if row["metadata"]["skill"] == "choice_prod_abstain"]
        assert len(commits) == n_commit
        assert len(abstains) == n_abstain
        field_ids = []
        for row in rows:
            assert row["target"]["function_calls"] == [] or row["metadata"]["skill"] == "choice_prod"
            field = row["context"]["fields"][0]
            field_ids.append(field["id"])
            assert field["id"] not in blocked
            for opt in field["options"]:
                assert opt["id"] not in blocked
                assert opt["aliases"][0] != opt["id"]
        assert len(field_ids) == len(set(field_ids))
    train_alias = next(row for row in by_split["train"] if row["utterance"].startswith("just "))
    call = train_alias["target"]["function_calls"][0]
    assert call["name"] == "set_choice"
    spoken = train_alias["utterance"].split()[1]
    assert call["arguments"]["value"] != spoken
    assert spoken in call["arguments"]["value"]
    train_abstain = next(row for row in by_split["train"] if row["metadata"]["skill"] == "choice_prod_abstain")
    assert train_abstain["target"]["function_calls"] == []
    train_utt = {row["utterance"].casefold() for row in by_split["train"]}
    val_utt = {row["utterance"].casefold() for row in by_split["validation"]}
    assert train_utt.isdisjoint(val_utt)


def test_v5_option_ids_hide_the_spoken_form():
    from form_sft.training.json_examples import _V5_COUNTS, _v4_blocked_ids

    blocked = _v4_blocked_ids()
    by_split = {}
    for split, (n_commit, n_abstain) in _V5_COUNTS.items():
        rows = v5_choice_rows(split)
        by_split[split] = rows
        assert len(rows) == n_commit + n_abstain
        utterances = [row["utterance"].casefold() for row in rows]
        assert len(utterances) == len(set(utterances))
        commits = [row for row in rows if row["metadata"]["skill"] == "choice_lookup"]
        abstains = [row for row in rows if row["metadata"]["skill"] == "choice_lookup_abstain"]
        assert len(commits) == n_commit
        assert len(abstains) == n_abstain
        assert len(abstains) < len(commits)
        for row in commits:
            call = row["target"]["function_calls"][0]
            assert call["name"] == "set_choice"
            assert call["arguments"]["value"].startswith("opt_")
            fields = row["context"]["fields"]
            assert call["arguments"]["field_id"] == fields[0]["id"]
            for field in fields:
                assert field["id"] not in blocked
                for opt in field.get("options") or []:
                    assert opt["id"] not in blocked
                    folded = opt["id"].casefold()
                    assert opt["label"].casefold() not in folded
                    for alias in opt["aliases"]:
                        assert alias.casefold() not in folded
        for row in abstains:
            assert row["target"]["function_calls"] == []
    train_utt = {row["utterance"].casefold() for row in by_split["train"]}
    val_utt = {row["utterance"].casefold() for row in by_split["validation"]}
    assert train_utt.isdisjoint(val_utt)
    two_fields = next(row for row in by_split["train"] if len(row["context"]["fields"]) == 2)
    assert two_fields["target"]["function_calls"][0]["arguments"]["field_id"] == two_fields["context"]["fields"][0]["id"]


def test_score_empty_and_multi_call():
    gold = {"function_calls": []}
    flags = score_function_calls(gold, {"function_calls": []})
    assert flags["exact_call"] and flags["no_change_tp"] and not flags["false_mutation"]
    bad = {"function_calls": [{"name": "set_integer", "arguments": {"field_id": "age", "value": 1}}]}
    mutated = score_function_calls(gold, bad)
    assert mutated["false_mutation"]
    parsed = parse_function_calls(
        'noise {"function_calls":[{"name":"set_integer","arguments":{"field_id":"age","value":32.0}}]}'
    )
    assert score_function_calls(
        {"function_calls": [{"name": "set_integer", "arguments": {"field_id": "age", "value": 32}}]},
        parsed,
    )["exact_call"]
    rows = [
        PredRow("1", gold, gold, "", True, {"flags": flags}),
        PredRow("2", gold, bad, "", False, {"flags": mutated}),
    ]
    metrics = aggregate_json_metrics(rows)
    assert metrics["false_mutation_rate"] == 0.5
    assert metrics["no_change_recall"] == 0.5
