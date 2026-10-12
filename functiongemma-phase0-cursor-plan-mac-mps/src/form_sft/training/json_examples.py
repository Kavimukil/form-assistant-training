"""Held-out fixtures and synthetic paraphrases for the JSON function-call format."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from form_sft.generation.values import MONTHS, digit_words, fraction_words, hundred_composite, int_words
from form_sft.training.json_prompt import clear_call, set_call, tool_line

_ONES = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
    13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
    18: "eighteen", 19: "nineteen",
}
_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}
_MONTHS = [
    (1, "January"), (2, "February"), (3, "March"), (4, "April"), (5, "May"), (6, "June"),
    (7, "July"), (8, "August"), (9, "September"), (10, "October"), (11, "November"), (12, "December"),
]
_ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 14: "fourteenth", 15: "fifteenth"}


def spoken_int(n: int) -> str:
    if n in _ONES:
        return _ONES[n]
    if n in _TENS:
        return _TENS[n]
    if n < 100:
        return f"{_TENS[n // 10 * 10]} {_ONES[n % 10]}"
    raise ValueError(n)


def _enum_options(specs: list[tuple[str, list[str]]]) -> list[dict[str, Any]]:
    options = []
    for oid, aliases in specs:
        opt: dict[str, Any] = {"id": oid, "label": oid}
        if aliases:
            opt["aliases"] = aliases
        options.append(opt)
    return options


def _field(fid: str, kind: str, label: str, **extra: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"id": fid, "type": kind, "label": label}
    item.update(extra)
    return item


_SEASONS = _enum_options([("fall", ["autumn"]), ("spring", []), ("summer", []), ("winter", [])])
_LANGS = _enum_options([("java", []), ("flutter", []), ("react", []), ("python", [])])
_COUNTRIES = _enum_options([("india", []), ("indonesia", [])])


def _example(
    example_id: str,
    utterance: str,
    fields: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    *,
    active: str | None,
    primitive: str,
    split: str,
    source: str,
    skill: str | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "primitive": primitive,
        "split": split,
        "source": source,
        "specialization_mode": mode or ("active_field" if active else "reachable_fields"),
    }
    if skill:
        metadata["skill"] = skill
    return {
        "id": example_id,
        "utterance": utterance,
        "context": {"locale": "en", "active": active, "fields": fields},
        "tools": tool_line(fields),
        "target": {"function_calls": calls},
        "metadata": metadata,
    }


def held_out_fixtures() -> list[dict[str, Any]]:
    age = _field("age", "integer", "Age")
    registers = _field("register_count", "integer", "Registers", aliases=["registers open"])
    registers_plain = _field("register_count", "integer", "Registers")
    weight = _field("weight_kg", "number", "Weight")
    discount = _field("discount_percent", "number", "Discount")
    consent = _field("consent", "boolean", "Consent")
    back = _field("has_back_room", "boolean", "Back room")
    name = _field("name", "string", "Name")
    city = _field("city", "string", "City")
    visit = _field("visit_date", "date", "Visit date")
    dob = _field("date_of_birth", "date", "Date of birth", aliases=["DOB"])
    promotion = _field("promotion", "enum", "Promotion", options=_SEASONS)
    promotion_spring = _field("promotion", "enum", "Promotion", options=_SEASONS, value="spring")
    techs = _field("technologies", "multi", "Technologies", options=_LANGS, aliases=["languages"])
    techs_java = _field("technologies", "multi", "Technologies", options=_LANGS, aliases=["languages"], value=["java"])
    techs_both = _field(
        "technologies", "multi", "Technologies", options=_LANGS, aliases=["languages"], value=["flutter", "java"]
    )
    age_31 = _field("age", "integer", "Age", value=31)
    age_cap = _field("age", "integer", "Age", constraints={"required": False, "min": 18, "max": 100})
    country = _field("country", "enum", "Country", options=_COUNTRIES, optionsTruncated=True)
    rows = [
        _example("integer_age", "Age thirty two", [age], [set_call("set_integer", "age", 32)], active="age", primitive="integer", split="fixture", source="fixture"),
        _example("integer_registers", "There were four registers open", [registers], [set_call("set_integer", "register_count", 4)], active="register_count", primitive="integer", split="fixture", source="fixture"),
        _example("number_weight", "Weight is seventy two point five kilos", [weight], [set_call("set_number", "weight_kg", 72.5)], active="weight_kg", primitive="number", split="fixture", source="fixture"),
        _example("number_discount", "Discount is twelve and a half percent", [discount], [set_call("set_number", "discount_percent", 12.5)], active="discount_percent", primitive="number", split="fixture", source="fixture"),
        _example("boolean_consent", "Yes I consent", [consent], [set_call("set_boolean", "consent", True)], active="consent", primitive="boolean", split="fixture", source="fixture"),
        _example("boolean_backroom", "No, there is no back room", [back], [set_call("set_boolean", "has_back_room", False)], active="has_back_room", primitive="boolean", split="fixture", source="fixture"),
        _example("string_name", "My name is Mukil", [name], [set_call("set_string", "name", "Mukil")], active="name", primitive="string", split="fixture", source="fixture"),
        _example("string_city", "City is Bangalore", [city], [set_call("set_string", "city", "Bangalore")], active="city", primitive="string", split="fixture", source="fixture"),
        _example("date_visit", "Visit date was October fifth 2026", [visit], [set_call("set_date", "visit_date", "2026-10-05")], active="visit_date", primitive="date", split="fixture", source="fixture"),
        _example("date_birth", "DOB is fourteenth August 1992", [dob], [set_call("set_date", "date_of_birth", "1992-08-14")], active="date_of_birth", primitive="date", split="fixture", source="fixture"),
        _example("choice_alias", "The promotion was autumn", [promotion], [set_call("set_choice", "promotion", "fall")], active="promotion", primitive="enum", split="fixture", source="fixture"),
        _example("multi_selection", "I use Java and Flutter", [techs], [set_call("set_multi_choice", "technologies", ["flutter", "java"])], active="technologies", primitive="multi", split="fixture", source="fixture"),
        _example("multi_add", "Also add Flutter and React", [techs_java], [set_call("set_multi_choice", "technologies", ["flutter", "java", "react"])], active="technologies", primitive="multi", split="fixture", source="fixture"),
        _example("clear_promotion", "Clear the promotion", [promotion_spring], [clear_call("promotion")], active="promotion", primitive="clear", split="fixture", source="fixture"),
        _example("correction_age", "Actually I'm thirty two", [age_31], [set_call("set_integer", "age", 32)], active="age", primitive="integer", split="fixture", source="fixture"),
        _example(
            "compound_store",
            "We have a back room, four registers were open and the promotion was autumn",
            [back, registers, promotion],
            [
                set_call("set_boolean", "has_back_room", True),
                set_call("set_integer", "register_count", 4),
                set_call("set_choice", "promotion", "fall"),
            ],
            active="has_back_room",
            primitive="compound",
            split="fixture",
            source="fixture",
        ),
        _example("negation_consent", "I do not consent", [consent], [set_call("set_boolean", "consent", False)], active="consent", primitive="boolean", split="fixture", source="fixture"),
        _example("negation_choice", "It wasn't the spring promotion, it was autumn", [promotion], [set_call("set_choice", "promotion", "fall")], active="promotion", primitive="enum", split="fixture", source="fixture"),
        _example("reference_age", "Change that to thirty two", [age_31], [set_call("set_integer", "age", 32)], active="age", primitive="integer", split="fixture", source="fixture"),
        _example("reference_multi", "Remove Flutter and keep Java", [techs_both], [set_call("set_multi_choice", "technologies", ["java"])], active="technologies", primitive="multi", split="fixture", source="fixture"),
        _example("ambiguous_number", "Maybe four or five", [registers_plain], [], active="register_count", primitive="no_change", split="fixture", source="fixture"),
        _example("ambiguous_option", "The promotion was the other one", [promotion], [], active="promotion", primitive="no_change", split="fixture", source="fixture"),
        _example("constraint_age", "Age is one hundred and fifty", [age_cap], [], active="age", primitive="no_change", split="fixture", source="fixture"),
        _example(
            "noise_compound",
            "age thirty to consent yes",
            [age, consent],
            [set_call("set_integer", "age", 30), set_call("set_boolean", "consent", True)],
            active="age",
            primitive="compound",
            split="fixture",
            source="fixture",
        ),
        _example("noise_registers", "four registers where open", [registers_plain], [set_call("set_integer", "register_count", 4)], active="register_count", primitive="integer", split="fixture", source="fixture"),
        _example("noise_choice", "promotion was auto man", [promotion], [set_call("set_choice", "promotion", "fall")], active="promotion", primitive="enum", split="fixture", source="fixture"),
        _example("out_of_context", "My country is India", [age, consent], [], active="age", primitive="no_change", split="fixture", source="fixture"),
        _example("option_truncated", "Japan", [country], [], active="country", primitive="no_change", split="fixture", source="fixture"),
        _example(
            "email_spoken",
            "kavi at gmail dot com",
            [_field("email", "string", "Email")],
            [set_call("set_string", "email", "kavi@gmail.com")],
            active="email",
            primitive="string",
            split="fixture",
            source="fixture",
        ),
    ]
    shared_calls = [set_call("set_integer", "age", 32), set_call("set_boolean", "consent", True)]
    for suffix in ("001", "002", "003"):
        rows.append(
            _example(
                f"compound_age_consent_{suffix}",
                "Age thirty two, consent yes",
                [age, consent],
                shared_calls,
                active="age",
                primitive="compound",
                split="fixture",
                source="fixture",
            )
        )
    return rows


def fixture_utterances() -> set[str]:
    return {row["utterance"] for row in held_out_fixtures()}


def synthetic_examples(split: str) -> list[dict[str, Any]]:
    banned = fixture_utterances()
    rows: list[dict[str, Any]] = []
    prefix = "json-syn-tr" if split == "train" else "json-syn-va"
    numbers = [21, 27, 33, 36, 41, 44, 48, 53, 58, 64, 69, 77, 81, 86, 93] if split == "train" else [22, 35, 46, 57, 68]
    counts = [2, 3, 6, 7, 8, 9, 11] if split == "train" else [12, 15]
    names = ["Priya", "Arun", "Leela", "Dev"] if split == "train" else ["Nora", "Sam"]
    cities = ["Chennai", "Mysore", "Pune", "Kochi"] if split == "train" else ["Surat", "Jaipur"]
    weights = [(68.5, "sixty eight point five"), (81.25, "eighty one point two five")] if split == "train" else [(54.5, "fifty four point five")]
    percents = [(7.5, "seven and a half"), (15.5, "fifteen and a half")] if split == "train" else [(9.5, "nine and a half")]

    def add(
        kind: str,
        utterance: str,
        fields: list[dict],
        calls: list[dict],
        active: str | None,
        skill: str | None = None,
        mode: str | None = None,
    ) -> None:
        if utterance in banned:
            raise RuntimeError(f"synthetic utterance collides with a held-out fixture: {utterance}")
        rows.append(
            _example(
                f"{prefix}-{len(rows):04d}",
                utterance,
                fields,
                calls,
                active=active,
                primitive=kind,
                split=split,
                source="synthetic_json",
                skill=skill,
                mode=mode,
            )
        )

    age = _field("age", "integer", "Age")
    for n in numbers:
        spoken = spoken_int(n)
        current = _field("age", "integer", "Age", value=n - 1)
        add("integer", f"The age is {spoken}", [age], [set_call("set_integer", "age", n)], "age")
        add("integer", f"I am {spoken} years old", [age], [set_call("set_integer", "age", n)], "age")
        add("integer", f"Actually I am {spoken}", [current], [set_call("set_integer", "age", n)], "age")
        add("integer", f"Change that to {spoken}", [current], [set_call("set_integer", "age", n)], "age")

    registers = _field("counter_count", "integer", "Counters")
    for n in counts:
        spoken = spoken_int(n)
        add("integer", f"There were {spoken} counters open", [registers], [set_call("set_integer", "counter_count", n)], "counter_count")
        add("integer", f"{spoken} counters where open", [registers], [set_call("set_integer", "counter_count", n)], "counter_count")

    for value, spoken in weights:
        add("number", f"Weight is {spoken} kilos", [_field("weight_kg", "number", "Weight")], [set_call("set_number", "weight_kg", value)], "weight_kg")
    for value, spoken in percents:
        add("number", f"Discount is {spoken} percent", [_field("discount_percent", "number", "Discount")], [set_call("set_number", "discount_percent", value)], "discount_percent")

    consent = _field("consent", "boolean", "Consent")
    back = _field("has_back_room", "boolean", "Back room")
    add("boolean", "Yes I agree", [consent], [set_call("set_boolean", "consent", True)], "consent")
    add("boolean", "I consent", [consent], [set_call("set_boolean", "consent", True)], "consent")
    add("boolean", "I do not agree", [consent], [set_call("set_boolean", "consent", False)], "consent")
    add("boolean", "No I do not consent", [consent], [set_call("set_boolean", "consent", False)], "consent")
    add("boolean", "Yes there is a back room", [back], [set_call("set_boolean", "has_back_room", True)], "has_back_room")
    add("boolean", "No back room here", [back], [set_call("set_boolean", "has_back_room", False)], "has_back_room")

    for person in names:
        add("string", f"My name is {person}", [_field("name", "string", "Name")], [set_call("set_string", "name", person)], "name")
    for city in cities:
        add("string", f"City is {city}", [_field("city", "string", "City")], [set_call("set_string", "city", city)], "city")

    months = _MONTHS[:8] if split == "train" else _MONTHS[:3]
    for month_i, month in months:
        day = 15 if month_i == 8 else 5
        year = 2024 + (month_i % 3)
        ordinal = _ORDINALS[day]
        add(
            "date",
            f"Visit date was {month} {ordinal} {year}",
            [_field("visit_date", "date", "Visit date")],
            [set_call("set_date", "visit_date", f"{year:04d}-{month_i:02d}-{day:02d}")],
            "visit_date",
        )
        add(
            "date",
            f"DOB is {ordinal} {month} {1990 + month_i}",
            [_field("date_of_birth", "date", "Date of birth", aliases=["DOB"])],
            [set_call("set_date", "date_of_birth", f"{1990 + month_i:04d}-{month_i:02d}-{day:02d}")],
            "date_of_birth",
        )

    promotion = _field("promotion", "enum", "Promotion", options=_SEASONS)
    for season in ("spring", "summer", "winter", "fall"):
        add("enum", f"The promotion was {season}", [promotion], [set_call("set_choice", "promotion", season)], "promotion")
    add("enum", "Pick autumn for the promotion", [promotion], [set_call("set_choice", "promotion", "fall")], "promotion")
    add("enum", "The season should be autumn", [promotion], [set_call("set_choice", "promotion", "fall")], "promotion")
    add("enum", "It was not spring, use autumn", [promotion], [set_call("set_choice", "promotion", "fall")], "promotion")
    add("enum", "the deal was auto man", [promotion], [set_call("set_choice", "promotion", "fall")], "promotion")
    add("clear", "Wipe the promotion", [_field("promotion", "enum", "Promotion", options=_SEASONS, value="spring")], [clear_call("promotion")], "promotion")
    add("clear", "Clear the city", [_field("city", "string", "City", value="Pune")], [clear_call("city")], "city")

    langs = _field("technologies", "multi", "Technologies", options=_LANGS, aliases=["languages"])
    combos = [("java", "python"), ("react", "flutter"), ("python", "java"), ("flutter", "react")]
    if split != "train":
        combos = [("python", "react")]
    for left, right in combos:
        add(
            "multi",
            f"I use {left.title()} and {right.title()}",
            [langs],
            [set_call("set_multi_choice", "technologies", sorted({left, right}))],
            "technologies",
        )
    add(
        "multi",
        "Also add Python and Flutter",
        [_field("technologies", "multi", "Technologies", options=_LANGS, value=["java"])],
        [set_call("set_multi_choice", "technologies", ["flutter", "java", "python"])],
        "technologies",
    )
    add(
        "multi",
        "Remove Java and keep Flutter",
        [_field("technologies", "multi", "Technologies", options=_LANGS, value=["flutter", "java"])],
        [set_call("set_multi_choice", "technologies", ["flutter"])],
        "technologies",
    )
    add(
        "compound",
        "Age forty one, consent yes",
        [age, consent],
        [set_call("set_integer", "age", 41), set_call("set_boolean", "consent", True)],
        "age",
    )
    add(
        "compound",
        "age forty to consent no",
        [age, consent],
        [set_call("set_integer", "age", 40), set_call("set_boolean", "consent", False)],
        "age",
    )
    add(
        "compound",
        "We have a back room, six counters were open and the promotion was summer",
        [back, registers, promotion],
        [
            set_call("set_boolean", "has_back_room", True),
            set_call("set_integer", "counter_count", 6),
            set_call("set_choice", "promotion", "summer"),
        ],
        "has_back_room",
    )
    add("no_change", "Age is two hundred", [_field("age", "integer", "Age", constraints={"required": False, "min": 18, "max": 100})], [], "age")
    add("no_change", "Maybe six or seven", [registers], [], "counter_count")
    add("no_change", "It was the other season", [promotion], [], "promotion")
    add("no_change", "My country is France", [age, consent], [], "age")
    add("no_change", "Brazil", [_field("country", "enum", "Country", options=_COUNTRIES, optionsTruncated=True)], [], "country")

    extra_ages = numbers if split == "train" else numbers[:2]
    for n in extra_ages:
        spoken = spoken_int(n)
        add(
            "clear",
            f"Clear the age, it is not {spoken}",
            [_field("age", "integer", "Age", value=n)],
            [clear_call("age")],
            "age",
        )
        add(
            "compound",
            f"Age {spoken}, consent no",
            [age, consent],
            [set_call("set_integer", "age", n), set_call("set_boolean", "consent", False)],
            "age",
        )
        add(
            "multi",
            f"Also add Java and Python to the {spoken} stack",
            [_field("technologies", "multi", "Technologies", options=_LANGS, value=["react"])],
            [set_call("set_multi_choice", "technologies", ["java", "python", "react"])],
            "technologies",
        )
        add(
            "no_change",
            f"Maybe {spoken} or {spoken_int(n + 1 if n < 99 else n)}",
            [registers],
            [],
            "counter_count",
        )
    _add_spoken_emails(add, split)
    _add_spoken_numbers(add, split)
    _add_holdout_skills(add, split)
    return rows


def _speak_email(local: tuple[str, ...], host: tuple[str, ...]) -> str:
    return f"{' dot '.join(local)} at {' dot '.join(host)}"


def _join_email(local: tuple[str, ...], host: tuple[str, ...]) -> str:
    return f"{'.'.join(local)}@{'.'.join(host)}".lower()


def _add_spoken_emails(add, split: str) -> None:
    email = _field("email", "string", "Email")
    if split == "train":
        locals_simple = (
            "meera", "arjun", "nisha", "rohan", "leela", "dev", "ananya", "vikram",
            "sana", "rahul", "pooja", "aditya", "deepak", "tanvi", "isha", "nikhil",
        )
        dotted = (("kavi", "r"), ("meera", "k"), ("arjun", "s"), ("nisha", "p"), ("leela", "m"), ("dev", "n"))
        hosts = (("gmail", "com"), ("yahoo", "com"), ("outlook", "com"), ("rediffmail", "com"))
        multi = (("rediffmail", "co", "in"), ("yahoo", "co", "uk"), ("outlook", "co", "in"))
        frames = ("{spoken}", "Email is {spoken}", "My email is {spoken}")
    else:
        locals_simple = ("nora", "sam", "rita", "omar")
        dotted = (("nora", "j"), ("sam", "p"))
        hosts = (("gmail", "com"), ("yahoo", "com"), ("outlook", "com"))
        multi = (("rediffmail", "co", "in"),)
        frames = ("{spoken}", "Email is {spoken}")
    pairs: list[tuple[tuple[str, ...], tuple[str, ...]]] = [( (name,), host) for name in locals_simple for host in hosts]
    pairs.extend((local, host) for local in dotted for host in hosts)
    pairs.extend(( (name,), host) for name in locals_simple[:8] for host in multi)
    held_out = ("kavi",), ("gmail", "com")
    for local, host in pairs:
        if (local, host) == held_out:
            continue
        spoken = _speak_email(local, host)
        address = _join_email(local, host)
        for frame in frames:
            add("string", frame.format(spoken=spoken), [email], [set_call("set_string", "email", address)], "email")


def _spoken_decimal(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    whole, frac = text.split(".")
    return f"{int_words(int(whole))} point {digit_words(frac)}"


def _add_spoken_numbers(add, split: str) -> None:
    age = _field("age", "integer", "Age")
    counters = _field("counter_count", "integer", "Counters")
    weight = _field("weight_kg", "number", "Weight")
    discount = _field("discount_percent", "number", "Discount")
    if split == "train":
        small = list(range(20, 100, 2))
        hundreds = [101, 120, 150, 240, 305, 418, 512, 640, 777, 890, 999]
        thousands = [1000, 1200, 2500, 3040, 4500, 10000, 12500, 20000]
        decimals = [18.5, 25.25, 41.5, 56.75, 63.5, 70.25, 72.5, 84.5, 91.75]
        halves = [7.5, 9.5, 12.5, 15.5, 20.5, 33.5]
    else:
        small = [23, 34, 47, 59, 88]
        hundreds = [110, 260, 480]
        thousands = [1800, 5600]
        decimals = [19.5, 44.25, 67.5]
        halves = [8.5, 14.5]
    for n in [*small, *hundreds, *thousands]:
        words = int_words(n)
        hyphen = int_words(n, True)
        add("integer", f"Age in years is {words}", [age], [set_call("set_integer", "age", n)], "age")
        if hyphen != words:
            add("integer", f"Age in years is {hyphen}", [age], [set_call("set_integer", "age", n)], "age")
        if n < 1000:
            add("integer", f"There were {words} counters open", [counters], [set_call("set_integer", "counter_count", n)], "counter_count")
    for value in decimals:
        words = _spoken_decimal(value)
        add("number", f"Weight is {words} kilograms", [weight], [set_call("set_number", "weight_kg", value)], "weight_kg")
    for value in halves:
        words = fraction_words(value)
        if words is None:
            continue
        add(
            "number",
            f"The discount is {words} percent",
            [discount],
            [set_call("set_number", "discount_percent", value)],
            "discount_percent",
        )


_SKILL_COUNTS = {
    "train": {
        "hundred_composite": 2000,
        "fraction_words": 1200,
        "field_copy": 2000,
        "enum_option": 3600,
        "correction": 500,
        "no_change_guard": 1200,
        "commit_near_miss": 1200,
        "round_hundred": 712,
        "small_fraction": 480,
        "glued_fraction": 60,
        "bare_boolean": 300,
        "bare_clear": 300,
        "hedged_commit": 200,
        "hedged_abstain": 200,
        "restart_correction": 320,
        "restart_abstain": 320,
        "enum_surface": 1200,
        "choice_map": 960,
        "choice_abstain": 480,
        "active_integer": 160,
        "hundred_remainder": 180,
        "large_integer": 120,
    },
    "validation": {
        "hundred_composite": 200,
        "fraction_words": 150,
        "field_copy": 200,
        "enum_option": 450,
        "correction": 60,
        "no_change_guard": 160,
        "commit_near_miss": 160,
        "round_hundred": 88,
        "small_fraction": 48,
        "glued_fraction": 15,
        "bare_boolean": 30,
        "bare_clear": 30,
        "hedged_commit": 20,
        "hedged_abstain": 20,
        "restart_correction": 40,
        "restart_abstain": 40,
        "enum_surface": 120,
        "choice_map": 128,
        "choice_abstain": 64,
        "active_integer": 16,
        "hundred_remainder": 20,
        "large_integer": 12,
    },
}

_DATE_LABELS = {
    "train": ("packing date", "audit date", "intake date", "review date", "seal date", "blend date"),
    "validation": ("invoice date", "renewal date", "handoff date", "census date", "filing date", "issue date"),
}
_BOOL_LABELS = {
    "train": ("rush order", "gift wrap", "fragile item", "signature required", "open box", "call ahead"),
    "validation": ("priority desk", "window seat", "paper receipt", "locker access", "rear door", "quiet car"),
}
_STR_LABELS = {
    "train": ("dock name", "bay code", "crew lead", "shift note", "lane tag", "yard note"),
    "validation": ("gate name", "aisle code", "shift lead", "bin note", "row tag", "cage note"),
}
_INT_LABELS = {
    "train": ("pallet count", "bin tally", "crate total", "lot size", "batch units", "reel count"),
    "validation": ("roll count", "tray tally", "sack total", "bundle size", "case units", "spool count"),
}
_NUM_LABELS = {
    "train": ("moisture reading", "batch yield", "fill level", "sample weight"),
    "validation": ("salinity reading", "yield ratio", "tank level", "tare weight"),
}
_ENUM_LABELS = {
    "train": (
        ("visit cadence", ("weekly", "daily", "monthly")),
        ("furnishing", ("unfurnished", "furnished", "partial")),
        ("priority band", ("low", "medium", "high")),
        ("packing style", ("boxed", "crated", "loose")),
    ),
    "validation": (
        ("delivery rhythm", ("weekly", "daily", "monthly")),
        ("fit out", ("unfurnished", "furnished", "partial")),
        ("urgency band", ("low", "medium", "high")),
        ("wrap style", ("boxed", "crated", "loose")),
    ),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def _frozen_test_utterances() -> frozenset[str]:
    path = _repo_root() / "generated" / "phase0-v1" / "test.jsonl"
    utterances: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                utterances.add(json.loads(line)["utterance"])
    return frozenset(utterances)


@lru_cache(maxsize=1)
def _banned_holdout_ids() -> frozenset[str]:
    from form_sft.dataset.families import HOLDOUT_FIELD_IDS_BY_PARTITION
    from form_sft.generation.catalog import (
        ENUM_HOLDOUT_DEFS,
        HOLDOUT_DOMAIN_DEFS,
        all_field_ids,
        all_option_ids,
    )

    banned: set[str] = set()
    for ids in HOLDOUT_FIELD_IDS_BY_PARTITION.values():
        banned.update(ids)
    banned.update(all_field_ids(ENUM_HOLDOUT_DEFS))
    banned.update(all_field_ids(HOLDOUT_DOMAIN_DEFS))
    banned.update(all_option_ids(ENUM_HOLDOUT_DEFS))
    banned.update(all_option_ids(HOLDOUT_DOMAIN_DEFS))
    return frozenset(banned)


def _iter_ids(fields: list[dict[str, Any]]) -> list[str]:
    found: list[str] = []
    for field in fields:
        found.append(field["id"])
        for opt in field.get("options") or []:
            found.append(str(opt["id"]))
    return found


def _speak_date(iso: str) -> str:
    year, month, day = (int(part) for part in iso.split("-"))
    return f"{day} {MONTHS[month - 1][:3]} {year}"


def _rotate(items: list[dict[str, Any]], shift: int) -> list[dict[str, Any]]:
    if not items:
        return items
    cut = shift % len(items)
    return items[cut:] + items[:cut]


def _syn_field(fid: str, kind: str, label: str) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if kind == "integer":
        extra["constraints"] = {"min": 0, "max": 100000}
    elif kind == "number":
        extra["constraints"] = {"min": 0, "max": 1000000}
    elif kind == "date":
        extra["constraints"] = {"min": "2020-01-01", "max": "2035-12-31"}
    return _field(fid, kind, label, **extra)


def _enum_field(fid: str, label: str, pool: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    options = [{"id": oid, "label": lab} for lab, oid in pool]
    return _field(fid, "enum", label, options=options)


def _opaque_id(prefix: str, key: str) -> str:
    """Option/field id with no shared stem across keys. Never contains the spoken label."""
    banned = _banned_holdout_ids()
    for extra in ("", "|a", "|b"):
        digest = hashlib.sha1(f"json-calls-v3|{prefix}|{key}{extra}".encode()).hexdigest()[:8]
        ident = f"{prefix}_{digest}"
        if ident not in banned:
            return ident
    raise RuntimeError(f"could not mint an id for {prefix} {key}")


def _choice_pool(split: str, index: int, labels: tuple[str, ...], salt: str) -> tuple[tuple[str, str], ...]:
    pool = []
    for slot, label in enumerate(labels):
        oid = _opaque_id("opt", f"{split}|{salt}|{index}|{slot}")
        if label.lower() in oid.lower():
            raise RuntimeError(f"option id {oid} contains label {label}")
        pool.append((label, oid))
    return tuple(pool)


def _add_holdout_skills(add, split: str) -> None:
    """Train the held-out numeric, field-copy, enum-id, and correction skills.

    Rows stay out of the frozen test split. Field and option ids are synthetic
    and must not collide with holdout registries or frozen test utterances.
    """
    if split not in _SKILL_COUNTS:
        raise ValueError(split)
    counts = _SKILL_COUNTS[split]
    banned_utt = _frozen_test_utterances()
    banned_ids = _banned_holdout_ids()

    def emit(
        skill: str,
        kind: str,
        utterance: str,
        fields: list[dict[str, Any]],
        calls: list[dict[str, Any]],
        active: str | None,
    ) -> None:
        if utterance in banned_utt:
            raise RuntimeError(f"synthetic utterance matches frozen test: {utterance}")
        for ident in _iter_ids(fields):
            if ident in banned_ids:
                raise RuntimeError(f"synthetic id {ident} is a holdout id")
        mode = "active_field" if active else "reachable_fields"
        add(kind, utterance, fields, calls, active, skill=skill, mode=mode)

    _add_hundreds(emit, split, counts["hundred_composite"])
    _add_fractions(emit, split, counts["fraction_words"])
    _add_field_copies(emit, split, counts["field_copy"])
    _add_enum_copies(emit, split, counts["enum_option"])
    _add_corrections(emit, split, counts["correction"])
    _add_no_change_guards(emit, split, counts["no_change_guard"])
    _add_round_hundreds(emit, split)
    _add_small_fractions(emit, split)
    _add_glued_fractions(emit, split)
    _add_bare_booleans(emit, split)
    _add_hedged_commits(emit, split)
    _add_restart_corrections(emit, split)
    _add_enum_surface(emit, split)
    _add_choice_maps(emit, split)
    _add_active_integers(emit, split)
    _add_hundred_remainders(emit, split)
    _add_large_integers(emit, split)


def _add_hundreds(emit, split: str, count: int) -> None:
    labels = _INT_LABELS[split]
    start = 1100 if split == "train" else 7000
    frames = (
        "It should be {spoken}",
        "For the {label}, {spoken} is correct",
        "Go ahead with {spoken} for {label}",
        "Please note {label} as {spoken}",
    )
    for i in range(count):
        value = start + i
        spoken = hundred_composite(value)
        label = labels[i % len(labels)]
        n_extra = 0 if i % 5 == 0 else min(2 + (i % 4), len(labels) - 1)
        fields = [_syn_field(f"syn_hun_{split[:2]}_{i:04d}_0", "integer", label)]
        for k in range(n_extra):
            distractor = labels[(i + k + 1) % len(labels)]
            fields.append(_syn_field(f"syn_hun_{split[:2]}_{i:04d}_{k + 1}", "integer", distractor))
        fields = _rotate(fields, i)
        target = next(field for field in fields if field["label"] == label and field["id"].endswith("_0"))
        use_active = i % 2 == 0
        frame = frames[0] if use_active else frames[1 + (i % 3)]
        utterance = frame.format(spoken=spoken, label=label)
        emit(
            "hundred_composite",
            "integer",
            utterance,
            fields,
            [set_call("set_integer", target["id"], value)],
            target["id"] if use_active else None,
        )


def _fraction_values(split: str, count: int) -> list[float]:
    fracs = (25, 50, 75)
    if split == "train":
        ips = list(range(40, 440)) + list(range(1500, 2300))
    else:
        ips = list(range(900, 950)) + list(range(5000, 5100))
    if len(ips) != count:
        raise RuntimeError(f"{split} fraction ip count {len(ips)} != {count}")
    return [round(ip + fracs[i % 3] / 100, 2) for i, ip in enumerate(ips)]


def _add_fractions(emit, split: str, count: int) -> None:
    labels = _NUM_LABELS[split]
    frames = (
        "The reading is {spoken}",
        "It is {spoken} for {label}",
        "Go ahead with {spoken}",
        "Please note {label} is {spoken}",
    )
    for i, value in enumerate(_fraction_values(split, count)):
        spoken = fraction_words(value)
        if spoken is None:
            raise RuntimeError(f"value {value} is not expressible as fraction words")
        label = labels[i % len(labels)]
        n_extra = 0 if i % 5 == 0 else min(2 + (i % 3), len(labels) - 1)
        fields = [_syn_field(f"syn_frc_{split[:2]}_{i:04d}_0", "number", label)]
        for k in range(n_extra):
            distractor = labels[(i + k + 1) % len(labels)]
            fields.append(_syn_field(f"syn_frc_{split[:2]}_{i:04d}_{k + 1}", "number", distractor))
        fields = _rotate(fields, i)
        target = next(field for field in fields if field["id"].endswith("_0"))
        use_active = i % 2 == 0
        frame = frames[0] if use_active else frames[1 + (i % 3)]
        emit(
            "fraction_words",
            "number",
            frame.format(spoken=spoken, label=label),
            fields,
            [set_call("set_number", target["id"], value)],
            target["id"] if use_active else None,
        )


def _field_copy_value(kind: str, index: int, split: str) -> tuple[str, Any]:
    if kind == "date":
        base = dt.date(2024, 3, 1) if split == "train" else dt.date(2028, 6, 1)
        iso = (base + dt.timedelta(days=index)).isoformat()
        return _speak_date(iso), iso
    if kind == "boolean":
        value = index % 2 == 0
        return ("true" if value else "false"), value
    if kind == "string":
        text = f"{'crew' if split == 'train' else 'unit'} {index:04d}"
        return text, text
    value = (10000 if split == "train" else 20000) + index
    return str(value), value


def _add_field_copies(emit, split: str, count: int) -> None:
    kinds = ("date", "boolean", "string", "integer")
    labels_for = {
        "date": _DATE_LABELS[split],
        "boolean": _BOOL_LABELS[split],
        "string": _STR_LABELS[split],
        "integer": _INT_LABELS[split],
    }
    setters = {"date": "set_date", "boolean": "set_boolean", "string": "set_string", "integer": "set_integer"}
    frames = (
        "Set {label} to {spoken}",
        "For {label}, go ahead with {spoken}",
        "Please note {label} is {spoken}",
        "{label} should be {spoken}",
    )
    for i in range(count):
        kind = kinds[i % 4]
        labels = labels_for[kind]
        spoken, value = _field_copy_value(kind, i, split)
        n_fields = min(len(labels), 3 + (i % 4))
        chosen = [labels[(i + k) % len(labels)] for k in range(n_fields)]
        fields = [
            _syn_field(f"syn_cpy_{kind[:3]}_{split[:2]}_{i:04d}_{k}", kind, lab)
            for k, lab in enumerate(chosen)
        ]
        fields = _rotate(fields, i)
        target = next(field for field in fields if field["label"] == chosen[0])
        name_field = kind == "boolean" or i % 2 == 1
        utterance = frames[i % 4].format(label=chosen[0], spoken=spoken) if name_field else f"Go ahead with {spoken}"
        emit(
            "field_copy",
            kind if kind != "date" else "date",
            utterance,
            fields,
            [set_call(setters[kind], target["id"], value)],
            target["id"] if not name_field else None,
        )


def _add_enum_copies(emit, split: str, count: int) -> None:
    groups = _ENUM_LABELS[split]
    for i in range(count):
        label, words = groups[i % len(groups)]
        other_label, other_words = groups[(i + 1) % len(groups)]
        pool = _choice_pool(split, i, words, "enum")
        other_pool = _choice_pool(split, i, other_words, "enum-other")
        correct = pool[i % len(pool)]
        wrong = pool[(i + 1) % len(pool)]
        target_id = _opaque_id("fld", f"{split}|enum|{i}|target")
        distractor_id = _opaque_id("fld", f"{split}|enum|{i}|other")
        target = _enum_field(target_id, label, pool)
        distractor = _enum_field(distractor_id, other_label, other_pool)
        fields = [target, distractor] if i % 2 == 0 else [distractor, target]
        style = i % 3
        if style == 0:
            utterance = f"Set {label} to {correct[0]}"
            active = None
        elif style == 1:
            utterance = f"For {label}, thanks, {wrong[0]}, actually {correct[0]} is what I have"
            active = target_id
        else:
            utterance = f"For {label}, it is {correct[0]}"
            active = target_id
        emit(
            "enum_option",
            "enum",
            utterance,
            fields,
            [set_call("set_choice", target_id, correct[1])],
            active,
        )


def _add_corrections(emit, split: str, count: int) -> None:
    labels = _INT_LABELS[split]
    start = 3200 if split == "train" else 7300
    enum_groups = _ENUM_LABELS[split]
    int_n = count * 4 // 5
    for i in range(count):
        if i < int_n:
            new_v = start + i
            old_v = 1100 + ((i * 17) % 800)
            if old_v == new_v:
                old_v += 100
            label = labels[i % len(labels)]
            fid = f"syn_cor_{split[:2]}_{i:04d}"
            field = _syn_field(fid, "integer", label)
            distractor = _syn_field(f"syn_corx_{split[:2]}_{i:04d}", "integer", labels[(i + 1) % len(labels)])
            spoken_new = hundred_composite(new_v)
            spoken_old = hundred_composite(old_v)
            emit(
                "correction",
                "integer",
                f"Got it, {spoken_old} no actually {spoken_new} for {label}",
                [field, distractor],
                [set_call("set_integer", fid, new_v)],
                fid,
            )
        else:
            j = i - int_n
            label, words = enum_groups[j % len(enum_groups)]
            pool = _choice_pool(split, j, words, "correction")
            correct = pool[j % len(pool)]
            wrong = pool[(j + 1) % len(pool)]
            fid = _opaque_id("fld", f"{split}|correction|{j}")
            emit(
                "correction",
                "enum",
                f"Sorry, {wrong[0]} not {correct[0]}, use {correct[0]} for {label}",
                [_enum_field(fid, label, pool)],
                [set_call("set_choice", fid, correct[1])],
                fid,
            )


def _add_no_change_guards(emit, split: str, count: int) -> None:
    labels = _INT_LABELS[split]
    start = 4000 if split == "train" else 7600
    for i in range(count):
        spoken = hundred_composite(start + i)
        other = hundred_composite(start + i + 25)
        label = labels[i % len(labels)]
        fid = f"syn_neg_{split[:2]}_{i:04d}"
        field = _syn_field(fid, "integer", label)
        style = i % 3
        if style == 0:
            utterance = f"My colleague said {spoken}"
            commit = f"I said {spoken} for {label}"
        elif style == 1:
            utterance = f"I have not decided between {spoken} and {other}"
            commit = f"I have decided on {spoken} for {label}"
        else:
            utterance = f"Not for this form, ignore {spoken}"
            commit = f"For this form, use {spoken} for {label}"
        emit("no_change_guard", "no_change", utterance, [field], [], fid)
        emit(
            "commit_near_miss",
            "integer",
            commit,
            [field],
            [set_call("set_integer", fid, start + i)],
            fid,
        )


def _avoid_test_utterance(text: str) -> str:
    banned = {utterance.casefold() for utterance in _frozen_test_utterances()}
    if text.casefold() not in banned:
        return text
    alt = f"kindly note {text}"
    if alt.casefold() in banned:
        raise RuntimeError(f"synthetic utterance matches frozen test: {text}")
    return alt


_ROUND_FRAMES = {
    "train": (
        "{spoken}",
        "{spoken} please",
        "make it {spoken}",
        "the figure is {spoken}",
        "set the {label} to {spoken}",
        "go with {spoken}",
        "put down {spoken}",
        "{label} is {spoken}",
    ),
    "validation": (
        "record {spoken} for {label}",
        "we need {spoken}",
        "enter {spoken}",
        "{spoken} exactly",
    ),
}
_SMALL_FRACTION_FRAMES = {
    "train": (
        "the figure is {spoken}",
        "put down {spoken}",
        "go with {spoken}",
        "{label} is {spoken}",
    ),
    "validation": (
        "record {spoken} for {label}",
        "enter {spoken}",
    ),
}


def _add_round_hundreds(emit, split: str) -> None:
    """Bare 'N hundred' with no remainder. The model was dividing these by 10."""
    labels = _INT_LABELS[split]
    values = list(range(1100, 10000, 100))
    if split == "validation":
        values = values[1::4]
    frames = _ROUND_FRAMES[split]
    n = 0
    for value in values:
        spoken = hundred_composite(value)
        if " and " in spoken:
            raise RuntimeError(f"round hundred {value} rendered with a remainder: {spoken}")
        for frame in frames:
            label = labels[n % len(labels)]
            fid = f"syn_rnd_{split[:2]}_{n:04d}"
            emit(
                "round_hundred",
                "integer",
                _avoid_test_utterance(frame.format(spoken=spoken, label=label)),
                [_syn_field(fid, "integer", label)],
                [set_call("set_integer", fid, value)],
                fid,
            )
            n += 1


def _add_small_fractions(emit, split: str) -> None:
    """Short 'and a quarter / half / three quarters' below the old fraction range."""
    labels = _NUM_LABELS[split]
    if split == "train":
        ips = range(1, 41)
    else:
        ips = (2, 5, 9, 14, 21, 28, 33, 37)
    frames = _SMALL_FRACTION_FRAMES[split]
    n = 0
    for ip in ips:
        for frac, _tail in ((25, "quarter"), (50, "half"), (75, "quarters")):
            value = round(ip + frac / 100, 2)
            spoken = fraction_words(value)
            if spoken is None:
                raise RuntimeError(value)
            for frame in frames:
                label = labels[n % len(labels)]
                fid = f"syn_sfr_{split[:2]}_{n:04d}"
                emit(
                    "small_fraction",
                    "number",
                    _avoid_test_utterance(frame.format(spoken=spoken, label=label)),
                    [_syn_field(fid, "number", label)],
                    [set_call("set_number", fid, value)],
                    fid,
                )
                n += 1


def _add_glued_fractions(emit, split: str) -> None:
    """'nine hundred and three quarters' is .75, not an extra integer 3."""
    labels = _NUM_LABELS[split]
    if split == "train":
        ips = list(range(100, 1000, 100)) + [1100, 1600, 2400, 3600, 4900, 5600, 7800, 9100, 12400, 18600, 22100]
        frame = "the figure is {spoken}"
    else:
        ips = [1500, 2700, 6400, 8300, 15200]
        frame = "record {spoken} for {label}"
    n = 0
    for ip in ips:
        for frac in (25, 50, 75):
            value = round(ip + frac / 100, 2)
            spoken = fraction_words(value)
            if spoken is None or "hundred" not in spoken:
                raise RuntimeError(f"{value} did not glue to hundred: {spoken}")
            label = labels[n % len(labels)]
            fid = f"syn_glu_{split[:2]}_{n:04d}"
            emit(
                "glued_fraction",
                "number",
                _avoid_test_utterance(frame.format(spoken=spoken, label=label)),
                [_syn_field(fid, "number", label)],
                [set_call("set_number", fid, value)],
                fid,
            )
            n += 1


_YES_WORDS = ("yes", "yep", "yeah", "true", "correct")
_NO_WORDS = ("no", "nah", "nope", "false", "negative")
_BOOL_WORDS = _YES_WORDS + _NO_WORDS
_RESTART_CUES = (
    "{old} hold on from the top {new}",
    "{old} let me say that again {new}",
    "{old} oops sorry {new}",
    "{old} scratch that {mid} scratch that too {new}",
)
_SURFACE_SAME = {
    "train": ("shift pattern", (("brisk", "brisk"), ("steady", "steady"), ("rare", "rare"))),
    "validation": ("pace pattern", (("swift", "swift"), ("even", "even"), ("sparse", "sparse"))),
}
_SURFACE_ALIAS_LABEL = "plate choice"
_SURFACE_ALIAS_WORDS = ("vegetarian", "vegan", "jain")
_SURFACE_OPAQUE = {
    "train": ("desk lane", ("alpha", "beta", "gamma")),
    "validation": ("bay lane", ("crimson", "amber", "cobalt")),
}
_SURFACE_SIBLING = {
    "train": ("stream lane", ("north", "south", "east")),
    "validation": ("row lane", ("inland", "coastal", "upland")),
}
_HUNDRED_BASES = [n for n in range(31, 100) if n not in (70, 71)]


def _ord_suffix(day: int) -> str:
    if 10 <= day % 100 <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def _speak_ord_date(iso: str) -> str:
    year, month, day = (int(part) for part in iso.split("-"))
    return f"{day}{_ord_suffix(day)} {MONTHS[month - 1]} {year}"


def _cycle_iso(index: int, year0: int) -> str:
    day = 1 + (index % 27)
    month = 1 + (index % 12)
    year = year0 + ((index // 12) % 4)
    return f"{year:04d}-{month:02d}-{day:02d}"


def _speak_decimal(value: float) -> str:
    text = f"{value:.1f}"
    whole, frac = text.split(".")
    return f"{int_words(int(whole))} point {int_words(int(frac))}"


def _bool_value(word: str) -> bool:
    if word in _YES_WORDS:
        return True
    if word in _NO_WORDS:
        return False
    raise ValueError(word)


def _label_options(pairs: tuple[tuple[str, str], ...]) -> list[dict[str, str]]:
    return [{"id": oid, "label": lab} for lab, oid in pairs]


def _add_bare_booleans(emit, split: str) -> None:
    """Bare yes/no on an active boolean is set_boolean. Clear stays a separate phrase."""
    labels = _BOOL_LABELS[split]
    n_set = _SKILL_COUNTS[split]["bare_boolean"]
    n_active = n_set * 4 // 5
    n_reach = n_set - n_active
    if n_active + n_reach != _SKILL_COUNTS[split]["bare_clear"]:
        raise RuntimeError("bare boolean and clear counts must match")
    for i in range(n_active):
        label = labels[i % len(labels)]
        word = _BOOL_WORDS[i % len(_BOOL_WORDS)]
        fid = f"syn_bb_{split[:2]}_{i:04d}"
        field = _syn_field(fid, "boolean", label)
        utterance = word if i % 2 == 0 else f"{word}."
        if split == "validation":
            utterance = f"{word} thanks"
        emit(
            "bare_boolean",
            "boolean",
            _avoid_test_utterance(utterance),
            [field],
            [set_call("set_boolean", fid, _bool_value(word))],
            fid,
        )
        clear = f"clear it on {label} {i}" if split == "train" else f"blank out {label} {i}"
        emit(
            "bare_clear",
            "clear",
            _avoid_test_utterance(clear),
            [field],
            [clear_call(fid)],
            fid,
        )
    for i in range(n_reach):
        label = labels[i % len(labels)]
        word = _BOOL_WORDS[i % len(_BOOL_WORDS)]
        fid = f"syn_bbr_{split[:2]}_{i:04d}"
        field = _syn_field(fid, "boolean", label)
        if split == "train":
            utterance = f"for {label} the answer is {word}"
            clear = f"clear the {label} please {i}"
        else:
            utterance = f"mark {label} {word}"
            clear = f"wipe {label} now {i}"
        emit(
            "bare_boolean",
            "boolean",
            _avoid_test_utterance(utterance),
            [field],
            [set_call("set_boolean", fid, _bool_value(word))],
            None,
        )
        emit(
            "bare_clear",
            "clear",
            _avoid_test_utterance(clear),
            [field],
            [clear_call(fid)],
            None,
        )


def _hedge_payload(split: str, index: int) -> tuple[str, str, list[dict[str, Any]], list[dict[str, Any]], str]:
    """Return kind, label, fields, calls, spoken for one hedged commit."""
    kind = ("date", "integer", "number", "boolean", "enum")[index % 5]
    if kind == "date":
        labels = _DATE_LABELS[split]
        label = labels[index % len(labels)]
        iso = _cycle_iso(index, 2024 if split == "train" else 2031)
        fid = f"syn_hd_{split[:2]}_{index:04d}"
        return kind, label, [_syn_field(fid, "date", label)], [set_call("set_date", fid, iso)], _speak_ord_date(iso)
    if kind == "integer":
        labels = _INT_LABELS[split]
        label = labels[index % len(labels)]
        value = (80 + index) if split == "train" else (900 + index)
        fid = f"syn_hd_{split[:2]}_{index:04d}"
        return kind, label, [_syn_field(fid, "integer", label)], [set_call("set_integer", fid, value)], int_words(value)
    if kind == "number":
        labels = _NUM_LABELS[split]
        label = labels[index % len(labels)]
        value = round((2 + (index % 40)) + ((index % 9) * 0.1), 1)
        if split == "validation":
            value = round(value + 40, 1)
        fid = f"syn_hd_{split[:2]}_{index:04d}"
        return kind, label, [_syn_field(fid, "number", label)], [set_call("set_number", fid, value)], _speak_decimal(value)
    if kind == "boolean":
        labels = _BOOL_LABELS[split]
        label = labels[index % len(labels)]
        word = _BOOL_WORDS[index % len(_BOOL_WORDS)]
        fid = f"syn_hd_{split[:2]}_{index:04d}"
        return (
            kind,
            label,
            [_syn_field(fid, "boolean", label)],
            [set_call("set_boolean", fid, _bool_value(word))],
            word,
        )
    label, pairs = _SURFACE_SAME[split]
    spoken, oid = pairs[index % len(pairs)]
    fid = f"syn_hd_{split[:2]}_{index:04d}"
    field = _field(fid, "enum", label, options=_label_options(pairs))
    return "enum", label, [field], [set_call("set_choice", fid, oid)], spoken


def _add_hedged_commits(emit, split: str) -> None:
    """'for now' and 'i guess' commit. Uncertainty with no value stays empty."""
    n = _SKILL_COUNTS[split]["hedged_commit"]
    if n != _SKILL_COUNTS[split]["hedged_abstain"]:
        raise RuntimeError("hedged commit and abstain counts must match")
    for i in range(n):
        kind, label, fields, calls, spoken = _hedge_payload(split, i)
        fid = fields[0]["id"]
        if split == "train":
            utterance = f"{spoken} for now" if i % 2 == 0 else f"{spoken}, i guess"
            abstain = f"not sure for now about {label} {i}"
        else:
            utterance = f"use {spoken} for now" if i % 2 == 0 else f"{spoken} i guess thanks"
            abstain = f"leave {label} not sure for now {i}"
        emit("hedged_commit", kind, _avoid_test_utterance(utterance), fields, calls, fid)
        emit("hedged_abstain", "no_change", _avoid_test_utterance(abstain), fields, [], fid)


def _restart_payload(split: str, index: int) -> tuple[str, list[dict[str, Any]], dict[str, Any], str, str, str]:
    kind = ("integer", "number", "date", "boolean", "enum")[index % 5]
    slot = index // 5
    if kind == "integer":
        labels = _INT_LABELS[split]
        label = labels[slot % len(labels)]
        base = 2 if split == "train" else 60
        new_v = base + (slot % 20)
        mid_v = new_v + 3
        old_v = new_v + 7
        fid = f"syn_rs_{split[:2]}_{index:04d}"
        field = _syn_field(fid, "integer", label)
        return (
            kind,
            [field],
            set_call("set_integer", fid, new_v),
            int_words(old_v),
            int_words(mid_v),
            int_words(new_v),
        )
    if kind == "number":
        labels = _NUM_LABELS[split]
        label = labels[slot % len(labels)]
        base = 1.2 if split == "train" else 30.2
        new_v = round(base + (slot % 9) * 0.3, 1)
        mid_v = round(new_v + 1.1, 1)
        old_v = round(new_v + 2.2, 1)
        fid = f"syn_rs_{split[:2]}_{index:04d}"
        field = _syn_field(fid, "number", label)
        return (
            kind,
            [field],
            set_call("set_number", fid, new_v),
            _speak_decimal(old_v),
            _speak_decimal(mid_v),
            _speak_decimal(new_v),
        )
    if kind == "date":
        labels = _DATE_LABELS[split]
        label = labels[slot % len(labels)]
        year0 = 2022 if split == "train" else 2032
        new_iso = _cycle_iso(slot, year0)
        mid_iso = _cycle_iso(slot + 40, year0)
        old_iso = _cycle_iso(slot + 80, year0)
        fid = f"syn_rs_{split[:2]}_{index:04d}"
        field = _syn_field(fid, "date", label)
        return (
            kind,
            [field],
            set_call("set_date", fid, new_iso),
            _speak_ord_date(old_iso),
            _speak_ord_date(mid_iso),
            _speak_ord_date(new_iso),
        )
    if kind == "boolean":
        labels = _BOOL_LABELS[split]
        label = labels[slot % len(labels)]
        new_word = _BOOL_WORDS[slot % len(_BOOL_WORDS)]
        other = _NO_WORDS if _bool_value(new_word) else _YES_WORDS
        mid_word = other[slot % len(other)]
        old_word = other[(slot + 1) % len(other)]
        fid = f"syn_rs_{split[:2]}_{index:04d}"
        field = _syn_field(fid, "boolean", label)
        return (
            kind,
            [field],
            set_call("set_boolean", fid, _bool_value(new_word)),
            old_word,
            mid_word,
            new_word,
        )
    label, pairs = _SURFACE_SAME[split]
    new_lab, new_id = pairs[slot % len(pairs)]
    mid_lab, _mid_id = pairs[(slot + 1) % len(pairs)]
    old_lab, _old_id = pairs[(slot + 2) % len(pairs)]
    fid = f"syn_rs_{split[:2]}_{index:04d}"
    field = _field(fid, "enum", label, options=_label_options(pairs))
    return kind, [field], set_call("set_choice", fid, new_id), old_lab, mid_lab, new_lab


def _add_restart_corrections(emit, split: str) -> None:
    """The last mention wins. 'I am not sure' after a restart stays empty."""
    n = _SKILL_COUNTS[split]["restart_correction"]
    n_abstain = _SKILL_COUNTS[split]["restart_abstain"]
    if n % len(_RESTART_CUES) != 0:
        raise RuntimeError("restart count must cover every cue evenly")
    for i in range(n):
        kind, fields, call, old, mid, new = _restart_payload(split, i)
        cue = _RESTART_CUES[i % len(_RESTART_CUES)]
        utterance = cue.format(old=old, mid=mid, new=new)
        if split == "validation":
            utterance = f"{utterance} on {fields[0]['label']}"
        emit(
            "restart_correction",
            kind,
            _avoid_test_utterance(utterance),
            fields,
            [call],
            fields[0]["id"],
        )
    labels = _INT_LABELS[split]
    for i in range(n_abstain):
        old = int_words(3 + (i % 15))
        label = labels[i % len(labels)]
        fid = f"syn_rsa_{split[:2]}_{i:04d}"
        field = _syn_field(fid, "integer", label)
        if split == "train":
            utterance = f"{old} hold on from the top, I am not sure about {label} {i}"
        else:
            utterance = f"{old} hold on from the top, leave {label} unresolved {i}"
        emit("restart_abstain", "no_change", _avoid_test_utterance(utterance), [field], [], fid)


def _opaque_pool(split: str, index: int, words: tuple[str, ...], salt: str) -> tuple[tuple[str, str], ...]:
    pool = []
    for slot, word in enumerate(words):
        oid = _opaque_id("opt", f"{split}|{salt}|{index}|{slot}")
        pool.append((word, oid))
    return tuple(pool)


def _add_enum_surface(emit, split: str) -> None:
    """Copy the option id exactly. A spoken alias is not the field id."""
    n = _SKILL_COUNTS[split]["enum_surface"]
    same_label, same_pairs = _SURFACE_SAME[split]
    opaque_label, opaque_words = _SURFACE_OPAQUE[split]
    sibling_label, sibling_words = _SURFACE_SIBLING[split]
    for i in range(n):
        style = i % 4
        sib_pairs = _opaque_pool(split, i, sibling_words, "surface-sib")
        sib_id = _opaque_id("fld", f"{split}|surface-sib|{i}")
        sibling = _field(sib_id, "enum", sibling_label, options=_label_options(sib_pairs))
        if style == 1:
            fid = _opaque_id("fld", f"{split}|surface-alias|{i}")
            pool = _opaque_pool(split, i, _SURFACE_ALIAS_WORDS, "surface-alias")
            veg_id = next(oid for word, oid in pool if word == "vegetarian")
            field = _field(fid, "enum", _SURFACE_ALIAS_LABEL, options=_label_options(pool))
            alias_frames = (
                ("it is veg", "veg please", "go with veg", "that is veg")
                if split == "train"
                else ("it is veg thanks", "veg please thanks", "go with veg thanks", "that is veg thanks")
            )
            spoken = alias_frames[(i // 4) % len(alias_frames)]
            emit(
                "enum_surface",
                "enum",
                _avoid_test_utterance(spoken),
                [field, sibling],
                [set_call("set_choice", fid, veg_id)],
                fid,
            )
            continue
        if style == 2:
            fid = _opaque_id("fld", f"{split}|surface|{i}")
            pool = _opaque_pool(split, i, opaque_words, "surface")
            word, oid = pool[i % len(pool)]
            field = _field(fid, "enum", opaque_label, options=_label_options(pool))
            emit(
                "enum_surface",
                "enum",
                _avoid_test_utterance(word if split == "train" else f"{word} thanks"),
                [sibling, field],
                [set_call("set_choice", fid, oid)],
                fid,
            )
            continue
        word, oid = same_pairs[i % len(same_pairs)]
        fid = f"syn_es_{split[:2]}_{i:04d}"
        field = _field(fid, "enum", same_label, options=_label_options(same_pairs))
        if style == 0:
            utterance = word if split == "train" else f"{word} thanks"
            active: str | None = fid
        else:
            utterance = (
                f"for {same_label} it is {word}" if split == "train" else f"record {same_label} as {word}"
            )
            active = None
        emit(
            "enum_surface",
            "enum",
            _avoid_test_utterance(utterance),
            [field, sibling],
            [set_call("set_choice", fid, oid)],
            active,
        )


# Spoken form -> option id. Ids and labels are invented so frozen test fields stay out.
# Each option is (id, spaced speech, short alias or "").
_CHOICE_MAPS = {
    "train": (
        (
            "dye lot",
            (
                ("indigo_wash", "indigo wash", "indigo"),
                ("rust_wash", "rust wash", "rust"),
                ("sand_wash", "sand wash", "sand"),
            ),
        ),
        (
            "crate grade",
            (
                ("export_grade", "export grade", "export"),
                ("local_grade", "local grade", "local"),
                ("scrap_grade", "scrap grade", "scrap"),
            ),
        ),
        (
            "bay light",
            (
                ("cool_white", "cool white", "cool"),
                ("warm_white", "warm white", "warm"),
                ("amber_glow", "amber glow", "amber"),
            ),
        ),
        (
            "route pace",
            (
                ("dawn_run", "dawn run", "dawn"),
                ("noon_run", "noon run", "noon"),
                ("dusk_run", "dusk run", "dusk"),
            ),
        ),
    ),
    "validation": (
        (
            "cloth weave",
            (
                ("plain_weave", "plain weave", "plain"),
                ("twill_weave", "twill weave", "twill"),
                ("satin_weave", "satin weave", "satin"),
            ),
        ),
        (
            "drum tone",
            (
                ("low_tone", "low tone", "low"),
                ("mid_tone", "mid tone", "mid"),
                ("bright_tone", "bright tone", "bright"),
            ),
        ),
        (
            "ink family",
            (
                ("matte_black", "matte black", "matte"),
                ("gloss_black", "gloss black", "gloss"),
                ("dye_black", "dye black", "dye"),
            ),
        ),
        (
            "shelf zone",
            (
                ("front_zone", "front zone", "front"),
                ("mid_zone", "mid zone", "middle"),
                ("back_zone", "back zone", "back"),
            ),
        ),
    ),
}


def _choice_map_field(fid: str, label: str, options: tuple[tuple[str, str, str], ...]) -> dict[str, Any]:
    rendered = []
    for oid, spaced, alias in options:
        item: dict[str, Any] = {"id": oid, "label": spaced}
        if alias:
            item["aliases"] = [alias]
        rendered.append(item)
    return _field(fid, "enum", label, options=rendered)


def _add_choice_maps(emit, split: str) -> None:
    """Map spoken aliases, spaces, rejections, and light disfluency onto option ids.

    A paired abstain set uses the same fields when the speaker does not commit.
    """
    groups = _CHOICE_MAPS[split]
    n_map = _SKILL_COUNTS[split]["choice_map"]
    n_abstain = _SKILL_COUNTS[split]["choice_abstain"]
    if n_map % (len(groups) * 4) != 0:
        raise RuntimeError("choice_map count must cover every group and style evenly")
    for i in range(n_map):
        label, options = groups[i % len(groups)]
        correct = options[i % len(options)]
        wrong = options[(i + 1) % len(options)]
        fid = f"syn_cm_{split[:2]}_{i:04d}"
        field = _choice_map_field(fid, label, options)
        style = (i // len(groups)) % 4
        oid, spaced, alias = correct
        if style == 0:
            utterance = f"go with {alias}" if split == "train" else f"mark {alias} thanks"
        elif style == 1:
            utterance = spaced if split == "train" else f"{spaced} thanks"
        elif style == 2:
            utterance = (
                f"{spaced} and not {wrong[1]}"
                if split == "train"
                else f"use {spaced}, not {wrong[1]}"
            )
        else:
            utterance = f"uh {spaced}" if split == "train" else f"so so {spaced}"
        emit(
            "choice_map",
            "enum",
            _avoid_test_utterance(utterance),
            [field],
            [set_call("set_choice", fid, oid)],
            fid,
        )
    abstain_frames = (
        "maybe {a} or {b}",
        "not sure if it is {a}",
        "leave the {label} undecided",
        "{a}, or actually never mind",
    )
    for i in range(n_abstain):
        label, options = groups[i % len(groups)]
        fid = f"syn_ca_{split[:2]}_{i:04d}"
        field = _choice_map_field(fid, label, options)
        a = options[i % len(options)][2]
        b = options[(i + 1) % len(options)][2]
        frame = abstain_frames[i % len(abstain_frames)]
        utterance = frame.format(a=a, b=b, label=label)
        if split == "validation":
            utterance = f"{utterance} thanks"
        emit(
            "choice_abstain",
            "no_change",
            _avoid_test_utterance(utterance),
            [field],
            [],
            fid,
        )


def _add_active_integers(emit, split: str) -> None:
    """The active field wins over a familiar distractor when the utterance is only the number."""
    labels = _INT_LABELS[split]
    n = _SKILL_COUNTS[split]["active_integer"]
    start = 30000 if split == "train" else 46000
    frame = "{spoken}" if split == "train" else "enter {spoken}"
    for i in range(n):
        value = start + i
        spoken = str(value) if i % 2 == 0 else int_words(value)
        label = labels[i % len(labels)]
        fid = f"syn_act_{split[:2]}_{i:04d}"
        distractor = _syn_field(f"syn_bud_{split[:2]}_{i:04d}", "integer", "trip budget")
        target = _syn_field(fid, "integer", label)
        fields = [distractor, target] if i % 2 == 0 else [target, distractor]
        emit(
            "active_integer",
            "integer",
            _avoid_test_utterance(frame.format(spoken=spoken)),
            fields,
            [set_call("set_integer", fid, value)],
            fid,
        )


def _add_hundred_remainders(emit, split: str) -> None:
    """'N hundred and <remainder>' outside the ranges the earlier hundred rows covered."""
    labels = _INT_LABELS[split]
    n = _SKILL_COUNTS[split]["hundred_remainder"]
    rem0 = 1 if split == "train" else 60
    frame = "note the total {spoken}" if split == "train" else "log the total {spoken}"
    for i in range(n):
        base = _HUNDRED_BASES[i % len(_HUNDRED_BASES)]
        rem = rem0 + (i % 39)
        value = base * 100 + rem
        spoken = hundred_composite(value)
        if " and " not in spoken:
            raise RuntimeError(f"{value} rendered without a remainder: {spoken}")
        label = labels[i % len(labels)]
        fid = f"syn_rem_{split[:2]}_{i:04d}"
        emit(
            "hundred_remainder",
            "integer",
            _avoid_test_utterance(frame.format(spoken=spoken)),
            [_syn_field(fid, "integer", label)],
            [set_call("set_integer", fid, value)],
            fid,
        )


def _add_large_integers(emit, split: str) -> None:
    """Keep every digit of a large spoken integer."""
    labels = _INT_LABELS[split]
    n = _SKILL_COUNTS[split]["large_integer"]
    start = 100000 if split == "train" else 500000
    for i in range(n):
        value = start + i * 10007
        spoken = int_words(value)
        label = labels[i % len(labels)]
        fid = f"syn_lrg_{split[:2]}_{i:04d}"
        target = _syn_field(fid, "integer", label)
        distractor = _syn_field(f"syn_lrx_{split[:2]}_{i:04d}", "integer", "trip budget")
        if i % 2 == 0:
            utterance = spoken if split == "train" else f"read {spoken}"
            active: str | None = fid
        else:
            utterance = (
                f"for {label} {spoken}" if split == "train" else f"record {label} as {spoken}"
            )
            active = None
        emit(
            "large_integer",
            "integer",
            _avoid_test_utterance(utterance),
            [target, distractor],
            [set_call("set_integer", fid, value)],
            active,
        )
