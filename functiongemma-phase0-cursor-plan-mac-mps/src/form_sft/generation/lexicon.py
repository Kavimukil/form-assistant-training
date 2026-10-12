"""Deterministic templates for G4 realization.

Every list that is indexed by template variant has exactly 6 entries; variant 5
(``tpl5``) is the held-out template family and is only used by ``unseen_templates``
test rows. Source styles are alternate deterministic paraphrase sets (no external
teacher is ever called).

Placeholders: {value} {field} {owner} {other} {minutes}; correction cues are
wrapped in [[...]] so ASR transforms can never touch them.
"""
from __future__ import annotations

# --- positive (extraction) frames, per source -------------------------------
ACTIVE_FRAMES = {
    "deterministic_template": ["{value}", "it is {value}", "{value} please", "make it {value}",
                               "the answer is {value}", "i would say {value}"],
    "teacher_generated": ["that would be {value}", "i believe it is {value}", "it should be {value}",
                          "go ahead with {value}", "{value} is what i have", "please note {value}"],
    "curated_manual": ["um {value}", "it's {value} for that one", "well, {value}", "that one is {value}",
                       "put down {value}", "yeah so {value}"],
    "realistic_asr_seed": ["uh {value} i think", "its {value} yeah", "so like {value}", "{value} yeah",
                           "let me see its {value}", "okay {value} thanks"],
}
REACHABLE_FRAMES = {
    "deterministic_template": ["set {field} to {value}", "{field} is {value}", "the {field} is {value}",
                               "my {field} is {value}", "update {field} to {value}", "put {value} for {field}"],
    "teacher_generated": ["please record {value} as the {field}", "for the {field}, {value} is correct",
                          "the {field} should be {value}", "i want the {field} to be {value}",
                          "{field} would be {value}", "go with {value} for {field}"],
    "curated_manual": ["regarding {field}, it is {value}", "{field}: {value}", "about the {field}, {value}",
                       "so {field} is {value}", "make {field} {value}", "for {field} use {value}"],
    "realistic_asr_seed": ["uh {field} is {value}", "so {field} its {value}", "okay {field} {value} yeah",
                           "{field} is like {value}", "put down {value} for the {field}", "yeah {field} is {value}"],
}

# --- wrappers: (prefix, suffix) per source/variant ---------------------------
WRAPPERS = {
    "deterministic_template": [("", "")] * 6,
    "teacher_generated": [("sure, ", ""), ("okay, ", ""), ("alright, ", " thanks"), ("right, ", ""),
                          ("thanks, ", ""), ("got it, ", " please")],
    "curated_manual": [("", ""), ("hmm ", ""), ("", " i guess"), ("well ", ""), ("ok ", ""), ("", " for now")],
    "realistic_asr_seed": [("uh ", ""), ("um ", " yeah"), ("so ", ""), ("hi ", ""), ("", " thanks"), ("okay so ", "")],
}

# --- no_change frames (shared across sources; wrappers add source style) -----
OWNERS = ["my wife's", "my brother's", "my manager's", "her", "his", "my neighbour's", "my father's"]
NC_FRAMES = {
    "other_person": ["{owner} {field} is {value}", "{owner} {field} was {value}", "i heard {owner} {field} is {value}",
                     "{owner} {field} is {value} not mine", "{value} is {owner} {field}",
                     "just so you know {owner} {field} is {value}"],
    "hypothetical": ["if the {field} were {value} would that be okay?", "what if the {field} was {value}?",
                     "suppose the {field} is {value}, would that matter?", "would {value} work for {field}?",
                     "is {value} allowed for the {field}?", "can the {field} be {value} later?"],
    "negated_assertion": ["the {field} is not {value}", "it is not {value}", "i did not say {value} for {field}",
                          "{value} is wrong for the {field}", "no, the {field} is definitely not {value}",
                          "never put {value} as the {field}"],
    "nonreachable_field": ["the {other} is {value}", "set {other} to {value}", "my {other} is {value}",
                           "{other} should be {value}", "please update {other} to {value}",
                           "about the {other}, it is {value}"],
    "unrelated": ["can you repeat the question", "hold on, give me {minutes} minutes", "what does that mean",
                  "i will check and get back to you", "let me find my documents first",
                  "sorry i did not catch that"],
    "declined": ["i do not know", "i would rather not say", "skip this one", "not sure about that",
                 "leave it blank for now", "i prefer not to answer"],
    "question": ["what is the {field} supposed to be?", "is the {field} required?", "why do you need the {field}?",
                 "what format should the {field} be in?", "where do i find the {field}?", "can you explain {field}?"],
}
NC_KIND_WEIGHTS = {"other_person": 25, "hypothetical": 10, "negated_assertion": 10, "nonreachable_field": 15,
                   "unrelated": 15, "declined": 15, "question": 10}
NC_ADVERSARIAL_WEIGHTS = {"other_person": 35, "hypothetical": 20, "negated_assertion": 25, "nonreachable_field": 20}
NC_NEEDS_VALUE = {"other_person", "hypothetical", "negated_assertion", "nonreachable_field"}

# --- correction expressions: replace the {value} slot -----------------------
# subtype -> 3 variants (0,1 general; 2 = held-out correction family)
CORRECTIONS = {
    "explicit_no_replacement": ["{old}, [[no]], {new}", "{old} [[no no]] {new}", "{old}, [[no that is wrong]], {new}"],
    "sorry_self_correction": ["{old}, [[sorry]], {new}", "{old} [[sorry i meant]] {new}", "{old} [[oops sorry]] {new}"],
    "actually_correction": ["{old}, [[actually]] {new}", "{old} [[no actually]] {new}", "{old} [[well actually]], {new}"],
    "restart_repetition": ["{old} [[let me say that again]] {new}", "{old} [[wait start over]] {new}",
                           "{old} [[hold on from the top]] {new}"],
    "negated_distractor": ["[[not]] {old}, {new}", "{new} [[and not]] {old}", "{new} [[rather than]] {old}"],
    "multiple_corrections": ["{old}, [[sorry]] {old2}, [[actually]] {new}", "{old} [[no]] {old2} [[no wait]] {new}",
                             "{old} [[scratch that]] {old2} [[scratch that too]] {new}"],
    "unresolved_to_no_change": ["{old}, [[sorry no, forget it]]", "{old} [[actually never mind]]",
                                "{old} [[no wait i am not sure]]"],
}

# --- ASR vocabulary ----------------------------------------------------------
# DELETABLE and HOMOPHONES are disjoint so compound transforms never starve each other.
DELETABLE = {"the", "a", "an", "is", "it", "its", "it's", "my", "that", "please", "just", "okay", "um", "uh",
             "yeah", "thanks", "hi", "hmm", "ok", "well", "i", "would"}
HOMOPHONES = {"so": "sew", "to": "too", "too": "to", "right": "write", "by": "buy", "there": "their",
              "their": "there", "your": "you're", "here": "hear", "hear": "here", "new": "knew", "be": "bee",
              "our": "are", "are": "our", "see": "sea", "wait": "weight", "sure": "shore", "heard": "herd"}
FILLERS = ["um", "uh", "like", "you know", "er", "hmm"]
ASR_LEAD = ["okay", "so"]
