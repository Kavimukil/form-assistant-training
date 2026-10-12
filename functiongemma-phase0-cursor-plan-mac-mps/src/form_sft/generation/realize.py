"""G4: utterance realization + correction/ASR transforms.

Semantic truth (``sample``) already exists and is never mutated by realization
(only ``sample["realization"]`` bookkeeping is added). Texts carry invisible
sentinels while in flight so transforms can prove they preserved the target:

    U+27E6 .. U+27E7   protected span (field label, option, cue words, old values)
    U+27EA .. U+27EB   protected *value* span (the target value; may be re-spelled
                       by the number-confusion transform, but only to another
                       value-preserving surface form)
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

from form_sft.dataset.families import variant_index
from form_sft.dataset.quota_planner import Slot
from form_sft.generation import lexicon as lex
from form_sft.generation.semantic import slot_rng
from form_sft.generation.teacher import TeacherAdapter, TeacherRejected
from form_sft.generation.values import render_value

PL, PR, VL, VR = "\u27e6", "\u27e7", "\u27ea", "\u27eb"
SENTINELS = PL + PR + VL + VR
VAL_RE = re.compile(VL + r"([^" + VR + r"]*)" + VR)
TOKEN_RE = re.compile(r"[\u27e6\u27ea][^\u27e7\u27eb]*[\u27e7\u27eb]|[^\s\u27e6\u27ea\u27e7\u27eb]+")
CUE_RE = re.compile(r"\[\[(.*?)\]\]")
WORD_RE = re.compile(r"^([^\w']*)([\w'’\-]+)(.*)$")


class ASRTransformError(RuntimeError):
    pass


def prot(s: str) -> str:
    return PL + s + PR


def val(s: str) -> str:
    return VL + s + VR


def strip_sentinels(t: str) -> str:
    return re.sub(f"[{SENTINELS}]", "", t)


def finalize_text(t: str) -> str:
    return " ".join(strip_sentinels(t).split())


def norm(t: str) -> str:
    t = strip_sentinels(t).lower().replace("-", " ").replace(",", "")
    t = re.sub(r"[^a-z0-9. ]", " ", t)
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)
    return " ".join(t.split())


def _contains(text: str, piece: str) -> bool:
    return f" {norm(piece)} " in f" {norm(text)} "


# ------------------------------------------------------------ base realize --

def _field_by_id(sample: dict) -> dict[str, dict]:
    return {f["field_id"]: f for f in sample["context"]["fields"]}


def realize_utterance(sample: dict, slot: Slot, seed: int) -> str:
    """Base (clean, unstyled) utterance with sentinels. Deterministic."""
    rng = slot_rng(seed, slot.id, "realize")
    v = variant_index(slot.families["template"])
    src, mode, kind = slot.source, slot.specialization_mode, sample["nc_kind"]
    fields = _field_by_id(sample)
    real = sample["realization"] = {"template_variant": v, "source": src}
    teacher = TeacherAdapter() if src == "teacher_generated" else None
    if teacher:
        real["teacher_mode"] = teacher.mode

    label = prot(sample["subject_label"])
    if kind is None or kind == "unresolved":
        holdout = slot.families["numeric"] == "numf_holdout" and kind is None
        rv = render_value(fields[sample["value_field_id"]], sample["value"], rng, holdout)
        real.update(value_text=rv.text, value_forms=rv.forms, value_form=rv.form, option_surface=rv.option_surface)
        if teacher:
            frame = teacher.frame(mode, v)
        else:
            frame = (lex.ACTIVE_FRAMES if mode == "active_field" else lex.REACHABLE_FRAMES)[src][v]
        real["template_id"] = f"{src}.positive.{mode}.v{v}"
        core = frame.replace("{value}", val(rv.text)).replace("{field}", label)
    else:
        frame = lex.NC_FRAMES[kind][v]
        real["template_id"] = f"{src}.nc_{kind}.v{v}"
        real.update(value_text=None, value_forms=[], value_form=None, option_surface=None)
        core = frame
        if "{value}" in core:
            rv = render_value(fields[sample["value_field_id"]], sample["value"], rng, False)
            real.update(value_text=rv.text, value_forms=rv.forms, value_form=rv.form)
            core = core.replace("{value}", prot(rv.text))
        core = (core.replace("{field}", label).replace("{owner}", rng.choice(lex.OWNERS))
                .replace("{minutes}", str(rng.randint(2, 30))))
        if "{other}" in core:
            core = core.replace("{other}", prot(sample["other_field"]["label"]))
    prefix, suffix = lex.WRAPPERS[src][v]
    if core.endswith("?"):
        suffix = ""
    return prefix + core + suffix


def apply_correction(text: str, sample: dict, slot: Slot, seed: int) -> str:
    """Replace the value expression by a correction expression. Target is untouched."""
    rng = slot_rng(seed, slot.id, "correction")
    sub = slot.correction_subtype
    var = variant_index(slot.families["correction"])
    tmpl = lex.CORRECTIONS[sub][var]
    m = VAL_RE.search(text)
    if not m:
        raise ValueError(f"{slot.id}: no value slot to correct")
    inner = m.group(1)
    fields = _field_by_id(sample)
    expr = CUE_RE.sub(lambda c: prot(c.group(1)), tmpl)
    if sub == "unresolved_to_no_change":
        expr = expr.replace("{old}", prot(inner))
    else:
        vf = fields[sample["value_field_id"]]
        olds = [render_value(vf, o, rng, False).text for o in sample["old_values"]]
        expr = expr.replace("{old}", prot(olds[0]))
        if "{old2}" in expr:
            expr = expr.replace("{old2}", prot(olds[1]))
        expr = expr.replace("{new}", val(inner))
        sample["realization"]["old_texts"] = olds
    sample["realization"]["correction_template"] = f"{sub}.v{var}"
    return text[: m.start()] + expr + text[m.end():]


# ------------------------------------------------------------------ style --

def capitalize_first(text: str) -> str:
    for i, ch in enumerate(text):
        if ch in SENTINELS:
            continue
        if ch.isalpha():
            return text[:i] + ch.upper() + text[i + 1:]
        return text
    return text


def style_text(text: str, rng: random.Random, force_punct: bool) -> str:
    text = " ".join(text.split())
    text = capitalize_first(text)
    if text and text[-1] not in ".?!" and (force_punct or rng.random() < 0.7):
        text += "."
    return text


# -------------------------------------------------------------------- ASR --

@dataclass
class Tok:
    kind: str  # w | p | v
    text: str
    sp: bool

    def render(self) -> str:
        body = {"w": self.text, "p": prot(self.text), "v": val(self.text)}[self.kind]
        return (" " if self.sp else "") + body


def tokenize(text: str) -> list[Tok]:
    toks = []
    for m in TOKEN_RE.finditer(text):
        s = m.group()
        sp = m.start() > 0 and text[m.start() - 1].isspace()
        if s[0] == PL:
            toks.append(Tok("p", s[1:-1], sp))
        elif s[0] == VL:
            toks.append(Tok("v", s[1:-1], sp))
        else:
            toks.append(Tok("w", s, sp))
    if toks:
        toks[0].sp = False
    return toks


def render_tokens(toks: list[Tok]) -> str:
    return "".join(t.render() for t in toks)


def _core(t: Tok):
    m = WORD_RE.match(t.text) if t.kind == "w" else None
    return m.groups() if m else None


def _recase_front(toks: list[Tok]) -> None:
    """Keep sentence-initial capitalisation on the first word token."""
    if toks and toks[0].kind == "w" and toks[0].text:
        toks[0].text = toks[0].text[0].upper() + toks[0].text[1:]


def _insert_before(toks: list[Tok], i: int, new: Tok) -> None:
    if i == 0:
        old = toks[0]
        was_cap = old.kind == "w" and old.text[:1].isupper() and old.text.lower() not in ("i", "i'm", "i'll")
        if was_cap:
            old.text = old.text[0].lower() + old.text[1:]
        new.sp = False
        old.sp = True
        toks.insert(0, new)
        if was_cap:
            _recase_front(toks)
    else:
        new.sp = toks[i].sp
        toks[i].sp = True
        toks.insert(i, new)


def _alpha_idx(toks):
    return [i for i, t in enumerate(toks) if (c := _core(t)) and len(c[1]) >= 2 and c[1].replace("'", "").isalpha()]


def op_filler(toks, rng, real):
    idxs = [i for i, t in enumerate(toks) if i == 0 or t.sp]
    if not idxs:
        return False
    _insert_before(toks, rng.choice(idxs), Tok("w", rng.choice(lex.FILLERS), False))
    return True


def op_stutter(toks, rng, real):
    idxs = _alpha_idx(toks)
    if not idxs:
        return False
    i = rng.choice(idxs)
    _insert_before(toks, i, Tok("w", _core(toks[i])[1][0].lower() + "-", False))
    return True


def op_duplicate(toks, rng, real):
    idxs = _alpha_idx(toks)
    if not idxs:
        return False
    i = rng.choice(idxs)
    _insert_before(toks, i, Tok("w", _core(toks[i])[1], False))
    return True


def op_deletion(toks, rng, real):
    idxs = [i for i, t in enumerate(toks) if t.kind == "w" and t.text.lower() in lex.DELETABLE]
    if not idxs or len(toks) < 3:
        return False
    i = rng.choice(idxs)
    del toks[i]
    if i == 0:
        toks[0].sp = False
        _recase_front(toks)
    return True


def op_homophone(toks, rng, real):
    idxs = [i for i, t in enumerate(toks) if (c := _core(t)) and c[1].lower() in lex.HOMOPHONES]
    if not idxs:
        return False
    i = rng.choice(idxs)
    lead, core, trail = _core(toks[i])
    rep = lex.HOMOPHONES[core.lower()]
    if core[0].isupper():
        rep = rep[0].upper() + rep[1:]
    toks[i].text = lead + rep + trail
    return True


def op_number_conf(toks, rng, real):
    idxs = [i for i, t in enumerate(toks) if t.kind == "v"]
    if not idxs:
        return False
    i = rng.choice(idxs)
    cur = toks[i].text
    alts = sorted({f for f in real["value_forms"] if f.lower() != cur.lower()})
    if not alts:
        return False
    toks[i].text = rng.choice(alts)
    return True


def op_punct_case(toks, rng, real):
    out = []
    for t in toks:
        if t.kind == "w":
            s = re.sub(r"[.,!?;:]+", "", t.text.lower()) if not re.search(r"\d[.,]\d", t.text) else t.text.lower()
            if not s:
                continue
            t.text = s
        else:
            t.text = t.text.lower()
        out.append(t)
    toks[:] = out
    if toks:
        toks[0].sp = False
    return True


ASR_OPS = {
    "punctuation_case_loss": [op_punct_case], "filler_insertion": [op_filler], "repetition_stutter": [op_stutter],
    "homophone_substitution": [op_homophone], "number_specific_confusion": [op_number_conf],
    "minor_word_deletion": [op_deletion], "minor_word_duplication": [op_duplicate],
}
COMPOUND_POOL = {"filler_insertion": op_filler, "repetition_stutter": op_stutter, "homophone_substitution": op_homophone,
                 "minor_word_deletion": op_deletion, "minor_word_duplication": op_duplicate,
                 "punctuation_case_loss": op_punct_case}


def _has_capacity(toks) -> bool:
    return (any(t.kind == "w" and t.text.lower() in lex.DELETABLE for t in toks)
            and any((c := _core(t)) and c[1].lower() in lex.HOMOPHONES for t in toks)
            and bool(_alpha_idx(toks)) and len(toks) >= 3)


def _protected_seq(toks):
    return [(t.kind, t.text.lower()) for t in toks if t.kind in ("p", "v")]


def apply_asr_noise(text: str, sample: dict, slot: Slot, seed: int) -> str:
    """Conservative synthetic ASR. Never changes field/option/value semantics."""
    rng = slot_rng(seed, slot.id, "asr")
    real = sample["realization"]
    toks = tokenize(text)
    led = False
    if not _has_capacity(toks):
        for w in reversed(lex.ASR_LEAD):
            _insert_before(toks, 0, Tok("w", w, False))
        _recase_front(toks)
        led = True
    seq_before = _protected_seq(toks)
    before_plain = strip_sentinels(render_tokens(toks))
    sub = slot.asr_subtype
    base = toks
    out = None
    for _try in range(25):  # compound ops can cancel (e.g. delete+duplicate of the same word): redraw
        toks = [Tok(t.kind, t.text, t.sp) for t in base]
        if sub == "compound_two_noise":
            pool = dict(COMPOUND_POOL)
            if sample["subject_type"] in ("integer", "number") and sample["nc_kind"] is None:
                pool["number_specific_confusion"] = op_number_conf
            pair = rng.sample(sorted(pool), 2)
            pair.sort(key=lambda n: n == "punctuation_case_loss")  # punctuation/case always last
            ops = [(n, pool[n]) for n in pair]
        else:
            ops = [(sub, f) for f in ASR_OPS[sub]]
        applied = []
        ok = True
        for name, fn in ops:
            if not fn(toks, rng, real):
                ok = False
                break
            applied.append(name)
        out = render_tokens(toks)
        if ok and strip_sentinels(out) != before_plain:
            break
    else:
        raise ASRTransformError(f"{slot.id}: ASR transform {sub} could not be applied effectively")
    # target-preservation proof for protected spans
    seq_after = _protected_seq(toks)
    allowed = {f.lower() for f in real.get("value_forms", [])}
    if len(seq_after) != len(seq_before):
        raise ASRTransformError(f"{slot.id}: protected span count changed")
    for (k0, t0), (k1, t1) in zip(seq_before, seq_after):
        if k0 != k1 or (t0 != t1 and not (k0 == "v" and t1 in allowed)):
            raise ASRTransformError(f"{slot.id}: protected span altered {t0!r}->{t1!r}")
    real["asr_applied"] = applied
    real["asr_lead_added"] = led
    return out


# ------------------------------------------------------------ orchestrate --

def prove_target_preserved(sample: dict, slot: Slot, utterance: str) -> bool:
    """Proof obligations: value surface (any value-preserving form), option surface, and
    (for reachable mode) the field label must still be recoverable from the utterance."""
    real = sample["realization"]
    if sample["nc_kind"] is not None and sample["nc_kind"] != "unresolved":
        return True
    if sample["nc_kind"] == "unresolved":
        return _contains(utterance, real["value_text"]) or any(_contains(utterance, f) for f in real["value_forms"])
    if not any(_contains(utterance, f) for f in real["value_forms"]):
        return False
    if slot.specialization_mode == "reachable_fields" and not _contains(utterance, sample["subject_label"]):
        return False
    return True


def realize_row(sample: dict, slot: Slot, seed: int) -> str:
    """Full pipeline: base -> correction -> style -> ASR -> finalize."""
    text = realize_utterance(sample, slot, seed)
    if slot.correction:
        text = apply_correction(text, sample, slot, seed)
    text = style_text(text, slot_rng(seed, slot.id, "style"), force_punct=slot.asr_noise)
    if slot.asr_noise:
        text = apply_asr_noise(text, sample, slot, seed)
    final = finalize_text(text)
    ok = prove_target_preserved(sample, slot, final)
    if slot.source == "teacher_generated":
        TeacherAdapter.require_preserved(ok, slot.id)
    if not ok:
        raise TeacherRejected(f"{slot.id}: target preservation not provable")
    sample["realization"]["utterance_chars"] = len(final)
    return final
