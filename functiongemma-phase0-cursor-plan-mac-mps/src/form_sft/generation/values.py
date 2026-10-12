"""Value generation + surface rendering (numbers, dates, strings, booleans, enums).

Value generation is *semantic truth* (what the operation will contain).
Rendering is *surface realization* (how the value is spoken/typed).
"""
from __future__ import annotations

import datetime as dt
import math
import random
from dataclasses import dataclass, field

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
         "sixteen seventeen eighteen nineteen").split()
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10**9, "billion"), (10**6, "million"), (10**3, "thousand")]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]

POS_WORDS = ["yes", "yeah", "yep", "true", "correct", "sure", "affirmative"]
NEG_WORDS = ["no", "nope", "false", "negative", "nah"]

FIRST = ["Priya", "Arjun", "Meera", "Rohan", "Kavya", "Vikram", "Ananya", "Suresh", "Divya", "Imran", "Neha", "Karthik",
         "Sana", "Rahul", "Pooja", "Aditya", "Lakshmi", "Manoj", "Farah", "Deepak", "Tanvi", "Harish", "Isha", "Nikhil"]
LAST = ["Nair", "Sharma", "Iyer", "Khan", "Reddy", "Patel", "Mehta", "Das", "Gupta", "Menon", "Joshi", "Kulkarni",
        "Singh", "Bose", "Pillai", "Verma", "Rao", "Chopra", "Banerjee", "Shetty"]
CITIES = ["Mumbai", "Pune", "Chennai", "Kochi", "Jaipur", "Lucknow", "Nagpur", "Indore", "Surat", "Bhopal", "Madurai",
          "Mysuru", "Patna", "Ranchi", "Guwahati", "Dehradun", "Vizag", "Coimbatore", "Nashik", "Raipur", "Agra",
          "Udaipur", "Jodhpur", "Thrissur", "Hubli", "Vadodara", "Amritsar", "Shimla"]
STREETS = ["Park Street", "Nehru Nagar", "Gandhi Road", "Lake View Lane", "Temple Road", "Station Road", "Rose Avenue",
           "Church Street", "Market Road", "River Side Drive", "Hill Crest Road", "Anna Salai"]
COMPANY_A = ["Sunrise", "Bluebird", "Greenfield", "Orion", "Lotus", "Silverline", "Pioneer", "Evergreen", "Horizon", "Zenith"]
COMPANY_B = ["Traders", "Logistics", "Foods", "Systems", "Industries", "Academy", "Insurance", "Exports", "Holdings", "Labs"]
PLATE_STATES = ["KA", "MH", "TN", "DL", "GJ", "KL", "RJ", "UP"]


# ---------------------------------------------------------------- numbers --

def int_words(n: int, hyphen: bool = False) -> str:
    if n < 0:
        return "minus " + int_words(-n, hyphen)
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, r = _TENS[n // 10], n % 10
        if not r:
            return t
        return f"{t}-{_ONES[r]}" if hyphen else f"{t} {_ONES[r]}"
    if n < 1000:
        h, r = divmod(n, 100)
        s = f"{_ONES[h]} hundred"
        return s + (f" and {int_words(r, hyphen)}" if r else "")
    for scale, name in _SCALES:
        if n >= scale:
            q, r = divmod(n, scale)
            s = f"{int_words(q, hyphen)} {name}"
            if r:
                s += (" and " if r < 100 else " ") + int_words(r, hyphen)
            return s
    raise AssertionError("unreachable")


def digit_words(s: str) -> str:
    return " ".join(_ONES[int(c)] for c in s)


def number_digits(v: float) -> str:
    s = f"{v:.2f}".rstrip("0")
    return s[:-1] if s.endswith(".") else s


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}" + {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def numeric_forms(value, ftype: str) -> dict[str, str]:
    """All value-preserving surface forms for an integer/number value."""
    forms: dict[str, str] = {}
    if ftype == "integer":
        forms["digits"] = str(value)
        if abs(value) >= 1000:
            forms["grouped"] = f"{value:,}"
        forms["words"] = int_words(value)
        if int_words(value, True) != forms["words"]:
            forms["hyphen_words"] = int_words(value, True)
        if value < 100 and value >= 20 and value % 10:
            forms["glued_words"] = int_words(value).replace(" ", "")
    else:
        ip = int(value)
        digits = number_digits(value)
        dec = digits.split(".")[1]
        forms["digits"] = digits
        if abs(ip) >= 1000:
            forms["grouped"] = f"{ip:,}." + dec
        forms["words"] = f"{int_words(ip)} point {digit_words(dec)}"
        hy = f"{int_words(ip, True)} point {digit_words(dec)}"
        if hy != forms["words"]:
            forms["hyphen_words"] = hy
    return forms


def hundred_composite(n: int) -> str:
    h, r = divmod(n, 100)
    return f"{int_words(h)} hundred" + (f" and {int_words(r)}" if r else "")


def fraction_words(v: float) -> str | None:
    ip = int(v)
    frac = round((v - ip) * 100)
    tail = {25: "and a quarter", 50: "and a half", 75: "and three quarters"}.get(frac)
    if tail is None or ip < 1:
        return None
    return f"{int_words(ip)} {tail}"


# ------------------------------------------------------------- generation --

def rand_int(rng: random.Random, lo: int, hi: int) -> int:
    if hi - lo <= 200:
        return rng.randint(lo, hi)
    lo_d, hi_d = len(str(max(lo, 1))), len(str(hi))
    d = rng.randint(lo_d, hi_d)
    a, b = max(lo, 10 ** (d - 1)), min(hi, 10**d - 1)
    if a > b:
        a, b = lo, hi
    v = rng.randint(a, b)
    if b - a > 1000 and rng.random() < 0.5:
        v = min(max(round(v, -2), lo), hi)
    return v


def gen_int(rng: random.Random, lo: int, hi: int, holdout: bool) -> int:
    if holdout:
        lo, hi = max(lo, 1100), min(hi, 9999)
        if lo > hi:
            raise ValueError("field cannot host hundred_composite values")
    return rand_int(rng, lo, hi)


def gen_number(rng: random.Random, lo: float, hi: float, holdout: bool) -> float:
    lo_i, hi_i = math.ceil(lo), math.floor(hi) - 1
    if hi_i < lo_i:
        raise ValueError("number field range too narrow")
    ip = rand_int(rng, lo_i, hi_i)
    if holdout:
        ip = max(ip, 1)
        frac = rng.choice([25, 50, 75])
    elif rng.random() < 0.6:
        frac = rng.randint(1, 9) * 10
    else:
        frac = rng.choice([f for f in range(1, 100) if f % 10])
    return round(ip + frac / 100, 2)


def gen_date(rng: random.Random, lo: str, hi: str) -> str:
    a, b = dt.date.fromisoformat(lo).toordinal(), dt.date.fromisoformat(hi).toordinal()
    return dt.date.fromordinal(rng.randint(a, b)).isoformat()


def gen_string(rng: random.Random, pool: str) -> str:
    if pool == "name":
        return f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    if pool == "city":
        return rng.choice(CITIES)
    if pool == "street":
        return rng.choice(STREETS)
    if pool == "company":
        return f"{rng.choice(COMPANY_A)} {rng.choice(COMPANY_B)}"
    if pool == "plate":
        return (f"{rng.choice(PLATE_STATES)} {rng.randint(1, 60):02d} "
                f"{rng.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}{rng.choice('ABCDEFGHJKLMNPRSTUVWXYZ')} {rng.randint(1000, 9999)}")
    raise ValueError(f"unknown pool {pool}")


def gen_value(defn, fld: dict, rng: random.Random, holdout: bool = False, avoid=()):
    """Generate a legal value for field ``fld`` (a SemanticField dict)."""
    t = fld["type"]
    for _ in range(200):
        if t == "integer":
            v = gen_int(rng, fld["minimum"], fld["maximum"], holdout)
        elif t == "number":
            v = gen_number(rng, fld["minimum"], fld["maximum"], holdout)
        elif t == "boolean":
            v = rng.random() < 0.5
        elif t == "enum":
            v = rng.choice(fld["options"])["id"]
        elif t == "string":
            v = gen_string(rng, defn.pool)
        elif t == "date":
            v = gen_date(rng, fld["minimum"], fld["maximum"])
        else:
            raise ValueError(t)
        if not any(_same(v, a) for a in avoid):
            return v
    raise RuntimeError("could not generate value distinct from avoid-set")


def _same(a, b) -> bool:
    return type(a) is type(b) and a == b


# -------------------------------------------------------------- rendering --

@dataclass
class Rendered:
    text: str
    form: str
    forms: list[str] = field(default_factory=list)  # every value-preserving surface
    option_surface: str | None = None


def render_value(fld: dict, value, rng: random.Random, holdout: bool = False) -> Rendered:
    t = fld["type"]
    if t in ("integer", "number"):
        forms = numeric_forms(value, t)
        extra = None
        if holdout:
            if t == "integer":
                extra = ("hundred_composite", hundred_composite(value))
            else:
                fw = fraction_words(value)
                if fw is None:
                    raise ValueError("value not expressible as fraction_words")
                extra = ("fraction_words", fw)
        allv = list(forms.values()) + ([extra[1]] if extra else [])
        if extra:
            return Rendered(extra[1], extra[0], allv)
        general = [k for k in ("digits", "words", "grouped") if k in forms]
        weights = {"digits": 4, "words": 4, "grouped": 2}
        k = rng.choices(general, weights=[weights[g] for g in general])[0]
        return Rendered(forms[k], k, allv)
    if t == "boolean":
        w = rng.choice(POS_WORDS if value else NEG_WORDS)
        return Rendered(w, "bool_word", [w])
    if t == "enum":
        opt = next(o for o in fld["options"] if o["id"] == value)
        surfaces = [opt["label"]] + list(opt.get("aliases", []))
        s = surfaces[0] if (len(surfaces) == 1 or rng.random() < 0.7) else rng.choice(surfaces[1:])
        return Rendered(s, "enum_label", surfaces, option_surface=s)
    if t == "string":
        return Rendered(value, "verbatim", [value])
    if t == "date":
        d = dt.date.fromisoformat(value)
        m, day, y = MONTHS[d.month - 1], d.day, d.year
        forms = {
            "day_month_year": f"{day} {m} {y}",
            "ordinal_day_month": f"{ordinal(day)} {m} {y}",
            "month_ordinal_year": f"{m} {ordinal(day)}, {y}",
            "ordinal_of_month": f"{ordinal(day)} of {m} {y}",
            "abbr_month": f"{day} {m[:3]} {y}",
        }
        k = rng.choice(list(forms))
        return Rendered(forms[k], k, list(forms.values()))
    raise ValueError(t)
