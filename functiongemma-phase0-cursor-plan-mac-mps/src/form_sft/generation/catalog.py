"""Field catalog for Phase-0 semantic truth.

The catalog is the *only* source of field/option identity. Field IDs are a pure
function of (field definition, identity class) so that holdout registries can
enumerate every ID a holdout definition can ever take.

Identity classes
  semantic            id = snake_case(label)            e.g. ``age_in_years``
  synthetic_semantic  id = abbreviated/structured code   e.g. ``hea_age_in_ye_0``
  opaque              id = hash-derived, no semantics    e.g. ``fld_3fa9c21b``
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

FIELD_TYPES = ("integer", "number", "boolean", "enum", "string", "date")
IDENTITIES = ("semantic", "synthetic_semantic", "opaque")
GENERAL_DOMAINS = ("home", "vehicle", "health", "education", "travel", "retail")
HOLDOUT_DOMAINS = ("agri", "logistics")
ALL_DOMAINS = GENERAL_DOMAINS + HOLDOUT_DOMAINS

_ID_SALT = "form-sft-phase0-ids-v1"


@dataclass(frozen=True)
class FieldDef:
    key: str
    domain: str
    type: str
    label: str
    minimum: int | float | str | None = None
    maximum: int | float | str | None = None
    options: tuple[tuple[str, tuple[str, ...]], ...] = ()  # (label, aliases)
    pool: str | None = None
    holdout_enum: bool = False
    index: int = 0  # position within its domain (stable)


def snake(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _h(*parts: str, n: int = 8) -> str:
    return hashlib.sha256(":".join((_ID_SALT, *parts)).encode()).hexdigest()[:n]


# domain -> type -> entries. The 2nd enum of every GENERAL domain is the enum holdout.
_RAW: dict[str, dict[str, list]] = {
    "home": {
        "integer": [("household size", 1, 12), ("monthly rent", 2000, 150000)],
        "number": [("floor area in square metres", 12.0, 600.0), ("monthly electricity bill", 150.0, 25000.0)],
        "boolean": ["balcony available", "pets allowed"],
        "enum": [("heating type", ["gas|piped gas", "electric", "solar", "none"]),
                 ("furnishing level", ["unfurnished", "semi furnished", "fully furnished"])],
        "string": [("landlord name", "name"), ("home city", "city")],
        "date": [("move in date", "2024-01-01", "2029-12-31"), ("lease end date", "2025-01-01", "2031-12-31")],
    },
    "vehicle": {
        "integer": [("seat count", 2, 9), ("odometer reading in kilometres", 100, 350000)],
        "number": [("engine capacity in litres", 0.6, 6.5), ("annual insurance premium", 500.0, 90000.0)],
        "boolean": ["airbags fitted", "second owner"],
        "enum": [("fuel type", ["petrol|gasoline", "diesel", "electric", "cng"]),
                 ("transmission mode", ["manual", "automatic", "semi automatic"])],
        "string": [("registration number", "plate"), ("insurer name", "company")],
        "date": [("purchase date", "2015-01-01", "2026-12-31"), ("insurance expiry date", "2026-01-01", "2031-12-31")],
    },
    "health": {
        "integer": [("age in years", 0, 120), ("daily step count", 0, 60000)],
        "number": [("body weight in kilograms", 2.0, 250.0), ("blood glucose level", 40.0, 600.0)],
        "boolean": ["currently smoking", "allergic to penicillin"],
        "enum": [("blood group", ["a positive", "b positive", "o positive", "ab positive"]),
                 ("exercise frequency", ["never", "occasionally", "weekly", "daily"])],
        "string": [("patient name", "name"), ("family doctor", "name")],
        "date": [("last checkup date", "2020-01-01", "2026-12-31"), ("next appointment date", "2026-01-01", "2028-12-31")],
    },
    "education": {
        "integer": [("number of siblings", 0, 15), ("annual tuition fee", 5000, 1500000)],
        "number": [("grade point average", 0.0, 10.0), ("attendance percentage", 0.0, 100.0)],
        "boolean": ["scholarship holder", "hostel resident"],
        "enum": [("study stream", ["science", "commerce", "arts", "vocational"]),
                 ("exam board", ["cbse", "icse", "state board", "ib"])],
        "string": [("student name", "name"), ("school name", "company")],
        "date": [("admission date", "2015-01-01", "2026-12-31"), ("graduation date", "2026-01-01", "2032-12-31")],
    },
    "travel": {
        "integer": [("number of passengers", 1, 12), ("trip budget", 1000, 2000000)],
        "number": [("checked baggage weight in kilograms", 0.0, 60.0), ("hotel nightly rate", 500.0, 80000.0)],
        "boolean": ["travel insurance required", "window seat preferred"],
        "enum": [("cabin class", ["economy", "premium economy", "business", "first"]),
                 ("meal preference", ["vegetarian|veg", "vegan", "non vegetarian|non veg", "jain"])],
        "string": [("destination city", "city"), ("lead traveller name", "name")],
        "date": [("departure date", "2026-01-01", "2028-12-31"), ("return date", "2026-01-01", "2029-12-31")],
    },
    "retail": {
        "integer": [("items in cart", 1, 200), ("loyalty points balance", 0, 500000)],
        "number": [("discount percentage", 0.0, 90.0), ("order total amount", 10.0, 500000.0)],
        "boolean": ["gift wrap requested", "express delivery selected"],
        "enum": [("payment method", ["card", "upi", "net banking", "cash on delivery|cod"]),
                 ("delivery slot", ["morning", "afternoon", "evening", "night"])],
        "string": [("shipping street", "street"), ("customer name", "name")],
        "date": [("order date", "2024-01-01", "2027-12-31"), ("delivery date", "2024-01-01", "2028-12-31")],
    },
    "agri": {
        "integer": [("number of cattle", 0, 500), ("harvest quantity in kilograms", 50, 900000)],
        "number": [("field area in hectares", 0.1, 200.0), ("seasonal rainfall in millimetres", 10.0, 3000.0)],
        "boolean": ["drip irrigation installed", "organic certified"],
        "enum": [("soil type", ["clay", "sandy", "loamy", "black cotton"]),
                 ("crop season", ["kharif", "rabi", "zaid"])],
        "string": [("farmer name", "name"), ("village name", "city")],
        "date": [("sowing date", "2024-01-01", "2028-12-31"), ("harvest date", "2024-01-01", "2029-12-31")],
    },
    "logistics": {
        "integer": [("pallet count", 1, 60), ("shipment value", 1000, 9000000)],
        "number": [("container load in tonnes", 0.5, 32.0), ("freight charge", 200.0, 400000.0)],
        "boolean": ["cold chain required", "customs cleared"],
        "enum": [("carrier mode", ["road", "rail", "air", "sea"]),
                 ("priority level", ["standard", "expedited", "critical"])],
        "string": [("consignee name", "company"), ("warehouse city", "city")],
        "date": [("dispatch date", "2024-01-01", "2028-12-31"), ("delivery deadline", "2024-01-01", "2029-12-31")],
    },
}


def _build_defs() -> tuple[FieldDef, ...]:
    out: list[FieldDef] = []
    for domain in ALL_DOMAINS:
        idx = 0
        for ftype in FIELD_TYPES:
            for n, entry in enumerate(_RAW[domain][ftype]):
                if ftype in ("integer", "number"):
                    label, lo, hi = entry
                    d = FieldDef(f"{domain}.{snake(label)}", domain, ftype, label, lo, hi, index=idx)
                elif ftype == "boolean":
                    d = FieldDef(f"{domain}.{snake(entry)}", domain, ftype, entry, index=idx)
                elif ftype == "enum":
                    label, opts = entry
                    options = []
                    for o in opts:
                        parts = o.split("|")
                        options.append((parts[0], tuple(parts[1:])))
                    d = FieldDef(f"{domain}.{snake(label)}", domain, ftype, label, options=tuple(options),
                                 holdout_enum=(domain in GENERAL_DOMAINS and n == 1), index=idx)
                elif ftype == "string":
                    label, pool = entry
                    d = FieldDef(f"{domain}.{snake(label)}", domain, ftype, label, pool=pool, index=idx)
                else:
                    label, lo, hi = entry
                    d = FieldDef(f"{domain}.{snake(label)}", domain, ftype, label, lo, hi, index=idx)
                out.append(d)
                idx += 1
    return tuple(out)


FIELD_DEFS: tuple[FieldDef, ...] = _build_defs()
DEF_BY_KEY = {d.key: d for d in FIELD_DEFS}
GENERAL_DEFS = tuple(d for d in FIELD_DEFS if d.domain in GENERAL_DOMAINS and not d.holdout_enum)
ENUM_HOLDOUT_DEFS = tuple(d for d in FIELD_DEFS if d.holdout_enum)
HOLDOUT_DOMAIN_DEFS = tuple(d for d in FIELD_DEFS if d.domain in HOLDOUT_DOMAINS)


def field_id(d: FieldDef, identity: str) -> str:
    if identity == "semantic":
        return snake(d.label)
    if identity == "synthetic_semantic":
        words = snake(d.label).split("_")
        return f"{d.domain[:3]}_{'_'.join(w[:4] for w in words[:3])}_{d.index}"
    if identity == "opaque":
        return f"fld_{_h(d.key)}"
    raise ValueError(f"unknown identity {identity}")


def option_id(d: FieldDef, i: int, identity: str) -> str:
    label = d.options[i][0]
    if identity == "semantic":
        return snake(label)
    if identity == "synthetic_semantic":
        return f"opt_{snake(label)[:8]}_{i}"
    if identity == "opaque":
        return f"opt_{_h(d.key, str(i), n=6)}"
    raise ValueError(f"unknown identity {identity}")


def build_field(d: FieldDef, identity: str) -> dict:
    """SemanticField as plain dict (validated later by pydantic)."""
    f: dict = {"field_id": field_id(d, identity), "label": d.label, "type": d.type}
    if d.type in ("integer", "number", "date"):
        f["minimum"] = d.minimum
        f["maximum"] = d.maximum
    if d.type == "enum":
        f["options"] = [
            {"id": option_id(d, i, identity), "label": lab, "aliases": list(al)}
            for i, (lab, al) in enumerate(d.options)
        ]
    return f


def all_field_ids(defs, identities=IDENTITIES) -> set[str]:
    return {field_id(d, i) for d in defs for i in identities}


def all_option_ids(defs, identities=IDENTITIES) -> set[str]:
    return {option_id(d, i, ident) for d in defs if d.type == "enum" for i in range(len(d.options)) for ident in identities}
