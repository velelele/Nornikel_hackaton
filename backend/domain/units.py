from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class UnitSpec:
    canonical: str
    aliases: tuple[str, ...]
    value_multiplier: float = 1.0
    dimension: str = "unknown"


UNIT_SPECS: tuple[UnitSpec, ...] = (
    UnitSpec("mg/L", ("мг/л", "мг / л", "mg/l", "mg / l"), 1.0, "concentration"),
    UnitSpec("mg/dm3", ("мг/дм3", "мг/дм³", "mg/dm3", "mg/dm³"), 1.0, "concentration"),
    UnitSpec("g/L", ("г/л", "г / л", "g/l", "g / l"), 1000.0, "concentration"),
    UnitSpec("ppm", ("ppm", "млн-1", "млн⁻¹"), 1.0, "concentration"),
    UnitSpec("percent", ("%", "проц.", "мас.%", "мас. %", "wt.%", "wt %"), 1.0, "fraction"),
    UnitSpec("C", ("°c", "ºc", "град c", "град. c", "градусов c", "градусов цельсия", "c"), 1.0, "temperature"),
    UnitSpec("K", ("k", "к"), 1.0, "temperature"),
    UnitSpec("L/min", ("л/мин", "л / мин", "l/min", "l / min"), 1.0, "flow_rate"),
    UnitSpec("m3/h", ("м3/ч", "м³/ч", "м3 / ч", "м³ / ч", "m3/h", "m³/h"), 1.0, "flow_rate"),
    UnitSpec("m/s", ("м/с", "m/s"), 1.0, "velocity"),
    UnitSpec("cm/s", ("см/с", "cm/s"), 0.01, "velocity"),
    UnitSpec("A/m2", ("а/м2", "а/м²", "a/m2", "a/m²"), 1.0, "current_density"),
    UnitSpec("A/dm2", ("а/дм2", "а/дм²", "a/dm2", "a/dm²"), 100.0, "current_density"),
    UnitSpec("pH", ("ph", "рн", "рh"), 1.0, "ph"),
    UnitSpec("t/day", ("т/сут", "т/сутки", "t/day", "ton/day"), 1.0, "productivity"),
    UnitSpec("t/h", ("т/ч", "t/h"), 1.0, "productivity"),
    UnitSpec("min", ("мин", "min"), 1.0, "time"),
    UnitSpec("h", ("ч", "час", "часов", "h", "hour", "hours"), 1.0, "time"),
)

_ALIAS_TO_SPEC: dict[str, UnitSpec] = {}
for spec in UNIT_SPECS:
    for alias in spec.aliases:
        _ALIAS_TO_SPEC[alias.lower()] = spec
    _ALIAS_TO_SPEC[spec.canonical.lower()] = spec

# Более длинные единицы должны проверяться первыми.
UNIT_PATTERN = re.compile(
    "|".join(
        re.escape(alias)
        for alias in sorted(_ALIAS_TO_SPEC, key=len, reverse=True)
    ),
    re.IGNORECASE,
)


def normalize_unit(unit: str) -> tuple[str, str, float]:
    key = unit.strip().lower().replace("ё", "е")
    key = re.sub(r"\s+", " ", key)
    spec = _ALIAS_TO_SPEC.get(key)
    if spec:
        return spec.canonical, spec.dimension, spec.value_multiplier
    return unit.strip(), "unknown", 1.0


def normalize_number(value: str) -> float:
    clean = value.strip().replace(" ", "").replace("\u00a0", "")
    clean = clean.replace(",", ".")
    return float(clean)
