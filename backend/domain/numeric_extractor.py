from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from backend.domain.terms import get_ontology
from backend.domain.units import UNIT_PATTERN, normalize_number, normalize_unit

_NUMBER = r"[-+]?\d+(?:[\s\u00a0]?\d{3})*(?:[,.]\d+)?"
_RANGE_SEP = r"(?:[-–—]|до|\.\.|to)"
_OPERATOR = r"(?:<=|≥|>=|≤|<|>|=|не\s+более|не\s+менее|до|от|около|примерно|порядка|~)?"
_VALUE_UNIT_RE = re.compile(
    rf"(?P<op>{_OPERATOR})\s*"
    rf"(?P<v1>{_NUMBER})"
    rf"(?:\s*{_RANGE_SEP}\s*(?P<v2>{_NUMBER}))?"
    rf"\s*(?P<unit>{UNIT_PATTERN.pattern})",
    re.IGNORECASE | re.UNICODE,
)
_PROPERTY_WINDOW = 96
_ENTITY_WINDOW = 120


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("ё", "е")).strip()


def _operator_to_canonical(raw: str, *, has_range: bool) -> str:
    clean = re.sub(r"\s+", " ", (raw or "").strip().lower())
    if has_range:
        return "range"
    if clean in {"<", "<=", "≤", "до", "не более"}:
        return "<=" if clean in {"<=", "≤", "до", "не более"} else "<"
    if clean in {">", ">=", "≥", "от", "не менее"}:
        return ">=" if clean in {">=", "≥", "от", "не менее"} else ">"
    if clean in {"около", "примерно", "порядка", "~"}:
        return "approx"
    if clean == "=":
        return "="
    return "observed"


@dataclass(slots=True)
class NumericFact:
    fact_id: str
    source_name: str
    object_id: str
    object_type: str
    property_id: str
    property_label: str
    entity_ids: list[str]
    entity_labels: list[str]
    operator: str
    value: float
    value_max: float | None
    raw_value: str
    raw_unit: str
    unit: str
    dimension: str
    normalized_value: float
    normalized_value_max: float | None
    context: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NumericExtractor:
    def __init__(self) -> None:
        self.ontology = get_ontology()
        self.property_labels = self.ontology.property_labels()

    def _detect_property(self, text: str, start: int, end: int, dimension: str) -> tuple[str, str, float]:
        left = max(0, start - _PROPERTY_WINDOW)
        right = min(len(text), end + _PROPERTY_WINDOW)
        window = text[left:right].lower().replace("ё", "е")
        local_start = start - left
        compatible = {
            "concentration": {"concentration", "dry_residue"},
            "temperature": {"temperature"},
            "flow_rate": {"flow_rate"},
            "velocity": {"flow_rate"},
            "current_density": {"current_density"},
            "ph": {"ph"},
            "fraction": {"extraction_recovery", "distribution"},
            "productivity": {"productivity"},
        }.get(dimension, set())

        best: tuple[str, str, float, int] | None = None
        for prop_id, labels in self.property_labels.items():
            for label in labels:
                clean = label.lower().replace("ё", "е")
                if not clean or (len(clean) <= 1 and clean not in {"t"}):
                    continue
                # Однобуквенное t слишком часто встречается внутри слов; учитываем только отдельный токен.
                pattern = rf"(?<![\wА-Яа-яЁё]){re.escape(clean)}(?![\wА-Яа-яЁё])" if len(clean) <= 2 else re.escape(clean)
                for m in re.finditer(pattern, window):
                    distance = min(abs(local_start - m.start()), abs(local_start - m.end()))
                    if prop_id not in compatible:
                        continue
                    if distance > _PROPERTY_WINDOW:
                        continue
                    # В технических текстах название параметра почти всегда стоит перед числом:
                    # "сухой остаток ≤1000 мг/дм3", "температура 60 °C".
                    # Следующий параметр после запятой не должен забирать предыдущее значение.
                    if m.start() > local_start + 5:
                        continue
                    score = 0.72 + max(0.0, (_PROPERTY_WINDOW - distance) / _PROPERTY_WINDOW) * 0.20
                    score += 0.05
                    candidate = (prop_id, label, min(score, 0.97), distance)
                    if best is None or candidate[2] > best[2] or (candidate[2] == best[2] and candidate[3] < best[3]):
                        best = candidate
        if best:
            return best[:3]

        fallback_by_dimension = {
            "concentration": ("concentration", "концентрация", 0.66),
            "temperature": ("temperature", "температура", 0.72),
            "flow_rate": ("flow_rate", "расход/скорость потока", 0.70),
            "velocity": ("flow_rate", "скорость потока", 0.68),
            "current_density": ("current_density", "плотность тока", 0.73),
            "ph": ("ph", "pH", 0.90),
            "fraction": ("extraction_recovery", "процентный показатель", 0.55),
            "productivity": ("productivity", "производительность", 0.72),
        }
        return fallback_by_dimension.get(dimension, ("numeric_value", "числовое значение", 0.50))

    def _detect_entities(self, text: str, start: int, end: int) -> tuple[list[str], list[str]]:
        left = max(0, start - _ENTITY_WINDOW)
        right = min(len(text), end + _ENTITY_WINDOW)
        window = text[left:right]
        matches = self.ontology.find_terms(window)
        ids: list[str] = []
        labels: list[str] = []
        for match in matches:
            if match.term_id not in ids:
                ids.append(match.term_id)
                labels.append(match.canonical_ru)
        return ids[:8], labels[:8]

    def extract(
        self,
        text: str,
        *,
        source_name: str,
        object_id: str,
        object_type: str = "chunk",
        metadata: dict[str, Any] | None = None,
    ) -> list[NumericFact]:
        clean_text = _norm_text(text)
        facts: list[NumericFact] = []
        for idx, m in enumerate(_VALUE_UNIT_RE.finditer(clean_text)):
            raw_unit = m.group("unit")
            unit, dimension, multiplier = normalize_unit(raw_unit)
            try:
                value = normalize_number(m.group("v1"))
                value_max = normalize_number(m.group("v2")) if m.group("v2") else None
            except ValueError:
                continue

            start, end = m.span()
            prop_id, prop_label, confidence = self._detect_property(clean_text, start, end, dimension)
            entity_ids, entity_labels = self._detect_entities(clean_text, start, end)
            context = clean_text[max(0, start - 180) : min(len(clean_text), end + 180)].strip()
            normalized_value = value * multiplier
            normalized_value_max = value_max * multiplier if value_max is not None else None
            op = _operator_to_canonical(m.group("op") or "", has_range=value_max is not None)
            raw_value = clean_text[start:end]
            seed = f"{source_name}|{object_id}|{idx}|{prop_id}|{raw_value}|{context[:40]}"
            fact_id = hashlib.sha1(seed.encode("utf-8", errors="ignore")).hexdigest()[:20]
            facts.append(
                NumericFact(
                    fact_id=fact_id,
                    source_name=source_name,
                    object_id=object_id,
                    object_type=object_type,
                    property_id=prop_id,
                    property_label=prop_label,
                    entity_ids=entity_ids,
                    entity_labels=entity_labels,
                    operator=op,
                    value=value,
                    value_max=value_max,
                    raw_value=raw_value,
                    raw_unit=raw_unit,
                    unit=unit,
                    dimension=dimension,
                    normalized_value=normalized_value,
                    normalized_value_max=normalized_value_max,
                    context=context,
                    confidence=confidence,
                    metadata=metadata or {},
                )
            )
        return facts


_default_extractor: NumericExtractor | None = None


def get_numeric_extractor() -> NumericExtractor:
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = NumericExtractor()
    return _default_extractor
