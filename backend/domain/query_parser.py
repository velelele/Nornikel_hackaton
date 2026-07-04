from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from backend.domain.numeric_extractor import get_numeric_extractor
from backend.domain.terms import get_ontology


@dataclass(slots=True)
class NumericConstraint:
    property_id: str
    entity_ids: list[str]
    operator: str
    value: float
    value_max: float | None
    unit: str
    dimension: str
    raw_value: str


@dataclass(slots=True)
class ParsedQuery:
    original_query: str
    intent: str
    term_ids: list[str]
    terms_by_class: dict[str, list[str]]
    numeric_constraints: list[NumericConstraint]
    geography: list[str]
    year_from: int | None = None
    year_to: int | None = None
    required_facets: list[str] = field(default_factory=list)
    expanded_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["numeric_constraints"] = [asdict(item) for item in self.numeric_constraints]
        return data


class DomainQueryParser:
    def __init__(self) -> None:
        self.ontology = get_ontology()
        self.numeric_extractor = get_numeric_extractor()

    @staticmethod
    def _detect_intent(text: str) -> str:
        n = text.lower().replace("ё", "е")
        if any(x in n for x in ("сравни", "сравнение", "вариант а", "вариант б", "vs", "против")):
            return "compare"
        if any(x in n for x in ("покажи все", "найди все", "перечисли", "какие публикации", "эксперименты")):
            return "find_evidence"
        if any(x in n for x in ("что будет", "если повыс", "если сниз", "влия", "как измен")):
            return "what_if"
        if any(x in n for x in ("почему", "причин", "из-за чего", "первоприч")):
            return "causal_hypothesis"
        if any(x in n for x in ("подходит", "рекоменд", "какие методы", "какие решения")):
            return "recommend_methods"
        return "technical_qa"

    @staticmethod
    def _detect_years(text: str) -> tuple[int | None, int | None]:
        now = datetime.now().year
        n = text.lower().replace("ё", "е")
        m = re.search(r"последн(?:ие|их)\s+(\d{1,2})\s+лет", n)
        if m:
            span = int(m.group(1))
            return now - span + 1, now
        years = [int(x) for x in re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", text)]
        if not years:
            return None, None
        return min(years), max(years)

    @staticmethod
    def _required_facets(text: str) -> list[str]:
        n = text.lower().replace("ё", "е")
        facets: list[str] = []
        checks = {
            "sources": ["источник", "публикац", "отчет", "патент", "гост"],
            "experiments": ["эксперимент", "опыт", "протокол"],
            "equipment": ["оборудован", "установк", "ячейк", "ванн", "печ"],
            "conditions": ["услов", "режим", "температур", "скорост", "концентрац", "ph", "рн"],
            "geography": ["росси", "отечествен", "зарубеж", "миров"],
            "economics": ["тэо", "эконом", "capex", "opex", "капитальн", "эксплуатацион"],
            "contradictions": ["противореч", "расхожд", "конфликт"],
        }
        for facet, needles in checks.items():
            if any(needle in n for needle in needles):
                facets.append(facet)
        return facets

    def parse(self, query: str) -> ParsedQuery:
        matches = self.ontology.find_terms(query)
        term_ids: list[str] = []
        terms_by_class: dict[str, list[str]] = {}
        for match in matches:
            if match.term_id not in term_ids:
                term_ids.append(match.term_id)
            terms_by_class.setdefault(match.class_name, [])
            if match.canonical_ru not in terms_by_class[match.class_name]:
                terms_by_class[match.class_name].append(match.canonical_ru)

        raw_numeric = self.numeric_extractor.extract(
            query,
            source_name="query",
            object_id="query",
            object_type="query",
        )
        constraints = [
            NumericConstraint(
                property_id=fact.property_id,
                entity_ids=fact.entity_ids,
                operator=fact.operator,
                value=fact.normalized_value,
                value_max=fact.normalized_value_max,
                unit=fact.unit,
                dimension=fact.dimension,
                raw_value=fact.raw_value,
            )
            for fact in raw_numeric
        ]
        year_from, year_to = self._detect_years(query)
        expanded = self.ontology.expand_term_ids(term_ids)
        return ParsedQuery(
            original_query=query,
            intent=self._detect_intent(query),
            term_ids=term_ids,
            terms_by_class=terms_by_class,
            numeric_constraints=constraints,
            geography=self.ontology.detect_geography(query),
            year_from=year_from,
            year_to=year_to,
            required_facets=self._required_facets(query),
            expanded_terms=expanded,
        )


_default_parser: DomainQueryParser | None = None


def get_query_parser() -> DomainQueryParser:
    global _default_parser
    if _default_parser is None:
        _default_parser = DomainQueryParser()
    return _default_parser
