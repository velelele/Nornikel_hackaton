from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from backend.domain.numeric_extractor import NumericFact
from backend.domain.query_parser import ParsedQuery


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()


class FactStore:
    """Простой JSONL sidecar для структурированных признаков первого этапа.

    На production-этапе этот слой заменяется на Neo4j/Postgres, но интерфейс уже отделён
    от LightRAG и может использоваться для числовых фильтров и provenance.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def replace_source_facts(self, source_name: str, source_hash: str, facts: list[NumericFact]) -> int:
        rows = [
            row
            for row in self._read_all()
            if not (row.get("source_name") == source_name or row.get("metadata", {}).get("source_hash") == source_hash)
        ]
        rows.extend(fact.to_dict() for fact in facts)
        self.path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
        return len(facts)

    def stats(self) -> dict[str, Any]:
        rows = self._read_all()
        by_property: dict[str, int] = {}
        by_source: dict[str, int] = {}
        for row in rows:
            by_property[row.get("property_id") or "unknown"] = by_property.get(row.get("property_id") or "unknown", 0) + 1
            by_source[row.get("source_name") or "unknown"] = by_source.get(row.get("source_name") or "unknown", 0) + 1
        return {"total_facts": len(rows), "by_property": by_property, "by_source": by_source}

    @staticmethod
    def _numeric_matches(fact: dict[str, Any], parsed: ParsedQuery) -> bool:
        if not parsed.numeric_constraints:
            return True
        fact_value = fact.get("normalized_value")
        fact_value_max = fact.get("normalized_value_max")
        fact_dimension = fact.get("dimension")
        fact_entities = set(fact.get("entity_ids") or [])
        for constraint in parsed.numeric_constraints:
            if constraint.dimension != "unknown" and fact_dimension != constraint.dimension:
                continue
            if constraint.entity_ids and fact_entities and not (set(constraint.entity_ids) & fact_entities):
                continue
            if fact_value is None:
                continue
            f_min = float(fact_value)
            f_max = float(fact_value_max) if fact_value_max is not None else f_min
            q_min = float(constraint.value)
            q_max = float(constraint.value_max) if constraint.value_max is not None else q_min
            if constraint.operator in {"range", "observed", "approx", "="}:
                if f_max >= q_min and f_min <= q_max:
                    return True
            elif constraint.operator in {"<=", "<"}:
                if f_min <= q_min:
                    return True
            elif constraint.operator in {">=", ">"}:
                if f_max >= q_min:
                    return True
        return False

    @staticmethod
    def _term_score(fact: dict[str, Any], parsed: ParsedQuery) -> int:
        score = 0
        fact_ids = set(fact.get("entity_ids") or [])
        for term_id in parsed.term_ids:
            if term_id in fact_ids:
                score += 4
        haystack = _norm(" ".join([
            str(fact.get("source_name") or ""),
            str(fact.get("property_label") or ""),
            " ".join(str(x) for x in fact.get("entity_labels") or []),
            str(fact.get("context") or ""),
        ]))
        for term in parsed.expanded_terms[:32]:
            if _norm(term) in haystack:
                score += 1
        return score

    def search(self, parsed: ParsedQuery, *, limit: int = 12) -> list[dict[str, Any]]:
        # Do not inject arbitrary numeric facts into LightRAG query expansion.
        # When a user query has no recognized domain terms, numeric constraints,
        # geography, or time filters, the previous implementation returned the
        # highest-confidence facts from unrelated sources. For queries such as
        # "особенности сейсмограмм при проведении взрывных работ" this polluted
        # LightRAG keyword extraction with source/person names (e.g. "Тяпкина")
        # and led to Query nodes/edges that produced 0 context.
        if not (parsed.term_ids or parsed.numeric_constraints or parsed.geography or parsed.year_from or parsed.year_to):
            return []

        rows = self._read_all()
        candidates: list[tuple[int, float, dict[str, Any]]] = []
        for row in rows:
            if not self._numeric_matches(row, parsed):
                continue
            score = self._term_score(row, parsed)
            if parsed.term_ids and score == 0:
                continue
            if parsed.geography:
                geography_text = _norm(str(row.get("metadata", {}).get("geography") or "") + " " + str(row.get("context") or ""))
                if not any(g in geography_text for g in parsed.geography):
                    # На первом этапе география часто не размечена; не отбрасываем полностью, но понижаем.
                    score -= 1
            confidence = float(row.get("confidence") or 0.0)
            candidates.append((score, confidence, row))
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [row for score, _, row in candidates[:limit] if score >= 0]
