from __future__ import annotations

from backend.domain.query_parser import ParsedQuery


def build_expanded_query(parsed: ParsedQuery, *, top_facts: list[dict] | None = None, max_terms: int = 40) -> str:
    """Контролируемое расширение запроса.

    Источники expansion: доменный словарь, KG-соседи в YAML-онтологии, найденные top-k факты.
    LLM здесь не используется, чтобы снизить topic drift и галлюцинации терминов.
    """
    terms = list(dict.fromkeys(parsed.expanded_terms[:max_terms]))
    if top_facts:
        for fact in top_facts[:8]:
            for label in fact.get("entity_labels") or []:
                if label not in terms:
                    terms.append(label)
            prop = fact.get("property_label")
            if prop and prop not in terms:
                terms.append(prop)
    numeric = "; ".join(item.raw_value for item in parsed.numeric_constraints)
    geography = ", ".join(parsed.geography)
    years = ""
    if parsed.year_from and parsed.year_to:
        years = f"{parsed.year_from}-{parsed.year_to}"
    return "\n".join(
        part
        for part in [
            f"Исходный запрос: {parsed.original_query}",
            f"Intent: {parsed.intent}",
            f"Двуязычные термины и KG-соседи: {', '.join(terms)}" if terms else "",
            f"Числовые ограничения: {numeric}" if numeric else "",
            f"География: {geography}" if geography else "",
            f"Временной диапазон: {years}" if years else "",
        ]
        if part
    )


def build_structured_context(facts: list[dict], *, limit: int = 10) -> str:
    if not facts:
        return ""
    lines = ["Структурированные факты, извлечённые из корпуса:"]
    for idx, fact in enumerate(facts[:limit], start=1):
        value = fact.get("value")
        value_max = fact.get("value_max")
        raw_value = fact.get("raw_value")
        value_part = raw_value or (f"{value}-{value_max} {fact.get('unit')}" if value_max is not None else f"{value} {fact.get('unit')}")
        entities = ", ".join(fact.get("entity_labels") or []) or "не определено"
        lines.append(
            f"{idx}. Источник={fact.get('source_name')}; объект={fact.get('object_id')}; "
            f"тип={fact.get('object_type')}; параметр={fact.get('property_label')} ({fact.get('property_id')}); "
            f"сущности={entities}; значение={value_part}; confidence={fact.get('confidence')}; "
            f"контекст={fact.get('context')}"
        )
    return "\n".join(lines)
