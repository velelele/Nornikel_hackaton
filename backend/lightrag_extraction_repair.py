from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Iterable

_DEFAULT_TUPLE_DELIMITER = "<|>"
_DEFAULT_RECORD_DELIMITER = "##"
_DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"
_DEFAULT_SOURCE_ID = "unknown_source"
_REPAIR_SCHEMA_VERSION = "nico_repair_v6_newline_records"

_ENTITY_TYPES = {
    "concept", "method", "process", "equipment", "material", "artifact", "event",
    "organization", "person", "location", "parameter", "property", "value", "unit",
    "experiment", "standard", "document", "software", "methodology", "technology",
    "sample", "condition", "result",
}
_TABLE_HEADER_TOKENS = {
    "название", "тип", "описание", "name", "type", "description",
    "entity", "entity name", "entity_type", "relation", "relationship",
    "source", "target", "сущность", "отношение", "источник", "цель",
}


_DOMAIN_ENTITY_PATTERNS: list[tuple[str, str, str, str]] = [
    (r"\b(?:бвр|буровзрыв\w*|взрывн\w*\s+работ\w*)\b", "взрывные работы", "Process", "Технологический процесс разрушения или подготовки горных пород с применением зарядов взрывчатых веществ."),
    (r"\bсейсмограм\w*\b", "сейсмограмма", "Concept", "Запись сейсмических колебаний, используемая для анализа воздействия взрывных работ и динамического расчета."),
    (r"\bвелосиграм\w*\b", "велосиграмма", "Concept", "Запись скорости колебаний или смещений, используемая при анализе сейсмовзрывного воздействия."),
    (r"\bакселерограм\w*\b", "акселерограмма", "Concept", "Запись ускорений, используемая при анализе динамического воздействия."),
    (r"\b(?:ppv|pvs|ppvx|ppvy|ppvz)\b|пиков\w*\s+скорост\w*|скорост\w*\s+(?:смещен\w*|колебан\w*)", "пиковая скорость колебаний", "Property", "Параметр сейсмического воздействия, фиксирующий максимальную скорость колебаний или смещения."),
    (r"прям\w*\s+динамическ\w*\s+расчет|динамическ\w*\s+расчет", "прямой динамический расчет", "Method", "Метод расчета реакции объекта или массива на динамическое воздействие по временной записи колебаний."),
    (r"\bPhantom\s+MIRO\s+C320\b|высокоскоростн\w*\s+(?:камера|видеокамера)", "Phantom MIRO C320", "Equipment", "Высокоскоростная видеокамера, применяемая для регистрации быстропротекающих процессов."),
    (r"\b(?:Гц|Hz)\b|частотн\w*\s+диапазон", "частотный диапазон", "Property", "Диапазон частот, характеризующий сейсмические или вибрационные сигналы."),
]

_NUMERIC_VALUE_RE = re.compile(
    r"(?P<label>\b(?:PPV|PVS|PPVX|PPVY|PPVZ)\b)?\s*=?\s*(?P<value>\d+(?:[,.]\d+)?)\s*(?P<unit>см/с|мм/с|м/с|Гц|Hz|кПа|МПа|%|°C|кадр(?:ов)?/с|fps)",
    re.IGNORECASE,
)


def is_lightrag_extraction_prompt(
    prompt: str,
    system_prompt: str | None = None,
    *,
    keyword_extraction: bool = False,
) -> bool:
    """Heuristically detect LightRAG entity/relation extraction prompts.

    Query-time and keyword-extraction prompts must not be repaired as entity
    extraction. The extraction prompt normally mentions entity/relationship
    records and the tuple/record/completion delimiters.
    """
    if keyword_extraction:
        return False
    joined = f"{system_prompt or ''}\n{prompt or ''}".lower()
    if "content_keywords" in joined or "tuple delimiter" in joined or "record delimiter" in joined:
        return "entity" in joined and ("relationship" in joined or "relation" in joined)
    if "entity_types" in joined or "entity types" in joined:
        return "relationship" in joined or "relation" in joined
    return False


def _extract_quoted_value_near(text: str, marker: str) -> str | None:
    marker_re = re.escape(marker)
    # Examples seen in LightRAG prompts: tuple_delimiter: <|>, tuple delimiter is "<|>".
    patterns = [
        rf"{marker_re}\s*[:=]\s*[`'\"]?([^`'\"\s]+)[`'\"]?",
        rf"{marker_re}[^\n]*?[`'\"]([^`'\"]+)[`'\"]",
        rf"{marker_re}[^\n]*?([<][^\s]+[>]|#+)",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip().strip("`'\"")
            value = value.rstrip(".,;:")
            if value:
                return value
    return None


def _sanitize_extraction_delimiters(tuple_delim: str, record_delim: str, completion_delim: str) -> tuple[str, str, str]:
    """Normalize delimiter guesses from LightRAG prompts.

    The repair layer is invoked with a prompt that may already include our
    appended hardening section. Loose regex extraction can therefore pick the
    tuple delimiter `<|>` as the record delimiter from the example record line.
    When that happens, repaired records are concatenated without `##`, and
    LightRAG reads the whole output as one giant ENTITY (`found 75/4 fields`).
    This guard keeps the three delimiters distinct and falls back to the
    canonical LightRAG defaults when the guess is suspicious.
    """
    tuple_delim = (tuple_delim or _DEFAULT_TUPLE_DELIMITER).strip().strip("`'\"")
    record_delim = (record_delim or _DEFAULT_RECORD_DELIMITER).strip().strip("`'\"")
    completion_delim = (completion_delim or _DEFAULT_COMPLETION_DELIMITER).strip().strip("`'\"")

    # LightRAG's defaults in this project. Preserve custom values only when
    # they are distinct and structurally plausible.
    if not tuple_delim or tuple_delim in {record_delim, completion_delim}:
        tuple_delim = _DEFAULT_TUPLE_DELIMITER
    if (
        not record_delim
        or record_delim == tuple_delim
        or record_delim == completion_delim
        or "<|" in record_delim
        or "|>" in record_delim
    ):
        record_delim = _DEFAULT_RECORD_DELIMITER
    if (
        not completion_delim
        or completion_delim == tuple_delim
        or completion_delim == record_delim
        or completion_delim in {"<", ">", "|"}
    ):
        completion_delim = _DEFAULT_COMPLETION_DELIMITER
    return tuple_delim, record_delim, completion_delim


def extraction_delimiters_from_prompt(prompt: str, system_prompt: str | None = None) -> tuple[str, str, str]:
    joined = f"{system_prompt or ''}\n{prompt or ''}"
    tuple_delim = (
        _extract_quoted_value_near(joined, "tuple_delimiter")
        or _extract_quoted_value_near(joined, "tuple delimiter")
        or _DEFAULT_TUPLE_DELIMITER
    )
    record_delim = (
        _extract_quoted_value_near(joined, "record_delimiter")
        or _extract_quoted_value_near(joined, "record delimiter")
        or _DEFAULT_RECORD_DELIMITER
    )
    completion_delim = (
        _extract_quoted_value_near(joined, "completion_delimiter")
        or _extract_quoted_value_near(joined, "completion delimiter")
        or _DEFAULT_COMPLETION_DELIMITER
    )
    return _sanitize_extraction_delimiters(tuple_delim, record_delim, completion_delim)


def harden_extraction_prompt(
    prompt: str,
    system_prompt: str | None = None,
) -> tuple[str, str | None]:
    """Append a short anti-Markdown instruction without changing LightRAG's own template."""
    tuple_delim, record_delim, completion_delim = extraction_delimiters_from_prompt(prompt, system_prompt)
    hardening = (
        "\n\nIMPORTANT LightRAG extraction format constraints:\n"
        "- Return ONLY extraction records. Do not use Markdown tables, bullet lists, prose, headings, or explanations.\n"
        f"- Use tuple delimiter exactly: {tuple_delim}\n"
        f"- Use record delimiter exactly: {record_delim}\n"
        f"- End the output with completion delimiter exactly: {completion_delim}\n"
        f'- Entity record format: ("entity"{tuple_delim}"<name>"{tuple_delim}"<type>"{tuple_delim}"<description>"){record_delim}\n'
        f'- Relationship record format: ("relationship"{tuple_delim}"<source>"{tuple_delim}"<target>"{tuple_delim}"<description>"{tuple_delim}"<keywords>"{tuple_delim}"<strength>"){record_delim}\n'
        f'- Content keyword record format: ("content_keywords"{tuple_delim}"<comma-separated keywords>"){record_delim}\n'
        "- Do not add extra fields to entity records. Entity records have exactly 4 fields including the record type.\n"
        "- Relationship records have exactly 6 fields including the record type. If strength is unknown, use 1.0.\n"
        "- Use concise Russian technical descriptions; preserve units and numeric values exactly as in the input text.\n"
        f"- Internal schema version marker for cache busting: {_REPAIR_SCHEMA_VERSION}\n"
    )
    if system_prompt:
        return prompt, f"{system_prompt.rstrip()}\n{hardening}"
    return f"{prompt.rstrip()}\n{hardening}", system_prompt


def _strip_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"```(?:\w+)?", "", cleaned)
    return cleaned.strip()


def _split_pipe_line(line: str) -> list[str]:
    raw = line.strip().strip("|").strip()
    # Do not treat valid LightRAG tuple delimiter records as Markdown tables.
    if "<|>" in raw or not raw or "|" not in raw:
        return []
    parts = [part.strip().strip('"').strip("'") for part in raw.split("|")]
    return [part for part in parts if part]


def _is_separator_line(line: str) -> bool:
    stripped = line.strip().strip("|").strip()
    if not stripped:
        return True
    return bool(re.fullmatch(r"[:\-\s|]+", line.strip()))


def _is_table_header(parts: list[str]) -> bool:
    if not parts:
        return True
    normalized = {re.sub(r"\s+", " ", p.lower().replace("ё", "е")).strip() for p in parts}
    if len(normalized & _TABLE_HEADER_TOKENS) >= 2:
        return True
    return any("----------" in p for p in parts)


def _is_entity_type(value: str) -> bool:
    normalized = re.sub(r"[^a-zа-я0-9_ ]+", "", value.lower().replace("ё", "е")).strip()
    if normalized in _ENTITY_TYPES:
        return True
    # Accept model-specific capitalized English types not listed explicitly.
    return bool(re.fullmatch(r"[a-z_ ]{3,32}", normalized)) and normalized not in {"uses", "used in", "part of", "contains"}


def _clean_field_value(
    value: object,
    tuple_delim: str = _DEFAULT_TUPLE_DELIMITER,
    record_delim: str = _DEFAULT_RECORD_DELIMITER,
    completion_delim: str = _DEFAULT_COMPLETION_DELIMITER,
) -> str:
    """Sanitize one LightRAG record field.

    The previous repair versions still let delimiter tokens leak into entity
    descriptions. LightRAG then split a single ENTITY into many tuple fields
    and logged errors like `found 13/4 fields on ENTITY`. This function removes
    the active delimiters and common delimiter fragments from every field before
    a record is serialized.
    """
    text = str(value or "").strip()
    text = text.replace("\n", " ").replace("\r", " ")
    for token in {
        tuple_delim,
        record_delim,
        completion_delim,
        _DEFAULT_TUPLE_DELIMITER,
        _DEFAULT_RECORD_DELIMITER,
        _DEFAULT_COMPLETION_DELIMITER,
    }:
        if token:
            text = text.replace(token, " ")
    # Remove delimiter fragments that often appear after malformed local-LLM
    # repairs or Markdown/code escaping. Keep ordinary pipe text out of records.
    text = text.replace("<|", " ").replace("|>", " ").replace("|", ";")
    text = text.replace('("entity"', " entity ").replace('("relationship"', " relationship ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip("()[]{} ")
    text = text.replace('"', "'")
    return text


def _quote(value: object, tuple_delim: str = _DEFAULT_TUPLE_DELIMITER) -> str:
    text = _clean_field_value(value, tuple_delim=tuple_delim)
    return f'"{text}"'


def _serialize_record(tag: str, fields: list[object], tuple_delim: str) -> str:
    safe = [tag] + [_clean_field_value(v, tuple_delim=tuple_delim) for v in fields]
    return "(" + tuple_delim.join(_quote(v, tuple_delim) for v in safe) + ")"

def _entity_record(parts: list[str], tuple_delim: str) -> str | None:
    """Build the exact LightRAG entity record shape.

    The LightRAG parser used here expects 4 fields including the record type:
    ("entity"<|>"name"<|>"type"<|>"description")
    Do not append source/chunk id here: it produces errors such as found 5/4
    or, after malformed repair, found 42/4 fields. Source grounding is preserved
    by the document/chunk id passed to LightRAG, not by the extraction record.
    """
    if len(parts) < 2:
        return None
    name = parts[0]
    ent_type = parts[1] if len(parts) >= 2 else "Concept"
    description = " ".join(parts[2:]).strip() if len(parts) >= 3 else f"{name} — сущность, упомянутая в техническом контексте."
    normalized_name = re.sub(r"\s+", " ", str(name).strip().lower())
    normalized_type = re.sub(r"\s+", " ", str(ent_type).strip().lower())
    if (
        not name
        or _is_table_header(parts)
        or normalized_name in {"<", "complete", "<|complete|>"}
        or normalized_type in {"complete", "<|complete|>"}
    ):
        return None
    return _serialize_record("entity", [name, ent_type, description], tuple_delim)


def _relationship_record(parts: list[str], tuple_delim: str) -> str | None:
    if len(parts) < 3:
        return None
    if len(parts) >= 5:
        source, target, description, keywords, strength = parts[:5]
    elif len(parts) == 4:
        # Common malformed form from local LLMs: source | predicate | target | description.
        source, predicate, target, description = parts
        keywords = predicate
        description = f"{source} {predicate} {target}. {description}"
        strength = "1.0"
    else:
        source, target, description = parts[:3]
        keywords = description[:80]
        strength = "1.0"
    if not source or not target or _is_table_header(parts):
        return None
    return _serialize_record("relationship", [source, target, description, keywords, strength], tuple_delim)


def _strip_completion(text: str, completion_delim: str) -> str:
    if not completion_delim:
        return text
    return text.replace(completion_delim, " ").strip()


def _split_tuple_parts(inner: str, tuple_delim: str) -> list[str]:
    return [p.strip().strip('"').strip("'") for p in inner.split(tuple_delim) if p.strip()]


def _normalize_existing_record(parts: list[str], tuple_delim: str) -> str | None:
    if not parts:
        return None
    tag = parts[0].strip().lower().strip('"').strip("'")
    if tag == "entity":
        # Exact LightRAG shape: tag, name, type, description. If extra fields
        # appear because several records were glued together or a table row was
        # inserted into description, fold them into description instead of
        # allowing parser errors like found 42/4.
        if len(parts) < 4:
            if len(parts) >= 3:
                parts.append(f"{parts[1]} — сущность, упомянутая в техническом контексте.")
            else:
                return None
        return _entity_record([parts[1], parts[2], " ".join(parts[3:])], tuple_delim)
    if tag in {"relationship", "relation"}:
        # Exact LightRAG shape: tag, source, target, description, keywords, strength.
        if len(parts) < 4:
            return None
        source = parts[1]
        target = parts[2]
        if len(parts) == 4:
            description = parts[3]
            keywords = description[:80]
            strength = "1.0"
        elif len(parts) == 5:
            description = parts[3]
            keywords = parts[4]
            strength = "1.0"
        else:
            strength = parts[-1] if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", parts[-1].strip()) else "1.0"
            keywords = parts[-2] if len(parts) >= 6 else parts[3][:80]
            description_fields = parts[3:-2] if len(parts) >= 6 else parts[3:]
            description = " ".join(description_fields).strip() or keywords
        return _relationship_record([source, target, description, keywords, strength], tuple_delim)
    if tag == "content_keywords":
        if len(parts) < 2:
            return None
        keywords = " ".join(parts[1:]).strip()
        return _serialize_record("content_keywords", [keywords], tuple_delim)
    return None


def _records_from_parenthesized_lightrag(text: str, tuple_delim: str) -> list[str]:
    records: list[str] = []
    # Non-greedy parenthesized scan. It handles adjacent records without record
    # delimiters: (...)(...) and records separated only by newlines.
    for match in re.finditer(r"\((.*?)\)", text, flags=re.DOTALL):
        inner = match.group(1)
        if tuple_delim not in inner:
            continue
        parts = _split_tuple_parts(inner, tuple_delim)
        rec = _normalize_existing_record(parts, tuple_delim)
        if rec:
            records.append(rec)
    return records


def _repair_parenthesized_records(text: str, tuple_delim: str) -> str:
    """Repair records that already use LightRAG tuple delimiter.

    This legacy helper is kept for compatibility, but the final repair path now
    normalizes records through _records_from_parenthesized_lightrag().
    """
    records = _records_from_parenthesized_lightrag(text, tuple_delim)
    return "##".join(records) if records else text


def _looks_like_relation(parts: list[str]) -> bool:
    if len(parts) < 3:
        return False
    if len(parts) >= 4 and not _is_entity_type(parts[1]):
        return True
    lowered = " ".join(parts[:3]).lower().replace("ё", "е")
    return any(token in lowered for token in ("использ", "исключ", "состоит", "примен", "связан", "входит", "получ", "измер", "used", "contains", "part of"))


def _records_from_pipe_lines(text: str, tuple_delim: str) -> list[str]:
    records: list[str] = []
    for line in text.splitlines():
        if _is_separator_line(line):
            continue
        parts = _split_pipe_line(line)
        if not parts or _is_table_header(parts):
            continue
        record: str | None = None
        first = parts[0].strip().lower()
        if first in {"entity", "сущность"} and len(parts) >= 4:
            record = _entity_record(parts[1:], tuple_delim)
        elif first in {"relationship", "relation", "отношение", "связь"} and len(parts) >= 4:
            record = _relationship_record(parts[1:], tuple_delim)
        elif len(parts) >= 3 and _is_entity_type(parts[1]) and not _looks_like_relation(parts):
            record = _entity_record(parts, tuple_delim)
        elif _looks_like_relation(parts):
            record = _relationship_record(parts, tuple_delim)
        if record:
            records.append(record)
    return records


def _records_from_label_lines(text: str, tuple_delim: str) -> list[str]:
    records: list[str] = []
    entity_patterns = [
        r"(?:ENTITY|Entity|Сущность)\s*[:\-]\s*([^|\n;]+)\s*[|;]\s*([^|\n;]+)\s*[|;]\s*([^\n]+)",
        r"`([^`]+)`\s*@\s*`?([A-Za-zА-Яа-я_ ]+)`?\s*[:\-]\s*([^\n]+)",
    ]
    for pat in entity_patterns:
        for match in re.finditer(pat, text):
            rec = _entity_record([match.group(1), match.group(2), match.group(3)], tuple_delim)
            if rec:
                records.append(rec)

    rel_patterns = [
        r"(?:RELATION|Relationship|Relation|Связь|Отношение)\s*[:\-]\s*([^|\n;]+)\s*[|;~]\s*([^|\n;]+)\s*[|;~]\s*([^|\n;]+)(?:\s*[|;]\s*([^\n]+))?",
    ]
    for pat in rel_patterns:
        for match in re.finditer(pat, text):
            parts = [g for g in match.groups() if g]
            rec = _relationship_record(parts, tuple_delim)
            if rec:
                records.append(rec)
    return records



def _source_text_from_prompt(prompt: str | None, system_prompt: str | None = None) -> str:
    """Extract the real chunk text from a LightRAG extraction prompt.

    LightRAG prompts usually include examples plus a final "Real Data" block.
    A domain fallback must operate only on the actual input chunk, otherwise it
    may extract entities from instructions or examples.
    """
    text = str(prompt or "")
    # Remove our own appended hardening, if it was appended to the user prompt.
    text = re.split(r"IMPORTANT\s+LightRAG\s+extraction\s+format\s+constraints", text, flags=re.IGNORECASE)[0]
    lower = text.lower()

    real_idx = max(lower.rfind("-real data-"), lower.rfind("real data"), lower.rfind("-данные-"))
    if real_idx >= 0:
        text = text[real_idx:]
        lower = text.lower()

    # Prefer the last Text: block before Output:. This matches LightRAG's
    # extraction template and avoids earlier examples.
    text_markers = ["text:", "текст:", "input text:", "input:"]
    starts = [lower.rfind(m) for m in text_markers]
    start = max(starts)
    if start >= 0:
        marker_len = len(text_markers[starts.index(start)]) if start in starts else 5
        candidate = text[start + marker_len :]
    else:
        candidate = text

    # Cut at the first Output marker after the real text.
    candidate = re.split(r"\n\s*(?:output|ответ|result)\s*:\s*\n", candidate, flags=re.IGNORECASE)[0]
    # Remove delimiter instructions/examples if still present.
    candidate = re.sub(r"#+\s*$", "", candidate, flags=re.MULTILINE)
    return candidate.strip()


def _domain_terms_present(source_text: str) -> list[tuple[str, str, str]]:
    entities: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for pattern, name, ent_type, description in _DOMAIN_ENTITY_PATTERNS:
        if re.search(pattern, source_text, flags=re.IGNORECASE):
            key = name.lower().replace("ё", "е")
            if key not in seen:
                entities.append((name, ent_type, description))
                seen.add(key)
    return entities


def _numeric_entities_from_source(source_text: str) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for m in _NUMERIC_VALUE_RE.finditer(source_text):
        label = (m.group("label") or "").upper()
        value = (m.group("value") or "").replace(".", ",")
        unit = m.group("unit") or ""
        if not value or not unit:
            continue
        name = f"{label} = {value} {unit}" if label else f"{value} {unit}"
        key = name.lower().replace("ё", "е")
        if key in seen:
            continue
        seen.add(key)
        desc = f"Числовое значение, извлеченное из исходного фрагмента: {name}."
        out.append((name, "Value", desc))
    return out[:12]


def _domain_records_from_source_text(source_text: str, tuple_delim: str) -> list[str]:
    """Minimal deterministic extraction when the LLM output is unrecoverable.

    This is intentionally conservative: it only emits entities/relations whose
    anchor terms are literally present in the source chunk. It is not a substitute
    for LLM extraction; it prevents LightRAG from producing an empty graph when
    the local LLM returns prose, an empty answer, or malformed tables.
    """
    if not source_text or len(source_text.strip()) < 40:
        return []
    source_norm = source_text.lower().replace("ё", "е")
    entity_specs = _domain_terms_present(source_text)
    numeric_specs = _numeric_entities_from_source(source_text)
    if not entity_specs and not numeric_specs:
        return []

    records: list[str] = []
    for name, ent_type, desc in entity_specs + numeric_specs:
        rec = _entity_record([name, ent_type, desc], tuple_delim)
        if rec:
            records.append(rec)

    names = {name.lower().replace("ё", "е"): name for name, _, _ in entity_specs}

    def has(name: str) -> bool:
        return name.lower().replace("ё", "е") in names

    rel_specs: list[list[str]] = []
    if has("взрывные работы") and has("сейсмограмма"):
        rel_specs.append([
            "взрывные работы",
            "сейсмограмма",
            "Сейсмограммы используются для регистрации и анализа колебаний при взрывных работах.",
            "регистрация колебаний, сейсмовзрывное воздействие",
            "1.0",
        ])
    if has("сейсмограмма") and has("прямой динамический расчет"):
        rel_specs.append([
            "сейсмограмма",
            "прямой динамический расчет",
            "Сейсмограмма используется как входная временная запись для прямого динамического расчета.",
            "используется в, динамический расчет",
            "1.0",
        ])
    if has("велосиграмма") and has("прямой динамический расчет"):
        rel_specs.append([
            "велосиграмма",
            "прямой динамический расчет",
            "Велосиграмма может использоваться как запись скорости колебаний для динамического расчета.",
            "скорость колебаний, динамический расчет",
            "1.0",
        ])
    if has("акселерограмма") and has("прямой динамический расчет"):
        rel_specs.append([
            "акселерограмма",
            "прямой динамический расчет",
            "Акселерограмма может использоваться как запись ускорений для динамического расчета.",
            "ускорение, динамический расчет",
            "1.0",
        ])
    if has("взрывные работы") and has("пиковая скорость колебаний"):
        rel_specs.append([
            "взрывные работы",
            "пиковая скорость колебаний",
            "Пиковая скорость колебаний является измеряемым параметром воздействия взрывных работ.",
            "PPV, PVS, скорость колебаний",
            "1.0",
        ])
    if has("Phantom MIRO C320") and has("взрывные работы"):
        rel_specs.append([
            "Phantom MIRO C320",
            "взрывные работы",
            "Высокоскоростная камера Phantom MIRO C320 упомянута в контексте регистрации процессов при взрывных работах.",
            "оборудование, высокоскоростная съемка",
            "1.0",
        ])
    if has("частотный диапазон") and (has("сейсмограмма") or has("взрывные работы")):
        target = "сейсмограмма" if has("сейсмограмма") else "взрывные работы"
        rel_specs.append([
            "частотный диапазон",
            target,
            "Частотный диапазон указан как характеристика сейсмических или вибрационных сигналов.",
            "частота, сигнал, диапазон",
            "1.0",
        ])

    # Link numeric values to a nearby property/entity only when the property is present.
    if numeric_specs:
        target = "пиковая скорость колебаний" if has("пиковая скорость колебаний") else None
        if not target and re.search(r"\b(?:PPV|PVS|PPVX|PPVY|PPVZ)\b", source_text, flags=re.IGNORECASE):
            target = "пиковая скорость колебаний"
            if not has(target):
                rec = _entity_record([target, "Property", "Параметр, фиксирующий максимальную скорость колебаний или смещения."], tuple_delim)
                if rec:
                    records.append(rec)
        if target:
            for value_name, _, _ in numeric_specs[:8]:
                rel_specs.append([
                    target,
                    value_name,
                    f"Для параметра '{target}' в источнике указано значение {value_name}.",
                    "числовое значение, единица измерения",
                    "1.0",
                ])

    for rel_parts in rel_specs:
        rec = _relationship_record(rel_parts, tuple_delim)
        if rec:
            records.append(rec)

    keywords = []
    for needle, kw in [
        ("сейсм", "сейсмограмма"), ("велосиграм", "велосиграмма"), ("акселерограм", "акселерограмма"),
        ("взрыв", "взрывные работы"), ("бвр", "БВР"), ("динамическ", "динамический расчет"),
        ("ppv", "PPV"), ("pvs", "PVS"), ("частот", "частотный диапазон"),
    ]:
        if needle in source_norm and kw not in keywords:
            keywords.append(kw)
    if keywords:
        records.append(_serialize_record("content_keywords", [", ".join(keywords)], tuple_delim))
    return records

def _dedupe_preserve_order(records: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for rec in records:
        key = re.sub(r"\s+", " ", rec.lower().replace("ё", "е")).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def _ensure_completion(text: str, record_delim: str, completion_delim: str) -> str:
    cleaned = _strip_completion(text.strip(), completion_delim)
    if cleaned and not cleaned.endswith(record_delim):
        cleaned += record_delim
    return cleaned + completion_delim


def _final_validate_records(records: list[str], tuple_delim: str, completion_delim: str) -> list[str]:
    """Ensure every emitted record has the exact parser shape.

    This is the last guardrail before returning text to LightRAG. It prevents
    errors like `found 13/4 fields on ENTITY` by re-parsing our own records and
    serializing them again with delimiter-safe fields.
    """
    valid: list[str] = []
    for rec in records:
        rec = _strip_completion(str(rec or "").strip(), completion_delim)
        if not rec:
            continue
        if rec == completion_delim or rec.strip("()\"\' ") in {"<", "COMPLETE", "< COMPLETE >"}:
            continue
        match = re.fullmatch(r"\((.*)\)", rec, flags=re.DOTALL)
        if not match:
            continue
        parts = _split_tuple_parts(match.group(1), tuple_delim)
        if not parts:
            continue
        tag = parts[0].strip().strip('"').strip("'").lower()
        rebuilt: str | None = None
        if tag == "entity":
            if len(parts) >= 4:
                rebuilt = _entity_record([parts[1], parts[2], " ".join(parts[3:])], tuple_delim)
        elif tag in {"relationship", "relation"}:
            if len(parts) >= 6:
                strength = parts[-1] if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", parts[-1].strip()) else "1.0"
                keywords = parts[-2]
                description = " ".join(parts[3:-2]).strip() or keywords
                rebuilt = _relationship_record([parts[1], parts[2], description, keywords, strength], tuple_delim)
            elif len(parts) >= 4:
                rebuilt = _relationship_record(parts[1:], tuple_delim)
        elif tag == "content_keywords":
            if len(parts) >= 2:
                rebuilt = _serialize_record("content_keywords", [" ".join(parts[1:])], tuple_delim)
        if rebuilt:
            # Verify exact shape after rebuilding. If it still contains active
            # tuple delimiters inside fields, drop it rather than poisoning the
            # whole extraction batch.
            m2 = re.fullmatch(r"\((.*)\)", rebuilt, flags=re.DOTALL)
            rebuilt_parts = _split_tuple_parts(m2.group(1), tuple_delim) if m2 else []
            rebuilt_tag = rebuilt_parts[0].strip().strip('"').strip("'").lower() if rebuilt_parts else ""
            expected = 4 if rebuilt_tag == "entity" else 6 if rebuilt_tag == "relationship" else 2 if rebuilt_tag == "content_keywords" else -1
            if expected > 0 and len(rebuilt_parts) == expected:
                valid.append(rebuilt)
    return _dedupe_preserve_order(valid)


def _format_records(records: list[str], tuple_delim: str, record_delim: str, completion_delim: str) -> str:
    tuple_delim, record_delim, completion_delim = _sanitize_extraction_delimiters(tuple_delim, record_delim, completion_delim)
    cleaned_records = _final_validate_records(records, tuple_delim, completion_delim)
    if not cleaned_records:
        return completion_delim

    # Use one physical line per record. Some LightRAG versions/templates include
    # a trailing newline in the record delimiter during parsing. Emitting
    # `record_delim + "\n"` is compatible with both `##` and `##\n` splitters,
    # and avoids the observed failure where many records are parsed as one giant
    # ENTITY (`found 75/4 fields`).
    record_sep = record_delim + "\n"
    text = "".join(rec + record_sep for rec in cleaned_records) + completion_delim

    # Fail closed if the output would still be parsed by LightRAG as an
    # overlong record. This catches delimiter-regression bugs before LightRAG
    # emits warnings such as `found 75/4 fields on ENTITY`.
    for raw_rec in _strip_completion(text, completion_delim).split(record_delim):
        raw_rec = raw_rec.strip()
        if not raw_rec:
            continue
        match = re.fullmatch(r"\((.*)\)", raw_rec, flags=re.DOTALL)
        if not match:
            return completion_delim
        parts = _split_tuple_parts(match.group(1), tuple_delim)
        if not parts:
            return completion_delim
        tag = parts[0].strip().strip('"').strip("'").lower()
        expected = 4 if tag == "entity" else 6 if tag == "relationship" else 2 if tag == "content_keywords" else -1
        if expected < 0 or len(parts) != expected:
            return completion_delim
    return text


def _dump_repair_io(raw: str, repaired: str, prompt: str | None = None) -> None:
    """Optional diagnostics for hard cases.

    Enable with LIGHTRAG_EXTRACTION_REPAIR_DUMP_DIR=path. The dump is disabled
    by default and never runs unless the environment variable is set.
    """
    dump_dir = os.getenv("LIGHTRAG_EXTRACTION_REPAIR_DUMP_DIR", "").strip()
    if not dump_dir:
        return
    try:
        root = Path(dump_dir)
        root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1((raw or "").encode("utf-8", errors="ignore")).hexdigest()[:16]
        (root / f"{digest}.raw.txt").write_text(raw or "", encoding="utf-8")
        (root / f"{digest}.repaired.txt").write_text(repaired or "", encoding="utf-8")
        if prompt:
            (root / f"{digest}.prompt.txt").write_text(prompt, encoding="utf-8")
    except Exception:
        # Repair must never fail ingestion due to diagnostics.
        return


def repair_lightrag_extraction_output(
    text: str,
    prompt: str | None = None,
    system_prompt: str | None = None,
) -> str:
    """Best-effort repair for local LLM extraction outputs.

    Targets three observed failure modes:
    1. Markdown/pipe tables instead of LightRAG records.
    2. Parenthesized records with missing record delimiters, e.g. (...)(...).
    3. Completion delimiter glued into a record, which produced errors such as
       ENTITY `"<"` @ `"COMPLETE"`.
    """
    if not text:
        return text
    tuple_delim, record_delim, completion_delim = extraction_delimiters_from_prompt(prompt or "", system_prompt)
    cleaned = _strip_thinking(text)
    cleaned = _strip_completion(cleaned, completion_delim)

    records: list[str] = []

    # First normalize any records that already use LightRAG tuple delimiters.
    # This fixes missing record delimiters and overlong entity records before
    # they reach LightRAG's parser.
    records.extend(_records_from_parenthesized_lightrag(cleaned, tuple_delim))

    # Then convert local-LLM Markdown/pipe outputs into exact LightRAG records.
    records.extend(_records_from_pipe_lines(cleaned, tuple_delim))
    records.extend(_records_from_label_lines(cleaned, tuple_delim))
    records = _dedupe_preserve_order(records)

    if not records:
        source_text = _source_text_from_prompt(prompt, system_prompt)
        records.extend(_domain_records_from_source_text(source_text, tuple_delim))

    if records:
        if not any("content_keywords" in r.lower() for r in records):
            keyword_record = _serialize_record("content_keywords", [
                "горно-металлургические процессы, материал, процесс, оборудование, свойство, числовые параметры, условия эксперимента",
            ], tuple_delim)
            records.append(keyword_record)
        repaired = _format_records(records, tuple_delim, record_delim, completion_delim)
        _dump_repair_io(text, repaired, prompt)
        return repaired

    # No recoverable records and no grounded domain anchors in the source text:
    # return only the completion delimiter. Returning prose plus ## tends to
    # create parser warnings and still gives 0 records.
    _dump_repair_io(text, completion_delim, prompt)
    return completion_delim
