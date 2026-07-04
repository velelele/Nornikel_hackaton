from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from backend.knowledge_store import KnowledgeStore, jsonl_read, jsonl_write, utc_now_iso
from backend.domain.chunker import clean_extracted_text

_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё%°./+-]{2,}", re.UNICODE)
_NUMERIC_RE = re.compile(
    # Linear-time numeric-with-unit detector. The previous pattern allowed an
    # empty numeric prefix and a repeated optional-space group, which caused
    # catastrophic backtracking on ordinary prose like "from 1956 to 2010 ... 40%".
    r"(?<![\w.,+-])[+-]?(?:(?:\d{1,3}(?:[\s\u00a0]\d{3})+)|\d+)(?:[\.,]\d+)?\s*"
    r"(?:%|°\s*C|degC|мг/л|мг/дм3|мг/дм³|г/л|кг/т|см/с|мм/с|м/с|Гц|Hz|A/m2|А/м2|ppm|ppb)(?!\w)",
    re.IGNORECASE | re.UNICODE,
)
_TABLE_LIKE_RE = re.compile(r"(?:\|.*\||\t| {2,}\S+ {2,}\S+|(?:табл|table)\.?\s*\d*)", re.IGNORECASE | re.UNICODE)
_SECTION_RE = re.compile(r"\b(?:abstract|аннотац|introduction|введение|results|результат|discussion|обсуждение|conclusion|вывод|табл|table)\b", re.IGNORECASE | re.UNICODE)


_DOCTYPE_EXT_RE = re.compile(r"\.(pptx?|pdf|docx?|xlsx?|html?|md|txt)(?:$|[#?\s])", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.%-]+@[\w.-]+\.[A-Za-zА-Яа-я]{2,}\b", re.UNICODE)
_PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)", re.UNICODE)
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_SLIDE_BOILERPLATE_RE = re.compile(
    r"\b(?:спасибо\s+за\s+внимание|thank\s+you|контакт(?:ы|ная\s+информация)|"
    r"содержание|agenda|план\s+доклада|актуальность\s*$|цель\s+работы\s*$|"
    r"структура\s+доклада|вопросы\?)\b",
    re.IGNORECASE | re.UNICODE,
)
_ARTICLE_NOISE_RE = re.compile(
    r"\b(?:список\s+литературы|литература|references|bibliography|acknowledg(?:e)?ments|"
    r"благодарности|copyright|all\s+rights\s+reserved|doi:|удк|issn)\b",
    re.IGNORECASE | re.UNICODE,
)
_NORMATIVE_RE = re.compile(r"\b(?:должен|должна|должны|следует|необходимо|требовани[ея]|норма|ГОСТ|GOST|ISO|shall|must|requirement)\b", re.IGNORECASE | re.UNICODE)
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*[-:]{2,}\s*(?:\|\s*[-:]{2,}\s*)+\|?\s*$", re.MULTILINE)
_SENTENCE_END_RE = re.compile(r"[.!?…]\s+")

PROCESS_TERMS = {
    "electrowinning", "electrorefining", "leaching", "smelting", "flotation", "roasting",
    "solvent extraction", "precipitation", "neutralization", "refining", "газоочист", "выщелач",
    "электроэкстрак", "электрорафинир", "плавк", "флотац", "обжиг", "экстракц", "осажд",
    "буровзрыв", "взрывн", "blasting", "blast", "сейсмограмм", "велосиграмм",
    "динамический расчет", "dynamic analysis",
}
MATERIAL_TERMS = {
    "nickel", "cobalt", "copper", "catholyte", "anolyte", "matte", "slag", "pgm", "so2",
    "никель", "кобальт", "медь", "католит", "анолит", "штейн", "шлак", "мпг", "сернист",
    "hydroxide", "гидроксид", "solution", "раствор", "ore", "руда", "concentrate", "концентрат",
}
PROPERTY_TERMS = {
    "temperature", "concentration", "flow", "current density", "ph", "pressure", "recovery", "distribution",
    "температур", "концентрац", "скорост", "расход", "плотность тока", "давлен", "извлечен", "распределен",
    "частот", "frequency", "ppv", "pvs", "скорость смещения", "скорость колеб",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("ё", "е")).strip()


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower().replace("ё", "е") for m in _WORD_RE.finditer(text or "")]


def _contains_any(text: str, terms: Iterable[str]) -> int:
    low = _norm(text)
    return sum(1 for term in terms if term in low)


def _text_fingerprint(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip().lower())[:1600]
    return hashlib.sha1(cleaned.encode("utf-8", errors="ignore")).hexdigest()


def _iter_jsonl_dicts(path: Path) -> Iterator[dict[str, Any]]:
    """Stream JSONL rows without loading large theme files into memory."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _candidate_sort_key(row: dict[str, Any]) -> tuple[float, int, int]:
    score, chars = _rough_candidate_score(row)
    return (float(score), int(chars), len(str(row.get("text") or row.get("lightrag_text") or "")))


def _load_candidate_chunks(
    store: KnowledgeStore,
    *,
    max_candidate_chunks_per_theme: int | None,
) -> tuple[list[dict[str, Any]], int, int]:
    """Load only the selected candidate window for compressed KG selection.

    Large journal themes can contain hundreds of MB of chunk JSONL. Stage 2 only
    needs the top-N rough candidates when max_candidate_chunks_per_theme is set,
    so keeping all chunks in memory is unnecessary and makes the UI look hung.
    """
    cap = int(max_candidate_chunks_per_theme or 0)
    if cap <= 0:
        chunks = jsonl_read(store.chunks_path)
        return chunks, 0, len(chunks)

    import heapq

    heap: list[tuple[tuple[float, int, int], int, dict[str, Any]]] = []
    total = 0
    for total, row in enumerate(_iter_jsonl_dicts(store.chunks_path), start=1):
        key = _candidate_sort_key(row)
        item = (key, total, row)
        if len(heap) < cap:
            heapq.heappush(heap, item)
        elif key > heap[0][0]:
            heapq.heapreplace(heap, item)

    kept = [row for _key, _idx, row in heap]
    kept.sort(key=lambda row: (str(row.get("source_name") or ""), str(row.get("citation_name") or ""), str(row.get("chunk_id") or "")))
    return kept, max(0, total - len(kept)), total


def _numeric_chunk_ids_for_candidates(
    store: KnowledgeStore,
    candidate_chunk_ids: set[str] | None,
) -> set[str]:
    ids: set[str] = set()
    for row in _iter_jsonl_dicts(store.numeric_facts_path):
        chunk_id = str(row.get("chunk_id") or "")
        if not chunk_id:
            continue
        if candidate_chunk_ids is None or chunk_id in candidate_chunk_ids:
            ids.add(chunk_id)
    return ids


def _source_key(row: dict[str, Any]) -> str:
    return str(row.get("source_id") or row.get("source_hash") or row.get("source_name") or "unknown")


def _rough_candidate_score(row: dict[str, Any]) -> tuple[float, int]:
    text = str(row.get("text") or row.get("lightrag_text") or "")
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    domain_tags = row.get("domain_tags") or metadata.get("domain_tags") or []
    if isinstance(domain_tags, str):
        domain_tags = [x.strip() for x in domain_tags.split(",") if x.strip()]
    score = 0.0
    if _NUMERIC_RE.search(text):
        score += 0.42
    score += min(0.22, 0.025 * len(domain_tags))
    score += min(0.24, 0.035 * (_contains_any(text, PROCESS_TERMS) + _contains_any(text, PROPERTY_TERMS)))
    score += min(0.08, len(text) / 12000.0)
    if str(metadata.get("section_type") or "") in {"methods", "results", "requirements", "table"}:
        score += 0.10
    return (score, int(row.get("chars") or len(text)))


def _limit_candidate_chunks(chunks: list[dict[str, Any]], *, max_candidate_chunks_per_theme: int | None) -> tuple[list[dict[str, Any]], int]:
    # Kept for compatibility with older callers. New Stage 2 selection uses
    # _load_candidate_chunks() to avoid loading large themes twice.
    if not max_candidate_chunks_per_theme or int(max_candidate_chunks_per_theme) <= 0:
        return chunks, 0
    cap = int(max_candidate_chunks_per_theme)
    if len(chunks) <= cap:
        return chunks, 0
    kept = sorted(chunks, key=_rough_candidate_score, reverse=True)[:cap]
    # Stable source-local order after rough pruning.
    kept.sort(key=lambda row: (str(row.get("source_name") or ""), str(row.get("citation_name") or ""), str(row.get("chunk_id") or "")))
    return kept, len(chunks) - len(kept)


def _source_char_count(rows: list[dict[str, Any]]) -> int:
    total = 0
    for row in rows:
        total += int(row.get("chars") or len(str(row.get("text") or row.get("lightrag_text") or "")))
    return total


def _per_document_limit_for_source(
    rows: list[dict[str, Any]],
    *,
    base_limit: int,
    short_document_max_chunks: int = 3,
    long_document_max_chunks: int = 6,
    short_document_max_source_chunks: int = 5,
    short_document_max_chars: int = 12000,
    long_document_min_source_chunks: int = 24,
    long_document_min_chars: int = 50000,
) -> int:
    source_chunks = len(rows)
    source_chars = _source_char_count(rows)
    base = max(1, int(base_limit or 1))
    if source_chunks <= int(short_document_max_source_chunks) or source_chars <= int(short_document_max_chars):
        return max(1, min(base, int(short_document_max_chunks)))
    if source_chunks >= int(long_document_min_source_chunks) or source_chars >= int(long_document_min_chars):
        return max(base, int(long_document_max_chunks))
    return base


def infer_document_type(row: dict[str, Any]) -> str:
    """Best-effort document type for compressed KG selection.

    Stage 1 stores mostly flat chunks, so we infer type from source/citation name
    and a few textual markers. The goal is not perfect classification; it is to
    apply different noise penalties for slide decks, papers, standards and
    tabular files before LightRAG extraction.
    """
    source = str(row.get("source_name") or row.get("citation_name") or row.get("source_id") or "")
    text = str(row.get("text") or row.get("lightrag_text") or "")
    hay = _norm(source + "\n" + text[:1200])
    m = _DOCTYPE_EXT_RE.search(source)
    ext = (m.group(1).lower() if m else "").lstrip(".")
    if ext in {"ppt", "pptx"} or "презентац" in hay or "slide" in hay:
        return "presentation"
    if ext in {"xls", "xlsx"}:
        return "spreadsheet"
    if "гост" in hay or "gost" in hay or " iso " in f" {hay} " or "стандарт" in hay:
        return "standard"
    if ext == "pdf" and any(x in hay for x in ("abstract", "аннотац", "references", "список литературы", "doi:")):
        return "article"
    if ext == "pdf":
        return "pdf_report"
    if ext in {"doc", "docx"}:
        return "report"
    return "generic"


def _line_stats(text: str) -> dict[str, float]:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return {"lines": 0.0, "short_line_ratio": 1.0, "table_separator_ratio": 0.0, "uppercase_ratio": 0.0}
    short_lines = sum(1 for line in lines if len(line) <= 38)
    table_separators = sum(1 for line in lines if _TABLE_SEPARATOR_RE.search(line))
    uppercase_lines = 0
    for line in lines:
        letters = [ch for ch in line if ch.isalpha()]
        if letters and sum(1 for ch in letters if ch.isupper()) / max(1, len(letters)) > 0.72:
            uppercase_lines += 1
    return {
        "lines": float(len(lines)),
        "short_line_ratio": short_lines / max(1, len(lines)),
        "table_separator_ratio": table_separators / max(1, len(lines)),
        "uppercase_ratio": uppercase_lines / max(1, len(lines)),
    }


def score_chunk_informativeness(row: dict[str, Any], *, scored: dict[str, Any] | None = None) -> dict[str, Any]:
    """Score whether a chunk is worth sending to LightRAG extraction.

    The score is document-type aware. It suppresses presentation title/contact
    slides, article reference sections, pure markdown table separators and other
    fragments that are expensive for LLM extraction but rarely answer technical
    questions. It boosts numeric/domain/process fragments, normative requirements
    and article result/method sections.
    """
    text = str(row.get("text") or row.get("lightrag_text") or "")
    cleaned = clean_extracted_text(text)
    normalized = _norm(cleaned)
    doc_type = infer_document_type(row)
    tokens = _tokenize(cleaned)
    token_count = len(tokens)
    unique_ratio = len(set(tokens)) / max(1, token_count)
    letters = sum(1 for ch in cleaned if ch.isalpha())
    alpha_ratio = letters / max(1, len(cleaned))
    numeric_hits = len(_NUMERIC_RE.findall(cleaned))
    domain_hits = (
        _contains_any(cleaned, PROCESS_TERMS)
        + _contains_any(cleaned, MATERIAL_TERMS)
        + _contains_any(cleaned, PROPERTY_TERMS)
    )
    line_stats = _line_stats(cleaned)
    sentence_like = len(_SENTENCE_END_RE.findall(cleaned))
    table_like = bool(_TABLE_LIKE_RE.search(cleaned))
    table_separator_heavy = line_stats["table_separator_ratio"] > 0.2
    contact_heavy = bool(_EMAIL_RE.search(cleaned) or _PHONE_RE.search(cleaned) or _URL_RE.search(cleaned))
    title_like = token_count <= 16 and numeric_hits == 0 and domain_hits <= 1 and line_stats["uppercase_ratio"] > 0.35
    boilerplate = False

    score = 0.10
    reasons: list[str] = [f"doc_type:{doc_type}"]

    if token_count >= 45:
        score += 0.18
        reasons.append("enough_words")
    elif token_count < 12:
        score -= 0.18
        reasons.append("too_short")

    if alpha_ratio >= 0.45:
        score += 0.08
    else:
        score -= 0.12
        reasons.append("low_alpha_ratio")

    if unique_ratio < 0.28 and token_count > 25:
        score -= 0.10
        reasons.append("repetitive")

    if numeric_hits:
        score += min(0.28, 0.07 * numeric_hits)
        reasons.append(f"numeric_hits:{numeric_hits}")
    if domain_hits:
        score += min(0.28, 0.06 * domain_hits)
        reasons.append(f"domain_hits:{domain_hits}")
    if sentence_like >= 2:
        score += 0.08
        reasons.append("sentence_context")
    if table_like and numeric_hits:
        score += 0.08
        reasons.append("numeric_table")
    if table_separator_heavy and not numeric_hits:
        score -= 0.22
        reasons.append("table_separator_noise")

    if doc_type == "presentation":
        if _SLIDE_BOILERPLATE_RE.search(normalized) or contact_heavy or title_like:
            score -= 0.25
            boilerplate = True
            reasons.append("presentation_boilerplate")
        if numeric_hits or domain_hits >= 2:
            score += 0.10
            reasons.append("informative_slide")
        if line_stats["short_line_ratio"] > 0.75 and not numeric_hits:
            score -= 0.10
            reasons.append("bullet_title_slide")
    elif doc_type == "article":
        if _ARTICLE_NOISE_RE.search(normalized) and not numeric_hits:
            score -= 0.24
            boilerplate = True
            reasons.append("article_tail_noise")
        if _SECTION_RE.search(normalized) and not _ARTICLE_NOISE_RE.search(normalized):
            score += 0.10
            reasons.append("article_section")
    elif doc_type == "standard":
        if _NORMATIVE_RE.search(cleaned):
            score += 0.18
            reasons.append("normative_text")
        if contact_heavy and not _NORMATIVE_RE.search(cleaned):
            score -= 0.16
            reasons.append("standard_cover_noise")
    elif doc_type == "spreadsheet":
        if table_like and numeric_hits:
            score += 0.18
            reasons.append("spreadsheet_numeric_rows")
        elif table_like:
            score -= 0.08
            reasons.append("spreadsheet_header_only")

    # KG score remains the main signal; quality decides whether the chunk is
    # worth paying LightRAG extraction cost.
    kg_score = float((scored or {}).get("score") or 0.0)
    if kg_score >= 0.30:
        score += 0.10
        reasons.append("high_kg_score")

    score = max(0.0, min(1.0, score))
    is_informative = bool(score >= 0.20 and not (boilerplate and not (numeric_hits or domain_hits >= 2)))
    return {
        "doc_type": doc_type,
        "chunk_quality": round(float(score), 4),
        "quality_reasons": reasons,
        "is_informative": is_informative,
        "numeric_hits": numeric_hits,
        "domain_hits": domain_hits,
    }


def score_chunk_for_kg(chunk: dict[str, Any], *, numeric_chunk_ids: set[str] | None = None) -> dict[str, Any]:
    """Return deterministic KG usefulness score for a chunk.

    The score is intentionally simple and explainable. It favours chunks likely to
    contain extractable metallurgical facts: numbers with units, ontology tags,
    process/material/property terms, tables and abstract/result/conclusion blocks.
    """
    numeric_chunk_ids = numeric_chunk_ids or set()
    text = str(chunk.get("text") or chunk.get("lightrag_text") or "")
    metadata = chunk.get("metadata") or {}
    domain_tags = chunk.get("domain_tags") or metadata.get("domain_tags") or []
    if isinstance(domain_tags, str):
        domain_tags = [x.strip() for x in domain_tags.split(",") if x.strip()]

    numeric_presence = 1.0 if chunk.get("chunk_id") in numeric_chunk_ids or _NUMERIC_RE.search(text) else 0.0
    ontology_terms_count = min(len(domain_tags), 12)
    process_count = min(_contains_any(text, PROCESS_TERMS), 8)
    material_count = min(_contains_any(text, MATERIAL_TERMS), 8)
    property_count = min(_contains_any(text, PROPERTY_TERMS), 8)
    table_like = 1.0 if _TABLE_LIKE_RE.search(text) else 0.0
    section_boost = 1.0 if _SECTION_RE.search(str(chunk.get("citation_name") or "") + "\n" + text[:500]) else 0.0

    score = (
        0.35 * numeric_presence
        + 0.20 * min(ontology_terms_count / 6.0, 1.0)
        + 0.15 * min(process_count / 3.0, 1.0)
        + 0.10 * min(material_count / 3.0, 1.0)
        + 0.10 * min(property_count / 3.0, 1.0)
        + 0.07 * table_like
        + 0.03 * section_boost
    )
    reasons: list[str] = []
    if numeric_presence:
        reasons.append("numeric_fact_or_unit")
    if ontology_terms_count:
        reasons.append(f"ontology_terms:{ontology_terms_count}")
    if process_count:
        reasons.append(f"process_terms:{process_count}")
    if material_count:
        reasons.append(f"material_terms:{material_count}")
    if property_count:
        reasons.append(f"property_terms:{property_count}")
    if table_like:
        reasons.append("table_like")
    if section_boost:
        reasons.append("section_boost")
    if not reasons:
        reasons.append("weak_context")
    return {"score": round(float(score), 4), "reasons": reasons}


def select_graph_chunks(
    store: KnowledgeStore,
    *,
    max_chunks_per_document: int = 8,
    min_kg_score: float = 0.15,
    max_chunks_per_theme: int | None = None,
    max_candidate_chunks_per_theme: int | None = None,
    deduplicate: bool = True,
    min_chunk_quality: float = 0.20,
    doc_type_aware: bool = True,
    dynamic_document_limits: bool = True,
    short_document_max_chunks: int = 3,
    long_document_max_chunks: int = 6,
) -> dict[str, Any]:
    chunks, candidate_pruned, total_chunks = _load_candidate_chunks(
        store,
        max_candidate_chunks_per_theme=max_candidate_chunks_per_theme,
    )
    candidate_chunk_ids = {str(row.get("chunk_id") or "") for row in chunks if row.get("chunk_id")}
    numeric_chunk_ids = _numeric_chunk_ids_for_candidates(
        store,
        candidate_chunk_ids if max_candidate_chunks_per_theme and int(max_candidate_chunks_per_theme) > 0 else None,
    )

    scored_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected_by_type: Counter[str] = Counter()
    selected_doc_types: Counter[str] = Counter()
    for row in chunks:
        scored = score_chunk_for_kg(row, numeric_chunk_ids=numeric_chunk_ids)
        quality = score_chunk_informativeness(row, scored=scored) if doc_type_aware else {
            "doc_type": infer_document_type(row),
            "chunk_quality": 1.0,
            "quality_reasons": ["doc_type_aware_disabled"],
            "is_informative": True,
        }
        raw_kg_score = float(scored["score"])
        chunk_quality = float(quality.get("chunk_quality") or 0.0)
        combined_score = round(0.72 * raw_kg_score + 0.28 * chunk_quality, 4)
        entry = {
            **row,
            "kg_score_raw": scored["score"],
            "kg_score": combined_score,
            "kg_reasons": scored["reasons"],
            "doc_type": quality.get("doc_type"),
            "chunk_quality": chunk_quality,
            "quality_reasons": quality.get("quality_reasons") or [],
            "is_informative_for_kg": bool(quality.get("is_informative")),
        }
        scored_by_source[_source_key(row)].append(entry)

    selected: list[dict[str, Any]] = []
    seen_fingerprints: set[str] = set()
    base_per_doc_limit = max(1, int(max_chunks_per_document))
    per_doc_limits_used: Counter[str] = Counter()
    for source_id, rows in scored_by_source.items():
        per_doc_limit = (
            _per_document_limit_for_source(
                rows,
                base_limit=base_per_doc_limit,
                short_document_max_chunks=short_document_max_chunks,
                long_document_max_chunks=long_document_max_chunks,
            )
            if dynamic_document_limits else base_per_doc_limit
        )
        per_doc_limits_used[str(per_doc_limit)] += 1
        rows.sort(
            key=lambda item: (
                bool(item.get("is_informative_for_kg")),
                float(item.get("kg_score") or 0.0),
                float(item.get("chunk_quality") or 0.0),
                int(item.get("chars") or 0),
            ),
            reverse=True,
        )
        taken = 0
        for row in rows:
            score = float(row.get("kg_score") or 0.0)
            quality = float(row.get("chunk_quality") or 0.0)
            doc_type = str(row.get("doc_type") or "generic")
            has_hard_signal = (
                "numeric_fact_or_unit" in set(row.get("kg_reasons") or [])
                or quality >= max(float(min_chunk_quality), 0.34)
            )
            if not row.get("is_informative_for_kg") and not has_hard_signal:
                rejected_by_type[doc_type] += 1
                continue
            if quality < float(min_chunk_quality) and not has_hard_signal:
                rejected_by_type[doc_type] += 1
                continue
            if taken > 0 and score < float(min_kg_score):
                continue
            fp = _text_fingerprint(str(row.get("text") or row.get("lightrag_text") or ""))
            if deduplicate and fp in seen_fingerprints:
                continue
            seen_fingerprints.add(fp)
            selected.append(row)
            selected_doc_types[doc_type] += 1
            taken += 1
            if taken >= per_doc_limit:
                break

    selected.sort(key=lambda item: (str(item.get("source_name") or ""), -float(item.get("kg_score") or 0.0)))
    if max_chunks_per_theme is not None and max_chunks_per_theme > 0:
        selected = sorted(selected, key=lambda item: float(item.get("kg_score") or 0.0), reverse=True)[: int(max_chunks_per_theme)]
        selected.sort(key=lambda item: (str(item.get("source_name") or ""), -float(item.get("kg_score") or 0.0)))

    # Recompute after the theme-level cap; the first pass counted all per-source
    # selected rows before max_chunks_per_theme truncation.
    selected_doc_types = Counter(str(row.get("doc_type") or "generic") for row in selected)

    out_path = store.root / "selected_graph_chunks.jsonl"
    jsonl_write(out_path, selected)
    return {
        "selected_chunks": len(selected),
        "total_chunks": total_chunks,
        "candidate_chunks": len(chunks),
        "candidate_chunks_pruned": candidate_pruned,
        "selection_ratio": round(len(selected) / total_chunks, 4) if total_chunks else 0.0,
        "max_chunks_per_document": base_per_doc_limit,
        "dynamic_document_limits": bool(dynamic_document_limits),
        "per_doc_limits_used": dict(per_doc_limits_used),
        "min_kg_score": float(min_kg_score),
        "min_chunk_quality": float(min_chunk_quality),
        "doc_type_aware": bool(doc_type_aware),
        "max_chunks_per_theme": max_chunks_per_theme,
        "max_candidate_chunks_per_theme": max_candidate_chunks_per_theme,
        "selected_doc_types": dict(selected_doc_types),
        "rejected_by_doc_type": dict(rejected_by_type),
        "path": str(out_path),
        "rows": selected,
    }


def _lightrag_items_from_selected_rows_chunk(rows: Iterable[dict[str, Any]]) -> list[tuple[str, str]]:
    """Previous compressed-LightRAG behaviour: one selected chunk = one runtime document.

    This mode is intentionally kept as the default. Empirically it gives
    LightRAG better retrieval granularity for PPT/PDF slide decks than grouping
    several selected fragments into one pseudo-document. The runtime citation is
    still made unique with #kgchunk:<suffix> to avoid duplicate filename/doc_status
    collisions inside LightRAG.
    """
    items: list[tuple[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        text = str(row.get("lightrag_text") or row.get("text") or "").strip()
        citation_base = str(row.get("citation_name") or row.get("source_name") or row.get("chunk_id") or "").strip()
        chunk_id = str(row.get("chunk_id") or "").replace(":", "_")
        suffix = chunk_id[-16:] if chunk_id else f"selected_{idx:06d}"
        citation = f"{citation_base}#kgchunk:{suffix}" if citation_base else f"kgchunk:{suffix}"
        if text and citation:
            items.append((clean_extracted_text(text), citation))
    return items


def _chunk_sort_key(row: dict[str, Any], fallback_idx: int) -> tuple[str, int, str]:
    """Best-effort source-local order for aggregated compressed LightRAG docs."""
    meta = row.get("metadata") or {}
    candidates = [
        row.get("chunk_index"),
        row.get("chunk_no"),
        row.get("page"),
        meta.get("chunk_index") if isinstance(meta, dict) else None,
        meta.get("page") if isinstance(meta, dict) else None,
    ]
    numeric = None
    for value in candidates:
        if value is None:
            continue
        text = str(value)
        if text.isdigit():
            numeric = int(text)
            break
    if numeric is None:
        m = re.search(r"(?:chunk|page|slide|p)[_: -]?(\d+)", str(row.get("chunk_id") or row.get("citation_name") or ""), re.I)
        numeric = int(m.group(1)) if m else fallback_idx
    return (str(row.get("source_name") or row.get("source_id") or ""), numeric, str(row.get("chunk_id") or ""))


def _runtime_source_name(rows: list[dict[str, Any]], group_idx: int, part_idx: int, chunk_ids: list[str]) -> str:
    first = rows[0] if rows else {}
    base = str(first.get("source_name") or first.get("citation_name") or first.get("source_id") or f"source_{group_idx:04d}").strip()
    base = re.sub(r"\s+", " ", base)[:180] or f"source_{group_idx:04d}"
    digest_src = "|".join(chunk_ids) or f"{group_idx}:{part_idx}:{base}"
    digest = hashlib.sha1(digest_src.encode("utf-8", errors="ignore")).hexdigest()[:12]
    suffix = f"kgcompressed:{digest}" if part_idx == 1 else f"kgcompressed:{digest}:part{part_idx}"
    return f"{base}#{suffix}"


def _lightrag_items_from_selected_rows_grouped(
    rows: Iterable[dict[str, Any]],
    *,
    max_chars_per_runtime_doc: int = 14000,
) -> list[tuple[str, str]]:
    """Optional source-level pseudo-document mode for compressed LightRAG."""
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, row in enumerate(rows, start=1):
        text = str(row.get("lightrag_text") or row.get("text") or "").strip()
        if not text:
            continue
        key = _source_key(row)
        grouped[key].append((idx, row))

    items: list[tuple[str, str]] = []
    max_chars = max(4000, int(max_chars_per_runtime_doc or 14000))
    for group_idx, (_source_id, indexed_rows) in enumerate(sorted(grouped.items(), key=lambda kv: str(kv[0])), start=1):
        ordered = [row for idx, row in sorted(indexed_rows, key=lambda pair: _chunk_sort_key(pair[1], pair[0]))]
        part_rows: list[dict[str, Any]] = []
        part_texts: list[str] = []
        part_chars = 0
        part_idx = 1

        def flush_part() -> None:
            nonlocal part_rows, part_texts, part_chars, part_idx
            if not part_texts:
                return
            chunk_ids = [str(r.get("chunk_id") or r.get("citation_name") or "") for r in part_rows]
            citation = _runtime_source_name(part_rows, group_idx, part_idx, chunk_ids)
            body = "\n\n".join(part_texts).strip()
            if body:
                items.append((body, citation))
            part_rows = []
            part_texts = []
            part_chars = 0
            part_idx += 1

        for local_idx, row in enumerate(ordered, start=1):
            citation = str(row.get("citation_name") or row.get("chunk_id") or f"fragment_{local_idx}").strip()
            text = clean_extracted_text(str(row.get("lightrag_text") or row.get("text") or ""))
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            fragment = f"[Фрагмент {local_idx}: {citation}]\n{text}"
            if part_texts and part_chars + len(fragment) > max_chars:
                flush_part()
            part_rows.append(row)
            part_texts.append(fragment)
            part_chars += len(fragment)
        flush_part()
    return items


def lightrag_items_from_selected_rows(
    rows: Iterable[dict[str, Any]],
    *,
    mode: str = "chunk",
    max_chars_per_runtime_doc: int = 14000,
) -> list[tuple[str, str]]:
    """Convert compressed KG rows to LightRAG runtime input.

    mode="chunk" restores the previously useful compressed-LightRAG behaviour:
    every selected chunk is a separate retrievable document with unique
    #kgchunk identity. mode="grouped" remains available for overnight tests, but
    it can reduce retrieval granularity and was observed to produce empty graphs
    on slide/table-heavy documents.
    """
    normalized_mode = str(mode or "chunk").strip().lower()
    if normalized_mode in {"group", "grouped", "source", "source_grouped"}:
        return _lightrag_items_from_selected_rows_grouped(
            rows,
            max_chars_per_runtime_doc=max_chars_per_runtime_doc,
        )
    return _lightrag_items_from_selected_rows_chunk(rows)


def build_cheap_kg(store: KnowledgeStore, *, top_n: int = 50) -> dict[str, Any]:
    """Build a deterministic lightweight KG summary from Stage-1 JSONL files.

    This is not a replacement for full LightRAG KG. It is a guaranteed fallback
    graph for routing, quality control and Stage-1/Stage-2 search readiness.
    """
    sources = jsonl_read(store.sources_path)
    chunks = jsonl_read(store.chunks_path)
    entities = jsonl_read(store.entities_path)
    triples = jsonl_read(store.triples_path)
    facts = jsonl_read(store.numeric_facts_path)

    node_counter: Counter[str] = Counter()
    edge_counter: Counter[tuple[str, str, str]] = Counter()
    cooccur_counter: Counter[tuple[str, str]] = Counter()
    chunk_entities: dict[str, set[str]] = defaultdict(set)

    entity_labels = {str(row.get("entity_id")): str(row.get("label") or row.get("term_id") or row.get("entity_id")) for row in entities}

    for triple in triples:
        s = str(triple.get("subject_id") or "")
        p = str(triple.get("predicate") or "")
        o = str(triple.get("object_id") or "")
        if not s or not p or not o:
            continue
        node_counter[s] += 1
        node_counter[o] += 1
        edge_counter[(s, p, o)] += 1
        chunk_id = str(triple.get("chunk_id") or "")
        if chunk_id and (s.startswith("onto:") or o.startswith("onto:")):
            if s.startswith("onto:"):
                chunk_entities[chunk_id].add(s)
            if o.startswith("onto:"):
                chunk_entities[chunk_id].add(o)

    for entity_set in chunk_entities.values():
        ordered = sorted(entity_set)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                cooccur_counter[(ordered[i], ordered[j])] += 1

    fact_by_property: Counter[str] = Counter(str(row.get("property_id") or row.get("property") or "unknown") for row in facts)
    fact_by_unit: Counter[str] = Counter(str(row.get("unit") or "unknown") for row in facts)
    top_nodes = [
        {"node_id": node_id, "label": entity_labels.get(node_id, node_id), "count": count}
        for node_id, count in node_counter.most_common(top_n)
    ]
    top_edges = [
        {"subject_id": s, "predicate": p, "object_id": o, "count": count}
        for (s, p, o), count in edge_counter.most_common(top_n)
    ]
    top_cooccurrence = [
        {"entity_a": a, "label_a": entity_labels.get(a, a), "entity_b": b, "label_b": entity_labels.get(b, b), "count": count}
        for (a, b), count in cooccur_counter.most_common(top_n)
    ]
    payload = {
        "graph_type": "NiCoDeterministicCheapKG",
        "updated_at": utc_now_iso(),
        "counts": {
            "sources": len(sources),
            "chunks": len(chunks),
            "entities": len(entities),
            "triples": len(triples),
            "numeric_facts": len(facts),
            "cooccurrence_edges": len(cooccur_counter),
        },
        "top_nodes": top_nodes,
        "top_edges": top_edges,
        "top_cooccurrence": top_cooccurrence,
        "numeric_fact_properties": dict(fact_by_property.most_common(top_n)),
        "numeric_fact_units": dict(fact_by_unit.most_common(top_n)),
    }
    path = store.root / "cheap_kg.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_compressed_kg_plan(
    store: KnowledgeStore,
    *,
    cheap_payload: dict[str, Any] | None = None,
    max_chunks_per_document: int = 8,
    min_kg_score: float = 0.15,
    max_chunks_per_theme: int | None = None,
    max_candidate_chunks_per_theme: int | None = None,
    min_chunk_quality: float = 0.20,
    doc_type_aware: bool = True,
    dynamic_document_limits: bool = True,
    short_document_max_chunks: int = 3,
    long_document_max_chunks: int = 6,
) -> dict[str, Any]:
    cheap = cheap_payload if cheap_payload is not None else build_cheap_kg(store)
    selection = select_graph_chunks(
        store,
        max_chunks_per_document=max_chunks_per_document,
        min_kg_score=min_kg_score,
        max_chunks_per_theme=max_chunks_per_theme,
        max_candidate_chunks_per_theme=max_candidate_chunks_per_theme,
        min_chunk_quality=min_chunk_quality,
        doc_type_aware=doc_type_aware,
        dynamic_document_limits=dynamic_document_limits,
        short_document_max_chunks=short_document_max_chunks,
        long_document_max_chunks=long_document_max_chunks,
    )
    payload = {
        "graph_type": "NiCoCompressedKGPlan",
        "updated_at": utc_now_iso(),
        "cheap_kg_path": str(store.root / "cheap_kg.json"),
        "selected_graph_chunks_path": selection["path"],
        "total_chunks": selection["total_chunks"],
        "selected_chunks": selection["selected_chunks"],
        "candidate_chunks": selection.get("candidate_chunks"),
        "candidate_chunks_pruned": selection.get("candidate_chunks_pruned"),
        "selection_ratio": selection["selection_ratio"],
        "max_chunks_per_document": selection["max_chunks_per_document"],
        "dynamic_document_limits": selection.get("dynamic_document_limits"),
        "per_doc_limits_used": selection.get("per_doc_limits_used") or {},
        "min_kg_score": selection["min_kg_score"],
        "min_chunk_quality": selection.get("min_chunk_quality"),
        "doc_type_aware": selection.get("doc_type_aware"),
        "max_chunks_per_theme": selection["max_chunks_per_theme"],
        "max_candidate_chunks_per_theme": selection.get("max_candidate_chunks_per_theme"),
        "selected_doc_types": selection.get("selected_doc_types") or {},
        "rejected_by_doc_type": selection.get("rejected_by_doc_type") or {},
        "cheap_counts": cheap.get("counts") or {},
    }
    (store.root / "compressed_kg_plan.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**payload, "rows": selection["rows"]}
