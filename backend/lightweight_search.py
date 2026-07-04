from __future__ import annotations

import math
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.config_manager import AppConfig
from backend.knowledge_store import KnowledgeStore, jsonl_read
from backend.domain.chunker import clean_extracted_text

TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё_.+-]{2,}", re.UNICODE)
STOPWORDS = {
    "как", "что", "это", "для", "или", "при", "про", "над", "под", "без", "есть", "где", "какие", "какая", "какой",
    "какое", "найди", "покажи", "расскажи", "сделай", "укажи", "между", "если", "были", "было", "from", "with",
    "the", "and", "for", "what", "where", "show", "find", "about",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("ё", "е")).strip()


def human_source_name(source: str) -> str:
    """Return a readable source label without internal kgchunk suffixes."""
    raw = str(source or "").replace("\\", "/").strip()
    if not raw:
        return "unknown"
    parts = raw.split("#")
    base = Path(parts[0]).name or parts[0] or raw
    useful_suffixes: list[str] = []
    for suffix in parts[1:]:
        if suffix.startswith(("kgchunk", "kgcompressed")):
            continue
        if suffix.startswith(("page:", "slide:", "chunk:")):
            useful_suffixes.append(suffix)
    return base + (("#" + "#".join(useful_suffixes)) if useful_suffixes else "")


def tokenize(text: str) -> list[str]:
    tokens = [t.lower().replace("ё", "е") for t in TOKEN_RE.findall(text or "")]
    return [t for t in tokens if t not in STOPWORDS and len(t) >= 2]


def _short_snippet(text: str, query_tokens: list[str], *, max_chars: int = 900) -> str:
    clean = re.sub(r"\s+", " ", clean_extracted_text(text or "")).strip()
    if len(clean) <= max_chars:
        return clean
    low = normalize_text(clean)
    positions = [low.find(tok) for tok in query_tokens if tok and low.find(tok) >= 0]
    pos = min(positions) if positions else 0
    start = max(0, pos - max_chars // 3)
    end = min(len(clean), start + max_chars)
    start = max(0, end - max_chars)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(clean) else ""
    return prefix + clean[start:end].strip() + suffix


def _metadata_matches(row: dict[str, Any], query_tokens: list[str]) -> float:
    fields = [
        str(row.get("collection") or ""),
        str(row.get("theme_name") or ""),
        str(row.get("theme_id") or ""),
        str(row.get("source_type") or ""),
        str(row.get("year") or ""),
        str(row.get("source_name") or ""),
        " ".join(str(x) for x in row.get("numeric_properties") or []),
        " ".join(str(x) for x in row.get("numeric_units") or []),
    ]
    hay = normalize_text(" ".join(fields))
    return sum(0.35 for tok in set(query_tokens) if tok in hay)


def _score_chunk(row: dict[str, Any], query: str, query_tokens: list[str]) -> float:
    text = str(row.get("text") or row.get("lightrag_text") or "")
    meta = row.get("metadata") or {}
    domain_tags = " ".join(str(x) for x in row.get("domain_tags") or [])
    entity_ids = " ".join(str(x).replace("onto:", "") for x in row.get("entity_ids") or [])
    numeric_props = " ".join(str(x) for x in row.get("numeric_properties") or [])
    haystack = normalize_text(" ".join([
        text,
        str(row.get("source_name") or ""),
        str(row.get("citation_name") or ""),
        str(row.get("theme_name") or ""),
        str(row.get("theme_id") or ""),
        str(row.get("collection") or ""),
        str(row.get("source_type") or ""),
        str(row.get("year") or ""),
        str(meta.get("title") or ""),
        str(meta.get("section") or ""),
        domain_tags,
        entity_ids,
        numeric_props,
    ]))
    if not haystack:
        return 0.0
    score = 0.0
    qnorm = normalize_text(query)
    if qnorm and qnorm in haystack:
        score += 8.0
    counts = Counter(tokenize(haystack))
    uniq = set(query_tokens)
    for tok in uniq:
        tf = counts.get(tok, 0)
        if tf:
            score += 1.0 + math.log1p(tf)
            if tok in domain_tags.lower() or tok in entity_ids.lower():
                score += 0.95
            if tok in normalize_text(str(row.get("source_name") or "")):
                score += 0.5
            if tok in normalize_text(str(row.get("collection") or "")) or tok in normalize_text(str(row.get("theme_name") or "")):
                score += 0.75
    if re.search(r"\d+(?:[,.]\d+)?\s*(?:°\s*c|c|к|%|мг/л|мг/дм3|мг/дм³|г/л|м/с|а/м2|a/m2|кпа|мпа)", haystack, re.I):
        score += 0.9
    score += min(2.5, float(row.get("graph_boost") or 0.0) * 0.35)
    score += min(1.2, float(row.get("numeric_fact_count") or 0) * 0.25)
    score += min(1.0, float(row.get("triple_count") or 0) * 0.04)
    score += _metadata_matches(row, query_tokens)
    obj_type = str(row.get("object_type") or "").lower()
    if obj_type in {"title", "abstract", "summary", "heading"}:
        score += 0.5
    return score


def _load_search_rows(store: KnowledgeStore) -> list[dict[str, Any]]:
    # Stage 2 retrieval_kg writes retrieval_index.jsonl. Prefer it because it
    # already contains graph/numeric/entity features. Fallback to raw chunks.
    retrieval_index = store.root / "retrieval_index.jsonl"
    rows = jsonl_read(retrieval_index)
    return rows if rows else jsonl_read(store.chunks_path)


@lru_cache(maxsize=1)
def _load_cross_encoder():
    if str(os.environ.get("NICO_ENABLE_LOCAL_RERANKER", "")).lower() not in {"1", "true", "yes", "on"}:
        return None
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
        model_name = os.environ.get("NICO_RERANKER_MODEL", "BAAI/bge-reranker-base")
        return CrossEncoder(model_name)
    except Exception as exc:
        print(f"[warn] local reranker disabled: {exc}")
        return None


def _rerank_if_enabled(query: str, items: list[dict[str, Any]], *, top_k: int = 30) -> list[dict[str, Any]]:
    model = _load_cross_encoder()
    if model is None or len(items) < 2:
        return items
    head = items[:top_k]
    tail = items[top_k:]
    try:
        pairs = [(query, str(item.get("text") or item.get("snippet") or "")[:1800]) for item in head]
        scores = model.predict(pairs)
        reranked = []
        for item, score in zip(head, scores):
            item = dict(item)
            item["rerank_score"] = float(score)
            item["score"] = round(float(item.get("score") or 0.0) + float(score), 4)
            reranked.append(item)
        reranked.sort(key=lambda item: (float(item.get("rerank_score") or 0.0), float(item.get("score") or 0.0)), reverse=True)
        return reranked + tail
    except Exception as exc:
        print(f"[warn] reranker failed, keeping lexical order: {exc}")
        return items


def search_theme_store(store: KnowledgeStore, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
    tokens = tokenize(query)
    if not tokens:
        return []
    rows = _load_search_rows(store)
    scored: list[dict[str, Any]] = []
    for row in rows:
        score = _score_chunk(row, query, tokens)
        if score <= 0:
            continue
        text = str(row.get("text") or row.get("lightrag_text") or "")
        scored.append({
            "score": round(score, 4),
            "chunk_id": row.get("chunk_id"),
            "source_name": row.get("source_name"),
            "citation_name": row.get("citation_name"),
            "file_path": row.get("citation_name") or row.get("source_name") or row.get("chunk_id"),
            "theme_id": row.get("theme_id"),
            "collection": row.get("collection"),
            "theme_name": row.get("theme_name"),
            "object_type": row.get("object_type"),
            "year": row.get("year"),
            "source_type": row.get("source_type"),
            "numeric_fact_count": row.get("numeric_fact_count", 0),
            "numeric_properties": row.get("numeric_properties") or [],
            "numeric_units": row.get("numeric_units") or [],
            "graph_boost": row.get("graph_boost", 0.0),
            "chars": len(text),
            "text": text,
            "snippet": _short_snippet(text, tokens),
        })
    scored.sort(key=lambda item: item["score"], reverse=True)
    scored = _rerank_if_enabled(query, scored)
    # MMR-lite: avoid returning several chunks with same source and near-identical prefix.
    selected: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in scored:
        prefix = normalize_text(item.get("snippet", ""))[:180]
        key = f"{item.get('source_name')}|{prefix}"
        if key in seen_keys:
            continue
        selected.append(item)
        seen_keys.add(key)
        if len(selected) >= limit:
            break
    return selected



_NUMERIC_VALUE_RE = re.compile(
    r"(?<!\w)[+-]?(?:\d+[\s\u00a0]?)*(?:[\.,]\d+)?\s*"
    r"(?:%|°\s*C|degC|мг/л|мг/дм3|мг/дм³|г/л|кг/т|м/с|см/с|мм/с|A/m2|А/м2|кА/м2|кПа|МПа|Гц|кадр(?:ов)?/с|ppm|ppb|pH)\b",
    re.IGNORECASE | re.UNICODE,
)

_DOMAIN_SIGNAL_RE = re.compile(
    r"\b(?:процесс|режим|услов|извлеч|концентрац|температур|скорост|расход|давлен|pH|"
    r"выщелач|экстракц|осажд|электро|флотац|плавк|обжиг|штейн|шлак|католит|анолит|"
    r"сейсм|взрыв|трещин|колебан|скорост[ьи]?\s+смещения|PVS|PPV|frequency|leaching|smelting|extraction)\b",
    re.IGNORECASE | re.UNICODE,
)

_NOISE_RE = re.compile(
    r"(?:^[#\s]*$|@|^\s*xxiv\b|^\s*\d+\s*$|^\s*источники\b|^\s*презентац|^\s*содержание\b)",
    re.IGNORECASE | re.UNICODE,
)


def _compact_text(text: str, *, max_chars: int = 520) -> str:
    clean = clean_extracted_text(text or "")
    clean = clean.replace("\u00a0", " ")
    clean = re.sub(r"\s+", " ", clean).strip(" -•#|\t")
    clean = re.sub(r"\s+([,.;:])", r"\1", clean)
    # Add missing whitespace after punctuation, but do not break decimal values
    # such as 0,68 or 1.36.
    clean = re.sub(r"([,.;:])(?!\s|$|\d)", r"\1 ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    if len(clean) <= max_chars:
        return clean
    cut = clean[:max_chars].rsplit(" ", 1)[0].strip()
    return (cut or clean[:max_chars]).rstrip(" ,;:") + "…"


def _split_sentences(text: str) -> list[str]:
    text = clean_extracted_text(text or "")
    text = text.replace("\r", "\n")
    if not text.strip():
        return []
    # PDF/PPT extraction often gives slide/table text as one long line. Treat
    # table pipes, markdown headings and numbered slide separators as weak
    # sentence boundaries, then score the resulting evidence units.
    text = re.sub(r"\s*\|\s*", " \n", text)
    text = re.sub(r"\s+(?=(?:\d{1,2}\s+)?[А-ЯЁA-Z][А-ЯЁA-Z\s]{12,})", "\n", text)
    parts = re.split(r"\n+|(?<=[.!?])\s+(?=[A-ZА-ЯЁ0-9])|(?<=[;])\s+(?=[A-ZА-ЯЁ0-9])", text)
    out: list[str] = []
    for part in parts:
        unit = _compact_text(part, max_chars=620)
        if len(unit) < 28:
            continue
        if _NOISE_RE.search(unit) and not _NUMERIC_VALUE_RE.search(unit):
            continue
        out.append(unit)
    return out


def _numeric_windows(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", clean_extracted_text(text or "")).strip()
    windows: list[str] = []
    seen: set[str] = set()
    for match in _NUMERIC_VALUE_RE.finditer(clean):
        left_candidates = [clean.rfind(sep, 0, match.start()) for sep in (". ", "; ", "|", "\n")]
        left = max(left_candidates)
        if left >= 0 and match.start() - left <= 220:
            start = left + 1
        else:
            start = max(0, match.start() - 140)
            while start > 0 and clean[start - 1].isalnum():
                start -= 1
        # Prefer semantic starts in OCR/PPT lines without punctuation.
        marker_start = -1
        for marker in ("Зарегистрированные", "Изменения", "Диапазон", "Взрывные работы", "где V", "PVS", "PPV"):
            pos = clean.rfind(marker, max(0, match.start() - 260), match.start() + 1)
            if pos > marker_start:
                marker_start = pos
        if marker_start >= 0:
            start = marker_start
        right_candidates = [clean.find(sep, match.end()) for sep in (". ", "; ", "|", "\n") if clean.find(sep, match.end()) >= 0]
        right = min(right_candidates) if right_candidates else -1
        if right >= 0 and right - match.end() <= 220:
            end = right + 1
        else:
            end = min(len(clean), match.end() + 140)
            while end < len(clean) and clean[end - 1].isalnum():
                end += 1
        window = clean[start:end].strip(" ,;:-|#")
        window = _compact_text(window, max_chars=420)
        norm = normalize_text(window)[:220]
        if len(window) >= 20 and norm not in seen:
            seen.add(norm)
            windows.append(window)
    return windows


def _unit_score(unit: str, query_tokens: list[str]) -> float:
    low = normalize_text(unit)
    q = set(query_tokens)
    score = 0.0
    score += sum(1.4 for tok in q if tok in low)
    if _NUMERIC_VALUE_RE.search(unit):
        score += 3.0
    if _DOMAIN_SIGNAL_RE.search(unit):
        score += 1.6
    if re.search(r"\b(?:вывод|результат|показано|установлено|составляет|диапазон|прогноз|мониторинг)\b", low, re.I):
        score += 0.8
    # Penalize pure title/author/contacts unless the query explicitly asks about people.
    if re.search(r"@|младший научный сотрудник|ведущий научный сотрудник|к\.т\.н", low) and not any(tok in low for tok in q):
        score -= 2.0
    if len(unit) > 480:
        score -= 0.4
    return score


def _evidence_units(item: dict[str, Any], query_tokens: list[str], *, limit: int = 3) -> list[str]:
    text = str(item.get("snippet") or item.get("text") or "")
    candidates = _numeric_windows(text) + _split_sentences(text)
    if not candidates:
        compact = _compact_text(text, max_chars=420)
        return [compact] if compact else []
    seen: set[str] = set()
    unique: list[str] = []
    for cand in candidates:
        norm = normalize_text(cand)[:180]
        if not norm or norm in seen:
            continue
        if len(cand) < 45 and not _NUMERIC_VALUE_RE.search(cand):
            continue
        seen.add(norm)
        unique.append(cand)
    unique.sort(key=lambda unit: _unit_score(unit, query_tokens), reverse=True)
    return unique[: max(1, int(limit))]


def _best_fact_sentence(item: dict[str, Any], query_tokens: list[str]) -> str:
    units = _evidence_units(item, query_tokens, limit=1)
    return units[0] if units else ""


def _source_ref(source: str, refs: dict[str, int]) -> int:
    if source not in refs:
        refs[source] = len(refs) + 1
    return refs[source]


def _add_source_once(
    source: str,
    item: dict[str, Any],
    *,
    ref_id: int,
    sources: list[dict[str, Any]],
    seen_sources: set[str],
) -> None:
    if source in seen_sources:
        return
    seen_sources.add(source)
    sources.append({
        "filename": human_source_name(source),
        "file_path": source,
        "chars": int(item.get("chars") or 0),
        "reference_id": str(ref_id),
        "theme_id": str(item.get("theme_id") or ""),
    })


def _ranked_claims(
    query: str,
    flat_items: list[dict[str, Any]],
    *,
    max_items: int = 10,
) -> tuple[list[str], list[str], list[dict[str, Any]], list[str]]:
    query_tokens = tokenize(query)
    refs: dict[str, int] = {}
    sources: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    seen_claims: set[str] = set()
    fact_claims: list[tuple[float, str]] = []
    numeric_claims: list[tuple[float, str]] = []
    themes: list[str] = []

    for item in flat_items[: max(1, int(max_items))]:
        theme_id = str(item.get("theme_id") or "unknown")
        if theme_id not in themes:
            themes.append(theme_id)
        source = str(item.get("source_name") or item.get("citation_name") or item.get("chunk_id") or "unknown")
        ref_id = _source_ref(source, refs)
        _add_source_once(source, item, ref_id=ref_id, sources=sources, seen_sources=seen_sources)
        for unit in _evidence_units(item, query_tokens, limit=2):
            nums = [m.group(0) for m in _NUMERIC_VALUE_RE.finditer(unit)]
            key = "NUM:" + "|".join(nums) if nums else normalize_text(unit)[:200]
            if not key or key in seen_claims:
                continue
            seen_claims.add(key)
            score = _unit_score(unit, query_tokens) + float(item.get("score") or 0.0) * 0.05
            line = f"{unit} [{ref_id}]"
            if _NUMERIC_VALUE_RE.search(unit):
                numeric_claims.append((score, line))
            else:
                fact_claims.append((score, line))

    numeric_claims.sort(key=lambda x: x[0], reverse=True)
    fact_claims.sort(key=lambda x: x[0], reverse=True)
    return [line for _, line in fact_claims], [line for _, line in numeric_claims], sources, themes

def build_lightweight_context(
    query: str,
    matches_by_theme: list[dict[str, Any]],
    *,
    max_items: int = 10,
    max_context_chars: int = 9000,
) -> dict[str, Any]:
    """Build a compact cited context block for LLM synthesis over retrieval_kg.

    The deterministic search layer returns scored chunks. This helper converts
    them into numbered evidence snippets so the chat answer can be synthesized
    as a normal RAG response instead of exposing raw chunks or router traces.
    """
    flat: list[dict[str, Any]] = []
    for theme_result in matches_by_theme:
        theme_id = theme_result.get("theme_id")
        for match in theme_result.get("matches") or []:
            m = dict(match)
            m.setdefault("theme_id", theme_id)
            flat.append(m)
    flat.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    query_tokens = tokenize(query)
    refs: dict[str, int] = {}
    sources: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    lines: list[str] = []
    themes: list[str] = []
    used_chars = 0

    for item in flat[: max(1, int(max_items))]:
        theme_id = str(item.get("theme_id") or "unknown")
        if theme_id not in themes:
            themes.append(theme_id)
        source = str(item.get("source_name") or item.get("citation_name") or item.get("chunk_id") or "unknown")
        ref_id = _source_ref(source, refs)
        snippet = clean_extracted_text(str(item.get("snippet") or item.get("text") or ""))
        snippet = re.sub(r"\s+", " ", snippet).strip()
        if not snippet:
            continue
        # Keep the best local sentence first, then a short context tail.
        best_sentence = _best_fact_sentence(item, query_tokens)
        if best_sentence and best_sentence not in snippet[:700]:
            snippet = f"{best_sentence} ... {snippet}"
        snippet = snippet[:1100].strip()
        meta = []
        if item.get("year"):
            meta.append(f"year={item.get('year')}")
        if item.get("source_type"):
            meta.append(f"type={item.get('source_type')}")
        if item.get("numeric_properties"):
            meta.append("numeric_properties=" + ",".join(str(x) for x in item.get("numeric_properties") or []))
        line = f"[{ref_id}] theme={theme_id}; source={human_source_name(source)}; score={item.get('score')}; {'; '.join(meta)}\n{snippet}"
        if used_chars + len(line) > max_context_chars and lines:
            break
        lines.append(line)
        used_chars += len(line)
        if source not in seen_sources:
            seen_sources.add(source)
            sources.append({
                "filename": human_source_name(source),
                "file_path": source,
                "chars": int(item.get("chars") or 0),
                "reference_id": str(ref_id),
                "theme_id": theme_id,
            })

    return {
        "context": "\n\n".join(lines),
        "sources": sources,
        "themes": themes,
        "matches": flat,
        "has_context": bool(lines),
    }

def format_lightweight_answer(
    query: str,
    matches_by_theme: list[dict[str, Any]],
    *,
    stage2_ready: bool = False,
    runtime_ready: bool = False,
) -> dict[str, Any]:
    """Format retrieval-KG results as a concise extractive RAG answer.

    This is the guaranteed fallback when the LLM synthesizer returns an empty
    response. It must not expose raw chunks, router traces or debug scores.
    """
    flat: list[dict[str, Any]] = []
    for theme_result in matches_by_theme:
        theme_id = theme_result.get("theme_id")
        for match in theme_result.get("matches") or []:
            m = dict(match)
            m.setdefault("theme_id", theme_id)
            flat.append(m)
    flat.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    if not flat:
        message = (
            "В построенном поисковом графе знаний релевантные сведения не найдены. "
            "Расширьте формулировку запроса или выберите тему/коллекцию явно."
            if stage2_ready else
            "В durable knowledge_store релевантные сведения не найдены. "
            "Расширьте формулировку запроса или выберите тему/коллекцию явно."
        )
        return {"answer": message, "sources": []}

    facts, numeric_facts, sources, themes = _ranked_claims(query, flat, max_items=12)
    primary_numeric = numeric_facts[:5]
    primary_facts = facts[: max(2, 6 - len(primary_numeric))]

    # Build a compact conclusion from the strongest evidence instead of saying
    # that the answer is a synthesis. The user needs the answer, not the trace.
    strongest = (primary_numeric + primary_facts)[:2]
    if strongest:
        ref_match = re.search(r"\[(\d+)\]\s*$", strongest[0])
        ref_suffix = f" [{ref_match.group(1)}]" if ref_match else ""
        conclusion_seed = re.sub(r"\s*\[\d+\]\s*$", "", strongest[0])
        conclusion = _compact_text(conclusion_seed, max_chars=360)
        short = f"По найденным источникам основной релевантный факт: {conclusion}{ref_suffix}"
    else:
        short = "В найденных источниках есть релевантные фрагменты, но они не дают достаточно связного фактического ответа без дополнительного контекста."

    lines: list[str] = []
    lines.append("### Краткий вывод")
    lines.append(short)
    if themes:
        theme_list = ", ".join(f"`{t}`" for t in themes[:5])
        lines.append(f"Использованные темы: {theme_list}.")

    if primary_facts:
        lines.append("")
        lines.append("### Подтверждённые факты")
        for fact in primary_facts[:6]:
            clean_fact = re.sub(r"\s+", " ", clean_extracted_text(fact)).strip()
            if clean_fact:
                lines.append(f"- {clean_fact}")

    if primary_numeric:
        lines.append("")
        lines.append("### Числовые параметры и условия")
        for fact in primary_numeric[:5]:
            clean_fact = re.sub(r"\s+", " ", clean_extracted_text(fact)).strip()
            if clean_fact:
                lines.append(f"- {clean_fact}")

    lines.append("")
    lines.append("### Ограничения и пробелы")
    if stage2_ready:
        lines.append("- Ответ построен по детерминированному retrieval_kg и исходным фрагментам knowledge_store; утверждения без ссылок на источники не добавлялись.")
    else:
        lines.append("- Ответ построен по быстрому durable knowledge_store; для более устойчивой маршрутизации желательно выполнить Stage 2 retrieval_kg.")
    if not primary_numeric:
        lines.append("- Числовые параметры, явно релевантные запросу, в найденном контексте не выделены.")
    if not primary_facts:
        lines.append("- Связные причинно-технологические утверждения в найденном контексте ограничены; требуется уточнить запрос или расширить набор тем.")

    if sources:
        lines.append("")
        lines.append("### Источники")
        for src in sources[:8]:
            ref = src.get("reference_id") or ""
            name = human_source_name(str(src.get("filename") or src.get("file_path") or "unknown"))
            lines.append(f"[{ref}] {name}")

    return {"answer": "\n".join(lines), "sources": sources}
