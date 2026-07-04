from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.domain.terms import get_ontology

_HEADING_RE = re.compile(r"^(#{1,6}\s+.+|\d+(?:\.\d+)*\s+.+|[А-ЯA-Z][^\n]{0,120})$", re.MULTILINE)
_TABLE_LINE_RE = re.compile(r"^\s*\|.+\|\s*$")
_GOST_RE = re.compile(r"\b(?:ГОСТ|GOST|СТО|ОСТ|ТУ)\s*[РR]?\s*\d+[\d\.\-:]*", re.IGNORECASE)
_JOURNAL_RE = re.compile(r"\b(?:аннотация|abstract|ключевые слова|keywords|удк|doi)\b", re.IGNORECASE)

_DOC_TYPE_EXT_RE = re.compile(r"\.([A-Za-z0-9]+)(?:$|[#?])")
_PAGE_NOISE_RE = re.compile(r"^\s*(?:стр\.?|страница|page|p\.?|с\.?)?\s*\d{1,4}\s*(?:из|of|/)??\s*\d{0,4}\s*$", re.IGNORECASE)
_BIBLIO_HEADER_RE = re.compile(r"^\s*(?:список\s+литературы|литература|библиограф(?:ия|ический\s+список)|references|bibliography|works\s+cited)\s*$", re.IGNORECASE)
_FOOTNOTE_RE = re.compile(r"^\s*(?:\[\d{1,3}\]|\d{1,3}[\).]|[*†‡])\s+(?:https?://|www\.|doi:|источник:|примечан|ibid\b|там\s+же)", re.IGNORECASE)
_SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("abstract", re.compile(r"\b(?:аннотация|abstract|резюме|summary)\b", re.IGNORECASE)),
    ("introduction", re.compile(r"\b(?:введение|introduction|актуальность)\b", re.IGNORECASE)),
    ("methods", re.compile(r"\b(?:методика|материалы\s+и\s+методы|методы|эксперимент|оборудование|methods?|experimental|procedure)\b", re.IGNORECASE)),
    ("results", re.compile(r"\b(?:результаты|обсуждение\s+результатов|results?|полученные\s+данные)\b", re.IGNORECASE)),
    ("discussion", re.compile(r"\b(?:обсуждение|discussion)\b", re.IGNORECASE)),
    ("conclusion", re.compile(r"\b(?:заключение|выводы|conclusions?)\b", re.IGNORECASE)),
    ("requirements", re.compile(r"\b(?:требования|нормы|нормативные\s+требования|shall|must|requirements?)\b", re.IGNORECASE)),
    ("definitions", re.compile(r"\b(?:термины\s+и\s+определения|definitions?|обозначения)\b", re.IGNORECASE)),
    ("appendix", re.compile(r"\b(?:приложение|appendix|annex)\b", re.IGNORECASE)),
    ("references", re.compile(r"\b(?:список\s+литературы|литература|references|bibliography)\b", re.IGNORECASE)),
]


_IMAGE_MARKUP_RE = re.compile(r"<!--\s*image\s*-->|<!-+\s*image\s*-+>|\[image\]|\(image\)", re.IGNORECASE)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# Typical PDF/OCR mojibake from embedded fonts. These characters are printable,
# but they render as unreadable sequences in snippets (e.g. "ɋɨɫɬɚɜ", "Ĭőņ").
_WEIRD_OCR_GLYPH_RE = re.compile(r"[\u00A1-\u00AC\u00AD-\u00AF\u00B4\u00B6-\u00BF\u00C0-\u00D6\u00D8-\u00FF\u0100-\u024F\u0250-\u02AF\u0300-\u036F\u1D00-\u1D7F\u2C60-\u2C7F\uA720-\uA7FF]")
_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060-\u206F]")
_LONG_PUNCT_RE = re.compile(r"(?:[\uFFFD\u0000-\u001F]|[\u0370-\u03FF]{8,}|[\u0180-\u024F]{8,})")
_SPACED_LETTERS_RE = re.compile(r"(?:(?<=^)|(?<=\s))(?:[A-Za-zА-Яа-яЁё]\s+){8,}[A-Za-zА-Яа-яЁё](?=\s|$)")


def clean_extracted_text(text: str) -> str:
    """Remove extractor artefacts before storing chunks.

    The goal is conservative cleanup: remove slide/image placeholders, control
    characters, repeated boilerplate spaces and obviously broken OCR/PDF lines,
    without rewriting technical formulas and units.
    """
    if not text:
        return ""
    text = text.replace("\u00a0", " ").replace("\ufeff", " ")
    text = text.replace("\ufffd", " ")
    text = _ZERO_WIDTH_RE.sub(" ", text)
    text = _CONTROL_RE.sub(" ", text)
    text = _WEIRD_OCR_GLYPH_RE.sub(" ", text)
    text = _IMAGE_MARKUP_RE.sub(" ", text)
    # Join words split by PDF line hyphenation: "техноло-\n гия" -> "технология".
    text = re.sub(r"(?<=[A-Za-zА-Яа-яЁё])-\s*\n\s*(?=[A-Za-zА-Яа-яЁё])", "", text)
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = _WEIRD_OCR_GLYPH_RE.sub(" ", raw_line)
        line = _ZERO_WIDTH_RE.sub(" ", line)
        line = re.sub(r"[ \t]{2,}", " ", line).strip()
        if not line:
            cleaned_lines.append("")
            continue
        # Drop pure image remnants and long sequences of isolated glyphs.
        if _IMAGE_MARKUP_RE.search(line):
            line = _IMAGE_MARKUP_RE.sub(" ", line).strip()
        compact = line.replace(" ", "")
        good_chars = sum(1 for ch in compact if ch.isalnum() or ch in "°%.,;:/\\-+()[]{}№#=<>≤≥±×·")
        if len(compact) >= 24 and good_chars / max(1, len(compact)) < 0.55:
            continue
        # Lines like "е л я ю т с я н а о с н о в е" are usually PDF glyph noise.
        spaced = _SPACED_LETTERS_RE.search(line)
        if spaced and len(line) > 40:
            continue
        if _LONG_PUNCT_RE.search(line) and len(line) > 30:
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()



@dataclass(slots=True)
class DocumentObject:
    object_id: str
    source_name: str
    object_type: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation_name(self) -> str:
        suffix = self.metadata.get("page") or self.metadata.get("section") or self.object_id
        return f"{self.source_name}#{self.object_type}:{suffix}"

    def to_lightrag_text(self) -> str:
        meta = {
            "source": self.source_name,
            "object_id": self.object_id,
            "object_type": self.object_type,
            **self.metadata,
        }
        header = "\n".join(f"{key}: {value}" for key, value in meta.items() if value not in (None, "", []))
        return f"[METADATA]\n{header}\n[/METADATA]\n\n{self.text.strip()}".strip()


def _hash_id(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8", errors="ignore")).hexdigest()[:16]


def infer_source_year(source_name: str, text: str = "") -> int | None:
    for blob in (source_name or "", text[:2000] if text else ""):
        matches = re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", blob)
        if matches:
            # Prefer the latest year mentioned in the source name/header.
            return max(int(x) for x in matches)
    return None


def infer_document_type(source_name: str, text: str = "", base_metadata: dict[str, Any] | None = None) -> str:
    metadata = base_metadata or {}
    explicit = str(metadata.get("document_type") or metadata.get("document_kind") or "").strip().lower()
    if explicit and explicit not in {"document", "generic"}:
        return explicit
    source = source_name or ""
    sample = f"{source}\n{text[:5000]}"
    low = sample.lower().replace("ё", "е")
    m = _DOC_TYPE_EXT_RE.search(source)
    ext = (m.group(1).lower() if m else str(metadata.get("extension") or "").lstrip(".").lower())
    if ext in {"ppt", "pptx"} or "презентац" in low or "slide" in low:
        return "presentation"
    if ext in {"xls", "xlsx", "csv", "tsv"}:
        return "spreadsheet"
    if _GOST_RE.search(sample) or "гост" in low or " iso " in f" {low} " or "стандарт" in low:
        return "standard"
    if re.search(r"\bпатент\b|\bpatent\b", sample, re.IGNORECASE):
        return "patent"
    if ext == "pdf" and _JOURNAL_RE.search(sample):
        return "article"
    if re.search(r"\bотчет\b|\bпротокол\b|\breport\b", sample, re.IGNORECASE):
        return "report"
    if ext == "pdf":
        return "pdf_report"
    return "generic"


def detect_document_kind(source_name: str, text: str) -> str:
    # Backward-compatible alias used by older code paths.
    return infer_document_type(source_name, text)


def infer_section_type(text: str, *, source_name: str = "", document_type: str = "generic", object_type: str = "chunk") -> str:
    sample = clean_extracted_text(text)[:2200]
    title = _section_title_for(sample).lower().replace("ё", "е") if sample else ""
    hay = f"{source_name}\n{title}\n{sample[:500]}"
    if object_type == "table":
        return "table"
    for section_type, pattern in _SECTION_PATTERNS:
        if pattern.search(hay):
            return section_type
    if document_type == "presentation":
        if re.search(r"\b(?:актуальность|цель|задачи)\b", hay, re.IGNORECASE):
            return "introduction"
        if re.search(r"\b(?:прибор|оборудование|методика|измерени)\b", hay, re.IGNORECASE):
            return "methods"
        if re.search(r"\b(?:результат|зарегистрирован|получен|анализ)\b", hay, re.IGNORECASE):
            return "results"
    return "body"


def _norm_line(line: str) -> str:
    return re.sub(r"\s+", " ", (line or "").strip().lower().replace("ё", "е"))


def detect_repeated_noise_lines(texts: list[str], *, min_repeats: int = 3) -> set[str]:
    """Find repeated page headers/footers/table headers across pages/slides.

    The function is conservative: it only removes short repeated lines or table
    header-like lines. Numeric/domain-rich rows are kept because they may be data.
    """
    counter: Counter[str] = Counter()
    examples: dict[str, str] = {}
    for text in texts:
        seen_in_doc: set[str] = set()
        for raw in (text or "").splitlines():
            line = re.sub(r"[ \t]{2,}", " ", raw).strip()
            norm = _norm_line(line)
            if not norm or len(norm) < 4:
                continue
            if len(norm) > 180 and "|" not in norm:
                continue
            if _PAGE_NOISE_RE.match(norm):
                continue
            seen_in_doc.add(norm)
            examples.setdefault(norm, line)
        counter.update(seen_in_doc)
    threshold = max(2, min_repeats)
    out: set[str] = set()
    for norm, count in counter.items():
        if count < threshold:
            continue
        example = examples.get(norm, norm)
        numeric_hits = len(re.findall(r"\d+(?:[,.]\d+)?", example))
        # Keep repeated rows with multiple values; drop repeated headers/footers.
        if numeric_hits >= 3 and "|" in example:
            continue
        if len(norm) <= 160 or "|" in norm:
            out.add(norm)
    return out


def preprocess_text_for_chunking(
    text: str,
    *,
    source_name: str = "",
    document_type: str | None = None,
    repeated_noise_lines: set[str] | None = None,
    drop_bibliography: bool = True,
) -> str:
    """Clean document-level noise before chunking/indexing.

    Removes page numbers, repeated headers/footers/table headers, conservative
    footnote/citation debris and bibliography tails. It does not rewrite formulas
    or units; richer LLM cleanup should remain an optional later stage.
    """
    text = clean_extracted_text(text)
    if not text:
        return ""
    doc_type = document_type or infer_document_type(source_name, text)
    repeated = repeated_noise_lines or set()
    cleaned_lines: list[str] = []
    in_bibliography = False
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]{2,}", " ", raw_line).strip()
        norm = _norm_line(line)
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        if _PAGE_NOISE_RE.match(line):
            continue
        if norm in repeated:
            continue
        if drop_bibliography and _BIBLIO_HEADER_RE.match(line) and doc_type in {"article", "pdf_report", "report", "standard", "generic"}:
            in_bibliography = True
            continue
        if in_bibliography:
            # Preserve a following appendix/section if the extractor concatenated
            # separate parts after references; otherwise skip bibliography rows.
            if re.match(r"^\s*(?:приложение|appendix|annex)\b", line, re.IGNORECASE):
                in_bibliography = False
            else:
                continue
        if _FOOTNOTE_RE.match(line):
            continue
        if re.match(r"^\s*(?:doi:|https?://|www\.)\S+\s*$", line, re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _domain_tags(text: str) -> list[str]:
    ontology = get_ontology()
    return list(dict.fromkeys(match.term_id for match in ontology.find_terms(text)))[:24]


def _parse_markdown_table(table: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw in (table or "").splitlines():
        line = raw.strip()
        if not _TABLE_LINE_RE.match(line):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and all(re.fullmatch(r"[-: ]+", c or "") for c in cells):
            continue
        if any(cells):
            rows.append(cells)
    if not rows:
        return []
    width = max(len(row) for row in rows)
    return [row + [""] * (width - len(row)) for row in rows]


def table_to_text_description(table: str, *, max_rows: int = 24) -> str:
    rows = _parse_markdown_table(table)
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]
    cols = [h or f"Колонка {i + 1}" for i, h in enumerate(header)]
    numeric_cells = sum(1 for row in body for cell in row if re.search(r"\d", cell))
    domain_tags = _domain_tags(table)
    parts: list[str] = []
    parts.append(f"Таблица содержит {len(body)} строк и {len(cols)} столбцов: {', '.join(cols[:12])}.")
    if numeric_cells:
        parts.append(f"В таблице обнаружены числовые значения: {numeric_cells} ячеек с числами.")
    if domain_tags:
        parts.append(f"Связанные доменные теги: {', '.join(domain_tags[:10])}.")
    for idx, row in enumerate(body[:max_rows], start=1):
        pairs = [f"{cols[i]} = {cell}" for i, cell in enumerate(row) if str(cell).strip()]
        if pairs:
            parts.append(f"Строка {idx}: " + "; ".join(pairs) + ".")
    if len(body) > max_rows:
        parts.append(f"Остальные строки сохранены в оригинальном табличном представлении: {len(body) - max_rows}.")
    return "\n".join(parts)


def _split_markdown_tables(text: str) -> tuple[str, list[str]]:
    lines = text.splitlines()
    normal_lines: list[str] = []
    tables: list[str] = []
    current: list[str] = []

    def flush_table() -> None:
        nonlocal current
        if len(current) >= 2:
            tables.append("\n".join(current).strip())
        elif current:
            normal_lines.extend(current)
        current = []

    for line in lines:
        if _TABLE_LINE_RE.match(line):
            current.append(line)
        else:
            flush_table()
            normal_lines.append(line)
    flush_table()
    return "\n".join(normal_lines), tables


def _section_title_for(text: str) -> str:
    for line in text.splitlines():
        clean = line.strip(" #\t")
        if 8 <= len(clean) <= 140:
            return clean
    return ""


def split_text_to_objects(
    text: str,
    *,
    source_name: str,
    base_metadata: dict[str, Any] | None = None,
    max_chars: int = 3200,
    overlap: int = 300,
) -> list[DocumentObject]:
    metadata = dict(base_metadata or {})
    document_type = infer_document_type(source_name, text, metadata)
    metadata.setdefault("document_type", document_type)
    metadata.setdefault("document_kind", document_type)
    year = infer_source_year(source_name, text)
    if year is not None and not metadata.get("year"):
        metadata["year"] = year
    repeated_noise = metadata.pop("repeated_noise_lines", None)
    repeated_set = set(repeated_noise or []) if not isinstance(repeated_noise, str) else {x.strip() for x in repeated_noise.split("||") if x.strip()}
    text = preprocess_text_for_chunking(text, source_name=source_name, document_type=document_type, repeated_noise_lines=repeated_set)
    if not text:
        return []
    text_without_tables, tables = _split_markdown_tables(text)
    objects: list[DocumentObject] = []

    for table_idx, table in enumerate(tables, start=1):
        object_id = _hash_id(source_name, "table", str(table_idx), table[:120])
        table_description = table_to_text_description(table)
        table_text = (
            "[Описание таблицы]\n" + table_description.strip() + "\n\n[Оригинальная таблица]\n" + table.strip()
            if table_description else table.strip()
        )
        table_meta = {
            **metadata,
            "table_index": table_idx,
            "section_type": "table",
            "table_to_text": True,
            "original_table_hash": _hash_id(source_name, "original_table", str(table_idx), table[:400]),
            "domain_tags": ",".join(_domain_tags(table + "\n" + table_description)),
        }
        objects.append(DocumentObject(object_id, source_name, "table", table_text, table_meta))

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text_without_tables) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}".strip()
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= max_chars:
            current = paragraph
        else:
            start = 0
            while start < len(paragraph):
                end = min(len(paragraph), start + max_chars)
                chunks.append(paragraph[start:end])
                if end == len(paragraph):
                    break
                start = max(0, end - overlap)
            current = ""
    if current:
        chunks.append(current)

    out_idx = 0
    for idx, chunk in enumerate(chunks, start=1):
        # Drop tiny non-technical fragments before they enter embeddings/LightRAG.
        domain_tags = _domain_tags(chunk)
        has_numeric = bool(re.search(r"\d", chunk))
        if len(re.sub(r"\s+", " ", chunk).strip()) < 60 and not (domain_tags or has_numeric):
            continue
        out_idx += 1
        object_id = _hash_id(source_name, "chunk", str(idx), chunk[:120])
        section = _section_title_for(chunk)
        section_type = infer_section_type(chunk, source_name=source_name, document_type=str(metadata.get("document_type") or "generic"), object_type="chunk")
        chunk_meta = {
            **metadata,
            "chunk_index": out_idx,
            "source_chunk_index": idx,
            "section": section,
            "section_type": section_type,
            "domain_tags": ",".join(domain_tags),
        }
        objects.append(DocumentObject(object_id, source_name, "chunk", chunk, chunk_meta))

    return objects


def source_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
