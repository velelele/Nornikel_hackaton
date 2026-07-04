from __future__ import annotations

import asyncio
import json
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Iterable

from backend.config_manager import AppConfig
from backend.document_loader import SUPPORTED_EXTENSIONS
from backend.domain.document_processor import ProcessedDocument
from backend.domain.numeric_extractor import NumericFact
from backend.domain.terms import get_ontology
from backend.knowledge_store import KnowledgeStore, jsonl_read, utc_now_iso
from backend.lightweight_search import build_lightweight_context, format_lightweight_answer, search_theme_store
from backend.graph_embedding_intelligence import (
    build_theme_embeddings,
    compute_graph_metrics,
    embed_texts_openai_compatible,
    load_graph_metrics,
    route_theme_scores,
    write_graph_metrics,
)
from backend.compressed_kg import (
    build_cheap_kg,
    build_compressed_kg_plan,
    lightrag_items_from_selected_rows,
)
from backend.retrieval_kg import build_retrieval_kg

if TYPE_CHECKING:
    from backend.rag_service import RagService


_READINESS_ORDER = {
    "failed": -1,
    "not_ready": 0,
    "parsed_ready": 1,
    "search_ready": 2,
    "cheap_kg_ready": 3,
    "retrieval_kg_ready": 4,
    "compressed_kg_ready": 4,  # backward-compatible status for old runs only
    "full_kg_ready": 5,
}

DEFAULT_INGESTION_PROFILES: dict[str, dict[str, Any]] = {
    "fast_fill": {
        "build_runtime_index": False,
        "compute_graph_metrics_during_ingest": False,
        "rebuild_graph_metrics_after": False,
        "rebuild_theme_embeddings_after": False,
        "build_global_router_after": True,
        "batch_size_docs": 32,
        "status": "search_ready",
        "search_backend": "knowledge_store_lightweight",
    },
    "fast_smoke": {
        "build_runtime_index": True,
        "compute_graph_metrics_during_ingest": False,
        "rebuild_graph_metrics_after": True,
        "rebuild_theme_embeddings_after": True,
        "build_global_router_after": True,
        "batch_size_docs": 16,
        "status": "search_ready",
    },
    "balanced": {
        "build_runtime_index": True,
        "compute_graph_metrics_during_ingest": True,
        "rebuild_graph_metrics_after": True,
        "rebuild_theme_embeddings_after": True,
        "build_global_router_after": True,
        "batch_size_docs": 8,
        "status": "search_ready",
    },
    "safe": {
        "build_runtime_index": True,
        "compute_graph_metrics_during_ingest": True,
        "rebuild_graph_metrics_after": True,
        "rebuild_theme_embeddings_after": True,
        "build_global_router_after": True,
        "batch_size_docs": 4,
        "status": "search_ready",
    },
    # Stage 2 default for the web UI: retrieval KG only. It does not call
    # LightRAG/Ollama and therefore cannot hang on LightRAG final graph write.
    "overnight_retrieval_kg": {
        "graph_mode": "retrieval_kg",
        "build_runtime_index": False,
        "clear_runtime": False,
        "wait": False,
        "batch_size_docs": 32,
        "poll_interval": 10.0,
        "timeout_sec": 0.0,
        "max_chunks_per_document_for_graph": 0,
        "min_kg_score": 0.0,
        "max_chunks_per_theme": 0,
        "fallback_to_lightweight_on_timeout": True,
        "rebuild_graph_metrics_after": False,
        "rebuild_theme_embeddings_after": False,
        "build_global_router_after": True,
        "target_status": "retrieval_kg_ready",
        "description": "Safe retrieval KG over knowledge_store: chunk index + numeric/entity/triple boosts + metadata-aware search; no LightRAG runtime.",
    },
    # Compact LightRAG remains available, but only as an explicit optional mode.
    # It is bounded by small batches, a per-theme timeout and a hard chunk cap.
    "overnight_compressed_lightrag": {
        "graph_mode": "compressed_kg",
        "build_runtime_index": True,
        "clear_runtime": False,
        "wait": True,
        "batch_size_docs": 2,
        "poll_interval": 5.0,
        "timeout_sec": 600.0,
        "max_chunks_per_document_for_graph": 4,
        "min_kg_score": 0.15,
        "max_chunks_per_theme": 32,
        "fallback_to_lightweight_on_timeout": True,
        "rebuild_graph_metrics_after": False,
        "rebuild_theme_embeddings_after": False,
        "build_global_router_after": True,
        "target_status": "compressed_kg_ready",
    },
    "overnight_full": {
        "graph_mode": "full_kg",
        "build_runtime_index": True,
        "clear_runtime": False,
        "wait": True,
        "batch_size_docs": 2,
        "poll_interval": 5.0,
        "timeout_sec": 900.0,
        "fallback_to_lightweight_on_timeout": True,
        "rebuild_graph_metrics_after": False,
        "rebuild_theme_embeddings_after": False,
        "build_global_router_after": True,
        "target_status": "full_kg_ready",
    },
}


def get_ingestion_profile(config: AppConfig, name: str) -> dict[str, Any]:
    profile = dict(DEFAULT_INGESTION_PROFILES.get(name, {}))
    custom_profiles = getattr(config, "ingestion_profiles", {}) or {}
    custom = custom_profiles.get(name) if isinstance(custom_profiles, dict) else None
    if isinstance(custom, dict):
        profile.update(custom)
    if not profile:
        raise KeyError(f"Unknown ingestion profile: {name}")
    profile["name"] = name
    return profile

YEAR_RE_DEFAULT = r"^(19|20)\d{2}$"
YEAR_ANYWHERE_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё_\-]{3,}", re.UNICODE)

DEFAULT_KNOWN_COLLECTIONS: dict[str, list[str]] = {
    "журналы": ["журналы", "журнал", "journal", "journals", "articles", "article", "papers", "статьи", "статья"],
    "конференции": ["конференции", "конференция", "conference", "conferences", "proceedings", "symposium", "alta", "conference_papers"],
    "патенты": ["патенты", "патент", "patent", "patents"],
    "отчёты": ["отчеты", "отчёты", "отчет", "отчёт", "reports", "report", "protocols", "протоколы", "протокол"],
    "стандарты": ["нормативы", "нормативная", "стандарты", "стандарт", "gost", "гост", "standards", "standard"],
    "презентации": ["презентации", "презентация", "presentations", "presentation", "slides"],
    "книги": ["книги", "книга", "book", "books", "monograph", "монография"],
    "геотехника": ["геотехника", "geotechnics", "geotech", "сейсмика", "seismics", "бвр", "буровзрыв"],
    "гидрометаллургия": ["гидрометаллургия", "hydrometallurgy", "leaching", "выщелачивание", "электроэкстракция"],
    "пирометаллургия": ["пирометаллургия", "pyrometallurgy", "smelting", "плавка", "штейн", "шлак"],
    "экология": ["экология", "environment", "environmental", "газоочистка", "water", "воды"],
}

SOURCE_TYPE_BY_COLLECTION = {
    "журналы": "journal_article",
    "конференции": "conference_paper",
    "патенты": "patent",
    "отчёты": "report_or_protocol",
    "стандарты": "standard",
    "презентации": "presentation",
    "книги": "book",
    "геотехника": "report_or_article",
    "гидрометаллургия": "report_or_article",
    "пирометаллургия": "report_or_article",
    "экология": "report_or_article",
}

GENERIC_DIRS = {
    "data", "datas", "docs", "documents", "document", "files", "file", "source", "sources", "исходники",
    "материалы", "corpus", "root", "root_dir", "nornikel", "норникель", "misc", "разное", "прочее",
}

DEFAULT_THEME_PATTERNS: list[dict[str, Any]] = [
    {"pattern": r".*ALTA.*Ni.*Co.*", "collection": "конференции", "theme_name": "ALTA_Ni_Co", "source_type": "conference_paper"},
    {"pattern": r".*(?:SO2|SO₂|сернист|sulfur\s*dioxide|sulphur\s*dioxide|газоочист).*", "collection": "экология", "theme_name": "SO2_removal", "source_type": "report_or_article"},
    {"pattern": r".*Горн(?:ая|ой).*промышленн.*", "collection": "журналы", "theme_name": "Горная промышленность", "source_type": "journal_article"},
    {"pattern": r".*(?:Цветные\s*металлы|ЦМ).*", "collection": "журналы", "theme_name": "Цветные металлы", "source_type": "journal_article"},
    {"pattern": r".*(?:Смеш.*гидроксид|mixed\s*hydroxide|MHP).*", "collection": "гидрометаллургия", "theme_name": "mixed_hydroxides", "source_type": "report_or_article"},
    {"pattern": r".*(?:католит|catholyte|electrowinning|electro\s*winning|электроэкстрак).*", "collection": "гидрометаллургия", "theme_name": "nickel_electrowinning", "source_type": "report_or_article"},
    {"pattern": r".*(?:штейн.*шлак|matte.*slag|slag.*matte|МПГ|PGM|platinum\s*group).*", "collection": "пирометаллургия", "theme_name": "matte_slag_partitioning", "source_type": "report_or_article"},
    {"pattern": r".*(?:шахтн.*вод|mine\s*water|закачк.*горизонт).*", "collection": "экология", "theme_name": "mine_water", "source_type": "report_or_article"},
    {"pattern": r".*(?:сейсмограмм|велосиграмм|акселерограмм|сейсмовзрыв|буровзрыв|взрывн(?:ые|ых)?\s+работ|PPV|PVS|пиков.*скорост|скорост.*смещ|динамическ.*расчет|динамическ.*расчёт).*", "collection": "геотехника", "theme_name": "blasting_seismics", "source_type": "report_or_article"},
    {"pattern": r".*(?:ГОСТ|GOST|standard|стандарт).*", "collection": "стандарты", "theme_name": "standards", "source_type": "standard"},
]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("ё", "е")).strip()


def _norm_key(text: str) -> str:
    return re.sub(r"[^0-9a-zа-я]+", "", _norm(text), flags=re.UNICODE)


def _clean_part(part: str) -> str:
    return (part or "").strip().strip("/\\")


def _split_relative_path(relative_path: str) -> list[str]:
    clean = (relative_path or "").replace("\\", "/").strip().lstrip("/")
    return [p for p in (_clean_part(x) for x in clean.split("/")) if p and p not in {".", ".."}]


def slugify_theme_part(value: str, *, fallback: str = "misc") -> str:
    value = (value or "").strip().replace("\\", "/")
    value = PurePosixPath(value).name if "/" in value else value
    value = value.replace("ё", "е").replace("Ё", "Е")
    value = re.sub(r"[^0-9A-Za-zА-Яа-я_.-]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"_+", "_", value).strip("_.-")
    return value or fallback


def safe_theme_id(collection: str, theme_name: str) -> str:
    return f"{slugify_theme_part(collection)}__{slugify_theme_part(theme_name)}"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _merge_known_collections(raw: dict[str, Any] | None) -> dict[str, list[str]]:
    merged = {key: list(dict.fromkeys([key, *aliases])) for key, aliases in DEFAULT_KNOWN_COLLECTIONS.items()}
    for collection, aliases in (raw or {}).items():
        if isinstance(aliases, dict):
            alias_values = aliases.get("aliases") or aliases.get("labels") or []
        else:
            alias_values = aliases
        merged[str(collection)] = list(dict.fromkeys([str(collection), *[str(x) for x in _as_list(alias_values)]]))
    return merged


def _compile_alias_map(known_collections: dict[str, list[str]]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for canonical, aliases in known_collections.items():
        alias_map[_norm_key(canonical)] = canonical
        for alias in aliases:
            alias_map[_norm_key(str(alias))] = canonical
    return alias_map


def _path_years(parts: Iterable[str]) -> list[int]:
    years: list[int] = []
    for part in parts:
        for match in YEAR_ANYWHERE_RE.finditer(part):
            year = int(match.group(1))
            if 1900 <= year <= 2099 and year not in years:
                years.append(year)
    return years


def _strip_extension(name: str) -> str:
    return Path(name).stem if name else ""


def _is_generic_segment(segment: str, year_re: re.Pattern[str]) -> bool:
    key = _norm_key(segment)
    return not key or key in GENERIC_DIRS or bool(year_re.match(segment.strip()))


def _clean_theme_from_filename(filename: str) -> str:
    stem = _strip_extension(filename)
    stem = re.sub(r"(?i)\b(?:proceedings|article|статья|готовая|final|draft|презентация|presentation)\b", " ", stem)
    stem = YEAR_ANYWHERE_RE.sub(" ", stem)
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    if len(stem) > 80:
        # Keep a stable but not absurdly long auto-theme name.
        words = stem.split()
        stem = " ".join(words[:8])
    return stem or "unclassified"


@dataclass(slots=True)
class ThemeInfo:
    theme_id: str
    collection: str
    theme_name: str
    relative_path: str
    source_name: str
    year: int | None = None
    source_type: str = "unknown"
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ThemeResolver:
    """Hybrid resolver: path + filename + year-folders + overrides + content preview.

    It is deliberately conservative: if the path is ambiguous, the file goes to
    misc/unclassified or to an auto theme derived from filename/content rather
    than polluting an unrelated explicit theme.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        default_collection: str = "misc",
        default_theme: str = "unclassified",
        year_dir_pattern: str = YEAR_RE_DEFAULT,
        known_collections: dict[str, Any] | None = None,
        theme_overrides: list[dict[str, Any]] | None = None,
        use_path: bool = True,
        use_filename: bool = True,
        use_content_preview: bool = True,
        content_preview_chars: int = 5000,
        min_confidence: float = 0.55,
        auto_theme_from_filename: bool = True,
        auto_theme_from_content_terms: bool = True,
        collapse_low_confidence: bool = True,
    ) -> None:
        self.enabled = enabled
        self.default_collection = default_collection
        self.default_theme = default_theme
        self.year_re = re.compile(year_dir_pattern or YEAR_RE_DEFAULT, re.IGNORECASE)
        self.known_collections = _merge_known_collections(known_collections)
        self.alias_map = _compile_alias_map(self.known_collections)
        self.overrides = [*DEFAULT_THEME_PATTERNS, *(theme_overrides or [])]
        self.use_path = use_path
        self.use_filename = use_filename
        self.use_content_preview = use_content_preview
        self.content_preview_chars = max(0, int(content_preview_chars))
        self.min_confidence = float(min_confidence)
        self.auto_theme_from_filename = bool(auto_theme_from_filename)
        self.auto_theme_from_content_terms = bool(auto_theme_from_content_terms)
        self.collapse_low_confidence = bool(collapse_low_confidence)

    @classmethod
    def from_config(cls, config: AppConfig) -> "ThemeResolver":
        return cls(
            enabled=bool(config.topic_sharding_enabled),
            default_collection=config.topic_default_collection,
            default_theme=config.topic_default_theme,
            year_dir_pattern=config.file_discovery_year_dir_pattern,
            known_collections=config.theme_known_collections,
            theme_overrides=config.theme_overrides,
            use_path=config.theme_use_path,
            use_filename=config.theme_use_filename,
            use_content_preview=config.theme_use_content_preview,
            content_preview_chars=config.theme_content_preview_chars,
            min_confidence=config.theme_min_confidence,
            auto_theme_from_filename=config.theme_auto_theme_from_filename,
            auto_theme_from_content_terms=config.theme_auto_theme_from_content_terms,
            collapse_low_confidence=config.theme_collapse_low_confidence,
        )

    def _collection_for_segment(self, segment: str) -> str | None:
        return self.alias_map.get(_norm_key(segment))

    def _normalize_parts(self, parts: list[str]) -> list[str]:
        # Browser folder picker often sends selected_root/collection/theme/file.
        # If the first segment is artificial and the second is a known collection,
        # drop the first segment. This keeps backend robust even if frontend did not strip it.
        if len(parts) >= 3 and self._collection_for_segment(parts[1]) and not self._collection_for_segment(parts[0]):
            return parts[1:]
        return parts

    def _match_override(self, haystack: str) -> dict[str, Any] | None:
        for raw in self.overrides:
            pattern = str(raw.get("pattern") or "")
            if not pattern:
                continue
            try:
                if re.search(pattern, haystack, flags=re.IGNORECASE | re.UNICODE):
                    return raw
            except re.error:
                continue
        return None

    def _source_type(self, collection: str, filename: str, haystack: str) -> str:
        if collection in SOURCE_TYPE_BY_COLLECTION:
            return SOURCE_TYPE_BY_COLLECTION[collection]
        low = _norm(haystack + " " + filename)
        if re.search(r"patent|патент", low):
            return "patent"
        if re.search(r"gost|гост|standard|стандарт", low):
            return "standard"
        if re.search(r"conference|proceedings|конференц|alta", low):
            return "conference_paper"
        if re.search(r"journal|журнал|статья|article|цветные металлы|горная промышленность", low):
            return "journal_article"
        if re.search(r"presentation|презентац|slides", low) or Path(filename).suffix.lower() in {".ppt", ".pptx"}:
            return "presentation"
        if re.search(r"report|отчет|отчёт|protocol|протокол", low):
            return "report_or_protocol"
        return "report_or_article"

    def _should_protect_theme(self, evidence: list[str]) -> bool:
        for item in evidence:
            if (
                item == "override_pattern"
                or item.startswith("path:theme")
                or item.startswith("filename:domain_pattern")
                or item.startswith("filename:journal_name")
                or item.startswith("filename:ALTA_Ni_Co")
                or item.startswith("content_preview:domain_pattern")
            ):
                return True
        return False

    def _collapse_if_low_confidence(
        self,
        collection: str,
        theme_name: str,
        confidence: float,
        evidence: list[str],
    ) -> tuple[str, float, list[str]]:
        if not self.collapse_low_confidence or not theme_name or theme_name == self.default_theme:
            return theme_name, confidence, evidence
        if confidence >= self.min_confidence or self._should_protect_theme(evidence):
            return theme_name, confidence, evidence
        collapsed_from = theme_name
        evidence = [
            *evidence,
            f"low_confidence_theme_collapsed:{collapsed_from}:{confidence:.2f}<min:{self.min_confidence:.2f}",
        ]
        return self.default_theme, max(confidence, min(self.min_confidence, 0.55)), evidence

    def _infer_theme_from_filename_or_content(self, filename: str, preview_text: str, collection: str | None) -> tuple[str, str | None, str | None, list[str], float]:
        haystack = f"{filename}\n{preview_text[:self.content_preview_chars]}"
        override = self._match_override(haystack)
        if override:
            return (
                str(override.get("theme_name") or self.default_theme),
                str(override.get("collection") or collection or self.default_collection),
                str(override.get("source_type") or ""),
                ["override_pattern"],
                0.98,
            )

        evidence: list[str] = []
        fname = _norm(filename)
        theme = ""
        if self.use_filename:
            if re.search(r"alta.*ni.*co", fname, flags=re.IGNORECASE):
                theme, evidence = "ALTA_Ni_Co", ["filename:ALTA_Ni_Co"]
            elif re.search(r"горн.*промышлен", fname):
                theme, evidence = "Горная промышленность", ["filename:journal_name"]
            elif re.search(r"цветн.*металл|\bцм\b", fname):
                theme, evidence = "Цветные металлы", ["filename:journal_name"]
            elif re.search(r"so2|so₂|сернист|газоочист", fname):
                theme, evidence = "SO2_removal", ["filename:domain_pattern"]
            elif re.search(r"смеш.*гидроксид|mixed.*hydroxide|\bmhp\b", fname):
                theme, evidence = "mixed_hydroxides", ["filename:domain_pattern"]
            elif re.search(r"католит|electrowinning|электроэкстрак", fname):
                theme, evidence = "nickel_electrowinning", ["filename:domain_pattern"]
            elif re.search(r"штейн|matte|slag|шлак|мпг|pgm", fname):
                theme, evidence = "matte_slag_partitioning", ["filename:domain_pattern"]
            elif re.search(r"шахт.*вод|mine.*water", fname):
                theme, evidence = "mine_water", ["filename:domain_pattern"]
            elif self.auto_theme_from_filename:
                cleaned = _clean_theme_from_filename(filename)
                if cleaned and _norm_key(cleaned) not in GENERIC_DIRS and cleaned != "unclassified":
                    theme, evidence = cleaned, ["filename:auto_theme"]

        if not theme and self.use_content_preview and preview_text:
            preview = _norm(preview_text[:self.content_preview_chars])
            content_rules = [
                (r"so2|so₂|сернист|газоочист|sulfur dioxide", "SO2_removal"),
                (r"католит|catholyte|electrowinning|электроэкстрак", "nickel_electrowinning"),
                (r"штейн|matte|slag|шлак|мпг|pgm|platinum group", "matte_slag_partitioning"),
                (r"смешан.*гидроксид|mixed hydroxide|\bmhp\b", "mixed_hydroxides"),
                (r"шахтн.*вод|mine water", "mine_water"),
                (r"сейсмограмм|велосиграмм|акселерограмм|сейсмовзрыв|буровзрыв|взрывн(?:ые|ых)?\s+работ|ppv|pvs|пиков.*скорост|скорост.*смещ|динамическ.*расчет|динамическ.*расчёт", "blasting_seismics"),
            ]
            for pattern, candidate in content_rules:
                if re.search(pattern, preview, flags=re.IGNORECASE):
                    theme, evidence = candidate, ["content_preview:domain_pattern"]
                    break
            if not theme and self.auto_theme_from_content_terms:
                try:
                    matches = get_ontology().find_terms(preview[:2500])[:8]
                    if matches:
                        # Weak auto theme only; collapsed by min_confidence in coarse/nightly mode.
                        theme = str(matches[0].canonical_en or matches[0].term_id)
                        evidence = ["content_preview:ontology_terms"]
                except Exception:
                    pass

        if theme:
            inferred_collection = collection
            if not inferred_collection and theme == "blasting_seismics":
                inferred_collection = "геотехника"
            if evidence and evidence[0].startswith("filename:auto"):
                confidence = 0.60
            elif evidence and evidence[0].startswith("content_preview:ontology_terms"):
                confidence = 0.55
            else:
                confidence = 0.75
            return theme, inferred_collection, None, evidence, confidence
        return self.default_theme, collection or self.default_collection, None, ["fallback:default_theme"], 0.25

    def resolve(self, relative_path: str, *, preview_text: str | None = None) -> ThemeInfo:
        clean = (relative_path or "").replace("\\", "/").strip().lstrip("/")
        parts = self._normalize_parts(_split_relative_path(clean))
        clean = "/".join(parts)
        source_name = parts[-1] if parts else "document"
        preview_text = preview_text or ""

        if not self.enabled:
            return ThemeInfo(
                theme_id=safe_theme_id("global", "default"),
                collection="global",
                theme_name="default",
                relative_path=clean or source_name,
                source_name=source_name,
                year=None,
                source_type="global",
                confidence=1.0,
                evidence=["topic_sharding_disabled"],
            )

        years = _path_years([*parts, source_name])
        year = years[0] if years else None
        evidence: list[str] = []
        confidence = 0.25
        haystack = f"{clean}\n{preview_text[:self.content_preview_chars]}"

        override = self._match_override(haystack)
        if override:
            collection = str(override.get("collection") or self.default_collection)
            theme_name = str(override.get("theme_name") or self.default_theme)
            source_type = str(override.get("source_type") or self._source_type(collection, source_name, haystack))
            return ThemeInfo(
                theme_id=safe_theme_id(collection, theme_name),
                collection=collection,
                theme_name=theme_name,
                relative_path=clean or source_name,
                source_name=source_name,
                year=year,
                source_type=source_type,
                confidence=0.98,
                evidence=["override_pattern", *(["year_detected"] if year else [])],
            )

        collection: str | None = None
        collection_idx: int | None = None
        if self.use_path:
            for idx, part in enumerate(parts[:-1] if len(parts) > 1 else parts):
                candidate = self._collection_for_segment(part)
                if candidate:
                    collection = candidate
                    collection_idx = idx
                    evidence.append(f"path:collection:{part}")
                    confidence = max(confidence, 0.55)
                    break

        if not collection:
            inferred_theme, inferred_collection, inferred_source_type, inf_evidence, inf_conf = self._infer_theme_from_filename_or_content(source_name, preview_text, None)
            collection = inferred_collection or self.default_collection
            evidence.extend(inf_evidence)
            confidence = max(confidence, min(inf_conf, 0.65))
        else:
            inferred_source_type = None

        theme_name = ""
        if self.use_path and collection_idx is not None:
            path_theme_candidates = []
            for part in parts[collection_idx + 1:-1]:
                if self._collection_for_segment(part):
                    continue
                if _is_generic_segment(part, self.year_re):
                    continue
                path_theme_candidates.append(part)
            if path_theme_candidates:
                theme_name = path_theme_candidates[0]
                evidence.append(f"path:theme:{theme_name}")
                confidence = max(confidence, 0.85)

        if not theme_name:
            theme_name, inferred_collection, inferred_source_type, inf_evidence, inf_conf = self._infer_theme_from_filename_or_content(source_name, preview_text, collection)
            if inferred_collection and collection == self.default_collection:
                collection = inferred_collection
            evidence.extend(inf_evidence)
            confidence = max(confidence, inf_conf)

        if not theme_name:
            theme_name = self.default_theme
            evidence.append("fallback:default_theme")

        if year:
            evidence.append("year_detected")
            confidence = min(0.99, confidence + 0.03)

        theme_name, confidence, evidence = self._collapse_if_low_confidence(collection, theme_name, confidence, evidence)
        source_type = inferred_source_type or self._source_type(collection, source_name, haystack)
        theme_id = safe_theme_id(collection, theme_name)
        return ThemeInfo(
            theme_id=theme_id,
            collection=collection,
            theme_name=theme_name,
            relative_path=clean or source_name,
            source_name=source_name,
            year=year,
            source_type=source_type,
            confidence=round(float(confidence), 3),
            evidence=list(dict.fromkeys(evidence)),
        )


class ThemeCatalog:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.root / "themes.jsonl"
        self.router_path = self.root / "global_router.json"
        self.theme_embeddings_path = self.root / "theme_embeddings.jsonl"

    def upsert_theme(self, theme: ThemeInfo, stats: dict[str, Any], *, status: str = "parsed_ready") -> None:
        now = utc_now_iso()
        rows = [row for row in jsonl_read(self.catalog_path) if row.get("theme_id") != theme.theme_id]
        previous = next((row for row in jsonl_read(self.catalog_path) if row.get("theme_id") == theme.theme_id), {})
        raw_years = [*(_as_list((previous.get("stats") or {}).get("years"))), *(_as_list(stats.get("years")))]
        years = sorted({int(y) for y in raw_years if str(y).isdigit()})
        if years:
            stats["years"] = years
        rows.append({
            **theme.to_dict(),
            "status": status,
            "stats": stats,
            "updated_at": now,
        })
        self.catalog_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )
        self.build_router()

    def list_themes(self) -> list[dict[str, Any]]:
        rows = jsonl_read(self.catalog_path)
        rows.sort(key=lambda row: (str(row.get("collection") or ""), str(row.get("theme_name") or ""), str(row.get("theme_id") or "")))
        return rows

    def build_router(self) -> dict[str, Any]:
        themes = self.list_themes()
        payload = {
            "router_type": "NiCoHybridGlobalThemeRouter",
            "updated_at": utc_now_iso(),
            "themes": [
                {
                    "theme_id": row.get("theme_id"),
                    "collection": row.get("collection"),
                    "theme_name": row.get("theme_name"),
                    "source_type": row.get("source_type"),
                    "year": row.get("year"),
                    "confidence": row.get("confidence"),
                    "evidence": row.get("evidence") or [],
                    "status": row.get("status"),
                    "stats": row.get("stats") or {},
                }
                for row in themes
            ],
        }
        self.router_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    def _theme_text(self, row: dict[str, Any]) -> str:
        stats = row.get("stats") or {}
        values: list[str] = [
            str(row.get("theme_id") or ""),
            str(row.get("collection") or ""),
            str(row.get("theme_name") or ""),
            str(row.get("source_type") or ""),
            " ".join(str(x) for x in _as_list(stats.get("top_terms"))),
            " ".join(str(x) for x in _as_list(stats.get("top_entities"))),
            " ".join(str(x) for x in _as_list(stats.get("top_processes"))),
            " ".join(str(x) for x in _as_list(stats.get("years"))),
            str(row.get("year") or ""),
        ]
        return _norm(" ".join(values))

    def route_scores(
        self,
        message: str,
        *,
        max_themes: int = 3,
        min_readiness: str = "search_ready",
        min_score: float = 0.10,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        rows = self.list_themes()
        if not rows:
            return []
        min_rank = _READINESS_ORDER.get(min_readiness, 2)
        scores = route_theme_scores(
            message,
            rows,
            global_dir=self.root,
            query_embedding=query_embedding,
            max_themes=max_themes,
            min_readiness_rank=min_rank,
            readiness_order=_READINESS_ORDER,
            min_score=min_score,
        )
        if scores:
            return scores
        # Fallback: return first ready themes with a low synthetic score.
        ready_rows = [row for row in rows if _READINESS_ORDER.get(str(row.get("status") or "not_ready"), 0) >= min_rank]
        return [
            {
                "theme_id": str(row.get("theme_id")),
                "score": 0.0,
                "vector": 0.0,
                "ontology": 0.0,
                "keyword": 0.0,
                "centrality": 0.0,
                "metadata": 0.0,
                "status": row.get("status"),
                "collection": row.get("collection"),
                "theme_name": row.get("theme_name"),
                "reason": "fallback_ready_theme",
            }
            for row in ready_rows[:max_themes]
            if row.get("theme_id")
        ]

    def route(
        self,
        message: str,
        *,
        max_themes: int = 3,
        min_readiness: str = "search_ready",
        min_score: float = 0.10,
        query_embedding: list[float] | None = None,
    ) -> list[str]:
        return [
            str(row.get("theme_id"))
            for row in self.route_scores(
                message,
                max_themes=max_themes,
                min_readiness=min_readiness,
                min_score=min_score,
                query_embedding=query_embedding,
            )
            if row.get("theme_id")
        ]


class ThemeShardManager:
    """Manage per-theme KnowledgeStore + per-theme LightRAG runtime indexes."""

    def __init__(self, root: Path, config: AppConfig) -> None:
        self.root = root
        self.config = config
        self.knowledge_root = (root / config.knowledge_store_dir).resolve() if not Path(config.knowledge_store_dir).is_absolute() else Path(config.knowledge_store_dir)
        self.runtime_root = (root / config.runtime_rag_dir).resolve() if not Path(config.runtime_rag_dir).is_absolute() else Path(config.runtime_rag_dir)
        self.themes_store_root = self.knowledge_root / "themes"
        self.themes_runtime_root = self.runtime_root / "themes"
        self.catalog = ThemeCatalog(self.knowledge_root / "global")
        self.resolver = ThemeResolver.from_config(config)
        self._services: dict[str, "RagService"] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def update_config(self, config: AppConfig) -> None:
        if config != self.config:
            self.config = config
            self.resolver = ThemeResolver.from_config(config)

    def resolve_theme(self, relative_path: str, *, preview_text: str | None = None) -> ThemeInfo:
        return self.resolver.resolve(relative_path, preview_text=preview_text)

    def theme_store_dir(self, theme_id: str) -> Path:
        return self.themes_store_root / slugify_theme_part(theme_id)

    def theme_runtime_dir(self, theme_id: str) -> Path:
        return self.themes_runtime_root / slugify_theme_part(theme_id)

    def theme_store(self, theme_id: str) -> KnowledgeStore:
        return KnowledgeStore(
            self.theme_store_dir(theme_id),
            schema_version=self.config.schema_version,
            ontology_version=self.config.ontology_version,
            app_version=self.config.app_version,
        )

    async def service_for_theme(self, theme_id: str) -> "RagService":
        service = self._services.get(theme_id)
        if service is None:
            from backend.rag_service import RagService
            service = RagService(self.theme_runtime_dir(theme_id), self.theme_store_dir(theme_id))
            self._services[theme_id] = service
            self._locks[theme_id] = asyncio.Lock()
        async with self._locks[theme_id]:
            await service.ensure_initialized(self.config)
        return service

    async def shutdown(self) -> None:
        for service in list(self._services.values()):
            await service.shutdown()
        self._services.clear()

    def apply_theme_metadata(self, processed: ProcessedDocument, theme: ThemeInfo) -> ProcessedDocument:
        for obj in processed.objects:
            obj.metadata.update({
                "theme_id": theme.theme_id,
                "collection": theme.collection,
                "theme_name": theme.theme_name,
                "relative_path": theme.relative_path,
                "year": theme.year,
                "source_type": theme.source_type,
                "theme_confidence": theme.confidence,
                "theme_evidence": ",".join(theme.evidence),
            })
        return processed

    async def ingest_processed_document(
        self,
        processed: ProcessedDocument,
        numeric_facts: list[NumericFact] | list[dict[str, Any]],
        *,
        original_path: str,
        build_runtime_index: bool = True,
        compute_graph_metrics: bool | None = None,
        status_if_no_runtime: str = "parsed_ready",
    ) -> dict[str, Any]:
        preview = "\n".join(obj.text for obj in processed.objects[:3])[: self.config.theme_content_preview_chars]
        theme = self.resolve_theme(original_path, preview_text=preview)
        self.apply_theme_metadata(processed, theme)
        store = self.theme_store(theme.theme_id)
        store_meta = store.upsert_processed_document(
            processed,
            numeric_facts,
            original_path=original_path,
            theme=theme.to_dict(),
        )
        should_compute_graph_metrics = self.config.graph_metrics_enabled if compute_graph_metrics is None else bool(compute_graph_metrics)
        if should_compute_graph_metrics:
            try:
                write_graph_metrics(
                    store,
                    top_n=self.config.graph_metrics_top_n,
                    pagerank_iterations=self.config.graph_pagerank_iterations,
                    betweenness_sample=self.config.graph_betweenness_sample,
                )
            except Exception as exc:
                print(f"[warn] graph metrics failed for {theme.theme_id}: {exc}")

        enqueue: dict[str, Any] = {"track_id": "", "items": []}
        if build_runtime_index:
            service = await self.service_for_theme(theme.theme_id)
            enqueue = await service.enqueue_documents_batch(
                [(obj.to_lightrag_text(), obj.citation_name) for obj in processed.objects]
            )
            status = "search_ready" if enqueue.get("track_id") else "parsed_ready"
        else:
            status = status_if_no_runtime

        self.catalog.upsert_theme(theme, self._theme_stats_for_catalog(store), status=status)
        return {
            "theme": theme.to_dict(),
            "store_meta": store_meta,
            "runtime_index_requested": bool(build_runtime_index),
            "track_id": enqueue.get("track_id") or "",
            "items": enqueue.get("items") or [],
        }

    def _theme_stats_for_catalog(self, store: KnowledgeStore) -> dict[str, Any]:
        stats = store.stats()
        entities = jsonl_read(store.entities_path)
        sources = jsonl_read(store.sources_path)
        top_terms = [str(row.get("label") or row.get("term_id")) for row in entities[:60] if row.get("label") or row.get("term_id")]
        years = sorted({int(row.get("year")) for row in sources if str(row.get("year") or "").isdigit()})
        source_types = list(dict.fromkeys(str(row.get("source_type")) for row in sources if row.get("source_type")))
        collections = list(dict.fromkeys(str(row.get("collection")) for row in sources if row.get("collection")))
        theme_names = list(dict.fromkeys(str(row.get("theme_name")) for row in sources if row.get("theme_name")))
        metrics = load_graph_metrics(store.root)
        top_pagerank = [str(row.get("label") or row.get("node_id")) for row in (metrics.get("top_pagerank") or [])]
        top_betweenness = [str(row.get("label") or row.get("node_id")) for row in (metrics.get("top_betweenness") or [])]
        cheap_kg = {}
        compressed_plan = {}
        try:
            cheap_path = store.root / "cheap_kg.json"
            if cheap_path.exists():
                cheap_kg = json.loads(cheap_path.read_text(encoding="utf-8"))
        except Exception:
            cheap_kg = {}
        try:
            compressed_path = store.root / "compressed_kg_plan.json"
            if compressed_path.exists():
                compressed_plan = json.loads(compressed_path.read_text(encoding="utf-8"))
        except Exception:
            compressed_plan = {}
        stats.update({
            "top_terms": list(dict.fromkeys([*top_terms, *top_pagerank[:20], *top_betweenness[:20]])),
            "top_entities": list(dict.fromkeys([*top_pagerank[:40], *top_betweenness[:40]])),
            "top_processes": [],
            "years": years,
            "source_types": source_types,
            "collections": collections,
            "theme_names": theme_names,
            "graph_metrics": metrics,
            "cheap_kg": {"counts": cheap_kg.get("counts", {})} if cheap_kg else {},
            "compressed_kg": {
                "selected_chunks": compressed_plan.get("selected_chunks"),
                "total_chunks": compressed_plan.get("total_chunks"),
                "selection_ratio": compressed_plan.get("selection_ratio"),
            } if compressed_plan else {},
        })
        return stats

    def list_themes(self) -> list[dict[str, Any]]:
        rows = self.catalog.list_themes()
        known = {row.get("theme_id"): row for row in rows}
        known_slugs = {slugify_theme_part(str(row.get("theme_id") or "")) for row in rows if row.get("theme_id")}
        if self.themes_store_root.exists():
            for theme_dir in self.themes_store_root.iterdir():
                if not theme_dir.is_dir() or theme_dir.name in known_slugs:
                    continue
                manifest = theme_dir / "manifest.json"
                if not manifest.exists():
                    continue
                store = KnowledgeStore(theme_dir)
                # Recover theme metadata from first source if available.
                first_source = next(iter(jsonl_read(store.sources_path)), {})
                theme = ThemeInfo(
                    theme_id=str(first_source.get("theme_id") or theme_dir.name),
                    collection=str(first_source.get("collection") or "unknown"),
                    theme_name=str(first_source.get("theme_name") or theme_dir.name),
                    relative_path="",
                    source_name="",
                    year=int(first_source.get("year")) if str(first_source.get("year") or "").isdigit() else None,
                    source_type=str(first_source.get("source_type") or "unknown"),
                    confidence=float(first_source.get("confidence") or 0.0),
                    evidence=list(first_source.get("evidence") or []),
                )
                self.catalog.upsert_theme(theme, self._theme_stats_for_catalog(store), status="parsed_ready")
        return self.catalog.list_themes()

    def list_documents(self, *, limit: int = 500) -> list[dict[str, Any]]:
        """Aggregate Stage-1 sources across theme stores for the UI file list."""
        documents: list[dict[str, Any]] = []
        themes = {str(row.get("theme_id") or ""): row for row in self.list_themes()}
        if not self.themes_store_root.exists():
            return documents
        for row in themes.values():
            theme_id = str(row.get("theme_id") or "")
            if not theme_id:
                continue
            store = self.theme_store(theme_id)
            status = str(row.get("status") or "search_ready")
            chunks_by_source: dict[str, int] = {}
            for chunk in jsonl_read(store.chunks_path):
                source_id = str(chunk.get("source_id") or "")
                if source_id:
                    chunks_by_source[source_id] = chunks_by_source.get(source_id, 0) + 1
            for src in jsonl_read(store.sources_path):
                source_id = str(src.get("source_id") or src.get("source_hash") or "")
                filename = str(src.get("original_path") or src.get("source_name") or source_id or "unknown")
                documents.append({
                    "id": source_id or filename,
                    "filename": filename,
                    "file_path": filename,
                    "status": "processed" if _READINESS_ORDER.get(status, 0) >= _READINESS_ORDER["search_ready"] else status,
                    "theme_status": status,
                    "theme_id": theme_id,
                    "collection": row.get("collection"),
                    "theme_name": row.get("theme_name"),
                    "chars": int(src.get("total_chars") or 0),
                    "chunks": int(chunks_by_source.get(source_id, src.get("objects_count") or 0)),
                    "updated_at": src.get("updated_at") or "",
                    "deletable": False,
                })
        documents.sort(key=lambda item: (str(item.get("updated_at") or ""), str(item.get("filename") or "")), reverse=True)
        return documents[: max(1, int(limit))]

    def rebuild_theme_deterministic_kg(
        self,
        theme_id: str,
        *,
        graph_mode: str = "compressed_kg",
        max_chunks_per_document_for_graph: int = 6,
        min_kg_score: float = 0.15,
        max_chunks_per_theme: int | None = None,
        max_candidate_chunks_per_theme: int | None = None,
        target_status: str | None = None,
        compute_graph_metrics: bool = False,
        compressed_runtime_doc_mode: str = "chunk",
        compressed_min_chunk_quality: float = 0.20,
        compressed_doc_type_aware: bool = True,
        compressed_dynamic_document_limits: bool = True,
        compressed_short_document_max_chunks: int = 3,
        compressed_long_document_max_chunks: int = 6,
    ) -> dict[str, Any]:
        """Build Stage-2 deterministic KG artefacts without touching LightRAG.

        This method is intentionally synchronous so the web layer can run it in
        a worker thread and keep `/api/health` and progress polling responsive.
        """
        graph_mode = str(graph_mode or "compressed_kg").strip().lower()
        if graph_mode not in {"vector_only", "cheap_kg", "compressed_kg", "retrieval_kg"}:
            raise ValueError(f"Deterministic Stage 2 does not support graph_mode={graph_mode}")
        store = self.theme_store(theme_id)
        if not store.root.exists():
            raise RuntimeError(f"theme knowledge_store not found: {theme_id}")
        theme = self._theme_from_store(theme_id)

        if graph_mode == "vector_only":
            status = target_status or "search_ready"
            self._clear_runtime_ready_marker(theme_id)
            self.catalog.upsert_theme(theme, self._theme_stats_for_catalog(store), status=status)
            return {"theme_id": theme_id, "status": status, "graph_mode": graph_mode, "track_ids": [], "chunks": store.stats().get("chunks", 0), "runtime_index_requested": False}

        if graph_mode == "retrieval_kg":
            retrieval_payload = build_retrieval_kg(store)
            status = target_status or "retrieval_kg_ready"
            self._clear_runtime_ready_marker(theme_id)
            self.catalog.upsert_theme(theme, self._theme_stats_for_catalog(store), status=status)
            return {
                "theme_id": theme_id,
                "status": status,
                "graph_mode": graph_mode,
                "track_ids": [],
                "chunks": int((retrieval_payload.get("counts") or {}).get("retrieval_rows") or 0),
                "retrieval_kg_path": str(store.root / "retrieval_kg.json"),
                "retrieval_index_path": str(store.root / "retrieval_index.jsonl"),
                "runtime_index_requested": False,
                "message": "retrieval_kg built from knowledge_store; LightRAG runtime was skipped.",
            }

        cheap_payload = build_cheap_kg(store)
        retrieval_payload: dict[str, Any] | None = None
        if graph_mode in {"compressed_kg"}:
            # Always build retrieval_kg artefacts as the guaranteed answer layer.
            # Compressed LightRAG can be empty/slow, but retrieval_index.jsonl keeps
            # RAG-style answers available through deterministic search.
            retrieval_payload = build_retrieval_kg(store)
        if graph_mode == "cheap_kg":
            status = target_status or "cheap_kg_ready"
            if compute_graph_metrics and self.config.graph_metrics_enabled:
                write_graph_metrics(store, top_n=self.config.graph_metrics_top_n, pagerank_iterations=self.config.graph_pagerank_iterations, betweenness_sample=self.config.graph_betweenness_sample)
            self._clear_runtime_ready_marker(theme_id)
            self.catalog.upsert_theme(theme, self._theme_stats_for_catalog(store), status=status)
            return {
                "theme_id": theme_id,
                "status": status,
                "graph_mode": graph_mode,
                "track_ids": [],
                "chunks": store.stats().get("chunks", 0),
                "cheap_kg": cheap_payload,
                "runtime_index_requested": False,
                "message": "cheap_kg built deterministically; LightRAG was skipped.",
            }

        max_theme = None if not max_chunks_per_theme or int(max_chunks_per_theme) <= 0 else int(max_chunks_per_theme)
        compressed_plan = build_compressed_kg_plan(
            store,
            cheap_payload=cheap_payload,
            max_chunks_per_document=max_chunks_per_document_for_graph,
            min_kg_score=min_kg_score,
            max_chunks_per_theme=max_theme,
            max_candidate_chunks_per_theme=max_candidate_chunks_per_theme,
            min_chunk_quality=compressed_min_chunk_quality,
            doc_type_aware=compressed_doc_type_aware,
            dynamic_document_limits=compressed_dynamic_document_limits,
            short_document_max_chunks=compressed_short_document_max_chunks,
            long_document_max_chunks=compressed_long_document_max_chunks,
        )
        selected_items = lightrag_items_from_selected_rows(
            compressed_plan.get("rows") or [],
            mode=compressed_runtime_doc_mode,
        )
        status = target_status or "compressed_kg_ready"
        if not selected_items:
            status = "cheap_kg_ready"
        if compute_graph_metrics and self.config.graph_metrics_enabled:
            write_graph_metrics(store, top_n=self.config.graph_metrics_top_n, pagerank_iterations=self.config.graph_pagerank_iterations, betweenness_sample=self.config.graph_betweenness_sample)
        self._clear_runtime_ready_marker(theme_id)
        self.catalog.upsert_theme(theme, self._theme_stats_for_catalog(store), status=status)
        return {
            "theme_id": theme_id,
            "status": status,
            "graph_mode": graph_mode,
            "track_ids": [],
            "chunks": len(selected_items),
            "cheap_kg_path": str(store.root / "cheap_kg.json"),
            "retrieval_kg_path": str(store.root / "retrieval_kg.json") if retrieval_payload else "",
            "retrieval_index_path": str(store.root / "retrieval_index.jsonl") if retrieval_payload else "",
            "compressed_plan": {k: v for k, v in compressed_plan.items() if k != "rows"},
            "runtime_index_requested": False,
            "message": "compressed_kg built deterministically; retrieval_index fallback is ready; LightRAG runtime was skipped.",
        }

    async def get_theme_status(self, theme_id: str) -> dict[str, Any]:
        store = self.theme_store(theme_id)
        state: dict[str, Any] = {"theme_id": theme_id, "store": store.stats(), "runtime": None}
        try:
            service = await self.service_for_theme(theme_id)
            state["runtime"] = await service.get_knowledge_state()
        except Exception as exc:
            state["runtime_error"] = str(exc)
        return state

    def _theme_from_store(self, theme_id: str) -> ThemeInfo:
        store = self.theme_store(theme_id)
        first_source = next(iter(jsonl_read(store.sources_path)), {})
        return ThemeInfo(
            theme_id=theme_id,
            collection=str(first_source.get("collection") or "unknown"),
            theme_name=str(first_source.get("theme_name") or theme_id),
            relative_path="",
            source_name="",
            year=int(first_source.get("year")) if str(first_source.get("year") or "").isdigit() else None,
            source_type=str(first_source.get("source_type") or "unknown"),
            confidence=float(first_source.get("confidence") or 0.0),
            evidence=list(first_source.get("evidence") or []),
        )

    def mark_theme_status(self, theme_id: str, status: str) -> None:
        store = self.theme_store(theme_id)
        if not store.root.exists():
            return
        self.catalog.upsert_theme(self._theme_from_store(theme_id), self._theme_stats_for_catalog(store), status=status)

    async def rebuild_theme_runtime(
        self,
        theme_id: str,
        *,
        clear_runtime: bool = False,
        batch_size: int = 64,
        wait: bool = False,
        poll_interval: float = 10.0,
        timeout_sec: float = 0.0,
        target_status: str | None = None,
        graph_mode: str = "full_kg",
        max_chunks_per_document_for_graph: int = 8,
        min_kg_score: float = 0.15,
        max_chunks_per_theme: int | None = None,
        max_candidate_chunks_per_theme: int | None = None,
        build_runtime_index: bool = True,
        compute_graph_metrics: bool = True,
        compressed_runtime_doc_mode: str = "chunk",
        compressed_min_chunk_quality: float = 0.20,
        compressed_doc_type_aware: bool = True,
        compressed_dynamic_document_limits: bool = True,
        compressed_short_document_max_chunks: int = 3,
        compressed_long_document_max_chunks: int = 6,
    ) -> dict[str, Any]:
        """Build a theme runtime graph from Stage-1 knowledge_store.

        graph_mode:
          - cheap_kg: deterministic KG only, no LightRAG calls.
          - retrieval_kg: NetworkX-style retrieval graph over knowledge_store; no LightRAG calls.
          - compressed_kg: deterministic KG + compressed selection plan. By default
            this does not call LightRAG and therefore cannot hang on final graph write.
          - full_kg: all chunks for LightRAG. Use only for selected important themes.
          - vector_only: no LightRAG; keep Stage-1 lightweight search/search_ready.
        """
        graph_mode = str(graph_mode or "full_kg").strip().lower()
        if graph_mode not in {"vector_only", "cheap_kg", "compressed_kg", "retrieval_kg", "full_kg"}:
            raise ValueError(f"Unknown graph_mode: {graph_mode}")

        store = self.theme_store(theme_id)
        if not store.root.exists():
            raise RuntimeError(f"theme knowledge_store not found: {theme_id}")
        theme = self._theme_from_store(theme_id)

        cheap_payload: dict[str, Any] | None = None
        compressed_plan: dict[str, Any] | None = None
        selected_items: list[tuple[str, str]] | None = None

        # Guaranteed deterministic graph/fallback. This is intentionally built
        # before any LightRAG call so timeout/failure still leaves useful KG files.
        retrieval_payload: dict[str, Any] | None = None
        if graph_mode in {"cheap_kg", "compressed_kg", "full_kg"}:
            cheap_payload = build_cheap_kg(store)
        if graph_mode in {"compressed_kg", "full_kg"}:
            # The deterministic retrieval layer is cheap compared with LightRAG
            # extraction and is the source-of-truth fallback for chat. Build it
            # before runtime indexing so the theme remains answerable even when
            # LightRAG writes an empty graph or query keyword extraction fails.
            retrieval_payload = build_retrieval_kg(store)

        if graph_mode == "vector_only":
            self.catalog.upsert_theme(theme, self._theme_stats_for_catalog(store), status=target_status or "search_ready")
            return {
                "theme_id": theme_id,
                "status": target_status or "search_ready",
                "graph_mode": graph_mode,
                "track_ids": [],
                "chunks": store.stats().get("chunks", 0),
                "message": "vector_only/lightweight mode: LightRAG runtime was not rebuilt; Stage-1 search remains available.",
            }


        if graph_mode == "retrieval_kg":
            return self.rebuild_theme_deterministic_kg(
                theme_id,
                graph_mode="retrieval_kg",
                target_status=target_status or "retrieval_kg_ready",
                compute_graph_metrics=False,
            )

        if graph_mode == "cheap_kg":
            return self.rebuild_theme_deterministic_kg(
                theme_id,
                graph_mode="cheap_kg",
                target_status=target_status or "cheap_kg_ready",
                compute_graph_metrics=compute_graph_metrics,
            )

        if graph_mode == "compressed_kg":
            max_theme = None if not max_chunks_per_theme or int(max_chunks_per_theme) <= 0 else int(max_chunks_per_theme)
            compressed_plan = build_compressed_kg_plan(
                store,
                cheap_payload=cheap_payload,
                max_chunks_per_document=max_chunks_per_document_for_graph,
                min_kg_score=min_kg_score,
                max_chunks_per_theme=max_theme,
                max_candidate_chunks_per_theme=max_candidate_chunks_per_theme,
                min_chunk_quality=compressed_min_chunk_quality,
                doc_type_aware=compressed_doc_type_aware,
                dynamic_document_limits=compressed_dynamic_document_limits,
                short_document_max_chunks=compressed_short_document_max_chunks,
                long_document_max_chunks=compressed_long_document_max_chunks,
            )
            selected_items = lightrag_items_from_selected_rows(
                compressed_plan.get("rows") or [],
                mode=compressed_runtime_doc_mode,
            )
            if not selected_items:
                self._clear_runtime_ready_marker(theme_id)
                self.catalog.upsert_theme(theme, self._theme_stats_for_catalog(store), status="cheap_kg_ready")
                return {
                    "theme_id": theme_id,
                    "status": "cheap_kg_ready",
                    "graph_mode": graph_mode,
                    "track_ids": [],
                    "chunks": 0,
                    "cheap_kg": cheap_payload,
                    "compressed_plan": {k: v for k, v in compressed_plan.items() if k != "rows"},
                    "message": "compressed_kg found no selected chunks; kept cheap_kg_ready.",
                }
            if not build_runtime_index:
                if compute_graph_metrics and self.config.graph_metrics_enabled:
                    try:
                        write_graph_metrics(
                            store,
                            top_n=self.config.graph_metrics_top_n,
                            pagerank_iterations=self.config.graph_pagerank_iterations,
                            betweenness_sample=self.config.graph_betweenness_sample,
                        )
                    except Exception as exc:
                        print(f"[warn] graph metrics failed for {theme_id}: {exc}")
                self._clear_runtime_ready_marker(theme_id)
                status = target_status or "compressed_kg_ready"
                self.catalog.upsert_theme(theme, self._theme_stats_for_catalog(store), status=status)
                return {
                    "theme_id": theme_id,
                    "status": status,
                    "graph_mode": graph_mode,
                    "track_ids": [],
                    "chunks": len(selected_items),
                    "cheap_kg": cheap_payload,
                    "cheap_kg_path": str(store.root / "cheap_kg.json") if cheap_payload else "",
                    "compressed_plan": {k: v for k, v in compressed_plan.items() if k != "rows"},
                    "runtime_index_requested": False,
                    "message": "compressed_kg built deterministically from Stage-1 store; LightRAG runtime was skipped to avoid final graph-write hangs.",
                }

        if not build_runtime_index:
            self._clear_runtime_ready_marker(theme_id)
            self.catalog.upsert_theme(theme, self._theme_stats_for_catalog(store), status=target_status or "cheap_kg_ready")
            return {
                "theme_id": theme_id,
                "status": target_status or "cheap_kg_ready",
                "graph_mode": graph_mode,
                "track_ids": [],
                "chunks": store.stats().get("chunks", 0),
                "runtime_index_requested": False,
                "message": "LightRAG runtime was skipped by build_runtime_index=false.",
            }

        runtime_dir = self.theme_runtime_dir(theme_id)
        if clear_runtime and runtime_dir.exists():
            shutil.rmtree(runtime_dir)
            self._services.pop(theme_id, None)
        service = await self.service_for_theme(theme_id)
        result = await service.rebuild_runtime_index_from_store(
            batch_size=batch_size,
            wait=wait,
            poll_interval=poll_interval,
            timeout_sec=timeout_sec,
            items_override=selected_items,
            build_label=graph_mode,
        )

        if compute_graph_metrics and self.config.graph_metrics_enabled:
            try:
                write_graph_metrics(
                    store,
                    top_n=self.config.graph_metrics_top_n,
                    pagerank_iterations=self.config.graph_pagerank_iterations,
                    betweenness_sample=self.config.graph_betweenness_sample,
                )
            except Exception as exc:
                print(f"[warn] graph metrics failed for {theme_id}: {exc}")

        timeout_count = int(result.get("timeout_count") or 0)
        failed_count = int(result.get("failed_count") or 0)
        has_tracks = bool(result.get("track_ids"))

        runtime_stats = self._runtime_index_stats(theme_id)
        runtime_has_vector_chunks = bool(runtime_stats.get("vdb_chunks", 0) > 0)
        runtime_has_graph = bool(runtime_stats.get("graph_nodes", 0) > 0 or runtime_stats.get("graph_edges", 0) > 0)
        runtime_has_context = bool(runtime_has_vector_chunks or runtime_has_graph)

        # A compressed LightRAG build may time out at the doc-status polling level
        # even after LightRAG has already flushed vdb_chunks. The KG graph can be
        # empty (0 nodes / 0 edges) because the extraction LLM returned malformed
        # entity/relation records, but the vector chunk index is still useful for
        # LightRAG naive retrieval. Do not demote such themes to cheap_kg_ready;
        # otherwise every query permanently falls through to deterministic RAG.
        runtime_ready = bool(
            wait
            and has_tracks
            and failed_count == 0
            and graph_mode in {"compressed_kg", "full_kg"}
            and runtime_has_context
        )

        runtime_ready_note = ""
        if runtime_ready:
            status = target_status or ("full_kg_ready" if graph_mode == "full_kg" else "compressed_kg_ready")
            runtime_ready_note = "graph+vector" if runtime_has_graph else "vector_only_after_timeout" if timeout_count else "vector_only"
            self._write_runtime_ready_marker(
                theme_id,
                graph_mode=graph_mode,
                track_ids=list(result.get("track_ids") or []),
                stats=runtime_stats,
                note=runtime_ready_note,
            )
        else:
            self._clear_runtime_ready_marker(theme_id)
            if graph_mode == "compressed_kg" and wait and has_tracks and failed_count == 0 and timeout_count == 0:
                # LightRAG processed the selected documents but produced neither
                # graph nor chunk vectors. Keep deterministic retrieval answerable,
                # but do not advertise a runtime graph for query routing.
                status = target_status or "compressed_kg_ready"
            elif graph_mode == "full_kg" and wait and has_tracks and failed_count == 0 and timeout_count == 0:
                status = target_status or "full_kg_ready"
            elif graph_mode in {"compressed_kg", "full_kg"} and (timeout_count or failed_count):
                status = "cheap_kg_ready"
                if timeout_count and failed_count == 0 and has_tracks:
                    # Web mode can time out while LightRAG is still flushing chunk
                    # vectors. Keep deterministic retrieval available immediately,
                    # but continue watching the runtime in the background and
                    # promote the theme when vdb_chunks/graph files appear.
                    try:
                        asyncio.create_task(
                            self._promote_runtime_when_available(
                                theme_id,
                                graph_mode=graph_mode,
                                track_ids=list(result.get("track_ids") or []),
                                target_status=target_status,
                                poll_interval=poll_interval,
                                timeout_sec=max(float(timeout_sec or 0.0), 1800.0),
                            )
                        )
                    except Exception:
                        pass
            else:
                status = "search_ready" if has_tracks else "cheap_kg_ready"

        self.catalog.upsert_theme(theme, self._theme_stats_for_catalog(store), status=status)
        result.update({
            "theme_id": theme_id,
            "status": status,
            "graph_mode": graph_mode,
            "cheap_kg_path": str(store.root / "cheap_kg.json") if cheap_payload else "",
            "retrieval_kg_path": str(store.root / "retrieval_kg.json") if retrieval_payload else "",
            "retrieval_index_path": str(store.root / "retrieval_index.jsonl") if retrieval_payload else "",
            "compressed_plan": {k: v for k, v in (compressed_plan or {}).items() if k != "rows"},
            "runtime_index_ready": runtime_ready,
            "runtime_index_ready_note": runtime_ready_note,
            "runtime_index_stats": runtime_stats,
        })
        return result

    async def rebuild_themes_runtime(
        self,
        theme_ids: list[str],
        *,
        max_parallel_themes: int = 1,
        on_theme_start: Any | None = None,
        on_theme_done: Any | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Rebuild several theme runtimes with a bounded asyncio task pool.

        This is intentionally placed in ThemeShardManager rather than only in
        CLI code so web and batch runners use the same concurrency guard. It is
        safe for separate theme runtime directories, while the semaphore prevents
        the local LLM/embedding server from being flooded during nightly builds.
        """
        max_parallel = max(1, int(max_parallel_themes or 1))
        semaphore = asyncio.Semaphore(max_parallel)
        results: list[dict[str, Any] | None] = [None] * len(theme_ids)

        async def _maybe_call(callback: Any | None, *args: Any) -> None:
            if callback is None:
                return
            value = callback(*args)
            if asyncio.iscoroutine(value):
                await value

        async def _run_one(index: int, theme_id: str) -> None:
            async with semaphore:
                await _maybe_call(on_theme_start, index, theme_id)
                try:
                    result = await self.rebuild_theme_runtime(theme_id, **kwargs)
                    results[index] = {"theme_id": theme_id, "ok": True, **result}
                except Exception as exc:
                    results[index] = {"theme_id": theme_id, "ok": False, "error": str(exc)}
                await _maybe_call(on_theme_done, index, theme_id, results[index])

        tasks = [asyncio.create_task(_run_one(idx, theme_id)) for idx, theme_id in enumerate(theme_ids)]
        if tasks:
            await asyncio.gather(*tasks)
        return [row for row in results if row is not None]


    async def _promote_runtime_when_available(
        self,
        theme_id: str,
        *,
        graph_mode: str,
        track_ids: list[str],
        target_status: str | None = None,
        poll_interval: float = 10.0,
        timeout_sec: float = 1800.0,
    ) -> None:
        """Promote a timed-out LightRAG runtime when it becomes queryable.

        LightRAG sometimes keeps processing after our per-track wait timed out.
        If chunk vectors or graph edges appear later, the theme should stop
        falling back permanently to deterministic retrieval. This method is
        best-effort and intentionally silent except for a compact log line.
        """
        start = asyncio.get_running_loop().time()
        while True:
            stats = self._runtime_index_stats(theme_id)
            if stats.get("vdb_chunks", 0) > 0 or stats.get("graph_nodes", 0) > 0 or stats.get("graph_edges", 0) > 0:
                status = target_status or ("full_kg_ready" if graph_mode == "full_kg" else "compressed_kg_ready")
                self._write_runtime_ready_marker(
                    theme_id,
                    graph_mode=graph_mode,
                    track_ids=track_ids,
                    stats=stats,
                    note="promoted_after_timeout",
                )
                try:
                    theme = self._theme_from_store(theme_id)
                    self.catalog.upsert_theme(theme, self._theme_stats_for_catalog(self.theme_store(theme_id)), status=status)
                except Exception:
                    pass
                print(f"[stage2] promoted runtime after timeout: theme={theme_id} status={status} stats={stats}")
                return
            if timeout_sec and (asyncio.get_running_loop().time() - start) >= timeout_sec:
                return
            await asyncio.sleep(max(5.0, float(poll_interval or 10.0)))


    async def get_track_status(self, track_id: str) -> dict[str, Any]:
        last_error = None
        for theme_id, service in list(self._services.items()):
            try:
                status = await service.get_track_status(track_id)
                status["theme_id"] = theme_id
                return status
            except Exception as exc:
                last_error = exc
        for row in self.list_themes():
            theme_id = str(row.get("theme_id") or "")
            if not theme_id or theme_id in self._services:
                continue
            try:
                service = await self.service_for_theme(theme_id)
                status = await service.get_track_status(track_id)
                status["theme_id"] = theme_id
                return status
            except Exception as exc:
                last_error = exc
        raise RuntimeError(str(last_error) if last_error else f"track_id not found: {track_id}")

    async def knowledge_state(self) -> dict[str, Any]:
        themes = self.list_themes()
        ready = 0
        stage2_ready = 0
        runtime_ready = 0
        failed = 0
        max_status = "not_ready"
        max_rank = _READINESS_ORDER["not_ready"]
        for row in themes:
            theme_id = str(row.get("theme_id") or "")
            status = str(row.get("status") or "not_ready")
            rank = _READINESS_ORDER.get(status, 0)
            if rank > max_rank:
                max_rank = rank
                max_status = status
            if rank >= _READINESS_ORDER["search_ready"]:
                ready += 1
            if rank >= _READINESS_ORDER["cheap_kg_ready"]:
                stage2_ready += 1
            if theme_id and self._has_runtime_index(theme_id):
                runtime_ready += 1
            if status == "failed":
                failed += 1
        return {
            "knowledge_ready": ready > 0,
            "stage2_ready": stage2_ready > 0,
            "stage2_ready_theme_count": stage2_ready,
            "runtime_ready_theme_count": runtime_ready,
            "max_theme_status": max_status,
            "document_count": sum(int((row.get("stats") or {}).get("sources") or 0) for row in themes),
            "processing_count": 0,
            "failed_count": failed,
            "pipeline_busy": False,
            "themes": themes,
        }

    def compute_theme_graph_metrics(self, theme_id: str | None = None, theme_ids: list[str] | None = None) -> dict[str, Any]:
        results: dict[str, Any] = {}
        rows = self.list_themes()
        if theme_ids:
            selected = {str(tid) for tid in theme_ids if tid}
        else:
            selected = {theme_id} if theme_id else {str(row.get("theme_id")) for row in rows if row.get("theme_id")}
        for tid in selected:
            store = self.theme_store(tid)
            if not store.root.exists():
                results[tid] = {"error": "theme store not found"}
                continue
            try:
                metrics = write_graph_metrics(
                    store,
                    top_n=self.config.graph_metrics_top_n,
                    pagerank_iterations=self.config.graph_pagerank_iterations,
                    betweenness_sample=self.config.graph_betweenness_sample,
                )
                results[tid] = metrics
                first_source = next(iter(jsonl_read(store.sources_path)), {})
                theme = ThemeInfo(
                    theme_id=tid,
                    collection=str(first_source.get("collection") or "unknown"),
                    theme_name=str(first_source.get("theme_name") or tid),
                    relative_path="",
                    source_name="",
                    year=int(first_source.get("year")) if str(first_source.get("year") or "").isdigit() else None,
                    source_type=str(first_source.get("source_type") or "unknown"),
                    confidence=float(first_source.get("confidence") or 0.0),
                    evidence=list(first_source.get("evidence") or []),
                )
                self.catalog.upsert_theme(theme, self._theme_stats_for_catalog(store), status=str((next((r for r in rows if r.get("theme_id") == tid), {}) or {}).get("status") or "parsed_ready"))
            except Exception as exc:
                results[tid] = {"error": str(exc)}
        return {"themes": results}

    async def rebuild_theme_embeddings(self, theme_ids: list[str] | None = None) -> dict[str, Any]:
        if not self.config.theme_embeddings_enabled:
            return {"updated": 0, "skipped": True, "reason": "theme_embeddings disabled in config"}
        return await build_theme_embeddings(
            project_root=self.root,
            config=self.config,
            theme_ids=theme_ids,
            max_chunks=self.config.theme_embeddings_max_chunks_per_theme,
        )

    async def debug_route(
        self,
        message: str,
        *,
        max_themes: int | None = None,
        min_readiness: str = "search_ready",
    ) -> dict[str, Any]:
        query_embedding = None
        embedding_error = ""
        if self.config.theme_embeddings_enabled and self.config.routing_use_theme_embeddings:
            try:
                vectors = await embed_texts_openai_compatible([message], config=self.config)
                query_embedding = vectors[0] if vectors else None
            except Exception as exc:
                embedding_error = str(exc)
        scores = self.catalog.route_scores(
            message,
            max_themes=max_themes or self.config.routing_top_k_themes,
            min_readiness=min_readiness,
            min_score=self.config.routing_min_theme_score,
            query_embedding=query_embedding,
        )
        return {"message": message, "embedding_error": embedding_error, "routes": scores}

    def _catalog_row_by_theme(self, theme_id: str) -> dict[str, Any]:
        for row in self.list_themes():
            if str(row.get("theme_id") or "") == theme_id:
                return row
        return {}

    def _runtime_ready_marker(self, theme_id: str) -> Path:
        return self.theme_runtime_dir(theme_id) / "nico_runtime_ready.json"

    def _write_runtime_ready_marker(
        self,
        theme_id: str,
        *,
        graph_mode: str,
        track_ids: list[str],
        stats: dict[str, int] | None = None,
        note: str = "",
    ) -> None:
        marker = self._runtime_ready_marker(theme_id)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "theme_id": theme_id,
                    "graph_mode": graph_mode,
                    "track_ids": track_ids,
                    "runtime_stats": stats or {},
                    "note": note,
                    "created_at": utc_now_iso(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _clear_runtime_ready_marker(self, theme_id: str) -> None:
        try:
            self._runtime_ready_marker(theme_id).unlink(missing_ok=True)
        except Exception:
            pass

    def _runtime_index_stats(self, theme_id: str) -> dict[str, int]:
        """Return cheap health stats for a theme LightRAG runtime directory.

        A runtime directory may contain non-empty files even when LightRAG wrote
        an empty graph and later KG/vector query cannot build context. Treating
        such a directory as fully query-ready makes it hijack the deterministic
        retrieval_kg fallback. These lightweight counters are intentionally
        best-effort and do not import LightRAG internals.
        """
        runtime_dir = self.theme_runtime_dir(theme_id)
        stats = {"files": 0, "graph_nodes": 0, "graph_edges": 0, "vdb_chunks": 0, "doc_status": 0}
        if not runtime_dir.exists():
            return stats
        try:
            stats["files"] = sum(1 for path in runtime_dir.rglob("*") if path.is_file())
        except Exception:
            pass

        graph_path = runtime_dir / "graph_chunk_entity_relation.graphml"
        if graph_path.exists():
            try:
                root = ET.parse(graph_path).getroot()
                stats["graph_nodes"] = len(root.findall(".//{*}node"))
                stats["graph_edges"] = len(root.findall(".//{*}edge"))
            except Exception:
                try:
                    text = graph_path.read_text(encoding="utf-8", errors="ignore")
                    stats["graph_nodes"] = text.count("<node")
                    stats["graph_edges"] = text.count("<edge")
                except Exception:
                    pass

        for name, key in (("vdb_chunks.json", "vdb_chunks"), ("kv_store_doc_status.json", "doc_status"), ("doc_status.json", "doc_status")):
            path = runtime_dir / name
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    if isinstance(data.get("data"), list):
                        stats[key] = max(stats[key], len(data["data"]))
                    elif isinstance(data.get("data"), dict):
                        stats[key] = max(stats[key], len(data["data"]))
                    else:
                        stats[key] = max(stats[key], len(data))
                elif isinstance(data, list):
                    stats[key] = max(stats[key], len(data))
            except Exception:
                pass
        return stats

    def _has_runtime_index(self, theme_id: str) -> bool:
        runtime_dir = self.theme_runtime_dir(theme_id)
        marker = self._runtime_ready_marker(theme_id)
        if not runtime_dir.exists() or not marker.exists():
            return False
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if payload.get("theme_id") != theme_id:
                return False
            stats = self._runtime_index_stats(theme_id)
            # Trust a runtime only if it has at least retrievable chunk vectors or
            # KG nodes/edges. Empty graph files and doc-status files alone are not
            # enough: they cause LightRAG to return no-result while suppressing the
            # deterministic retrieval_kg fallback.
            return bool(stats.get("vdb_chunks", 0) > 0 or stats.get("graph_nodes", 0) > 0 or stats.get("graph_edges", 0) > 0)
        except Exception:
            return False


    def _query_theme_lightweight(self, theme_id: str, message: str, *, limit: int = 8) -> dict[str, Any]:
        store = self.theme_store(theme_id)
        matches = search_theme_store(store, message, limit=limit)
        return {"theme_id": theme_id, "matches": matches}

    def _route_by_lightweight_retrieval(
        self,
        message: str,
        *,
        max_themes: int,
        min_readiness: str = "search_ready",
    ) -> list[dict[str, Any]]:
        """Route by actual chunk/retrieval hits when global router has no signal.

        This avoids the old behaviour where the router returned the first ready
        themes with score=0.0 and reason=fallback_ready_theme. The fallback is
        still deterministic, but it is now grounded in retrieval scores from
        each theme's knowledge_store/retrieval_index.jsonl.
        """
        min_rank = _READINESS_ORDER.get(min_readiness, _READINESS_ORDER["search_ready"])
        candidates: list[dict[str, Any]] = []
        for row in self.list_themes():
            theme_id = str(row.get("theme_id") or "")
            if not theme_id:
                continue
            if _READINESS_ORDER.get(str(row.get("status") or "not_ready"), 0) < min_rank:
                continue
            try:
                matches = search_theme_store(self.theme_store(theme_id), message, limit=1)
            except Exception:
                matches = []
            if not matches:
                continue
            top = matches[0]
            score = float(top.get("score") or 0.0)
            if score <= 0:
                continue
            candidates.append({
                "theme_id": theme_id,
                "score": round(score, 4),
                "vector": 0.0,
                "ontology": round(float(top.get("graph_boost") or 0.0), 4),
                "keyword": round(score, 4),
                "centrality": round(float(top.get("graph_boost") or 0.0), 4),
                "metadata": 0.0,
                "status": row.get("status"),
                "collection": row.get("collection"),
                "theme_name": row.get("theme_name"),
                "reason": "retrieval_kg_fallback",
            })
        candidates.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        return candidates[: max(1, int(max_themes))]

    async def _synthesize_lightweight_answer(
        self,
        message: str,
        lightweight_results: list[dict[str, Any]],
        *,
        stage2_ready: bool,
        runtime_ready: bool,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        payload = build_lightweight_context(
            message,
            lightweight_results,
            max_items=10,
            max_context_chars=9000,
        )
        if not payload.get("has_context"):
            return {"answer": "", "sources": payload.get("sources") or []}

        context = str(payload.get("context") or "")
        themes = ", ".join(str(t) for t in payload.get("themes") or [])
        history_tail = ""
        if history:
            normalized = []
            for item in history[-6:]:
                role = str(item.get("role") or "").strip()
                content = str(item.get("content") or "").strip()
                if role in {"user", "assistant"} and content:
                    normalized.append(f"{role}: {content[:700]}")
            history_tail = "\n".join(normalized)

        prompt = (
            "Сформируй RAG-ответ по найденному контексту из детерминированного retrieval_kg.\n"
            "Не перечисляй чанки и не показывай router/debug scores. Синтезируй инженерный ответ.\n"
            "Обязательно отделяй факты от ограничений. Для чисел сохраняй единицы измерения ровно как в источнике.\n"
            "Ставь ссылки в виде [1], [2] после утверждений, если соответствующий номер есть в CONTEXT.\n"
            "Если данных недостаточно, прямо напиши: В источниках недостаточно данных.\n\n"
            f"THEMES: {themes}\n"
            f"RUNTIME_READY: {runtime_ready}\n"
            f"STAGE2_READY: {stage2_ready}\n\n"
            f"HISTORY:\n{history_tail}\n\n"
            f"USER_QUESTION:\n{message}\n\n"
            f"CONTEXT:\n{context}\n\n"
            "Формат ответа:\n"
            "### Краткий вывод\n"
            "### Найденные факты\n"
            "### Ограничения и пробелы\n"
        )
        system_prompt = (
            "Ты — ◈NiCo, R&D ассистент для горно-металлургической карты знаний. "
            "Отвечай на русском языке, технически точно и только по CONTEXT. "
            "Роль: экспертный совет SME-гидрометаллург/онтолог/graph DB/backend. "
            "Не выдумывай численные значения, материалы, процессы, годы, источники. "
            "Не вставляй служебные трассы маршрутизатора и не называй ответ перечнем чанков."
        )
        try:
            from lightrag.llm.openai import openai_complete_if_cache

            api_key = (self.config.llm_api_key or "").strip() or "EMPTY"
            raw = await openai_complete_if_cache(
                self.config.llm_model,
                prompt,
                system_prompt=system_prompt,
                api_key=api_key,
                base_url=self.config.llm_base_url.rstrip("/"),
                max_tokens=900,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            answer = re.sub(r"\s+\n", "\n", str(raw or "")).strip()
            if answer:
                return {"answer": answer, "sources": payload.get("sources") or []}
        except Exception as exc:
            print(f"[warn] retrieval_kg LLM synthesis failed, fallback to deterministic formatter: {exc}")
        return format_lightweight_answer(
            message,
            lightweight_results,
            stage2_ready=stage2_ready,
            runtime_ready=runtime_ready,
        )

    @staticmethod
    def _is_no_result_answer(answer: str, sources: list[dict[str, Any]] | None = None) -> bool:
        low = re.sub(r"\s+", " ", (answer or "").lower().replace("ё", "е")).strip()
        if not low:
            return True
        no_source = not bool(sources)
        hard_markers = (
            "sorry, i'm not able to provide an answer",
            "sorry, i am not able to provide an answer",
            "no query context could be built",
            "returning no-result",
            "no-result",
            "[no-context]",
            "ответ не получен",
            "контекст не найден",
            "не удалось построить контекст",
            "релевантные сведения не найдены",
            "в источниках недостаточно данных",
        )
        if any(marker in low for marker in hard_markers):
            return True
        if no_source and re.search(r"\b(?:не найден[оы]?|недостаточно данных|не могу ответить|cannot answer|unable to answer)\b", low):
            return True
        return False

    async def query(
        self,
        message: str,
        *,
        mode: str,
        history: list[dict[str, str]] | None = None,
        theme_ids: list[str] | None = None,
        routing: str = "auto",
        max_themes: int | None = None,
        min_readiness: str = "search_ready",
    ) -> dict[str, Any]:
        max_themes = max_themes or self.config.routing_top_k_themes
        selected = list(dict.fromkeys(theme_ids or []))
        route_scores_debug: list[dict[str, Any]] = []
        if not selected and routing == "auto":
            query_embedding = None
            if self.config.theme_embeddings_enabled and self.config.routing_use_theme_embeddings:
                try:
                    vectors = await embed_texts_openai_compatible([message], config=self.config)
                    query_embedding = vectors[0] if vectors else None
                except Exception as exc:
                    print(f"[warn] theme embedding routing failed, fallback to lexical routing: {exc}")
            route_scores_debug = self.catalog.route_scores(
                message,
                max_themes=max_themes,
                min_readiness=min_readiness,
                min_score=self.config.routing_min_theme_score,
                query_embedding=query_embedding,
            )
            if route_scores_debug and all(str(row.get("reason") or "") == "fallback_ready_theme" for row in route_scores_debug):
                retrieval_route = self._route_by_lightweight_retrieval(
                    message,
                    max_themes=max_themes,
                    min_readiness=min_readiness,
                )
                if retrieval_route:
                    route_scores_debug = retrieval_route
            # If Stage 1 completed but no full runtime exists yet, search_ready themes
            # may still be served by knowledge_store_lightweight search. If the caller
            # requested search_ready and routing returned nothing, fall back to parsed_ready.
            if not route_scores_debug and min_readiness != "parsed_ready":
                route_scores_debug = self.catalog.route_scores(
                    message,
                    max_themes=max_themes,
                    min_readiness="parsed_ready",
                    min_score=0.0,
                    query_embedding=query_embedding,
                )
                if route_scores_debug and all(str(row.get("reason") or "") == "fallback_ready_theme" for row in route_scores_debug):
                    retrieval_route = self._route_by_lightweight_retrieval(
                        message,
                        max_themes=max_themes,
                        min_readiness="parsed_ready",
                    )
                    if retrieval_route:
                        route_scores_debug = retrieval_route
            selected = [str(row.get("theme_id")) for row in route_scores_debug if row.get("theme_id")]
        if not selected:
            raise RuntimeError("Нет готовых тематических графов для запроса.")

        answers: list[str] = []
        sources: list[dict[str, Any]] = []
        errors: list[str] = []
        lightweight_results: list[dict[str, Any]] = []
        selected_statuses: dict[str, str] = {}
        selected_runtime_ready: dict[str, bool] = {}
        for theme_id in selected:
            row = self._catalog_row_by_theme(theme_id)
            status = str(row.get("status") or "not_ready")
            runtime_ready_for_theme = self._has_runtime_index(theme_id)
            selected_statuses[theme_id] = status
            selected_runtime_ready[theme_id] = runtime_ready_for_theme
            prefer_runtime = _READINESS_ORDER.get(status, 0) >= _READINESS_ORDER["search_ready"] and runtime_ready_for_theme
            if prefer_runtime:
                try:
                    service = await self.service_for_theme(theme_id)
                    result = await service.query(message, mode=mode, history=history)
                    runtime_answer = str(result.get("answer") or "").strip()
                    runtime_sources = list(result.get("sources") or [])
                    if runtime_answer and not self._is_no_result_answer(runtime_answer, runtime_sources):
                        # Do not expose internal theme routing when it adds no value.
                        # A theme heading is useful only when several themes really
                        # contribute substantive answers. The final join keeps answers
                        # readable for single-theme runs.
                        answers.append(runtime_answer)
                        for src in runtime_sources:
                            src = dict(src)
                            src.setdefault("theme_id", theme_id)
                            sources.append(src)
                        continue
                    errors.append(
                        f"{theme_id}: LightRAG runtime returned no query context; fallback на deterministic retrieval_kg"
                    )
                except Exception as exc:
                    errors.append(f"{theme_id}: LightRAG runtime недоступен, fallback на lightweight search ({exc})")
            # Fallback: search over durable knowledge_store chunks. This is used both
            # after Stage 1 and after deterministic/compressed Stage 2 when no
            # completed LightRAG runtime marker exists.
            try:
                lightweight_results.append(self._query_theme_lightweight(theme_id, message, limit=8))
            except Exception as exc:
                errors.append(f"{theme_id}: lightweight search failed: {exc}")

        # If routing selected a ready LightRAG theme but both runtime and local
        # search produced no useful context, try retrieval-grounded routing across
        # all searchable themes. This fixes cases where LightRAG keyword extraction
        # searches irrelevant nodes/edges (e.g. author name) and returns
        # 0 entities / 0 relations / 0 vector chunks.
        explicit_theme_selection = bool(theme_ids)
        if (not any(item.get("matches") for item in lightweight_results)) and not answers and not explicit_theme_selection:
            retrieval_route = self._route_by_lightweight_retrieval(
                message,
                max_themes=max(max_themes, self.config.routing_top_k_themes, 8),
                min_readiness="search_ready",
            )
            already = {str(item.get("theme_id")) for item in lightweight_results}
            for route in retrieval_route:
                routed_theme = str(route.get("theme_id") or "")
                if not routed_theme or routed_theme in already:
                    continue
                try:
                    lightweight_results.append(self._query_theme_lightweight(routed_theme, message, limit=8))
                    row = self._catalog_row_by_theme(routed_theme)
                    selected_statuses[routed_theme] = str(row.get("status") or "not_ready")
                    selected_runtime_ready[routed_theme] = self._has_runtime_index(routed_theme)
                    selected.append(routed_theme)
                    already.add(routed_theme)
                except Exception as exc:
                    errors.append(f"{routed_theme}: global lightweight fallback failed: {exc}")

        if lightweight_results:
            stage2_ready = any(
                _READINESS_ORDER.get(str(selected_statuses.get(str(item.get("theme_id")) or "not_ready")), 0)
                >= _READINESS_ORDER["cheap_kg_ready"]
                for item in lightweight_results
            )
            runtime_ready = any(bool(selected_runtime_ready.get(str(item.get("theme_id")) or "")) for item in lightweight_results)
            fallback = await self._synthesize_lightweight_answer(
                message,
                lightweight_results,
                stage2_ready=stage2_ready,
                runtime_ready=runtime_ready,
                history=history,
            )
            if fallback.get("answer"):
                heading = "### Ответ по поисковому графу знаний" if stage2_ready else "### Быстрый поиск по durable knowledge_store"
                answers.append(heading + "\n" + fallback["answer"])
            for src in fallback.get("sources") or []:
                sources.append(src)

        if not answers and errors:
            raise RuntimeError("; ".join(errors[:3]))
        answer = "\n\n".join(answers).strip() or "В выбранных тематических графах релевантные сведения не найдены."
        # Do not prepend raw routing diagnostics to the chat answer. They are
        # returned as structured route_scores for debugging, but the user-facing
        # answer should stay RAG-like and not look like a router trace.
        if errors and not answers:
            answer += "\n\n### Диагностика поиска\n" + "\n".join(f"- {err}" for err in errors[:5])
        return {"answer": answer, "sources": sources, "theme_ids": selected, "route_scores": route_scores_debug}


def should_ignore_path(path: Path, root: Path, ignore_dirs: Iterable[str]) -> bool:
    ignore = {_norm_key(x) for x in ignore_dirs}
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = path.parts
    return any(_norm_key(part) in ignore for part in rel_parts[:-1])


def iter_supported_documents(root: Path, config: AppConfig, *, limit: int | None = None) -> Iterable[tuple[Path, str]]:
    supported = set(config.file_discovery_supported_extensions or SUPPORTED_EXTENSIONS)
    supported = {ext if str(ext).startswith(".") else f".{ext}" for ext in supported}
    iterator = root.rglob("*") if config.file_discovery_recursive else root.glob("*")
    count = 0
    for path in sorted(iterator):
        if not path.is_file():
            continue
        if should_ignore_path(path, root, config.file_discovery_ignore_dirs):
            continue
        if path.suffix.lower() not in supported:
            continue
        rel = path.relative_to(root).as_posix()
        yield path, rel
        count += 1
        if limit is not None and count >= limit:
            break
