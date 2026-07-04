from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.base import DocStatus
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.kg.shared_storage import get_namespace_data
from lightrag.utils import generate_track_id, setup_logger, wrap_embedding_func_with_attrs

from backend.config_manager import AppConfig
from backend.embedding_client import embed_texts_http_json, is_openai_embedding_url
from backend.knowledge_store import KnowledgeStore
from backend.domain.fact_store import FactStore
from backend.domain.query_expansion import build_expanded_query, build_structured_context
from backend.domain.query_parser import get_query_parser
from backend.lightrag_extraction_repair import (
    harden_extraction_prompt,
    is_lightrag_extraction_prompt,
    repair_lightrag_extraction_output,
)

setup_logger("lightrag", level="INFO")
logger = logging.getLogger(__name__)

_LOCAL_API_KEY_PLACEHOLDER = "EMPTY"
_MAX_HISTORY_MESSAGES = 20
_STATS_LABELS_TIMEOUT_SEC = 2.0

_INFLIGHT_DOC_STATUSES = frozenset(
    {
        DocStatus.PENDING,
        DocStatus.PARSING,
        DocStatus.ANALYZING,
        DocStatus.PROCESSING,
        DocStatus.PREPROCESSED,
    }
)

NIKOLA_USER_PROMPT = (
    "Ты — ◈NiCo, R&D ассистент для горно-металлургической карты знаний. "
    "Отвечай на русском языке, технически точно и только на основании найденного контекста. "
    "Для каждого существенного вывода указывай: материал, процесс, оборудование, условия, числовые параметры, "
    "географию и источник, если они есть в контексте. "
    "Не выдумывай численные значения, единицы измерения, годы и источники. "
    "Если данных недостаточно, явно пиши: 'В источниках недостаточно данных'. "
    "Если есть расхождения между источниками, выделяй раздел 'Противоречия'. "
    "Для инженерных запросов используй структуру: Краткий вывод; Найденные факты; Ограничения; Пробелы. "
    "На общие вопросы отвечай кратко. Не добавляй отдельный раздел References — источники покажет интерфейс."
)

_REFERENCE_SECTION_MARKERS = (
    "\n### References",
    "\n## References",
    "\n### Источники",
    "\n## Источники",
    "\nReferences\n",
    "\nИсточники\n",
)
_GENERAL_QUESTION_PATTERN = re.compile(
    r"^\s*(?:"
    r"привет|здравствуй|добрый\s+(?:день|утро|вечер)|"
    r"что\s+умеешь|что\s+ты\s+умеешь|"
    r"что\s+можешь\s+подсказать|чем\s+можешь\s+помочь|"
    r"кто\s+ты|расскажи\s+о\s+себе"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)


def _resolve_api_key(api_key: str) -> str:
    return api_key.strip() or _LOCAL_API_KEY_PLACEHOLDER


def _normalize_history(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    if not history:
        return []

    normalized: list[dict[str, str]] = []
    for item in history:
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": content})

    return normalized[-_MAX_HISTORY_MESSAGES:]


def _strip_reference_section(text: str) -> str:
    cleaned = text.strip()
    for marker in _REFERENCE_SECTION_MARKERS:
        idx = cleaned.find(marker)
        if idx != -1:
            cleaned = cleaned[:idx].strip()
    return cleaned


def _clean_answer(text: str) -> str:
    cleaned = _strip_reference_section(text)
    cleaned = cleaned.replace("[no-context]", "").strip()
    return cleaned or "Ответ не получен."


def _normalize_match_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()


def _human_source_name(source: str) -> str:
    raw = str(source or "").replace("\\", "/").strip()
    if not raw:
        return "unknown"
    parts = raw.split("#")
    base = Path(parts[0]).name or parts[0] or raw
    useful_suffixes = []
    for suffix in parts[1:]:
        if suffix.startswith(("kgchunk", "kgcompressed")):
            continue
        if suffix.startswith(("page:", "slide:", "chunk:")):
            useful_suffixes.append(suffix)
    return base + (("#" + "#".join(useful_suffixes)) if useful_suffixes else "")


def _is_general_question(text: str) -> bool:
    normalized = _normalize_match_text(text)
    if _GENERAL_QUESTION_PATTERN.search(normalized):
        return True
    if re.search(r"\b(реферат|отзыв|документ|docx|pdf|файл|стать|глав)\b", normalized):
        return False
    return len(normalized) <= 36 and normalized.count(" ") <= 4


def _has_retrieved_context(
    references: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> bool:
    for ref in references:
        file_path = str(ref.get("file_path") or "").strip()
        if file_path and file_path != "unknown_source":
            return True
    for chunk in chunks:
        file_path = str(chunk.get("file_path") or "").strip()
        if file_path and file_path != "unknown_source":
            return True
    return False


def _should_show_sources(
    message: str,
    raw_answer: str,
    references: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> bool:
    if _is_general_question(message):
        return False
    if "[no-context]" in (raw_answer or ""):
        return False
    return _has_retrieved_context(references, chunks)


class RagService:
    def __init__(self, working_dir: Path, knowledge_store_dir: Path | None = None) -> None:
        self.working_dir = working_dir
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_store = KnowledgeStore(knowledge_store_dir or (self.working_dir.parent / "knowledge_store"))
        self._rag: LightRAG | None = None
        self._config: AppConfig | None = None
        self._lock = asyncio.Lock()
        self._pipeline_tasks: set[asyncio.Task] = set()
        legacy_facts = self.working_dir.parent / "domain_facts.jsonl"
        self.knowledge_store.import_legacy_numeric_facts(legacy_facts)
        self.fact_store = FactStore(self.knowledge_store.numeric_facts_path)
        self.query_parser = get_query_parser()

    @property
    def is_ready(self) -> bool:
        return self._rag is not None

    async def ensure_initialized(self, config: AppConfig) -> None:
        async with self._lock:
            if self._rag is not None and self._config == config:
                return

            if self._rag is not None:
                await self._rag.finalize_storages()
                self._rag = None

            self._config = config
            self.knowledge_store.schema_version = config.schema_version
            self.knowledge_store.ontology_version = config.ontology_version
            self.knowledge_store.app_version = config.app_version
            self.knowledge_store._ensure_manifest()
            self._rag = await self._build_rag(config)

    async def reinitialize(self, config: AppConfig) -> None:
        async with self._lock:
            if self._rag is not None:
                await self._rag.finalize_storages()
                self._rag = None
            self._config = config
            self.knowledge_store.schema_version = config.schema_version
            self.knowledge_store.ontology_version = config.ontology_version
            self.knowledge_store.app_version = config.app_version
            self.knowledge_store._ensure_manifest()
            self._rag = await self._build_rag(config)

    async def _complete_llm(self, prompt: str, *, system_prompt: str, max_tokens: int = 256) -> str:
        config = self._config
        if config is None:
            raise RuntimeError("База знаний не инициализирована.")

        return await openai_complete_if_cache(
            config.llm_model,
            prompt,
            system_prompt=system_prompt,
            api_key=_resolve_api_key(config.llm_api_key),
            base_url=config.llm_base_url.rstrip("/"),
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    async def _build_rag(self, config: AppConfig) -> LightRAG:
        llm_base_url = config.llm_base_url.rstrip("/")
        embedding_base_url = (config.embedding_base_url or config.llm_base_url).rstrip("/")
        use_openai_embeddings = is_openai_embedding_url(embedding_base_url)

        async def llm_model_func(
            prompt,
            system_prompt=None,
            history_messages=None,
            keyword_extraction=False,
            **kwargs,
        ) -> str:
            llm_kwargs = dict(kwargs)
            extra_body = dict(llm_kwargs.pop("extra_body", {}) or {})
            template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
            template_kwargs["enable_thinking"] = False
            extra_body["chat_template_kwargs"] = template_kwargs
            llm_kwargs["extra_body"] = extra_body

            is_extraction = is_lightrag_extraction_prompt(
                str(prompt or ""),
                str(system_prompt or ""),
                keyword_extraction=bool(keyword_extraction),
            )
            effective_prompt = prompt
            effective_system_prompt = system_prompt
            if is_extraction and bool(getattr(config, "lightrag_extraction_prompt_hardening", True)):
                effective_prompt, effective_system_prompt = harden_extraction_prompt(
                    str(prompt or ""),
                    str(system_prompt or "") if system_prompt else None,
                )

            response = await openai_complete_if_cache(
                config.llm_model,
                effective_prompt,
                system_prompt=effective_system_prompt,
                history_messages=history_messages or [],
                api_key=_resolve_api_key(config.llm_api_key),
                base_url=llm_base_url,
                **llm_kwargs,
            )
            if is_extraction and bool(getattr(config, "lightrag_extraction_output_repair", True)):
                repaired = repair_lightrag_extraction_output(
                    response,
                    prompt=str(effective_prompt or prompt or ""),
                    system_prompt=str(effective_system_prompt or "") if effective_system_prompt else None,
                )
                if repaired != response:
                    logger.info("LightRAG extraction output repaired before parser")
                return repaired
            return response

        @wrap_embedding_func_with_attrs(
            embedding_dim=config.embedding_dim,
            max_token_size=8192,
            model_name=config.embedding_model,
        )
        async def embedding_func(texts: list[str]) -> np.ndarray:
            if use_openai_embeddings:
                return await openai_embed.func(
                    texts,
                    model=config.embedding_model,
                    api_key=_resolve_api_key(config.embedding_api_key or config.llm_api_key),
                    base_url=embedding_base_url,
                )
            return await embed_texts_http_json(
                texts,
                url=embedding_base_url,
                api_key=_resolve_api_key(config.embedding_api_key or config.llm_api_key),
            )

        rag = LightRAG(
            working_dir=str(self.working_dir),
            llm_model_func=llm_model_func,
            embedding_func=embedding_func,
            addon_params={"language": "Russian"},
        )
        await rag.initialize_storages()
        return rag

    def _require_rag(self) -> LightRAG:
        if self._rag is None:
            raise RuntimeError("База знаний не инициализирована. Сохраните настройки LLM.")
        return self._rag

    async def shutdown(self) -> None:
        async with self._lock:
            for task in list(self._pipeline_tasks):
                task.cancel()
            if self._pipeline_tasks:
                await asyncio.gather(*self._pipeline_tasks, return_exceptions=True)
                self._pipeline_tasks.clear()
            if self._rag is not None:
                await self._rag.finalize_storages()
                self._rag = None

    async def insert_text(self, text: str, *, citation_name: str) -> dict[str, Any]:
        result = await self.enqueue_documents_batch([(text, citation_name)])
        if not result["items"]:
            raise RuntimeError("Не удалось поставить документ в очередь.")
        return result["items"][0]

    async def enqueue_documents_batch(
        self,
        items: list[tuple[str, str]],
        *,
        schedule_processing: bool = True,
    ) -> dict[str, Any]:
        if not items:
            return {"track_id": "", "items": []}

        rag = self._require_rag()
        track_id = generate_track_id("insert")
        texts = [text for text, _ in items]
        paths = [name for _, name in items]

        await rag.apipeline_enqueue_documents(texts, file_paths=paths, track_id=track_id)
        if schedule_processing:
            self.schedule_document_processing()
        items_out = await self._build_queued_outcomes(track_id, paths)
        return {"track_id": track_id, "items": items_out}

    def schedule_document_processing(self) -> None:
        task = asyncio.create_task(self._run_pipeline_safe())
        self._pipeline_tasks.add(task)
        task.add_done_callback(lambda t: self._pipeline_tasks.discard(t))

    async def _run_pipeline_safe(self) -> None:
        rag = self._require_rag()
        try:
            await rag.apipeline_process_enqueue_documents()
        except Exception as exc:
            logger.warning("Не удалось обработать очередь документов: %s", exc)

    async def resume_pending_documents(self) -> None:
        self.schedule_document_processing()

    async def is_pipeline_busy(self) -> bool:
        if self._rag is None:
            return False
        try:
            workspace = getattr(self._rag, "workspace", None)
            pipeline_status = await get_namespace_data("pipeline_status", workspace=workspace)
            return bool(pipeline_status.get("busy"))
        except Exception:
            return False

    async def get_track_status(self, track_id: str) -> dict[str, Any]:
        rag = self._require_rag()
        docs = await rag.doc_status.get_docs_by_track_id(track_id)
        documents: list[dict[str, Any]] = []
        status_summary: dict[str, int] = {}

        for doc_id, doc in docs.items():
            status = doc.status.value if isinstance(doc.status, DocStatus) else str(doc.status)
            status_summary[status] = status_summary.get(status, 0) + 1
            documents.append(
                {
                    "id": doc_id,
                    "filename": Path(doc.file_path).name,
                    "file_path": doc.file_path,
                    "status": status,
                    "chars": doc.content_length,
                    "chunks": doc.chunks_count or 0,
                    "error": doc.error_msg,
                    "updated_at": doc.updated_at,
                }
            )

        is_complete = bool(documents) and all(
            doc["status"] not in {status.value for status in _INFLIGHT_DOC_STATUSES}
            for doc in documents
        )
        processed_count = sum(1 for doc in documents if doc["status"] == DocStatus.PROCESSED.value)
        failed_count = sum(1 for doc in documents if doc["status"] == DocStatus.FAILED.value)

        return {
            "track_id": track_id,
            "documents": documents,
            "total_count": len(documents),
            "status_summary": status_summary,
            "is_complete": is_complete,
            "processed_count": processed_count,
            "failed_count": failed_count,
        }

    async def wait_for_track(
        self,
        track_id: str,
        *,
        poll_interval: float = 10.0,
        timeout_sec: float = 0.0,
        max_empty_polls: int = 30,
    ) -> dict[str, Any]:
        """Poll LightRAG doc status until one track is complete.

        timeout_sec <= 0 means no timeout. This is intended for unattended
        overnight rebuilds where the shell script must not exit before the
        background LightRAG pipeline finishes.
        """
        start = asyncio.get_running_loop().time()
        empty_polls = 0
        last_status: dict[str, Any] = {"track_id": track_id, "is_complete": False}
        while True:
            last_status = await self.get_track_status(track_id)
            if int(last_status.get("total_count") or 0) == 0:
                empty_polls += 1
                if max_empty_polls > 0 and empty_polls >= max_empty_polls:
                    last_status["timeout"] = True
                    last_status["error"] = "No documents appeared for track_id while waiting."
                    return last_status
            else:
                empty_polls = 0
            if last_status.get("is_complete"):
                return last_status
            if timeout_sec and timeout_sec > 0:
                elapsed = asyncio.get_running_loop().time() - start
                if elapsed > timeout_sec:
                    last_status["timeout"] = True
                    return last_status
            await asyncio.sleep(max(1.0, float(poll_interval)))

    async def _build_queued_outcomes(
        self,
        track_id: str,
        filenames: list[str],
    ) -> list[dict[str, Any]]:
        rag = self._require_rag()
        docs = await rag.doc_status.get_docs_by_track_id(track_id)
        outcomes: list[dict[str, Any]] = []

        for filename in filenames:
            matching = [
                doc for doc in docs.values() if Path(doc.file_path).name == Path(filename).name
            ]
            if not matching:
                outcomes.append(
                    {
                        "filename": filename,
                        "success": False,
                        "status": "failed",
                        "track_id": track_id,
                        "message": "Документ не попал в очередь обработки.",
                    }
                )
                continue

            duplicate = next(
                (doc for doc in matching if (doc.metadata or {}).get("is_duplicate")),
                None,
            )
            if duplicate:
                outcomes.append(
                    {
                        "filename": filename,
                        "success": False,
                        "status": "failed",
                        "track_id": track_id,
                        "message": self._format_upload_error(duplicate),
                    }
                )
                continue

            processed = next(
                (doc for doc in matching if doc.status == DocStatus.PROCESSED),
                None,
            )
            if processed:
                outcomes.append(
                    {
                        "filename": filename,
                        "success": True,
                        "status": "processed",
                        "chars": processed.content_length,
                        "track_id": track_id,
                        "message": "Документ добавлен в базу знаний.",
                    }
                )
                continue

            doc = matching[-1]
            outcomes.append(
                {
                    "filename": filename,
                    "success": True,
                    "status": "queued",
                    "chars": doc.content_length,
                    "track_id": track_id,
                    "message": "Документ поставлен в очередь обработки.",
                }
            )

        return outcomes

    @staticmethod
    def _format_upload_error(doc: Any) -> str:
        metadata = doc.metadata or {}
        filename = Path(doc.file_path).name
        if metadata.get("is_duplicate"):
            if metadata.get("duplicate_kind") == "content_hash":
                return f"Документ «{filename}» уже есть в базе (совпадает содержимое)."
            return f"Файл «{filename}» уже загружен. Переименуйте файл или удалите старую версию."

        if doc.error_msg:
            if "timeout" in doc.error_msg.lower():
                return (
                    f"Не удалось обработать «{filename}»: LLM не ответил вовремя. "
                    "Попробуйте позже или загрузите файл меньшего размера."
                )
            return doc.error_msg

        if doc.status in _INFLIGHT_DOC_STATUSES:
            return f"Документ «{filename}» всё ещё обрабатывается."

        return f"Не удалось обработать «{filename}»."

    async def query(
        self,
        message: str,
        mode: str = "hybrid",
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        rag = self._require_rag()
        conversation_history = _normalize_history(history)

        parsed = self.query_parser.parse(message)
        structured_facts = self.fact_store.search(parsed, limit=12)
        expanded_query = build_expanded_query(parsed, top_facts=structured_facts)
        structured_context = build_structured_context(structured_facts)
        retrieval_message = (
            f"{expanded_query}\n\n"
            f"{structured_context}\n\n"
            f"Ответь на исходный запрос пользователя: {message}"
        ).strip()

        query_param = QueryParam(
            mode=mode,
            conversation_history=conversation_history,
            user_prompt=NIKOLA_USER_PROMPT,
            response_type="Structured technical answer",
            enable_rerank=True,
            include_references=True,
        )
        result = await rag.aquery_llm(retrieval_message, param=query_param)

        llm_response = result.get("llm_response", {})
        raw_answer = llm_response.get("content") or ""
        references = result.get("data", {}).get("references", [])
        chunks = result.get("data", {}).get("chunks", [])

        # Compressed LightRAG can build chunk vectors but fail KG keyword routing
        # (e.g. Query nodes: author name; Raw search results: 0/0/0). Retry with
        # vector-only/naive retrieval before declaring no-context. This preserves
        # the useful old compressed-LightRAG behaviour for slide/PDF fragments.
        if (not _has_retrieved_context(references, chunks)) and str(mode).lower() != "naive":
            try:
                retry_result = await rag.aquery_llm(
                    retrieval_message,
                    param=QueryParam(
                        mode="naive",
                        conversation_history=conversation_history,
                        user_prompt=NIKOLA_USER_PROMPT,
                        response_type="Structured technical answer",
                        enable_rerank=True,
                        include_references=True,
                    ),
                )
                retry_refs = retry_result.get("data", {}).get("references", [])
                retry_chunks = retry_result.get("data", {}).get("chunks", [])
                retry_raw = (retry_result.get("llm_response", {}) or {}).get("content") or ""
                if _has_retrieved_context(retry_refs, retry_chunks) or (retry_raw and "[no-context]" not in retry_raw):
                    result = retry_result
                    llm_response = result.get("llm_response", {})
                    raw_answer = llm_response.get("content") or ""
                    references = retry_refs
                    chunks = retry_chunks
            except Exception as exc:
                logger.warning("LightRAG naive retry failed: %s", exc)

        answer = _clean_answer(raw_answer)
        if answer == "Ответ не получен." and result.get("status") == "failure":
            answer = result.get("message") or answer

        sources = (
            self._format_sources(references, chunks)
            if _should_show_sources(message, raw_answer, references, chunks)
            else []
        )

        return {
            "answer": answer,
            "sources": sources,
            "parsed_query": parsed.to_dict(),
            "structured_facts": structured_facts,
        }

    def parse_query_debug(self, message: str) -> dict[str, Any]:
        parsed = self.query_parser.parse(message)
        facts = self.fact_store.search(parsed, limit=12)
        return {
            "parsed_query": parsed.to_dict(),
            "expanded_query": build_expanded_query(parsed, top_facts=facts),
            "structured_context": build_structured_context(facts),
            "facts": facts,
        }

    @staticmethod
    def _format_sources(
        references: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Источники из retrieval LightRAG: reference_list и chunks контекста."""
        chars_by_path: dict[str, int] = {}
        for chunk in chunks:
            file_path = str(chunk.get("file_path") or "").strip()
            if not file_path or file_path == "unknown_source":
                continue
            content = str(chunk.get("content") or "")
            chars_by_path[file_path] = chars_by_path.get(file_path, 0) + len(content)

        sources: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add_source(file_path: str, reference_id: str = "") -> None:
            filename = _human_source_name(file_path)
            dedupe_key = file_path
            if dedupe_key in seen:
                return
            seen.add(dedupe_key)
            entry: dict[str, Any] = {"filename": filename, "file_path": file_path}
            if reference_id:
                entry["reference_id"] = reference_id
            chunk_chars = chars_by_path.get(file_path)
            if chunk_chars:
                entry["chars"] = chunk_chars
            sources.append(entry)

        for ref in references:
            file_path = str(ref.get("file_path") or "").strip()
            if not file_path or file_path == "unknown_source":
                continue
            add_source(file_path, str(ref.get("reference_id") or ""))

        if not sources:
            for chunk in chunks:
                file_path = str(chunk.get("file_path") or "").strip()
                if not file_path or file_path == "unknown_source":
                    continue
                add_source(file_path, str(chunk.get("reference_id") or ""))

        return sorted(
            sources,
            key=lambda item: int(item["reference_id"])
            if str(item.get("reference_id") or "").isdigit()
            else 999,
        )

    async def rebuild_runtime_index_from_store(
        self,
        *,
        batch_size: int = 64,
        wait: bool = False,
        poll_interval: float = 10.0,
        timeout_sec: float = 0.0,
        items_override: list[tuple[str, str]] | None = None,
        build_label: str = "full",
    ) -> dict[str, Any]:
        """Rebuild LightRAG runtime index from durable knowledge_store chunks.

        This does not re-parse PDF/DOCX/PPTX files. It reuses lightrag_text from
        data/knowledge_store/chunks.jsonl and re-enqueues it to the current
        LightRAG working_dir. If wait=True, batches are processed serially and
        the function returns only after LightRAG has finished each track.
        """
        self._require_rag()
        items = list(items_override) if items_override is not None else list(self.knowledge_store.iter_lightrag_items())
        if not items:
            return {"track_ids": [], "chunks": 0, "message": "knowledge_store не содержит chunks для выбранного режима.", "build_label": build_label}

        track_ids: list[str] = []
        track_statuses: list[dict[str, Any]] = []
        processed_count = 0
        failed_count = 0
        timeout_count = 0
        bs = max(1, int(batch_size))
        for start in range(0, len(items), bs):
            batch = items[start : start + bs]
            # Always schedule processing after enqueue. When wait=True we wait for
            # the current track status, not for the whole LightRAG pipeline task.
            # Waiting on apipeline_process_enqueue_documents() directly can look
            # like a web/API hang, especially when LightRAG is finalizing graph
            # storages or processing stale queue entries from previous runs.
            result = await self.enqueue_documents_batch(batch, schedule_processing=True)
            track_id = str(result.get("track_id") or "")
            if not track_id:
                continue
            track_ids.append(track_id)
            if wait:
                status = await self.wait_for_track(
                    track_id,
                    poll_interval=poll_interval,
                    timeout_sec=timeout_sec,
                    max_empty_polls=6,
                )
                if status.get("timeout"):
                    timeout_count += 1
                track_statuses.append(status)
                processed_count += int(status.get("processed_count") or 0)
                failed_count += int(status.get("failed_count") or 0)
        return {
            "track_ids": track_ids,
            "track_statuses": track_statuses,
            "chunks": len(items),
            "waited": bool(wait),
            "processed_count": processed_count,
            "failed_count": failed_count,
            "timeout_count": timeout_count,
            "build_label": build_label,
            "message": f"Поставлено chunks на пересборку runtime-индекса: {len(items)} ({build_label}).",
        }

    async def summarize_chat_title(self, messages: list[dict[str, str]]) -> str:
        self._require_rag()
        normalized = _normalize_history(messages)
        if not normalized:
            return "Новый чат"

        transcript = "\n".join(
            f"{'Пользователь' if msg['role'] == 'user' else '◈NiCo'}: {msg['content'][:400]}"
            for msg in normalized[-8:]
        )
        try:
            title = await self._complete_llm(
                transcript,
                system_prompt=(
                    "Сформулируй короткое название диалога на русском языке (4–8 слов). "
                    "Ответь только названием, без кавычек, без точки в конце."
                ),
                max_tokens=48,
            )
        except Exception:
            first_user = next((m["content"] for m in normalized if m["role"] == "user"), "Новый чат")
            return first_user[:60].strip() or "Новый чат"

        cleaned = title.strip().strip('"').strip("'").strip("«»").split("\n")[0].strip()
        return cleaned[:80] or "Новый чат"

    async def get_documents(self) -> list[dict[str, Any]]:
        rag = self._require_rag()
        statuses = [
            DocStatus.PROCESSED,
            DocStatus.PENDING,
            DocStatus.PARSING,
            DocStatus.ANALYZING,
            DocStatus.PROCESSING,
            DocStatus.PREPROCESSED,
            DocStatus.FAILED,
        ]
        all_docs = await rag.doc_status.get_docs_by_statuses(statuses)

        latest_by_name: dict[str, dict[str, Any]] = {}

        for doc_id, doc in all_docs.items():
            if doc_id.startswith("dup-"):
                continue
            if (doc.metadata or {}).get("is_duplicate"):
                continue

            filename = Path(doc.file_path).name
            if not filename or filename == "unknown_source":
                continue

            entry = {
                "id": doc_id,
                "filename": filename,
                "file_path": doc.file_path,
                "status": doc.status.value if isinstance(doc.status, DocStatus) else str(doc.status),
                "chars": doc.content_length,
                "chunks": doc.chunks_count or 0,
                "updated_at": doc.updated_at,
                "error": doc.error_msg,
            }

            existing = latest_by_name.get(filename)
            if not existing or (entry.get("updated_at") or "") >= (existing.get("updated_at") or ""):
                latest_by_name[filename] = entry

        documents = list(latest_by_name.values())
        documents.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
        return documents

    async def delete_document(self, doc_id: str) -> dict[str, Any]:
        rag = self._require_rag()
        result = await rag.adelete_by_doc_id(doc_id.strip(), delete_llm_cache=True)
        return {
            "status": result.status,
            "doc_id": result.doc_id,
            "message": result.message,
            "filename": Path(result.file_path).name if result.file_path else "",
        }

    async def get_knowledge_state(self) -> dict[str, Any]:
        documents = await self.get_documents()
        pipeline_busy = await self.is_pipeline_busy()
        processed_count = sum(1 for doc in documents if doc.get("status") == DocStatus.PROCESSED.value)
        failed_count = sum(1 for doc in documents if doc.get("status") == DocStatus.FAILED.value)
        processing_count = sum(
            1
            for doc in documents
            if doc.get("status") in {status.value for status in _INFLIGHT_DOC_STATUSES}
        )
        knowledge_ready = processed_count > 0 and processing_count == 0 and not pipeline_busy
        return {
            "knowledge_ready": knowledge_ready,
            "document_count": processed_count,
            "processing_count": processing_count,
            "failed_count": failed_count,
            "pipeline_busy": pipeline_busy,
            "documents": documents,
        }

    async def get_stats(self) -> dict[str, Any]:
        rag = self._require_rag()
        state = await self.get_knowledge_state()
        try:
            labels = await asyncio.wait_for(
                rag.get_graph_labels(),
                timeout=_STATS_LABELS_TIMEOUT_SEC,
            )
        except TimeoutError:
            labels = []
        return {
            "entities": len(labels),
            "labels_preview": labels[:12],
            **state,
            "working_dir": str(self.working_dir),
            "knowledge_store": self.knowledge_store.stats(),
        }
