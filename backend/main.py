from __future__ import annotations

import asyncio
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config_manager import AppConfig, ConfigManager, read_project_yaml, resolve_project_path
from backend.document_loader import SUPPORTED_EXTENSIONS
from backend.domain.document_processor import process_document
from backend.domain.numeric_extractor import get_numeric_extractor
from backend.knowledge_store import jsonl_append, jsonl_read, utc_now_iso
from backend.rag_service import RagService
from backend.theme_sharding import (
    _READINESS_ORDER,
    ThemeShardManager,
    get_ingestion_profile,
    iter_supported_documents,
)

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

CONFIG_PATH = Path(os.getenv("APP_CONFIG_PATH", ROOT / "config.yaml"))
PROJECT_CONFIG = read_project_yaml(CONFIG_PATH)
FRONTEND_DIR = ROOT / "frontend"


def _cfg_get(path: tuple[str, ...], default=None):
    node = PROJECT_CONFIG
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _env_or_yaml_path(env_name: str, yaml_path: tuple[str, ...], default: str) -> Path:
    return resolve_project_path(ROOT, os.getenv(env_name) or _cfg_get(yaml_path), default)


UPLOAD_DIR = _env_or_yaml_path("UPLOAD_DIR", ("upload", "store_dir"), "./uploads")
WORKING_DIR = _env_or_yaml_path("RAG_WORKING_DIR", ("rag", "working_dir"), "./data/rag_storage")
KNOWLEDGE_STORE_DIR = _env_or_yaml_path("KNOWLEDGE_STORE_DIR", ("storage", "knowledge_store_dir"), "./data/knowledge_store")
STAGE1_STAGING_DIR = UPLOAD_DIR / "stage1_staging"

config_manager = ConfigManager(CONFIG_PATH)
rag_service = RagService(WORKING_DIR, KNOWLEDGE_STORE_DIR)
theme_shards = ThemeShardManager(ROOT, config_manager.load())

INGESTION_RUNS: dict[str, dict[str, Any]] = {}
INGESTION_TASKS: dict[str, asyncio.Task] = {}


def _active_ingestion_runs() -> list[dict[str, Any]]:
    return [
        run
        for run in INGESTION_RUNS.values()
        if run.get("status") in {"queued", "running", "cancelling"}
    ]


def _is_ingestion_busy() -> bool:
    return bool(_active_ingestion_runs())


def _bool_value(value, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_flag(name: str, *, default: bool = False) -> bool:
    return _bool_value(os.getenv(name), default=default)


def _runtime_flag(env_name: str, yaml_path: tuple[str, ...], *, default: bool = False) -> bool:
    value = os.getenv(env_name)
    if value is not None:
        return _bool_value(value, default=default)
    return _bool_value(_cfg_get(yaml_path), default=default)


def _client_filename(raw: str | None) -> str:
    name = (raw or "без_имени").replace("\\", "/").strip().lstrip("/")
    if not name:
        return "без_имени"
    parts = [part for part in name.split("/") if part not in {"", ".", ".."}]
    return "/".join(parts) or "без_имени"


def _client_suffix(filename: str) -> str:
    return PurePosixPath(filename).suffix.lower()


def _safe_upload_leaf(filename: str) -> str:
    leaf = PurePosixPath(filename).name.strip() or "document"
    return "".join(ch if ch not in '<>:"/\\|?*' else "_" for ch in leaf)


def _safe_relative_upload_path(filename: str) -> Path:
    """Return a safe relative path preserving folder picker structure."""
    clean = _client_filename(filename)
    parts: list[str] = []
    for raw in clean.split("/"):
        raw = raw.strip()
        if not raw or raw in {".", ".."}:
            continue
        safe = "".join(ch if ch not in '<>:"/\\|?*' else "_" for ch in raw).strip()
        if safe:
            parts.append(safe)
    if not parts:
        parts = ["без_имени"]
    return Path(*parts)


def _safe_stage_session_id(session_id: str | None) -> str:
    value = (session_id or "").strip()
    if not value:
        value = uuid.uuid4().hex
    safe = "".join(ch for ch in value if ch.isalnum() or ch in {"-", "_"})[:80]
    return safe or uuid.uuid4().hex


@asynccontextmanager
async def lifespan(_: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    STAGE1_STAGING_DIR.mkdir(parents=True, exist_ok=True)
    config = config_manager.load()
    try:
        theme_shards.update_config(config)
        if not config.topic_sharding_enabled:
            await rag_service.ensure_initialized(config)
            if _runtime_flag("RAG_AUTO_RESUME", ("rag", "auto_resume"), default=False):
                rag_service.schedule_document_processing()
    except Exception as exc:
        print(f"[warn] База знаний не инициализирована при старте: {exc}")
    yield
    await rag_service.shutdown()
    await theme_shards.shutdown()


app = FastAPI(
    title=str(_cfg_get(("app", "title"), "◈NiCo")),
    description="Графовая база знаний для горно-металлургических R&D-документов",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConfigUpdateRequest(BaseModel):
    app_symbol: str | None = None
    app_title: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_model: str | None = None
    embedding_dim: int | None = None
    embedding_api_key: str | None = None
    query_mode: str | None = None
    knowledge_store_dir: str | None = None
    runtime_rag_dir: str | None = None
    vector_cache_dir: str | None = None
    snapshots_dir: str | None = None
    schema_version: str | None = None
    ontology_version: str | None = None
    app_version: str | None = None
    topic_sharding_enabled: bool | None = None
    topic_collection_level: int | None = None
    topic_theme_level: int | None = None
    topic_default_collection: str | None = None
    topic_default_theme: str | None = None
    topic_batch_size_docs: int | None = None
    topic_max_parallel_themes: int | None = None
    topic_max_parallel_docs_per_theme: int | None = None
    global_router_dir: str | None = None
    file_discovery_recursive: bool | None = None
    file_discovery_include_root_files: bool | None = None
    file_discovery_supported_extensions: list[str] | None = None
    file_discovery_ignore_dirs: list[str] | None = None
    file_discovery_year_dir_pattern: str | None = None
    theme_resolution_strategy: str | None = None
    theme_use_path: bool | None = None
    theme_use_filename: bool | None = None
    theme_use_content_preview: bool | None = None
    theme_content_preview_chars: int | None = None
    theme_min_confidence: float | None = None
    theme_known_collections: dict[str, Any] | None = None
    theme_overrides: list[dict[str, Any]] | None = None
    routing_top_k_themes: int | None = None
    routing_min_theme_score: float | None = None
    routing_fallback_to_global_chunks: bool | None = None
    graph_metrics_enabled: bool | None = None
    graph_metrics_top_n: int | None = None
    graph_pagerank_iterations: int | None = None
    graph_betweenness_sample: int | None = None
    theme_embeddings_enabled: bool | None = None
    theme_embeddings_max_chunks_per_theme: int | None = None
    theme_embeddings_embedding_base_url: str | None = None
    theme_embeddings_embedding_model: str | None = None
    theme_embeddings_embedding_dim: int | None = None
    theme_embeddings_embedding_api_key: str | None = None
    routing_use_theme_embeddings: bool | None = None
    routing_explain: bool | None = None


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1)


class ChatTitleRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)


class ChatTitleResponse(BaseModel):
    title: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    mode: str | None = None
    history: list[ChatMessage] = Field(default_factory=list)
    theme_ids: list[str] = Field(default_factory=list)
    routing: str = "auto"
    min_readiness: str = "search_ready"


class SourceReference(BaseModel):
    filename: str
    file_path: str | None = None
    chars: int | None = None
    reference_id: str = ""


class ChatResponse(BaseModel):
    answer: str
    mode: str
    sources: list[SourceReference] = Field(default_factory=list)


class UploadResponse(BaseModel):
    track_id: str
    filename: str
    chars: int
    message: str


class BatchUploadItem(BaseModel):
    filename: str
    success: bool
    status: str = "queued"
    chars: int = 0
    track_id: str = ""
    message: str = ""


class BatchUploadResponse(BaseModel):
    track_id: str = ""
    results: list[BatchUploadItem]
    succeeded: int
    failed: int
    message: str


class TrackStatusResponse(BaseModel):
    track_id: str
    documents: list[dict]
    total_count: int
    status_summary: dict[str, int]
    is_complete: bool
    processed_count: int
    failed_count: int


class DeleteDocumentResponse(BaseModel):
    status: str
    doc_id: str
    message: str
    filename: str = ""


class DebugQueryRequest(BaseModel):
    message: str = Field(min_length=1)


class DebugExtractRequest(BaseModel):
    text: str = Field(min_length=1)
    source_name: str = "debug.txt"


class Stage1StartRequest(BaseModel):
    root_path: str = Field(min_length=1)
    limit: int | None = None
    profile: str = "fast_fill"
    skip_existing: bool = True
    build_runtime_index: bool = False
    rebuild_graph_metrics_after: bool = False
    rebuild_theme_embeddings_after: bool = False
    build_global_router_after: bool = True


class Stage1StagedStartRequest(BaseModel):
    session_id: str = Field(min_length=1)
    limit: int | None = None
    profile: str = "fast_fill"
    skip_existing: bool = True
    build_runtime_index: bool = False
    rebuild_graph_metrics_after: bool = False
    rebuild_theme_embeddings_after: bool = False
    build_global_router_after: bool = True
    cleanup_after: bool = False


class Stage2StartRequest(BaseModel):
    profile: str = "overnight_retrieval_kg"
    theme_ids: list[str] = Field(default_factory=list)
    max_themes: int | None = None
    batch_size: int | None = None
    clear_runtime: bool | None = None
    force: bool = False
    wait: bool = True
    poll_interval: float | None = None
    timeout_sec: float | None = None
    graph_mode: str | None = None
    build_runtime_index: bool | None = None
    max_chunks_per_document: int | None = None
    min_kg_score: float | None = None
    max_chunks_per_theme: int | None = None
    max_candidate_chunks_per_theme: int | None = None
    max_parallel_themes: int | None = None
    compressed_dynamic_document_limits: bool | None = None
    compressed_short_document_max_chunks: int | None = None
    compressed_long_document_max_chunks: int | None = None
    rebuild_graph_metrics_after: bool = True
    rebuild_theme_embeddings_after: bool = False
    build_global_router_after: bool = True


@app.get("/api/health")
async def health():
    pipeline_busy = False
    knowledge_ready = False
    document_count = 0
    processing_count = 0
    config = config_manager.load()
    if config.topic_sharding_enabled:
        try:
            state = await theme_shards.knowledge_state()
            pipeline_busy = bool(state.get("pipeline_busy")) or _is_ingestion_busy()
            knowledge_ready = bool(state.get("knowledge_ready"))
            document_count = int(state.get("document_count") or 0)
            processing_count = int(state.get("processing_count") or 0)
        except Exception:
            pipeline_busy = False
    elif rag_service.is_ready:
        try:
            state = await rag_service.get_knowledge_state()
            pipeline_busy = bool(state.get("pipeline_busy")) or _is_ingestion_busy()
            knowledge_ready = bool(state.get("knowledge_ready"))
            document_count = int(state.get("document_count") or 0)
            processing_count = int(state.get("processing_count") or 0)
        except Exception:
            pipeline_busy = False
    return {
        "status": "ok",
        "app_symbol": config.app_symbol,
        "app_title": config.app_title,
        "rag_ready": True if config.topic_sharding_enabled else rag_service.is_ready,
        "topic_sharding_enabled": config.topic_sharding_enabled,
        "knowledge_ready": knowledge_ready,
        "document_count": document_count,
        "processing_count": processing_count,
        "pipeline_busy": pipeline_busy or _is_ingestion_busy(),
        "active_ingestion_runs": _active_ingestion_runs(),
    }


@app.get("/api/config", response_model=AppConfig)
async def get_config():
    return config_manager.load()


@app.put("/api/config", response_model=AppConfig)
async def update_config(payload: ConfigUpdateRequest):
    updates = payload.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="Нет полей для обновления")

    config = config_manager.update(updates)
    theme_shards.update_config(config)
    try:
        if not config.topic_sharding_enabled:
            await rag_service.reinitialize(config)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Не удалось применить настройки LLM: {exc}") from exc
    return config


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    config = config_manager.load()
    mode = payload.mode or config.query_mode

    if config.topic_sharding_enabled:
        state = await theme_shards.knowledge_state()
        if not state.get("knowledge_ready"):
            raise HTTPException(
                status_code=409,
                detail="Тематические графы ещё не построены. Загрузите документы и дождитесь уровня search_ready хотя бы для одной темы.",
            )
        try:
            result = await theme_shards.query(
                payload.message.strip(),
                mode=mode,
                history=[item.model_dump() for item in payload.history],
                theme_ids=payload.theme_ids,
                routing=payload.routing,
                min_readiness=payload.min_readiness,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Ошибка запроса к тематическим графам: {exc}") from exc
    else:
        await rag_service.ensure_initialized(config)
        state = await rag_service.get_knowledge_state()
        if not state.get("knowledge_ready"):
            raise HTTPException(
                status_code=409,
                detail="Граф знаний ещё не построен. Загрузите документы и дождитесь завершения обработки.",
            )
        try:
            result = await rag_service.query(
                payload.message.strip(),
                mode=mode,
                history=[item.model_dump() for item in payload.history],
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Ошибка запроса к базе знаний: {exc}") from exc

    return ChatResponse(
        answer=result["answer"],
        mode=mode,
        sources=result.get("sources", []),
    )


@app.post("/api/chat/title", response_model=ChatTitleResponse)
async def chat_title(payload: ChatTitleRequest):
    config = config_manager.load()
    await rag_service.ensure_initialized(config)
    try:
        title = await rag_service.summarize_chat_title(
            [item.model_dump() for item in payload.messages],
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Не удалось сформировать название: {exc}") from exc
    return ChatTitleResponse(title=title)


@app.post("/api/knowledge/upload", response_model=UploadResponse)
async def upload_knowledge(file: UploadFile = File(...), profile: str = "balanced"):
    config = config_manager.load()
    result = await (_process_upload_sharded(file, profile_name=profile) if config.topic_sharding_enabled else _process_upload(file))
    if not result.success:
        raise HTTPException(status_code=400, detail=result.message)
    return UploadResponse(
        track_id=result.track_id,
        filename=result.filename,
        chars=result.chars,
        message=result.message,
    )


@app.post("/api/knowledge/upload/batch", response_model=BatchUploadResponse)
async def upload_knowledge_batch(files: list[UploadFile] = File(...), profile: str = "balanced"):
    if not files:
        raise HTTPException(status_code=400, detail="Не выбрано ни одного файла")

    config = config_manager.load()
    if config.topic_sharding_enabled:
        results: list[BatchUploadItem] = []
        for file in files:
            results.append(await _process_upload_sharded(file, profile_name=profile))
        succeeded = sum(1 for item in results if item.success)
        failed = len(results) - succeeded
        first_track = next((item.track_id for item in results if item.track_id), "")
        return BatchUploadResponse(
            track_id=first_track,
            results=results,
            succeeded=succeeded,
            failed=failed,
            message=(
                f"В тематические shards поставлено {succeeded} из {len(results)} документов"
                if succeeded else f"Не удалось загрузить ни одного документа ({failed})"
            ),
        )

    await rag_service.ensure_initialized(config)

    prepared: list[tuple[str, str, str, Path]] = []
    prepared_by_file: dict[str, dict] = {}
    results: list[BatchUploadItem] = []
    numeric_extractor = get_numeric_extractor()

    for file in files:
        filename = _client_filename(file.filename)
        if not filename:
            results.append(BatchUploadItem(filename="без_имени", success=False, message="Имя файла не указано"))
            continue

        suffix = _client_suffix(filename)
        if suffix not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS))
            results.append(
                BatchUploadItem(
                    filename=filename,
                    success=False,
                    message=f"Неподдерживаемый формат. Доступно: {supported}",
                )
            )
            continue

        safe_name = f"{uuid.uuid4().hex}_{_safe_upload_leaf(filename)}"
        saved_path = UPLOAD_DIR / safe_name

        try:
            with saved_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            processed = await asyncio.to_thread(process_document, saved_path, original_name=filename)
            facts = []
            for obj in processed.objects:
                facts.extend(
                    numeric_extractor.extract(
                        obj.text,
                        source_name=processed.source_name,
                        object_id=obj.object_id,
                        object_type=obj.object_type,
                        metadata={**obj.metadata, "source_hash": processed.source_hash},
                    )
                )
            store_meta = rag_service.knowledge_store.upsert_processed_document(
                processed,
                facts,
                original_path=filename,
            )

            prepared_by_file[filename] = {
                "source_name": processed.source_name,
                "objects_count": len(processed.objects),
                "facts_count": len(facts),
                "store_meta": store_meta,
                "chars": processed.total_chars,
                "saved_path": saved_path,
            }
            for obj in processed.objects:
                prepared.append((obj.to_lightrag_text(), obj.citation_name, filename, saved_path))
        except Exception as exc:
            if saved_path.exists():
                saved_path.unlink(missing_ok=True)
            results.append(BatchUploadItem(filename=filename, success=False, message=str(exc)))

    batch_track_id = ""
    if prepared:
        try:
            enqueue_result = await rag_service.enqueue_documents_batch(
                [(text, citation_name) for text, citation_name, _, _ in prepared]
            )
            batch_track_id = enqueue_result.get("track_id", "")
            success_by_file: dict[str, bool] = {filename: False for _, _, filename, _ in prepared}
            outcomes_by_name = {item["filename"]: item for item in enqueue_result.get("items", [])}

            for _, citation_name, filename, _ in prepared:
                outcome = outcomes_by_name.get(citation_name)
                if outcome and outcome.get("success", False):
                    success_by_file[filename] = True

            for filename, meta in prepared_by_file.items():
                saved_path = meta["saved_path"]
                if saved_path.exists():
                    saved_path.unlink(missing_ok=True)
                success = success_by_file.get(filename, False)
                results.append(
                    BatchUploadItem(
                        filename=filename,
                        success=success,
                        status="queued" if success else "failed",
                        chars=int(meta["chars"] or 0),
                        track_id=batch_track_id,
                        message=(
                            f"Документ разложен на объектов: {meta['objects_count']}; "
                            f"структурированных фактов: {meta['facts_count']}; обработка LightRAG идёт в фоне."
                            if success
                            else "Объекты документа не попали в очередь LightRAG."
                        ),
                    )
                )
        except Exception as exc:
            for _, _, filename, saved_path in prepared:
                if saved_path.exists():
                    saved_path.unlink(missing_ok=True)
            for filename in prepared_by_file:
                results.append(
                    BatchUploadItem(
                        filename=filename,
                        success=False,
                        status="failed",
                        message=str(exc),
                    )
                )

    succeeded = sum(1 for item in results if item.success)
    failed = len(results) - succeeded
    if failed == 0:
        if batch_track_id:
            message = f"Документы поставлены в очередь ({succeeded}). Извлечены metadata-aware объекты и числовые факты."
        else:
            message = f"Загружено документов: {succeeded}"
    elif succeeded == 0:
        message = f"Не удалось загрузить ни одного документа ({failed})"
    else:
        message = f"В очередь поставлено {succeeded} из {len(results)} документов"

    return BatchUploadResponse(
        track_id=batch_track_id,
        results=results,
        succeeded=succeeded,
        failed=failed,
        message=message,
    )

async def _process_upload_sharded(file: UploadFile, *, profile_name: str = "balanced") -> BatchUploadItem:
    filename = _client_filename(file.filename)
    if not filename:
        return BatchUploadItem(filename="без_имени", success=False, message="Имя файла не указано")

    suffix = _client_suffix(filename)
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS))
        return BatchUploadItem(filename=filename, success=False, message=f"Неподдерживаемый формат. Доступно: {supported}")

    config = config_manager.load()
    theme_shards.update_config(config)
    safe_name = f"{uuid.uuid4().hex}_{_safe_upload_leaf(filename)}"
    saved_path = UPLOAD_DIR / safe_name
    numeric_extractor = get_numeric_extractor()

    try:
        with saved_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        processed = await asyncio.to_thread(process_document, saved_path, original_name=filename)
        facts = []
        for obj in processed.objects:
            facts.extend(
                numeric_extractor.extract(
                    obj.text,
                    source_name=processed.source_name,
                    object_id=obj.object_id,
                    object_type=obj.object_type,
                    metadata={**obj.metadata, "source_hash": processed.source_hash},
                )
            )
        profile = get_ingestion_profile(config, profile_name)
        build_runtime_index = bool(profile.get("build_runtime_index", True))
        outcome = await theme_shards.ingest_processed_document(
            processed,
            facts,
            original_path=filename,
            build_runtime_index=build_runtime_index,
            compute_graph_metrics=bool(profile.get("compute_graph_metrics_during_ingest", False)),
            status_if_no_runtime=str(profile.get("status") or "search_ready"),
        )
        theme = outcome.get("theme") or {}
        track_id = str(outcome.get("track_id") or "")
        return BatchUploadItem(
            filename=filename,
            success=True,
            status="queued" if track_id else "stored",
            chars=processed.total_chars,
            track_id=track_id,
            message=(
                f"Тема: {theme.get('collection', 'misc')}/{theme.get('theme_name', 'unclassified')}; "
                f"объектов: {len(processed.objects)}; фактов: {len(facts)}; runtime-index строится в тематическом shard." if track_id else "Stage 1: сохранено в search-ready knowledge_store без LightRAG runtime."
            ),
        )
    except Exception as exc:
        return BatchUploadItem(filename=filename, success=False, status="failed", message=str(exc))
    finally:
        if saved_path.exists():
            saved_path.unlink(missing_ok=True)

async def _process_upload(file: UploadFile) -> BatchUploadItem:
    filename = _client_filename(file.filename)
    if not filename:
        return BatchUploadItem(filename="без_имени", success=False, message="Имя файла не указано")

    suffix = _client_suffix(filename)
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS))
        return BatchUploadItem(
            filename=filename,
            success=False,
            message=f"Неподдерживаемый формат. Доступно: {supported}",
        )

    config = config_manager.load()
    await rag_service.ensure_initialized(config)

    safe_name = f"{uuid.uuid4().hex}_{_safe_upload_leaf(filename)}"
    saved_path = UPLOAD_DIR / safe_name
    numeric_extractor = get_numeric_extractor()

    try:
        with saved_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        processed = await asyncio.to_thread(process_document, saved_path, original_name=filename)
        facts = []
        for obj in processed.objects:
            facts.extend(
                numeric_extractor.extract(
                    obj.text,
                    source_name=processed.source_name,
                    object_id=obj.object_id,
                    object_type=obj.object_type,
                    metadata={**obj.metadata, "source_hash": processed.source_hash},
                )
            )
        rag_service.knowledge_store.upsert_processed_document(
            processed,
            facts,
            original_path=filename,
        )
        outcome = await rag_service.enqueue_documents_batch(
            [(obj.to_lightrag_text(), obj.citation_name) for obj in processed.objects]
        )
    except Exception as exc:
        if saved_path.exists():
            saved_path.unlink(missing_ok=True)
        return BatchUploadItem(filename=filename, success=False, message=str(exc))

    if saved_path.exists():
        saved_path.unlink(missing_ok=True)

    return BatchUploadItem(
        filename=filename,
        success=bool(outcome.get("items")),
        status="queued",
        chars=processed.total_chars,
        track_id=str(outcome.get("track_id") or ""),
        message=f"Документ разложен на объектов: {len(processed.objects)}; структурированных фактов: {len(facts)}.",
    )

@app.post("/api/knowledge/resume")
async def resume_knowledge_processing():
    config = config_manager.load()
    if config.topic_sharding_enabled:
        return {"status": "noop", "message": "В topic-sharded режиме обработка запускается по тематическим shards при загрузке или rebuild темы."}
    await rag_service.ensure_initialized(config)
    rag_service.schedule_document_processing()
    return {"status": "started", "message": "Обработка очереди документов запущена в фоне."}


@app.get("/api/knowledge/track/{track_id}", response_model=TrackStatusResponse)
async def knowledge_track_status(track_id: str):
    config = config_manager.load()
    try:
        if config.topic_sharding_enabled:
            return await theme_shards.get_track_status(track_id.strip())
        await rag_service.ensure_initialized(config)
        return await rag_service.get_track_status(track_id.strip())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.delete("/api/knowledge/documents/{doc_id}", response_model=DeleteDocumentResponse)
async def delete_knowledge_document(doc_id: str):
    doc_id = doc_id.strip()
    if not doc_id:
        raise HTTPException(status_code=400, detail="ID документа не указан")

    config = config_manager.load()
    await rag_service.ensure_initialized(config)

    try:
        outcome = await rag_service.delete_document(doc_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Не удалось удалить документ: {exc}") from exc

    status = outcome.get("status")
    message = outcome.get("message") or "Не удалось удалить документ."

    if status == "success":
        filename = outcome.get("filename") or doc_id
        return DeleteDocumentResponse(
            status="success",
            doc_id=doc_id,
            message=f"Документ «{filename}» удалён из базы знаний.",
            filename=filename,
        )
    if status == "not_found":
        raise HTTPException(status_code=404, detail=message)
    if status == "not_allowed":
        raise HTTPException(status_code=409, detail=message)

    raise HTTPException(status_code=502, detail=message)


@app.get("/api/knowledge/stats")
async def knowledge_stats():
    config = config_manager.load()
    try:
        if config.topic_sharding_enabled:
            state = await theme_shards.knowledge_state()
            themes = theme_shards.list_themes()
            documents = theme_shards.list_documents()
            return {
                "entities": sum(int((row.get("stats") or {}).get("entities") or 0) for row in themes),
                "labels_preview": [str(row.get("theme_name") or row.get("theme_id")) for row in themes[:12]],
                "documents": documents,
                "themes": themes,
                "active_ingestion_runs": _active_ingestion_runs(),
                "topic_sharding_enabled": True,
                "working_dir": str(theme_shards.themes_runtime_root),
                **state,
            }
        await rag_service.ensure_initialized(config)
        stats = await rag_service.get_stats()
        stats["structured_facts"] = rag_service.fact_store.stats()
        stats["topic_sharding_enabled"] = False
        return stats
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/knowledge/store")
async def knowledge_store_stats():
    config = config_manager.load()
    if config.topic_sharding_enabled:
        return {"themes": theme_shards.list_themes(), "router": theme_shards.catalog.build_router()}
    await rag_service.ensure_initialized(config)
    validation = rag_service.knowledge_store.validate()
    return {"stats": rag_service.knowledge_store.stats(), "validation": validation}


@app.post("/api/knowledge/rebuild-from-store")
async def rebuild_knowledge_from_store():
    config = config_manager.load()
    try:
        if config.topic_sharding_enabled:
            results = {}
            for row in theme_shards.list_themes():
                theme_id = str(row.get("theme_id") or "")
                if theme_id:
                    results[theme_id] = await theme_shards.rebuild_theme_runtime(theme_id, batch_size=config.topic_batch_size_docs)
            return {"themes": results}
        await rag_service.ensure_initialized(config)
        return await rag_service.rebuild_runtime_index_from_store()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Не удалось пересобрать runtime-индекс: {exc}") from exc



def _new_ingestion_run(stage: str, payload: dict[str, Any]) -> str:
    run_id = f"{stage}:{uuid.uuid4().hex[:12]}"
    INGESTION_RUNS[run_id] = {
        "run_id": run_id,
        "stage": stage,
        "status": "queued",
        "payload": payload,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "total": 0,
        "processed": 0,
        "ok": 0,
        "skipped": 0,
        "failed": 0,
        "current": "",
        "messages": [],
        "errors": [],
        "phase": "queued",
        "progress_detail": "",
    }
    return run_id


def _update_run(run_id: str, **updates: Any) -> None:
    run = INGESTION_RUNS.get(run_id)
    if not run:
        return
    for key, value in updates.items():
        if key == "message":
            run.setdefault("messages", []).append({"at": utc_now_iso(), "message": str(value)})
            run["messages"] = run["messages"][-80:]
        elif key == "error":
            run.setdefault("errors", []).append({"at": utc_now_iso(), "error": str(value)})
            run["errors"] = run["errors"][-80:]
        else:
            run[key] = value
    run["updated_at"] = utc_now_iso()


def _collect_existing_original_paths() -> set[str]:
    existing: set[str] = set()
    root = theme_shards.themes_store_root
    if not root.exists():
        return existing
    for theme_dir in root.iterdir():
        if not theme_dir.is_dir():
            continue
        for row in jsonl_read(theme_dir / "sources.jsonl"):
            original_path = str(row.get("original_path") or "").replace("\\", "/")
            if original_path:
                existing.add(original_path)
    return existing


async def _run_stage1_web(run_id: str, payload: Stage1StartRequest) -> None:
    _update_run(run_id, status="running", message="Stage 1 started")
    config = config_manager.load()
    theme_shards.update_config(config)
    profile = get_ingestion_profile(config, payload.profile)
    root = Path(payload.root_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise RuntimeError(f"root_path не существует или не является папкой: {root}")
    limit = payload.limit if payload.limit and payload.limit > 0 else None
    files = list(iter_supported_documents(root, config, limit=limit))
    _update_run(run_id, total=len(files), message=f"Stage 1 files selected: {len(files)}")
    extractor = get_numeric_extractor()
    existing = _collect_existing_original_paths() if payload.skip_existing else set()
    ok = skipped = failed = 0
    report_rows: list[dict[str, Any]] = []
    build_runtime_index = bool(payload.build_runtime_index or profile.get("build_runtime_index", False))
    status_if_no_runtime = str(profile.get("status") or "search_ready")
    try:
        for idx, (path, rel) in enumerate(files, start=1):
            task = INGESTION_TASKS.get(run_id)
            if task and task.cancelled():
                raise asyncio.CancelledError()
            _update_run(run_id, processed=idx - 1, current=rel)
            if rel in existing:
                skipped += 1
                report_rows.append({"idx": idx, "status": "skipped", "file_path": str(path), "relative_path": rel})
                continue
            try:
                processed_doc = await asyncio.to_thread(process_document, path, original_name=rel)
                facts = []
                for obj in processed_doc.objects:
                    facts.extend(extractor.extract(
                        obj.text,
                        source_name=processed_doc.source_name,
                        object_id=obj.object_id,
                        object_type=obj.object_type,
                        metadata={**obj.metadata, "source_hash": processed_doc.source_hash},
                    ))
                result = await theme_shards.ingest_processed_document(
                    processed_doc,
                    facts,
                    original_path=rel,
                    build_runtime_index=build_runtime_index,
                    compute_graph_metrics=bool(profile.get("compute_graph_metrics_during_ingest", False)),
                    status_if_no_runtime=status_if_no_runtime,
                )
                theme = result.get("theme") or {}
                ok += 1
                report_rows.append({
                    "idx": idx,
                    "status": "ok",
                    "file_path": str(path),
                    "relative_path": rel,
                    "theme_id": theme.get("theme_id"),
                    "collection": theme.get("collection"),
                    "theme_name": theme.get("theme_name"),
                    "year": theme.get("year"),
                    "source_type": theme.get("source_type"),
                    "confidence": theme.get("confidence"),
                    "chunks": len(processed_doc.objects),
                    "numeric_facts": len(facts),
                    "error": "",
                })
                _update_run(run_id, ok=ok, skipped=skipped, failed=failed, processed=idx, message=f"Stage 1 stored {rel} → {theme.get('theme_id')}")
            except Exception as exc:
                failed += 1
                report_rows.append({"idx": idx, "status": "failed", "file_path": str(path), "relative_path": rel, "error": str(exc)})
                _update_run(run_id, failed=failed, processed=idx, error=f"{rel}: {exc}")
        if payload.rebuild_graph_metrics_after and bool(profile.get("rebuild_graph_metrics_after", False)):
            _update_run(run_id, message="Rebuilding graph metrics")
            await asyncio.to_thread(theme_shards.compute_theme_graph_metrics)
        if payload.rebuild_theme_embeddings_after and bool(profile.get("rebuild_theme_embeddings_after", False)):
            _update_run(run_id, phase="theme_embeddings", progress_detail=f"selected themes={len(selected_theme_ids)}", message="Rebuilding theme embeddings")
            await theme_shards.rebuild_theme_embeddings()
        if payload.build_global_router_after and bool(profile.get("build_global_router_after", True)):
            _update_run(run_id, phase="global_router", progress_detail="build_router", message="Rebuilding global router")
            theme_shards.catalog.build_router()
        report_path = theme_shards.knowledge_root / "global" / f"stage1_web_{run_id.replace(':', '_')}.jsonl"
        for row in report_rows:
            jsonl_append(report_path, row)
        jsonl_append(theme_shards.knowledge_root / "global" / "two_stage_runs.jsonl", {
            "run_id": run_id,
            "stage": "stage1_web_fast_fill",
            "root": str(root),
            "limit": limit,
            "profile": payload.profile,
            "runtime_index": build_runtime_index,
            "ok": ok,
            "skipped": skipped,
            "failed": failed,
            "report": str(report_path),
            "created_at": utc_now_iso(),
        })
        _update_run(run_id, status="completed", current="", processed=len(files), ok=ok, skipped=skipped, failed=failed, report=str(report_path), message="Stage 1 completed; search is available via knowledge_store_lightweight")
    except asyncio.CancelledError:
        _update_run(run_id, status="cancelled", message="Stage 1 cancelled")
    except Exception as exc:
        _update_run(run_id, status="failed", error=str(exc))


def _norm_selector(text: str) -> str:
    import re
    return re.sub(r"[^0-9a-zа-я]+", "", (text or "").lower().replace("ё", "е"), flags=re.UNICODE)


def _filter_theme_rows_by_selectors(rows: list[dict[str, Any]], selectors: list[str]) -> list[dict[str, Any]]:
    """Accept theme_id, collection folder name, theme_name or partial folder token.

    Examples: "конференции", "ALTA", "конференции/ALTA", "конференции__ALTA_Ni_Co".
    """
    cleaned = [s.strip() for s in selectors if str(s or "").strip()]
    if not cleaned:
        return rows
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        values = [
            str(row.get("theme_id") or ""),
            str(row.get("collection") or ""),
            str(row.get("theme_name") or ""),
            str(row.get("relative_path") or ""),
            str(row.get("source_name") or ""),
        ]
        raw_blob = " / ".join(values).lower().replace("ё", "е")
        norm_values = [_norm_selector(v) for v in values]
        norm_blob = _norm_selector(" ".join(values))
        matched = False
        for selector in cleaned:
            sel_low = selector.lower().replace("ё", "е").strip()
            sel_norm = _norm_selector(selector)
            if not sel_norm:
                continue
            # Exact theme_id is strongest; collection exact selects the whole folder.
            if selector == str(row.get("theme_id") or "") or sel_norm in norm_values or sel_low in raw_blob or sel_norm in norm_blob:
                matched = True
                break
        if matched:
            theme_id = str(row.get("theme_id") or "")
            if theme_id not in seen:
                result.append(row)
                seen.add(theme_id)
    return result


async def _run_stage2_web(run_id: str, payload: Stage2StartRequest) -> None:
    _update_run(run_id, status="running", phase="select_themes", message="Stage 2 started")
    config = config_manager.load()
    theme_shards.update_config(config)
    profile = get_ingestion_profile(config, payload.profile)
    batch_size = int(payload.batch_size or profile.get("batch_size_docs") or config.topic_batch_size_docs)
    requested_graph_mode = str(payload.graph_mode or profile.get("graph_mode") or "retrieval_kg").strip().lower()
    compressed_lightrag_alias = requested_graph_mode in {"compressed_lightrag", "compressed_lightrag_kg", "compressed_kg_lightrag"}
    graph_mode = "compressed_kg" if compressed_lightrag_alias else requested_graph_mode
    build_runtime_index = bool(
        payload.build_runtime_index
        if payload.build_runtime_index is not None
        else profile.get("build_runtime_index", True if compressed_lightrag_alias else graph_mode == "full_kg")
    )
    if graph_mode == "compressed_kg" and not build_runtime_index and not compressed_lightrag_alias:
        graph_mode = "retrieval_kg"
    if graph_mode in {"cheap_kg", "vector_only", "retrieval_kg"}:
        build_runtime_index = False
    # LightRAG runtime builds must be bounded. A stale frontend may still send
    # wait=false/timeout=0; force controlled waiting for LightRAG modes so the
    # worker can mark timeout/fallback instead of leaving the UI in an endless run.
    is_lightrag_runtime_mode = bool(build_runtime_index and graph_mode in {"compressed_kg", "full_kg"})
    wait = bool(is_lightrag_runtime_mode or (build_runtime_index and payload.wait and profile.get("wait", True) and graph_mode not in {"cheap_kg", "vector_only", "retrieval_kg"}))
    clear_runtime = bool(payload.clear_runtime if payload.clear_runtime is not None else profile.get("clear_runtime", False))
    poll_interval = float(payload.poll_interval if payload.poll_interval is not None else profile.get("poll_interval", 10.0))
    # Web default must not wait forever. Retrieval KG does not use LightRAG;
    # LightRAG modes get a hard per-theme timeout even when the request says 0.
    timeout_sec = float(payload.timeout_sec if payload.timeout_sec is not None else profile.get("timeout_sec", 0.0 if not build_runtime_index else 1800.0))
    if is_lightrag_runtime_mode and timeout_sec <= 0:
        timeout_sec = 900.0 if graph_mode == "full_kg" else 600.0
    max_chunks_per_document = int(payload.max_chunks_per_document or profile.get("max_chunks_per_document_for_graph") or 8)
    min_kg_score = float(payload.min_kg_score if payload.min_kg_score is not None else profile.get("min_kg_score", 0.15))
    max_chunks_per_theme = payload.max_chunks_per_theme if payload.max_chunks_per_theme is not None else profile.get("max_chunks_per_theme", 0)
    max_chunks_per_theme = None if not max_chunks_per_theme or int(max_chunks_per_theme) <= 0 else int(max_chunks_per_theme)
    max_candidate_chunks_per_theme = payload.max_candidate_chunks_per_theme if payload.max_candidate_chunks_per_theme is not None else profile.get("max_candidate_chunks_per_theme", 1000)
    max_candidate_chunks_per_theme = None if not max_candidate_chunks_per_theme or int(max_candidate_chunks_per_theme) <= 0 else int(max_candidate_chunks_per_theme)
    max_parallel_themes = int(payload.max_parallel_themes or profile.get("max_parallel_themes") or config.topic_max_parallel_themes or 1)
    compressed_runtime_doc_mode = str(profile.get("compressed_runtime_doc_mode") or profile.get("compressed_lightrag_doc_mode") or "chunk").strip().lower()
    compressed_min_chunk_quality = float(profile.get("compressed_min_chunk_quality", 0.20))
    compressed_doc_type_aware = bool(profile.get("compressed_doc_type_aware", True))
    compressed_dynamic_document_limits = bool(payload.compressed_dynamic_document_limits if payload.compressed_dynamic_document_limits is not None else profile.get("compressed_dynamic_document_limits", True))
    compressed_short_document_max_chunks = int(payload.compressed_short_document_max_chunks or profile.get("compressed_short_document_max_chunks") or 3)
    compressed_long_document_max_chunks = int(payload.compressed_long_document_max_chunks or profile.get("compressed_long_document_max_chunks") or 6)
    target_status = str(profile.get("target_status") or ({"cheap_kg": "cheap_kg_ready", "compressed_kg": "compressed_kg_ready", "retrieval_kg": "retrieval_kg_ready", "full_kg": "full_kg_ready"}.get(graph_mode, "search_ready")))
    rows = sorted(theme_shards.list_themes(), key=lambda row: (str(row.get("collection") or ""), str(row.get("theme_name") or ""), str(row.get("theme_id") or "")))
    theme_selectors = [x for x in payload.theme_ids if x]
    if theme_selectors:
        rows = _filter_theme_rows_by_selectors(rows, theme_selectors)
        if not rows:
            raise RuntimeError(f"Не найдены темы/коллекции по селекторам: {', '.join(theme_selectors)}")
    elif build_runtime_index and graph_mode in {"compressed_kg", "full_kg"}:
        # LightRAG can be launched without a theme selector. In that case run it
        # for all parsed/searchable themes, instead of filtering everything out
        # because target_status=search_ready is already reached after Stage 1.
        rows = [row for row in rows if _READINESS_ORDER.get(str(row.get("status") or "not_ready"), 0) >= _READINESS_ORDER["parsed_ready"]]
    elif not payload.force:
        target_rank = _READINESS_ORDER.get(target_status, _READINESS_ORDER["compressed_kg_ready"])
        rows = [row for row in rows if _READINESS_ORDER.get(str(row.get("status") or "not_ready"), 0) < target_rank]
    if payload.max_themes is not None:
        rows = rows[: max(0, payload.max_themes)]
    _update_run(run_id, total=len(rows), processed=0, phase="selected", progress_detail=f"graph_mode={graph_mode}; runtime={build_runtime_index}; wait={wait}; clear_runtime={clear_runtime}", message=f"Stage 2 themes selected: {len(rows)}")
    print(f"[stage2:{run_id}] selected themes={len(rows)} graph_mode={graph_mode} build_runtime_index={build_runtime_index} wait={wait} clear_runtime={clear_runtime} timeout_sec={timeout_sec} max_chunks_per_doc={max_chunks_per_document} max_parallel_themes={max_parallel_themes} max_chunks_per_theme={max_chunks_per_theme} max_candidate_chunks_per_theme={max_candidate_chunks_per_theme} compressed_runtime_doc_mode={compressed_runtime_doc_mode} compressed_min_chunk_quality={compressed_min_chunk_quality} doc_type_aware={compressed_doc_type_aware} dynamic_doc_limits={compressed_dynamic_document_limits}")
    ok = failed = timeout_count = 0
    results: list[dict[str, Any]] = []
    try:
        for idx, row in enumerate(rows, start=1):
            theme_id = str(row.get("theme_id") or "")
            if not theme_id:
                continue
            _update_run(run_id, processed=idx - 1, current=theme_id, message=f"Stage 2 {graph_mode} for {theme_id}")
            print(f"[stage2:{run_id}] [{idx}/{len(rows)}] theme={theme_id} mode={graph_mode} runtime={build_runtime_index}")
            try:
                # Deterministic Stage 2 is intentionally executed in a worker
                # thread, because large JSONL scans/graph-plan writes should not
                # block /api/health and UI progress polling.
                if not build_runtime_index or graph_mode in {"cheap_kg", "vector_only", "retrieval_kg"}:
                    _update_run(
                        run_id,
                        phase="deterministic_kg",
                        progress_detail=f"{theme_id}: build {graph_mode} without LightRAG",
                        message=f"Building deterministic {graph_mode} for {theme_id}",
                    )
                    deterministic_timeout = float(profile.get("deterministic_timeout_sec", 300.0) or 300.0)
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            theme_shards.rebuild_theme_deterministic_kg,
                            theme_id,
                            graph_mode=graph_mode if graph_mode in {"cheap_kg", "compressed_kg", "vector_only", "retrieval_kg"} else "compressed_kg",
                            max_chunks_per_document_for_graph=max_chunks_per_document,
                            min_kg_score=min_kg_score,
                            max_chunks_per_theme=max_chunks_per_theme,
                            max_candidate_chunks_per_theme=max_candidate_chunks_per_theme,
                            target_status=target_status,
                            compute_graph_metrics=False,
                            compressed_runtime_doc_mode=compressed_runtime_doc_mode,
                            compressed_min_chunk_quality=compressed_min_chunk_quality,
                            compressed_doc_type_aware=compressed_doc_type_aware,
                            compressed_dynamic_document_limits=compressed_dynamic_document_limits,
                            compressed_short_document_max_chunks=compressed_short_document_max_chunks,
                            compressed_long_document_max_chunks=compressed_long_document_max_chunks,
                        ),
                        timeout=deterministic_timeout,
                    )
                else:
                    # LightRAG is allowed only as an explicit runtime mode. In
                    # web mode wait=False is preferred: the Stage 2 run enqueues
                    # shortened chunks and returns control to the UI instead of
                    # waiting for LightRAG's final graph write.
                    _update_run(
                        run_id,
                        phase="lightrag_enqueue" if not wait else "lightrag_build",
                        progress_detail=f"{theme_id}: runtime={build_runtime_index}, wait={wait}",
                        message=f"LightRAG {graph_mode} for {theme_id}: {'wait' if wait else 'enqueue only'}",
                    )
                    theme_chunks = int((row.get("stats") or {}).get("chunks") or 0)
                    estimated_runtime_chunks = min(theme_chunks, max_chunks_per_theme or theme_chunks) if graph_mode == "compressed_kg" else theme_chunks
                    batches = max(1, (estimated_runtime_chunks + max(1, batch_size) - 1) // max(1, batch_size))
                    whole_theme_timeout = None
                    if timeout_sec and timeout_sec > 0:
                        whole_theme_timeout = max(float(timeout_sec) + 300.0, float(timeout_sec) * min(batches, 8))
                    coro = theme_shards.rebuild_theme_runtime(
                        theme_id,
                        clear_runtime=clear_runtime,
                        batch_size=batch_size,
                        wait=wait,
                        poll_interval=poll_interval,
                        timeout_sec=timeout_sec,
                        graph_mode=graph_mode,
                        max_chunks_per_document_for_graph=max_chunks_per_document,
                        min_kg_score=min_kg_score,
                        max_chunks_per_theme=max_chunks_per_theme,
                        max_candidate_chunks_per_theme=max_candidate_chunks_per_theme,
                        build_runtime_index=build_runtime_index,
                        target_status=target_status,
                        compute_graph_metrics=False,
                        compressed_runtime_doc_mode=compressed_runtime_doc_mode,
                        compressed_min_chunk_quality=compressed_min_chunk_quality,
                        compressed_doc_type_aware=compressed_doc_type_aware,
                        compressed_dynamic_document_limits=compressed_dynamic_document_limits,
                        compressed_short_document_max_chunks=compressed_short_document_max_chunks,
                        compressed_long_document_max_chunks=compressed_long_document_max_chunks,
                    )
                    result = await asyncio.wait_for(coro, timeout=whole_theme_timeout) if whole_theme_timeout else await coro
                theme_timeouts = int(result.get("timeout_count") or 0)
                timeout_count += theme_timeouts
                if theme_timeouts:
                    _update_run(run_id, error=f"{theme_id}: LightRAG timeout; theme remains searchable via Stage 1 lightweight search")
                ok += 1
                result_short = {
                    "theme_id": theme_id,
                    "ok": True,
                    "status": result.get("status"),
                    "graph_mode": result.get("graph_mode") or graph_mode,
                    "chunks": result.get("chunks"),
                    "compressed_plan": result.get("compressed_plan"),
                    "track_ids": result.get("track_ids"),
                    "failed_count": result.get("failed_count"),
                    "timeout_count": theme_timeouts,
                }
                results.append(result_short)
                _update_run(run_id, ok=ok, failed=failed, processed=idx, timeout_count=timeout_count, message=f"Stage 2 theme done: {theme_id} status={result.get('status')}")
                print(f"[stage2:{run_id}] done theme={theme_id} status={result.get('status')} chunks={result.get('chunks')} timeout_count={theme_timeouts}")
            except Exception as exc:
                failed += 1
                results.append({"theme_id": theme_id, "ok": False, "error": str(exc)})
                try:
                    # A failed/timeout full/compressed LightRAG build must not make the
                    # theme unavailable. Keep deterministic Stage-1/cheap KG search.
                    if graph_mode in {"compressed_kg", "full_kg"}:
                        theme_shards.mark_theme_status(theme_id, "cheap_kg_ready")
                except Exception:
                    pass
                _update_run(run_id, failed=failed, processed=idx, error=f"{theme_id}: {exc}; kept as cheap_kg_ready/searchable if Stage 1 exists")
        selected_theme_ids = [str(row.get("theme_id")) for row in rows if row.get("theme_id")]
        if payload.rebuild_graph_metrics_after and bool(profile.get("rebuild_graph_metrics_after", True)):
            _update_run(run_id, phase="graph_metrics", progress_detail=f"selected themes={len(selected_theme_ids)}", message="Rebuilding graph metrics for selected themes")
            print(f"[stage2:{run_id}] post: graph_metrics themes={len(selected_theme_ids)}")
            await asyncio.to_thread(theme_shards.compute_theme_graph_metrics, None, selected_theme_ids)
        if payload.rebuild_theme_embeddings_after and bool(profile.get("rebuild_theme_embeddings_after", False)):
            _update_run(run_id, phase="theme_embeddings", progress_detail=f"selected themes={len(selected_theme_ids)}", message="Rebuilding theme embeddings")
            print(f"[stage2:{run_id}] post: theme_embeddings themes={len(selected_theme_ids)}")
            await theme_shards.rebuild_theme_embeddings(selected_theme_ids)
        else:
            print(f"[stage2:{run_id}] post: theme_embeddings skipped")
        if payload.build_global_router_after and bool(profile.get("build_global_router_after", True)):
            _update_run(run_id, phase="global_router", progress_detail="build_router", message="Rebuilding global router")
            print(f"[stage2:{run_id}] post: global_router")
            theme_shards.catalog.build_router()
        jsonl_append(theme_shards.knowledge_root / "global" / "two_stage_runs.jsonl", {
            "run_id": run_id,
            "stage": "stage2_web_build",
            "profile": payload.profile,
            "selected_themes": [str(row.get("theme_id")) for row in rows],
            "batch_size": batch_size,
            "graph_mode": graph_mode,
            "max_chunks_per_document": max_chunks_per_document,
            "min_kg_score": min_kg_score,
            "compressed_min_chunk_quality": compressed_min_chunk_quality,
            "compressed_doc_type_aware": compressed_doc_type_aware,
            "compressed_dynamic_document_limits": compressed_dynamic_document_limits,
            "compressed_short_document_max_chunks": compressed_short_document_max_chunks,
            "compressed_long_document_max_chunks": compressed_long_document_max_chunks,
            "max_chunks_per_theme": max_chunks_per_theme,
            "max_candidate_chunks_per_theme": max_candidate_chunks_per_theme,
            "max_parallel_themes": max_parallel_themes,
            "wait": wait,
            "timeout_sec": timeout_sec,
            "build_runtime_index": build_runtime_index,
            "ok": ok,
            "failed": failed,
            "timeout_count": timeout_count,
            "results": results,
            "created_at": utc_now_iso(),
        })
        _update_run(run_id, status="completed", phase="completed", progress_detail="", current="", processed=len(rows), ok=ok, failed=failed, timeout_count=timeout_count, results=results, message="Stage 2 completed")
    except asyncio.CancelledError:
        _update_run(run_id, status="cancelled", message="Stage 2 cancelled")
    except Exception as exc:
        _update_run(run_id, status="failed", error=str(exc))


@app.post("/api/ingestion/stage1/upload-batch")
async def upload_stage1_batch(session_id: str, files: list[UploadFile] = File(...)):
    """Stage browser-selected files without parsing them.

    This endpoint is intentionally cheap: the browser uploads the selected folder
    into a server-side staging directory, preserving relative paths. Real parsing
    and knowledge_store filling are performed by /stage1/staged/start as a
    background ingestion run, so the UI can poll progress reliably.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Не выбрано ни одного файла")
    sid = _safe_stage_session_id(session_id)
    root = STAGE1_STAGING_DIR / sid
    root.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    failed: list[dict[str, str]] = []
    for file in files:
        filename = _client_filename(file.filename)
        suffix = _client_suffix(filename)
        if suffix not in SUPPORTED_EXTENSIONS:
            failed.append({"filename": filename, "error": "unsupported_extension"})
            continue
        rel = _safe_relative_upload_path(filename)
        dest = root / rel
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved.append(str(rel).replace(os.sep, "/"))
        except Exception as exc:
            failed.append({"filename": filename, "error": str(exc)})
    return {
        "session_id": sid,
        "root_path": str(root),
        "saved": len(saved),
        "failed": len(failed),
        "files": saved[:20],
        "errors": failed[:20],
    }


@app.post("/api/ingestion/stage1/staged/start")
async def start_staged_stage1(payload: Stage1StagedStartRequest):
    if _is_ingestion_busy():
        raise HTTPException(status_code=409, detail="Уже выполняется ingestion-задача. Дождитесь завершения или отмените её.")
    sid = _safe_stage_session_id(payload.session_id)
    root = STAGE1_STAGING_DIR / sid
    if not root.exists():
        raise HTTPException(status_code=404, detail="Staging-сессия не найдена. Повторите выбор папки.")
    inner = Stage1StartRequest(
        root_path=str(root),
        limit=payload.limit,
        profile=payload.profile,
        skip_existing=payload.skip_existing,
        build_runtime_index=payload.build_runtime_index,
        rebuild_graph_metrics_after=payload.rebuild_graph_metrics_after,
        rebuild_theme_embeddings_after=payload.rebuild_theme_embeddings_after,
        build_global_router_after=payload.build_global_router_after,
    )
    run_id = _new_ingestion_run("stage1", {**payload.model_dump(), "root_path": str(root), "staged": True})

    async def runner() -> None:
        try:
            await _run_stage1_web(run_id, inner)
        finally:
            if payload.cleanup_after:
                shutil.rmtree(root, ignore_errors=True)

    task = asyncio.create_task(runner())
    INGESTION_TASKS[run_id] = task
    task.add_done_callback(lambda _: INGESTION_TASKS.pop(run_id, None))
    return {"run_id": run_id, "status": "queued", "message": "Stage 1 запущен из загруженной папки. Поиск станет доступен после уровня search_ready."}


@app.get("/api/ingestion/runs")
async def list_ingestion_runs():
    runs = sorted(INGESTION_RUNS.values(), key=lambda row: str(row.get("updated_at") or ""), reverse=True)
    return {"runs": runs[:50], "active": _active_ingestion_runs()}


@app.get("/api/ingestion/runs/{run_id}")
async def get_ingestion_run(run_id: str):
    run = INGESTION_RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Запуск ingestion не найден")
    return run


@app.post("/api/ingestion/runs/{run_id}/cancel")
async def cancel_ingestion_run(run_id: str):
    task = INGESTION_TASKS.get(run_id)
    if not task:
        raise HTTPException(status_code=404, detail="Активный запуск не найден")
    _update_run(run_id, status="cancelling", message="Cancellation requested")
    task.cancel()
    return {"run_id": run_id, "status": "cancelling"}


@app.post("/api/ingestion/stage1/start")
async def start_stage1(payload: Stage1StartRequest):
    if _is_ingestion_busy():
        raise HTTPException(status_code=409, detail="Уже выполняется ingestion-задача. Дождитесь завершения или отмените её.")
    run_id = _new_ingestion_run("stage1", payload.model_dump())
    task = asyncio.create_task(_run_stage1_web(run_id, payload))
    INGESTION_TASKS[run_id] = task
    task.add_done_callback(lambda _: INGESTION_TASKS.pop(run_id, None))
    return {"run_id": run_id, "status": "queued", "message": "Stage 1 запущен. Если limit не указан, будут обработаны все найденные файлы."}


@app.post("/api/ingestion/stage2/start")
async def start_stage2(payload: Stage2StartRequest):
    if _is_ingestion_busy():
        raise HTTPException(status_code=409, detail="Уже выполняется ingestion-задача. Дождитесь завершения или отмените её.")
    run_id = _new_ingestion_run("stage2", payload.model_dump())
    task = asyncio.create_task(_run_stage2_web(run_id, payload))
    INGESTION_TASKS[run_id] = task
    task.add_done_callback(lambda _: INGESTION_TASKS.pop(run_id, None))
    return {"run_id": run_id, "status": "queued", "message": "Stage 2 запущен. По умолчанию строится поисковый KG без LightRAG; поиск Stage 1 остаётся доступным."}


@app.get("/api/themes")
async def list_themes():
    return {"themes": theme_shards.list_themes()}


@app.get("/api/themes/{theme_id}")
async def theme_status(theme_id: str):
    try:
        return await theme_shards.get_theme_status(theme_id.strip())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/themes/{theme_id}/rebuild")
async def rebuild_theme(
    theme_id: str,
    clear_runtime: bool = False,
    batch_size: int | None = None,
    graph_mode: str = "compressed_kg",
    max_chunks_per_document: int = 8,
    min_kg_score: float = 0.15,
):
    config = config_manager.load()
    try:
        return await theme_shards.rebuild_theme_runtime(
            theme_id.strip(),
            clear_runtime=clear_runtime,
            batch_size=batch_size or config.topic_batch_size_docs,
            graph_mode=graph_mode,
            wait=False,
            max_chunks_per_document_for_graph=max_chunks_per_document,
            min_kg_score=min_kg_score,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/themes/rebuild-router")
async def rebuild_theme_router():
    return theme_shards.catalog.build_router()


@app.post("/api/themes/graph-metrics/rebuild")
async def rebuild_all_graph_metrics():
    try:
        return theme_shards.compute_theme_graph_metrics()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/themes/{theme_id}/graph-metrics")
async def get_theme_graph_metrics(theme_id: str):
    try:
        result = theme_shards.compute_theme_graph_metrics(theme_id.strip())
        return result.get("themes", {}).get(theme_id.strip()) or result
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/themes/embeddings/rebuild")
async def rebuild_theme_embeddings():
    try:
        return await theme_shards.rebuild_theme_embeddings()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/routing/debug")
async def debug_routing(payload: DebugQueryRequest):
    try:
        return await theme_shards.debug_route(payload.message.strip())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/debug/query-parse")
async def debug_query_parse(payload: DebugQueryRequest):
    config = config_manager.load()
    await rag_service.ensure_initialized(config)
    return rag_service.parse_query_debug(payload.message.strip())


@app.post("/api/debug/extract")
async def debug_extract(payload: DebugExtractRequest):
    extractor = get_numeric_extractor()
    facts = extractor.extract(
        payload.text,
        source_name=payload.source_name,
        object_id="debug",
        object_type="debug_text",
    )
    return {"facts": [fact.to_dict() for fact in facts]}


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
