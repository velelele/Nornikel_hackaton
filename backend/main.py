from __future__ import annotations

import asyncio
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config_manager import AppConfig, ConfigManager
from backend.document_loader import SUPPORTED_EXTENSIONS, extract_text
from backend.rag_service import RagService

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

SETTINGS_PATH = ROOT / "data" / "settings.json"
UPLOAD_DIR = ROOT / "uploads"
WORKING_DIR = Path(os.getenv("RAG_WORKING_DIR", ROOT / "data" / "rag_storage"))
FRONTEND_DIR = ROOT / "frontend"

config_manager = ConfigManager(SETTINGS_PATH)
rag_service = RagService(WORKING_DIR)


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    config = config_manager.load()
    try:
        await rag_service.ensure_initialized(config)
        if _env_flag("RAG_AUTO_RESUME", default=False):
            rag_service.schedule_document_processing()
    except Exception as exc:
        print(f"[warn] База знаний не инициализирована при старте: {exc}")
    yield
    await rag_service.shutdown()


app = FastAPI(
    title="Научный клубок",
    description="Графовая база знаний для научных документов, экспериментов и материалов",
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
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_model: str | None = None
    embedding_dim: int | None = None
    embedding_api_key: str | None = None
    query_mode: str | None = None


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


@app.get("/api/health")
async def health():
    pipeline_busy = False
    if rag_service.is_ready:
        try:
            pipeline_busy = await rag_service.is_pipeline_busy()
        except Exception:
            pipeline_busy = False
    return {
        "status": "ok",
        "rag_ready": rag_service.is_ready,
        "pipeline_busy": pipeline_busy,
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
    try:
        await rag_service.reinitialize(config)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Не удалось применить настройки LLM: {exc}") from exc
    return config


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    config = config_manager.load()
    await rag_service.ensure_initialized(config)
    mode = payload.mode or config.query_mode

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
        sources=result["sources"],
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
async def upload_knowledge(file: UploadFile = File(...)):
    result = await _process_upload(file)
    if not result.success:
        raise HTTPException(status_code=400, detail=result.message)
    return UploadResponse(
        track_id=result.track_id,
        filename=result.filename,
        chars=result.chars,
        message=result.message,
    )


@app.post("/api/knowledge/upload/batch", response_model=BatchUploadResponse)
async def upload_knowledge_batch(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Не выбрано ни одного файла")

    config = config_manager.load()
    await rag_service.ensure_initialized(config)

    prepared: list[tuple[str, str, str, Path]] = []
    results: list[BatchUploadItem] = []

    for file in files:
        filename = file.filename or "без_имени"
        if not file.filename:
            results.append(BatchUploadItem(filename=filename, success=False, message="Имя файла не указано"))
            continue

        suffix = Path(file.filename).suffix.lower()
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

        safe_name = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
        saved_path = UPLOAD_DIR / safe_name

        try:
            with saved_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            text = await asyncio.to_thread(extract_text, saved_path)
            citation_name = Path(file.filename).name
            prepared.append((text, citation_name, filename, saved_path))
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
            outcomes_by_name = {item["filename"]: item for item in enqueue_result.get("items", [])}

            for text, citation_name, filename, saved_path in prepared:
                outcome = outcomes_by_name.get(citation_name) or outcomes_by_name.get(filename)
                if saved_path.exists():
                    saved_path.unlink(missing_ok=True)

                if not outcome:
                    results.append(
                        BatchUploadItem(
                            filename=filename,
                            success=False,
                            status="failed",
                            message="Не получен результат постановки в очередь.",
                        )
                    )
                    continue

                results.append(
                    BatchUploadItem(
                        filename=filename,
                        success=outcome.get("success", False),
                        status=str(outcome.get("status") or ("queued" if outcome.get("success") else "failed")),
                        chars=int(outcome.get("chars") or 0),
                        track_id=str(outcome.get("track_id") or batch_track_id),
                        message=outcome.get("message") or "Неизвестная ошибка.",
                    )
                )
        except Exception as exc:
            for _, _, filename, saved_path in prepared:
                if saved_path.exists():
                    saved_path.unlink(missing_ok=True)
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
            message = f"Документы поставлены в очередь ({succeeded}). Обработка идёт в фоне."
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


async def _process_upload(file: UploadFile) -> BatchUploadItem:
    filename = file.filename or "без_имени"
    if not file.filename:
        return BatchUploadItem(filename=filename, success=False, message="Имя файла не указано")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS))
        return BatchUploadItem(
            filename=filename,
            success=False,
            message=f"Неподдерживаемый формат. Доступно: {supported}",
        )

    config = config_manager.load()
    await rag_service.ensure_initialized(config)

    safe_name = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    saved_path = UPLOAD_DIR / safe_name

    try:
        with saved_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        text = await asyncio.to_thread(extract_text, saved_path)
        outcome = await rag_service.insert_text(text, citation_name=Path(file.filename).name)
    except Exception as exc:
        if saved_path.exists():
            saved_path.unlink(missing_ok=True)
        return BatchUploadItem(filename=filename, success=False, message=str(exc))

    if saved_path.exists():
        saved_path.unlink(missing_ok=True)

    return BatchUploadItem(
        filename=filename,
        success=outcome.get("success", False),
        status=str(outcome.get("status") or ("queued" if outcome.get("success") else "failed")),
        chars=int(outcome.get("chars") or 0),
        track_id=str(outcome.get("track_id") or ""),
        message=outcome.get("message") or "Неизвестная ошибка.",
    )


@app.post("/api/knowledge/resume")
async def resume_knowledge_processing():
    config = config_manager.load()
    await rag_service.ensure_initialized(config)
    rag_service.schedule_document_processing()
    return {"status": "started", "message": "Обработка очереди документов запущена в фоне."}


@app.get("/api/knowledge/track/{track_id}", response_model=TrackStatusResponse)
async def knowledge_track_status(track_id: str):
    config = config_manager.load()
    await rag_service.ensure_initialized(config)
    try:
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
    await rag_service.ensure_initialized(config)
    try:
        return await rag_service.get_stats()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
