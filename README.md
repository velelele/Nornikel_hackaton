# ◈NiCo

Доменная RAG/KG-платформа для горно-металлургических R&D-документов: LightRAG + metadata-aware parsing + structured numeric fact extraction + RU/EN domain query expansion.

## Что изменено в текущей сборке

- Конфигурация перенесена в `config.yaml`. Старый `data/settings.json` больше не является источником истины и не сможет случайно вернуть `nomic-embed-text:latest`.
- Дефолтный локальный профиль: `qwen3:4b` для генерации и `bge-m3` для embeddings с размерностью `1024`.
- В UI добавлена исправленная кнопка **«Загрузить папку»**: picker создаётся динамически и использует `webkitdirectory`.
- Документы из папки загружаются постепенно: по одному файлу, с ожиданием завершения построения графа для текущего пакета перед следующим.
- Относительный путь файла из папки сохраняется в имени источника, например `subdir/report.pdf`.
- Чат заблокирован, пока граф знаний не построен: нужен хотя бы один обработанный документ и отсутствие активного ingestion.
- Название приложения возвращено к `◈NiCo`; этот же маркер используется в чате и системном prompt.

## Быстрый старт

```powershell
cd D:\dev\metallum-rag
conda activate metallum-rag

pip install -r requirements.txt
# optional full parsing profile
pip install -r requirements-parsing-full.txt

ollama pull qwen3:4b
ollama pull bge-m3

python run.py
```

Открыть:

```text
http://localhost:8090
```

## Основной config.yaml

```yaml
app:
  symbol: "◈NiCo"
  title: "◈NiCo"

server:
  host: "0.0.0.0"
  port: 8090
  reload: false

rag:
  working_dir: "./data/rag_storage"
  auto_resume: false

models:
  llm:
    base_url: "http://localhost:11434/v1"
    model: "qwen3:4b"
    api_key: "ollama"
  embedding:
    base_url: "http://localhost:11434/v1"
    model: "bge-m3"
    dim: 1024
    api_key: "ollama"

retrieval:
  query_mode: "hybrid"

upload:
  store_dir: "./uploads"
```

`.env` теперь используется только для локальных override/secrets. YAML остаётся основной конфигурацией.

## Важная проверка перед ingestion

Проверь, что embedding-модель реально доступна в Ollama:

```powershell
ollama list
python -c "from openai import OpenAI; c=OpenAI(base_url='http://localhost:11434/v1', api_key='ollama'); r=c.embeddings.create(model='bge-m3', input=['тест']); print(len(r.data[0].embedding))"
```

Ожидаемо:

```text
1024
```

Если ранее запускался индекс с другой embedding-моделью, очисти старое состояние:

```powershell
if (Test-Path data) { Remove-Item -Recurse -Force data }
```

## Загрузка документов

UI поддерживает два режима:

1. **Загрузить документы** — выбор одного или нескольких файлов.
2. **Загрузить папку** — выбор папки; браузер отправляет документы рекурсивно с относительными путями. Загрузка идёт постепенно, по одному файлу, чтобы не создавать несколько конкурирующих задач LightRAG.

Поддерживаемые расширения:

```text
txt, md, csv, tsv, json, xml, yaml, yml, log, rtf,
pdf, doc, docx, ppt, pptx, xls, xlsx, html, htm, epub
```

Для smoke-теста сначала загружай 1–10 файлов, не весь корпус. Чат станет активен только после завершения обработки и построения графа знаний.

## CLI ingestion

```powershell
python scripts\ingest_folder.py D:\datasets\nornikel_sample --limit 10
python scripts\ingest_folder.py D:\datasets\datas.zip --limit 10
```

CLI также читает `config.yaml` и использует `rag.working_dir`.

## API

| Метод | Путь | Описание |
|---|---|---|
| GET | `/api/health` | Статус сервиса, символ и название приложения |
| GET/PUT | `/api/config` | Плоская проекция YAML-настроек для UI/API |
| POST | `/api/knowledge/upload` | Загрузка одного документа |
| POST | `/api/knowledge/upload/batch` | Загрузка нескольких файлов или файлов из папки |
| POST | `/api/knowledge/resume` | Возобновить обработку очереди |
| GET | `/api/knowledge/track/{track_id}` | Статус batch-ingestion |
| GET | `/api/knowledge/stats` | Статистика графа и structured facts |
| POST | `/api/debug/query-parse` | Проверка domain query parser |
| POST | `/api/debug/extract` | Проверка numeric extractor |
| POST | `/api/chat` | Вопрос к базе знаний |

## Рекомендуемый локальный профиль

- Python 3.11
- Conda env: `metallum-rag`
- LLM runtime: Ollama
- LLM: `qwen3:4b`
- Embeddings: `bge-m3`, `dim=1024`
- Full parsing profile: `requirements-parsing-full.txt`
- Проект лучше держать в пути без кириллицы: `D:\dev\metallum-rag`

## Stage 1: доменный RAG/KG слой

- LightRAG подключается через `lightrag-hku` из pip.
- Документы раскладываются на metadata-aware объекты: страницы PDF, chunk-и, таблицы DOCX/XLSX/CSV.
- Добавлен domain query parser: материалы, процессы, оборудование, числовые ограничения, география, временной диапазон.
- Добавлен controlled query expansion через RU/EN словарь, KG-соседей из `backend/domain/ontology.yaml` и top-k structured facts.
- Числовые факты сохраняются в `data/domain_facts.jsonl`.
- Добавлены debug endpoints: `/api/debug/query-parse`, `/api/debug/extract`.

Подробности: `docs/STAGE1_RAG_IMPLEMENTATION.md`.

## Durable knowledge_store между версиями

В этой версии `data/rag_storage` считается runtime-кэшем LightRAG, а не единственным источником истины. Извлечённые знания дополнительно сохраняются в `data/knowledge_store`:

```text
data/knowledge_store/
  manifest.json
  sources.jsonl
  source_fragments.jsonl
  chunks.jsonl
  numeric_facts.jsonl
  entities.jsonl
  relations.jsonl
  triples.jsonl
  claims.jsonl
  ingestion_runs.jsonl
```

Практическое правило:

```text
Сохранять между версиями: data/knowledge_store
Можно пересоздавать: data/rag_storage, data/vector_cache
```

Перед обновлением версии:

```powershell
cd D:\dev\metallum-rag
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
mkdir "D:\dev\nico_snapshots\$stamp"
robocopy data\knowledge_store "D:\dev\nico_snapshots\$stamp\knowledge_store" /MIR
copy config.yaml "D:\dev\nico_snapshots\$stamp\config.yaml"
```

Проверка durable-хранилища:

```powershell
python scripts\kb_validate.py
```

Экспорт knowledge_store в отдельную папку:

```powershell
python scripts\kb_export.py --out D:\dev\nico_exports\kb_20260703
```

Пересборка LightRAG runtime-индекса из knowledge_store без повторного parsing PDF/DOCX/PPTX:

```powershell
python scripts\kb_rebuild_index.py --clear-runtime --batch-size 64
```

API:

```text
GET  /api/knowledge/store
POST /api/knowledge/rebuild-from-store
```

`/api/knowledge/store` возвращает статистику и валидацию `knowledge_store`. `/api/knowledge/rebuild-from-store` переиспользует `chunks.jsonl` и заново ставит их в очередь LightRAG.

Если при обновлении рядом уже есть старый `data/domain_facts.jsonl`, ◈NiCo автоматически импортирует его в `data/knowledge_store/numeric_facts.jsonl`, если новое хранилище ещё пустое. Это сохраняет структурированные числовые факты, но не восстанавливает chunks для пересборки vector index. Для полной пересборки без parsing нужны `chunks.jsonl`, созданные новой версией.

## Topic-sharded ingestion for large corpora

For large corpora use the new topic-sharded mode. See [`README_TOPIC_SHARDS.md`](README_TOPIC_SHARDS.md).
