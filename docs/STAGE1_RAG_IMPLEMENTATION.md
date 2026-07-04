# Этап 1. Доменный RAG для Nornikel Hackaton

## Что добавлено

1. `lightrag-hku` используется как pip-зависимость вместо пустого `vendor/LightRAG`.
2. Добавлен доменный слой `backend/domain`:
   - `ontology.yaml` — минимальная RU/EN онтология материалов, процессов, оборудования, свойств и KG-соседей;
   - `query_parser.py` — domain query parser: сущности, классы, числовые ограничения, география, временной диапазон, intent;
   - `numeric_extractor.py` — извлечение чисел, диапазонов, операторов и единиц измерения;
   - `document_processor.py` + `chunker.py` — metadata-aware chunking, страницы PDF, таблицы DOCX/XLSX/CSV как отдельные объекты;
   - `fact_store.py` — JSONL sidecar для структурированных фактов;
   - `query_expansion.py` — controlled expansion через словарь, KG-соседей и найденные top-k факты;
   - `optimizer.py` — заготовка PSO/stigmergy/niching слоя для offline-подбора retrieval-параметров.
3. Добавлены debug endpoints:
   - `POST /api/debug/query-parse`
   - `POST /api/debug/extract`
4. `/api/knowledge/stats` теперь возвращает статистику структурированных фактов.
5. Загрузка документов теперь сначала раскладывает файл на объекты, извлекает числовые факты, затем ставит объекты в LightRAG.

## Проверка на Windows

```bat
cd Nornikel_hackaton-main
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
python run.py
```

Для Ollama/OpenAI-compatible embeddings рекомендуется начать с:

```bat
ollama pull qwen2.5:7b
ollama pull bge-m3
```

В `.env`:

```env
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:7b
LLM_API_KEY=ollama

EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_MODEL=bge-m3
EMBEDDING_DIM=1024
EMBEDDING_API_KEY=ollama
```

Альтернативы:

- `nomic-embed-text:latest`, обычно `EMBEDDING_DIM=768`;
- OpenAI-compatible сервер с `BAAI/bge-m3`;
- OpenAI-compatible сервер с Qwen embedding model, если он поднят локально.

## CLI импорт корпуса

```bat
python scripts\ingest_folder.py D:\data\datas_unpacked --limit 5
```

Или zip:

```bat
python scripts\ingest_folder.py D:\data\datas.zip --limit 5
```

RAR внутри zip пока не распаковывается автоматически. Для Windows его лучше заранее распаковать через 7-Zip в папку корпуса.

## Проверка domain query parser

```bat
curl -X POST http://localhost:8090/api/debug/query-parse ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"Какие технические решения циркуляции католита при электроэкстракции никеля описаны в мировой практике и какая скорость потока оптимальна?\"}"
```

Ожидается, что parser найдёт:

- `electrowinning`
- `catholyte`
- `nickel`
- `flow_rate`
- geography: `foreign`, если есть формулировка `мировая практика` / `зарубежная`.

## Проверка числового извлечения

```bat
curl -X POST http://localhost:8090/api/debug/extract ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"Исходная вода содержит сульфаты 200-300 мг/л, хлориды <=300 мг/л, сухой остаток ≤1000 мг/дм³.\"}"
```

Ожидаются факты по концентрации/сухому остатку с единицами `mg/L` или `mg/dm3`.

## Границы этапа 1

Это ещё не production KG. На этом этапе структурированные факты хранятся в JSONL sidecar, а графовая часть остаётся в LightRAG. Следующий этап — перенос ABox-фактов в Neo4j, SHACL-валидация и полноценные Cypher-запросы.
