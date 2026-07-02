# OSR + LightRAG Platform

Гибридная аналитическая платформа на базе [LightRAG](https://github.com/HKUDS/LightRAG) с веб-интерфейсом в корпоративном стиле Норникель.

## Возможности

- **LightRAG Core** — инкрементальная индексация документов и построение графовой базы знаний
- **Настройка LLM** — URL OpenAI-совместимого API (Ollama, vLLM, OpenAI и др.)
- **Чат** — вопросы к базе знаний в режимах `hybrid`, `mix`, `local`, `global`, `naive`
- **Загрузка базы знаний** — TXT, MD, CSV, JSON, PDF

## Архитектура (MVP)

1. **Парсинг документов** — извлечение текста из файлов (PDF через pypdf)
2. **LightRAG Core** — граф сущностей + векторный индекс (NetworkX + NanoVectorDB)
3. **Семантический поиск** — гибридный режим LightRAG
4. **UI** — чат + настройки LLM + загрузка документов

Полная схема из ТЗ (OSR, PostgreSQL, Qdrant, re-ranking) заложена как следующий этап развития.

## Быстрый старт

```bash
cd light-rag-platform
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env
python run.py
```

Откройте http://localhost:8090

### Пример с Ollama

```bash
ollama pull llama3.2:latest
ollama pull nomic-embed-text:latest
```

В UI укажите:
- **URL LLM API:** `http://localhost:11434/v1`
- **Модель LLM:** `llama3.2:latest`
- **Модель эмбеддингов:** `nomic-embed-text:latest`
- **Размерность:** `768`

## Структура репозитория

```
light-rag-platform/
├── backend/           # FastAPI + LightRAG
├── frontend/          # UI (чат, настройки, загрузка)
├── vendor/LightRAG/   # исходники LightRAG (клон upstream)
├── data/              # хранилище графа и настроек
├── uploads/           # загруженные файлы
└── run.py
```

## API

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/api/health` | Статус сервиса |
| GET/PUT | `/api/config` | Настройки LLM |
| POST | `/api/knowledge/upload` | Загрузка документа в LightRAG |
| GET | `/api/knowledge/stats` | Статистика графа |
| POST | `/api/chat` | Вопрос к базе знаний |

## Лицензия

MIT. LightRAG — MIT (HKUDS).
