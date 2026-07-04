import re
from typing import Dict, Any, List

from backend.theme_sharding import ThemeShardManager


class ChainOfThoughts:
    """
    Chain-of-Thoughts reasoning pipeline для NiCo системы.
    Адаптировано под ThemeShardManager + retrieval_kg + lightweight search.
    """

    def __init__(self, shard_manager: ThemeShardManager):
        self.manager = shard_manager
        self.doc_num: str | None = None

    async def _call_llm(self, prompt: str, system_prompt: str | None = None) -> str:
        """Удобный shortcut до LLM"""
        if system_prompt is None:
            system_prompt = (
                "Ты — строгий эксперт в металлургии и нормативно-технической документации. "
                "Отвечай точно, лаконично, сохраняя единицы измерения и ссылки на источники."
            )
        return await self.manager._call_llm_for_reasoning(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.25,
            max_tokens=1100
        )

    # ====================== ШАГ 1: Извлечение номера НТД ======================
    async def _extract_doc_number(self, question: str) -> str | None:
        prompt = f"""Найди в запросе идентификатор нормативно-технического документа (ГОСТ, ТУ, СТО и т.д.).

Запрос: {question}

Правила:
- Ищи конструкции вида: ГОСТ 12345-2016, ТУ 14-123-2005 и т.п.
- Верни только номер документа (например "ГОСТ 12345-2016") или слово "нет".
- Ничего больше не пиши.

Ответ:"""

        answer = await self._call_llm(prompt)
        answer = answer.strip()

        if "нет" in answer.lower() or len(answer) < 4:
            return None

        # Простой парсинг
        match = re.search(r'(ГОСТ|ТУ|СТО|ОСТ)\s*[\d\-]+', answer, re.IGNORECASE)
        if match:
            self.doc_num = match.group(0).upper()
            return self.doc_num
        return None

    # ====================== ШАГ 2: Определение типа данных ======================
    async def _classify_query_type(self, question: str) -> str:
        prompt = f"""Проанализируй запрос и определи, в каком типе данных наиболее вероятно содержится ответ.

Запрос: {question}

Ответь только одним словом:
- текст
- таблица
- схема
- смешанный

Ответ:"""

        answer = (await self._call_llm(prompt)).lower()
        if "таблиц" in answer:
            return "table"
        elif any(x in answer for x in ["схем", "рисун", "чертеж"]):
            return "schema"
        return "text"

    # ====================== Поиск по тексту ======================
    async def _find_by_text(self, question: str, selected_themes: List[str]) -> Dict[str, Any]:
        context = []
        for theme_id in selected_themes[:4]:
            try:
                result = self.manager._query_theme_lightweight(theme_id, question, limit=12)
                if result and result.get("matches"):
                    context.extend(result["matches"][:10])
            except:
                continue

        if not context:
            return {"answer": "Не удалось найти релевантную информацию.", "sources": []}

        prompt = self._make_final_text_prompt(question, context)
        final_answer = await self._call_llm(prompt)

        return {
            "answer": final_answer,
            "sources": context[:8],
            "reasoning": "text_search"
        }

    # ====================== Поиск по таблицам (заглушка) ======================
    async def _find_by_tables(self, question: str, selected_themes: List[str]) -> Dict[str, Any]:
        # TODO: в будущем здесь будет полноценная работа с table_meta
        print("[CoT] Табличный поиск пока в разработке → fallback на текст")
        return await self._find_by_text(question, selected_themes)

    # ====================== Финальный промпт ======================
    def _make_final_text_prompt(self, question: str, context: List[Dict]) -> str:
        context_str = "\n\n".join([
            f"[Источник {i+1}] (тема: {c.get('theme_name','')})\n{c.get('text') or c.get('content','')[:900]}"
            for i, c in enumerate(context)
        ])

        return f"""Ты — эксперт-металлург. Используй только информацию из контекста ниже.

Вопрос: {question}

Контекст:
{context_str}

Требования к ответу:
- Будь точен и технически корректен.
- Сохраняй единицы измерения.
- Приводи ссылки на источники в формате [1], [2].
- Если информации недостаточно — прямо скажи об этом.

Ответ:"""

    # ====================== ГЛАВНЫЙ МЕТОД ======================
    async def process(self, question: str, selected_themes: List[str] | None = None) -> Dict[str, Any]:
        """
        Основной входной метод Chain of Thoughts.
        """
        steps: List[Dict] = []

        # Шаг 1: Номер документа
        self.doc_num = await self._extract_doc_number(question)
        steps.append({"step": "extract_doc_num", "result": self.doc_num or "не найден"})

        # Шаг 2: Тип запроса
        query_type = await self._classify_query_type(question)
        steps.append({"step": "classify_type", "result": query_type})

        # Шаг 3: Поиск
        if query_type == "table":
            result = await self._find_by_tables(question, selected_themes or [])
        else:
            result = await self._find_by_text(question, selected_themes or [])

        result["steps"] = steps
        result["doc_num"] = self.doc_num
        return result