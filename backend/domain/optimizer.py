from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass(slots=True)
class RetrievalParams:
    chunk_size: int = 3200
    chunk_overlap: int = 300
    top_k: int = 12
    rerank_top_k: int = 50
    expansion_terms: int = 40


class ExpertFeedbackMemory:
    """Stigmergy-like память экспертной обратной связи.

    Эксперты оставляют оценки удачных/неудачных ответов. Эти оценки затем могут быть
    использованы как феромонный след для автоматического подбора retrieval-параметров.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, query: str, params: RetrievalParams, score: float, comment: str = "") -> None:
        row = {"query": query, "params": asdict(params), "score": score, "comment": comment}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]


class SimplePSOOptimizer:
    """Лёгкий PSO для offline-подбора chunking/retrieval/reranking параметров.

    Это не runtime-зависимость. Используется в scripts/tune_retrieval.py на наборе competency questions.
    """

    bounds = {
        "chunk_size": (1200, 6000),
        "chunk_overlap": (80, 900),
        "top_k": (4, 30),
        "rerank_top_k": (20, 120),
        "expansion_terms": (8, 80),
    }

    def optimize(self, objective: Callable[[RetrievalParams], float], *, particles: int = 12, iterations: int = 20) -> RetrievalParams:
        keys = list(self.bounds)
        positions = [self._random_vector(keys) for _ in range(particles)]
        velocities = [{key: 0.0 for key in keys} for _ in range(particles)]
        personal_best = [dict(pos) for pos in positions]
        personal_scores = [objective(self._to_params(pos)) for pos in positions]
        best_idx = max(range(particles), key=lambda i: personal_scores[i])
        global_best = dict(personal_best[best_idx])
        global_score = personal_scores[best_idx]

        for _ in range(iterations):
            for i in range(particles):
                for key in keys:
                    lo, hi = self.bounds[key]
                    r1, r2 = random.random(), random.random()
                    velocities[i][key] = (
                        0.55 * velocities[i][key]
                        + 1.3 * r1 * (personal_best[i][key] - positions[i][key])
                        + 1.3 * r2 * (global_best[key] - positions[i][key])
                    )
                    positions[i][key] = min(hi, max(lo, positions[i][key] + velocities[i][key]))
                score = objective(self._to_params(positions[i]))
                if score > personal_scores[i]:
                    personal_scores[i] = score
                    personal_best[i] = dict(positions[i])
                    if score > global_score:
                        global_score = score
                        global_best = dict(positions[i])
        return self._to_params(global_best)

    def _random_vector(self, keys: list[str]) -> dict[str, float]:
        return {key: random.uniform(*self.bounds[key]) for key in keys}

    @staticmethod
    def _to_params(vector: dict[str, float]) -> RetrievalParams:
        return RetrievalParams(
            chunk_size=int(round(vector["chunk_size"])),
            chunk_overlap=int(round(vector["chunk_overlap"])),
            top_k=int(round(vector["top_k"])),
            rerank_top_k=int(round(vector["rerank_top_k"])),
            expansion_terms=int(round(vector["expansion_terms"])),
        )


def niche_key(answer_or_facts: str) -> str:
    """Грубый ключ ниши для multi-modal/niching retrieval.

    На следующем этапе сюда подключается кластеризация результатов по методам/процессам.
    """
    text = answer_or_facts.lower()
    if any(x in text for x in ("обратный осмос", "reverse osmosis", "мембран")):
        return "membrane_desalination"
    if any(x in text for x in ("ионный обмен", "ion exchange")):
        return "ion_exchange"
    if any(x in text for x in ("католит", "catholyte")):
        return "electrolyte_circulation"
    if any(x in text for x in ("штейн", "matte", "шлак", "slag")):
        return "matte_slag_distribution"
    return "general"
