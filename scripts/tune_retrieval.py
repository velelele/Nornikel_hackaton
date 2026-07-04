from __future__ import annotations

"""Заготовка offline-подбора параметров retrieval через PSO.

Запускать после подготовки competency questions и разметки expected_sources:
    python scripts/tune_retrieval.py

На первом этапе objective_stub нужно заменить на реальную метрику Recall@k / MRR / nDCG.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.domain.optimizer import RetrievalParams, SimplePSOOptimizer


def objective_stub(params: RetrievalParams) -> float:
    # Заглушка: штрафуем слишком мелкие chunks и слишком большой top_k.
    return -abs(params.chunk_size - 3200) / 3200 - abs(params.chunk_overlap - 300) / 900 - params.top_k / 100


def main() -> None:
    best = SimplePSOOptimizer().optimize(objective_stub, particles=8, iterations=8)
    print(best)


if __name__ == "__main__":
    main()
