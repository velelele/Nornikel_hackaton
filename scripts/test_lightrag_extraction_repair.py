from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.lightrag_extraction_repair import (  # noqa: E402
    harden_extraction_prompt,
    is_lightrag_extraction_prompt,
    repair_lightrag_extraction_output,
)

_SAMPLE_PROMPT = (
    "Extract entities and relationships from the input text. "
    "Use tuple delimiter: <|>; record delimiter: ##; completion delimiter: <|COMPLETE|>. "
    "Return entity and relationship records, plus content_keywords."
)

_SAMPLE_OUTPUT = """| Название | Тип | Описание |
|---|---|---|
| акселерограмма | Concept | запись ускорений при взрывных работах |
| велосиграмма | Concept | запись скоростей колебаний |
| сейсмограмма | Concept | источник для прямого динамического расчета |
| динамический расчет | Method | метод расчёта реакции конструкций на сейсмовзрывное воздействие |
| сейсмограмма | используется в | динамический расчет | для прямого динамического расчета зданий |
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test LightRAG extraction prompt hardening/output repair.")
    parser.add_argument("--input", type=Path, default=None, help="Optional raw LLM extraction output to repair")
    parser.add_argument("--prompt", type=Path, default=None, help="Optional LightRAG extraction prompt file")
    args = parser.parse_args()

    prompt = args.prompt.read_text(encoding="utf-8", errors="replace") if args.prompt else _SAMPLE_PROMPT
    raw = args.input.read_text(encoding="utf-8", errors="replace") if args.input else _SAMPLE_OUTPUT

    print("is_extraction_prompt:", is_lightrag_extraction_prompt(prompt))
    hardened_prompt, hardened_system = harden_extraction_prompt(prompt)
    print("\n[hardened prompt tail]")
    print(hardened_prompt[-900:])
    if hardened_system:
        print("\n[hardened system tail]")
        print(hardened_system[-900:])
    print("\n[repaired output]")
    print(repair_lightrag_extraction_output(raw, prompt=prompt))

    pathological = {
        "adjacent_records_without_delimiter": '("entity"<|>"Взрывные работы на поверхности"<|>"Concept"<|>"описание")("entity"<|>"Сейсмограмма"<|>"Concept"<|>"описание")<|COMPLETE|>',
        "overlong_entity_record": '("entity"<|>"Взрывные работы на поверхности"<|>"Concept"<|>"описание"<|>"лишнее"<|>"ещё")<|COMPLETE|>',
        "completion_marker_as_entity": '("entity"<|>"<"<|>"COMPLETE"<|>"bad")<|COMPLETE|>',
        "relation_missing_strength": '("relationship"<|>"сейсмограмма"<|>"динамический расчет"<|>"используется"<|>"используется в")<|COMPLETE|>',
    }
    for name, sample in pathological.items():
        print(f"\n[pathological: {name}]")
        print(repair_lightrag_extraction_output(sample, prompt=prompt))

    # Regression: if delimiter extraction accidentally treats tuple delimiter as
    # record delimiter, LightRAG sees several records as one overlong ENTITY
    # (`found 75/4 fields`). The repaired output must contain ## between records.
    already_records = (
        '("entity"<|>"взрывные работы"<|>"Process"<|>"описание")'
        '("entity"<|>"сейсмограмма"<|>"Concept"<|>"описание")'
        '<|COMPLETE|>'
    )
    print("\n[pathological: delimiter_regression_hardened_prompt]")
    print(repair_lightrag_extraction_output(already_records, prompt=hardened_prompt, system_prompt=hardened_system))

    domain_prompt = (
        "-Real Data-\n"
        "Text:\n"
        "При проведении взрывных работ зарегистрированы сейсмограмма, велосиграмма и акселерограмма. "
        "Данные используются для прямого динамического расчета. PPVX = 0,68 см/с, PPVY = 1,36 см/с, PPVZ = 1,31 см/с. "
        "Применялась высокоскоростная камера Phantom MIRO C320.\n"
        "Output:\n"
    )
    print("\n[domain fallback from source text]")
    print(repair_lightrag_extraction_output("Извините, не удалось извлечь знания.", prompt=domain_prompt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
