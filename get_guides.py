import re
import csv
from pathlib import Path

import pandas as pd


# Ключевые слова для отнесения к справочникам / reference works [web:35][web:51]
RU_KEYWORDS = [
    "справочник",
    "словарь",
    "энциклопедия",
    #"руководство",
    "глоссарий",
]

EN_KEYWORDS = [
    "handbook",
    #"manual",
    #"reference book",
    #"reference work",
    "dictionary",
    "encyclopedia",
    #"encyclopaedia",
    "guide",
    "glossary",
]


def detect_language(text: str) -> str:
    """
    Очень грубая эвристика: считаем кириллицу и латиницу.
    """
    if not isinstance(text, str):
        return "unknown"

    t = text.lower()
    ru = len(re.findall(r"[а-яё]", t))
    en = len(re.findall(r"[a-z]", t))

    if ru > en * 1.2 and ru > 5:
        return "ru"
    if en > ru * 1.2 and en > 5:
        return "en"
    return "unknown"


def is_guide(text: str) -> bool:
    if not isinstance(text, str):
        return False

    t = text.lower()
    if any(k in t for k in RU_KEYWORDS):
        return True
    if any(k in t for k in EN_KEYWORDS):
        return True
    return False


def classify_row(row) -> str:
    """
    Возвращает class_label:
    - 'ru_guide'
    - 'en_guide'
    - 'other_guide' (справочник, но язык нераспознан)
    - 'not_guide'
    """
    txt = row.get("citation_text", "")
    if not is_guide(txt):
        return "not_guide"

    lang = detect_language(txt)
    if lang == "ru":
        return "ru_guide"
    if lang == "en":
        return "en_guide"
    return "other_guide"


def main(input_csv="sources.csv"):
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    df = pd.read_csv(input_path, encoding="utf-8")

    if "citation_text" not in df.columns:
        raise ValueError("Expected column 'citation_text' in CSV")

    df["language"] = df["citation_text"].apply(detect_language)
    df["class_label"] = df.apply(classify_row, axis=1)

    # Все справочники
    guides = df[df["class_label"] != "not_guide"].copy()

    ru_guides = guides[guides["class_label"] == "ru_guide"]
    en_guides = guides[guides["class_label"] == "en_guide"]
    other_guides = guides[guides["class_label"] == "other_guide"]

    if not ru_guides.empty:
        ru_guides.to_csv("ru_guides.csv", index=False, encoding="utf-8-sig")
        print(f"Saved {len(ru_guides)} ru_guides to ru_guides.csv")
    else:
        print("No ru_guides found")

    if not en_guides.empty:
        en_guides.to_csv("en_guides.csv", index=False, encoding="utf-8-sig")
        print(f"Saved {len(en_guides)} en_guides to en_guides.csv")
    else:
        print("No en_guides found")

    if not other_guides.empty:
        other_guides.to_csv("other_guides.csv", index=False, encoding="utf-8-sig")
        print(f"Saved {len(other_guides)} other_guides to other_guides.csv")
    else:
        print("No other_guides found")


if __name__ == "__main__":
    main("all_citations.csv")