import re
import csv
from pathlib import Path

import pdfplumber


REFERENCE_HEADERS = [
    r"references?",
    r"bibliography",
    r"works cited",
    r"список литературы",
    r"литература",
    r"использованные источники",
    r"источники",
]

HEADER_RE = re.compile(
    r"(?is)\b(?:"
    + "|".join(REFERENCE_HEADERS)
    + r")\b"
)

# Любой «крупный» заголовок следующего раздела (примеры, можно дописать под свои документы)
NEXT_SECTION_RE = re.compile(
    r"(?im)^\s*(?:приложение|application|appendix|chapter|глава|section)\b"
)

# Начало новой записи в списке литературы
START_RE = re.compile(
    r"(?m)^\s*(?:\[\d+\]|\d+\.\s|\d+\)\s|[•\-–]\s)"
)


def extract_pdf_text(pdf_path: str) -> str:
    parts = []
    with pdfplumber.open(pdf_path) as pdf:  # [web:11]
        for page in pdf.pages:
            txt = page.extract_text() or ""
            parts.append(txt)
    text = "\n".join(parts)
    text = text.replace("\xa0", " ")
    return text


def find_reference_blocks(text: str) -> list[str]:
    """
    Возвращает список текстовых блоков – по одному на каждый раздел с источниками.
    """
    blocks = []
    matches = list(HEADER_RE.finditer(text))

    if not matches:
        return blocks

    # Берём все найденные заголовки
    for i, m in enumerate(matches):
        start = m.end()

        # Граница – либо начало следующего заголовка источников,
        # либо начало «большого» раздела (appendix/глава и т.п.), либо конец текста.
        if i + 1 < len(matches):
            next_header_start = matches[i + 1].start()
        else:
            next_header_start = len(text)

        # Попробуем найти «следующий раздел» внутри интервала
        segment = text[start:next_header_start]
        next_section_match = NEXT_SECTION_RE.search(segment)
        if next_section_match:
            end = start + next_section_match.start()
        else:
            end = next_header_start

        block = text[start:end].strip()
        if block:
            blocks.append(block)

    return blocks


def split_citations(ref_text: str) -> list[str]:
    """
    Делит текст одного блока на отдельные записи источников.
    """
    ref_text = ref_text.strip()
    if not ref_text:
        return []

    ref_text = re.sub(r"\r", "\n", ref_text)
    ref_text = re.sub(r"\n{2,}", "\n", ref_text)

    lines = [ln.strip() for ln in ref_text.split("\n") if ln.strip()]
    if not lines:
        return []

    citations = []
    current = ""

    for line in lines:
        is_new = bool(START_RE.match(line))
        if is_new and current:
            citations.append(current.strip())
            current = START_RE.sub("", line).strip()
        else:
            if current:
                current += " " + line
            else:
                current = line

    if current:
        citations.append(current.strip())

    cleaned = []
    for c in citations:
        c = re.sub(r"^\s*(?:\[\d+\]|\d+\.\s|\d+\)\s)", "", c).strip()
        c = re.sub(r"\s+", " ", c)
        if len(c) > 5:
            cleaned.append(c)

    return cleaned


def extract_all_citations(text: str) -> list[str]:
    """
    Находит все блоки источников и собирает источники из каждого блока.
    """
    blocks = find_reference_blocks(text)
    all_citations = []
    for b in blocks:
        all_citations.extend(split_citations(b))

    # Можно убрать дубли (часто в сборниках встречаются одинаковые записи)
    unique = []
    seen = set()
    for c in all_citations:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def save_to_csv(citations: list[str], csv_path: str):
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["citation_id", "citation_text"])
        for i, cit in enumerate(citations, start=1):
            writer.writerow([i, cit])


def extract_citations_from_pdf(pdf_path: str, csv_path: str = "citations.csv"):
    text = extract_pdf_text(pdf_path)
    citations = extract_all_citations(text)
    #save_to_csv(citations, csv_path)
    return citations


def process_directory_to_single_csv(input_dir: str, output_csv: str):
    input_path = Path(input_dir)
    pdf_files = sorted(input_path.rglob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {input_path}")
        return

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["pdf_name", "citation_id", "citation_text"])

        for pdf_file in pdf_files:
            if not pdf_file.is_file():
                continue
            if pdf_file.suffix.lower() != ".pdf":
                continue

            print(f"Processing {pdf_file}")

            try:
                citations = extract_citations_from_pdf(str(pdf_file), csv_path=None)
            except TypeError:
                # если твоя функция обязательно требует csv_path – делаем обёртку
                citations = extract_citations_from_pdf(str(pdf_file), "tmp.csv")
            except Exception as e:
                print(f"  Error processing {pdf_file}: {e}")
                continue

            for i, cit in enumerate(citations, start=1):
                writer.writerow([pdf_file.name, i, cit])

            print(f"  Extracted {len(citations)} citations")

if __name__ == "__main__":
    input_dir = "C:\\Users\\User\\Documents\\Projects\\Nornikel_hakathon\\Задача 2. Научный клубок\\Задача 2. Научный клубок"
    output_csv = "all_citations.csv"

    process_directory_to_single_csv(input_dir, output_csv)
