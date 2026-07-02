from __future__ import annotations

from pathlib import Path

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".tsv",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".log",
    ".rtf",
}

OFFICE_EXTENSIONS = {
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
}

OTHER_BINARY_EXTENSIONS = {
    ".pdf",
    ".html",
    ".htm",
    ".epub",
}

SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | OFFICE_EXTENSIONS | OTHER_BINARY_EXTENSIONS

# Docling не поддерживает старые бинарные форматы .doc / .ppt / .xls
DOCLING_EXTENSIONS = {".docx", ".pptx", ".xlsx", ".pdf", ".html", ".htm", ".epub"}


def _decode_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "cp1251"):
        try:
            text = raw.decode(encoding).strip()
            if text:
                return text
        except UnicodeDecodeError:
            continue
    raise ValueError("Не удалось прочитать файл в поддерживаемой кодировке")


def _extract_with_docling(path: Path) -> str:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise ValueError(
            "Формат требует пакет docling. Установите зависимости: pip install -r requirements.txt"
        ) from exc

    result = DocumentConverter().convert(str(path))
    text = (result.document.export_to_markdown() or "").strip()
    if not text:
        text = (result.document.export_to_markdown(strict_text=True) or "").strip()
    if not text:
        raise ValueError("Docling не извлёк текст из документа")
    return text


def _extract_with_markitdown(path: Path) -> str:
    try:
        from markitdown import MarkItDown
    except ImportError as exc:
        raise ValueError(
            f"Формат {path.suffix} требует пакет markitdown. Установите зависимости: pip install -r requirements.txt"
        ) from exc

    result = MarkItDown().convert(str(path))
    text = (getattr(result, "text_content", None) or result.markdown or "").strip()
    if not text:
        raise ValueError("Файл не содержит извлекаемого текста")
    return text


def _extract_via_soffice(path: Path) -> str:
    import shutil
    import subprocess
    import tempfile

    candidates = [
        shutil.which("soffice"),
        shutil.which("libreoffice"),
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    ]
    soffice = next((cmd for cmd in candidates if cmd and Path(cmd).exists()), None)
    if not soffice:
        raise ValueError("LibreOffice не найден")

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)
        proc = subprocess.run(
            [soffice, "--headless", "--convert-to", "txt", "--outdir", str(out_dir), str(path.resolve())],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if proc.returncode != 0:
            raise ValueError(proc.stderr.strip() or "LibreOffice не смог конвертировать файл")

        txt_files = sorted(out_dir.glob("*.txt"))
        if not txt_files:
            raise ValueError("LibreOffice не создал текстовый файл")

        for encoding in ("utf-8", "utf-8-sig", "cp1251"):
            try:
                text = txt_files[0].read_text(encoding=encoding).strip()
                if text:
                    return text
            except UnicodeDecodeError:
                continue

    raise ValueError("LibreOffice вернул пустой текст")


def _extract_doc(file_bytes: bytes, path: Path | None = None) -> str:
    from lightrag.parser.legacy.extractors import _extract_docx

    errors: list[str] = []

    try:
        text = _extract_docx(file_bytes).strip()
        if text:
            return text
    except Exception as exc:
        errors.append(f"docx-parser: {exc}")

    if path is not None and path.exists():
        for extractor, label in (
            (_extract_with_markitdown, "markitdown"),
            (_extract_via_soffice, "libreoffice"),
        ):
            try:
                return extractor(path)
            except Exception as exc:
                errors.append(f"{label}: {exc}")

    import tempfile

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)
        for extractor, label in (
            (_extract_with_markitdown, "markitdown"),
            (_extract_via_soffice, "libreoffice"),
        ):
            try:
                return extractor(tmp_path)
            except Exception as exc:
                errors.append(f"{label}: {exc}")
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    detail = "; ".join(errors) if errors else "неизвестная ошибка"
    raise ValueError(
        f"Не удалось прочитать .doc ({detail}). Сохраните файл как .docx или установите LibreOffice."
    )


def _extract_with_legacy(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    file_bytes = path.read_bytes()

    from lightrag.parser.legacy.extractors import LegacyExtractionError, extract_text as legacy_extract

    try:
        text = legacy_extract(file_bytes, suffix).strip()
    except LegacyExtractionError as exc:
        raise ValueError(str(exc)) from exc

    if not text:
        raise ValueError("Файл не содержит извлекаемого текста")
    return text


def _extract_office_or_binary(path: Path) -> str:
    suffix = path.suffix.lower()
    errors: list[str] = []

    if suffix == ".doc":
        return _extract_doc(path.read_bytes(), path)

    if suffix in {".docx", ".pptx", ".xlsx", ".pdf"}:
        try:
            return _extract_with_legacy(path)
        except Exception as exc:
            errors.append(f"legacy: {exc}")

    if suffix in {".html", ".htm", ".epub", ".ppt", ".xls"}:
        try:
            return _extract_with_markitdown(path)
        except Exception as exc:
            errors.append(f"markitdown: {exc}")

    if suffix in DOCLING_EXTENSIONS:
        try:
            return _extract_with_docling(path)
        except Exception as exc:
            errors.append(f"docling: {exc}")

    detail = "; ".join(errors) if errors else "неизвестная ошибка"
    raise ValueError(f"Не удалось извлечь текст из документа ({detail})")


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Неподдерживаемый формат: {suffix}")

    if suffix in TEXT_EXTENSIONS:
        return _decode_text_file(path)

    return _extract_office_or_binary(path)


def supported_formats_label() -> str:
    return ", ".join(sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS))
