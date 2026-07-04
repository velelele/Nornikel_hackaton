from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_ONTOLOGY_PATH = ROOT / "ontology.yaml"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()


@dataclass(frozen=True)
class DomainTerm:
    term_id: str
    canonical_ru: str
    canonical_en: str
    class_name: str
    labels_ru: tuple[str, ...]
    labels_en: tuple[str, ...]
    neighbors: tuple[str, ...]

    @property
    def all_labels(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.canonical_ru, self.canonical_en, *self.labels_ru, *self.labels_en)))


@dataclass(frozen=True)
class TermMatch:
    term_id: str
    label: str
    canonical_ru: str
    canonical_en: str
    class_name: str
    start: int
    end: int


class Ontology:
    def __init__(self, path: Path = DEFAULT_ONTOLOGY_PATH) -> None:
        self.path = path
        self.raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.terms: dict[str, DomainTerm] = self._load_terms()
        self.properties: dict[str, dict[str, Any]] = dict(self.raw.get("properties") or {})
        self.geography: dict[str, dict[str, Any]] = dict(self.raw.get("geography") or {})
        self._label_index = self._build_label_index()

    def _load_terms(self) -> dict[str, DomainTerm]:
        out: dict[str, DomainTerm] = {}
        for term_id, payload in (self.raw.get("terms") or {}).items():
            labels = payload.get("labels") or {}
            out[term_id] = DomainTerm(
                term_id=term_id,
                canonical_ru=str(payload.get("canonical_ru") or term_id),
                canonical_en=str(payload.get("canonical_en") or term_id),
                class_name=str(payload.get("class") or "Unknown"),
                labels_ru=tuple(str(x) for x in (labels.get("ru") or [])),
                labels_en=tuple(str(x) for x in (labels.get("en") or [])),
                neighbors=tuple(str(x) for x in (payload.get("neighbors") or [])),
            )
        return out

    def _build_label_index(self) -> list[tuple[str, str, str]]:
        rows: list[tuple[str, str, str]] = []
        for term in self.terms.values():
            for label in term.all_labels:
                clean = _norm(label)
                if clean:
                    rows.append((clean, label, term.term_id))
        rows.sort(key=lambda item: len(item[0]), reverse=True)
        return rows

    def find_terms(self, text: str) -> list[TermMatch]:
        normalized = _norm(text)
        matches: list[TermMatch] = []
        occupied: list[tuple[int, int]] = []

        for normalized_label, original_label, term_id in self._label_index:
            if not normalized_label:
                continue
            pattern = re.compile(rf"(?<![\wА-Яа-яЁё]){re.escape(normalized_label)}(?![\wА-Яа-яЁё])", re.IGNORECASE)
            for m in pattern.finditer(normalized):
                start, end = m.span()
                if any(not (end <= s or start >= e) for s, e in occupied):
                    continue
                term = self.terms[term_id]
                matches.append(
                    TermMatch(
                        term_id=term_id,
                        label=original_label,
                        canonical_ru=term.canonical_ru,
                        canonical_en=term.canonical_en,
                        class_name=term.class_name,
                        start=start,
                        end=end,
                    )
                )
                occupied.append((start, end))
        return sorted(matches, key=lambda item: (item.start, item.end))

    def expand_term_ids(self, term_ids: list[str], *, max_terms: int = 48) -> list[str]:
        expansions: list[str] = []
        seen: set[str] = set()

        def add(value: str) -> None:
            clean = value.strip()
            key = clean.lower()
            if clean and key not in seen:
                seen.add(key)
                expansions.append(clean)

        for term_id in term_ids:
            term = self.terms.get(term_id)
            if not term:
                continue
            add(term.canonical_ru)
            add(term.canonical_en)
            for label in term.all_labels:
                add(label)
            for neighbor_id in term.neighbors:
                neighbor = self.terms.get(neighbor_id)
                if neighbor:
                    add(neighbor.canonical_ru)
                    add(neighbor.canonical_en)
                    for label in neighbor.all_labels[:4]:
                        add(label)
                else:
                    add(neighbor_id)
            if len(expansions) >= max_terms:
                break

        return expansions[:max_terms]

    def property_labels(self) -> dict[str, list[str]]:
        return {
            prop_id: [str(x) for x in (payload.get("labels") or [])]
            for prop_id, payload in self.properties.items()
        }

    def detect_geography(self, text: str) -> list[str]:
        normalized = _norm(text)
        out: list[str] = []
        for geography_id, payload in self.geography.items():
            for label in payload.get("labels") or []:
                clean = _norm(str(label))
                if clean and re.search(rf"(?<![\wА-Яа-яЁё]){re.escape(clean)}(?![\wА-Яа-яЁё])", normalized):
                    out.append(geography_id)
                    break
        return out


@lru_cache(maxsize=1)
def get_ontology() -> Ontology:
    return Ontology()
