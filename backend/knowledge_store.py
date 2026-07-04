from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from backend.domain.document_processor import ProcessedDocument
from backend.domain.numeric_extractor import NumericFact

SCHEMA_VERSION = "0.2"
DEFAULT_ONTOLOGY_VERSION = "nornickel-metallurgy-0.1"
DEFAULT_APP_VERSION = "nico-0.4.0"
DEFAULT_CHUNKER_VERSION = "metadata-aware-chunker-0.2"
DEFAULT_NUMERIC_EXTRACTOR_VERSION = "numeric-extractor-0.2"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def jsonl_read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def jsonl_write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    payload = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows_list)
    path.write_text((payload + "\n") if payload else "", encoding="utf-8")


def jsonl_append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _replace_rows(path: Path, predicate: Callable[[dict[str, Any]], bool], new_rows: Iterable[dict[str, Any]]) -> None:
    kept = [row for row in jsonl_read(path) if not predicate(row)]
    kept.extend(new_rows)
    jsonl_write(path, kept)


def _metadata_value(metadata: dict[str, Any], key: str, default: Any = None) -> Any:
    value = metadata.get(key, default)
    if value is None:
        return default
    return value


class KnowledgeStore:
    """Versioned, LightRAG-independent persistent knowledge store.

    This is the source-of-truth layer. LightRAG's vector/KG files remain a
    rebuildable runtime index. The store keeps parsed chunks, extracted facts,
    lightweight entities/relations and provenance in JSONL files that survive
    application upgrades and embedding-model changes.
    """

    def __init__(
        self,
        root: Path,
        *,
        schema_version: str = SCHEMA_VERSION,
        ontology_version: str = DEFAULT_ONTOLOGY_VERSION,
        app_version: str = DEFAULT_APP_VERSION,
        chunker_version: str = DEFAULT_CHUNKER_VERSION,
        numeric_extractor_version: str = DEFAULT_NUMERIC_EXTRACTOR_VERSION,
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.schema_version = schema_version
        self.ontology_version = ontology_version
        self.app_version = app_version
        self.chunker_version = chunker_version
        self.numeric_extractor_version = numeric_extractor_version

        self.manifest_path = self.root / "manifest.json"
        self.sources_path = self.root / "sources.jsonl"
        self.fragments_path = self.root / "source_fragments.jsonl"
        self.chunks_path = self.root / "chunks.jsonl"
        self.numeric_facts_path = self.root / "numeric_facts.jsonl"
        self.entities_path = self.root / "entities.jsonl"
        self.relations_path = self.root / "relations.jsonl"
        self.triples_path = self.root / "triples.jsonl"
        self.claims_path = self.root / "claims.jsonl"
        self.ingestion_runs_path = self.root / "ingestion_runs.jsonl"
        self._ensure_manifest()

    def _base_meta(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ontology_version": self.ontology_version,
            "app_version": self.app_version,
        }

    def _ensure_manifest(self) -> None:
        manifest = {
            "store_type": "NiCoKnowledgeStore",
            "schema_version": self.schema_version,
            "ontology_version": self.ontology_version,
            "app_version": self.app_version,
            "chunker_version": self.chunker_version,
            "numeric_extractor_version": self.numeric_extractor_version,
            "created_or_updated_at": utc_now_iso(),
            "files": {
                "sources": self.sources_path.name,
                "source_fragments": self.fragments_path.name,
                "chunks": self.chunks_path.name,
                "numeric_facts": self.numeric_facts_path.name,
                "entities": self.entities_path.name,
                "relations": self.relations_path.name,
                "triples": self.triples_path.name,
                "claims": self.claims_path.name,
                "ingestion_runs": self.ingestion_runs_path.name,
            },
        }
        if self.manifest_path.exists():
            try:
                previous = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if isinstance(previous, dict):
                    manifest["created_at"] = previous.get("created_at") or previous.get("created_or_updated_at")
            except Exception:
                pass
        manifest.setdefault("created_at", utc_now_iso())
        self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _source_id(source_hash: str) -> str:
        return f"sha256:{source_hash}"

    @staticmethod
    def _chunk_id(source_hash: str, object_id: str, text: str) -> str:
        return "chunk:" + hashlib.sha1(f"{source_hash}|{object_id}|{sha256_text(text)}".encode("utf-8")).hexdigest()

    @staticmethod
    def _fragment_id(source_hash: str, object_id: str) -> str:
        return "fragment:" + hashlib.sha1(f"{source_hash}|{object_id}".encode("utf-8")).hexdigest()

    def upsert_processed_document(
        self,
        processed: ProcessedDocument,
        numeric_facts: list[NumericFact] | list[dict[str, Any]],
        *,
        original_path: str | None = None,
        run_id: str | None = None,
        theme: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = utc_now_iso()
        run_id = run_id or "run:" + hashlib.sha1(f"{processed.source_hash}|{now}".encode("utf-8")).hexdigest()[:16]
        source_id = self._source_id(processed.source_hash)
        source_name = processed.source_name

        theme_meta = dict(theme or {})

        source_row = {
            **self._base_meta(),
            **theme_meta,
            "source_id": source_id,
            "source_hash": processed.source_hash,
            "source_name": source_name,
            "original_path": original_path or source_name,
            "total_chars": processed.total_chars,
            "objects_count": len(processed.objects),
            "updated_at": now,
        }

        fragment_rows: list[dict[str, Any]] = []
        chunk_rows: list[dict[str, Any]] = []
        entity_rows_by_id: dict[str, dict[str, Any]] = {}
        relation_rows: list[dict[str, Any]] = []
        triple_rows: list[dict[str, Any]] = []

        object_to_ids: dict[str, tuple[str, str]] = {}
        for obj in processed.objects:
            text_hash = sha256_text(obj.text)
            fragment_id = self._fragment_id(processed.source_hash, obj.object_id)
            chunk_id = self._chunk_id(processed.source_hash, obj.object_id, obj.text)
            object_to_ids[obj.object_id] = (fragment_id, chunk_id)
            metadata = dict(obj.metadata or {})
            domain_tags = [tag.strip() for tag in str(metadata.get("domain_tags") or "").split(",") if tag.strip()]
            citation_name = obj.citation_name

            common = {
                **self._base_meta(),
                **theme_meta,
                "source_id": source_id,
                "source_hash": processed.source_hash,
                "source_name": source_name,
                "object_id": obj.object_id,
                "object_type": obj.object_type,
                "citation_name": citation_name,
                "text_hash": text_hash,
                "chars": len(obj.text),
                "metadata": metadata,
                "domain_tags": domain_tags,
                "updated_at": now,
            }
            fragment_rows.append({**common, "fragment_id": fragment_id})
            chunk_rows.append(
                {
                    **common,
                    "chunk_id": chunk_id,
                    "fragment_id": fragment_id,
                    "chunker_version": self.chunker_version,
                    "text": obj.text,
                    "lightrag_text": obj.to_lightrag_text(),
                }
            )
            triple_rows.append(
                {
                    **self._base_meta(),
                    **theme_meta,
                    "triple_id": f"triple:{source_id}:{chunk_id}:contains",
                    "subject_id": source_id,
                    "predicate": "CONTAINS",
                    "object_id": chunk_id,
                    "source_id": source_id,
                    "chunk_id": chunk_id,
                    "confidence": 1.0,
                    "updated_at": now,
                }
            )
            for tag in domain_tags:
                entity_id = f"onto:{tag}"
                entity_rows_by_id[entity_id] = {
                    **self._base_meta(),
                    **theme_meta,
                    "entity_id": entity_id,
                    "term_id": tag,
                    "label": tag,
                    "class_name": "OntologyTerm",
                    "source_ids": [source_id],
                    "updated_at": now,
                }
                triple_rows.append(
                    {
                        **self._base_meta(),
                        **theme_meta,
                        "triple_id": f"triple:{chunk_id}:{entity_id}:mentions",
                        "subject_id": chunk_id,
                        "predicate": "MENTIONS",
                        "object_id": entity_id,
                        "source_id": source_id,
                        "chunk_id": chunk_id,
                        "confidence": 0.75,
                        "updated_at": now,
                    }
                )

        fact_rows: list[dict[str, Any]] = []
        for item in numeric_facts:
            fact_dict = item.to_dict() if hasattr(item, "to_dict") else dict(item)
            object_id = str(fact_dict.get("object_id") or "")
            fragment_id, chunk_id = object_to_ids.get(object_id, ("", ""))
            fact_id = str(fact_dict.get("fact_id") or "")
            if not fact_id:
                fact_id = "fact:" + hashlib.sha1(json.dumps(fact_dict, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:20]
            else:
                fact_id = f"fact:{fact_id}" if not fact_id.startswith("fact:") else fact_id

            metadata = dict(fact_dict.get("metadata") or {})
            metadata.update({"source_hash": processed.source_hash})
            row = {
                **self._base_meta(),
                **theme_meta,
                **fact_dict,
                "fact_id": fact_id,
                "source_id": source_id,
                "source_hash": processed.source_hash,
                "source_name": source_name,
                "fragment_id": fragment_id,
                "chunk_id": chunk_id,
                "extractor_version": self.numeric_extractor_version,
                "metadata": metadata,
                "updated_at": now,
            }
            fact_rows.append(row)
            if chunk_id:
                triple_rows.append(
                    {
                        **self._base_meta(),
                        **theme_meta,
                        "triple_id": f"triple:{chunk_id}:{fact_id}:has_numeric_fact",
                        "subject_id": chunk_id,
                        "predicate": "HAS_NUMERIC_FACT",
                        "object_id": fact_id,
                        "source_id": source_id,
                        "chunk_id": chunk_id,
                        "confidence": float(row.get("confidence") or 0.0),
                        "updated_at": now,
                    }
                )
            for entity_id in row.get("entity_ids") or []:
                ent_id = f"onto:{entity_id}"
                entity_rows_by_id.setdefault(
                    ent_id,
                    {
                        **self._base_meta(),
                        **theme_meta,
                        "entity_id": ent_id,
                        "term_id": entity_id,
                        "label": entity_id,
                        "class_name": "OntologyTerm",
                        "source_ids": [source_id],
                        "updated_at": now,
                    },
                )
                triple_rows.append(
                    {
                        **self._base_meta(),
                        **theme_meta,
                        "triple_id": f"triple:{fact_id}:{ent_id}:about",
                        "subject_id": fact_id,
                        "predicate": "ABOUT_ENTITY",
                        "object_id": ent_id,
                        "source_id": source_id,
                        "chunk_id": chunk_id,
                        "confidence": float(row.get("confidence") or 0.0),
                        "updated_at": now,
                    }
                )

        def same_source(row: dict[str, Any]) -> bool:
            return row.get("source_id") == source_id or row.get("source_hash") == processed.source_hash or row.get("source_name") == source_name

        _replace_rows(self.sources_path, same_source, [source_row])
        _replace_rows(self.fragments_path, same_source, fragment_rows)
        _replace_rows(self.chunks_path, same_source, chunk_rows)
        _replace_rows(self.numeric_facts_path, same_source, fact_rows)
        _replace_rows(self.relations_path, same_source, relation_rows)
        _replace_rows(self.triples_path, same_source, triple_rows)

        # Entities can be shared across sources. Merge by entity_id and append source_id.
        existing_entities = {row.get("entity_id"): row for row in jsonl_read(self.entities_path) if row.get("entity_id")}
        for entity_id, row in entity_rows_by_id.items():
            existing = existing_entities.get(entity_id)
            if existing:
                source_ids = list(dict.fromkeys([*(existing.get("source_ids") or []), source_id]))
                existing.update(row)
                existing["source_ids"] = source_ids
                existing_entities[entity_id] = existing
            else:
                existing_entities[entity_id] = row
        jsonl_write(self.entities_path, existing_entities.values())

        jsonl_append(
            self.ingestion_runs_path,
            {
                **self._base_meta(),
                **theme_meta,
                "run_id": run_id,
                "source_id": source_id,
                "source_name": source_name,
                "source_hash": processed.source_hash,
                "objects_count": len(processed.objects),
                "numeric_facts_count": len(fact_rows),
                "chunks_count": len(chunk_rows),
                "status": "stored",
                "created_at": now,
            },
        )
        self._ensure_manifest()
        return {
            "source_id": source_id,
            "source_name": source_name,
            "theme": theme_meta,
            "chunks_count": len(chunk_rows),
            "numeric_facts_count": len(fact_rows),
            "entities_count": len(entity_rows_by_id),
        }

    def import_legacy_numeric_facts(self, legacy_path: Path, *, overwrite: bool = False) -> dict[str, Any]:
        """Import old data/domain_facts.jsonl into numeric_facts.jsonl.

        Legacy rows do not contain full chunks, so they are usable for structured
        query context but not for rebuilding the vector index.
        """
        if not legacy_path.exists():
            return {"imported": 0, "skipped": True, "reason": "legacy file not found"}
        if self.numeric_facts_path.exists() and not overwrite and jsonl_read(self.numeric_facts_path):
            return {"imported": 0, "skipped": True, "reason": "knowledge_store already has numeric facts"}

        rows = jsonl_read(legacy_path)
        now = utc_now_iso()
        migrated: list[dict[str, Any]] = []
        for row in rows:
            metadata = dict(row.get("metadata") or {})
            source_hash = str(metadata.get("source_hash") or row.get("source_hash") or "legacy")
            source_id = self._source_id(source_hash) if source_hash != "legacy" else "legacy:unknown_source"
            fact_id = str(row.get("fact_id") or "")
            if not fact_id:
                fact_id = "fact:legacy:" + hashlib.sha1(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:20]
            elif not fact_id.startswith("fact:"):
                fact_id = "fact:" + fact_id
            migrated.append({
                **self._base_meta(),
                **row,
                "fact_id": fact_id,
                "source_id": source_id,
                "source_hash": source_hash,
                "extractor_version": str(row.get("extractor_version") or "legacy"),
                "legacy_import": True,
                "metadata": metadata,
                "updated_at": now,
            })
        jsonl_write(self.numeric_facts_path, migrated)
        jsonl_append(self.ingestion_runs_path, {
            **self._base_meta(),
            "run_id": "legacy-import:" + hashlib.sha1(str(legacy_path).encode("utf-8")).hexdigest()[:16],
            "source_name": str(legacy_path),
            "numeric_facts_count": len(migrated),
            "status": "legacy_imported",
            "created_at": now,
        })
        self._ensure_manifest()
        return {"imported": len(migrated), "skipped": False, "source": str(legacy_path)}

    def iter_lightrag_items(self, *, source_id: str | None = None) -> Iterable[tuple[str, str]]:
        for idx, row in enumerate(jsonl_read(self.chunks_path), start=1):
            if source_id and row.get("source_id") != source_id:
                continue
            text = str(row.get("lightrag_text") or row.get("text") or "").strip()
            citation_base = str(row.get("citation_name") or row.get("source_name") or row.get("chunk_id") or "").strip()
            chunk_id = str(row.get("chunk_id") or "").replace(":", "_")
            suffix = chunk_id[-16:] if chunk_id else f"row_{idx:06d}"
            # LightRAG treats file_path as a document identity in several storages.
            # Multiple chunks from one page/section can have the same citation_name;
            # keep the human-readable citation, but make the runtime file_path unique.
            citation = f"{citation_base}#kgchunk:{suffix}" if citation_base else f"kgchunk:{suffix}"
            if text and citation:
                yield text, citation

    def stats(self) -> dict[str, Any]:
        sources = jsonl_read(self.sources_path)
        chunks = jsonl_read(self.chunks_path)
        facts = jsonl_read(self.numeric_facts_path)
        entities = jsonl_read(self.entities_path)
        triples = jsonl_read(self.triples_path)
        by_source: dict[str, int] = {}
        by_property: dict[str, int] = {}
        for fact in facts:
            by_source[fact.get("source_name") or "unknown"] = by_source.get(fact.get("source_name") or "unknown", 0) + 1
            by_property[fact.get("property_id") or "unknown"] = by_property.get(fact.get("property_id") or "unknown", 0) + 1
        return {
            "store_dir": str(self.root),
            "schema_version": self.schema_version,
            "ontology_version": self.ontology_version,
            "app_version": self.app_version,
            "sources": len(sources),
            "chunks": len(chunks),
            "numeric_facts": len(facts),
            "entities": len(entities),
            "triples": len(triples),
            "by_property": by_property,
            "by_source": by_source,
        }

    def validate(self) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        sources = jsonl_read(self.sources_path)
        chunks = jsonl_read(self.chunks_path)
        facts = jsonl_read(self.numeric_facts_path)
        source_ids = {row.get("source_id") for row in sources}
        chunk_ids = {row.get("chunk_id") for row in chunks}
        if len(chunk_ids) != len(chunks):
            warnings.append("Есть дубли chunk_id в chunks.jsonl")
        for row in chunks:
            if not row.get("source_id"):
                errors.append(f"chunk без source_id: {row.get('chunk_id')}")
            elif row.get("source_id") not in source_ids:
                errors.append(f"chunk с неизвестным source_id: {row.get('chunk_id')}")
            if not row.get("text"):
                errors.append(f"chunk без text: {row.get('chunk_id')}")
        for row in facts:
            if not row.get("source_id"):
                errors.append(f"fact без source_id: {row.get('fact_id')}")
            if not row.get("chunk_id"):
                warnings.append(f"fact без chunk_id: {row.get('fact_id')}")
            elif row.get("chunk_id") not in chunk_ids:
                warnings.append(f"fact с неизвестным chunk_id: {row.get('fact_id')}")
            if row.get("value") is not None and not row.get("unit"):
                errors.append(f"numeric fact без unit: {row.get('fact_id')}")
        return {"ok": not errors, "errors": errors[:200], "warnings": warnings[:200], "stats": self.stats()}

    def export_to(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        for path in (
            self.manifest_path,
            self.sources_path,
            self.fragments_path,
            self.chunks_path,
            self.numeric_facts_path,
            self.entities_path,
            self.relations_path,
            self.triples_path,
            self.claims_path,
            self.ingestion_runs_path,
        ):
            if path.exists():
                shutil.copy2(path, out_dir / path.name)
        export_manifest = {
            "exported_at": utc_now_iso(),
            "source_store": str(self.root),
            "stats": self.stats(),
        }
        (out_dir / "export_manifest.json").write_text(
            json.dumps(export_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return out_dir
