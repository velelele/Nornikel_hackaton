from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from backend.knowledge_store import KnowledgeStore, jsonl_read, jsonl_write, utc_now_iso
from backend.domain.chunker import clean_extracted_text

TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё_.+-]{2,}", re.UNICODE)
NUMERIC_UNIT_RE = re.compile(
    r"(?<![\w.,+-])[+-]?(?:(?:\d{1,3}(?:[\s\u00a0]\d{3})+)|\d+)(?:[,.]\d+)?\s*"
    r"(?:%|°\s*C|degC|мг/л|мг/дм3|мг/дм³|г/л|кг/т|см/с|мм/с|м/с|A/m2|А/м2|ppm|ppb|кПа|МПа)(?!\w)",
    re.IGNORECASE | re.UNICODE,
)
PROCESS_HINT_RE = re.compile(
    r"\b(?:electrowinning|electrorefining|leaching|smelting|flotation|roasting|solvent extraction|precipitation|neutralization|refining)\b|"
    r"(?:выщелач|электроэкстрак|электрорафинир|плавк|флотац|обжиг|экстракц|осажд|нейтрализ|рафинир|газоочист)",
    re.IGNORECASE | re.UNICODE,
)
PROPERTY_HINT_RE = re.compile(
    r"\b(?:temperature|concentration|flow|current density|pressure|recovery|distribution|yield|ph)\b|"
    r"(?:температур|концентрац|скорост|расход|плотность\s+тока|давлен|извлечен|распределен|выход|pH)",
    re.IGNORECASE | re.UNICODE,
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("ё", "е")).strip()


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower().replace("ё", "е") for m in TOKEN_RE.finditer(text or "")]


def _node_label(node_id: str, entity_labels: dict[str, str]) -> str:
    return entity_labels.get(node_id, node_id.replace("onto:", "").replace("fact:", ""))


def _ontology_similarity(a: str, b: str) -> float:
    an = _norm(a).replace("onto:", "")
    bn = _norm(b).replace("onto:", "")
    if not an or not bn or an == bn:
        return 0.0
    at = set(_tokenize(an.replace("_", " ")))
    bt = set(_tokenize(bn.replace("_", " ")))
    if not at or not bt:
        return 0.0
    jaccard = len(at & bt) / max(1, len(at | bt))
    prefix = 0.0
    for x in at:
        for y in bt:
            if len(x) >= 5 and len(y) >= 5 and (x.startswith(y[:5]) or y.startswith(x[:5])):
                prefix = max(prefix, 0.35)
    return max(jaccard, prefix)


def _safe_centrality(nodes: set[str], weighted_edges: Counter[tuple[str, str, str]]) -> dict[str, float]:
    """Return a cheap degree/PageRank-like centrality without forcing NetworkX."""
    if not nodes:
        return {}
    try:
        import networkx as nx  # type: ignore

        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        for (s, _p, o), weight in weighted_edges.items():
            if s and o and s != o:
                graph.add_edge(s, o, weight=float(weight))
        if graph.number_of_edges() == 0:
            return {node: 0.0 for node in nodes}
        # pagerank on the compact retrieval graph only; bounded iterations to avoid long post-processing.
        pr = nx.pagerank(graph, alpha=0.85, max_iter=30, tol=1e-4, weight="weight")
        return {str(k): round(float(v), 8) for k, v in pr.items()}
    except Exception:
        degree: Counter[str] = Counter()
        for (s, _p, o), weight in weighted_edges.items():
            degree[s] += weight
            degree[o] += weight
        total = float(sum(degree.values()) or 1.0)
        return {node: round(float(degree.get(node, 0)) / total, 8) for node in nodes}


def _source_metadata_by_id(sources: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for src in sources:
        source_id = str(src.get("source_id") or src.get("source_hash") or src.get("source_name") or "")
        if source_id:
            result[source_id] = src
    return result


def build_retrieval_kg(
    store: KnowledgeStore,
    *,
    max_entity_pairs_per_chunk: int = 24,
    max_similarity_edges: int = 2000,
    top_n: int = 80,
) -> dict[str, Any]:
    """Build a safe NetworkX-style retrieval KG over durable knowledge_store.

    This is the default non-LightRAG Stage 2 layer. It does not call an LLM,
    does not write LightRAG runtime files and is bounded by per-chunk caps. The
    output is intended for graph-aware retrieval rather than for final ontology
    validation.
    """
    sources = jsonl_read(store.sources_path)
    chunks = jsonl_read(store.chunks_path)
    entities = jsonl_read(store.entities_path)
    triples = jsonl_read(store.triples_path)
    facts = jsonl_read(store.numeric_facts_path)

    entity_labels = {
        str(row.get("entity_id")): str(row.get("label") or row.get("term_id") or row.get("entity_id"))
        for row in entities
        if row.get("entity_id")
    }
    source_by_id = _source_metadata_by_id(sources)

    facts_by_chunk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fact_entity_edges: Counter[tuple[str, str, str]] = Counter()
    for fact in facts:
        chunk_id = str(fact.get("chunk_id") or "")
        if chunk_id:
            facts_by_chunk[chunk_id].append(fact)
        fact_id = str(fact.get("fact_id") or "")
        for ent in fact.get("entity_ids") or []:
            ent_id = str(ent)
            if ent_id and not ent_id.startswith("onto:"):
                ent_id = f"onto:{ent_id}"
            if fact_id and ent_id:
                fact_entity_edges[(fact_id, "ABOUT_ENTITY", ent_id)] += 1

    nodes: set[str] = set()
    edge_counter: Counter[tuple[str, str, str]] = Counter()
    chunk_entities: dict[str, set[str]] = defaultdict(set)
    chunk_triple_count: Counter[str] = Counter()

    for triple in triples:
        s = str(triple.get("subject_id") or "")
        p = str(triple.get("predicate") or "")
        o = str(triple.get("object_id") or "")
        if not s or not p or not o:
            continue
        nodes.add(s)
        nodes.add(o)
        edge_counter[(s, p, o)] += 1
        chunk_id = str(triple.get("chunk_id") or "")
        if chunk_id:
            chunk_triple_count[chunk_id] += 1
        if chunk_id:
            if s.startswith("onto:"):
                chunk_entities[chunk_id].add(s)
            if o.startswith("onto:"):
                chunk_entities[chunk_id].add(o)

    # Explicitly add fact-entity edges from numeric facts, even if triples.jsonl is partial.
    for edge, weight in fact_entity_edges.items():
        edge_counter[edge] += weight
        nodes.add(edge[0])
        nodes.add(edge[2])

    cooccur_edges: Counter[tuple[str, str, str]] = Counter()
    for chunk_id, entity_set in chunk_entities.items():
        ordered = sorted(entity_set)[: max(2, int(max_entity_pairs_per_chunk))]
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                cooccur_edges[(ordered[i], "CO_OCCURS_IN_CHUNK", ordered[j])] += 1
    edge_counter.update(cooccur_edges)

    # Ontology-similarity edges are bounded and lexical. They help queries like
    # "ALTA" / "конференции" / process synonyms without embedding rebuilds.
    onto_ids = sorted([node for node in nodes if node.startswith("onto:")])
    sim_candidates: list[tuple[float, str, str]] = []
    for i, a in enumerate(onto_ids[:3000]):
        label_a = _node_label(a, entity_labels)
        for b in onto_ids[i + 1 : min(len(onto_ids), i + 401)]:
            sim = _ontology_similarity(label_a, _node_label(b, entity_labels))
            if sim >= 0.35:
                sim_candidates.append((sim, a, b))
    sim_candidates.sort(reverse=True)
    for sim, a, b in sim_candidates[: max(0, int(max_similarity_edges))]:
        edge_counter[(a, "SIMILAR_ONTOLOGY", b)] += max(1, int(round(sim * 10)))

    centrality = _safe_centrality(nodes, edge_counter)

    retrieval_rows: list[dict[str, Any]] = []
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id") or "")
        text = clean_extracted_text(str(chunk.get("text") or chunk.get("lightrag_text") or ""))
        domain_tags = list(chunk.get("domain_tags") or [])
        if not domain_tags and isinstance((chunk.get("metadata") or {}).get("domain_tags"), str):
            domain_tags = [x.strip() for x in str((chunk.get("metadata") or {}).get("domain_tags")).split(",") if x.strip()]
        ent_ids = sorted(chunk_entities.get(chunk_id) or {f"onto:{tag}" for tag in domain_tags if tag})
        fact_rows = facts_by_chunk.get(chunk_id, [])
        properties = sorted({str(f.get("property_id") or f.get("property") or "") for f in fact_rows if f.get("property_id") or f.get("property")})
        units = sorted({str(f.get("unit") or "") for f in fact_rows if f.get("unit")})
        source_id = str(chunk.get("source_id") or "")
        src = source_by_id.get(source_id, {})
        numeric_score = min(2.0, len(fact_rows) * 0.35) + (0.5 if NUMERIC_UNIT_RE.search(text) else 0.0)
        ontology_score = min(2.0, len(ent_ids) * 0.12)
        centrality_score = sum(float(centrality.get(e, 0.0)) for e in ent_ids[:16]) * 10.0
        process_score = 0.45 if PROCESS_HINT_RE.search(text) else 0.0
        property_score = 0.35 if PROPERTY_HINT_RE.search(text) else 0.0
        graph_boost = round(numeric_score + ontology_score + centrality_score + process_score + property_score, 4)
        retrieval_rows.append(
            {
                "chunk_id": chunk_id,
                "source_id": source_id,
                "source_name": chunk.get("source_name"),
                "citation_name": chunk.get("citation_name"),
                "theme_id": chunk.get("theme_id") or src.get("theme_id"),
                "theme_name": chunk.get("theme_name") or src.get("theme_name"),
                "collection": chunk.get("collection") or src.get("collection"),
                "year": chunk.get("year") or src.get("year"),
                "source_type": chunk.get("source_type") or src.get("source_type"),
                "object_type": chunk.get("object_type"),
                "domain_tags": domain_tags,
                "entity_ids": ent_ids,
                "numeric_fact_count": len(fact_rows),
                "numeric_properties": properties,
                "numeric_units": units,
                "triple_count": int(chunk_triple_count.get(chunk_id, 0)),
                "graph_boost": graph_boost,
                "chars": len(text),
                "text": text,
            }
        )

    retrieval_rows.sort(key=lambda row: (float(row.get("graph_boost") or 0.0), int(row.get("chars") or 0)), reverse=True)
    jsonl_write(store.root / "retrieval_index.jsonl", retrieval_rows)

    top_nodes = sorted(
        (
            {"node_id": node, "label": _node_label(node, entity_labels), "centrality": centrality.get(node, 0.0)}
            for node in nodes
        ),
        key=lambda row: float(row["centrality"]),
        reverse=True,
    )[:top_n]
    top_edges = [
        {"subject_id": s, "predicate": p, "object_id": o, "weight": weight}
        for (s, p, o), weight in edge_counter.most_common(top_n)
    ]
    payload = {
        "graph_type": "NiCoRetrievalKG",
        "updated_at": utc_now_iso(),
        "store": str(store.root),
        "index_path": str(store.root / "retrieval_index.jsonl"),
        "counts": {
            "sources": len(sources),
            "chunks": len(chunks),
            "entities": len(entities),
            "facts": len(facts),
            "nodes": len(nodes),
            "edges": len(edge_counter),
            "retrieval_rows": len(retrieval_rows),
        },
        "top_nodes": top_nodes,
        "top_edges": top_edges,
        "retrieval_features": [
            "lexical BM25-lite",
            "metadata-aware boosts",
            "numeric_facts boost",
            "entity/triples boost",
            "ontology co-occurrence graph",
            "optional NetworkX PageRank if networkx is installed",
        ],
    }
    (store.root / "retrieval_kg.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
