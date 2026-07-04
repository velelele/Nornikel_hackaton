from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx

from backend.config_manager import AppConfig, resolve_project_path
from backend.knowledge_store import KnowledgeStore, jsonl_read, jsonl_write, utc_now_iso

TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё_\-]{3,}", re.UNICODE)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("ё", "е")).strip()


def _tokens(text: str) -> list[str]:
    return list(dict.fromkeys(TOKEN_RE.findall(_norm(text))))


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_id(value: Any) -> str:
    return str(value or "").strip()


def cosine_similarity(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        fx = float(x)
        fy = float(y)
        dot += fx * fy
        na += fx * fx
        nb += fy * fy
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _theme_embedding_base_url(config: AppConfig) -> str:
    return str(config.theme_embeddings_embedding_base_url or config.embedding_base_url).rstrip("/")


def _theme_embedding_model(config: AppConfig) -> str:
    return str(config.theme_embeddings_embedding_model or config.embedding_model)


def _theme_embedding_api_key(config: AppConfig) -> str:
    return str(config.theme_embeddings_embedding_api_key or config.embedding_api_key or "")


async def embed_texts_openai_compatible(
    texts: list[str],
    *,
    config: AppConfig,
    timeout: float = 180.0,
) -> list[list[float]]:
    """Use Ollama/OpenAI-compatible /v1 embeddings endpoint for theme routing.

    The theme router may use a lighter model than the main LightRAG chunk
    embeddings via intelligence.theme_embeddings.embedding_model. Query vectors
    and stored theme-profile vectors must use the same model.
    """
    if not texts:
        return []
    url = _theme_embedding_base_url(config) + "/embeddings"
    model = _theme_embedding_model(config)
    api_key = _theme_embedding_api_key(config)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json={"model": model, "input": texts}, headers=headers)
        response.raise_for_status()
        payload = response.json()
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError(f"Некорректный ответ embedding endpoint: {payload!r}")
    vectors: list[list[float]] = []
    for row in data:
        emb = (row or {}).get("embedding")
        if not isinstance(emb, list) or not emb:
            raise ValueError(f"Некорректный embedding row: {row!r}")
        vectors.append([float(x) for x in emb])
    return vectors


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    sums = [0.0] * dim
    count = 0
    for vec in vectors:
        if len(vec) != dim:
            continue
        count += 1
        for i, value in enumerate(vec):
            sums[i] += float(value)
    if count <= 0:
        return []
    return [x / count for x in sums]


def _stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


@dataclass(slots=True)
class SimpleGraph:
    nodes: set[str]
    labels: dict[str, str]
    adjacency: dict[str, set[str]]
    directed_out: dict[str, set[str]]
    directed_in: dict[str, set[str]]


def build_store_graph(store: KnowledgeStore) -> SimpleGraph:
    nodes: set[str] = set()
    labels: dict[str, str] = {}
    adjacency: dict[str, set[str]] = defaultdict(set)
    directed_out: dict[str, set[str]] = defaultdict(set)
    directed_in: dict[str, set[str]] = defaultdict(set)

    for row in jsonl_read(store.entities_path):
        node_id = _safe_id(row.get("entity_id") or row.get("term_id") or row.get("label"))
        if not node_id:
            continue
        nodes.add(node_id)
        labels[node_id] = str(row.get("label") or row.get("term_id") or node_id)

    for row in jsonl_read(store.numeric_facts_path):
        fact_id = _safe_id(row.get("fact_id"))
        if fact_id:
            nodes.add(fact_id)
            labels.setdefault(fact_id, str(row.get("property_id") or row.get("property") or fact_id))
        for ent in row.get("entity_ids") or []:
            ent_id = f"onto:{ent}" if not str(ent).startswith("onto:") else str(ent)
            nodes.add(ent_id)
            labels.setdefault(ent_id, str(ent))
            if fact_id:
                adjacency[fact_id].add(ent_id)
                adjacency[ent_id].add(fact_id)
                directed_out[fact_id].add(ent_id)
                directed_in[ent_id].add(fact_id)

    for row in jsonl_read(store.triples_path):
        src = _safe_id(row.get("subject_id") or row.get("source"))
        dst = _safe_id(row.get("object_id") or row.get("target"))
        if not src or not dst:
            continue
        nodes.add(src)
        nodes.add(dst)
        labels.setdefault(src, src)
        labels.setdefault(dst, dst)
        adjacency[src].add(dst)
        adjacency[dst].add(src)
        directed_out[src].add(dst)
        directed_in[dst].add(src)

    # Ensure every known node has dict entries.
    for node in list(nodes):
        adjacency.setdefault(node, set())
        directed_out.setdefault(node, set())
        directed_in.setdefault(node, set())
        labels.setdefault(node, node)

    return SimpleGraph(nodes=nodes, labels=labels, adjacency=adjacency, directed_out=directed_out, directed_in=directed_in)


def connected_components(graph: SimpleGraph) -> list[set[str]]:
    seen: set[str] = set()
    comps: list[set[str]] = []
    for start in graph.nodes:
        if start in seen:
            continue
        comp: set[str] = set()
        q = deque([start])
        seen.add(start)
        while q:
            node = q.popleft()
            comp.add(node)
            for nb in graph.adjacency.get(node, set()):
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def pagerank(graph: SimpleGraph, *, iterations: int = 40, damping: float = 0.85) -> dict[str, float]:
    nodes = list(graph.nodes)
    n = len(nodes)
    if n == 0:
        return {}
    rank = {node: 1.0 / n for node in nodes}
    base = (1.0 - damping) / n
    for _ in range(max(1, iterations)):
        new_rank = {node: base for node in nodes}
        dangling = sum(rank[node] for node in nodes if not graph.directed_out.get(node))
        dangling_share = damping * dangling / n
        for node in nodes:
            new_rank[node] += dangling_share
        for src in nodes:
            outs = graph.directed_out.get(src) or set()
            if not outs:
                continue
            share = damping * rank[src] / len(outs)
            for dst in outs:
                new_rank[dst] = new_rank.get(dst, base) + share
        rank = new_rank
    return rank


def approximate_betweenness(graph: SimpleGraph, *, sample_size: int = 64, seed: int = 13) -> dict[str, float]:
    nodes = list(graph.nodes)
    if not nodes:
        return {}
    if len(nodes) <= sample_size:
        roots = nodes
    else:
        rnd = random.Random(seed)
        # Bias sampling towards connected/high-degree nodes.
        ranked = sorted(nodes, key=lambda n: len(graph.adjacency.get(n, set())), reverse=True)
        head = ranked[: sample_size // 2]
        tail_pool = ranked[sample_size // 2:]
        roots = head + rnd.sample(tail_pool, min(len(tail_pool), sample_size - len(head)))
    cb = {v: 0.0 for v in nodes}
    for s in roots:
        stack: list[str] = []
        pred: dict[str, list[str]] = {w: [] for w in nodes}
        sigma = dict.fromkeys(nodes, 0.0)
        sigma[s] = 1.0
        dist = dict.fromkeys(nodes, -1)
        dist[s] = 0
        queue = deque([s])
        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in graph.adjacency.get(v, set()):
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = dict.fromkeys(nodes, 0.0)
        while stack:
            w = stack.pop()
            if sigma[w] > 0:
                for v in pred[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                cb[w] += delta[w]
    scale = 1.0 / max(1, len(roots))
    return {node: value * scale for node, value in cb.items()}


def _top_labeled(scores: dict[str, float], labels: dict[str, str], *, top_n: int = 20) -> list[dict[str, Any]]:
    rows = []
    for node, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]:
        rows.append({"node_id": node, "label": labels.get(node, node), "score": round(float(score), 6)})
    return rows


def compute_graph_metrics(store: KnowledgeStore, *, top_n: int = 20, pagerank_iterations: int = 40, betweenness_sample: int = 64) -> dict[str, Any]:
    graph = build_store_graph(store)
    node_count = len(graph.nodes)
    edge_count = sum(len(v) for v in graph.adjacency.values()) // 2
    comps = connected_components(graph)
    largest = len(comps[0]) if comps else 0
    isolated = sum(1 for node in graph.nodes if not graph.adjacency.get(node))
    density = 0.0 if node_count <= 1 else (2.0 * edge_count) / (node_count * (node_count - 1))
    degree = {node: float(len(graph.adjacency.get(node, set()))) for node in graph.nodes}
    pr = pagerank(graph, iterations=pagerank_iterations)
    btw = approximate_betweenness(graph, sample_size=betweenness_sample)

    metrics = {
        "metrics_type": "NiCoThemeGraphMetrics",
        "computed_at": utc_now_iso(),
        "store_dir": str(store.root),
        "node_count": node_count,
        "edge_count": edge_count,
        "density": round(density, 8),
        "average_degree": round((2.0 * edge_count / node_count) if node_count else 0.0, 4),
        "connected_components": len(comps),
        "largest_component_size": largest,
        "largest_component_ratio": round((largest / node_count) if node_count else 0.0, 4),
        "isolated_nodes": isolated,
        "isolated_nodes_ratio": round((isolated / node_count) if node_count else 0.0, 4),
        "top_degree": _top_labeled(degree, graph.labels, top_n=top_n),
        "top_pagerank": _top_labeled(pr, graph.labels, top_n=top_n),
        "top_betweenness": _top_labeled(btw, graph.labels, top_n=top_n),
    }

    if isolated > max(10, int(0.25 * max(1, node_count))):
        metrics["health"] = "weak_graph"
        metrics["health_reason"] = "many_isolated_nodes"
    elif len(comps) > max(10, int(0.15 * max(1, node_count))):
        metrics["health"] = "split_needed"
        metrics["health_reason"] = "many_components"
    else:
        metrics["health"] = "ok"
        metrics["health_reason"] = ""
    return metrics


def write_graph_metrics(store: KnowledgeStore, *, top_n: int = 20, pagerank_iterations: int = 40, betweenness_sample: int = 64) -> dict[str, Any]:
    metrics = compute_graph_metrics(store, top_n=top_n, pagerank_iterations=pagerank_iterations, betweenness_sample=betweenness_sample)
    (store.root / "graph_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def load_graph_metrics(store_dir: Path) -> dict[str, Any]:
    path = store_dir / "graph_metrics.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _chunk_score(row: dict[str, Any]) -> tuple[int, int, int]:
    text = str(row.get("text") or "")
    tags = row.get("domain_tags") or []
    has_number = 1 if re.search(r"\d", text) else 0
    return (len(tags), has_number, min(len(text), 8000))


def build_theme_profile_text(store: KnowledgeStore, *, max_chunks: int = 32, max_chars: int = 24000) -> str:
    sources = jsonl_read(store.sources_path)
    chunks = jsonl_read(store.chunks_path)
    entities = jsonl_read(store.entities_path)
    facts = jsonl_read(store.numeric_facts_path)
    metrics = load_graph_metrics(store.root)

    parts: list[str] = []
    if sources:
        first = sources[0]
        parts.append(" ".join([
            str(first.get("collection") or ""),
            str(first.get("theme_name") or ""),
            str(first.get("source_type") or ""),
        ]))
    entity_labels = [str(row.get("label") or row.get("term_id") or "") for row in entities[:200]]
    parts.append("Top ontology terms: " + " ".join(entity_labels))
    fact_props = Counter(str(row.get("property_id") or row.get("property") or "unknown") for row in facts)
    parts.append("Numeric properties: " + " ".join(name for name, _ in fact_props.most_common(50)))
    central = []
    for key in ("top_pagerank", "top_betweenness", "top_degree"):
        central.extend(str(row.get("label") or "") for row in metrics.get(key) or [])
    parts.append("Graph central entities: " + " ".join(central[:120]))

    ranked_chunks = sorted(chunks, key=_chunk_score, reverse=True)[:max_chunks]
    used = 0
    for row in ranked_chunks:
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        text = re.sub(r"\s+", " ", text)
        if used + len(text) > max_chars:
            text = text[: max(0, max_chars - used)]
        if text:
            parts.append(text)
            used += len(text)
        if used >= max_chars:
            break
    return "\n".join(part for part in parts if part.strip())


def _theme_embedding_row_path(global_dir: Path) -> Path:
    return global_dir / "theme_embeddings.jsonl"


def load_theme_embeddings(global_dir: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("theme_id")): row for row in jsonl_read(_theme_embedding_row_path(global_dir)) if row.get("theme_id")}


async def build_theme_embeddings(
    *,
    project_root: Path,
    config: AppConfig,
    theme_ids: list[str] | None = None,
    max_chunks: int = 32,
) -> dict[str, Any]:
    knowledge_root = resolve_project_path(project_root, config.knowledge_store_dir, "./data/knowledge_store")
    themes_root = knowledge_root / "themes"
    global_dir = knowledge_root / "global"
    global_dir.mkdir(parents=True, exist_ok=True)
    if not themes_root.exists():
        return {"updated": 0, "skipped": True, "reason": "themes directory not found"}

    selected = set(theme_ids or [])
    rows_existing = load_theme_embeddings(global_dir)
    rows_out = dict(rows_existing)
    updates = 0
    errors: list[str] = []
    for theme_dir in sorted(p for p in themes_root.iterdir() if p.is_dir()):
        store = KnowledgeStore(theme_dir, schema_version=config.schema_version, ontology_version=config.ontology_version, app_version=config.app_version)
        first = next(iter(jsonl_read(store.sources_path)), {})
        theme_id = str(first.get("theme_id") or theme_dir.name)
        if selected and theme_id not in selected and theme_dir.name not in selected:
            continue
        text = build_theme_profile_text(store, max_chunks=max_chunks)
        if not text.strip():
            continue
        content_hash = _stable_hash(text)
        previous = rows_out.get(theme_id) or {}
        embedding_model = _theme_embedding_model(config)
        if previous.get("content_hash") == content_hash and previous.get("embedding_model") == embedding_model:
            continue
        try:
            vectors = await embed_texts_openai_compatible([text], config=config)
            vector = vectors[0]
        except Exception as exc:
            errors.append(f"{theme_id}: {exc}")
            continue
        metrics = load_graph_metrics(theme_dir)
        rows_out[theme_id] = {
            "theme_id": theme_id,
            "theme_dir": theme_dir.name,
            "collection": first.get("collection"),
            "theme_name": first.get("theme_name"),
            "source_type": first.get("source_type"),
            "embedding_model": embedding_model,
            "embedding_dim": len(vector),
            "content_hash": content_hash,
            "updated_at": utc_now_iso(),
            "profile_text_preview": text[:1200],
            "graph_health": metrics.get("health"),
            "vectors": {"profile": vector},
        }
        updates += 1
    jsonl_write(_theme_embedding_row_path(global_dir), rows_out.values())
    return {"updated": updates, "total": len(rows_out), "errors": errors[:50], "path": str(_theme_embedding_row_path(global_dir))}


def _overlap_score(query_tokens: set[str], candidate_text: str) -> float:
    if not query_tokens:
        return 0.0
    text = _norm(candidate_text)
    hits = sum(1 for token in query_tokens if token in text)
    return min(1.0, hits / max(1, min(len(query_tokens), 8)))


def _centrality_boost(query_tokens: set[str], stats: dict[str, Any]) -> float:
    metrics = stats.get("graph_metrics") or {}
    if not metrics:
        return 0.0
    score = 0.0
    for key, weight in (("top_pagerank", 1.0), ("top_betweenness", 0.8), ("top_degree", 0.5)):
        for row in metrics.get(key) or []:
            label = _norm(str(row.get("label") or row.get("node_id") or ""))
            if not label:
                continue
            if any(token in label or label in token for token in query_tokens):
                score += weight * min(1.0, _safe_float(row.get("score"), 0.0) * 10.0 + 0.1)
    return min(1.0, score)


def route_theme_scores(
    message: str,
    themes: list[dict[str, Any]],
    *,
    global_dir: Path,
    query_embedding: list[float] | None = None,
    max_themes: int = 5,
    min_readiness_rank: int = 2,
    readiness_order: dict[str, int] | None = None,
    min_score: float = 0.10,
) -> list[dict[str, Any]]:
    readiness_order = readiness_order or {"failed": -1, "not_ready": 0, "parsed_ready": 1, "search_ready": 2, "cheap_kg_ready": 3, "retrieval_kg_ready": 4, "compressed_kg_ready": 4, "full_kg_ready": 5}
    embeddings = load_theme_embeddings(global_dir)
    q_tokens = set(_tokens(message))
    q_norm = _norm(message)
    q_years = set(int(m.group(1)) for m in re.finditer(r"(?<!\d)((?:19|20)\d{2})(?!\d)", q_norm))
    scored: list[dict[str, Any]] = []
    for row in themes:
        status = str(row.get("status") or "not_ready")
        if readiness_order.get(status, 0) < min_readiness_rank:
            continue
        stats = row.get("stats") or {}
        text_parts = [
            str(row.get("theme_id") or ""),
            str(row.get("collection") or ""),
            str(row.get("theme_name") or ""),
            str(row.get("source_type") or ""),
            " ".join(str(x) for x in _as_list(stats.get("top_terms"))),
            " ".join(str(x) for x in _as_list(stats.get("top_entities"))),
            " ".join(str(x) for x in _as_list(stats.get("top_processes"))),
            " ".join(str(x) for x in _as_list(stats.get("years"))),
        ]
        theme_text = " ".join(text_parts)
        keyword = _overlap_score(q_tokens, theme_text)
        ontology = _overlap_score(q_tokens, " ".join(str(x) for x in _as_list(stats.get("top_terms")) + _as_list(stats.get("top_entities")) + _as_list(stats.get("top_processes"))))
        metadata = 0.0
        if _norm(str(row.get("theme_name") or "")) and _norm(str(row.get("theme_name") or "")) in q_norm:
            metadata += 0.6
        if _norm(str(row.get("collection") or "")) and _norm(str(row.get("collection") or "")) in q_norm:
            metadata += 0.25
        years = set(int(x) for x in _as_list(stats.get("years")) if str(x).isdigit())
        if row.get("year") and str(row.get("year")).isdigit():
            years.add(int(row.get("year")))
        if q_years and years.intersection(q_years):
            metadata += 0.35
        metadata = min(1.0, metadata)

        vector = 0.0
        emb_row = embeddings.get(str(row.get("theme_id") or "")) or {}
        theme_vec = ((emb_row.get("vectors") or {}).get("profile") or None)
        if query_embedding and theme_vec:
            vector = max(0.0, cosine_similarity(query_embedding, theme_vec))
        centrality = _centrality_boost(q_tokens, stats)
        # If embeddings are absent, redistribute vector mass to lexical and ontology signals.
        if query_embedding and theme_vec:
            score = 0.35 * vector + 0.25 * ontology + 0.20 * keyword + 0.10 * centrality + 0.10 * metadata
        else:
            score = 0.35 * ontology + 0.35 * keyword + 0.15 * centrality + 0.15 * metadata
        if int(stats.get("sources") or 0) > 0:
            score += 0.02
        item = {
            "theme_id": row.get("theme_id"),
            "score": round(score, 4),
            "vector": round(vector, 4),
            "ontology": round(ontology, 4),
            "keyword": round(keyword, 4),
            "centrality": round(centrality, 4),
            "metadata": round(metadata, 4),
            "status": status,
            "collection": row.get("collection"),
            "theme_name": row.get("theme_name"),
            "reason": _route_reason(q_tokens, row, stats),
        }
        if score >= min_score:
            scored.append(item)
    scored.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    return scored[:max_themes]


def _route_reason(query_tokens: set[str], row: dict[str, Any], stats: dict[str, Any]) -> str:
    hay = " ".join([
        str(row.get("theme_name") or ""),
        str(row.get("collection") or ""),
        " ".join(str(x) for x in _as_list(stats.get("top_terms"))),
        " ".join(str(x) for x in _as_list(stats.get("top_entities"))),
        " ".join(str(x) for x in _as_list(stats.get("top_processes"))),
    ])
    hnorm = _norm(hay)
    hits = [token for token in query_tokens if token in hnorm]
    if hits:
        return ", ".join(hits[:8])
    return "weak lexical/semantic match"


def build_quality_report(knowledge_root: Path) -> dict[str, Any]:
    themes_root = knowledge_root / "themes"
    report: dict[str, Any] = {"generated_at": utc_now_iso(), "themes": [], "duplicates": [], "warnings": []}
    if not themes_root.exists():
        report["warnings"].append("themes directory not found")
        return report
    doc_profiles: list[tuple[str, str, str]] = []
    for theme_dir in sorted(p for p in themes_root.iterdir() if p.is_dir()):
        store = KnowledgeStore(theme_dir)
        sources = jsonl_read(store.sources_path)
        facts = jsonl_read(store.numeric_facts_path)
        metrics = load_graph_metrics(theme_dir)
        first = sources[0] if sources else {}
        theme_id = str(first.get("theme_id") or theme_dir.name)
        issue_flags: list[str] = []
        if not sources:
            issue_flags.append("empty_theme")
        if metrics.get("health") in {"weak_graph", "split_needed"}:
            issue_flags.append(str(metrics.get("health")))
        no_unit = [row.get("fact_id") for row in facts if row.get("value") is not None and not row.get("unit")]
        if no_unit:
            issue_flags.append("numeric_facts_without_unit")
        unlinked = [row.get("fact_id") for row in facts if not row.get("chunk_id")]
        if unlinked:
            issue_flags.append("numeric_facts_without_chunk")
        report["themes"].append({
            "theme_id": theme_id,
            "theme_dir": theme_dir.name,
            "sources": len(sources),
            "numeric_facts": len(facts),
            "graph_health": metrics.get("health"),
            "isolated_nodes_ratio": metrics.get("isolated_nodes_ratio"),
            "largest_component_ratio": metrics.get("largest_component_ratio"),
            "issues": issue_flags,
        })
        for source in sources:
            text = " ".join([
                str(source.get("source_name") or ""),
                str(source.get("collection") or ""),
                str(source.get("theme_name") or ""),
            ])
            doc_profiles.append((theme_id, str(source.get("source_name") or ""), _stable_hash(_norm(text))))
    # Cheap exact-title-ish duplicate signal. Semantic duplicate detection is handled by embeddings later.
    seen: dict[str, tuple[str, str]] = {}
    for theme_id, source_name, h in doc_profiles:
        previous = seen.get(h)
        if previous:
            report["duplicates"].append({"hash": h, "a": {"theme_id": previous[0], "source_name": previous[1]}, "b": {"theme_id": theme_id, "source_name": source_name}})
        else:
            seen[h] = (theme_id, source_name)
    return report
