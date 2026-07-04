from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError


class AppConfig(BaseModel):
    """Runtime LLM/RAG settings exposed to the UI and RagService.

    YAML is the source of truth. Environment variables are used only to create the
    initial YAML file or to override values explicitly supplied by the operator.
    """

    app_symbol: str = Field(default="◈NiCo")
    app_title: str = Field(default="◈NiCo")
    llm_base_url: str = Field(default="http://localhost:11434/v1")
    llm_model: str = Field(default="qwen3:4b")
    llm_api_key: str = Field(default="ollama")
    embedding_base_url: str = Field(default="http://localhost:11434/v1")
    embedding_model: str = Field(default="bge-m3")
    embedding_dim: int = Field(default=1024)
    embedding_api_key: str = Field(default="ollama")
    query_mode: str = Field(default="hybrid")
    knowledge_store_dir: str = Field(default="./data/knowledge_store")
    runtime_rag_dir: str = Field(default="./data/rag_storage")
    lightrag_extraction_prompt_hardening: bool = Field(default=True)
    lightrag_extraction_output_repair: bool = Field(default=True)
    vector_cache_dir: str = Field(default="./data/vector_cache")
    snapshots_dir: str = Field(default="./data/snapshots")
    schema_version: str = Field(default="0.2")
    ontology_version: str = Field(default="nornickel-metallurgy-0.1")
    app_version: str = Field(default="nico-0.5.0")
    topic_sharding_enabled: bool = Field(default=True)
    topic_collection_level: int = Field(default=1)
    topic_theme_level: int = Field(default=2)
    topic_default_collection: str = Field(default="misc")
    topic_default_theme: str = Field(default="unclassified")
    topic_batch_size_docs: int = Field(default=8)
    topic_max_parallel_themes: int = Field(default=1)
    topic_max_parallel_docs_per_theme: int = Field(default=1)
    global_router_dir: str = Field(default="./data/knowledge_store/global")
    file_discovery_recursive: bool = Field(default=True)
    file_discovery_include_root_files: bool = Field(default=True)
    file_discovery_supported_extensions: list[str] = Field(default_factory=lambda: [
        ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".csv", ".txt", ".md", ".html", ".htm", ".epub"
    ])
    file_discovery_ignore_dirs: list[str] = Field(default_factory=lambda: [
        "__MACOSX", ".git", "node_modules", ".venv", "venv", "data", "uploads", "__pycache__"
    ])
    file_discovery_year_dir_pattern: str = Field(default=r"^(19|20)\d{2}$")
    theme_resolution_strategy: str = Field(default="hybrid")
    theme_use_path: bool = Field(default=True)
    theme_use_filename: bool = Field(default=True)
    theme_use_content_preview: bool = Field(default=True)
    theme_content_preview_chars: int = Field(default=5000)
    theme_min_confidence: float = Field(default=0.55)
    theme_auto_theme_from_filename: bool = Field(default=True)
    theme_auto_theme_from_content_terms: bool = Field(default=True)
    theme_collapse_low_confidence: bool = Field(default=True)
    theme_known_collections: dict[str, Any] = Field(default_factory=dict)
    theme_overrides: list[dict[str, Any]] = Field(default_factory=list)
    routing_top_k_themes: int = Field(default=5)
    routing_min_theme_score: float = Field(default=0.15)
    routing_fallback_to_global_chunks: bool = Field(default=True)
    graph_metrics_enabled: bool = Field(default=True)
    graph_metrics_top_n: int = Field(default=20)
    graph_pagerank_iterations: int = Field(default=25)
    graph_betweenness_sample: int = Field(default=32)
    theme_embeddings_enabled: bool = Field(default=False)
    theme_embeddings_max_chunks_per_theme: int = Field(default=12)
    theme_embeddings_embedding_base_url: str | None = Field(default=None)
    theme_embeddings_embedding_model: str | None = Field(default=None)
    theme_embeddings_embedding_dim: int | None = Field(default=None)
    theme_embeddings_embedding_api_key: str | None = Field(default=None)
    routing_use_theme_embeddings: bool = Field(default=False)
    routing_explain: bool = Field(default=True)
    ingestion_profiles: dict[str, Any] = Field(default_factory=dict)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return data


def _nested_get(data: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    node: Any = data
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _flatten_config(data: dict[str, Any]) -> dict[str, Any]:
    """Accept both nested config.yaml and legacy flat config payloads."""

    if not data:
        return {}

    # Flat payload from the old /api/config JSON contract.
    flat_keys = set(AppConfig.model_fields)
    if any(key in data for key in flat_keys):
        return {key: value for key, value in data.items() if key in flat_keys}

    return {
        "app_symbol": _nested_get(data, ("app", "symbol")),
        "app_title": _nested_get(data, ("app", "title")),
        "llm_base_url": _nested_get(data, ("models", "llm", "base_url")),
        "llm_model": _nested_get(data, ("models", "llm", "model")),
        "llm_api_key": _nested_get(data, ("models", "llm", "api_key")),
        "embedding_base_url": _nested_get(data, ("models", "embedding", "base_url")),
        "embedding_model": _nested_get(data, ("models", "embedding", "model")),
        "embedding_dim": _nested_get(data, ("models", "embedding", "dim")),
        "embedding_api_key": _nested_get(data, ("models", "embedding", "api_key")),
        "query_mode": _nested_get(data, ("retrieval", "query_mode")),
        "knowledge_store_dir": _nested_get(data, ("storage", "knowledge_store_dir")),
        "runtime_rag_dir": _nested_get(data, ("storage", "runtime_rag_dir")),
        "lightrag_extraction_prompt_hardening": _nested_get(data, ("rag", "extraction_prompt_hardening")),
        "lightrag_extraction_output_repair": _nested_get(data, ("rag", "extraction_output_repair")),
        "vector_cache_dir": _nested_get(data, ("storage", "vector_cache_dir")),
        "snapshots_dir": _nested_get(data, ("storage", "snapshots_dir")),
        "schema_version": _nested_get(data, ("versioning", "schema_version")),
        "ontology_version": _nested_get(data, ("versioning", "ontology_version")),
        "app_version": _nested_get(data, ("versioning", "app_version")),
        "topic_sharding_enabled": _nested_get(data, ("ingestion", "topic_layout", "enabled")),
        "topic_collection_level": _nested_get(data, ("ingestion", "topic_layout", "path_depth", "collection_level")),
        "topic_theme_level": _nested_get(data, ("ingestion", "topic_layout", "path_depth", "theme_level")),
        "topic_default_collection": _nested_get(data, ("ingestion", "topic_layout", "default_collection")),
        "topic_default_theme": _nested_get(data, ("ingestion", "topic_layout", "default_theme")),
        "topic_batch_size_docs": _nested_get(data, ("ingestion", "batching", "batch_size_docs")),
        "topic_max_parallel_themes": _nested_get(data, ("ingestion", "batching", "max_parallel_themes")),
        "topic_max_parallel_docs_per_theme": _nested_get(data, ("ingestion", "batching", "max_parallel_docs_per_theme")),
        "global_router_dir": _nested_get(data, ("storage", "global_router_dir")),
        "file_discovery_recursive": _nested_get(data, ("ingestion", "file_discovery", "recursive")),
        "file_discovery_include_root_files": _nested_get(data, ("ingestion", "file_discovery", "include_root_files")),
        "file_discovery_supported_extensions": _nested_get(data, ("ingestion", "file_discovery", "supported_extensions")),
        "file_discovery_ignore_dirs": _nested_get(data, ("ingestion", "file_discovery", "ignore_dirs")),
        "file_discovery_year_dir_pattern": _nested_get(data, ("ingestion", "file_discovery", "year_dir_pattern")),
        "theme_resolution_strategy": _nested_get(data, ("ingestion", "theme_resolution", "strategy")),
        "theme_use_path": _nested_get(data, ("ingestion", "theme_resolution", "use_path")),
        "theme_use_filename": _nested_get(data, ("ingestion", "theme_resolution", "use_filename")),
        "theme_use_content_preview": _nested_get(data, ("ingestion", "theme_resolution", "use_content_preview")),
        "theme_content_preview_chars": _nested_get(data, ("ingestion", "theme_resolution", "content_preview_chars")),
        "theme_min_confidence": _nested_get(data, ("ingestion", "theme_resolution", "min_confidence")),
        "theme_auto_theme_from_filename": _nested_get(data, ("ingestion", "theme_resolution", "auto_theme_from_filename")),
        "theme_auto_theme_from_content_terms": _nested_get(data, ("ingestion", "theme_resolution", "auto_theme_from_content_terms")),
        "theme_collapse_low_confidence": _nested_get(data, ("ingestion", "theme_resolution", "collapse_low_confidence")),
        "theme_known_collections": _nested_get(data, ("ingestion", "theme_resolution", "known_collections")),
        "theme_overrides": _nested_get(data, ("ingestion", "theme_overrides")),
        "routing_top_k_themes": _nested_get(data, ("ingestion", "routing", "top_k_themes")),
        "routing_min_theme_score": _nested_get(data, ("ingestion", "routing", "min_theme_score")),
        "routing_fallback_to_global_chunks": _nested_get(data, ("ingestion", "routing", "fallback_to_global_chunks")),
        "graph_metrics_enabled": _nested_get(data, ("intelligence", "graph_metrics", "enabled")),
        "graph_metrics_top_n": _nested_get(data, ("intelligence", "graph_metrics", "top_n")),
        "graph_pagerank_iterations": _nested_get(data, ("intelligence", "graph_metrics", "pagerank_iterations")),
        "graph_betweenness_sample": _nested_get(data, ("intelligence", "graph_metrics", "betweenness_sample")),
        "theme_embeddings_enabled": _nested_get(data, ("intelligence", "theme_embeddings", "enabled")),
        "theme_embeddings_max_chunks_per_theme": _nested_get(data, ("intelligence", "theme_embeddings", "max_chunks_per_theme")),
        "theme_embeddings_embedding_base_url": _nested_get(data, ("intelligence", "theme_embeddings", "embedding_base_url")),
        "theme_embeddings_embedding_model": _nested_get(data, ("intelligence", "theme_embeddings", "embedding_model")),
        "theme_embeddings_embedding_dim": _nested_get(data, ("intelligence", "theme_embeddings", "embedding_dim")),
        "theme_embeddings_embedding_api_key": _nested_get(data, ("intelligence", "theme_embeddings", "embedding_api_key")),
        "routing_use_theme_embeddings": _nested_get(data, ("ingestion", "routing", "use_theme_embeddings")),
        "routing_explain": _nested_get(data, ("ingestion", "routing", "explain")),
        "ingestion_profiles": _nested_get(data, ("ingestion", "profiles")),
    }


def _to_nested_yaml(config: AppConfig, existing: dict[str, Any] | None = None) -> dict[str, Any]:
    data = dict(existing or {})
    data.setdefault("app", {})
    data.setdefault("models", {})
    data.setdefault("retrieval", {})
    data.setdefault("server", {})
    data.setdefault("rag", {})
    data.setdefault("upload", {})
    data.setdefault("storage", {})
    data.setdefault("versioning", {})
    data.setdefault("ingestion", {})

    data["app"].update(
        {
            "symbol": config.app_symbol,
            "title": config.app_title,
        }
    )
    data["models"].setdefault("llm", {})
    data["models"]["llm"].update(
        {
            "base_url": config.llm_base_url,
            "model": config.llm_model,
            "api_key": config.llm_api_key,
        }
    )
    data["models"].setdefault("embedding", {})
    data["models"]["embedding"].update(
        {
            "base_url": config.embedding_base_url,
            "model": config.embedding_model,
            "dim": int(config.embedding_dim),
            "api_key": config.embedding_api_key,
        }
    )
    data["retrieval"].update({"query_mode": config.query_mode})
    data["storage"].update({
        "knowledge_store_dir": config.knowledge_store_dir,
        "runtime_rag_dir": config.runtime_rag_dir,
        "vector_cache_dir": config.vector_cache_dir,
        "snapshots_dir": config.snapshots_dir,
    })
    data["versioning"].update({
        "schema_version": config.schema_version,
        "ontology_version": config.ontology_version,
        "app_version": config.app_version,
    })

    data["ingestion"].setdefault("mode", "topic_sharded")
    data["ingestion"].setdefault("topic_layout", {})
    data["ingestion"]["topic_layout"].update({
        "enabled": bool(config.topic_sharding_enabled),
        "default_collection": config.topic_default_collection,
        "default_theme": config.topic_default_theme,
    })
    data["ingestion"]["topic_layout"].setdefault("path_depth", {})
    data["ingestion"]["topic_layout"]["path_depth"].update({
        "collection_level": int(config.topic_collection_level),
        "theme_level": int(config.topic_theme_level),
    })
    data["ingestion"].setdefault("batching", {})
    data["ingestion"]["batching"].update({
        "batch_size_docs": int(config.topic_batch_size_docs),
        "max_parallel_themes": int(config.topic_max_parallel_themes),
        "max_parallel_docs_per_theme": int(config.topic_max_parallel_docs_per_theme),
        "stop_on_theme_error": False,
    })
    data["storage"].setdefault("global_router_dir", config.global_router_dir)

    data["ingestion"].setdefault("file_discovery", {})
    data["ingestion"]["file_discovery"].update({
        "recursive": bool(config.file_discovery_recursive),
        "include_root_files": bool(config.file_discovery_include_root_files),
        "supported_extensions": list(config.file_discovery_supported_extensions),
        "ignore_dirs": list(config.file_discovery_ignore_dirs),
        "year_dir_pattern": config.file_discovery_year_dir_pattern,
    })
    data["ingestion"].setdefault("theme_resolution", {})
    data["ingestion"]["theme_resolution"].update({
        "strategy": config.theme_resolution_strategy,
        "use_path": bool(config.theme_use_path),
        "use_filename": bool(config.theme_use_filename),
        "use_content_preview": bool(config.theme_use_content_preview),
        "content_preview_chars": int(config.theme_content_preview_chars),
        "min_confidence": float(config.theme_min_confidence),
        "auto_theme_from_filename": bool(config.theme_auto_theme_from_filename),
        "auto_theme_from_content_terms": bool(config.theme_auto_theme_from_content_terms),
        "collapse_low_confidence": bool(config.theme_collapse_low_confidence),
        "known_collections": dict(config.theme_known_collections),
    })
    data["ingestion"]["theme_overrides"] = list(config.theme_overrides)
    data["ingestion"].setdefault("routing", {})
    data["ingestion"]["routing"].update({
        "top_k_themes": int(config.routing_top_k_themes),
        "min_theme_score": float(config.routing_min_theme_score),
        "fallback_to_global_chunks": bool(config.routing_fallback_to_global_chunks),
        "use_theme_embeddings": bool(config.routing_use_theme_embeddings),
        "explain": bool(config.routing_explain),
    })
    if config.ingestion_profiles:
        data["ingestion"]["profiles"] = dict(config.ingestion_profiles)

    data.setdefault("intelligence", {})
    data["intelligence"].setdefault("graph_metrics", {})
    data["intelligence"]["graph_metrics"].update({
        "enabled": bool(config.graph_metrics_enabled),
        "top_n": int(config.graph_metrics_top_n),
        "pagerank_iterations": int(config.graph_pagerank_iterations),
        "betweenness_sample": int(config.graph_betweenness_sample),
    })
    data["intelligence"].setdefault("theme_embeddings", {})
    data["intelligence"]["theme_embeddings"].update({
        "enabled": bool(config.theme_embeddings_enabled),
        "max_chunks_per_theme": int(config.theme_embeddings_max_chunks_per_theme),
    })
    if config.theme_embeddings_embedding_base_url:
        data["intelligence"]["theme_embeddings"]["embedding_base_url"] = config.theme_embeddings_embedding_base_url
    if config.theme_embeddings_embedding_model:
        data["intelligence"]["theme_embeddings"]["embedding_model"] = config.theme_embeddings_embedding_model
    if config.theme_embeddings_embedding_dim:
        data["intelligence"]["theme_embeddings"]["embedding_dim"] = int(config.theme_embeddings_embedding_dim)
    if config.theme_embeddings_embedding_api_key:
        data["intelligence"]["theme_embeddings"]["embedding_api_key"] = config.theme_embeddings_embedding_api_key

    data["server"].setdefault("port", 8090)
    data["server"].setdefault("reload", False)
    data["rag"].setdefault("working_dir", config.runtime_rag_dir)
    data["rag"]["extraction_prompt_hardening"] = bool(config.lightrag_extraction_prompt_hardening)
    data["rag"]["extraction_output_repair"] = bool(config.lightrag_extraction_output_repair)
    data["rag"].setdefault("auto_resume", False)
    data["upload"].setdefault("store_dir", "./uploads")
    return data


def _env_or_default(name: str, default: Any) -> Any:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _config_from_env() -> AppConfig:
    defaults = AppConfig()
    return AppConfig(
        app_symbol=_env_or_default("APP_SYMBOL", defaults.app_symbol),
        app_title=_env_or_default("APP_TITLE", defaults.app_title),
        llm_base_url=_env_or_default("LLM_BASE_URL", defaults.llm_base_url),
        llm_model=_env_or_default("LLM_MODEL", defaults.llm_model),
        llm_api_key=_env_or_default("LLM_API_KEY", defaults.llm_api_key),
        embedding_base_url=_env_or_default("EMBEDDING_BASE_URL", defaults.embedding_base_url),
        embedding_model=_env_or_default("EMBEDDING_MODEL", defaults.embedding_model),
        embedding_dim=int(_env_or_default("EMBEDDING_DIM", defaults.embedding_dim)),
        embedding_api_key=_env_or_default("EMBEDDING_API_KEY", defaults.embedding_api_key),
        query_mode=_env_or_default("QUERY_MODE", defaults.query_mode),
        knowledge_store_dir=_env_or_default("KNOWLEDGE_STORE_DIR", defaults.knowledge_store_dir),
        runtime_rag_dir=_env_or_default("RUNTIME_RAG_DIR", defaults.runtime_rag_dir),
        lightrag_extraction_prompt_hardening=str(_env_or_default("LIGHTRAG_EXTRACTION_PROMPT_HARDENING", defaults.lightrag_extraction_prompt_hardening)).lower() in {"1", "true", "yes", "on"},
        lightrag_extraction_output_repair=str(_env_or_default("LIGHTRAG_EXTRACTION_OUTPUT_REPAIR", defaults.lightrag_extraction_output_repair)).lower() in {"1", "true", "yes", "on"},
        vector_cache_dir=_env_or_default("VECTOR_CACHE_DIR", defaults.vector_cache_dir),
        snapshots_dir=_env_or_default("SNAPSHOTS_DIR", defaults.snapshots_dir),
        schema_version=_env_or_default("SCHEMA_VERSION", defaults.schema_version),
        ontology_version=_env_or_default("ONTOLOGY_VERSION", defaults.ontology_version),
        app_version=_env_or_default("APP_VERSION", defaults.app_version),
        topic_sharding_enabled=str(_env_or_default("TOPIC_SHARDING_ENABLED", defaults.topic_sharding_enabled)).lower() in {"1", "true", "yes", "on"},
        topic_collection_level=int(_env_or_default("TOPIC_COLLECTION_LEVEL", defaults.topic_collection_level)),
        topic_theme_level=int(_env_or_default("TOPIC_THEME_LEVEL", defaults.topic_theme_level)),
        topic_default_collection=_env_or_default("TOPIC_DEFAULT_COLLECTION", defaults.topic_default_collection),
        topic_default_theme=_env_or_default("TOPIC_DEFAULT_THEME", defaults.topic_default_theme),
        topic_batch_size_docs=int(_env_or_default("TOPIC_BATCH_SIZE_DOCS", defaults.topic_batch_size_docs)),
        topic_max_parallel_themes=int(_env_or_default("TOPIC_MAX_PARALLEL_THEMES", defaults.topic_max_parallel_themes)),
        topic_max_parallel_docs_per_theme=int(_env_or_default("TOPIC_MAX_PARALLEL_DOCS_PER_THEME", defaults.topic_max_parallel_docs_per_theme)),
        global_router_dir=_env_or_default("GLOBAL_ROUTER_DIR", defaults.global_router_dir),
        file_discovery_recursive=str(_env_or_default("FILE_DISCOVERY_RECURSIVE", defaults.file_discovery_recursive)).lower() in {"1", "true", "yes", "on"},
        file_discovery_include_root_files=str(_env_or_default("FILE_DISCOVERY_INCLUDE_ROOT_FILES", defaults.file_discovery_include_root_files)).lower() in {"1", "true", "yes", "on"},
        file_discovery_supported_extensions=defaults.file_discovery_supported_extensions,
        file_discovery_ignore_dirs=defaults.file_discovery_ignore_dirs,
        file_discovery_year_dir_pattern=_env_or_default("FILE_DISCOVERY_YEAR_DIR_PATTERN", defaults.file_discovery_year_dir_pattern),
        theme_resolution_strategy=_env_or_default("THEME_RESOLUTION_STRATEGY", defaults.theme_resolution_strategy),
        theme_use_path=str(_env_or_default("THEME_USE_PATH", defaults.theme_use_path)).lower() in {"1", "true", "yes", "on"},
        theme_use_filename=str(_env_or_default("THEME_USE_FILENAME", defaults.theme_use_filename)).lower() in {"1", "true", "yes", "on"},
        theme_use_content_preview=str(_env_or_default("THEME_USE_CONTENT_PREVIEW", defaults.theme_use_content_preview)).lower() in {"1", "true", "yes", "on"},
        theme_content_preview_chars=int(_env_or_default("THEME_CONTENT_PREVIEW_CHARS", defaults.theme_content_preview_chars)),
        theme_min_confidence=float(_env_or_default("THEME_MIN_CONFIDENCE", defaults.theme_min_confidence)),
        theme_auto_theme_from_filename=str(_env_or_default("THEME_AUTO_THEME_FROM_FILENAME", defaults.theme_auto_theme_from_filename)).lower() in {"1", "true", "yes", "on"},
        theme_auto_theme_from_content_terms=str(_env_or_default("THEME_AUTO_THEME_FROM_CONTENT_TERMS", defaults.theme_auto_theme_from_content_terms)).lower() in {"1", "true", "yes", "on"},
        theme_collapse_low_confidence=str(_env_or_default("THEME_COLLAPSE_LOW_CONFIDENCE", defaults.theme_collapse_low_confidence)).lower() in {"1", "true", "yes", "on"},
        theme_known_collections=defaults.theme_known_collections,
        theme_overrides=defaults.theme_overrides,
        routing_top_k_themes=int(_env_or_default("ROUTING_TOP_K_THEMES", defaults.routing_top_k_themes)),
        routing_min_theme_score=float(_env_or_default("ROUTING_MIN_THEME_SCORE", defaults.routing_min_theme_score)),
        routing_fallback_to_global_chunks=str(_env_or_default("ROUTING_FALLBACK_TO_GLOBAL_CHUNKS", defaults.routing_fallback_to_global_chunks)).lower() in {"1", "true", "yes", "on"},
        graph_metrics_enabled=str(_env_or_default("GRAPH_METRICS_ENABLED", defaults.graph_metrics_enabled)).lower() in {"1", "true", "yes", "on"},
        graph_metrics_top_n=int(_env_or_default("GRAPH_METRICS_TOP_N", defaults.graph_metrics_top_n)),
        graph_pagerank_iterations=int(_env_or_default("GRAPH_PAGERANK_ITERATIONS", defaults.graph_pagerank_iterations)),
        graph_betweenness_sample=int(_env_or_default("GRAPH_BETWEENNESS_SAMPLE", defaults.graph_betweenness_sample)),
        theme_embeddings_enabled=str(_env_or_default("THEME_EMBEDDINGS_ENABLED", defaults.theme_embeddings_enabled)).lower() in {"1", "true", "yes", "on"},
        theme_embeddings_max_chunks_per_theme=int(_env_or_default("THEME_EMBEDDINGS_MAX_CHUNKS_PER_THEME", defaults.theme_embeddings_max_chunks_per_theme)),
        theme_embeddings_embedding_base_url=_env_or_default("THEME_EMBEDDINGS_EMBEDDING_BASE_URL", defaults.theme_embeddings_embedding_base_url),
        theme_embeddings_embedding_model=_env_or_default("THEME_EMBEDDINGS_EMBEDDING_MODEL", defaults.theme_embeddings_embedding_model),
        theme_embeddings_embedding_dim=(int(_env_or_default("THEME_EMBEDDINGS_EMBEDDING_DIM", defaults.theme_embeddings_embedding_dim)) if _env_or_default("THEME_EMBEDDINGS_EMBEDDING_DIM", defaults.theme_embeddings_embedding_dim) else None),
        theme_embeddings_embedding_api_key=_env_or_default("THEME_EMBEDDINGS_EMBEDDING_API_KEY", defaults.theme_embeddings_embedding_api_key),
        routing_use_theme_embeddings=str(_env_or_default("ROUTING_USE_THEME_EMBEDDINGS", defaults.routing_use_theme_embeddings)).lower() in {"1", "true", "yes", "on"},
        routing_explain=str(_env_or_default("ROUTING_EXPLAIN", defaults.routing_explain)).lower() in {"1", "true", "yes", "on"},
    )


class ConfigManager:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

    def load_raw(self) -> dict[str, Any]:
        return _read_yaml(self.config_path)

    def load(self) -> AppConfig:
        if not self.config_path.exists():
            config = _config_from_env()
            self.save(config)
            return config

        raw = self.load_raw()
        flat = {key: value for key, value in _flatten_config(raw).items() if value is not None}
        try:
            return AppConfig.model_validate(flat)
        except ValidationError as exc:
            raise ValueError(f"Некорректный YAML config {self.config_path}: {exc}") from exc

    def save(self, config: AppConfig) -> None:
        existing = self.load_raw() if self.config_path.exists() else {}
        nested = _to_nested_yaml(config, existing)
        self.config_path.write_text(
            yaml.safe_dump(nested, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def update(self, payload: dict[str, Any]) -> AppConfig:
        current = self.load()
        updated = current.model_copy(update=payload)
        self.save(updated)
        return updated


def read_project_yaml(config_path: Path) -> dict[str, Any]:
    return _read_yaml(config_path)


def resolve_project_path(root: Path, value: str | os.PathLike[str] | None, default: str) -> Path:
    raw = Path(str(value or default)).expanduser()
    if raw.is_absolute():
        return raw
    return (root / raw).resolve()
