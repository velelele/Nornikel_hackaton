from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    llm_base_url: str = Field(default="http://localhost:11434/v1")
    llm_model: str = Field(default="llama3.2:latest")
    llm_api_key: str = Field(default="ollama")
    embedding_base_url: str = Field(default="http://localhost:11434/v1")
    embedding_model: str = Field(default="nomic-embed-text:latest")
    embedding_dim: int = Field(default=768)
    embedding_api_key: str = Field(default="ollama")
    query_mode: str = Field(default="hybrid")


class ConfigManager:
    def __init__(self, settings_path: Path) -> None:
        self.settings_path = settings_path
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> AppConfig:
        if self.settings_path.exists():
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
            return AppConfig.model_validate(data)

        config = AppConfig(
            llm_base_url=os.getenv("LLM_BASE_URL", AppConfig.model_fields["llm_base_url"].default),
            llm_model=os.getenv("LLM_MODEL", AppConfig.model_fields["llm_model"].default),
            llm_api_key=os.getenv("LLM_API_KEY", AppConfig.model_fields["llm_api_key"].default),
            embedding_base_url=os.getenv(
                "EMBEDDING_BASE_URL",
                AppConfig.model_fields["embedding_base_url"].default,
            ),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                AppConfig.model_fields["embedding_model"].default,
            ),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
            embedding_api_key=os.getenv(
                "EMBEDDING_API_KEY",
                AppConfig.model_fields["embedding_api_key"].default,
            ),
        )
        self.save(config)
        return config

    def save(self, config: AppConfig) -> None:
        self.settings_path.write_text(
            config.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def update(self, payload: dict[str, Any]) -> AppConfig:
        current = self.load()
        updated = current.model_copy(update=payload)
        self.save(updated)
        return updated
