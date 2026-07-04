import os
from pathlib import Path

import uvicorn
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
CONFIG_PATH = Path(os.getenv("APP_CONFIG_PATH", ROOT / "config.yaml"))


def _read_yaml() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _get(data: dict, path: tuple[str, ...], default=None):
    node = data
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _bool_value(value, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    project_config = _read_yaml()
    host = os.getenv("HOST") or _get(project_config, ("server", "host"), "0.0.0.0")
    port = int(os.getenv("PORT") or _get(project_config, ("server", "port"), 8090))
    reload = _bool_value(os.getenv("RAG_RELOAD"), default=_bool_value(_get(project_config, ("server", "reload")), default=False))

    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=reload,
    )
