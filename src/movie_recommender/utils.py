"""Paths, seeds, logging, and small I/O helpers."""

from __future__ import annotations

import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is a required dependency
    torch = None  # type: ignore[assignment]


def project_root() -> Path:
    """Return the repository root that contains ``pyproject.toml``."""
    here = Path(__file__).resolve()
    candidates = [here.parents[2], Path.cwd().resolve(), *Path.cwd().resolve().parents]
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() and (
            candidate / "src" / "movie_recommender"
        ).exists():
            return candidate
    return here.parents[2]


def resolve_path(path: str | Path) -> Path:
    """Resolve a project-relative path without depending on the shell cwd."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch. GPU kernels may still be nondeterministic."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str | None = None):
    """Select cuda, then Apple MPS, then CPU."""
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RedactingFormatter(logging.Formatter):
    """Strip the API key from any log line if it is present in the environment."""

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        key = os.environ.get("LLM_API_KEY", "").strip()
        if key and key in text:
            text = text.replace(key, "[REDACTED]")
        return text


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        RedactingFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def json_safe(value: Any) -> Any:
    """Convert NumPy values and non-finite floats into JSON-friendly objects."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.floating):
        value = float(value)
    elif isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def save_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=False) + "\n")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def estimate_tokens(text: str) -> int:
    """Rough token estimate. Four characters is about one token for English text."""
    return max(1, len(text) // 4)
