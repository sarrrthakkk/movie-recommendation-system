"""YAML configuration loaded into validated models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from movie_recommender.utils import project_root


class DataConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    relevance_threshold: float = 4.0
    minimum_positive_interactions: int = 3

    @field_validator("relevance_threshold")
    @classmethod
    def threshold_in_range(cls, value: float) -> float:
        if not 1.0 <= value <= 5.0:
            raise ValueError("relevance_threshold must be between 1 and 5")
        return value

    @field_validator("minimum_positive_interactions")
    @classmethod
    def minimum_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("minimum_positive_interactions must be at least 1")
        return value


class MatrixFactorizationConfig(BaseModel):
    latent_dimensions: int = 50
    learning_rate: float = 0.005
    regularization: float = 0.01
    epochs: int = 50
    batch_size: int = 1024
    patience: int = 5
    device: str = "auto"

    @field_validator("latent_dimensions", "epochs", "batch_size", "patience")
    @classmethod
    def positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("matrix factorization settings must be positive")
        return value

    @field_validator("learning_rate", "regularization")
    @classmethod
    def non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("learning_rate and regularization must be non-negative")
        return value


class PopularityConfig(BaseModel):
    prior_strength: int = 10

    @field_validator("prior_strength")
    @classmethod
    def positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("prior_strength must be at least 1")
        return value


class HybridConfig(BaseModel):
    alpha_values: list[float] = Field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    top_k_candidates: int = 20

    @field_validator("alpha_values")
    @classmethod
    def alphas_valid(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("alpha_values must not be empty")
        if any(alpha < 0 or alpha > 1 for alpha in value):
            raise ValueError("each alpha must be within [0, 1]")
        return value

    @field_validator("top_k_candidates")
    @classmethod
    def candidates_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("top_k_candidates must be at least 1")
        return value


class EvaluationConfig(BaseModel):
    k_values: list[int] = Field(default_factory=lambda: [5, 10])
    llm_sample_users: int = 100
    bootstrap_resamples: int = 1000

    @field_validator("k_values")
    @classmethod
    def k_positive(cls, value: list[int]) -> list[int]:
        if not value or any(k < 1 for k in value):
            raise ValueError("k_values must be positive")
        return value

    @field_validator("llm_sample_users", "bootstrap_resamples")
    @classmethod
    def positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("evaluation sample sizes must be positive")
        return value


class LLMConfig(BaseModel):
    model: str = "qwen35-27b"
    temperature: float = 0.0
    requests_per_minute: int = 6
    concurrency: int = 1
    request_timeout_seconds: float = 90
    maximum_retries: int = 3
    retry_base_seconds: float = 2
    maximum_candidates: int = 20
    maximum_positive_history: int = 10
    maximum_negative_history: int = 5

    @field_validator("temperature")
    @classmethod
    def temperature_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("temperature must be non-negative")
        return value

    @field_validator("requests_per_minute", "concurrency", "maximum_retries")
    @classmethod
    def positive_limits(cls, value: int) -> int:
        if value < 1:
            raise ValueError("LLM rate, concurrency, and retries must be at least 1")
        return value

    @field_validator("maximum_candidates", "maximum_positive_history", "maximum_negative_history")
    @classmethod
    def positive_context(cls, value: int) -> int:
        if value < 1:
            raise ValueError("LLM context limits must be at least 1")
        return value


class AppConfig(BaseModel):
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    matrix_factorization: MatrixFactorizationConfig = Field(default_factory=MatrixFactorizationConfig)
    popularity: PopularityConfig = Field(default_factory=PopularityConfig)
    hybrid: HybridConfig = Field(default_factory=HybridConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    artifacts_dir: str = "artifacts"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        config_path = Path(path)
        if not config_path.is_absolute():
            config_path = project_root() / config_path
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with config_path.open(encoding="utf-8") as handle:
            raw: Any = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError(f"Configuration file must contain a mapping: {config_path}")
        return cls.model_validate(raw)

    def with_smoke_defaults(self) -> "AppConfig":
        """Return a copy with a short training schedule for smoke experiments."""
        cfg = self.model_copy(deep=True)
        cfg.matrix_factorization.epochs = min(cfg.matrix_factorization.epochs, 5)
        cfg.matrix_factorization.latent_dimensions = min(
            cfg.matrix_factorization.latent_dimensions, 16
        )
        cfg.matrix_factorization.patience = min(cfg.matrix_factorization.patience, 2)
        cfg.evaluation.llm_sample_users = min(cfg.evaluation.llm_sample_users, 25)
        cfg.evaluation.bootstrap_resamples = min(cfg.evaluation.bootstrap_resamples, 200)
        return cfg
