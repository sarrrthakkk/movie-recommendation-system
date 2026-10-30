"""Prepare MovieLens files and the chronological split."""

from __future__ import annotations

from pathlib import Path

import yaml

from movie_recommender.config import AppConfig
from movie_recommender.data import load_movielens, save_processed
from movie_recommender.plotting import save_eda_plots
from movie_recommender.splitting import make_chronological_split
from movie_recommender.utils import ensure_dir, get_logger, resolve_path, save_json

logger = get_logger(__name__)


def prepare_dataset(
    data_dir: str | Path,
    config: AppConfig,
    *,
    processed_dir: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
) -> dict:
    """Parse MovieLens, split it chronologically, and write processed files."""
    processed = Path(processed_dir) if processed_dir else resolve_path(config.data.processed_dir)
    artifacts = Path(artifacts_dir) if artifacts_dir else resolve_path(config.artifacts_dir)
    ratings, movies, genres, report = load_movielens(data_dir)
    split = make_chronological_split(
        ratings,
        relevance_threshold=config.data.relevance_threshold,
        minimum_positive_interactions=config.data.minimum_positive_interactions,
        n_movies=len(movies),
    )
    summary = {**report, **split.summary, "seed": config.seed}
    save_processed(
        processed,
        ratings,
        movies,
        genres,
        split.train,
        split.validation,
        split.test,
        summary,
    )
    metrics_dir = ensure_dir(artifacts / "metrics")
    plots_dir = ensure_dir(artifacts / "plots")
    save_json(metrics_dir / "split_summary.json", summary)
    save_eda_plots(ratings, split.train, plots_dir)
    (metrics_dir / "run_config.yaml").write_text(
        yaml.safe_dump(config.model_dump(), sort_keys=False),
        encoding="utf-8",
    )
    logger.info(
        "Prepared %d eligible users, %d train rows, %d validation rows, %d test rows",
        summary["n_eligible_users"],
        summary["n_train_interactions"],
        summary["n_validation_interactions"],
        summary["n_test_interactions"],
    )
    return summary
