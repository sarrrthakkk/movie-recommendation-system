"""Fit traditional models and choose the hybrid alpha on validation data only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from movie_recommender.config import AppConfig
from movie_recommender.data import load_processed
from movie_recommender.evaluation import ndcg_at_k, rank_of
from movie_recommender.models.content_based import ContentModel
from movie_recommender.models.matrix_factorization import MatrixFactorizationRecommender
from movie_recommender.models.popularity import PopularityModel
from movie_recommender.plotting import plot_alpha, plot_training_curves
from movie_recommender.ranking import prepare_users, rankings_for_score
from movie_recommender.utils import ensure_dir, get_logger, resolve_path, save_json

logger = get_logger(__name__)


def _dirs(
    config: AppConfig,
    processed_dir: str | Path | None,
    artifacts_dir: str | Path | None,
) -> tuple[Path, Path]:
    processed = Path(processed_dir) if processed_dir else resolve_path(config.data.processed_dir)
    artifacts = Path(artifacts_dir) if artifacts_dir else resolve_path(config.artifacts_dir)
    return processed, artifacts


def select_alpha(
    prepared,
    targets: dict[int, int],
    alpha_values: list[float],
    k: int = 10,
) -> tuple[float, pd.DataFrame]:
    """Choose alpha by mean validation NDCG@K.

    Ties prefer the alpha closest to 0.5, then the smaller alpha. The targets
    passed here must be validation targets. This function does not read the test set.
    """
    rows = []
    for alpha in alpha_values:
        rankings = rankings_for_score(prepared, "hybrid", alpha=float(alpha))
        scores = [
            ndcg_at_k(rank_of(ranked, int(targets[user_id])), k)
            for user_id, ranked in rankings.items()
        ]
        rows.append(
            {
                "alpha": float(alpha),
                "ndcg@10": float(np.mean(scores)) if scores else float("nan"),
                "n_users": len(scores),
                "distance_to_half": abs(float(alpha) - 0.5),
            }
        )
        logger.info("Validation NDCG@%d at alpha %.2f is %.4f", k, alpha, rows[-1]["ndcg@10"])
    table = pd.DataFrame(rows).sort_values(
        ["ndcg@10", "distance_to_half", "alpha"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    if table.empty or pd.isna(table.iloc[0]["ndcg@10"]):
        raise RuntimeError("Alpha selection produced no validation scores")
    return float(table.iloc[0]["alpha"]), table.reset_index(drop=True)


def train_models(
    config: AppConfig,
    *,
    smoke: bool = False,
    processed_dir: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    device: str | None = None,
) -> dict:
    """Train on the chronological training split and select alpha on validation."""
    if smoke:
        config = config.with_smoke_defaults()
    processed, artifacts = _dirs(config, processed_dir, artifacts_dir)
    data = load_processed(processed)
    model_dir = ensure_dir(artifacts / "models")
    metrics_dir = ensure_dir(artifacts / "metrics")
    plots_dir = ensure_dir(artifacts / "plots")
    movie_ids = data.movies["movie_id"].to_numpy()
    popularity = PopularityModel(prior_strength=config.popularity.prior_strength).fit(
        data.train,
        movie_ids,
        relevance_threshold=config.data.relevance_threshold,
    )
    popularity.save(model_dir / "popularity.json")
    content = ContentModel().fit(movie_ids, data.genres)
    content.save(model_dir)
    mf = MatrixFactorizationRecommender()
    mf_config = config.matrix_factorization
    mf_metadata = mf.fit(
        data.train,
        data.validation,
        movie_ids,
        n_factors=mf_config.latent_dimensions,
        learning_rate=mf_config.learning_rate,
        regularization=mf_config.regularization,
        epochs=mf_config.epochs,
        batch_size=mf_config.batch_size,
        patience=mf_config.patience,
        seed=config.seed,
        device=device or mf_config.device,
        checkpoint_dir=model_dir,
    )
    plot_training_curves(mf.history, plots_dir / "mf_training_curves.png")
    user_ids = data.validation["user_id"].astype(int).tolist()
    targets = dict(zip(data.validation["user_id"].astype(int), data.validation["movie_id"].astype(int)))
    prepared = prepare_users(user_ids, data.train, movie_ids, mf, content, popularity)
    selected, alpha_table = select_alpha(prepared, targets, config.hybrid.alpha_values, k=10)
    alpha_table.to_csv(metrics_dir / "alpha_validation.csv", index=False)
    save_json(metrics_dir / "alpha_validation.json", alpha_table.to_dict(orient="records"))
    plot_alpha(alpha_table.sort_values("alpha"), plots_dir / "alpha_validation.png")
    payload = {
        "alpha": selected,
        "validation_ndcg@10": float(alpha_table.loc[alpha_table["alpha"] == selected, "ndcg@10"].iloc[0]),
        "tie_break": "highest validation NDCG@10, then closest to 0.5, then smaller alpha",
        "smoke": smoke,
        "seed": config.seed,
        "result_type": "smoke" if smoke else "full",
        "note": "Alpha was selected on validation rankings only. Test labels were not used.",
    }
    save_json(model_dir / "selected_alpha.json", payload)
    mf.metadata.update({"smoke": smoke, "result_type": payload["result_type"]})
    mf.save(model_dir)
    run_metadata = {
        "smoke": smoke,
        "result_type": payload["result_type"],
        "seed": config.seed,
        "selected_alpha": selected,
        "matrix_factorization": mf_metadata,
        "config": config.model_dump(),
    }
    save_json(model_dir / "run_metadata.json", run_metadata)
    (metrics_dir / "run_config.yaml").write_text(
        yaml.safe_dump(config.model_dump(), sort_keys=False),
        encoding="utf-8",
    )
    logger.info("Selected hybrid alpha %.2f on validation NDCG@10", selected)
    return run_metadata
