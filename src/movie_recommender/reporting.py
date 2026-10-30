"""Test-set evaluation for the non-LLM recommenders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from movie_recommender.config import AppConfig
from movie_recommender.data import load_processed
from movie_recommender.evaluation import (
    ndcg_at_k,
    per_user_metrics,
    popularity_buckets,
    rank_of,
    summarize_metric_frame,
)
from movie_recommender.models.content_based import ContentModel
from movie_recommender.models.matrix_factorization import MatrixFactorizationRecommender
from movie_recommender.models.popularity import PopularityModel
from movie_recommender.plotting import plot_buckets, plot_model_comparison, save_eda_plots
from movie_recommender.ranking import prepare_users, rankings_for_score
from movie_recommender.utils import ensure_dir, get_logger, load_json, resolve_path, save_json

logger = get_logger(__name__)

MODEL_NAMES = [
    "popularity_bayesian",
    "popularity_positive_count",
    "matrix_factorization",
    "content",
    "hybrid",
]


def _dirs(config: AppConfig, processed_dir, artifacts_dir) -> tuple[Path, Path]:
    processed = Path(processed_dir) if processed_dir else resolve_path(config.data.processed_dir)
    artifacts = Path(artifacts_dir) if artifacts_dir else resolve_path(config.artifacts_dir)
    return processed, artifacts


def _genre_lookup(movies: pd.DataFrame, genres: np.ndarray) -> dict[int, np.ndarray]:
    return {int(movie_id): genres[row] for row, movie_id in enumerate(movies["movie_id"].astype(int))}


def _flat_summary(model: str, summary: dict) -> dict:
    row: dict = {"model": model, "n_users": summary["n_users"]}
    for key, value in summary.items():
        if isinstance(value, dict) and "mean" in value:
            row[key] = value["mean"]
            row[f"{key}_low"] = value["low"]
            row[f"{key}_high"] = value["high"]
        elif key != "n_users":
            row[key] = value
    return row


def _truncate_history(history: pd.DataFrame, threshold: float, keep_positive: int = 3) -> pd.DataFrame | None:
    ordered = history.sort_values(["timestamp", "movie_id"], kind="mergesort")
    positives = ordered.loc[ordered["rating"] >= threshold]
    if len(positives) < keep_positive + 3:
        return None
    cutoff = positives.iloc[keep_positive - 1]
    cutoff_key = (int(cutoff["timestamp"]), int(cutoff["movie_id"]))
    keep = (ordered["timestamp"] < cutoff_key[0]) | (
        (ordered["timestamp"] == cutoff_key[0]) & (ordered["movie_id"] <= cutoff_key[1])
    )
    return ordered.loc[keep]


def evaluate_models(
    config: AppConfig,
    *,
    processed_dir: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    device: str | None = None,
) -> dict:
    """Evaluate saved models once on the test split.

    Alpha is read from the validation selection file and is not retuned here.
    """
    processed, artifacts = _dirs(config, processed_dir, artifacts_dir)
    data = load_processed(processed)
    model_dir = artifacts / "models"
    metrics_dir = ensure_dir(artifacts / "metrics")
    plots_dir = ensure_dir(artifacts / "plots")
    popularity = PopularityModel.load(model_dir / "popularity.json")
    content = ContentModel.load(model_dir)
    mf = MatrixFactorizationRecommender.load(model_dir, device=device or config.matrix_factorization.device)
    alpha_payload = load_json(model_dir / "selected_alpha.json")
    alpha = float(alpha_payload["alpha"])
    run_metadata = load_json(model_dir / "run_metadata.json") if (model_dir / "run_metadata.json").exists() else {}
    result_type = run_metadata.get("result_type", alpha_payload.get("result_type", "unknown"))
    movie_ids = data.movies["movie_id"].to_numpy()
    user_ids = data.test["user_id"].astype(int).tolist()
    targets = dict(zip(data.test["user_id"].astype(int), data.test["movie_id"].astype(int)))
    train_pairs = set(zip(data.train["user_id"].astype(int), data.train["movie_id"].astype(int)))
    leaked = [user_id for user_id, movie_id in targets.items() if (user_id, movie_id) in train_pairs]
    if leaked:
        raise RuntimeError(f"Test movies leaked into training for users {leaked[:5]}")
    prepared = prepare_users(user_ids, data.train, movie_ids, mf, content, popularity)
    genre_by_movie = _genre_lookup(data.movies, data.genres)
    counts = {int(movie_id): float(value) for movie_id, value in popularity.interaction_count.items()}
    k_values = list(config.evaluation.k_values)
    rankings = {
        name: rankings_for_score(prepared, name, alpha=alpha if name == "hybrid" else None)
        for name in MODEL_NAMES
    }
    per_user_frames = []
    summaries = {}
    for name, ranking in rankings.items():
        frame = per_user_metrics(
            ranking,
            targets,
            k_values,
            genre_by_movie=genre_by_movie,
            interaction_count=counts,
            n_users_train=popularity.n_users,
        )
        frame.insert(0, "model", name)
        per_user_frames.append(frame)
        summaries[name] = summarize_metric_frame(
            frame,
            ranking,
            k_values,
            n_movies=len(movie_ids),
            n_resamples=config.evaluation.bootstrap_resamples,
            seed=config.seed,
        )
        logger.info(
            "%s test NDCG@10=%.4f HR@10=%.4f MRR=%.4f",
            name,
            summaries[name]["ndcg@10"]["mean"],
            summaries[name]["hit@10"]["mean"],
            summaries[name]["mrr"]["mean"],
        )
    per_user = pd.concat(per_user_frames, ignore_index=True)
    per_user.to_csv(metrics_dir / "per_user_metrics.csv", index=False)
    comparison = pd.DataFrame([_flat_summary(name, summary) for name, summary in summaries.items()])
    comparison.to_csv(metrics_dir / "model_comparison.csv", index=False)
    rating_metrics = {
        "validation": mf.rating_metrics(data.validation),
        "test": mf.rating_metrics(data.test),
        "note": "RMSE and MAE are secondary explicit-rating metrics. Ranking metrics are primary.",
    }
    buckets = popularity_buckets(popularity.interaction_count)
    bucket_rows = []
    for name in ("popularity_bayesian", "content", "matrix_factorization", "hybrid"):
        frame = per_user.loc[per_user["model"] == name, ["user_id", "target_movie_id", "ndcg@10", "hit@10"]]
        for record in frame.to_dict(orient="records"):
            bucket_rows.append(
                {
                    "model": name,
                    "user_id": int(record["user_id"]),
                    "bucket": buckets[int(record["target_movie_id"])],
                    "ndcg@10": float(record["ndcg@10"]),
                    "hit@10": float(record["hit@10"]),
                }
            )
    bucket_user = pd.DataFrame(bucket_rows)
    bucket_summary = (
        bucket_user.groupby(["model", "bucket"], as_index=False)
        .agg(n_users=("user_id", "size"), **{"ndcg@10": ("ndcg@10", "mean"), "hit@10": ("hit@10", "mean")})
    )
    bucket_summary.to_csv(metrics_dir / "bucket_metrics.csv", index=False)
    sparse = _sparse_user_experiment(
        data.train,
        data.test,
        prepared,
        content,
        config.data.relevance_threshold,
    )
    sparse_frame = pd.DataFrame(sparse["per_user"])
    if not sparse_frame.empty:
        sparse_frame.to_csv(metrics_dir / "sparse_user_metrics.csv", index=False)
    save_json(metrics_dir / "sparse_user_summary.json", sparse["summary"])
    representative = _representative_recommendations(
        data, prepared, alpha, targets, buckets
    )
    save_json(metrics_dir / "representative_recommendations.json", representative)
    plot_frame = pd.DataFrame(
        {
            "model": comparison["model"],
            "HitRate@10": comparison["hit@10"],
            "NDCG@10": comparison["ndcg@10"],
            "MRR": comparison["mrr"],
            "Coverage@10": comparison["coverage@10"],
        }
    )
    plot_model_comparison(plot_frame, plots_dir / "model_comparison.png")
    plot_buckets_frame = bucket_summary.copy()
    plot_buckets_frame["bucket"] = pd.Categorical(
        plot_buckets_frame["bucket"], categories=["tail", "mid", "head"], ordered=True
    )
    plot_buckets(plot_buckets_frame.sort_values(["bucket", "model"]), plots_dir / "head_mid_tail.png")
    save_eda_plots(data.ratings, data.train, plots_dir)
    result = {
        "result_type": result_type,
        "smoke": bool(run_metadata.get("smoke", result_type == "smoke")),
        "seed": config.seed,
        "selected_alpha": alpha,
        "alpha_selection": alpha_payload,
        "test_metrics": summaries,
        "rating_metrics": rating_metrics,
        "bucket_metrics": bucket_summary.to_dict(orient="records"),
        "sparse_user": sparse["summary"],
        "content_fallback_users": int(sum(user.content_used_fallback for user in prepared)),
        "caveat": (
            "Offline proxy metrics on a chronological MovieLens split. "
            "Smoke results are not the full training schedule. "
            "Do not read these numbers as user satisfaction or as a result about a live LLM."
        ),
    }
    save_json(metrics_dir / "model_comparison.json", result)
    _write_results_markdown(metrics_dir / "results_table.md", result, comparison)
    logger.info("Wrote test metrics for alpha %.2f (%s run)", alpha, result_type)
    return result


def _sparse_user_experiment(train, test, prepared, content: ContentModel, threshold: float) -> dict:
    """Re-score content profiles after keeping only an early slice of training history.

    The candidate pool stays the full-history unseen set, so the comparison
    isolates the shorter genre profile. Matrix factorization is not re-scored:
    its saved user embedding was fit on the full training history.
    """
    histories = {int(user_id): frame for user_id, frame in train.groupby("user_id")}
    targets = dict(zip(test["user_id"].astype(int), test["movie_id"].astype(int)))
    rows = []
    for user in prepared:
        truncated = _truncate_history(histories[user.user_id], threshold)
        if truncated is None:
            continue
        profile = content.build_profile(truncated)
        if content.profile_is_empty(profile):
            scores = np.zeros(len(user.movie_ids), dtype=float)
        else:
            scores = content.score_movies(profile, user.movie_ids)
        order = np.lexsort((user.movie_ids, -np.nan_to_num(scores, nan=-1e18)))
        ranked = user.movie_ids[order]
        full_rank = rank_of(
            user.movie_ids[np.lexsort((user.movie_ids, -np.nan_to_num(user.content_raw, nan=-1e18)))],
            targets[user.user_id],
        )
        short_rank = rank_of(ranked, targets[user.user_id])
        rows.append(
            {
                "user_id": user.user_id,
                "ndcg@10_full_history": ndcg_at_k(full_rank, 10),
                "ndcg@10_truncated_history": ndcg_at_k(short_rank, 10),
            }
        )
    if not rows:
        summary = {"n_users": 0, "note": "No user had enough positive history to truncate."}
    else:
        frame = pd.DataFrame(rows)
        summary = {
            "n_users": int(len(frame)),
            "mean_ndcg@10_full_history": float(frame["ndcg@10_full_history"].mean()),
            "mean_ndcg@10_truncated_history": float(frame["ndcg@10_truncated_history"].mean()),
            "mean_difference_truncated_minus_full": float(
                (frame["ndcg@10_truncated_history"] - frame["ndcg@10_full_history"]).mean()
            ),
            "note": (
                "Secondary sparsity check for the content model. "
                "Metadata can help when a short history still has genre signal, "
                "but it cannot recommend for a user with no preference history. "
                "Matrix factorization was not retrained on the truncated histories."
            ),
        }
    return {"per_user": rows, "summary": summary}


def _representative_recommendations(data, prepared, alpha: float, targets: dict[int, int], buckets: dict[int, str]) -> dict:
    from movie_recommender.ranking import hybrid_component_frame

    counts = data.train.groupby("user_id").size()
    eligible = [user.user_id for user in prepared]
    ordered = counts.loc[eligible].sort_values(kind="mergesort")
    chosen = {
        "most_training_ratings": int(ordered.index[-1]),
        "median_training_ratings": int(ordered.index[len(ordered) // 2]),
        "fewest_training_ratings": int(ordered.index[0]),
    }
    by_bucket: dict[str, int] = {}
    for user_id in eligible:
        bucket = buckets.get(int(targets[user_id]))
        if bucket and bucket not in by_bucket:
            by_bucket[bucket] = user_id
    for bucket, user_id in by_bucket.items():
        chosen[f"test_item_{bucket}"] = int(user_id)
    lookup = {
        int(row.movie_id): {"title": row.title, "year": None if pd.isna(row.year) else int(row.year), "genres": row.genres}
        for row in data.movies.itertuples(index=False)
    }
    prepared_by_id = {user.user_id: user for user in prepared}
    examples = []
    for label, user_id in chosen.items():
        frame = hybrid_component_frame(prepared_by_id[user_id], alpha).head(10)
        recommendations = []
        for row in frame.itertuples(index=False):
            info = lookup[int(row.movie_id)]
            recommendations.append(
                {
                    "movie_id": int(row.movie_id),
                    "title": info["title"],
                    "year": info["year"],
                    "genres": info["genres"],
                    "mf_score": float(row.mf_score),
                    "content_score": float(row.content_score),
                    "hybrid_score": float(row.hybrid_score),
                }
            )
        examples.append(
            {
                "label": label,
                "user_id": user_id,
                "n_training_ratings": int(counts.loc[user_id]),
                "held_out_test_movie_id": int(targets[user_id]),
                "held_out_bucket": buckets.get(int(targets[user_id])),
                "recommendations": recommendations,
                "analysis_note": "The held-out id is an evaluation label and must not be sent to a model or prompt.",
            }
        )
    return {"alpha": alpha, "users": examples}


def _write_results_markdown(path: Path, result: dict, comparison: pd.DataFrame) -> None:
    lines = [
        f"# Results ({result['result_type']})",
        "",
        result["caveat"],
        "",
        f"Selected alpha: `{result['selected_alpha']}`",
        "",
        "| Model | HR@10 | NDCG@10 | MRR | Coverage@10 | Diversity@10 | Novelty@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in comparison.to_dict(orient="records"):
        lines.append(
            "| {model} | {hit:.4f} | {ndcg:.4f} | {mrr:.4f} | {coverage:.4f} | {diversity:.4f} | {novelty:.4f} |".format(
                model=record["model"],
                hit=record["hit@10"],
                ndcg=record["ndcg@10"],
                mrr=record["mrr"],
                coverage=record["coverage@10"],
                diversity=record["diversity@10"],
                novelty=record["novelty@10"],
            )
        )
    lines.extend(
        [
            "",
            "## Explicit-rating metrics",
            "",
            f"- Validation RMSE: {result['rating_metrics']['validation']['rmse']}",
            f"- Validation MAE: {result['rating_metrics']['validation']['mae']}",
            f"- Test RMSE: {result['rating_metrics']['test']['rmse']}",
            f"- Test MAE: {result['rating_metrics']['test']['mae']}",
            "",
            "RMSE and MAE are secondary. The research comparison uses the ranking metrics above.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
