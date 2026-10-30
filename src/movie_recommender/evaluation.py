"""Top-K ranking metrics, bootstrap intervals, and popularity buckets."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


def rank_of(ranked_movie_ids: np.ndarray, target: int) -> int | None:
    """Return a 1-based rank, or None when the target is absent."""
    matches = np.flatnonzero(np.asarray(ranked_movie_ids) == int(target))
    if len(matches) == 0:
        return None
    return int(matches[0]) + 1


def hit_at_k(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def recall_at_k(rank: int | None, k: int) -> float:
    """Leave-one-out recall. With one relevant item this equals hit-rate@K."""
    return hit_at_k(rank, k)


def ndcg_at_k(rank: int | None, k: int) -> float:
    """Single-relevant NDCG. Ideal DCG is 1, so this is ``1 / log2(rank + 1)``."""
    if rank is None or rank > k or rank < 1:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def reciprocal_rank(rank: int | None) -> float:
    if rank is None or rank < 1:
        return 0.0
    return 1.0 / rank


def genre_diversity(vectors: np.ndarray) -> float:
    """Mean pairwise cosine distance among the recommended genre vectors."""
    if len(vectors) < 2:
        return 0.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.where(norms < 1e-12, 1.0, norms)
    normalized = np.where(norms < 1e-12, 0.0, vectors / safe)
    similarity = normalized @ normalized.T
    upper = np.triu_indices(len(vectors), k=1)
    return float(np.mean(1.0 - similarity[upper]))


def novelty_of_counts(counts: np.ndarray, n_users: int) -> np.ndarray:
    """Popularity-based novelty ``-log2(p_i)``.

    ``p_i = n_i / N`` where ``n_i`` is the number of training users who rated
    the movie and ``N`` is the number of training users. Movies with no
    training ratings use ``p_i = 1 / (N + 1)`` so the logarithm stays defined.
    """
    n_users = max(int(n_users), 1)
    probability = np.asarray(counts, dtype=float) / n_users
    probability = np.where(probability > 0, probability, 1.0 / (n_users + 1))
    return -np.log2(probability)


def catalog_coverage(rankings: dict[int, np.ndarray], k: int, n_movies: int) -> float:
    if n_movies <= 0:
        return 0.0
    unique: set[int] = set()
    for ranked in rankings.values():
        unique.update(int(movie_id) for movie_id in np.asarray(ranked)[:k])
    return len(unique) / n_movies


def bootstrap_ci(values: Sequence[float] | np.ndarray, n_resamples: int, seed: int) -> dict[str, float]:
    """Percentile bootstrap confidence interval for a user-level mean.

    This is a resampling interval. It is not a p-value.
    """
    array = np.asarray(list(values), dtype=float)
    n = len(array)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "low": float("nan"), "high": float("nan")}
    rng = np.random.default_rng(seed)
    sample_index = rng.integers(0, n, size=(n_resamples, n))
    means = array[sample_index].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"n": int(n), "mean": float(array.mean()), "low": float(low), "high": float(high)}


def paired_difference_ci(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
    n_resamples: int,
    seed: int,
) -> dict[str, float | bool | str]:
    """Bootstrap the mean of paired user-level differences ``left - right``.

    A practical margin of 0.01 NDCG (or the same units as the input) is reported
    only as a descriptive benchmark. It is not a hypothesis test.
    """
    difference = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    interval = bootstrap_ci(difference, n_resamples=n_resamples, seed=seed)
    low = float(interval["low"])
    high = float(interval["high"])
    excludes_zero = bool(low > 0 or high < 0) if interval["n"] else False
    margin = 0.01
    if interval["n"] < 30:
        reading = "The sample has fewer than 30 users, so the interval is too small for a reliable conclusion."
    elif excludes_zero and (low > margin or high < -margin):
        reading = (
            "The 95% bootstrap interval excludes zero and also excludes differences smaller "
            "than 0.01 in absolute value. That is a descriptive statement about this sample, "
            "not a p-value and not proof of practical importance outside this dataset."
        )
    elif excludes_zero:
        reading = (
            "The 95% bootstrap interval excludes zero, but it still includes differences smaller "
            "than 0.01. Any average change is small relative to the descriptive 0.01 margin."
        )
    else:
        reading = (
            "The 95% bootstrap interval includes zero. This sample does not show a stable "
            "average difference. No p-value is reported."
        )
    return {
        **interval,
        "excludes_zero": excludes_zero,
        "practical_margin": margin,
        "reading": reading,
    }


def popularity_buckets(interaction_count: pd.Series) -> dict[int, str]:
    """Split the catalog into equal-sized tail, mid, and head groups.

    Movies are ordered by training interaction count, with movie id as a
    tie-break, then cut into three nearly equal groups. The lowest group is
    the sparse tail and the highest group is the popular head.
    """
    frame = interaction_count.rename("count").reset_index()
    if "movie_id" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "movie_id"})
    frame = frame.sort_values(["count", "movie_id"], kind="mergesort").reset_index(drop=True)
    n = len(frame)
    labels = np.empty(n, dtype=object)
    first = n // 3
    second = 2 * n // 3
    labels[:first] = "tail"
    labels[first:second] = "mid"
    labels[second:] = "head"
    return {int(movie_id): str(label) for movie_id, label in zip(frame["movie_id"], labels)}


def per_user_metrics(
    rankings: dict[int, np.ndarray],
    targets: dict[int, int],
    k_values: list[int],
    *,
    genre_by_movie: dict[int, np.ndarray],
    interaction_count: dict[int, float],
    n_users_train: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for user_id, ranked in rankings.items():
        target = int(targets[int(user_id)])
        rank = rank_of(ranked, target)
        row: dict = {"user_id": int(user_id), "target_movie_id": target, "rank": rank if rank is not None else np.nan}
        row["mrr"] = reciprocal_rank(rank)
        for k in k_values:
            row[f"hit@{k}"] = hit_at_k(rank, k)
            row[f"recall@{k}"] = recall_at_k(rank, k)
            row[f"ndcg@{k}"] = ndcg_at_k(rank, k)
            top = [int(movie_id) for movie_id in np.asarray(ranked)[:k]]
            vectors = np.vstack([genre_by_movie[movie_id] for movie_id in top]) if top else np.zeros((0, 1))
            counts = np.asarray([interaction_count.get(movie_id, 0.0) for movie_id in top], dtype=float)
            row[f"diversity@{k}"] = genre_diversity(vectors) if top else 0.0
            row[f"popularity@{k}"] = float(counts.mean()) if len(counts) else 0.0
            row[f"novelty@{k}"] = float(novelty_of_counts(counts, n_users_train).mean()) if len(counts) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_metric_frame(
    frame: pd.DataFrame,
    rankings: dict[int, np.ndarray],
    k_values: list[int],
    *,
    n_movies: int,
    n_resamples: int,
    seed: int,
) -> dict:
    summary: dict = {"n_users": int(len(frame))}
    metric_names = ["mrr"]
    for k in k_values:
        metric_names.extend(
            [f"hit@{k}", f"recall@{k}", f"ndcg@{k}", f"diversity@{k}", f"popularity@{k}", f"novelty@{k}"]
        )
    for name in metric_names:
        summary[name] = bootstrap_ci(frame[name].to_numpy(), n_resamples=n_resamples, seed=seed)
    for k in k_values:
        summary[f"coverage@{k}"] = catalog_coverage(rankings, k, n_movies)
    return summary
