"""Candidate records and compact training-only user profiles for prompts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from movie_recommender.models.content_based import ContentModel


def sample_user_ids(user_ids: np.ndarray | list[int], n: int, seed: int) -> np.ndarray:
    """Deterministically sample and then sort user ids."""
    ids = np.array(sorted(int(user_id) for user_id in user_ids), dtype=int)
    if n >= len(ids):
        return ids
    rng = np.random.default_rng(seed)
    chosen = rng.choice(ids, size=n, replace=False)
    return np.sort(chosen.astype(int))


def movie_lookup(movies: pd.DataFrame) -> dict[int, dict]:
    lookup: dict[int, dict] = {}
    for row in movies.itertuples(index=False):
        year = None if pd.isna(row.year) else int(row.year)
        lookup[int(row.movie_id)] = {
            "movie_id": int(row.movie_id),
            "title": str(row.title),
            "year": year,
            "genres": str(row.genres),
        }
    return lookup


def build_candidate_records(
    ranked_frame: pd.DataFrame,
    movies: pd.DataFrame,
    popularity_counts: pd.Series,
    limit: int,
) -> list[dict]:
    """Attach metadata to the hybrid ranking. Scores stay here and out of the prompt."""
    lookup = movie_lookup(movies)
    records: list[dict] = []
    for row in ranked_frame.head(limit).itertuples(index=False):
        movie_id = int(row.movie_id)
        info = lookup[movie_id]
        records.append(
            {
                **info,
                "mf_score": float(row.mf_score),
                "content_score": float(row.content_score),
                "mf_score_normalized": float(row.mf_score_normalized),
                "content_score_normalized": float(row.content_score_normalized),
                "hybrid_score": float(row.hybrid_score),
                "popularity_count": float(popularity_counts.get(movie_id, 0.0)),
                "popularity_positive_count": float(row.positive_count),
                "bayesian_average": float(row.popularity_score),
            }
        )
    return records


def _history_rows(
    history: pd.DataFrame,
    lookup: dict[int, dict],
    *,
    ascending_rating: bool,
    limit: int,
    positive_only: bool = False,
    relevance_threshold: float = 4.0,
) -> list[dict]:
    frame = history
    if positive_only:
        frame = frame.loc[frame["rating"] >= relevance_threshold]
    if frame.empty or limit < 1:
        return []
    ordered = frame.sort_values(
        ["rating", "timestamp", "movie_id"],
        ascending=[ascending_rating, False, True],
        kind="mergesort",
    ).head(limit)
    rows: list[dict] = []
    for row in ordered.itertuples(index=False):
        info = lookup[int(row.movie_id)]
        rows.append({**info, "rating": float(row.rating)})
    return rows


def build_prompt_profile(
    history: pd.DataFrame,
    movies: pd.DataFrame,
    content: ContentModel,
    *,
    maximum_positive_history: int = 10,
    maximum_negative_history: int = 5,
    maximum_recent_positive: int = 5,
    history_mode: str = "combined",
    relevance_threshold: float = 4.0,
) -> dict:
    """Compact training-only profile. Held-out rows must not be in ``history``."""
    if history_mode not in {"combined", "highest_rated", "recent_positive"}:
        raise ValueError("history_mode must be combined, highest_rated, or recent_positive")
    history = history.copy()
    if "timestamp" not in history.columns:
        history["timestamp"] = np.arange(len(history))
    lookup = movie_lookup(movies)
    profile = content.build_profile(history)
    preferred = [
        {"genre": name, "share": share}
        for name, share in content.preferred_genres(profile, limit=5)
    ]
    highest = _history_rows(
        history,
        lookup,
        ascending_rating=False,
        limit=maximum_positive_history,
    )
    lowest = _history_rows(
        history,
        lookup,
        ascending_rating=True,
        limit=maximum_negative_history,
    )
    recent_frame = history.sort_values(["timestamp", "movie_id"], ascending=[False, True], kind="mergesort")
    recent_positive = recent_frame.loc[recent_frame["rating"] >= relevance_threshold].head(
        maximum_recent_positive
    )
    recent = []
    for row in recent_positive.itertuples(index=False):
        info = lookup[int(row.movie_id)]
        recent.append({**info, "rating": float(row.rating)})
    highest_ids = {int(row["movie_id"]) for row in highest}
    lowest = [row for row in lowest if int(row["movie_id"]) not in highest_ids]
    if history_mode == "highest_rated":
        lowest = []
        recent = []
    elif history_mode == "recent_positive":
        highest = []
        lowest = []
    return {
        "history_mode": history_mode,
        "preferred_genres": preferred,
        "highest_rated": highest,
        "lowest_rated": lowest,
        "recent_positive": recent,
        "profile_empty": content.profile_is_empty(profile),
    }
