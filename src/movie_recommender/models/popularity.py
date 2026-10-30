"""Training-only popularity baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from movie_recommender.models.hybrid import rank_order
from movie_recommender.utils import load_json, save_json


class PopularityModel:
    """Rank unseen movies by a Bayesian average or by positive-rating count.

    The Bayesian average uses ``prior_strength`` pseudo-ratings at the training
    global mean:

        bayes_i = (C * mu + sum of observed ratings) / (C + n_i)

    Movies with no training ratings receive the global mean under the Bayesian
    score and zero under the positive-count score. Ties break toward the
    smaller movie id.
    """

    def __init__(self, prior_strength: int = 10) -> None:
        if prior_strength < 1:
            raise ValueError("prior_strength must be at least 1")
        self.prior_strength = int(prior_strength)
        self.global_mean = 0.0
        self.relevance_threshold = 4.0
        self.n_users = 0
        self.movie_ids = np.asarray([], dtype=int)
        self.interaction_count = pd.Series(dtype=float)
        self.positive_count = pd.Series(dtype=float)
        self.rating_sum = pd.Series(dtype=float)
        self.bayesian_average = pd.Series(dtype=float)

    def fit(
        self,
        train: pd.DataFrame,
        movie_ids: np.ndarray,
        *,
        relevance_threshold: float = 4.0,
    ) -> "PopularityModel":
        self.relevance_threshold = float(relevance_threshold)
        self.movie_ids = np.asarray(sorted(int(movie_id) for movie_id in movie_ids), dtype=int)
        self.n_users = int(train["user_id"].nunique()) if not train.empty else 0
        self.global_mean = float(train["rating"].mean()) if not train.empty else 0.0
        grouped = train.groupby("movie_id")["rating"]
        counts = grouped.size().astype(float)
        sums = grouped.sum().astype(float)
        positives = (
            train.loc[train["rating"] >= relevance_threshold]
            .groupby("movie_id")
            .size()
            .astype(float)
        )
        index = pd.Index(self.movie_ids, name="movie_id")
        self.interaction_count = counts.reindex(index, fill_value=0.0)
        self.rating_sum = sums.reindex(index, fill_value=0.0)
        self.positive_count = positives.reindex(index, fill_value=0.0)
        strength = float(self.prior_strength)
        self.bayesian_average = (strength * self.global_mean + self.rating_sum) / (
            strength + self.interaction_count
        )
        return self

    def scores(self, movie_ids: np.ndarray, mode: str = "bayesian") -> np.ndarray:
        ids = pd.Index(np.asarray(movie_ids, dtype=int))
        if mode == "bayesian":
            return self.bayesian_average.reindex(ids).fillna(self.global_mean).to_numpy(dtype=float)
        if mode == "positive_count":
            return self.positive_count.reindex(ids).fillna(0.0).to_numpy(dtype=float)
        raise ValueError("mode must be 'bayesian' or 'positive_count'")

    def recommend(
        self,
        seen_movie_ids: set[int] | list[int],
        k: int,
        mode: str = "bayesian",
    ) -> tuple[np.ndarray, np.ndarray]:
        seen = {int(movie_id) for movie_id in seen_movie_ids}
        candidates = np.asarray([movie_id for movie_id in self.movie_ids if movie_id not in seen], dtype=int)
        if len(candidates) == 0 or k < 1:
            return np.asarray([], dtype=int), np.asarray([], dtype=float)
        values = self.scores(candidates, mode=mode)
        order = rank_order(candidates, values)
        chosen = order[:k]
        return candidates[chosen], values[chosen]

    def interaction_count_for(self, movie_ids: np.ndarray) -> np.ndarray:
        ids = pd.Index(np.asarray(movie_ids, dtype=int))
        return self.interaction_count.reindex(ids).fillna(0.0).to_numpy(dtype=float)

    def save(self, path: str | Path) -> None:
        payload = {
            "prior_strength": self.prior_strength,
            "global_mean": self.global_mean,
            "relevance_threshold": self.relevance_threshold,
            "n_users": self.n_users,
            "movie_ids": self.movie_ids.astype(int).tolist(),
            "interaction_count": self.interaction_count.astype(float).tolist(),
            "positive_count": self.positive_count.astype(float).tolist(),
            "rating_sum": self.rating_sum.astype(float).tolist(),
            "bayesian_average": self.bayesian_average.astype(float).tolist(),
        }
        save_json(path, payload)

    @classmethod
    def load(cls, path: str | Path) -> "PopularityModel":
        payload = load_json(path)
        model = cls(prior_strength=int(payload["prior_strength"]))
        model.global_mean = float(payload["global_mean"])
        model.relevance_threshold = float(payload["relevance_threshold"])
        model.n_users = int(payload["n_users"])
        model.movie_ids = np.asarray(payload["movie_ids"], dtype=int)
        index = pd.Index(model.movie_ids, name="movie_id")
        model.interaction_count = pd.Series(payload["interaction_count"], index=index, dtype=float)
        model.positive_count = pd.Series(payload["positive_count"], index=index, dtype=float)
        model.rating_sum = pd.Series(payload["rating_sum"], index=index, dtype=float)
        model.bayesian_average = pd.Series(payload["bayesian_average"], index=index, dtype=float)
        return model
