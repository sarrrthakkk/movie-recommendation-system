"""Genre-vector content model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from movie_recommender.data import EXPECTED_GENRE_DIM, GENRE_NAMES
from movie_recommender.models.hybrid import rank_order
from movie_recommender.models.popularity import PopularityModel

# Ratings at or below this value contribute no weight. A rating of 4 contributes 1
# and a rating of 5 contributes 2, matching w_ui = max(r_ui - 3, 0).
WEIGHT_OFFSET = 3.0


class ContentModel:
    """Score movies by cosine similarity to a weighted genre profile.

    The profile is the weighted average of genre vectors for training movies
    with positive weight. If that profile is all zeros, callers should fall
    back to popularity. Previously observed movies are excluded by the caller.
    """

    def __init__(self) -> None:
        self.movie_ids = np.asarray([], dtype=int)
        self.movie_to_row: dict[int, int] = {}
        self.genres = np.zeros((0, EXPECTED_GENRE_DIM), dtype=float)
        self.genre_names = list(GENRE_NAMES)

    def fit(self, movie_ids: np.ndarray, genres: np.ndarray) -> "ContentModel":
        ids = np.asarray(movie_ids, dtype=int)
        matrix = np.asarray(genres, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != EXPECTED_GENRE_DIM:
            raise ValueError(f"Expected genre matrix of width {EXPECTED_GENRE_DIM}")
        if len(ids) != len(matrix):
            raise ValueError("movie ids and genre matrix have different lengths")
        self.movie_ids = ids
        self.genres = matrix
        self.movie_to_row = {int(movie_id): row for row, movie_id in enumerate(ids)}
        return self

    def build_profile(self, history: pd.DataFrame) -> np.ndarray:
        """Weighted average of positively weighted training genre vectors."""
        if history.empty:
            return np.zeros(EXPECTED_GENRE_DIM, dtype=float)
        weights = np.maximum(history["rating"].to_numpy(dtype=float) - WEIGHT_OFFSET, 0.0)
        mask = weights > 0
        if not np.any(mask):
            return np.zeros(EXPECTED_GENRE_DIM, dtype=float)
        rows = [self.movie_to_row[int(movie_id)] for movie_id in history.loc[mask, "movie_id"]]
        vectors = self.genres[rows]
        used_weights = weights[mask]
        return (vectors * used_weights[:, None]).sum(axis=0) / used_weights.sum()

    @staticmethod
    def profile_is_empty(profile: np.ndarray) -> bool:
        return float(np.linalg.norm(profile)) < 1e-12

    def score_movies(self, profile: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        """Cosine similarity between one profile and the requested movies."""
        ids = np.asarray(movie_ids, dtype=int)
        if ids.size == 0:
            return np.asarray([], dtype=float)
        if self.profile_is_empty(profile):
            return np.zeros(ids.shape, dtype=float)
        rows = [self.movie_to_row[int(movie_id)] for movie_id in ids]
        vectors = self.genres[rows]
        profile_norm = float(np.linalg.norm(profile))
        vector_norms = np.linalg.norm(vectors, axis=1)
        dots = vectors @ np.asarray(profile, dtype=float)
        scores = np.zeros(len(ids), dtype=float)
        valid = vector_norms > 1e-12
        scores[valid] = dots[valid] / (vector_norms[valid] * profile_norm)
        return scores

    def recommend(
        self,
        history: pd.DataFrame,
        seen_movie_ids: set[int] | list[int],
        k: int,
        popularity: PopularityModel | None = None,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Return movie ids, scores, and whether popularity replaced an empty profile."""
        seen = {int(movie_id) for movie_id in seen_movie_ids}
        candidates = np.asarray([movie_id for movie_id in self.movie_ids if int(movie_id) not in seen], dtype=int)
        profile = self.build_profile(history)
        used_fallback = False
        if self.profile_is_empty(profile):
            used_fallback = True
            if popularity is None:
                scores = np.zeros(len(candidates), dtype=float)
            else:
                scores = popularity.scores(candidates, mode="bayesian")
        else:
            scores = self.score_movies(profile, candidates)
        if len(candidates) == 0 or k < 1:
            return np.asarray([], dtype=int), np.asarray([], dtype=float), used_fallback
        order = rank_order(candidates, scores)[:k]
        return candidates[order], scores[order], used_fallback

    def preferred_genres(self, profile: np.ndarray, limit: int = 5) -> list[tuple[str, float]]:
        if self.profile_is_empty(profile):
            return []
        total = float(profile.sum())
        shares = profile / total if total > 0 else profile
        order = np.argsort(-shares)
        named: list[tuple[str, float]] = []
        for row in order:
            if shares[row] <= 0 or len(named) >= limit:
                break
            named.append((self.genre_names[int(row)], float(shares[int(row)])))
        return named

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "content_movie_ids.npy", self.movie_ids)
        np.save(path / "content_genres.npy", self.genres)

    @classmethod
    def load(cls, directory: str | Path) -> "ContentModel":
        path = Path(directory)
        model = cls()
        movie_ids = np.load(path / "content_movie_ids.npy")
        genres = np.load(path / "content_genres.npy")
        return model.fit(movie_ids, genres)
