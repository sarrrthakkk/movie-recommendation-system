"""Per-user component scores used by hybrid tuning and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from movie_recommender.models.content_based import ContentModel
from movie_recommender.models.hybrid import combine_scores, minmax_normalize, rank_order
from movie_recommender.models.matrix_factorization import MatrixFactorizationRecommender
from movie_recommender.models.popularity import PopularityModel
from movie_recommender.utils import get_logger

logger = get_logger(__name__)


@dataclass
class PreparedUser:
    user_id: int
    movie_ids: np.ndarray
    mf_raw: np.ndarray
    content_raw: np.ndarray
    popularity_raw: np.ndarray
    positive_count: np.ndarray
    mf_norm: np.ndarray
    content_norm: np.ndarray
    content_used_fallback: bool


def prepare_users(
    user_ids: list[int] | np.ndarray,
    train: pd.DataFrame,
    catalog_ids: np.ndarray,
    mf: MatrixFactorizationRecommender,
    content: ContentModel,
    popularity: PopularityModel,
) -> list[PreparedUser]:
    """Score every unseen catalog movie once per user.

    Content scores fall back to the Bayesian popularity score when the user has
    an empty genre profile. Matrix-factorization scores are raw predicted ratings.
    Normalization is per user and is used only by the hybrid combination.
    """
    histories = {int(user_id): frame for user_id, frame in train.groupby("user_id")}
    catalog = np.asarray(catalog_ids, dtype=int)
    prepared: list[PreparedUser] = []
    total = len(list(user_ids))
    logger.info("Scoring %d users against the unseen catalog", total)
    for index, user_id in enumerate(user_ids, start=1):
        if index == 1 or index % 200 == 0 or index == total:
            logger.info("Scoring user %d/%d", index, total)
        user_id = int(user_id)
        history = histories.get(user_id)
        if history is None:
            raise KeyError(f"User {user_id} has no training history")
        seen = set(history["movie_id"].astype(int))
        unseen = catalog[~np.isin(catalog, list(seen))]
        profile = content.build_profile(history)
        used_fallback = content.profile_is_empty(profile)
        content_raw = (
            popularity.scores(unseen, mode="bayesian")
            if used_fallback
            else content.score_movies(profile, unseen)
        )
        mf_raw = mf.score_movies(user_id, unseen)
        prepared.append(
            PreparedUser(
                user_id=user_id,
                movie_ids=unseen,
                mf_raw=mf_raw,
                content_raw=content_raw,
                popularity_raw=popularity.scores(unseen, mode="bayesian"),
                positive_count=popularity.scores(unseen, mode="positive_count"),
                mf_norm=minmax_normalize(mf_raw),
                content_norm=minmax_normalize(content_raw),
                content_used_fallback=used_fallback,
            )
        )
    return prepared


def ranked_ids(movie_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    order = rank_order(movie_ids, scores)
    return np.asarray(movie_ids, dtype=int)[order]


def rankings_for_score(prepared: list[PreparedUser], score_name: str, alpha: float | None = None) -> dict[int, np.ndarray]:
    rankings: dict[int, np.ndarray] = {}
    for user in prepared:
        if score_name == "popularity_bayesian":
            scores = user.popularity_raw
        elif score_name == "popularity_positive_count":
            scores = user.positive_count
        elif score_name == "matrix_factorization":
            scores = user.mf_raw
        elif score_name == "content":
            scores = user.content_raw
        elif score_name == "hybrid":
            if alpha is None:
                raise ValueError("hybrid ranking requires alpha")
            scores = combine_scores(user.mf_norm, user.content_norm, alpha)
        else:
            raise ValueError(f"Unknown score name: {score_name}")
        rankings[user.user_id] = ranked_ids(user.movie_ids, scores)
    return rankings


def hybrid_component_frame(user: PreparedUser, alpha: float) -> pd.DataFrame:
    hybrid = combine_scores(user.mf_norm, user.content_norm, alpha)
    order = rank_order(user.movie_ids, hybrid)
    return pd.DataFrame(
        {
            "movie_id": user.movie_ids[order],
            "mf_score": user.mf_raw[order],
            "content_score": user.content_raw[order],
            "mf_score_normalized": user.mf_norm[order],
            "content_score_normalized": user.content_norm[order],
            "hybrid_score": hybrid[order],
            "popularity_score": user.popularity_raw[order],
            "positive_count": user.positive_count[order],
        }
    )
