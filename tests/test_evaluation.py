"""Ranking metrics and bootstrap intervals."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from movie_recommender.evaluation import (
    bootstrap_ci,
    catalog_coverage,
    hit_at_k,
    ndcg_at_k,
    paired_difference_ci,
    per_user_metrics,
    popularity_buckets,
    rank_of,
    recall_at_k,
    reciprocal_rank,
)


def test_single_relevant_metrics():
    ranked = np.array([10, 11, 12, 13, 14])
    assert rank_of(ranked, 11) == 2
    assert hit_at_k(2, 1) == 0
    assert hit_at_k(2, 5) == 1
    assert recall_at_k(2, 5) == 1
    assert reciprocal_rank(2) == 0.5
    assert ndcg_at_k(1, 5) == 1
    assert np.isclose(ndcg_at_k(2, 5), 1 / math.log2(3))
    assert ndcg_at_k(None, 5) == 0
    assert reciprocal_rank(None) == 0


def test_coverage_and_buckets():
    rankings = {1: np.array([1, 2, 3]), 2: np.array([2, 4, 5])}
    assert catalog_coverage(rankings, k=2, n_movies=10) == 0.3
    counts = pd.Series([0, 1, 5, 9], index=pd.Index([10, 11, 12, 13], name="movie_id"))
    buckets = popularity_buckets(counts)
    assert set(buckets.values()) == {"tail", "mid", "head"}
    assert buckets[10] == "tail"
    assert buckets[13] == "head"


def test_bootstrap_constant_and_paired_interval():
    interval = bootstrap_ci([1, 1, 1, 1], n_resamples=50, seed=1)
    assert interval["mean"] == 1
    assert interval["low"] == 1
    assert interval["high"] == 1
    paired = paired_difference_ci([0.2, 0.2, 0.2], [0.2, 0.2, 0.2], n_resamples=30, seed=1)
    assert paired["mean"] == 0
    assert "p-value" not in paired["reading"]
    assert paired["n"] == 3


def test_per_user_frame_contains_requested_cutoffs():
    rankings = {1: np.array([5, 6, 7, 8, 9, 10])}
    genres = {movie_id: np.array([1.0, 0.0]) for movie_id in range(5, 11)}
    frame = per_user_metrics(
        rankings,
        {1: 6},
        [5, 10],
        genre_by_movie=genres,
        interaction_count={movie_id: 2.0 for movie_id in range(5, 11)},
        n_users_train=10,
    )
    assert frame.loc[0, "hit@5"] == 1
    assert frame.loc[0, "mrr"] == 0.5
    assert frame.loc[0, "novelty@5"] > 0
