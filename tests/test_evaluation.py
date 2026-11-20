"""Ranking metrics and bootstrap intervals."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from movie_recommender.candidates import sample_stratified_user_ids, sample_user_ids
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


def test_sample_user_ids_behavior_unchanged():
    users = [10, 5, 20, 15, 30, 25]
    assert np.array_equal(sample_user_ids(users, n=10, seed=42), np.array([5, 10, 15, 20, 25, 30]))
    sampled = sample_user_ids(users, n=3, seed=42)
    assert len(sampled) == 3
    assert np.array_equal(sampled, np.sort(sampled))
    sampled2 = sample_user_ids(users, n=3, seed=42)
    assert np.array_equal(sampled, sampled2)


def test_sample_stratified_user_ids_deterministic_and_tier_balance():
    train_rows = []
    for u in range(1, 151):
        for m in range(u):
            train_rows.append({"user_id": u, "movie_id": m + 1})
    train_df = pd.DataFrame(train_rows)
    eligible = list(range(1, 151))

    sampled1, tiers1 = sample_stratified_user_ids(train_df, eligible, n=100, seed=42)
    sampled2, tiers2 = sample_stratified_user_ids(train_df, eligible, n=100, seed=42)

    assert np.array_equal(sampled1, sampled2)
    assert tiers1 == tiers2
    assert len(sampled1) == 100
    assert np.array_equal(sampled1, np.sort(sampled1))
    assert len(tiers1) == 150
    assert set(tiers1.values()) == {"low", "medium", "high"}

    tier_counts = {
        tier: sum(1 for u in sampled1 if tiers1[int(u)] == tier)
        for tier in ("low", "medium", "high")
    }
    assert tier_counts["low"] == 34
    assert tier_counts["medium"] == 33
    assert tier_counts["high"] == 33

    for u in range(1, 51):
        assert tiers1[u] == "low"
    for u in range(51, 101):
        assert tiers1[u] == "medium"
    for u in range(101, 151):
        assert tiers1[u] == "high"


def test_sample_stratified_user_ids_edge_cases():
    train_df = pd.DataFrame([{"user_id": 1, "movie_id": 10}, {"user_id": 2, "movie_id": 20}])
    eligible = [1, 2]

    sampled, tiers = sample_stratified_user_ids(train_df, eligible, n=5, seed=42)
    assert np.array_equal(sampled, np.array([1, 2]))
    assert len(tiers) == 2

    sampled_exact, _ = sample_stratified_user_ids(train_df, eligible, n=2, seed=42)
    assert np.array_equal(sampled_exact, np.array([1, 2]))

    sampled_zero, tiers_zero = sample_stratified_user_ids(train_df, eligible, n=0, seed=42)
    assert len(sampled_zero) == 0
    assert len(tiers_zero) == 2

    sampled_empty, tiers_empty = sample_stratified_user_ids(train_df, [], n=10, seed=42)
    assert len(sampled_empty) == 0
    assert tiers_empty == {}
