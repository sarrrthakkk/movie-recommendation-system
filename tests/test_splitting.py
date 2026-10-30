"""Chronological split and leakage tests."""

from __future__ import annotations

import pandas as pd

from movie_recommender.splitting import ChronologicalSplit, leakage_violations, make_chronological_split
from tests.helpers import synthetic_catalog


def _ratings(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])


def test_holds_out_latest_relevant_items_and_drops_the_future():
    ratings = _ratings(
        [
            (1, 1, 5, 1),
            (1, 2, 5, 2),
            (1, 3, 5, 3),
            (1, 4, 1, 4),
            (1, 5, 5, 5),
            (1, 6, 5, 6),
            (1, 7, 2, 7),
            (1, 8, 5, 8),
        ]
    )
    split = make_chronological_split(ratings, minimum_positive_interactions=3, n_movies=8)
    assert split.summary["n_eligible_users"] == 1
    assert int(split.test.iloc[0]["movie_id"]) == 8
    assert int(split.validation.iloc[0]["movie_id"]) == 6
    train_ids = set(split.train["movie_id"])
    assert train_ids == {1, 2, 3, 4, 5}
    assert 7 not in train_ids
    assert leakage_violations(split, minimum_positive_interactions=3) == []


def test_removes_users_without_enough_positive_history():
    ratings = _ratings(
        [
            (1, 1, 5, 1),
            (1, 2, 4, 2),
            (1, 3, 5, 3),
            (2, 1, 5, 1),
            (2, 2, 5, 2),
            (2, 3, 5, 3),
            (2, 4, 5, 4),
            (2, 5, 5, 5),
        ]
    )
    split = make_chronological_split(ratings, minimum_positive_interactions=3, n_movies=5)
    assert split.summary["removed_users"]["insufficient_positive_training_history"] == 1
    assert set(split.test["user_id"]) == {2}
    assert leakage_violations(split, minimum_positive_interactions=3) == []


def test_removes_users_with_fewer_than_two_relevant_ratings():
    ratings = _ratings([(1, 1, 5, 1), (1, 2, 2, 2), (1, 3, 1, 3)])
    split = make_chronological_split(ratings, minimum_positive_interactions=1, n_movies=3)
    assert split.summary["n_eligible_users"] == 0
    assert split.summary["removed_users"]["fewer_than_two_relevant"] == 1


def test_hand_built_overlap_is_reported():
    train = _ratings([(1, 1, 5, 1)])
    validation = _ratings([(1, 1, 5, 2)])
    test = _ratings([(1, 2, 5, 3)])
    problems = leakage_violations(
        ChronologicalSplit(train, validation, test, {}),
        minimum_positive_interactions=1,
    )
    assert any("overlap validation" in problem for problem in problems)


def test_synthetic_catalog_has_no_leakage():
    ratings, _movies, _genres = synthetic_catalog()
    split = make_chronological_split(ratings, n_movies=12)
    assert split.summary["n_eligible_users"] == 6
    assert leakage_violations(split) == []
    merged = split.train.merge(split.test, on=["user_id", "movie_id"], how="inner")
    assert merged.empty
    merged_val = split.train.merge(split.validation, on=["user_id", "movie_id"], how="inner")
    assert merged_val.empty
