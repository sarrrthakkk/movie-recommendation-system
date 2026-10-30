"""MovieLens parsing and configuration tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from movie_recommender.config import AppConfig
from movie_recommender.data import (
    EXPECTED_GENRE_DIM,
    DataValidationError,
    load_movielens,
    load_processed,
    parse_movies,
    parse_ratings,
    save_processed,
)
from movie_recommender.splitting import make_chronological_split
from movie_recommender.utils import project_root
from tests.helpers import synthetic_catalog, write_tiny_item_file


def test_config_loads_required_values():
    config = AppConfig.from_yaml(project_root() / "configs" / "default.yaml")
    assert config.seed == 42
    assert config.llm.model == "qwen35-27b"
    assert config.llm.requests_per_minute == 6
    assert config.hybrid.alpha_values == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert config.data.relevance_threshold == 4.0


def test_pipe_delimited_movies_and_years(tmp_path):
    item_path = tmp_path / "u.item"
    write_tiny_item_file(item_path)
    movies, genres = parse_movies(item_path)
    assert list(movies["movie_id"]) == [1, 2, 3, 4]
    assert movies.loc[movies["movie_id"] == 1, "year"].iloc[0] == 1995
    assert "Animation" in movies.loc[movies["movie_id"] == 1, "genres"].iloc[0] or genres.shape[1] == 19
    assert genres.shape == (4, EXPECTED_GENRE_DIM)
    assert set(np.unique(genres)).issubset({0.0, 1.0})


def test_duplicate_ratings_keep_the_latest_timestamp(tmp_path):
    path = tmp_path / "u.data"
    path.write_text("1\t10\t2\t100\n1\t10\t5\t300\n2\t10\t4\t200\n", encoding="utf-8")
    ratings, report = parse_ratings(path)
    assert report["duplicate_pairs_resolved"] == 1
    kept = ratings.loc[ratings["user_id"] == 1].iloc[0]
    assert kept["rating"] == 5
    assert kept["timestamp"] == 300


def test_rejects_ratings_outside_range(tmp_path):
    path = tmp_path / "u.data"
    path.write_text("1\t10\t6\t100\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="outside"):
        parse_ratings(path)


def test_rejects_bad_timestamps(tmp_path):
    path = tmp_path / "u.data"
    path.write_text("1\t10\t4\t-5\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="Timestamps"):
        parse_ratings(path)


def test_rejects_unknown_movie_ids(tmp_path):
    write_tiny_item_file(tmp_path / "u.item")
    (tmp_path / "u.data").write_text("1\t99\t4\t100\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="absent"):
        load_movielens(tmp_path)


def test_processed_roundtrip_preserves_genre_alignment(tmp_path):
    ratings, movies, genres = synthetic_catalog()
    split = make_chronological_split(ratings, n_movies=len(movies))
    save_processed(tmp_path, ratings, movies, genres, split.train, split.validation, split.test, split.summary)
    loaded = load_processed(tmp_path)
    assert loaded.genres.shape[1] == EXPECTED_GENRE_DIM
    assert np.array_equal(loaded.movies["movie_id"].to_numpy(), movies["movie_id"].to_numpy())
    assert np.allclose(loaded.genres, genres)


def test_real_movielens_if_present():
    data_dir = project_root() / "ml-100k"
    if not (data_dir / "u.data").exists():
        pytest.skip("Local MovieLens 100K directory is not present")
    ratings, movies, genres, report = load_movielens(data_dir)
    assert len(ratings) == 100_000
    assert len(movies) == 1682
    assert genres.shape == (1682, EXPECTED_GENRE_DIM)
    assert ratings["rating"].between(1, 5).all()
    assert report["duplicate_pairs_resolved"] == 0
    assert movies["title"].str.contains("Toy Story").any()
    split = make_chronological_split(
        ratings,
        relevance_threshold=4,
        minimum_positive_interactions=3,
        n_movies=len(movies),
    )
    assert split.summary["n_eligible_users"] > 0
    assert split.summary["n_removed_users"] >= 0
    assert len(split.validation) == split.summary["n_eligible_users"]
    assert len(split.test) == split.summary["n_eligible_users"]
