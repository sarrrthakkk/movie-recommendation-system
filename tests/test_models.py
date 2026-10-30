"""Model behavior that does not require the full MovieLens run."""

from __future__ import annotations

import numpy as np
import pandas as pd

from movie_recommender.data import EXPECTED_GENRE_DIM
from movie_recommender.models.content_based import ContentModel
from movie_recommender.models.hybrid import combine_scores, minmax_normalize
from movie_recommender.models.matrix_factorization import MatrixFactorizationRecommender
from movie_recommender.models.popularity import PopularityModel
from movie_recommender.splitting import make_chronological_split
from tests.helpers import synthetic_catalog


def test_minmax_and_hybrid_formula():
    assert np.allclose(minmax_normalize(np.array([10.0, 20.0, 30.0])), [0.0, 0.5, 1.0])
    assert np.allclose(minmax_normalize(np.array([5.0, 5.0, 5.0])), [0.5, 0.5, 0.5])
    assert np.allclose(combine_scores(np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.25), [0.25, 0.75])


def test_popularity_bayesian_average_and_seen_exclusion():
    train = pd.DataFrame(
        {
            "user_id": [1] * 20 + [2] + [3] * 20,
            "movie_id": [1] * 20 + [2] + [3] * 20,
            "rating": [4.0] * 20 + [5.0] + [1.0] * 20,
            "timestamp": list(range(41)),
        }
    )
    model = PopularityModel(prior_strength=10).fit(train, np.array([1, 2, 3]), relevance_threshold=4)
    # Movie 2 has one rating of 5. Movie 1 has twenty ratings of 4.
    # Bayes(2) = (10 * mu + 5) / 11. Bayes(1) = (10 * mu + 80) / 30.
    assert model.scores(np.array([1]), mode="positive_count")[0] == 20
    assert model.scores(np.array([2]), mode="positive_count")[0] == 1
    assert model.scores(np.array([1]), mode="bayesian")[0] > model.scores(np.array([2]), mode="bayesian")[0]
    recommended, _scores = model.recommend({1}, k=2, mode="positive_count")
    assert 1 not in recommended
    assert list(recommended)[0] == 2


def test_content_profile_weights_and_empty_fallback():
    genres = np.zeros((3, EXPECTED_GENRE_DIM))
    genres[0, 0] = 1
    genres[1, 1] = 1
    model = ContentModel().fit(np.array([1, 2, 3]), genres)
    history = pd.DataFrame({"movie_id": [1, 2], "rating": [5.0, 4.0]})
    profile = model.build_profile(history)
    assert np.allclose(profile[:2], [2 / 3, 1 / 3])
    empty = model.build_profile(pd.DataFrame({"movie_id": [1], "rating": [2.0]}))
    assert model.profile_is_empty(empty)
    popularity = PopularityModel().fit(
        pd.DataFrame({"user_id": [1, 1], "movie_id": [2, 3], "rating": [5.0, 4.0], "timestamp": [1, 2]}),
        np.array([1, 2, 3]),
    )
    ids, scores, used_fallback = model.recommend(history.iloc[0:0], seen_movie_ids={1}, k=2, popularity=popularity)
    assert used_fallback
    assert 1 not in ids
    assert scores[0] >= scores[-1]


def test_matrix_factorization_shapes_seen_exclusion_and_checkpoint(tmp_path):
    ratings, movies, genres = synthetic_catalog(n_users=6, n_movies=8)
    split = make_chronological_split(ratings, n_movies=len(movies))
    model = MatrixFactorizationRecommender()
    model.fit(
        split.train,
        split.validation,
        movies["movie_id"].to_numpy(),
        n_factors=4,
        learning_rate=0.01,
        regularization=0.01,
        epochs=2,
        batch_size=8,
        patience=2,
        seed=7,
        device="cpu",
        checkpoint_dir=tmp_path,
    )
    user_id = int(split.train["user_id"].iloc[0])
    seen = set(split.train.loc[split.train["user_id"] == user_id, "movie_id"].astype(int))
    recommended, scores = model.recommend(user_id, seen, k=3)
    assert len(recommended) == 3
    assert scores.shape == (3,)
    assert seen.isdisjoint(recommended)
    unseen = np.asarray([movie_id for movie_id in movies["movie_id"] if int(movie_id) not in seen], dtype=int)
    assert model.score_movies(user_id, unseen).shape == (len(unseen),)
    assert np.isfinite(model.predict(user_id, int(unseen[0])))
    restored = MatrixFactorizationRecommender.load(tmp_path, device="cpu")
    assert np.allclose(
        model.score_movies(user_id, unseen),
        restored.score_movies(user_id, unseen),
    )
    assert len(model.history["train_loss"]) == len(model.history["val_rmse"])
    _ = genres
