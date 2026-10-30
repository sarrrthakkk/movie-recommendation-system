"""Shared builders for unit tests. No network and no real API keys."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from movie_recommender.data import EXPECTED_GENRE_DIM, GENRE_NAMES


def item_line(movie_id: int, title: str, flags: list[int] | None = None, release_date: str = "01-Jan-1995") -> str:
    genre_flags = flags or [0] * EXPECTED_GENRE_DIM
    encoded = "|".join(str(flag) for flag in genre_flags)
    return f"{movie_id}|{title}|{release_date}||http://example.test/m{movie_id}|{encoded}"


def write_tiny_item_file(path: Path) -> None:
    flags = []
    for index in range(4):
        row = [0] * EXPECTED_GENRE_DIM
        row[index % EXPECTED_GENRE_DIM] = 1
        row[(index + 1) % EXPECTED_GENRE_DIM] = 1
        flags.append(row)
    lines = [
        item_line(1, "Toy Story (1995)", flags[0]),
        item_line(2, "GoldenEye (1995)", flags[1]),
        item_line(3, "Four Rooms (1995)", flags[2]),
        item_line(4, "Get Shorty (1995)", flags[3], release_date=""),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="latin-1")


def synthetic_catalog(n_users: int = 6, n_movies: int = 12) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    genres = np.zeros((n_movies, EXPECTED_GENRE_DIM), dtype=float)
    names = []
    for index in range(n_movies):
        genres[index, index % EXPECTED_GENRE_DIM] = 1
        genres[index, (index + 3) % EXPECTED_GENRE_DIM] = 1
        labels = [GENRE_NAMES[column] for column in range(EXPECTED_GENRE_DIM) if genres[index, column] == 1]
        names.append("|".join(labels))
    movies = pd.DataFrame(
        {
            "movie_id": np.arange(1, n_movies + 1),
            "title": [f"Movie {movie_id} (1995)" for movie_id in range(1, n_movies + 1)],
            "year": [1995] * n_movies,
            "release_date": ["01-Jan-1995"] * n_movies,
            "genres": names,
        }
    )
    rows = []
    timestamp = 1_000
    pattern = [5, 5, 4, 5, 4, 5, 2]
    for user_id in range(1, n_users + 1):
        for offset, rating in enumerate(pattern):
            movie_id = ((user_id + offset - 1) % n_movies) + 1
            rows.append((user_id, movie_id, float(rating), timestamp))
            timestamp += 1
    ratings = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])
    return ratings, movies, genres
