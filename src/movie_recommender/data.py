"""MovieLens parsing, validation, and processed-data I/O."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from movie_recommender.utils import load_json, save_json

GENRE_NAMES: tuple[str, ...] = (
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)
EXPECTED_GENRE_DIM = len(GENRE_NAMES)
ITEM_FIELD_COUNT = 5 + EXPECTED_GENRE_DIM
_YEAR_IN_TITLE = re.compile(r"\((\d{4})\)\s*$")


class DataValidationError(ValueError):
    """Raised when MovieLens files violate a hard validation rule."""


@dataclass
class ProcessedData:
    ratings: pd.DataFrame
    movies: pd.DataFrame
    genres: np.ndarray
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    summary: dict


def discover_data_dir(explicit: str | Path | None = None) -> Path:
    """Find a MovieLens directory containing ``u.data`` and ``u.item``."""
    from movie_recommender.utils import project_root, resolve_path

    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(resolve_path(explicit))
    root = project_root()
    candidates.extend(
        [
            root / "ml-100k",
            root / "data" / "raw" / "ml-100k",
            root / "data" / "raw",
        ]
    )
    for candidate in candidates:
        if (candidate / "u.data").exists() and (candidate / "u.item").exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find MovieLens files u.data and u.item. "
        f"Searched: {searched}. Pass --data-dir or run python scripts/download_data.py."
    )


def _detect_delimiter(header_line: str) -> str:
    if header_line.count("|") >= ITEM_FIELD_COUNT - 1:
        return "|"
    if header_line.count("\t") >= ITEM_FIELD_COUNT - 1:
        return "\t"
    raise DataValidationError(
        "u.item is neither pipe-delimited nor tab-delimited with the expected 24 fields. "
        "The original MovieLens 100K movie file is pipe-delimited."
    )


def _parse_year(title: str, release_date: str) -> int | None:
    match = _YEAR_IN_TITLE.search(title)
    if match:
        year = int(match.group(1))
        if 1800 <= year <= 2100:
            return year
    if release_date:
        try:
            return datetime.strptime(release_date, "%d-%b-%Y").year
        except ValueError:
            return None
    return None


def parse_movies(item_path: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Parse ``u.item`` into a movie table and a multi-hot genre matrix.

    The original MovieLens 100K file is pipe-delimited and encoded in latin-1.
    A tab-delimited file with the same 24 columns is also accepted.
    """
    path = Path(item_path)
    if not path.exists():
        raise FileNotFoundError(f"Movie file not found: {path}")
    text = path.read_text(encoding="latin-1")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise DataValidationError(f"Movie file is empty: {path}")
    delimiter = _detect_delimiter(lines[0])
    rows: list[dict] = []
    vectors: list[list[int]] = []
    seen_ids: set[int] = set()
    for line_number, line in enumerate(lines, start=1):
        parts = line.split(delimiter)
        if len(parts) != ITEM_FIELD_COUNT:
            raise DataValidationError(
                f"{path}:{line_number}: expected {ITEM_FIELD_COUNT} fields, found {len(parts)}"
            )
        try:
            movie_id = int(parts[0])
        except ValueError as exc:
            raise DataValidationError(f"{path}:{line_number}: invalid movie id") from exc
        if movie_id in seen_ids:
            raise DataValidationError(f"{path}:{line_number}: duplicate movie id {movie_id}")
        seen_ids.add(movie_id)
        flags: list[int] = []
        for raw_flag in parts[5:]:
            if raw_flag not in {"0", "1"}:
                raise DataValidationError(
                    f"{path}:{line_number}: genre flags must be 0 or 1, found {raw_flag!r}"
                )
            flags.append(int(raw_flag))
        if len(flags) != EXPECTED_GENRE_DIM:
            raise DataValidationError(
                f"{path}:{line_number}: expected {EXPECTED_GENRE_DIM} genre flags"
            )
        title = parts[1].strip()
        release_date = parts[2].strip()
        genre_labels = [GENRE_NAMES[i] for i, flag in enumerate(flags) if flag == 1]
        rows.append(
            {
                "movie_id": movie_id,
                "title": title,
                "year": _parse_year(title, release_date),
                "release_date": release_date,
                "genres": "|".join(genre_labels),
            }
        )
        vectors.append(flags)
    movies = pd.DataFrame(rows).sort_values("movie_id").reset_index(drop=True)
    order = movies["movie_id"].to_numpy()
    by_id = {int(row["movie_id"]): vectors[i] for i, row in enumerate(rows)}
    genres = np.asarray([by_id[int(movie_id)] for movie_id in order], dtype=np.float64)
    if genres.shape[1] != EXPECTED_GENRE_DIM:
        raise DataValidationError(
            f"Genre matrix has width {genres.shape[1]}, expected {EXPECTED_GENRE_DIM}"
        )
    return movies, genres


def parse_ratings(data_path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Parse ``u.data`` and resolve duplicate user-movie rows by keeping the latest."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Ratings file not found: {path}")
    ratings = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
        dtype={"user_id": int, "movie_id": int, "rating": float, "timestamp": int},
    )
    if ratings.empty:
        raise DataValidationError(f"Ratings file is empty: {path}")
    report: dict = {"rows_read": int(len(ratings))}
    if ratings[["user_id", "movie_id", "rating", "timestamp"]].isna().any().any():
        raise DataValidationError("Ratings contain missing values")
    if not ratings["rating"].between(1, 5).all():
        bad = ratings.loc[~ratings["rating"].between(1, 5), "rating"].unique()
        raise DataValidationError(f"Ratings outside [1, 5]: {bad.tolist()}")
    timestamps = pd.to_numeric(ratings["timestamp"], errors="coerce")
    if timestamps.isna().any() or (timestamps < 0).any():
        raise DataValidationError("Timestamps must be non-negative Unix seconds")
    parsed = pd.to_datetime(timestamps, unit="s", errors="coerce")
    if parsed.isna().any():
        raise DataValidationError("One or more timestamps could not be parsed")
    ratings = ratings.copy()
    ratings["timestamp"] = timestamps.astype(np.int64)
    duplicate_mask = ratings.duplicated(["user_id", "movie_id"], keep=False)
    report["duplicate_rows"] = int(duplicate_mask.sum())
    if duplicate_mask.any():
        ratings = (
            ratings.sort_values(["user_id", "movie_id", "timestamp"])
            .drop_duplicates(["user_id", "movie_id"], keep="last")
            .reset_index(drop=True)
        )
    report["rows_after_dedup"] = int(len(ratings))
    report["duplicate_pairs_resolved"] = int(report["rows_read"] - report["rows_after_dedup"])
    return ratings.reset_index(drop=True), report


def validate_id_consistency(ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
    missing = set(ratings["movie_id"].unique()) - set(movies["movie_id"].unique())
    if missing:
        sample = sorted(missing)[:10]
        raise DataValidationError(
            f"{len(missing)} movie ids in ratings are absent from u.item. Examples: {sample}"
        )


def load_movielens(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict]:
    directory = Path(data_dir)
    movies, genres = parse_movies(directory / "u.item")
    ratings, report = parse_ratings(directory / "u.data")
    validate_id_consistency(ratings, movies)
    report["n_users"] = int(ratings["user_id"].nunique())
    report["n_movies"] = int(len(movies))
    report["n_ratings"] = int(len(ratings))
    report["genre_dimensions"] = int(genres.shape[1])
    report["missing_year"] = int(movies["year"].isna().sum())
    report["sparsity"] = float(
        1.0 - report["n_ratings"] / (report["n_users"] * report["n_movies"])
    )
    report["rating_distribution"] = {
        str(int(value) if float(value).is_integer() else value): int(count)
        for value, count in ratings["rating"].value_counts().sort_index().items()
    }
    return ratings, movies, genres, report


def save_processed(
    processed_dir: str | Path,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    genres: np.ndarray,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    summary: dict,
) -> None:
    directory = Path(processed_dir)
    directory.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(directory / "ratings.csv", index=False)
    movies.to_csv(directory / "movies.csv", index=False)
    train.to_csv(directory / "train.csv", index=False)
    validation.to_csv(directory / "validation.csv", index=False)
    test.to_csv(directory / "test.csv", index=False)
    np.save(directory / "genres.npy", genres)
    np.save(directory / "movie_ids.npy", movies["movie_id"].to_numpy())
    save_json(directory / "genre_names.json", list(GENRE_NAMES))
    save_json(directory / "split_summary.json", summary)


def _read_split_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={"user_id": int, "movie_id": int, "rating": float, "timestamp": int},
    )
    return frame


def load_processed(processed_dir: str | Path) -> ProcessedData:
    directory = Path(processed_dir)
    required = [
        "ratings.csv",
        "movies.csv",
        "genres.npy",
        "movie_ids.npy",
        "train.csv",
        "validation.csv",
        "test.csv",
        "split_summary.json",
    ]
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Processed data in {directory} is missing {missing}. Run python scripts/prepare_data.py."
        )
    movies = pd.read_csv(directory / "movies.csv")
    movies["movie_id"] = movies["movie_id"].astype(int)
    movies["year"] = movies["year"].astype("Int64")
    genres = np.load(directory / "genres.npy")
    movie_ids = np.load(directory / "movie_ids.npy")
    if len(movies) != len(genres) or len(movies) != len(movie_ids):
        raise DataValidationError("Processed movie, genre, and id files have different lengths")
    if not np.array_equal(movie_ids.astype(int), movies["movie_id"].to_numpy()):
        raise DataValidationError("genres.npy is not aligned with movies.csv")
    if genres.shape[1] != EXPECTED_GENRE_DIM:
        raise DataValidationError(
            f"Loaded genre matrix has width {genres.shape[1]}, expected {EXPECTED_GENRE_DIM}"
        )
    return ProcessedData(
        ratings=_read_split_frame(directory / "ratings.csv"),
        movies=movies,
        genres=genres,
        train=_read_split_frame(directory / "train.csv"),
        validation=_read_split_frame(directory / "validation.csv"),
        test=_read_split_frame(directory / "test.csv"),
        summary=load_json(directory / "split_summary.json"),
    )
