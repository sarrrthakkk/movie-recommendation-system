"""Per-user chronological train, validation, and test splits."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ChronologicalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    summary: dict


def _empty_ratings() -> pd.DataFrame:
    return pd.DataFrame(columns=["user_id", "movie_id", "rating", "timestamp"])


def make_chronological_split(
    ratings: pd.DataFrame,
    *,
    relevance_threshold: float = 4.0,
    minimum_positive_interactions: int = 3,
    n_movies: int | None = None,
) -> ChronologicalSplit:
    """Hold out each eligible user's last two relevant interactions.

    Relevant means ``rating >= relevance_threshold``. For each user the most
    recent relevant interaction is the test item and the second-most-recent
    relevant interaction is the validation item. Training history contains
    only interactions that sort strictly before the validation interaction.
    Users without two relevant interactions, or without
    ``minimum_positive_interactions`` positive training ratings, are removed.
    """
    required = {"user_id", "movie_id", "rating", "timestamp"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"Ratings are missing columns: {sorted(missing)}")

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    removed = {
        "fewer_than_two_relevant": 0,
        "insufficient_positive_training_history": 0,
    }
    ordered = ratings.sort_values(["user_id", "timestamp", "movie_id"], kind="mergesort")
    for user_id, group in ordered.groupby("user_id", sort=False):
        group = group.sort_values(["timestamp", "movie_id"], kind="mergesort")
        relevant = group.loc[group["rating"] >= relevance_threshold]
        if len(relevant) < 2:
            removed["fewer_than_two_relevant"] += 1
            continue
        validation_row = relevant.iloc[-2]
        test_row = relevant.iloc[-1]
        val_key = (int(validation_row["timestamp"]), int(validation_row["movie_id"]))
        before_validation = (group["timestamp"] < val_key[0]) | (
            (group["timestamp"] == val_key[0]) & (group["movie_id"] < val_key[1])
        )
        history = group.loc[before_validation]
        n_positive = int((history["rating"] >= relevance_threshold).sum())
        if n_positive < minimum_positive_interactions:
            removed["insufficient_positive_training_history"] += 1
            continue
        train_parts.append(history)
        val_parts.append(relevant.iloc[[-2]])
        test_parts.append(relevant.iloc[[-1]])

    train = pd.concat(train_parts, ignore_index=True) if train_parts else _empty_ratings()
    validation = pd.concat(val_parts, ignore_index=True) if val_parts else _empty_ratings()
    test = pd.concat(test_parts, ignore_index=True) if test_parts else _empty_ratings()
    for frame in (train, validation, test):
        if not frame.empty:
            frame["user_id"] = frame["user_id"].astype(int)
            frame["movie_id"] = frame["movie_id"].astype(int)
            frame["rating"] = frame["rating"].astype(float)
            frame["timestamp"] = frame["timestamp"].astype(int)

    n_users = int(ratings["user_id"].nunique())
    n_movies = int(n_movies if n_movies is not None else ratings["movie_id"].nunique())
    n_ratings = int(len(ratings))
    n_eligible = int(validation["user_id"].nunique()) if not validation.empty else 0
    summary = {
        "n_users": n_users,
        "n_movies": n_movies,
        "n_ratings": n_ratings,
        "n_eligible_users": n_eligible,
        "n_removed_users": int(n_users - n_eligible),
        "removed_users": removed,
        "n_train_interactions": int(len(train)),
        "n_validation_interactions": int(len(validation)),
        "n_test_interactions": int(len(test)),
        "n_discarded_interactions": int(n_ratings - len(train) - len(validation) - len(test)),
        "sparsity": float(1.0 - n_ratings / (n_users * n_movies)) if n_users and n_movies else 1.0,
        "rating_distribution": {
            str(int(value) if float(value).is_integer() else value): int(count)
            for value, count in ratings["rating"].value_counts().sort_index().items()
        },
        "relevance_threshold": relevance_threshold,
        "minimum_positive_interactions": minimum_positive_interactions,
        "split_rule": (
            "Per user, sort by timestamp then movie id. "
            "Test is the latest relevant rating. "
            "Validation is the second-latest relevant rating. "
            "Training is every interaction strictly before the validation key."
        ),
    }
    split = ChronologicalSplit(train=train, validation=validation, test=test, summary=summary)
    violations = leakage_violations(
        split,
        relevance_threshold=relevance_threshold,
        minimum_positive_interactions=minimum_positive_interactions,
    )
    if violations:
        raise RuntimeError("Chronological split leaked held-out interactions: " + "; ".join(violations))
    return split


def _pairs(frame: pd.DataFrame) -> set[tuple[int, int]]:
    if frame.empty:
        return set()
    return set(zip(frame["user_id"].astype(int), frame["movie_id"].astype(int)))


def leakage_violations(
    split: ChronologicalSplit,
    *,
    relevance_threshold: float = 4.0,
    minimum_positive_interactions: int = 3,
) -> list[str]:
    """Return human-readable leakage problems. An empty list means the split is safe."""
    problems: list[str] = []
    train_pairs = _pairs(split.train)
    val_pairs = _pairs(split.validation)
    test_pairs = _pairs(split.test)
    if train_pairs & val_pairs:
        problems.append("training pairs overlap validation pairs")
    if train_pairs & test_pairs:
        problems.append("training pairs overlap test pairs")
    if val_pairs & test_pairs:
        problems.append("validation pairs overlap test pairs")
    if not split.validation.empty and split.validation["rating"].lt(relevance_threshold).any():
        problems.append("validation contains a non-relevant rating")
    if not split.test.empty and split.test["rating"].lt(relevance_threshold).any():
        problems.append("test contains a non-relevant rating")

    train_by_user = {int(uid): g for uid, g in split.train.groupby("user_id")} if not split.train.empty else {}
    val_by_user = (
        {int(row.user_id): row for row in split.validation.itertuples(index=False)}
        if not split.validation.empty
        else {}
    )
    test_by_user = (
        {int(row.user_id): row for row in split.test.itertuples(index=False)}
        if not split.test.empty
        else {}
    )
    if set(val_by_user) != set(test_by_user):
        problems.append("validation users and test users differ")
    for user_id, val_row in val_by_user.items():
        test_row = test_by_user[user_id]
        val_key = (int(val_row.timestamp), int(val_row.movie_id))
        test_key = (int(test_row.timestamp), int(test_row.movie_id))
        if not val_key < test_key:
            problems.append(f"user {user_id}: validation is not strictly before test")
        history = train_by_user.get(user_id)
        if history is None or history.empty:
            problems.append(f"user {user_id}: missing training history")
            continue
        keys = list(zip(history["timestamp"].astype(int), history["movie_id"].astype(int)))
        if any(key >= val_key for key in keys):
            problems.append(f"user {user_id}: training history is not strictly before validation")
        if int(val_row.movie_id) in set(history["movie_id"].astype(int)):
            problems.append(f"user {user_id}: validation movie appears in training")
        if int(test_row.movie_id) in set(history["movie_id"].astype(int)):
            problems.append(f"user {user_id}: test movie appears in training")
        n_positive = int((history["rating"] >= relevance_threshold).sum())
        if n_positive < minimum_positive_interactions:
            problems.append(f"user {user_id}: only {n_positive} positive training ratings")
    return problems
