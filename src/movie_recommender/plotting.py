"""Matplotlib figures for the offline and LLM experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from movie_recommender.utils import ensure_dir


def _finish(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_rating_distribution(ratings: pd.DataFrame, path: Path) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7, 4))
    counts = ratings["rating"].value_counts().sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values, color="#4C78A8")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.title("Rating distribution (cleaned MovieLens ratings)")
    _finish(path)


def plot_user_activity(ratings: pd.DataFrame, path: Path) -> None:
    sns.set_theme(style="whitegrid")
    counts = ratings.groupby("user_id").size()
    plt.figure(figsize=(7, 4))
    sns.histplot(counts, bins=30, color="#4C78A8")
    plt.xlabel("Ratings per user")
    plt.ylabel("Users")
    plt.title("User interaction distribution")
    _finish(path)


def plot_movie_popularity(train: pd.DataFrame, path: Path) -> None:
    sns.set_theme(style="whitegrid")
    counts = train.groupby("movie_id").size()
    plt.figure(figsize=(7, 4))
    sns.histplot(counts, bins=30, color="#F58518")
    plt.xlabel("Training ratings per movie")
    plt.ylabel("Movies")
    plt.title("Movie popularity distribution (training interactions only)")
    _finish(path)


def plot_training_curves(history: dict, path: Path) -> None:
    sns.set_theme(style="whitegrid")
    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)
    panels = 3 if history.get("val_ndcg") else 2
    fig, axes = plt.subplots(1, panels, figsize=(4.2 * panels, 4))
    axes[0].plot(epochs, history.get("train_loss", []), marker="o")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE + L2")
    axes[1].plot(epochs, history.get("val_rmse", []), marker="o", color="#E45756")
    axes[1].set_title("Validation RMSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE")
    if panels == 3:
        axes[2].plot(epochs, history.get("val_ndcg", []), marker="o", color="#54A24B")
        axes[2].set_title("Validation NDCG@10")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("NDCG@10")
    _finish(path)


def plot_alpha(alpha_table: pd.DataFrame, path: Path) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7, 4))
    sns.lineplot(data=alpha_table, x="alpha", y="ndcg@10", marker="o")
    plt.xlabel("Hybrid alpha (weight on matrix factorization)")
    plt.ylabel("Validation NDCG@10")
    plt.title("Alpha selection on validation data")
    _finish(path)


def plot_model_comparison(comparison: pd.DataFrame, path: Path) -> None:
    sns.set_theme(style="whitegrid")
    long = comparison.melt(id_vars="model", var_name="metric", value_name="value")
    plt.figure(figsize=(11, 5))
    sns.barplot(data=long, x="metric", y="value", hue="model")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Score")
    plt.title("Model comparison")
    plt.legend(loc="best", fontsize=8)
    _finish(path)


def plot_buckets(bucket_frame: pd.DataFrame, path: Path) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 4))
    sns.barplot(data=bucket_frame, x="bucket", y="ndcg@10", hue="model")
    plt.xlabel("Training-popularity bucket of the held-out test movie")
    plt.ylabel("NDCG@10")
    plt.title("Head, mid, and tail performance")
    _finish(path)


def plot_rank_displacement(displacements: np.ndarray, path: Path) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7, 4))
    sns.histplot(displacements, bins=20, color="#54A24B")
    plt.xlabel("Mean absolute candidate-rank displacement")
    plt.ylabel("Users")
    plt.title("LLM rank displacement within the candidate set")
    _finish(path)


def plot_quality_vs_cost(frame: pd.DataFrame, path: Path) -> None:
    """Compare NDCG@10 with the number of uncached client calls and mean latency."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.barplot(data=frame, x="method", y="ndcg@10", ax=axes[0], color="#4C78A8")
    axes[0].set_title("Quality")
    axes[0].tick_params(axis="x", rotation=20)
    sns.barplot(data=frame, x="method", y="api_calls", ax=axes[1], color="#F58518")
    axes[1].set_title("Uncached client calls")
    axes[1].tick_params(axis="x", rotation=20)
    _finish(path)


def save_eda_plots(ratings: pd.DataFrame, train: pd.DataFrame, directory: Path) -> None:
    ensure_dir(directory)
    plot_rating_distribution(ratings, directory / "rating_distribution.png")
    plot_user_activity(ratings, directory / "user_activity.png")
    plot_movie_popularity(train, directory / "movie_popularity.png")
