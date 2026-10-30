"""Recommendation helpers for the Streamlit demo.

A brand-new user has no matrix-factorization embedding. Cold-start hybrid
scoring replaces that embedding with the training-set popularity prior and
says so. It does not invent a collaborative user vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from movie_recommender.config import AppConfig
from movie_recommender.data import load_processed
from movie_recommender.llm.client import LLMGateway
from movie_recommender.llm.explanations import explain_movies, template_explanation
from movie_recommender.llm.reranker import rerank_candidates
from movie_recommender.models.content_based import ContentModel
from movie_recommender.models.hybrid import combine_scores, minmax_normalize, rank_order
from movie_recommender.models.matrix_factorization import MatrixFactorizationRecommender
from movie_recommender.models.popularity import PopularityModel
from movie_recommender.candidates import build_prompt_profile
from movie_recommender.utils import load_json, resolve_path


@dataclass
class DemoResources:
    movies: pd.DataFrame
    train: pd.DataFrame
    popularity: PopularityModel
    content: ContentModel
    mf: MatrixFactorizationRecommender
    alpha: float
    smoke: bool
    relevance_threshold: float


def load_demo_resources(config: AppConfig, artifacts_dir: str | Path | None = None, processed_dir: str | Path | None = None) -> DemoResources:
    processed = Path(processed_dir) if processed_dir else resolve_path(config.data.processed_dir)
    artifacts = Path(artifacts_dir) if artifacts_dir else resolve_path(config.artifacts_dir)
    data = load_processed(processed)
    model_dir = artifacts / "models"
    metadata_path = model_dir / "run_metadata.json"
    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    return DemoResources(
        movies=data.movies,
        train=data.train,
        popularity=PopularityModel.load(model_dir / "popularity.json"),
        content=ContentModel.load(model_dir),
        mf=MatrixFactorizationRecommender.load(model_dir, device="cpu"),
        alpha=float(load_json(model_dir / "selected_alpha.json")["alpha"]),
        smoke=bool(metadata.get("smoke", False)),
        relevance_threshold=config.data.relevance_threshold,
    )


def _lookup(movies: pd.DataFrame) -> dict[int, dict]:
    return {
        int(row.movie_id): {
            "title": str(row.title),
            "year": None if pd.isna(row.year) else int(row.year),
            "genres": str(row.genres),
        }
        for row in movies.itertuples(index=False)
    }


def _rows_from_order(
    movie_ids: np.ndarray,
    scores: np.ndarray,
    components: dict[str, np.ndarray],
    movies: pd.DataFrame,
    k: int,
    *,
    cold_start: bool,
) -> list[dict]:
    info = _lookup(movies)
    rows = []
    for index in range(min(k, len(movie_ids))):
        movie_id = int(movie_ids[index])
        meta = info[movie_id]
        row = {
            "rank": index + 1,
            "movie_id": movie_id,
            "title": meta["title"],
            "year": meta["year"],
            "genres": meta["genres"],
            "score": float(scores[index]),
            "cold_start": cold_start,
        }
        for name, values in components.items():
            row[name] = None if values is None else float(values[index])
        rows.append(row)
    return rows


def recommend_new_user(
    resources: DemoResources,
    ratings: pd.DataFrame,
    method: str,
    k: int,
    *,
    gateway: LLMGateway | None = None,
    candidate_k: int = 20,
) -> tuple[list[dict], list[str], str | None]:
    """Recommend for a user who was not part of matrix-factorization training."""
    warnings: list[str] = []
    if ratings.empty:
        return [], ["Add at least one rated movie before generating recommendations."], None
    seen = set(ratings["movie_id"].astype(int))
    catalog = resources.popularity.movie_ids
    unseen = np.asarray([movie_id for movie_id in catalog if int(movie_id) not in seen], dtype=int)
    if len(unseen) == 0:
        return [], ["Every catalog movie is already in the supplied ratings."], None
    profile = resources.content.build_profile(ratings)
    if resources.content.profile_is_empty(profile):
        content_scores = resources.popularity.scores(unseen, mode="bayesian")
        warnings.append("No positive genre profile was available, so content scores use popularity.")
    else:
        content_scores = resources.content.score_movies(profile, unseen)
    popularity_scores = resources.popularity.scores(unseen, mode="bayesian")
    content_norm = minmax_normalize(content_scores)
    popularity_norm = minmax_normalize(popularity_scores)
    cold_hybrid = combine_scores(popularity_norm, content_norm, resources.alpha)
    warnings.append(
        "Cold-start recommendation: this user has no trained collaborative embedding. "
        "The hybrid score uses the popularity prior in place of matrix factorization."
    )
    if method == "popularity":
        order = rank_order(unseen, popularity_scores)
        chosen = unseen[order]
        return (
            _rows_from_order(
                chosen,
                popularity_scores[order],
                {"popularity_score": popularity_scores[order], "content_score": content_scores[order], "mf_score": None, "hybrid_score": cold_hybrid[order]},
                resources.movies,
                k,
                cold_start=True,
            ),
            warnings,
            None,
        )
    if method == "content":
        order = rank_order(unseen, content_scores)
        chosen = unseen[order]
        return (
            _rows_from_order(
                chosen,
                content_scores[order],
                {"popularity_score": popularity_scores[order], "content_score": content_scores[order], "mf_score": None, "hybrid_score": cold_hybrid[order]},
                resources.movies,
                k,
                cold_start=True,
            ),
            warnings,
            None,
        )
    order = rank_order(unseen, cold_hybrid)
    ranked_ids = unseen[order]
    component_frame = pd.DataFrame(
        {
            "movie_id": ranked_ids,
            "mf_score": np.nan,
            "content_score": content_scores[order],
            "mf_score_normalized": popularity_norm[order],
            "content_score_normalized": content_norm[order],
            "hybrid_score": cold_hybrid[order],
            "popularity_score": popularity_scores[order],
            "positive_count": resources.popularity.scores(ranked_ids, mode="positive_count"),
        }
    )
    summary = None
    if method == "hybrid_llm":
        if gateway is None:
            warnings.append("The LLM API is unavailable. Showing the cold-start hybrid ranking instead.")
        else:
            from movie_recommender.candidates import build_candidate_records

            records = build_candidate_records(
                component_frame,
                resources.movies,
                resources.popularity.interaction_count,
                limit=candidate_k,
            )
            prompt_profile = build_prompt_profile(
                ratings,
                resources.movies,
                resources.content,
                relevance_threshold=resources.relevance_threshold,
            )
            outcome = rerank_candidates(gateway, prompt_profile, records)
            summary = outcome.reasoning_summary
            if outcome.fallback_reason not in {"none", "repaired"}:
                warnings.append(f"LLM reranking fell back ({outcome.fallback_reason}). The hybrid order was preserved where needed.")
            ranked_ids = _full_ranking(ranked_ids, outcome.ranked_movie_ids)
            component_frame = component_frame.set_index("movie_id").loc[list(ranked_ids)].reset_index()
    display = component_frame.head(k)
    rows = []
    info = _lookup(resources.movies)
    for position, record in enumerate(display.to_dict(orient="records"), start=1):
        meta = info[int(record["movie_id"])]
        rows.append(
            {
                "rank": position,
                "movie_id": int(record["movie_id"]),
                "title": meta["title"],
                "year": meta["year"],
                "genres": meta["genres"],
                "score": float(record["hybrid_score"]),
                "popularity_score": float(record["popularity_score"]),
                "content_score": float(record["content_score"]),
                "mf_score": None,
                "hybrid_score": float(record["hybrid_score"]),
                "cold_start": True,
            }
        )
    return rows, warnings, summary


def recommend_existing_user(
    resources: DemoResources,
    user_id: int,
    method: str,
    k: int,
    *,
    gateway: LLMGateway | None = None,
    candidate_k: int = 20,
) -> tuple[list[dict], list[str], str | None]:
    """Recommend from an existing user's training history only."""
    warnings: list[str] = []
    history = resources.train.loc[resources.train["user_id"] == int(user_id)]
    if history.empty or int(user_id) not in resources.mf.user_to_index:
        return [], [f"User {user_id} has no trained embedding. Use the new-user cold-start form instead."], None
    seen = set(history["movie_id"].astype(int))
    unseen = np.asarray([movie_id for movie_id in resources.popularity.movie_ids if int(movie_id) not in seen], dtype=int)
    mf_scores = resources.mf.score_movies(int(user_id), unseen)
    profile = resources.content.build_profile(history)
    if resources.content.profile_is_empty(profile):
        content_scores = resources.popularity.scores(unseen, mode="bayesian")
        warnings.append("This training history has an empty genre profile, so content scores use popularity.")
    else:
        content_scores = resources.content.score_movies(profile, unseen)
    popularity_scores = resources.popularity.scores(unseen, mode="bayesian")
    hybrid_scores = combine_scores(minmax_normalize(mf_scores), minmax_normalize(content_scores), resources.alpha)
    if method == "popularity":
        primary = popularity_scores
    elif method == "content":
        primary = content_scores
    else:
        primary = hybrid_scores
    order = rank_order(unseen, primary)
    ranked_ids = unseen[order]
    summary = None
    if method == "hybrid_llm":
        if gateway is None:
            warnings.append("The LLM API is unavailable. Showing the hybrid ranking instead.")
        else:
            from movie_recommender.candidates import build_candidate_records

            frame = pd.DataFrame(
                {
                    "movie_id": ranked_ids,
                    "mf_score": mf_scores[order],
                    "content_score": content_scores[order],
                    "mf_score_normalized": minmax_normalize(mf_scores)[order],
                    "content_score_normalized": minmax_normalize(content_scores)[order],
                    "hybrid_score": hybrid_scores[order],
                    "popularity_score": popularity_scores[order],
                    "positive_count": resources.popularity.scores(ranked_ids, mode="positive_count"),
                }
            )
            records = build_candidate_records(frame, resources.movies, resources.popularity.interaction_count, limit=candidate_k)
            prompt_profile = build_prompt_profile(
                history,
                resources.movies,
                resources.content,
                relevance_threshold=resources.relevance_threshold,
            )
            outcome = rerank_candidates(gateway, prompt_profile, records)
            summary = outcome.reasoning_summary
            if outcome.fallback_reason not in {"none", "repaired"}:
                warnings.append(f"LLM reranking fell back ({outcome.fallback_reason}).")
            ranked_ids = _full_ranking(ranked_ids, outcome.ranked_movie_ids)
    info = _lookup(resources.movies)
    position = {int(movie_id): index for index, movie_id in enumerate(unseen)}
    rows = []
    for rank, movie_id in enumerate(ranked_ids[:k], start=1):
        source = position[int(movie_id)]
        meta = info[int(movie_id)]
        rows.append(
            {
                "rank": rank,
                "movie_id": int(movie_id),
                "title": meta["title"],
                "year": meta["year"],
                "genres": meta["genres"],
                "score": float(primary[source]) if method != "hybrid_llm" else float(hybrid_scores[source]),
                "popularity_score": float(popularity_scores[source]),
                "content_score": float(content_scores[source]),
                "mf_score": float(mf_scores[source]),
                "hybrid_score": float(hybrid_scores[source]),
                "cold_start": False,
            }
        )
    return rows, warnings, summary


def explain_recommendations(
    resources: DemoResources,
    history: pd.DataFrame,
    recommendations: list[dict],
    gateway: LLMGateway | None,
) -> list[dict]:
    movies = recommendations[:5]
    if gateway is None:
        return [
            {
                "movie_id": int(row["movie_id"]),
                "explanation": template_explanation(int(row["movie_id"]), row["title"], row["genres"]),
            }
            for row in movies
        ]
    profile = build_prompt_profile(
        history,
        resources.movies,
        resources.content,
        relevance_threshold=resources.relevance_threshold,
    )
    outcome = explain_movies(gateway, profile, movies)
    return outcome.explanations


def _full_ranking(hybrid_order: np.ndarray, reranked_head: list[int]) -> np.ndarray:
    head = [int(movie_id) for movie_id in reranked_head]
    head_set = set(head)
    tail = [int(movie_id) for movie_id in hybrid_order if int(movie_id) not in head_set]
    return np.asarray(head + tail, dtype=int)
