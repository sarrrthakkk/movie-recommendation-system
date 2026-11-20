"""End-to-end smoke test on a tiny synthetic catalog. No network access."""

from __future__ import annotations

import pandas as pd

from movie_recommender.config import AppConfig
from movie_recommender.data import save_processed
from movie_recommender.llm_experiment import run_llm_evaluation
from movie_recommender.reporting import evaluate_models
from movie_recommender.splitting import leakage_violations, make_chronological_split
from movie_recommender.training import train_models
from tests.helpers import synthetic_catalog


def test_tiny_pipeline_trains_evaluates_and_mock_reranks(tmp_path):
    ratings, movies, genres = synthetic_catalog(n_users=6, n_movies=12)
    split = make_chronological_split(ratings, n_movies=len(movies))
    assert leakage_violations(split) == []
    processed = tmp_path / "processed"
    artifacts = tmp_path / "artifacts"
    save_processed(
        processed,
        ratings,
        movies,
        genres,
        split.train,
        split.validation,
        split.test,
        split.summary,
    )
    config = AppConfig()
    config.matrix_factorization.epochs = 2
    config.matrix_factorization.latent_dimensions = 4
    config.matrix_factorization.patience = 2
    config.matrix_factorization.batch_size = 16
    config.matrix_factorization.device = "cpu"
    config.evaluation.bootstrap_resamples = 25
    config.evaluation.llm_sample_users = 4
    config.evaluation.explanation_audit_samples = 4
    config.hybrid.top_k_candidates = 5
    config.llm.maximum_candidates = 5
    metadata = train_models(
        config,
        smoke=False,
        processed_dir=processed,
        artifacts_dir=artifacts,
        device="cpu",
    )
    assert metadata["selected_alpha"] in config.hybrid.alpha_values
    offline = evaluate_models(config, processed_dir=processed, artifacts_dir=artifacts, device="cpu")
    assert "hybrid" in offline["test_metrics"]
    assert offline["test_metrics"]["hybrid"]["n_users"] == split.summary["n_eligible_users"]
    llm = run_llm_evaluation(
        config,
        mock=True,
        dry_run=False,
        confirm_api_calls=False,
        processed_dir=processed,
        artifacts_dir=artifacts,
        run_ablations=False,
        explanations=True,
        sample_users=4,
        device="cpu",
    )
    assert llm["mock_llm"] is True
    assert llm["n_users"] == 4
    assert "paired_ndcg@10_llm_minus_hybrid" in llm
    assert "p-value" not in llm["paired_ndcg@10_llm_minus_hybrid"]["reading"]
    assert (artifacts / "plots" / "model_comparison.png").exists()
    assert (artifacts / "plots" / "mf_training_curves.png").exists()
    assert (artifacts / "metrics" / "explanation_examples.json").exists()
    assert (artifacts / "metrics" / "llm_ablations.csv").exists()
    assert (artifacts / "metrics" / "manual_audit_explanations.csv").exists()
    audit_df = pd.read_csv(artifacts / "metrics" / "manual_audit_explanations.csv", keep_default_na=False)
    assert len(audit_df) == 4
    assert list(audit_df.columns) == [
        "user_id",
        "activity_tier",
        "movie_id",
        "movie_title",
        "ground_truth_genres",
        "user_recent_positive_history",
        "explanation_text",
        "used_fallback",
        "fallback_reason",
        "factual_support_1_5",
        "history_relevance_1_5",
        "user_utility_1_5",
        "auditor_notes",
    ]
    for col in [
        "user_id",
        "activity_tier",
        "movie_id",
        "movie_title",
        "ground_truth_genres",
        "user_recent_positive_history",
        "explanation_text",
        "used_fallback",
        "fallback_reason",
    ]:
        assert (audit_df[col].astype(str).str.strip() != "").all()
    for col in ["factual_support_1_5", "history_relevance_1_5", "user_utility_1_5", "auditor_notes"]:
        assert (audit_df[col] == "").all()
    cache = (artifacts / "cache" / "llm_cache.sqlite").read_bytes()
    assert b"LLM_API_KEY" not in cache


def test_smoke_with_tmdb_metadata_and_ablation(tmp_path):
    from movie_recommender.tmdb import save_tmdb_metadata

    ratings, movies, genres = synthetic_catalog(n_users=6, n_movies=12)
    split = make_chronological_split(ratings, n_movies=len(movies))
    processed = tmp_path / "processed"
    artifacts = tmp_path / "artifacts"
    save_processed(
        processed,
        ratings,
        movies,
        genres,
        split.train,
        split.validation,
        split.test,
        split.summary,
    )
    config = AppConfig()
    config.matrix_factorization.epochs = 2
    config.matrix_factorization.latent_dimensions = 4
    config.matrix_factorization.patience = 2
    config.matrix_factorization.batch_size = 16
    config.matrix_factorization.device = "cpu"
    config.evaluation.bootstrap_resamples = 25
    config.evaluation.llm_sample_users = 4
    config.evaluation.explanation_audit_samples = 4
    config.hybrid.top_k_candidates = 5
    config.llm.maximum_candidates = 5

    # 1. Missing cache path: should fall back to genre-only cleanly without crashing
    config.llm.use_tmdb_metadata = True
    config.llm.tmdb_cache_path = str(tmp_path / "non_existent.json")

    train_models(config, smoke=False, processed_dir=processed, artifacts_dir=artifacts, device="cpu")
    evaluate_models(config, processed_dir=processed, artifacts_dir=artifacts, device="cpu")

    llm_fallback = run_llm_evaluation(
        config,
        mock=True,
        dry_run=False,
        confirm_api_calls=False,
        processed_dir=processed,
        artifacts_dir=artifacts,
        run_ablations=False,
        explanations=True,
        sample_users=4,
        device="cpu",
    )
    assert llm_fallback["use_tmdb_metadata"] is False
    assert llm_fallback["tmdb_metadata_count"] == 0

    # 2. Populated cache path: runs with use_tmdb_metadata=True and registers llm_k5_genre_only ablation
    tmdb_cache = tmp_path / "mock_tmdb.json"
    mock_meta = {
        int(m): {
            "movie_id": int(m),
            "tmdb_id": 1000 + int(m),
            "matched_title": f"Synthetic Movie {m}",
            "overview": f"Plot overview for movie {m} with deep themes.",
            "director": f"Director {m}",
            "top_cast": [f"Actor {m}A", f"Actor {m}B"],
            "match_status": "matched",
        }
        for m in movies["movie_id"]
    }
    save_tmdb_metadata(tmdb_cache, mock_meta)
    config.llm.tmdb_cache_path = str(tmdb_cache)

    llm_tmdb = run_llm_evaluation(
        config,
        mock=True,
        dry_run=False,
        confirm_api_calls=False,
        processed_dir=processed,
        artifacts_dir=artifacts,
        run_ablations=True,
        explanations=True,
        refresh_cache=True,
        sample_users=4,
        device="cpu",
    )
    assert llm_tmdb["use_tmdb_metadata"] is True
    assert llm_tmdb["tmdb_metadata_count"] == len(movies)

    ablation_df = pd.read_csv(artifacts / "metrics" / "llm_ablations.csv")
    job_names = set(ablation_df["name"])
    assert "llm_k5_genre_only" in job_names
    assert "use_tmdb" in ablation_df.columns
    genre_only_row = ablation_df[ablation_df["name"] == "llm_k5_genre_only"].iloc[0]
    assert genre_only_row["use_tmdb"] == False
    primary_row = ablation_df[ablation_df["name"] == "llm_combined"].iloc[0]
    assert primary_row["use_tmdb"] == True
