"""End-to-end smoke test on a tiny synthetic catalog. No network access."""

from __future__ import annotations

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
    cache = (artifacts / "cache" / "llm_cache.sqlite").read_bytes()
    assert b"LLM_API_KEY" not in cache
