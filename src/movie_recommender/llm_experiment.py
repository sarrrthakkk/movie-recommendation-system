"""Mock or live LLM reranking on a fixed sample of hybrid candidates."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from movie_recommender.candidates import (
    build_candidate_records,
    build_prompt_profile,
    sample_stratified_user_ids,
    sample_user_ids,
)
from movie_recommender.config import AppConfig
from movie_recommender.data import load_processed
from movie_recommender.evaluation import (
    paired_difference_ci,
    per_user_metrics,
    summarize_metric_frame,
)
from movie_recommender.llm.client import (
    LLMGateway,
    MockLLMClient,
    OpenAICompatibleClient,
    SQLiteCache,
    llm_credentials,
    request_interval_seconds,
)
from movie_recommender.llm.explanations import explain_movies
from movie_recommender.llm.prompts import PROMPT_VERSION, build_rerank_messages, prompt_token_estimate
from movie_recommender.llm.reranker import rerank_candidates
from movie_recommender.models.content_based import ContentModel
from movie_recommender.models.matrix_factorization import MatrixFactorizationRecommender
from movie_recommender.models.popularity import PopularityModel
from movie_recommender.plotting import plot_quality_vs_cost, plot_rank_displacement
from movie_recommender.ranking import hybrid_component_frame, prepare_users
from movie_recommender.utils import ensure_dir, get_logger, load_json, resolve_path, save_json

logger = get_logger(__name__)


class LLMDisabled(RuntimeError):
    """Raised when live LLM calls are requested without credentials."""


class ConfirmationRequired(RuntimeError):
    """Raised when a live run is missing ``--confirm-api-calls``."""


def _full_ranking(hybrid_order: np.ndarray, reranked_head: list[int]) -> np.ndarray:
    head = [int(movie_id) for movie_id in reranked_head]
    head_set = set(head)
    tail = [int(movie_id) for movie_id in hybrid_order if int(movie_id) not in head_set]
    return np.asarray(head + tail, dtype=int)


def _displacement(hybrid_head: list[int], reranked_head: list[int]) -> float:
    hybrid_pos = {movie_id: index for index, movie_id in enumerate(hybrid_head)}
    reranked_pos = {movie_id: index for index, movie_id in enumerate(reranked_head)}
    deltas = [abs(hybrid_pos[movie_id] - reranked_pos[movie_id]) for movie_id in hybrid_head]
    return float(np.mean(deltas)) if deltas else 0.0


def run_llm_evaluation(
    config: AppConfig,
    *,
    mock: bool = False,
    dry_run: bool = False,
    confirm_api_calls: bool = False,
    maximum_api_calls: int | None = None,
    refresh_cache: bool = False,
    run_ablations: bool | None = None,
    explanations: bool = False,
    sample_users: int | None = None,
    processed_dir: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    device: str | None = None,
) -> dict:
    """Compare hybrid rankings with LLM reranks on one shared user sample.

    Live HTTP calls require credentials and ``confirm_api_calls``. The mock client
    never touches the network. A negative or null difference is a valid result.
    """
    if run_ablations is None:
        run_ablations = bool(mock)
    processed = Path(processed_dir) if processed_dir else resolve_path(config.data.processed_dir)
    artifacts = Path(artifacts_dir) if artifacts_dir else resolve_path(config.artifacts_dir)
    metrics_dir = ensure_dir(artifacts / "metrics")
    plots_dir = ensure_dir(artifacts / "plots")
    data = load_processed(processed)
    model_dir = artifacts / "models"
    popularity = PopularityModel.load(model_dir / "popularity.json")
    content = ContentModel.load(model_dir)
    mf = MatrixFactorizationRecommender.load(model_dir, device=device or "cpu")
    alpha = float(load_json(model_dir / "selected_alpha.json")["alpha"])
    run_metadata = load_json(model_dir / "run_metadata.json") if (model_dir / "run_metadata.json").exists() else {}
    result_type = "smoke" if run_metadata.get("smoke") else run_metadata.get("result_type", "full")
    if mock:
        result_type = f"{result_type}_mock_llm"
    n_sample = sample_users or config.evaluation.llm_sample_users
    eligible = data.test["user_id"].astype(int).to_numpy()
    if config.evaluation.stratify_sample_users:
        sampled, user_tiers = sample_stratified_user_ids(
            data.train, eligible, n_sample, config.seed
        )
    else:
        sampled = sample_user_ids(eligible, n_sample, config.seed)
        user_tiers = {}
    if len(sampled) == 0:
        raise RuntimeError("No eligible users are available for LLM evaluation. Prepare the split first.")
    tier_counts = (
        {
            tier: int(sum(1 for u in sampled if user_tiers.get(int(u)) == tier))
            for tier in ("low", "medium", "high")
        }
        if user_tiers
        else {}
    )
    if tier_counts:
        logger.info(
            "LLM evaluation sample has %d users (stratified: low=%d, medium=%d, high=%d)",
            len(sampled),
            tier_counts["low"],
            tier_counts["medium"],
            tier_counts["high"],
        )
    else:
        logger.info("LLM evaluation sample has %d users", len(sampled))
    movie_ids = data.movies["movie_id"].to_numpy()
    prepared = prepare_users(sampled.tolist(), data.train, movie_ids, mf, content, popularity)
    prepared_by_id = {user.user_id: user for user in prepared}
    histories = {int(user_id): frame for user_id, frame in data.train.groupby("user_id")}
    test_targets = dict(zip(data.test["user_id"].astype(int), data.test["movie_id"].astype(int)))
    val_targets = dict(zip(data.validation["user_id"].astype(int), data.validation["movie_id"].astype(int)))
    for user_id in sampled.tolist():
        history_ids = set(histories[int(user_id)]["movie_id"].astype(int))
        if int(test_targets[int(user_id)]) in history_ids or int(val_targets[int(user_id)]) in history_ids:
            raise RuntimeError(f"Held-out interaction leaked into the prompt history for user {user_id}")

    candidate_limit = min(config.hybrid.top_k_candidates, config.llm.maximum_candidates)
    jobs = [{"name": "llm_combined", "history_mode": "combined", "k": candidate_limit, "primary": True}]
    if run_ablations:
        if 10 not in {candidate_limit}:
            jobs.append({"name": "llm_candidates_10", "history_mode": "combined", "k": min(10, candidate_limit), "primary": False})
        elif candidate_limit != 10:
            jobs.append({"name": "llm_candidates_10", "history_mode": "combined", "k": 10, "primary": False})
        else:
            jobs.append({"name": "llm_candidates_10", "history_mode": "combined", "k": 10, "primary": False})
        jobs.append({"name": "llm_highest_rated", "history_mode": "highest_rated", "k": candidate_limit, "primary": False})
        jobs.append({"name": "llm_recent_positive", "history_mode": "recent_positive", "k": candidate_limit, "primary": False})
    # The block above always appends the size-10 job when ablations are on. Keep it, including
    # the case where the primary job is already size 10, so the ablation file has an explicit row.

    gateway, client = _make_gateway(
        config,
        artifacts=artifacts,
        mock=mock,
        dry_run=True,
        maximum_api_calls=maximum_api_calls,
        refresh_cache=refresh_cache,
    )
    try:
        estimate = _estimate(gateway, jobs, prepared_by_id, histories, data, content, popularity, config, alpha, refresh_cache)
        if explanations:
            estimate["explanation_calls_upper_bound"] = min(5, len(sampled))
            estimate["uncached_upper_bound"] += estimate["explanation_calls_upper_bound"]
        estimate["repairs_not_included"] = (
            "A repair call is made only after an invalid ranking. It is not part of the base estimate."
        )
        save_json(metrics_dir / "llm_estimate.json", estimate)
        logger.info(
            "Estimated uncached LLM calls: %d (cached prompts: %d, users: %d)",
            estimate["uncached_calls"],
            estimate["cached_prompts"],
            estimate["n_users"],
        )
        if dry_run:
            estimate["executed"] = False
            estimate["user_tiers"] = user_tiers
            estimate["sampled_tier_counts"] = tier_counts
            save_json(metrics_dir / "llm_estimate.json", estimate)
            return estimate
        if not mock and llm_credentials() is None:
            raise LLMDisabled(
                "LLM_API_KEY or LLM_BASE_URL is missing. LLM functionality is disabled. "
                "Popularity, matrix factorization, content, and hybrid recommendations still run."
            )
        if not mock and not confirm_api_calls:
            raise ConfirmationRequired(
                "Refusing to make live API calls without --confirm-api-calls. "
                f"Estimated uncached calls: {estimate['uncached_calls']}."
            )
    finally:
        _close_client(client)

    live_gateway, live_client = _make_gateway(
        config,
        artifacts=artifacts,
        mock=mock,
        dry_run=False,
        maximum_api_calls=maximum_api_calls,
        refresh_cache=refresh_cache,
    )
    try:
        return _execute(
            config,
            live_gateway,
            jobs,
            prepared_by_id,
            histories,
            data,
            content,
            popularity,
            alpha,
            test_targets,
            sampled,
            metrics_dir,
            plots_dir,
            explanations=explanations,
            mock=mock,
            result_type=result_type,
            estimate=estimate,
            user_tiers=user_tiers,
            tier_counts=tier_counts,
        )
    finally:
        _close_client(live_client)


def _make_gateway(config: AppConfig, *, artifacts: Path, mock: bool, dry_run: bool, maximum_api_calls: int | None, refresh_cache: bool):
    cache = SQLiteCache(artifacts / "cache" / "llm_cache.sqlite")
    model = config.llm.model
    if os.environ.get("LLM_MODEL", "").strip():
        model = os.environ["LLM_MODEL"].strip()
    if mock or dry_run:
        client = MockLLMClient()
        interval = 0.0 if mock or dry_run else request_interval_seconds(config.llm.requests_per_minute)
        # Dry runs never call the client. Live mock runs also skip the network sleep.
        # The 10-second floor applies only when the OpenAI-compatible client is used.
    else:
        creds = llm_credentials()
        if creds is None:
            raise LLMDisabled(
                "LLM_API_KEY or LLM_BASE_URL is missing. LLM functionality is disabled."
            )
        api_key, base_url, _env_model = creds
        client = OpenAICompatibleClient(
            api_key=api_key,
            base_url=base_url,
            timeout=config.llm.request_timeout_seconds,
        )
        interval = request_interval_seconds(config.llm.requests_per_minute)
        logger.info("Live LLM calls use a minimum spacing of %.1f seconds", interval)
    if config.llm.concurrency != 1:
        logger.warning("Concurrency %s is clamped to a single in-flight request", config.llm.concurrency)
    gateway = LLMGateway(
        client,
        cache,
        model=model,
        temperature=config.llm.temperature,
        timeout=config.llm.request_timeout_seconds,
        maximum_retries=config.llm.maximum_retries,
        retry_base_seconds=config.llm.retry_base_seconds,
        min_interval_seconds=interval,
        maximum_api_calls=maximum_api_calls,
        dry_run=dry_run,
        refresh_cache=refresh_cache,
    )
    return gateway, client


def _close_client(client: object) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _estimate(gateway, jobs, prepared_by_id, histories, data, content, popularity, config, alpha, refresh_cache) -> dict:
    uncached = 0
    cached = 0
    prompt_tokens = []
    for job in jobs:
        for user in prepared_by_id.values():
            profile, candidates = _profile_and_candidates(
                job, user, histories, data, content, popularity, config, alpha
            )
            messages = build_rerank_messages(profile, candidates)
            prompt_tokens.append(prompt_token_estimate(messages))
            if refresh_cache or not gateway.is_cached(task="rerank", messages=messages):
                uncached += 1
            else:
                cached += 1
    return {
        "n_users": len(prepared_by_id),
        "n_jobs": len(jobs),
        "uncached_calls": uncached,
        "cached_prompts": cached,
        "uncached_upper_bound": uncached,
        "mean_prompt_tokens": float(np.mean(prompt_tokens)) if prompt_tokens else 0.0,
        "max_prompt_tokens": int(max(prompt_tokens) if prompt_tokens else 0),
        "prompt_version": PROMPT_VERSION,
        "dry_run_note": "No API call is made during a dry run.",
    }


def _profile_and_candidates(job, user, histories, data, content, popularity, config, alpha):
    profile = build_prompt_profile(
        histories[user.user_id],
        data.movies,
        content,
        maximum_positive_history=config.llm.maximum_positive_history,
        maximum_negative_history=config.llm.maximum_negative_history,
        history_mode=job["history_mode"],
        relevance_threshold=config.data.relevance_threshold,
    )
    frame = hybrid_component_frame(user, alpha)
    candidates = build_candidate_records(
        frame,
        data.movies,
        popularity.interaction_count,
        limit=int(job["k"]),
    )
    return profile, candidates


def _execute(
    config,
    gateway,
    jobs,
    prepared_by_id,
    histories,
    data,
    content,
    popularity,
    alpha,
    test_targets,
    sampled,
    metrics_dir,
    plots_dir,
    *,
    explanations: bool,
    mock: bool,
    result_type: str,
    estimate: dict,
    user_tiers: dict[int, str] | None = None,
    tier_counts: dict[str, int] | None = None,
) -> dict:
    genre_by_movie = {
        int(movie_id): data.genres[row] for row, movie_id in enumerate(data.movies["movie_id"].astype(int))
    }
    counts = {int(movie_id): float(value) for movie_id, value in popularity.interaction_count.items()}
    k_values = list(config.evaluation.k_values)
    primary_rows = []
    ablation_rows = []
    explanation_examples = []
    explained = 0
    per_job_rankings: dict[str, dict[int, np.ndarray]] = {}
    hybrid_rankings: dict[int, np.ndarray] = {}
    for job in jobs:
        rankings: dict[int, np.ndarray] = {}
        job_calls = 0
        displacements = []
        top1_changes = 0
        invalid = 0
        fallback = 0
        user_ids = sampled.tolist()
        for position, user_id in enumerate(user_ids, start=1):
            logger.info("Reranking user %s (%d of %d)", int(user_id), position, len(user_ids))
            user = prepared_by_id[int(user_id)]
            profile, candidates = _profile_and_candidates(
                job, user, histories, data, content, popularity, config, alpha
            )
            hybrid_head = [int(row["movie_id"]) for row in candidates]
            hybrid_full = hybrid_component_frame(user, alpha)["movie_id"].to_numpy(dtype=int)
            hybrid_rankings[int(user_id)] = hybrid_full
            outcome = rerank_candidates(gateway, profile, candidates)
            job_calls += outcome.api_calls
            reranked = _full_ranking(hybrid_full, outcome.ranked_movie_ids)
            rankings[int(user_id)] = reranked
            changed = int(outcome.ranked_movie_ids[0] != hybrid_head[0]) if hybrid_head else 0
            top1_changes += changed
            invalid += int(outcome.invalid_response)
            fallback += int(outcome.fallback_reason not in {"none", "repaired"})
            displacements.append(_displacement(hybrid_head, outcome.ranked_movie_ids))
            if job["primary"]:
                primary_rows.append(
                    {
                        "user_id": int(user_id),
                        "fallback_reason": outcome.fallback_reason,
                        "invalid_response": outcome.invalid_response,
                        "cache_hit": outcome.cache_hit,
                        "latency_s": outcome.latency_s,
                        "prompt_tokens": outcome.prompt_tokens,
                        "api_calls": outcome.api_calls,
                        "top1_changed": changed,
                        "mean_abs_displacement": displacements[-1],
                        "reasoning_summary": outcome.reasoning_summary,
                    }
                )
                if explanations and explained < 5 and outcome.fallback_reason != "dry_run":
                    top_movies = [row for row in candidates if int(row["movie_id"]) in set(outcome.ranked_movie_ids[:5])]
                    # Preserve the reranked top-5 order.
                    by_id = {int(row["movie_id"]): row for row in candidates}
                    top_movies = [by_id[movie_id] for movie_id in outcome.ranked_movie_ids[:5] if movie_id in by_id]
                    explanation = explain_movies(gateway, profile, top_movies)
                    explanation_examples.append(
                        {
                            "user_id": int(user_id),
                            "fallback_reason": explanation.fallback_reason,
                            "explanations": explanation.explanations,
                            "mock": mock,
                        }
                    )
                    explained += 1
        per_job_rankings[job["name"]] = rankings
        frame = per_user_metrics(
            rankings,
            {int(user_id): int(test_targets[int(user_id)]) for user_id in sampled.tolist()},
            k_values,
            genre_by_movie=genre_by_movie,
            interaction_count=counts,
            n_users_train=popularity.n_users,
        )
        summary = summarize_metric_frame(
            frame,
            rankings,
            k_values,
            n_movies=len(data.movies),
            n_resamples=config.evaluation.bootstrap_resamples,
            seed=config.seed,
        )
        ablation_rows.append(
            {
                "name": job["name"],
                "history_mode": job["history_mode"],
                "candidate_k": job["k"],
                "ndcg@10": summary["ndcg@10"]["mean"],
                "hit@10": summary["hit@10"]["mean"],
                "mrr": summary["mrr"]["mean"],
                "api_calls": job_calls,
                "invalid_response_rate": invalid / max(len(sampled), 1),
                "fallback_rate": fallback / max(len(sampled), 1),
                "top1_change_rate": top1_changes / max(len(sampled), 1),
                "mean_abs_displacement": float(np.mean(displacements)) if displacements else 0.0,
                "primary": job["primary"],
            }
        )
        if job["primary"]:
            primary_frame = frame
            primary_summary = summary
            primary_displacement = np.asarray(displacements, dtype=float)

    hybrid_frame = per_user_metrics(
        hybrid_rankings,
        {int(user_id): int(test_targets[int(user_id)]) for user_id in sampled.tolist()},
        k_values,
        genre_by_movie=genre_by_movie,
        interaction_count=counts,
        n_users_train=popularity.n_users,
    )
    hybrid_summary = summarize_metric_frame(
        hybrid_frame,
        hybrid_rankings,
        k_values,
        n_movies=len(data.movies),
        n_resamples=config.evaluation.bootstrap_resamples,
        seed=config.seed,
    )
    merged = primary_frame.merge(hybrid_frame, on="user_id", suffixes=("_llm", "_hybrid"))
    ndcg_ci = paired_difference_ci(
        merged["ndcg@10_llm"],
        merged["ndcg@10_hybrid"],
        n_resamples=config.evaluation.bootstrap_resamples,
        seed=config.seed,
    )
    hit_ci = paired_difference_ci(
        merged["hit@10_llm"],
        merged["hit@10_hybrid"],
        n_resamples=config.evaluation.bootstrap_resamples,
        seed=config.seed + 1,
    )
    detail = pd.DataFrame(primary_rows).merge(merged, on="user_id")
    if user_tiers:
        detail["user_tier"] = detail["user_id"].map(user_tiers)
    detail.to_csv(metrics_dir / "llm_per_user.csv", index=False)
    total_requests = gateway.cache_hits + gateway.cache_misses
    result = {
        "result_type": result_type,
        "mock_llm": mock,
        "prompt_version": PROMPT_VERSION,
        "seed": config.seed,
        "n_users": int(len(sampled)),
        "user_tiers": user_tiers or {},
        "sampled_tier_counts": tier_counts or {},
        "tier_counts": tier_counts or {},
        "selected_alpha": alpha,
        "estimate": estimate,
        "hybrid_on_sample": hybrid_summary,
        "llm_on_sample": primary_summary,
        "paired_ndcg@10_llm_minus_hybrid": ndcg_ci,
        "paired_hit@10_llm_minus_hybrid": hit_ci,
        "top1_change_rate": float(np.mean(detail["top1_changed"])) if len(detail) else 0.0,
        "mean_abs_rank_displacement": float(np.mean(detail["mean_abs_displacement"])) if len(detail) else 0.0,
        "invalid_response_rate": float(np.mean(detail["invalid_response"])) if len(detail) else 0.0,
        "fallback_rate": float(np.mean(~detail["fallback_reason"].isin(["none", "repaired"]))) if len(detail) else 0.0,
        "cache_hit_rate": (gateway.cache_hits / total_requests) if total_requests else None,
        "average_request_latency_s": float(np.mean(gateway.latencies)) if gateway.latencies else None,
        "average_prompt_tokens": float(np.mean(detail["prompt_tokens"])) if len(detail) else None,
        "api_calls": gateway.api_calls,
        "ablations": ablation_rows,
        "llm_disabled_behavior": "If the API is unavailable, reranking keeps the hybrid order.",
        "caveat": (
            "The LLM can only reorder the hybrid candidate set, so it cannot recover a relevant movie "
            "that the hybrid model did not nominate. Mock-client numbers are not measurements of qwen35-27b. "
            "A bootstrap interval is not a p-value. A small public dataset cannot establish online satisfaction."
        ),
    }
    if len(sampled) < 30:
        result["sample_limitation"] = (
            f"Only {len(sampled)} users were evaluated. That is too small for a reliable conclusion."
        )
    save_json(metrics_dir / "llm_comparison.json", result)
    save_json(metrics_dir / "llm_ablations.json", {"rows": ablation_rows, "mock_llm": mock, "result_type": result_type})
    if explanation_examples:
        save_json(metrics_dir / "explanation_examples.json", explanation_examples)
    if len(primary_displacement):
        plot_rank_displacement(primary_displacement, plots_dir / "llm_rank_displacement.png")
    cost = pd.DataFrame(
        [
            {
                "method": "hybrid",
                "ndcg@10": hybrid_summary["ndcg@10"]["mean"],
                "api_calls": 0,
            },
            *[
                {"method": row["name"], "ndcg@10": row["ndcg@10"], "api_calls": row["api_calls"]}
                for row in ablation_rows
            ],
        ]
    )
    plot_quality_vs_cost(cost, plots_dir / "quality_vs_api_calls.png")
    _append_llm_markdown(metrics_dir / "results_table.md", result)
    logger.info(
        "LLM minus hybrid NDCG@10 mean difference %.4f (interval %.4f to %.4f)",
        ndcg_ci["mean"],
        ndcg_ci["low"],
        ndcg_ci["high"],
    )
    return result


def _append_llm_markdown(path: Path, result: dict) -> None:
    ndcg = result["paired_ndcg@10_llm_minus_hybrid"]
    block = [
        "",
        "## LLM rerank on the shared sample",
        "",
        result["caveat"],
        "",
        f"- Result type: `{result['result_type']}`",
        f"- Mock client: `{result['mock_llm']}`",
        f"- Users: {result['n_users']}",
        f"- Mean paired NDCG@10 difference (LLM - hybrid): {ndcg['mean']:.4f}",
        f"- 95% bootstrap interval: [{ndcg['low']:.4f}, {ndcg['high']:.4f}]",
        f"- Interval excludes zero: `{ndcg['excludes_zero']}`",
        f"- Reading: {ndcg['reading']}",
        f"- Top-1 change rate: {result['top1_change_rate']:.4f}",
        f"- Mean absolute rank displacement: {result['mean_abs_rank_displacement']:.4f}",
        f"- Invalid response rate: {result['invalid_response_rate']:.4f}",
        f"- Fallback rate: {result['fallback_rate']:.4f}",
        f"- Cache hit rate: {result['cache_hit_rate']}",
        f"- Average request latency seconds: {result['average_request_latency_s']}",
        f"- Average prompt tokens: {result['average_prompt_tokens']}",
        f"- Uncached client calls: {result['api_calls']}",
        "",
    ]
    if result.get("sample_limitation"):
        block.append(result["sample_limitation"])
        block.append("")
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Results\n"
    path.write_text(existing.rstrip() + "\n" + "\n".join(block), encoding="utf-8")
