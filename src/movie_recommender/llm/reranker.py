"""Candidate reranking with one repair attempt and a hybrid fallback."""

from __future__ import annotations

from dataclasses import dataclass

from movie_recommender.llm.client import BudgetExceeded, GenerationResult, LLMGateway, LLMRequestError
from movie_recommender.llm.prompts import (
    TOKEN_WARNING_THRESHOLD,
    build_repair_messages,
    build_rerank_messages,
    prompt_token_estimate,
)
from movie_recommender.llm.schemas import parse_json_object, parse_rerank, valid_id_count
from movie_recommender.utils import get_logger

logger = get_logger(__name__)


@dataclass
class RerankOutcome:
    ranked_movie_ids: list[int]
    reasoning_summary: str
    fallback_reason: str
    invalid_response: bool
    cache_hit: bool
    latency_s: float
    prompt_tokens: int
    api_calls: int


def _hybrid_ids(candidates: list[dict]) -> list[int]:
    return [int(row["movie_id"]) for row in candidates]


def _from_result(result: GenerationResult, candidates: list[int]) -> tuple[list[int], bool, str]:
    if result.text is None:
        completed, _, _ = parse_rerank("{}", candidates)
        return completed, False, ""
    completed, exact, summary, _error = parse_rerank(result.text, candidates)
    return completed, exact, summary


def rerank_candidates(
    gateway: LLMGateway,
    profile: dict,
    candidates: list[dict],
    original_candidates: list[dict] | None = None,
    tmdb_metadata: dict[int, dict] | None = None,
    max_plot_chars: int = 220,
) -> RerankOutcome:
    """Rerank ``candidates``.

    If ``original_candidates`` is supplied, it provides the canonical hybrid order
    used for membership validation, error fallbacks, and completing partial permutations.
    """
    original = _hybrid_ids(original_candidates if original_candidates is not None else candidates)
    messages = build_rerank_messages(
        profile,
        candidates,
        tmdb_metadata=tmdb_metadata,
        max_plot_chars=max_plot_chars,
    )
    prompt_tokens = prompt_token_estimate(messages)
    if prompt_tokens > TOKEN_WARNING_THRESHOLD:
        logger.warning("Rerank prompt estimates %d tokens, above the %d token budget", prompt_tokens, TOKEN_WARNING_THRESHOLD)
    try:
        first = gateway.generate(task="rerank", messages=messages)
    except BudgetExceeded:
        return _outcome(original, "", "budget_exhausted", True, False, 0.0, prompt_tokens, 0)
    except LLMRequestError as exc:
        logger.info("Rerank request failed: %s", exc)
        return _outcome(original, "", "request_failed", True, False, 0.0, prompt_tokens, 0)

    if first.dry_run:
        return _outcome(original, "", "dry_run", False, False, 0.0, prompt_tokens, 0)

    ranked, exact, summary = _from_result(first, original)
    api_calls = int(first.api_call)
    latency = first.latency_s
    cache_hit = first.cache_hit
    if exact:
        return _outcome(ranked, summary, "none", False, cache_hit, latency, prompt_tokens, api_calls)

    repair_messages = build_repair_messages(
        profile,
        candidates,
        first.text or "",
        tmdb_metadata=tmdb_metadata,
        max_plot_chars=max_plot_chars,
    )
    try:
        repair = gateway.generate(task="repair", messages=repair_messages)
    except BudgetExceeded:
        partial, _, partial_summary = _from_result(first, original)
        return _outcome(partial, partial_summary, "partial_completion", True, cache_hit, latency, prompt_tokens, api_calls)
    except LLMRequestError:
        partial, _, partial_summary = _from_result(first, original)
        return _outcome(partial, partial_summary, "partial_completion", True, cache_hit, latency, prompt_tokens, api_calls)

    api_calls += int(repair.api_call)
    latency += repair.latency_s
    cache_hit = cache_hit and repair.cache_hit
    if repair.dry_run:
        return _outcome(original, "", "dry_run", True, False, 0.0, prompt_tokens, api_calls)
    repaired, repair_exact, repair_summary = _from_result(repair, original)
    if repair_exact:
        return _outcome(repaired, repair_summary, "repaired", True, cache_hit, latency, prompt_tokens, api_calls)
    repair_ids = _response_ids(repair.text)
    if valid_id_count(repair_ids, original) > 0:
        return _outcome(
            repaired,
            repair_summary or summary,
            "partial_completion",
            True,
            cache_hit,
            latency,
            prompt_tokens,
            api_calls,
        )
    return _outcome(ranked, summary, "partial_completion", True, cache_hit, latency, prompt_tokens, api_calls)


def _response_ids(text: str | None) -> list:
    if not text:
        return []
    try:
        payload = parse_json_object(text)
    except ValueError:
        return []
    ids = payload.get("ranked_movie_ids", [])
    return ids if isinstance(ids, list) else []


def _outcome(
    ranked: list[int],
    summary: str,
    reason: str,
    invalid: bool,
    cache_hit: bool,
    latency: float,
    prompt_tokens: int,
    api_calls: int,
) -> RerankOutcome:
    return RerankOutcome(
        ranked_movie_ids=ranked,
        reasoning_summary=summary,
        fallback_reason=reason,
        invalid_response=invalid,
        cache_hit=cache_hit,
        latency_s=latency,
        prompt_tokens=prompt_tokens,
        api_calls=api_calls,
    )
