"""One-sentence recommendation explanations with a template fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import ValidationError

from movie_recommender.llm.client import BudgetExceeded, LLMGateway, LLMRequestError
from movie_recommender.llm.prompts import build_explanation_messages
from movie_recommender.llm.schemas import ExplanationResponse, parse_json_object

_FORBIDDEN = re.compile(
    r"\b(actor|actress|director|starring|certainly|definitely|test label|hidden score|ndcg)\b|"
    r"i am sure|i know you will",
    re.IGNORECASE,
)


def template_explanation(movie_id: int, title: str, genres: str) -> str:
    genre_text = genres.replace("|", ", ") if genres else "no listed genres"
    safe_title = title.replace("|", "/")
    return (
        f"Movie {movie_id} ({safe_title}) is recommended because its listed genres "
        f"({genre_text}) overlap preferences in the supplied training history."
    )


def _first_sentence(text: str) -> str:
    stripped = " ".join(text.strip().split())
    if not stripped:
        return ""
    match = re.match(r"(.+?[.!?])(?:\s|$)", stripped)
    return match.group(1) if match else stripped


@dataclass
class ExplanationOutcome:
    explanations: list[dict]
    fallback_reason: str
    api_calls: int


def explain_movies(gateway: LLMGateway, profile: dict, movies: list[dict]) -> ExplanationOutcome:
    """Explain the supplied movies. Any failure becomes a deterministic template."""
    templates = {
        int(movie["movie_id"]): template_explanation(
            int(movie["movie_id"]), str(movie.get("title", "")), str(movie.get("genres", ""))
        )
        for movie in movies
    }
    messages = build_explanation_messages(profile, movies)
    try:
        result = gateway.generate(task="explain", messages=messages)
    except (BudgetExceeded, LLMRequestError):
        return ExplanationOutcome(_rows(templates, movies), "request_failed", 0)
    if result.dry_run or not result.text:
        return ExplanationOutcome(_rows(templates, movies), "dry_run" if result.dry_run else "request_failed", 0)
    try:
        payload = ExplanationResponse.model_validate(parse_json_object(result.text))
    except (ValueError, ValidationError):
        return ExplanationOutcome(_rows(templates, movies), "invalid_json", int(result.api_call))
    by_id = {int(item.movie_id): item.explanation for item in payload.explanations}
    rows: list[dict] = []
    used_template = False
    for movie in movies:
        movie_id = int(movie["movie_id"])
        text = _first_sentence(by_id.get(movie_id, ""))
        if (
            not text
            or str(movie_id) not in text
            or _FORBIDDEN.search(text)
            or movie_id not in by_id
        ):
            text = templates[movie_id]
            used_template = True
        rows.append({"movie_id": movie_id, "explanation": text})
    reason = "partial_template" if used_template else "none"
    return ExplanationOutcome(rows, reason, int(result.api_call))


def _rows(templates: dict[int, str], movies: list[dict]) -> list[dict]:
    return [{"movie_id": int(movie["movie_id"]), "explanation": templates[int(movie["movie_id"])]} for movie in movies]
