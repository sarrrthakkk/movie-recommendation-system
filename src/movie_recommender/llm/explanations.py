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

_GENRE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:sci[\s-]fi|science[\s-]fiction)\b", re.IGNORECASE), "Sci-Fi"),
    (re.compile(r"\bfilm[\s-]noir\b", re.IGNORECASE), "Film-Noir"),
    (re.compile(r"\bchildren(?:['’]s)?\b", re.IGNORECASE), "Children's"),
    (re.compile(r"\baction\b", re.IGNORECASE), "Action"),
    (re.compile(r"\badventure\b", re.IGNORECASE), "Adventure"),
    (re.compile(r"\banimation\b", re.IGNORECASE), "Animation"),
    (re.compile(r"\bcomedy\b", re.IGNORECASE), "Comedy"),
    (re.compile(r"\bcrime\b", re.IGNORECASE), "Crime"),
    (re.compile(r"\bdocumentary\b", re.IGNORECASE), "Documentary"),
    (re.compile(r"\bdrama\b", re.IGNORECASE), "Drama"),
    (re.compile(r"\bfantasy\b", re.IGNORECASE), "Fantasy"),
    (re.compile(r"\bhorror\b", re.IGNORECASE), "Horror"),
    (re.compile(r"\bmusical\b", re.IGNORECASE), "Musical"),
    (re.compile(r"\bmystery\b", re.IGNORECASE), "Mystery"),
    (re.compile(r"\bromance\b", re.IGNORECASE), "Romance"),
    (re.compile(r"\bthriller\b", re.IGNORECASE), "Thriller"),
    (re.compile(r"\bwar\b", re.IGNORECASE), "War"),
    (re.compile(r"\bwestern\b", re.IGNORECASE), "Western"),
]


def extract_mentioned_genres(text: str) -> set[str]:
    """Extract canonical MovieLens genre mentions from explanation text."""
    mentioned: set[str] = set()
    for pattern, canonical in _GENRE_PATTERNS:
        if pattern.search(text):
            mentioned.add(canonical)
    return mentioned


def parse_movie_genres(genres_obj: object) -> set[str]:
    """Parse movie genres into a set of canonical genre names."""
    if not genres_obj:
        return set()
    if isinstance(genres_obj, (list, set, tuple)):
        raw_list = [str(g).strip() for g in genres_obj]
    elif isinstance(genres_obj, str):
        raw_list = [p.strip() for p in re.split(r"[|,]", genres_obj) if p.strip()]
    else:
        return set()

    canonical: set[str] = set()
    for g in raw_list:
        low = g.lower()
        if low in {"sci-fi", "science fiction", "science-fiction", "sci fi"}:
            canonical.add("Sci-Fi")
        elif low in {"film-noir", "film noir"}:
            canonical.add("Film-Noir")
        elif low in {"children", "children's", "children’s"}:
            canonical.add("Children's")
        else:
            for _, c in _GENRE_PATTERNS:
                if low == c.lower():
                    canonical.add(c)
                    break
            else:
                canonical.add(g)
    return canonical


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


def explain_movies(
    gateway: LLMGateway,
    profile: dict,
    movies: list[dict],
    ground_truth_genres: dict[int, object] | None = None,
) -> ExplanationOutcome:
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
    hallucinated_genre = False
    for movie in movies:
        movie_id = int(movie["movie_id"])
        text = _first_sentence(by_id.get(movie_id, ""))
        is_invalid = (
            not text
            or str(movie_id) not in text
            or _FORBIDDEN.search(text)
            or movie_id not in by_id
        )
        is_hallucinated = False
        if not is_invalid:
            gt = (
                ground_truth_genres.get(movie_id)
                if (ground_truth_genres is not None and movie_id in ground_truth_genres)
                else movie.get("genres")
            )
            if gt is not None:
                valid_genres = parse_movie_genres(gt)
                mentioned = extract_mentioned_genres(text)
                if not mentioned.issubset(valid_genres):
                    is_invalid = True
                    is_hallucinated = True
        if is_invalid:
            text = templates[movie_id]
            used_template = True
            if is_hallucinated:
                hallucinated_genre = True
        rows.append({"movie_id": movie_id, "explanation": text})
    if hallucinated_genre:
        reason = "hallucinated_genre"
    elif used_template:
        reason = "partial_template"
    else:
        reason = "none"
    return ExplanationOutcome(rows, reason, int(result.api_call))


def _rows(templates: dict[int, str], movies: list[dict]) -> list[dict]:
    return [{"movie_id": int(movie["movie_id"]), "explanation": templates[int(movie["movie_id"])]} for movie in movies]
