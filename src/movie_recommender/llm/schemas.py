"""Pydantic schemas and ranking checks for LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class RerankResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ranked_movie_ids: list[int]
    reasoning_summary: str = ""


class ExplanationItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    movie_id: int
    explanation: str = ""


class ExplanationResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    explanations: list[ExplanationItem] = Field(default_factory=list)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, including a fenced block or surrounding prose."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = _FENCE.sub("", cleaned).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Response does not contain a JSON object")
        payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("JSON response must be an object")
    return payload


def valid_id_count(returned: list[Any], candidates: list[int]) -> int:
    """Count unique candidate ids that can be kept from a model response."""
    allowed = {int(movie_id) for movie_id in candidates}
    seen: set[int] = set()
    for raw in returned:
        try:
            movie_id = int(raw)
        except (TypeError, ValueError):
            continue
        if movie_id in allowed and movie_id not in seen:
            seen.add(movie_id)
    return len(seen)


def complete_ranking(returned: list[Any], candidates: list[int]) -> tuple[list[int], bool, list[Any]]:
    """Keep valid unique ids, then append missing candidates in hybrid order.

    The boolean is true only when ``returned`` is exactly a permutation of
    ``candidates``.
    """
    allowed = set(int(movie_id) for movie_id in candidates)
    kept: list[int] = []
    seen: set[int] = set()
    rejected: list[Any] = []
    for raw in returned:
        try:
            movie_id = int(raw)
        except (TypeError, ValueError):
            rejected.append(raw)
            continue
        if movie_id not in allowed or movie_id in seen:
            rejected.append(movie_id)
            continue
        kept.append(movie_id)
        seen.add(movie_id)
    missing = [int(movie_id) for movie_id in candidates if int(movie_id) not in seen]
    exact = not rejected and not missing and len(kept) == len(candidates)
    return kept + missing, exact, rejected


def parse_rerank(text: str, candidates: list[int]) -> tuple[list[int], bool, str, str]:
    """Return completed ids, whether the response was exact, summary, and error."""
    try:
        payload = RerankResponse.model_validate(parse_json_object(text))
    except (ValueError, ValidationError, json.JSONDecodeError) as exc:
        completed, _, _ = complete_ranking([], candidates)
        return completed, False, "", f"invalid_json: {type(exc).__name__}"
    completed, exact, rejected = complete_ranking(payload.ranked_movie_ids, candidates)
    error = "" if exact else f"invalid_ranking rejected={len(rejected)}"
    summary = payload.reasoning_summary.strip()[:600]
    return completed, exact, summary, error
