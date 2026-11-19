"""LLM parsing, fallback, cache, and retry tests. These never call a network."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import httpx
import pytest

from movie_recommender.candidates import build_prompt_profile
from movie_recommender.data import EXPECTED_GENRE_DIM, GENRE_NAMES
from movie_recommender.llm.client import (
    LLMGateway,
    LLMRequestError,
    MockLLMClient,
    OpenAICompatibleClient,
    RateLimiter,
    SQLiteCache,
    TransientLLMError,
    request_interval_seconds,
)
from movie_recommender.llm.explanations import (
    explain_movies,
    extract_mentioned_genres,
    template_explanation,
)
from movie_recommender.llm.prompts import build_rerank_messages, prompt_token_estimate
from movie_recommender.llm.reranker import rerank_candidates
from movie_recommender.llm.schemas import complete_ranking, parse_json_object, parse_rerank
from movie_recommender.models.content_based import ContentModel
from movie_recommender.utils import RedactingFormatter
import numpy as np
import pandas as pd


CANDIDATES = [
    {"movie_id": 1, "title": "Alpha (1994)", "year": 1994, "genres": "Action"},
    {"movie_id": 2, "title": "Beta (1994)", "year": 1994, "genres": "Comedy"},
    {"movie_id": 3, "title": "Gamma (1994)", "year": 1994, "genres": "Drama"},
    {"movie_id": 4, "title": "Delta (1994)", "year": 1994, "genres": "Drama"},
]


def _gateway(tmp_path: Path, client, **kwargs) -> LLMGateway:
    defaults = dict(
        model="qwen35-27b",
        temperature=0.0,
        timeout=5,
        maximum_retries=3,
        retry_base_seconds=2,
        min_interval_seconds=0,
        sleeper=lambda _delay: None,
        rng=random.Random(0),
    )
    defaults.update(kwargs)
    return LLMGateway(client, SQLiteCache(tmp_path / "cache.sqlite"), **defaults)


def _profile() -> dict:
    return {
        "preferred_genres": [{"genre": "Drama", "share": 0.5}],
        "highest_rated": [{"movie_id": 10, "title": "Seen (1990)", "year": 1990, "genres": "Drama", "rating": 5}],
        "lowest_rated": [],
        "recent_positive": [],
    }


def test_json_fence_and_exact_permutation():
    text = '```json\n{"ranked_movie_ids": [4, 3, 2, 1], "reasoning_summary": "Drama first."}\n```'
    payload = parse_json_object(text)
    assert payload["ranked_movie_ids"] == [4, 3, 2, 1]
    ranked, exact, summary, error = parse_rerank(text, [1, 2, 3, 4])
    assert exact
    assert error == ""
    assert ranked == [4, 3, 2, 1]
    assert "Drama" in summary


def test_duplicate_and_invalid_ids_are_completed_in_hybrid_order():
    completed, exact, rejected = complete_ranking([4, 4, 99, 2], [1, 2, 3, 4])
    assert not exact
    assert completed == [4, 2, 1, 3]
    assert 99 in rejected


def test_missing_candidate_is_appended():
    completed, exact, _rejected = complete_ranking([1, 2, 3], [1, 2, 3, 4])
    assert not exact
    assert completed == [1, 2, 3, 4]


def test_request_failure_preserves_hybrid_order(tmp_path):
    class Broken:
        def chat(self, messages, *, model, temperature, timeout=None):
            raise LLMRequestError("HTTP 400")

    outcome = rerank_candidates(_gateway(tmp_path, Broken(), maximum_retries=1), _profile(), CANDIDATES)
    assert outcome.ranked_movie_ids == [1, 2, 3, 4]
    assert outcome.fallback_reason == "request_failed"


def test_invalid_then_repair_and_partial(tmp_path):
    class Scripted:
        def __init__(self, responses):
            self.responses = list(responses)

        def chat(self, messages, *, model, temperature, timeout=None):
            item = self.responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    good = json.dumps({"ranked_movie_ids": [4, 3, 2, 1], "reasoning_summary": "Repaired."})
    repaired = rerank_candidates(
        _gateway(tmp_path / "repair", Scripted(["not json", good])),
        _profile(),
        CANDIDATES,
    )
    assert repaired.fallback_reason == "repaired"
    assert repaired.ranked_movie_ids == [4, 3, 2, 1]
    assert repaired.invalid_response

    partial = rerank_candidates(
        _gateway(
            tmp_path / "partial",
            Scripted(['{"ranked_movie_ids": [4, 99], "reasoning_summary": "bad"}', "still bad"]),
        ),
        _profile(),
        CANDIDATES,
    )
    assert partial.fallback_reason == "partial_completion"
    assert partial.ranked_movie_ids[0] == 4
    assert sorted(partial.ranked_movie_ids) == [1, 2, 3, 4]


def test_retries_transient_failures_then_succeeds(tmp_path):
    delays = []
    client = MockLLMClient(
        handler=lambda messages, model, temperature: json.dumps(
            {"ranked_movie_ids": [1, 2, 3, 4], "reasoning_summary": "ok"}
        ),
        fail_times=2,
    )
    gateway = _gateway(
        tmp_path,
        client,
        retry_base_seconds=2,
        sleeper=delays.append,
        rng=random.Random(0),
    )
    result = gateway.generate(task="rerank", messages=[{"role": "user", "content": "hello"}])
    assert result.text is not None
    assert client.calls == 3
    assert len(delays) == 2
    assert delays[0] >= 2
    assert delays[1] >= 4


def test_cache_skips_second_call_and_stores_no_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "super-secret-test-key")
    calls = {"n": 0}

    def handler(messages, *, model, temperature):
        calls["n"] += 1
        return json.dumps({"ranked_movie_ids": [2, 1, 3, 4], "reasoning_summary": "cached"})

    gateway = _gateway(tmp_path, MockLLMClient(handler=handler))
    messages = [{"role": "user", "content": "same prompt"}]
    first = gateway.generate(task="rerank", messages=messages)
    second = gateway.generate(task="rerank", messages=messages)
    assert calls["n"] == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    stored = (tmp_path / "cache.sqlite").read_bytes()
    assert b"super-secret-test-key" not in stored


def test_dry_run_and_budget(tmp_path):
    client = MockLLMClient(handler=lambda messages, model, temperature: "unused")
    dry = _gateway(tmp_path, client, dry_run=True)
    result = dry.generate(task="rerank", messages=[{"role": "user", "content": "x"}])
    assert result.dry_run
    assert client.calls == 0
    assert dry.rate_limiter.waits == 0

    limited = _gateway(tmp_path, MockLLMClient(handler=lambda messages, model, temperature: "ok"), maximum_api_calls=1)
    limited.generate(task="rerank", messages=[{"role": "user", "content": "one"}])
    with pytest.raises(Exception):
        limited.generate(task="rerank", messages=[{"role": "user", "content": "two"}])


def test_rate_limit_floor_and_spacing():
    assert request_interval_seconds(6) >= 10
    assert request_interval_seconds(100) >= 10
    assert request_interval_seconds(1) >= 60

    class Clock:
        def __init__(self):
            self.t = 0.0

        def __call__(self):
            return self.t

    clock = Clock()
    slept = []

    def sleeper(delay):
        slept.append(delay)
        clock.t += delay

    limiter = RateLimiter(10, sleeper=sleeper, clock=clock)
    assert limiter.wait() == 0
    assert limiter.wait() >= 10
    assert slept[0] >= 10


def test_openai_adapter_maps_status_and_hides_key():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret-token"
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(400, json={"error": "bad"})
        return httpx.Response(404)

    client = OpenAICompatibleClient(
        "secret-token",
        "https://example.test/v1",
        timeout=5,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(LLMRequestError) as caught:
        client.chat([{"role": "user", "content": "hi"}], model="qwen35-27b", temperature=0)
    assert "secret-token" not in str(caught.value)
    assert "Authorization" not in str(caught.value)

    calls = {"n": 0}

    def retry_handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": "rate"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "{\"ok\": true}"}}]},
        )

    live = OpenAICompatibleClient(
        "secret-token",
        "https://example.test/v1",
        timeout=5,
        transport=httpx.MockTransport(retry_handler),
    )
    try:
        live.chat([{"role": "user", "content": "hi"}], model="qwen35-27b", temperature=0)
        raise AssertionError("429 should be transient")
    except TransientLLMError as exc:
        assert "secret-token" not in str(exc)
    text = live.chat([{"role": "user", "content": "hi"}], model="qwen35-27b", temperature=0)
    assert text == "{\"ok\": true}"
    assert calls["n"] == 2


def test_prompt_stays_small_and_omits_held_out_title():
    genres = np.zeros((6, EXPECTED_GENRE_DIM))
    genres[:, 8] = 1
    content = ContentModel().fit(np.arange(1, 7), genres)
    history = pd.DataFrame(
        {
            "user_id": [1] * 4,
            "movie_id": [1, 2, 3, 4],
            "rating": [5, 5, 4, 2],
            "timestamp": [1, 2, 3, 4],
        }
    )
    movies = pd.DataFrame(
        {
            "movie_id": np.arange(1, 7),
            "title": [f"History {i} (1995)" for i in range(1, 6)] + ["Held Out Title (1995)"],
            "year": [1995] * 6,
            "genres": ["Drama"] * 6,
        }
    )
    profile = build_prompt_profile(history, movies, content)
    candidates = [
        {"movie_id": 5, "title": "Candidate (1995)", "year": 1995, "genres": "Drama"},
    ]
    messages = build_rerank_messages(profile, candidates)
    text = messages[1]["content"]
    assert "Held Out Title" not in text
    assert "id=5" in text
    assert prompt_token_estimate(messages) < 8000
    _ = GENRE_NAMES


def test_explanation_template_when_generation_fails(tmp_path):
    class Broken:
        def chat(self, messages, *, model, temperature, timeout=None):
            raise LLMRequestError("HTTP 401")

    movies = [{"movie_id": 12, "title": "Toy Story (1995)", "genres": "Animation|Comedy"}]
    outcome = explain_movies(_gateway(tmp_path, Broken(), maximum_retries=0), _profile(), movies)
    assert outcome.fallback_reason == "request_failed"
    assert outcome.explanations[0]["explanation"] == template_explanation(12, "Toy Story (1995)", "Animation|Comedy")
    assert "actor" not in outcome.explanations[0]["explanation"].lower()


def test_log_formatter_redacts_api_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "super-secret-value")
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "key is super-secret-value", (), None)
    formatted = RedactingFormatter("%(message)s").format(record)
    assert "super-secret-value" not in formatted
    assert "[REDACTED]" in formatted


def test_tracked_files_do_not_contain_a_filled_api_key():
    root = Path(__file__).resolve().parents[1]
    blocked = {".git", ".venv", "ml-100k", "artifacts", "data", "__pycache__"}
    offenders = []
    for path in root.rglob("*"):
        if not path.is_file() or blocked.intersection(path.parts):
            continue
        if path.suffix.lower() not in {".py", ".md", ".yml", ".yaml", ".toml", ".txt", ".example", ".ipynb"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if line.startswith("LLM_API_KEY=") and line.strip() != "LLM_API_KEY=":
                offenders.append(str(path))
    assert offenders == []


def test_openai_client_max_tokens_handling():
    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client_512 = OpenAICompatibleClient(
        "test-key",
        "https://example.test/v1",
        timeout=5,
        transport=httpx.MockTransport(handler),
        max_tokens=512,
    )
    client_512.chat([{"role": "user", "content": "hi"}], model="qwen35-27b", temperature=0)
    assert captured[0]["max_tokens"] == 512

    client_none = OpenAICompatibleClient(
        "test-key",
        "https://example.test/v1",
        timeout=5,
        transport=httpx.MockTransport(handler),
        max_tokens=None,
    )
    client_none.chat([{"role": "user", "content": "hi"}], model="qwen35-27b", temperature=0)
    assert "max_tokens" not in captured[1]


def test_explanation_genre_validation(tmp_path):
    class MockChat:
        def __init__(self, text: str):
            self.text = text

        def chat(self, messages, *, model, temperature, timeout=None):
            return self.text

    valid_resp = json.dumps({
        "explanations": [
            {
                "movie_id": 12,
                "explanation": "Movie 12 (Toy Story) is recommended because the user enjoys animation and comedy.",
            }
        ]
    })
    movies_toy_story = [{"movie_id": 12, "title": "Toy Story (1995)", "genres": "Animation|Comedy"}]
    outcome_valid = explain_movies(_gateway(tmp_path / "valid", MockChat(valid_resp)), _profile(), movies_toy_story)
    assert outcome_valid.fallback_reason == "none"
    assert "animation and comedy" in outcome_valid.explanations[0]["explanation"]

    horror_resp = json.dumps({
        "explanations": [
            {
                "movie_id": 12,
                "explanation": "Movie 12 (Toy Story) is an intense thriller and horror experience.",
            }
        ]
    })
    outcome_horror = explain_movies(_gateway(tmp_path / "horror", MockChat(horror_resp)), _profile(), movies_toy_story)
    assert outcome_horror.fallback_reason == "hallucinated_genre"
    assert outcome_horror.explanations[0]["explanation"] == template_explanation(12, "Toy Story (1995)", "Animation|Comedy")

    scifi_resp = json.dumps({
        "explanations": [
            {
                "movie_id": 5,
                "explanation": "Movie 5 (Drama Film) explores science fiction themes.",
            }
        ]
    })
    movies_drama = [{"movie_id": 5, "title": "Drama Film (1995)", "genres": "Drama"}]
    outcome_scifi = explain_movies(_gateway(tmp_path / "scifi", MockChat(scifi_resp)), _profile(), movies_drama)
    assert outcome_scifi.fallback_reason == "hallucinated_genre"
    assert outcome_scifi.explanations[0]["explanation"] == template_explanation(5, "Drama Film (1995)", "Drama")

    unknown_resp = json.dumps({
        "explanations": [
            {
                "movie_id": 12,
                "explanation": "Movie 12 (Toy Story) has unknown charm but great animation.",
            }
        ]
    })
    outcome_unknown = explain_movies(_gateway(tmp_path / "unknown", MockChat(unknown_resp)), _profile(), movies_toy_story)
    assert outcome_unknown.fallback_reason == "none"
    assert "unknown charm" in outcome_unknown.explanations[0]["explanation"]


def test_extract_mentioned_genres_aliases():
    assert extract_mentioned_genres("A great sci-fi adventure") == {"Sci-Fi", "Adventure"}
    assert extract_mentioned_genres("Classic science fiction drama") == {"Sci-Fi", "Drama"}
    assert extract_mentioned_genres("Dark film-noir mystery") == {"Film-Noir", "Mystery"}
    assert extract_mentioned_genres("Dark film noir mystery") == {"Film-Noir", "Mystery"}
    assert extract_mentioned_genres("Fun for children's entertainment") == {"Children's"}
    assert extract_mentioned_genres("Fun for children everywhere") == {"Children's"}
    assert extract_mentioned_genres("An unknown plot with high tension") == set()
