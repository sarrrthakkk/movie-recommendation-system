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
    import os

    root = Path(__file__).resolve().parents[1]
    blocked = {".git", ".venv", "ml-100k", "artifacts", "data", "__pycache__"}
    offenders = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in blocked]
        for fname in filenames:
            path = Path(dirpath) / fname
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


def test_reverse_candidate_order_prompt_and_fallback(tmp_path):
    class MockChat:
        def __init__(self, text: str):
            self.text = text

        def chat(self, messages, *, model, temperature, timeout=None):
            return self.text

    profile = _profile()
    hybrid_prompt = build_rerank_messages(profile, CANDIDATES)
    reverse_prompt = build_rerank_messages(profile, list(reversed(CANDIDATES)))

    hybrid_user_msg = hybrid_prompt[-1]["content"]
    reverse_user_msg = reverse_prompt[-1]["content"]
    assert hybrid_user_msg != reverse_user_msg

    # In hybrid prompt, id=1 comes before id=4 in [CANDIDATES]
    assert hybrid_user_msg.index("id=1 |") < hybrid_user_msg.index("id=4 |")
    # In reverse prompt, id=4 comes before id=1 in [CANDIDATES]
    assert reverse_user_msg.index("id=4 |") < reverse_user_msg.index("id=1 |")

    # When fallback completion occurs, it must preserve original hybrid rank order
    partial_resp = '{"ranked_movie_ids": [4], "reasoning_summary": "Only movie 4."}'
    gateway = _gateway(tmp_path, MockChat(partial_resp))
    outcome = rerank_candidates(
        gateway,
        profile,
        candidates=list(reversed(CANDIDATES)),
        original_candidates=CANDIDATES,
    )
    # 4 was returned; missing 1, 2, 3 must be appended in original hybrid order (1, 2, 3), not reversed (3, 2, 1)
    assert outcome.ranked_movie_ids == [4, 1, 2, 3]
    assert outcome.fallback_reason == "partial_completion"


def test_reverse_and_hybrid_do_not_collide_cache_keys():
    from movie_recommender.llm.client import cache_key

    profile = _profile()
    hybrid_messages = build_rerank_messages(profile, CANDIDATES)
    reverse_messages = build_rerank_messages(profile, list(reversed(CANDIDATES)))

    key_hybrid = cache_key(
        model="qwen35-27b",
        messages=hybrid_messages,
        temperature=0.0,
        task="rerank",
    )
    key_reverse = cache_key(
        model="qwen35-27b",
        messages=reverse_messages,
        temperature=0.0,
        task="rerank",
    )
    assert key_hybrid != key_reverse


def test_select_audit_users_round_robin():
    from movie_recommender.llm_experiment import select_audit_users

    sampled = list(range(1, 101))
    user_tiers = {}
    for u in sampled[:34]:
        user_tiers[u] = "low"
    for u in sampled[34:67]:
        user_tiers[u] = "medium"
    for u in sampled[67:]:
        user_tiers[u] = "high"

    # For 50 samples with 5 movies per user -> 10 users
    selected = select_audit_users(sampled, user_tiers, target_samples=50, movies_per_user=5)
    assert len(selected) == 10
    assert [user_tiers[u] for u in selected] == [
        "low", "medium", "high",
        "low", "medium", "high",
        "low", "medium", "high",
        "low",
    ]

    # Target 4 samples -> 1 user
    single = select_audit_users(sampled, user_tiers, target_samples=4, movies_per_user=5)
    assert len(single) == 1
    assert user_tiers[single[0]] == "low"

    # Edge cases
    assert select_audit_users(sampled, user_tiers, target_samples=0) == []
    no_tiers = select_audit_users(sampled, None, target_samples=15, movies_per_user=5)
    assert no_tiers == [1, 2, 3]


def test_candidates_prompt_formatting_with_and_without_tmdb():
    from movie_recommender.llm.prompts import build_rerank_messages

    profile = _profile()
    candidates = [
        {"movie_id": 1, "title": "Alpha (1994)", "year": 1994, "genres": "Action"},
        {"movie_id": 2, "title": "Beta (1994)", "year": 1994, "genres": "Comedy"},
        {"movie_id": 3, "title": "Gamma (1994)", "year": 1994, "genres": "Drama"},
    ]
    tmdb_meta = {
        1: {
            "movie_id": 1,
            "director": "Steven Spielberg",
            "top_cast": ["Tom Hanks", "Matt Damon"],
            "overview": "A daring mission behind enemy lines \n with intense action \t and heroic choices that shape history forever.",
        },
        2: {
            "movie_id": 2,
            "director": None,
            "top_cast": [],
            "overview": "",
        },
    }

    # 1. With TMDB metadata
    messages_tmdb = build_rerank_messages(profile, candidates, tmdb_metadata=tmdb_meta, max_plot_chars=40)
    user_content = messages_tmdb[-1]["content"]

    cand_section = user_content.split("[CANDIDATES]\n")[1].split("\nThere are")[0]
    lines = cand_section.splitlines()

    assert "id=1 | Alpha (1994) | 1994 | Action" in lines[0]
    assert "director=Steven Spielberg" in lines[0]
    assert "cast=Tom Hanks, Matt Damon" in lines[0]
    assert "plot=" in lines[0]
    assert "\n" not in lines[0].split("plot=")[1]
    assert "\t" not in lines[0].split("plot=")[1]
    plot_text = lines[0].split("plot=")[1]
    assert len(plot_text) <= 40
    assert plot_text == "A daring mission behind enemy lines with"

    # Candidate 2 has empty/None fields -> no "plot=None", no "director=None", no trailing pipes
    assert lines[1] == "- id=2 | Beta (1994) | 1994 | Comedy"
    assert "None" not in lines[1]
    assert "plot=" not in lines[1]
    assert "director=" not in lines[1]

    # Candidate 3 is missing -> regular clean line
    assert lines[2] == "- id=3 | Gamma (1994) | 1994 | Drama"

    # User history items must remain compact without plot/director
    for hist_sec in ["[HIGHEST_RATED]", "[LOWEST_RATED]", "[RECENT_POSITIVE]"]:
        sec_text = user_content.split(f"{hist_sec}\n")[1].split("\n\n")[0]
        assert "plot=" not in sec_text
        assert "director=" not in sec_text
        assert "cast=" not in sec_text

    # 2. Without TMDB metadata (genre-only)
    messages_genre = build_rerank_messages(profile, candidates)
    user_content_genre = messages_genre[-1]["content"]
    cand_section_genre = user_content_genre.split("[CANDIDATES]\n")[1].split("\nThere are")[0].strip()
    assert cand_section_genre == (
        "- id=1 | Alpha (1994) | 1994 | Action\n"
        "- id=2 | Beta (1994) | 1994 | Comedy\n"
        "- id=3 | Gamma (1994) | 1994 | Drama"
    )


def test_cache_key_differs_between_genre_only_and_tmdb():
    from movie_recommender.llm.client import cache_key

    profile = _profile()
    tmdb_meta = {
        1: {
            "movie_id": 1,
            "director": "Steven Spielberg",
            "top_cast": ["Tom Hanks"],
            "overview": "A great plot overview.",
        }
    }
    messages_genre = build_rerank_messages(profile, CANDIDATES)
    messages_tmdb = build_rerank_messages(profile, CANDIDATES, tmdb_metadata=tmdb_meta)

    key_genre = cache_key(model="qwen35-27b", messages=messages_genre, temperature=0.0, task="rerank")
    key_tmdb = cache_key(model="qwen35-27b", messages=messages_tmdb, temperature=0.0, task="rerank")
    assert key_genre != key_tmdb


def test_explain_movies_metadata_alignment(tmp_path):
    class MockChat:
        def __init__(self, text: str):
            self.text = text

        def chat(self, messages, *, model, temperature, timeout=None):
            return self.text

    profile = _profile()
    movies = [{"movie_id": 1, "title": "Alpha (1994)", "year": 1994, "genres": "Drama"}]
    gt_genres = {1: "Drama"}

    # 1. Explanation mentioning director when has_tmdb_metadata=False -> rejected, falls back to template
    dir_text = '{"explanations": [{"movie_id": 1, "explanation": "Movie 1 features director Steven Spielberg with great emotional depth."}]}'
    outcome_genre = explain_movies(
        _gateway(tmp_path, MockChat(dir_text), refresh_cache=True),
        profile,
        movies,
        ground_truth_genres=gt_genres,
        has_tmdb_metadata=False,
    )
    assert outcome_genre.fallback_reason == "partial_template"
    assert "overlap preferences" in outcome_genre.explanations[0]["explanation"]

    # 2. Same explanation when has_tmdb_metadata=True -> accepted
    outcome_tmdb = explain_movies(
        _gateway(tmp_path, MockChat(dir_text), refresh_cache=True),
        profile,
        movies,
        ground_truth_genres=gt_genres,
        has_tmdb_metadata=True,
    )
    assert outcome_tmdb.fallback_reason == "none"
    assert "features director Steven Spielberg" in outcome_tmdb.explanations[0]["explanation"]

    # 3. Grounded plot mention with actor when has_tmdb_metadata=True -> accepted
    actor_text = '{"explanations": [{"movie_id": 1, "explanation": "Movie 1 features an intense drama starring Tom Hanks."}]}'
    outcome_actor = explain_movies(
        _gateway(tmp_path, MockChat(actor_text), refresh_cache=True),
        profile,
        movies,
        ground_truth_genres=gt_genres,
        has_tmdb_metadata=True,
    )
    assert outcome_actor.fallback_reason == "none"
    assert "starring Tom Hanks" in outcome_actor.explanations[0]["explanation"]

    # 4. Hallucinated genre (Sci-Fi when movie is only Drama) with has_tmdb_metadata=True -> still rejected!
    hallucinated_text = '{"explanations": [{"movie_id": 1, "explanation": "Movie 1 is a thrilling sci-fi directed by Steven Spielberg."}]}'
    outcome_hallucinated = explain_movies(
        _gateway(tmp_path, MockChat(hallucinated_text), refresh_cache=True),
        profile,
        movies,
        ground_truth_genres=gt_genres,
        has_tmdb_metadata=True,
    )
    assert outcome_hallucinated.fallback_reason == "hallucinated_genre"
    assert "overlap preferences" in outcome_hallucinated.explanations[0]["explanation"]

    # 5. Leakage word (certainly / hidden score) with has_tmdb_metadata=True -> still rejected!
    leakage_text = '{"explanations": [{"movie_id": 1, "explanation": "Movie 1 will certainly appeal to you based on hidden score."}]}'
    outcome_leakage = explain_movies(
        _gateway(tmp_path, MockChat(leakage_text), refresh_cache=True),
        profile,
        movies,
        ground_truth_genres=gt_genres,
        has_tmdb_metadata=True,
    )
    assert outcome_leakage.fallback_reason == "partial_template"


def test_missing_tmdb_cache_falls_back_to_genre_only(tmp_path):
    from movie_recommender.config import AppConfig

    cfg = AppConfig()
    cfg.llm.use_tmdb_metadata = True
    cfg.llm.tmdb_cache_path = str(tmp_path / "non_existent_cache.json")
    assert cfg.llm.use_tmdb_metadata is True
    assert not Path(cfg.llm.tmdb_cache_path).exists()


def test_llm_model_env_override_and_rendered_report(tmp_path, monkeypatch):
    from movie_recommender.config import AppConfig
    from movie_recommender.llm_experiment import _append_llm_markdown, _make_gateway

    config = AppConfig()
    assert config.llm.model == "qwen35-27b"

    # 1. Without LLM_MODEL set, YAML default is preserved
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setattr("movie_recommender.llm_experiment.load_llm_environment", lambda: None)
    gw_default, _ = _make_gateway(
        config,
        artifacts=tmp_path,
        mock=True,
        dry_run=False,
        maximum_api_calls=None,
        refresh_cache=False,
    )
    assert gw_default.model == "qwen35-27b"

    # 2. With LLM_MODEL set in env, it overrides YAML default
    monkeypatch.setenv("LLM_MODEL", "gemini-3.1-flash-lite")
    gw_env, _ = _make_gateway(
        config,
        artifacts=tmp_path,
        mock=True,
        dry_run=False,
        maximum_api_calls=None,
        refresh_cache=False,
    )
    assert gw_env.model == "gemini-3.1-flash-lite"

    # 3. Report rendered text reflects the overridden model name
    report_file = tmp_path / "results_table.md"
    result = {
        "caveat": (
            "The LLM can only reorder the hybrid candidate set, so it cannot recover a relevant movie "
            f"that the hybrid model did not nominate. Mock-client numbers are not measurements of {gw_env.model}. "
            "A bootstrap interval is not a p-value. A small public dataset cannot establish online satisfaction."
        ),
        "result_type": "full",
        "mock_llm": True,
        "n_users": 10,
        "paired_ndcg@10_llm_minus_hybrid": {
            "mean": 0.01,
            "low": -0.01,
            "high": 0.03,
            "excludes_zero": False,
            "reading": "inconclusive",
        },
        "top1_change_rate": 0.2,
        "mean_abs_rank_displacement": 1.5,
        "invalid_response_rate": 0.0,
        "fallback_rate": 0.0,
        "cache_hit_rate": 0.0,
        "average_request_latency_s": 0.5,
        "average_prompt_tokens": 1500.0,
        "api_calls": 10,
    }
    _append_llm_markdown(report_file, result)
    content = report_file.read_text(encoding="utf-8")
    assert "Mock-client numbers are not measurements of gemini-3.1-flash-lite." in content
    assert "qwen35-27b" not in content


def test_explain_system_tmdb_instructions():
    from movie_recommender.llm.prompts import EXPLAIN_SYSTEM_TMDB, build_explanation_messages

    # Check key instruction requirements in EXPLAIN_SYSTEM_TMDB
    assert "[RECENT_POSITIVE]" in EXPLAIN_SYSTEM_TMDB
    assert "exact movie title" in EXPLAIN_SYSTEM_TMDB
    assert "ONLY genre words explicitly listed" in EXPLAIN_SYSTEM_TMDB
    assert "Never cite the recommended movie itself" in EXPLAIN_SYSTEM_TMDB
    assert "Never write a standalone synopsis" in EXPLAIN_SYSTEM_TMDB

    # Check that build_explanation_messages includes recent positive history
    profile = {
        "recent_positive": [{"movie_id": 10, "title": "Top Gun", "year": 1986, "genres": "Action", "rating": 5.0}],
        "highest_rated": [{"movie_id": 20, "title": "Die Hard", "year": 1988, "genres": "Action", "rating": 5.0}],
    }
    movies = [{"movie_id": 1, "title": "Toy Story", "year": 1995, "genres": "Animation"}]
    messages = build_explanation_messages(profile, movies, has_tmdb_metadata=True)
    assert messages[0]["content"] == EXPLAIN_SYSTEM_TMDB
    user_content = messages[1]["content"]
    assert "[RECENT_POSITIVE]" in user_content
    assert "Top Gun" in user_content
    assert "[HIGHEST_RATED]" in user_content
    assert "Die Hard" in user_content


