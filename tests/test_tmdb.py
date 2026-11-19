"""Unit tests for TMDB title parsing, API client, search fallback, and caching."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from movie_recommender.tmdb import (
    ParsedTitle,
    TMDBClient,
    clean_trailing_article,
    load_tmdb_metadata,
    parse_movielens_title,
    save_tmdb_metadata,
)


def test_clean_trailing_article():
    assert clean_trailing_article("Godfather, The") == "The Godfather"
    assert clean_trailing_article("Quiet Man, A") == "A Quiet Man"
    assert clean_trailing_article("American in Paris, An") == "An American in Paris"
    assert clean_trailing_article("Samouraï, Le") == "Le Samouraï"
    assert clean_trailing_article("Femme Nikita, La") == "La Femme Nikita"
    assert clean_trailing_article("Misérables, Les") == "Les Misérables"
    assert clean_trailing_article("Atalante, L'") == "L'Atalante"
    assert clean_trailing_article("Postino, Il") == "Il Postino"
    assert clean_trailing_article("Boot, Das") == "Das Boot"
    assert clean_trailing_article("Blaue Engel, Der") == "Der Blaue Engel"
    assert clean_trailing_article("Blechtrommel, Die") == "Die Blechtrommel"
    # Non-article commas must remain unchanged
    assert clean_trailing_article("Corrina, Corrina") == "Corrina, Corrina"
    assert clean_trailing_article("Paris, Texas") == "Paris, Texas"


def test_parse_movielens_title_edge_cases():
    cases = [
        ("Godfather, The (1972)", ParsedTitle("The Godfather", None, 1972)),
        ("Silence of the Lambs, The (1991)", ParsedTitle("The Silence of the Lambs", None, 1991)),
        ("Postino, Il (1994)", ParsedTitle("Il Postino", None, 1994)),
        ("Seven (Se7en) (1995)", ParsedTitle("Seven", "Se7en", 1995)),
        (
            "Land Before Time III: The Time of the Great Giving (1995) (V)",
            ParsedTitle("Land Before Time III: The Time of the Great Giving", None, 1995),
        ),
        (
            "Horseman on the Roof, The (Hussard sur le toit, Le) (1995)",
            ParsedTitle("The Horseman on the Roof", "Le Hussard sur le toit", 1995),
        ),
        (
            "Star Maker, The (Uomo delle stelle, L') (1995)",
            ParsedTitle("The Star Maker", "L'Uomo delle stelle", 1995),
        ),
        ("Boot, Das (1981)", ParsedTitle("Das Boot", None, 1981)),
        ("Blaue Engel, Der (1930)", ParsedTitle("Der Blaue Engel", None, 1930)),
        ("Blechtrommel, Die (1979)", ParsedTitle("Die Blechtrommel", None, 1979)),
        ("Misérables, Les (1995)", ParsedTitle("Les Misérables", None, 1995)),
        ("unknown", ParsedTitle("unknown", None, None)),
    ]
    for raw, expected in cases:
        parsed = parse_movielens_title(raw)
        assert parsed == expected

    # Year argument override / float handling
    assert parse_movielens_title("Toy Story", 1995.0) == ParsedTitle("Toy Story", None, 1995)
    assert parse_movielens_title("Toy Story (1995)", 1996) == ParsedTitle("Toy Story", None, 1996)


def test_tmdb_auth_mode_selection():
    # 1. v3 query param auth
    v3_requests: list[httpx.Request] = []

    def v3_handler(request: httpx.Request) -> httpx.Response:
        v3_requests.append(request)
        return httpx.Response(200, json={"results": []})

    client_v3 = TMDBClient(
        api_key="v3_dummy_key",
        transport=httpx.MockTransport(v3_handler),
        min_request_interval=0.0,
    )
    client_v3.search_movie(ParsedTitle("Toy Story", None, 1995))
    assert len(v3_requests) > 0
    assert v3_requests[0].url.params.get("api_key") == "v3_dummy_key"
    assert "Authorization" not in v3_requests[0].headers

    # 2. v4 Bearer token auth (starts with eyJ)
    v4_requests: list[httpx.Request] = []

    def v4_handler(request: httpx.Request) -> httpx.Response:
        v4_requests.append(request)
        return httpx.Response(200, json={"results": []})

    client_v4 = TMDBClient(
        api_key="eyJdummy_jwt_token",
        transport=httpx.MockTransport(v4_handler),
        min_request_interval=0.0,
    )
    client_v4.search_movie(ParsedTitle("Toy Story", None, 1995))
    assert len(v4_requests) > 0
    assert "api_key" not in v4_requests[0].url.params
    assert v4_requests[0].headers.get("Authorization") == "Bearer eyJdummy_jwt_token"


def test_mocked_search_cascade_and_credits_extraction():
    searches: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "search/movie" in url_str:
            searches.append(url_str)
            # Step 1: primary with year misses
            if "year=" in url_str:
                return httpx.Response(200, json={"results": []})
            # Step 2: primary without year hits
            return httpx.Response(
                200,
                json={"results": [{"id": 101, "title": "Fallback Search Title", "overview": "Search overview"}]},
            )
        elif "movie/101" in url_str:
            return httpx.Response(
                200,
                json={
                    "id": 101,
                    "title": "Godfather Detailed",
                    "overview": "The aging patriarch of an organized crime dynasty transfers control to his reluctant son.",
                    "credits": {
                        "cast": [
                            {"name": "Marlon Brando"},
                            {"name": "Al Pacino"},
                            {"name": "James Caan"},
                            {"name": "Robert Duvall"},
                        ],
                        "crew": [
                            {"job": "Producer", "name": "Albert S. Ruddy"},
                            {"job": "Director", "name": "Francis Ford Coppola"},
                        ],
                    },
                },
            )
        return httpx.Response(404)

    client = TMDBClient(
        api_key="test_key",
        transport=httpx.MockTransport(handler),
        min_request_interval=0.0,
    )
    meta = client.fetch_metadata_for_movie(1, "Godfather, The (1972)")

    assert len(searches) >= 2
    assert "year=1972" in searches[0]
    assert "year=" not in searches[1]
    assert meta["movie_id"] == 1
    assert meta["tmdb_id"] == 101
    assert meta["match_status"] == "title_only_fallback"
    assert meta["matched_title"] == "Godfather Detailed"
    assert "aging patriarch" in meta["overview"]
    assert meta["director"] == "Francis Ford Coppola"
    # Top cast capped at 3
    assert meta["top_cast"] == ["Marlon Brando", "Al Pacino", "James Caan"]


def test_mocked_search_alternate_title_fallback():
    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "search/movie" in url_str:
            if "query=Seven" in url_str:
                return httpx.Response(200, json={"results": []})
            if "query=Se7en" in url_str:
                return httpx.Response(
                    200,
                    json={"results": [{"id": 202, "title": "Se7en", "overview": "Two detectives hunt a serial killer."}]},
                )
        elif "movie/202" in url_str:
            return httpx.Response(
                200,
                json={
                    "id": 202,
                    "title": "Se7en",
                    "overview": "Two detectives hunt a serial killer.",
                    "credits": {
                        "cast": [{"name": "Brad Pitt"}, {"name": "Morgan Freeman"}],
                        "crew": [{"job": "Director", "name": "David Fincher"}],
                    },
                },
            )
        return httpx.Response(404)

    client = TMDBClient(
        api_key="test_key",
        transport=httpx.MockTransport(handler),
        min_request_interval=0.0,
    )
    meta = client.fetch_metadata_for_movie(11, "Seven (Se7en) (1995)")

    assert meta["movie_id"] == 11
    assert meta["tmdb_id"] == 202
    assert meta["match_status"] == "title_only_fallback"
    assert meta["director"] == "David Fincher"
    assert meta["top_cast"] == ["Brad Pitt", "Morgan Freeman"]


def test_mocked_search_missing_movie():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    client = TMDBClient(
        api_key="test_key",
        transport=httpx.MockTransport(handler),
        min_request_interval=0.0,
    )
    meta = client.fetch_metadata_for_movie(999, "Nonexistent Obscure Film (1990)")

    assert meta["movie_id"] == 999
    assert meta["tmdb_id"] is None
    assert meta["match_status"] == "missing"
    assert meta["matched_title"] == "Nonexistent Obscure Film"
    assert meta["overview"] == ""
    assert meta["director"] is None
    assert meta["top_cast"] == []


def test_load_save_tmdb_metadata_roundtrip_and_missing(tmp_path: Path):
    non_existent = tmp_path / "does_not_exist.json"
    assert load_tmdb_metadata(non_existent) == {}

    data = {
        1: {
            "movie_id": 1,
            "tmdb_id": 862,
            "matched_title": "Toy Story",
            "overview": "Led by Woody, Andy's toys live happily...",
            "director": "John Lasseter",
            "top_cast": ["Tom Hanks", "Tim Allen", "Don Rickles"],
            "match_status": "matched",
        },
        2: {
            "movie_id": 2,
            "tmdb_id": 710,
            "matched_title": "GoldenEye",
            "overview": "James Bond teams up with the sole survivor...",
            "director": "Martin Campbell",
            "top_cast": ["Pierce Brosnan", "Sean Bean", "Izabella Scorupco"],
            "match_status": "matched",
        },
    }

    cache_file = tmp_path / "metadata.json"
    save_tmdb_metadata(cache_file, data)
    assert cache_file.is_file()

    loaded = load_tmdb_metadata(cache_file)
    assert set(loaded.keys()) == {1, 2}
    assert loaded[1]["matched_title"] == "Toy Story"
    assert loaded[2]["director"] == "Martin Campbell"

    # Malformed file handling
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("invalid json contents {", encoding="utf-8")
    assert load_tmdb_metadata(bad_file) == {}


def test_incremental_resume_skips_cached(tmp_path: Path):
    cache_file = tmp_path / "cache.json"
    existing = {
        1: {"movie_id": 1, "tmdb_id": 862, "matched_title": "Toy Story", "match_status": "matched"}
    }
    save_tmdb_metadata(cache_file, existing)

    fetched_ids: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "movie/2" in url_str:
            return httpx.Response(200, json={"id": 2, "title": "GoldenEye", "overview": "Bond."})
        return httpx.Response(200, json={"results": [{"id": 2, "title": "GoldenEye"}]})

    client = TMDBClient(
        api_key="test_key",
        transport=httpx.MockTransport(handler),
        min_request_interval=0.0,
    )

    # Simulate movies to process: [1, 2]
    all_movies = [(1, "Toy Story (1995)", 1995), (2, "GoldenEye (1995)", 1995)]
    cached = load_tmdb_metadata(cache_file)
    to_fetch = [m for m in all_movies if m[0] not in cached]

    assert len(to_fetch) == 1
    assert to_fetch[0][0] == 2

    for mid, title, year in to_fetch:
        fetched_ids.append(mid)
        cached[mid] = client.fetch_metadata_for_movie(mid, title, year)

    assert fetched_ids == [2]
    save_tmdb_metadata(cache_file, cached)

    reloaded = load_tmdb_metadata(cache_file)
    assert set(reloaded.keys()) == {1, 2}


def test_tmdb_rate_limiting_and_retry():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0.01"})
        return httpx.Response(200, json={"results": [{"id": 99, "title": "Retry Success"}]})

    client = TMDBClient(
        api_key="test_key",
        transport=httpx.MockTransport(handler),
        min_request_interval=0.0,
    )
    result, status = client.search_movie(ParsedTitle("Retry Film", None, None))
    assert calls == 2
    assert result is not None
    assert result["title"] == "Retry Success"
