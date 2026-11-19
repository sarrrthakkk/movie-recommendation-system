"""TMDB API client, title normalization, and offline metadata caching."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any, NamedTuple

import httpx

logger = logging.getLogger(__name__)


class ParsedTitle(NamedTuple):
    primary_title: str
    alternate_title: str | None
    year: int | None


def clean_trailing_article(s: str) -> str:
    """Move trailing grammatical articles to the front of a movie title."""
    s = s.strip()
    m = re.match(
        r"^(.*?),\s*(The|A|An|Le|La|Les|L'|Il|Das|Der|Die)$",
        s,
        re.IGNORECASE,
    )
    if m:
        base, art = m.group(1).strip(), m.group(2).strip()
        if art.lower() == "l'":
            art = "L'"
        else:
            art = art.capitalize()

        if art.endswith("'"):
            return f"{art}{base}"
        return f"{art} {base}"
    return s


def parse_movielens_title(raw_title: str, year: int | float | None = None) -> ParsedTitle:
    """Clean MovieLens 100K titles for TMDB search.

    Strips trailing video/format tags (e.g., '(V)') and year (e.g., '(1995)'),
    repositions trailing articles, and extracts parenthetical alternate titles.
    """
    text = raw_title.strip()
    # Strip trailing format/video tags like (V)
    text = re.sub(r"\s*\([Vv]\)\s*$", "", text).strip()

    # Determine year from argument if valid, or extract trailing (YYYY)
    parsed_year: int | None = None
    if year is not None:
        try:
            if not math.isnan(float(year)):
                parsed_year = int(year)
        except (ValueError, TypeError):
            pass

    m_year = re.search(r"\s*\((\d{4})\)\s*$", text)
    if m_year:
        if parsed_year is None:
            parsed_year = int(m_year.group(1))
        text = text[: m_year.start()].strip()

    # Check for alternate title in parentheses at the end of the text
    alt_title: str | None = None
    m_alt = re.search(r"\s*\(([^()]+)\)\s*$", text)
    if m_alt:
        alt_part = m_alt.group(1).strip()
        primary_part = text[: m_alt.start()].strip()
        primary_title = clean_trailing_article(primary_part)
        alt_title = clean_trailing_article(alt_part)
    else:
        primary_title = clean_trailing_article(text)

    return ParsedTitle(
        primary_title=primary_title,
        alternate_title=alt_title,
        year=parsed_year,
    )


class TMDBClient:
    """HTTP client for TMDB v3/v4 endpoints with rate limiting and exponential backoff."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.themoviedb.org/3",
        min_request_interval: float = 0.04,  # ~25 req/sec
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.api_key = (api_key or os.getenv("TMDB_API_KEY", "")).strip()
        self.base_url = base_url.rstrip("/")
        self.min_request_interval = min_request_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self._last_request_time: float = 0.0

        headers: dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "movie-recommender/0.1.0",
        }
        self.default_params: dict[str, str] = {}

        if self.api_key.startswith("eyJ"):
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.api_key:
            self.default_params["api_key"] = self.api_key

        self.client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            transport=transport,
            timeout=timeout,
        )

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> TMDBClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> httpx.Response:
        """Execute a GET request with rate-limiting and exponential backoff."""
        if not self.has_api_key:
            raise ValueError(
                "TMDB API key is not configured. Set TMDB_API_KEY environment variable or pass api_key."
            )

        # Normalize endpoint path
        clean_endpoint = endpoint.strip()
        if clean_endpoint.startswith("/3/"):
            clean_endpoint = clean_endpoint[2:]
        elif clean_endpoint.startswith("3/"):
            clean_endpoint = clean_endpoint[1:]
        if not clean_endpoint.startswith("/"):
            clean_endpoint = "/" + clean_endpoint

        merged_params = dict(self.default_params)
        if params:
            merged_params.update({k: v for k, v in params.items() if v is not None})

        backoff = 1.0
        resp: httpx.Response | None = None
        for attempt in range(self.max_retries + 1):
            elapsed = time.time() - self._last_request_time
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)

            self._last_request_time = time.time()
            try:
                resp = self.client.get(clean_endpoint, params=merged_params)
                if resp.status_code == 429 or resp.status_code >= 500:
                    if attempt < self.max_retries:
                        retry_after = resp.headers.get("Retry-After")
                        sleep_time = float(retry_after) if retry_after else backoff
                        time.sleep(sleep_time)
                        backoff *= 2.0
                        continue
                resp.raise_for_status()
                return resp
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                if attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                raise e

        if resp is not None:
            return resp
        raise RuntimeError(f"Failed to execute GET request for {clean_endpoint}")

    def search_movie(self, parsed: ParsedTitle) -> tuple[dict[str, Any] | None, str]:
        """Search TMDB cascading through primary title + year, primary title, and alternate title."""
        # 1. Primary title + year
        if parsed.year is not None:
            try:
                resp = self._get("/search/movie", {"query": parsed.primary_title, "year": parsed.year})
                results = resp.json().get("results", [])
                if results:
                    return results[0], "matched"
            except Exception as e:
                logger.debug(f"Search failed for '{parsed.primary_title}' ({parsed.year}): {e}")

        # 2. Primary title without year
        try:
            resp = self._get("/search/movie", {"query": parsed.primary_title})
            results = resp.json().get("results", [])
            if results:
                status = "matched" if parsed.year is None else "title_only_fallback"
                return results[0], status
        except Exception as e:
            logger.debug(f"Search failed for '{parsed.primary_title}': {e}")

        # 3. Alternate title
        if parsed.alternate_title:
            if parsed.year is not None:
                try:
                    resp = self._get("/search/movie", {"query": parsed.alternate_title, "year": parsed.year})
                    results = resp.json().get("results", [])
                    if results:
                        return results[0], "title_only_fallback"
                except Exception as e:
                    logger.debug(f"Search failed for alt '{parsed.alternate_title}' ({parsed.year}): {e}")

            try:
                resp = self._get("/search/movie", {"query": parsed.alternate_title})
                results = resp.json().get("results", [])
                if results:
                    return results[0], "title_only_fallback"
            except Exception as e:
                logger.debug(f"Search failed for alt '{parsed.alternate_title}': {e}")

        return None, "missing"

    def fetch_movie_details(self, tmdb_id: int) -> dict[str, Any]:
        """Fetch details and credits for a TMDB movie ID."""
        resp = self._get(f"/movie/{tmdb_id}", {"append_to_response": "credits"})
        return resp.json()

    def fetch_metadata_for_movie(
        self,
        movie_id: int,
        raw_title: str,
        year: int | float | None = None,
    ) -> dict[str, Any]:
        """Search and extract enriched metadata for a single MovieLens movie."""
        parsed = parse_movielens_title(raw_title, year)
        result, match_status = self.search_movie(parsed)

        if result is None:
            return {
                "movie_id": movie_id,
                "tmdb_id": None,
                "matched_title": parsed.primary_title,
                "overview": "",
                "director": None,
                "top_cast": [],
                "match_status": "missing",
            }

        tmdb_id = result.get("id")
        details: dict[str, Any] = {}
        if tmdb_id is not None:
            try:
                details = self.fetch_movie_details(int(tmdb_id))
            except Exception as e:
                logger.warning(f"Failed to fetch details for tmdb_id={tmdb_id} (movie_id={movie_id}): {e}")

        # Director extraction
        director: str | None = None
        crew = details.get("credits", {}).get("crew", [])
        for member in crew:
            if member.get("job") == "Director":
                director = member.get("name")
                break

        # Top cast extraction (up to 3)
        cast = details.get("credits", {}).get("cast", [])
        top_cast: list[str] = []
        for member in cast:
            name = member.get("name")
            if name:
                top_cast.append(name)
            if len(top_cast) == 3:
                break

        overview = (details.get("overview") or result.get("overview") or "").strip()
        matched_title = details.get("title") or result.get("title") or parsed.primary_title

        return {
            "movie_id": movie_id,
            "tmdb_id": tmdb_id,
            "matched_title": matched_title,
            "overview": overview,
            "director": director,
            "top_cast": top_cast,
            "match_status": match_status,
        }


def load_tmdb_metadata(path: str | Path) -> dict[int, dict[str, Any]]:
    """Load cached TMDB metadata JSON mapping movie_id (int) to metadata dict."""
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {int(k): v for k, v in data.items()}
    except Exception as e:
        logger.warning(f"Could not load TMDB metadata from {path}: {e}")
        return {}


def save_tmdb_metadata(path: str | Path, metadata: dict[int, dict[str, Any]]) -> None:
    """Atomically save TMDB metadata to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    serializable = {str(k): v for k, v in metadata.items()}
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    temp_path.replace(path)
