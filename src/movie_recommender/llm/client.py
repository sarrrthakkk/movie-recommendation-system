"""Provider-neutral LLM client, cache, retries, and rate limiting.

Add a new provider by implementing ``ChatClient.chat`` and passing that object
to ``LLMGateway``. ``OpenAICompatibleClient`` posts to ``{LLM_BASE_URL}/chat/completions``.
Set ``LLM_BASE_URL`` to the server's API root, for example a URL that already
ends in ``/v1``. This module does not choose a provider URL.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import httpx
from dotenv import load_dotenv

from movie_recommender.llm.prompts import PROMPT_VERSION
from movie_recommender.utils import get_logger, project_root

logger = get_logger(__name__)

MINIMUM_REQUEST_INTERVAL_SECONDS = 10.0


class TransientLLMError(Exception):
    """Timeouts, HTTP 429, and server errors. These may be retried."""


class LLMRequestError(Exception):
    """A non-retryable client or response error."""


class BudgetExceeded(Exception):
    """The uncached-call budget has been reached."""


class ChatClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        timeout: float | None = None,
    ) -> str:
        """Return the assistant text. Do not log authorization headers."""


def request_interval_seconds(requests_per_minute: int) -> float:
    """Enforce at least 10 seconds between uncached calls, and also the RPM cap."""
    if requests_per_minute < 1:
        raise ValueError("requests_per_minute must be at least 1")
    return max(MINIMUM_REQUEST_INTERVAL_SECONDS, 60.0 / requests_per_minute)


def chat_completions_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if not base:
        raise ValueError("LLM_BASE_URL is empty")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def load_llm_environment() -> None:
    env_path = project_root() / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def llm_credentials() -> tuple[str, str, str] | None:
    """Return ``(api_key, base_url, model)`` or None when LLM calls must be disabled."""
    load_llm_environment()
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    base_url = os.environ.get("LLM_BASE_URL", "").strip()
    model = os.environ.get("LLM_MODEL", "").strip() or "qwen35-27b"
    if not api_key or not base_url:
        return None
    return api_key, base_url, model


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text", "")))
        return "".join(parts)
    raise LLMRequestError("unexpected response shape: content")


class OpenAICompatibleClient:
    """POST an OpenAI-style chat-completions request.

    This is the only provider adapter. To support a different protocol, write
    another ``ChatClient`` and pass it to ``LLMGateway`` instead of this class.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key or not base_url:
            raise ValueError("API key and base URL are required")
        self._url = chat_completions_url(base_url)
        self._client = httpx.Client(
            timeout=timeout,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    def close(self) -> None:
        self._client.close()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        timeout: float | None = None,
    ) -> str:
        payload = {"model": model, "temperature": temperature, "messages": messages}
        limit = self._client.timeout if timeout is None else timeout
        prompt_chars = sum(len(message.get("content", "")) for message in messages)
        logger.info(
            "Sending chat completion model=%s timeout=%.0fs prompt_chars=%d",
            model,
            float(limit.read if hasattr(limit, "read") else limit),
            prompt_chars,
        )
        try:
            response = self._client.post(self._url, json=payload, timeout=timeout)
        except httpx.TimeoutException:
            raise TransientLLMError("request timed out") from None
        except httpx.TransportError as exc:
            raise TransientLLMError(f"transport error: {type(exc).__name__}") from None
        if response.status_code == 429 or response.status_code >= 500:
            raise TransientLLMError(f"HTTP {response.status_code}") from None
        if response.status_code >= 400:
            raise LLMRequestError(f"HTTP {response.status_code}") from None
        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError):
            raise LLMRequestError(f"HTTP {response.status_code} unexpected response shape") from None
        text = _content_to_text(content)
        logger.info("Chat completion HTTP %s response_chars=%d", response.status_code, len(text))
        return text


class MockLLMClient:
    """Deterministic stand-in used by tests and offline demonstrations.

    The mock reranks candidates by overlap between their genres and the genres
    listed in the prompt. It is not ``qwen35-27b`` and its metrics must not be
    reported as model quality.
    """

    def __init__(
        self,
        handler: Callable[..., str] | None = None,
        fail_times: int = 0,
    ) -> None:
        self.handler = handler or default_mock_handler
        self.fail_times = fail_times
        self.calls = 0

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        timeout: float | None = None,
    ) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise TransientLLMError("mock transient failure")
        return self.handler(messages, model=model, temperature=temperature)


def default_mock_handler(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float,
) -> str:
    system = messages[0]["content"] if messages else ""
    user = messages[-1]["content"] if messages else ""
    if "TASK: explain" in system:
        import re

        ids = [int(value) for value in re.findall(r"movie_id=(\d+)", user)]
        return json.dumps(
            {
                "explanations": [
                    {
                        "movie_id": movie_id,
                        "explanation": (
                            f"Movie {movie_id} is recommended because its listed genres "
                            "overlap the supplied training history."
                        ),
                    }
                    for movie_id in ids
                ]
            }
        )
    return json.dumps(_mock_rerank_payload(user))


def _section(text: str, name: str) -> str:
    import re

    match = re.search(rf"\[{name}\](.*?)(?:\n\[|\Z)", text, flags=re.S)
    return match.group(1) if match else ""


def _lines(section: str) -> list[tuple[int, str]]:
    import re

    found = []
    for movie_id, genres in re.findall(r"id=(\d+)\s*\|[^|]*\|[^|]*\|\s*([^|\n]+)", section):
        found.append((int(movie_id), genres))
    return found


def _mock_rerank_payload(user_text: str) -> dict:
    history = "\n".join(
        _section(user_text, name) for name in ("HIGHEST_RATED", "LOWEST_RATED", "RECENT_POSITIVE", "PREFERRED_GENRES")
    )
    history_genres = {genre.strip() for _, genres in _lines(history) for genre in genres.split(",") if genre.strip()}
    for token in _section(user_text, "PREFERRED_GENRES").replace(",", " ").split():
        if token and not token.startswith("(") and not token[0].isdigit():
            history_genres.add(token.strip("()"))
    candidates = _lines(_section(user_text, "CANDIDATES"))
    if not candidates and "[ALLOWED_IDS]" in user_text:
        import re

        allowed = [int(value) for value in re.findall(r"\d+", _section(user_text, "ALLOWED_IDS"))]
        candidates = [(movie_id, "") for movie_id in allowed]
    scored: list[tuple[int, int, int]] = []
    for position, (movie_id, genres) in enumerate(candidates):
        names = {genre.strip() for genre in genres.split(",") if genre.strip() and genre.strip() != "none"}
        overlap = len(names & history_genres)
        scored.append((overlap, -position, movie_id))
    scored.sort(reverse=True)
    ordered = [movie_id for _, _, movie_id in scored]
    return {
        "ranked_movie_ids": ordered,
        "reasoning_summary": "Mock reranker ordered candidates by genre overlap with the supplied history.",
    }


class RateLimiter:
    """Sleep so uncached requests start at least ``min_interval`` seconds apart."""

    def __init__(
        self,
        min_interval: float,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.min_interval = float(min_interval)
        self._sleeper = sleeper
        self._clock = clock
        self._last: float | None = None
        self.waits = 0

    def wait(self) -> float:
        now = self._clock()
        delay = 0.0
        if self._last is not None:
            delay = self.min_interval - (now - self._last)
            if delay > 0:
                self._sleeper(delay)
                self.waits += 1
        self._last = self._clock()
        return max(delay, 0.0)


class SQLiteCache:
    """Persistent cache of normalized responses. The API key is never written."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    task TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def get(self, key: str) -> str | None:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT response_text FROM llm_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        return None if row is None else str(row[0])

    def set(
        self,
        key: str,
        *,
        model: str,
        temperature: float,
        task: str,
        prompt_version: str,
        response_text: str,
    ) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO llm_cache
                (cache_key, model, temperature, task, prompt_version, response_text, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    model,
                    float(temperature),
                    task,
                    prompt_version,
                    response_text,
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                ),
            )


def cache_key(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    task: str,
    prompt_version: str = PROMPT_VERSION,
) -> str:
    blob = json.dumps(
        {
            "model": model,
            "temperature": temperature,
            "task": task,
            "prompt_version": prompt_version,
            "messages": messages,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class GenerationResult:
    text: str | None
    cache_hit: bool
    dry_run: bool
    latency_s: float
    wait_s: float
    error: str | None
    api_call: bool


class LLMGateway:
    """Cache, rate limit, retry, and budget control in front of a ``ChatClient``."""

    def __init__(
        self,
        client: ChatClient,
        cache: SQLiteCache,
        *,
        model: str,
        temperature: float,
        timeout: float,
        maximum_retries: int,
        retry_base_seconds: float,
        min_interval_seconds: float,
        maximum_api_calls: int | None = None,
        dry_run: bool = False,
        refresh_cache: bool = False,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        self.client = client
        self.cache = cache
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.maximum_retries = maximum_retries
        self.retry_base_seconds = retry_base_seconds
        self.maximum_api_calls = maximum_api_calls
        self.dry_run = dry_run
        self.refresh_cache = refresh_cache
        self.rate_limiter = RateLimiter(min_interval_seconds, sleeper=sleeper, clock=clock)
        self._sleeper = sleeper
        self._clock = clock
        self._rng = rng or random.Random()
        self._lock = threading.Lock()
        self.api_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.planned_calls = 0
        self.latencies: list[float] = []

    def is_cached(self, *, task: str, messages: list[dict[str, str]]) -> bool:
        key = cache_key(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            task=task,
            prompt_version=PROMPT_VERSION,
        )
        return self.cache.get(key) is not None

    def generate(self, *, task: str, messages: list[dict[str, str]]) -> GenerationResult:
        key = cache_key(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            task=task,
            prompt_version=PROMPT_VERSION,
        )
        with self._lock:
            if not self.refresh_cache:
                cached = self.cache.get(key)
                if cached is not None:
                    self.cache_hits += 1
                    return GenerationResult(
                        text=cached,
                        cache_hit=True,
                        dry_run=False,
                        latency_s=0.0,
                        wait_s=0.0,
                        error=None,
                        api_call=False,
                    )
            self.cache_misses += 1
            if self.dry_run:
                self.planned_calls += 1
                return GenerationResult(
                    text=None,
                    cache_hit=False,
                    dry_run=True,
                    latency_s=0.0,
                    wait_s=0.0,
                    error="dry_run",
                    api_call=False,
                )
            if self.maximum_api_calls is not None and self.api_calls >= self.maximum_api_calls:
                raise BudgetExceeded(f"API call budget of {self.maximum_api_calls} reached")
            wait_s = self.rate_limiter.wait()
            started = self._clock()
            text = self._call_with_retries(messages)
            latency = self._clock() - started
            self.api_calls += 1
            self.latencies.append(latency)
            self.cache.set(
                key,
                model=self.model,
                temperature=self.temperature,
                task=task,
                prompt_version=PROMPT_VERSION,
                response_text=text,
            )
            return GenerationResult(
                text=text,
                cache_hit=False,
                dry_run=False,
                latency_s=latency,
                wait_s=wait_s,
                error=None,
                api_call=True,
            )

    def _call_with_retries(self, messages: list[dict[str, str]]) -> str:
        last_error: Exception | None = None
        for attempt in range(self.maximum_retries + 1):
            if attempt > 0:
                # Retries are additional uncached requests, so they keep the same spacing.
                self.rate_limiter.wait()
                delay = self.retry_base_seconds * (2 ** (attempt - 1)) + self._rng.uniform(
                    0, self.retry_base_seconds
                )
                logger.info(
                    "Attempt %d failed (%s). Waiting %.1fs before retry %d of %d.",
                    attempt,
                    last_error,
                    delay,
                    attempt + 1,
                    self.maximum_retries + 1,
                )
                self._sleeper(delay)
            logger.info(
                "LLM attempt %d of %d",
                attempt + 1,
                self.maximum_retries + 1,
            )
            try:
                return self.client.chat(
                    messages,
                    model=self.model,
                    temperature=self.temperature,
                    timeout=self.timeout,
                )
            except TransientLLMError as exc:
                last_error = exc
                if attempt >= self.maximum_retries:
                    break
            except LLMRequestError:
                raise
        raise LLMRequestError(f"transient failure after retries: {last_error}") from None
