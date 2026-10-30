"""Prompt templates. Bump PROMPT_VERSION whenever the text changes."""

from __future__ import annotations

from movie_recommender.utils import estimate_tokens

PROMPT_VERSION = "v1"
MAX_RECENT_POSITIVE = 5
TOKEN_WARNING_THRESHOLD = 8000

RERANK_SYSTEM = """TASK: rerank
You rerank a fixed set of movie candidates for one user.
Return JSON only, with exactly this structure:
{"ranked_movie_ids": [<movie id>, ...], "reasoning_summary": "<one or two short sentences>"}
Rules:
- Use only candidate movie IDs from the [CANDIDATES] section.
- Do not add movies.
- Include every candidate ID exactly once.
- ranked_movie_ids must be a permutation of the candidate IDs.
- reasoning_summary is a short user-facing rationale, not private chain-of-thought.
- Do not mention hidden model scores, test labels, validation labels, or future ratings.
- Use only the supplied training history and candidate metadata.
"""

REPAIR_SYSTEM = """TASK: repair
The previous ranking was invalid. Return corrected JSON only:
{"ranked_movie_ids": [<movie id>, ...], "reasoning_summary": "<one or two short sentences>"}
ranked_movie_ids must contain each allowed candidate ID exactly once and no other IDs.
"""

EXPLAIN_SYSTEM = """TASK: explain
Write one-sentence explanations for the listed movies.
Return JSON only:
{"explanations": [{"movie_id": <int>, "explanation": "<one sentence containing the movie id>"}]}
Rules:
- Explain only the supplied movie IDs.
- Each explanation is one sentence and includes the movie ID.
- Use only the supplied training history and movie metadata.
- Do not mention actors, directors, writers, or themes that were not provided.
- Do not claim certainty about the user.
- Do not mention scores, model internals, validation labels, or test labels.
"""


def _fmt_movie(row: dict, include_rating: bool) -> str:
    title = str(row.get("title", "")).replace("|", "/")
    year = "unknown" if row.get("year") in (None, "") else row["year"]
    genres = str(row.get("genres") or "none").replace("|", ", ")
    base = f"- id={int(row['movie_id'])} | {title} | {year} | {genres}"
    if include_rating:
        return f"{base} | rating={float(row['rating']):.1f}"
    return base


def _section(name: str, rows: list[dict], include_rating: bool) -> str:
    if not rows:
        return f"[{name}]\n- none"
    lines = "\n".join(_fmt_movie(row, include_rating) for row in rows)
    return f"[{name}]\n{lines}"


def build_rerank_messages(profile: dict, candidates: list[dict]) -> list[dict[str, str]]:
    preferred = profile.get("preferred_genres") or []
    if preferred:
        preferred_text = ", ".join(f"{item['genre']} ({item['share']:.2f})" for item in preferred)
    else:
        preferred_text = "none"
    user = "\n\n".join(
        [
            "Rank the candidate movies for this user.",
            f"[PREFERRED_GENRES]\n{preferred_text}",
            _section("HIGHEST_RATED", profile.get("highest_rated", []), True),
            _section("LOWEST_RATED", profile.get("lowest_rated", []), True),
            _section("RECENT_POSITIVE", profile.get("recent_positive", []), True),
            _section("CANDIDATES", candidates, False),
            f"There are {len(candidates)} candidates. Return each candidate id exactly once.",
        ]
    )
    return [
        {"role": "system", "content": RERANK_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_repair_messages(profile: dict, candidates: list[dict], invalid_response: str) -> list[dict[str, str]]:
    allowed = ", ".join(str(int(row["movie_id"])) for row in candidates)
    user = "\n\n".join(
        [
            "Fix the ranking.",
            f"[ALLOWED_IDS]\n{allowed}",
            _section("CANDIDATES", candidates, False),
            "[INVALID_RESPONSE]\n" + invalid_response[:2000],
        ]
    )
    return [
        {"role": "system", "content": REPAIR_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_explanation_messages(profile: dict, movies: list[dict]) -> list[dict[str, str]]:
    requested = "\n".join(
        f"- movie_id={int(row['movie_id'])} | {str(row.get('title', '')).replace('|', '/')} | "
        f"{row.get('year') or 'unknown'} | {str(row.get('genres') or 'none').replace('|', ', ')}"
        for row in movies
    )
    history = _section("HIGHEST_RATED", profile.get("highest_rated", []), True)
    user = "\n\n".join(
        [
            "Explain these recommendations.",
            history,
            "[MOVIES]\n" + requested,
        ]
    )
    return [
        {"role": "system", "content": EXPLAIN_SYSTEM},
        {"role": "user", "content": user},
    ]


def prompt_token_estimate(messages: list[dict[str, str]]) -> int:
    return estimate_tokens("\n".join(message.get("content", "") for message in messages))
