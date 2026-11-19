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

EXPLAIN_SYSTEM_TMDB = """TASK: explain
Write one-sentence explanations for the listed movies.
Return JSON only:
{"explanations": [{"movie_id": <int>, "explanation": "<one sentence containing the movie id>"}]}
Rules:
- Explain only the supplied movie IDs.
- Each explanation is one sentence and includes the movie ID.
- Use only the supplied training history and movie metadata.
- Ground explanations in the supplied plot, director, cast, and genres. Do not mention unprovided facts.
- Do not claim certainty about the user.
- Do not mention scores, model internals, validation labels, or test labels.
"""


def truncate_plot(overview: str, max_chars: int = 220) -> str:
    """Collapse whitespace and truncate overview cleanly at a word boundary."""
    if not overview:
        return ""
    text = " ".join(overview.strip().split())
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    if text[max_chars] == " ":
        return truncated.rstrip()
    last_space = truncated.rfind(" ")
    if last_space > 0:
        return truncated[:last_space].rstrip()
    return truncated.rstrip()


def _format_meta_suffix(meta: dict | None, max_plot_chars: int = 220) -> str:
    if not meta:
        return ""
    parts: list[str] = []
    director = meta.get("director")
    if director:
        if isinstance(director, (list, tuple)):
            director_str = ", ".join(str(d).strip() for d in director if d).replace("|", "/")
        else:
            director_str = str(director).strip().replace("|", "/")
        if director_str and director_str != "None":
            parts.append(f"director={director_str}")

    top_cast = meta.get("top_cast") or meta.get("cast")
    if top_cast:
        if isinstance(top_cast, (list, tuple)):
            cast_str = ", ".join(str(c).strip() for c in top_cast if c).replace("|", "/")
        else:
            cast_str = str(top_cast).strip().replace("|", "/")
        if cast_str and cast_str != "None":
            parts.append(f"cast={cast_str}")

    overview = meta.get("overview") or meta.get("plot")
    if overview:
        plot_str = truncate_plot(str(overview), max_chars=max_plot_chars).replace("|", "/")
        if plot_str and plot_str != "None":
            parts.append(f"plot={plot_str}")

    if parts:
        return " | " + " | ".join(parts)
    return ""


def _fmt_movie(
    row: dict,
    include_rating: bool,
    meta: dict | None = None,
    max_plot_chars: int = 220,
) -> str:
    title = str(row.get("title", "")).replace("|", "/")
    year = "unknown" if row.get("year") in (None, "") else row["year"]
    genres = str(row.get("genres") or "none").replace("|", ", ")
    base = f"- id={int(row['movie_id'])} | {title} | {year} | {genres}"
    if include_rating:
        return f"{base} | rating={float(row['rating']):.1f}"
    if meta is None and any(k in row for k in ("director", "top_cast", "cast", "overview", "plot")):
        meta = row
    return f"{base}{_format_meta_suffix(meta, max_plot_chars)}"


def _section(
    name: str,
    rows: list[dict],
    include_rating: bool,
    tmdb_metadata: dict[int, dict] | None = None,
    max_plot_chars: int = 220,
) -> str:
    if not rows:
        return f"[{name}]\n- none"
    lines = []
    for row in rows:
        meta = None
        if not include_rating and tmdb_metadata is not None:
            meta = tmdb_metadata.get(int(row["movie_id"]))
        lines.append(_fmt_movie(row, include_rating, meta=meta, max_plot_chars=max_plot_chars))
    return f"[{name}]\n" + "\n".join(lines)


def build_rerank_messages(
    profile: dict,
    candidates: list[dict],
    tmdb_metadata: dict[int, dict] | None = None,
    max_plot_chars: int = 220,
) -> list[dict[str, str]]:
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
            _section(
                "CANDIDATES",
                candidates,
                False,
                tmdb_metadata=tmdb_metadata,
                max_plot_chars=max_plot_chars,
            ),
            f"There are {len(candidates)} candidates. Return each candidate id exactly once.",
        ]
    )
    return [
        {"role": "system", "content": RERANK_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_repair_messages(
    profile: dict,
    candidates: list[dict],
    invalid_response: str,
    tmdb_metadata: dict[int, dict] | None = None,
    max_plot_chars: int = 220,
) -> list[dict[str, str]]:
    allowed = ", ".join(str(int(row["movie_id"])) for row in candidates)
    user = "\n\n".join(
        [
            "Fix the ranking.",
            f"[ALLOWED_IDS]\n{allowed}",
            _section(
                "CANDIDATES",
                candidates,
                False,
                tmdb_metadata=tmdb_metadata,
                max_plot_chars=max_plot_chars,
            ),
            "[INVALID_RESPONSE]\n" + invalid_response[:2000],
        ]
    )
    return [
        {"role": "system", "content": REPAIR_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_explanation_messages(
    profile: dict,
    movies: list[dict],
    tmdb_metadata: dict[int, dict] | None = None,
    max_plot_chars: int = 220,
    has_tmdb_metadata: bool = False,
) -> list[dict[str, str]]:
    has_meta = has_tmdb_metadata or bool(tmdb_metadata)
    lines = []
    for row in movies:
        m_id = int(row["movie_id"])
        meta = tmdb_metadata.get(m_id) if tmdb_metadata else None
        if not meta and any(k in row for k in ("director", "top_cast", "cast", "overview", "plot")):
            meta = row
            has_meta = True
        line = (
            f"- movie_id={m_id} | {str(row.get('title', '')).replace('|', '/')} | "
            f"{row.get('year') or 'unknown'} | {str(row.get('genres') or 'none').replace('|', ', ')}"
        )
        suffix = _format_meta_suffix(meta, max_plot_chars)
        lines.append(f"{line}{suffix}")

    requested = "\n".join(lines)
    history = _section("HIGHEST_RATED", profile.get("highest_rated", []), True)
    user = "\n\n".join(
        [
            "Explain these recommendations.",
            history,
            "[MOVIES]\n" + requested,
        ]
    )
    system_prompt = EXPLAIN_SYSTEM_TMDB if has_meta else EXPLAIN_SYSTEM
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def prompt_token_estimate(messages: list[dict[str, str]]) -> int:
    return estimate_tokens("\n".join(message.get("content", "") for message in messages))
