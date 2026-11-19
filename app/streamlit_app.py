"""Small demo for cold-start and existing-user recommendations.

The non-LLM methods run without credentials. Live reranking is off unless the
operator selects it and checks the confirmation box.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from movie_recommender.config import AppConfig
from movie_recommender.demo import (
    explain_recommendations,
    load_demo_resources,
    recommend_existing_user,
    recommend_new_user,
)
from movie_recommender.llm.client import (
    LLMGateway,
    MockLLMClient,
    OpenAICompatibleClient,
    SQLiteCache,
    llm_credentials,
    request_interval_seconds,
)
from movie_recommender.utils import project_root

st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")
st.title("Hybrid Movie Recommender")
st.caption("Matrix factorization, genre content, and optional LLM reranking on MovieLens 100K.")


@st.cache_resource
def resources():
    config = AppConfig.from_yaml(project_root() / "configs" / "default.yaml")
    return config, load_demo_resources(config)


try:
    config, bundle = resources()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Run `python scripts/run_pipeline.py --config configs/default.yaml --data-dir ml-100k --smoke` first.")
    st.stop()

if bundle.smoke:
    st.info("Loaded artifacts came from a smoke training run, not the full epoch budget.")

credentials = llm_credentials()
if credentials is None:
    st.warning("LLM_API_KEY or LLM_BASE_URL is missing. LLM reranking is disabled and the other methods still work.")

mode = st.sidebar.radio("User", ["New user (cold start)", "Existing MovieLens user"])
method = st.sidebar.selectbox(
    "Recommender",
    ["popularity", "content", "hybrid", "hybrid_llm"],
    format_func=lambda name: {
        "popularity": "Popularity",
        "content": "Content-based",
        "hybrid": "Hybrid",
        "hybrid_llm": "Hybrid plus LLM reranking",
    }[name],
)
top_k = st.sidebar.slider("How many recommendations", 5, 20, 10)
use_mock = st.sidebar.checkbox("Use mock LLM (no API call)", value=True)
confirm_live = st.sidebar.checkbox("I confirm a live API call", value=False)
explain = st.sidebar.checkbox("Add explanations", value=True)


def build_gateway(live: bool):
    """Return a mock gateway, a confirmed live gateway, or None when live use is blocked.

    Live demo calls use the model from the environment. The measured rerank
    latency is about one minute, and a few calls approached 100 seconds, so the
    demo allows 180 seconds and one retry instead of four 100-second attempts.
    """
    cache = SQLiteCache(project_root() / "artifacts" / "cache" / "llm_cache.sqlite")
    if live:
        if credentials is None or not confirm_live:
            return None
        api_key, base_url, model_name = credentials
        timeout = 180.0
        retries = 1
        client = OpenAICompatibleClient(api_key, base_url, timeout=timeout, max_tokens=config.llm.max_tokens)
        interval = request_interval_seconds(config.llm.requests_per_minute)
    else:
        model_name = config.llm.model
        timeout = config.llm.request_timeout_seconds
        retries = config.llm.maximum_retries
        client = MockLLMClient()
        interval = 0.0
    return LLMGateway(
        client,
        cache,
        model=model_name,
        temperature=config.llm.temperature,
        timeout=timeout,
        maximum_retries=retries,
        retry_base_seconds=config.llm.retry_base_seconds,
        min_interval_seconds=interval,
        maximum_api_calls=4,
        dry_run=False,
    )


def show_recommendations(rows, warnings, summary, heading):
    """Render one recommendation result. Return whether any rows were shown."""
    st.subheader(heading)
    for message in warnings:
        st.warning(message)
    if summary:
        st.write(summary)
    if not rows:
        return False
    if rows[0]["cold_start"]:
        st.info("These are cold-start recommendations. The matrix-factorization column is empty on purpose.")
    frame = pd.DataFrame(rows)
    st.dataframe(
        frame[
            [
                "rank",
                "movie_id",
                "title",
                "year",
                "genres",
                "score",
                "mf_score",
                "content_score",
                "popularity_score",
                "hybrid_score",
            ]
        ],
        hide_index=True,
    )
    return True


if mode.startswith("New"):
    st.subheader("Rate a few movies")
    query = st.text_input("Search by title", value="Toy")
    matches = bundle.movies
    if query.strip():
        matches = matches.loc[matches["title"].str.contains(query.strip(), case=False, na=False, regex=False)]
    matches = matches.head(30)
    labels = {f"{row.title} ({int(row.movie_id)})": int(row.movie_id) for row in matches.itertuples(index=False)}
    if "basket" not in st.session_state:
        st.session_state.basket = []
    if labels:
        choice = st.selectbox("Matching movies", list(labels))
        rating = st.slider("Rating", 1, 5, 5)
        if st.button("Add rating"):
            movie_id = labels[choice]
            st.session_state.basket = [item for item in st.session_state.basket if item["movie_id"] != movie_id]
            title = bundle.movies.loc[bundle.movies["movie_id"] == movie_id, "title"].iloc[0]
            st.session_state.basket.append({"movie_id": movie_id, "title": title, "rating": rating})
    else:
        st.write("No titles match that search.")
    if st.session_state.basket:
        st.dataframe(pd.DataFrame(st.session_state.basket), hide_index=True)
        if st.button("Clear ratings"):
            st.session_state.basket = []
            st.rerun()
    history = pd.DataFrame(st.session_state.basket or [], columns=["movie_id", "title", "rating"])
    if "rating" in history.columns:
        history["rating"] = history["rating"].astype(float)
else:
    user_id = st.number_input("Training user id", min_value=1, value=1, step=1)
    history = bundle.train.loc[bundle.train["user_id"] == int(user_id), ["movie_id", "rating", "timestamp"]]
    st.caption(f"{len(history)} training ratings will be used. Held-out ratings are not shown.")

if st.button("Generate recommendations"):
    gateway = None
    live_requested = method == "hybrid_llm" and not use_mock
    if live_requested and (credentials is None or not confirm_live):
        st.warning("Live LLM reranking was not confirmed or the API is not configured. Falling back automatically.")
    elif use_mock and (method == "hybrid_llm" or explain):
        gateway = build_gateway(live=False)
    elif live_requested and confirm_live and credentials is not None:
        gateway = build_gateway(live=True)

    def recommend(active_gateway, selected_method=method):
        if mode.startswith("New"):
            return recommend_new_user(
                bundle,
                history,
                selected_method,
                top_k,
                gateway=active_gateway if selected_method == "hybrid_llm" else None,
                candidate_k=config.hybrid.top_k_candidates,
            )
        return recommend_existing_user(
            bundle,
            int(user_id),
            selected_method,
            top_k,
            gateway=active_gateway if selected_method == "hybrid_llm" else None,
            candidate_k=config.hybrid.top_k_candidates,
        )

    if gateway is not None and not use_mock:
        st.info(
            "A live rerank usually takes about one minute, and explanations are a second call. "
            "The hybrid list is shown first. This demo waits up to 180 seconds and retries once."
        )
        preview_rows, preview_warnings, _preview_summary = recommend(None, "hybrid")
        if not show_recommendations(preview_rows, preview_warnings, None, "Hybrid order"):
            st.stop()
        with st.spinner("Reranking the top 20 candidates..."):
            rows, warnings, summary = recommend(gateway)
        show_recommendations(rows, warnings, summary, "After LLM reranking")
    else:
        rows, warnings, summary = recommend(gateway)
        if not show_recommendations(rows, warnings, summary, "Recommendations"):
            st.stop()
    if explain and rows:
        explanation_gateway = gateway if use_mock or confirm_live else None
        if explanation_gateway is not None and not use_mock:
            st.caption("Requesting explanations. This is another API call and usually takes about one minute.")
            with st.spinner("Writing explanations..."):
                explanations = explain_recommendations(bundle, history, rows, explanation_gateway)
        else:
            explanations = explain_recommendations(bundle, history, rows, explanation_gateway)
        st.subheader("Explanations")
        if explanation_gateway is None:
            st.caption("Deterministic template explanations, because the LLM was not called.")
        for item in explanations:
            st.write(f"**Movie {item['movie_id']}.** {item['explanation']}")
