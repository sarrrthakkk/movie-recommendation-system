"""Fetch and cache TMDB metadata for processed MovieLens movies."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movie_recommender.tmdb import TMDBClient, load_tmdb_metadata, save_tmdb_metadata
from movie_recommender.utils import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Fetch TMDB metadata for movies")
    parser.add_argument("--processed-dir", default="data/processed", help="Path to processed data directory")
    parser.add_argument("--output", default="data/processed/tmdb_metadata.json", help="Output JSON cache path")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of movies to fetch")
    parser.add_argument("--refresh", action="store_true", help="Force re-fetch all movies, ignoring existing cache")
    parser.add_argument("--api-key", default=None, help="TMDB API key override (defaults to TMDB_API_KEY env var)")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    configure_logging(args.log_level)

    api_key = args.api_key or os.getenv("TMDB_API_KEY")
    if not api_key:
        logger.error("TMDB_API_KEY is not set. Please set TMDB_API_KEY in your .env or via --api-key.")
        sys.exit(1)

    movies_path = Path(args.processed_dir) / "movies.csv"
    if not movies_path.is_file():
        logger.error(f"Processed movies file not found at {movies_path}. Run python scripts/prepare_data.py first.")
        sys.exit(1)

    output_path = Path(args.output)
    cached_metadata = {} if args.refresh else load_tmdb_metadata(output_path)
    logger.info(f"Loaded {len(cached_metadata)} movies from cache: {output_path}")

    movies_df = pd.read_csv(movies_path)
    logger.info(f"Loaded {len(movies_df)} movies from {movies_path}")

    # Determine movies to fetch
    to_fetch: list[tuple[int, str, int | float | None]] = []
    for _, row in movies_df.iterrows():
        movie_id = int(row["movie_id"])
        if movie_id in cached_metadata:
            continue
        raw_title = str(row["title"])
        year = row["year"] if "year" in row and pd.notna(row["year"]) else None
        to_fetch.append((movie_id, raw_title, year))

    if args.limit is not None and args.limit > 0:
        to_fetch = to_fetch[: args.limit]

    logger.info(f"Need to fetch metadata for {len(to_fetch)} movies (limit={args.limit}, refresh={args.refresh})")

    metadata = dict(cached_metadata)
    processed_count = 0

    if to_fetch:
        with TMDBClient(api_key=api_key) as client:
            for movie_id, raw_title, year in to_fetch:
                movie_meta = client.fetch_metadata_for_movie(movie_id, raw_title, year)
                metadata[movie_id] = movie_meta
                processed_count += 1

                if processed_count % 25 == 0:
                    save_tmdb_metadata(output_path, metadata)
                    logger.info(f"Progress: fetched {processed_count}/{len(to_fetch)} movies. Saved to {output_path}")

        save_tmdb_metadata(output_path, metadata)
        logger.info(f"Final save completed to {output_path}")

    # Summary statistics
    total_cached = len(metadata)
    matched_count = sum(1 for m in metadata.values() if m.get("match_status") == "matched")
    fallback_count = sum(1 for m in metadata.values() if m.get("match_status") == "title_only_fallback")
    missing_count = sum(1 for m in metadata.values() if m.get("match_status") == "missing")
    overview_count = sum(1 for m in metadata.values() if m.get("overview", "").strip())

    print("\n" + "=" * 50)
    print("TMDB Metadata Summary:")
    print(f"  Total movies in cache:       {total_cached}")
    print(f"  Movies fetched this run:     {processed_count}")
    print(f"  Status 'matched':            {matched_count}")
    print(f"  Status 'title_only_fallback':{fallback_count}")
    print(f"  Status 'missing':            {missing_count}")
    print(f"  With non-empty overview:     {overview_count}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
