"""Parse a local MovieLens 100K directory and write the chronological split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movie_recommender.config import AppConfig
from movie_recommender.data import discover_data_dir
from movie_recommender.prepare import prepare_dataset
from movie_recommender.utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MovieLens 100K")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default=None, help="Directory containing u.data and u.item")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    config = AppConfig.from_yaml(args.config)
    data_dir = discover_data_dir(args.data_dir)
    prepare_dataset(data_dir, config)


if __name__ == "__main__":
    main()
