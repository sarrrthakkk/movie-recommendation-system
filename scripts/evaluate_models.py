"""Evaluate saved models once on the chronological test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movie_recommender.config import AppConfig
from movie_recommender.reporting import evaluate_models
from movie_recommender.utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate non-LLM recommenders on the test split")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    config = AppConfig.from_yaml(args.config)
    evaluate_models(config, device=args.device)


if __name__ == "__main__":
    main()
