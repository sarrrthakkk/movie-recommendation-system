"""Train popularity, matrix factorization, and the hybrid alpha."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movie_recommender.config import AppConfig
from movie_recommender.training import train_models
from movie_recommender.utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Train recommendation models")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true", help="Short training schedule for a smoke experiment")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--latent-dimensions", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or mps")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    config = AppConfig.from_yaml(args.config)
    if args.epochs is not None:
        config.matrix_factorization.epochs = args.epochs
    if args.latent_dimensions is not None:
        config.matrix_factorization.latent_dimensions = args.latent_dimensions
    if args.device is not None:
        config.matrix_factorization.device = args.device
    train_models(config, smoke=args.smoke, device=args.device)


if __name__ == "__main__":
    main()
