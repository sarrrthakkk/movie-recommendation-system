"""Run preparation, training, offline evaluation, and a mock LLM evaluation.

This command does not make live API calls. Pass ``--live-llm --confirm-api-calls``
only when you intend to spend the configured request budget.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movie_recommender.config import AppConfig
from movie_recommender.data import discover_data_dir
from movie_recommender.llm_experiment import ConfirmationRequired, LLMDisabled, run_llm_evaluation
from movie_recommender.prepare import prepare_dataset
from movie_recommender.reporting import evaluate_models
from movie_recommender.training import train_models
from movie_recommender.utils import configure_logging, get_logger

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce the offline pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--smoke", action="store_true", help="Short matrix-factorization schedule and a 25-user LLM sample")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--live-llm", action="store_true", help="Call the configured API instead of the mock client")
    parser.add_argument("--confirm-api-calls", action="store_true")
    parser.add_argument("--maximum-api-calls", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    config = AppConfig.from_yaml(args.config)
    if args.smoke:
        config = config.with_smoke_defaults()
    data_dir = discover_data_dir(args.data_dir)
    prepare_dataset(data_dir, config)
    train_models(config, smoke=args.smoke, device=args.device)
    evaluate_models(config, device=args.device or "cpu")
    if args.skip_llm:
        logger.info("Skipped LLM evaluation")
        return 0
    try:
        run_llm_evaluation(
            config,
            mock=not args.live_llm,
            dry_run=False,
            confirm_api_calls=args.confirm_api_calls,
            maximum_api_calls=args.maximum_api_calls,
            explanations=not args.live_llm,
            run_ablations=not args.live_llm,
        )
    except (LLMDisabled, ConfirmationRequired) as exc:
        logger.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
