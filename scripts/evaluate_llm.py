"""Estimate or run LLM reranking. Live calls require an explicit confirmation flag."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movie_recommender.config import AppConfig
from movie_recommender.llm.client import load_llm_environment
from movie_recommender.llm_experiment import ConfirmationRequired, LLMDisabled, run_llm_evaluation
from movie_recommender.utils import configure_logging, get_logger

logger = get_logger(__name__)


def main() -> int:
    load_llm_environment()
    parser = argparse.ArgumentParser(description="Evaluate LLM reranking on hybrid candidates")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Count uncached calls and do not contact an API")
    parser.add_argument("--mock-llm", action="store_true", help="Use the local mock client instead of the API")
    parser.add_argument("--confirm-api-calls", action="store_true", help="Required before any live API call")
    parser.add_argument("--maximum-api-calls", type=int, default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--sample-users", type=int, default=None)
    parser.add_argument("--ablations", action="store_true", help="Also run candidate-size and history ablations")
    parser.add_argument("--no-ablations", action="store_true")
    parser.add_argument(
        "--explanations",
        action="store_true",
        help="Generate explanations for evaluation audit cohort (up to config.evaluation.explanation_audit_samples items)",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=None,
        help="Override the per-request API timeout",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    config = AppConfig.from_yaml(args.config)
    resolved_model = os.environ.get("LLM_MODEL", "").strip() or config.llm.model
    logger.info("Using LLM model: %s", resolved_model)
    if args.smoke:
        config = config.with_smoke_defaults()
    if args.request_timeout_seconds is not None:
        config = config.model_copy(
            update={
                "llm": config.llm.model_copy(
                    update={"request_timeout_seconds": args.request_timeout_seconds}
                )
            }
        )
        logger.info("API request timeout set to %.0f seconds", args.request_timeout_seconds)
    ablations = True if args.ablations else False if args.no_ablations else None
    try:
        run_llm_evaluation(
            config,
            mock=args.mock_llm,
            dry_run=args.dry_run,
            confirm_api_calls=args.confirm_api_calls,
            maximum_api_calls=args.maximum_api_calls,
            refresh_cache=args.refresh_cache,
            run_ablations=ablations,
            explanations=args.explanations,
            sample_users=args.sample_users,
        )
    except LLMDisabled as exc:
        logger.error("%s", exc)
        return 2
    except ConfirmationRequired as exc:
        logger.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
