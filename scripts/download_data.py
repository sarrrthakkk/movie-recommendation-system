"""Download the official MovieLens 100K archive into ``data/raw``."""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movie_recommender.utils import configure_logging, get_logger, project_root

OFFICIAL_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
logger = get_logger(__name__)


def download(destination: Path, force: bool = False) -> Path:
    target = destination / "ml-100k"
    if (target / "u.data").exists() and (target / "u.item").exists() and not force:
        logger.info("MovieLens 100K already exists at %s", target)
        return target
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "ml-100k.zip"
    logger.info("Downloading %s", OFFICIAL_URL)
    try:
        urllib.request.urlretrieve(OFFICIAL_URL, archive)
    except OSError as exc:
        raise RuntimeError(f"Could not download MovieLens 100K from {OFFICIAL_URL}: {exc}") from exc
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
    if not (target / "u.data").exists():
        raise RuntimeError(f"Download completed but {target / 'u.data'} is missing")
    logger.info("Extracted MovieLens 100K to %s", target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MovieLens 100K from GroupLens")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to data/raw")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    output = args.output_dir or (project_root() / "data" / "raw")
    download(output, force=args.force)


if __name__ == "__main__":
    main()
