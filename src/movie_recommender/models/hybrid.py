"""Score normalization and the hybrid linear combination."""

from __future__ import annotations

import numpy as np


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    """Scale scores to [0, 1] independently for one user.

    A constant vector becomes 0.5 so it does not invent a preference order.
    """
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return values.copy()
    lo = float(np.min(values))
    hi = float(np.max(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.full(values.shape, 0.5, dtype=float)
    return (values - lo) / (hi - lo)


def combine_scores(mf_norm: np.ndarray, content_norm: np.ndarray, alpha: float) -> np.ndarray:
    """Return ``alpha * mf + (1 - alpha) * content`` after per-user normalization."""
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be within [0, 1]")
    mf_values = np.asarray(mf_norm, dtype=float)
    content_values = np.asarray(content_norm, dtype=float)
    if mf_values.shape != content_values.shape:
        raise ValueError("mf and content scores must have the same shape")
    return float(alpha) * mf_values + (1.0 - float(alpha)) * content_values


def rank_order(movie_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Return indices that sort by score descending, then movie id ascending."""
    ids = np.asarray(movie_ids)
    values = np.nan_to_num(np.asarray(scores, dtype=float), nan=-1e18)
    return np.lexsort((ids, -values))
