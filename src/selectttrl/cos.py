"""Portable COS direction, scoring, and percentile routing primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def mean_difference(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """Return the raw CAA direction, positive answers minus negative answers."""
    positive = np.asarray(positive, dtype=np.float32)
    negative = np.asarray(negative, dtype=np.float32)
    if positive.ndim != 2 or negative.shape != positive.shape:
        raise ValueError("positive and negative must have the same [n, hidden] shape")
    return (positive - negative).mean(axis=0)


def normalize(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("direction must be finite and nonzero")
    return direction / norm


def cosine_scores(hidden: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Compute one COS score per response row."""
    hidden = np.asarray(hidden, dtype=np.float32)
    if hidden.ndim != 2:
        raise ValueError("hidden must have shape [n, hidden]")
    unit = normalize(direction)
    norms = np.linalg.norm(hidden, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms == 0.0):
        raise ValueError("hidden rows must have finite nonzero norms")
    return (hidden @ unit) / norms


@dataclass(frozen=True)
class Route:
    selected_indices: tuple[int, ...]
    selected_fraction: float
    lower_percentile: float
    upper_percentile: float


def select_percentile_interval(
    scores: Sequence[float], *, lower: float = 0.25, upper: float = 0.60,
    question_ids: Iterable[str] | None = None,
) -> Route:
    """Select the one-based rank interval defined in the paper.

    Scores are sorted ascending and question IDs deterministically break ties;
    the zero-based slice is ``[floor(lower*M):floor(upper*M)]``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("scores must be a nonempty vector")
    if not (0.0 <= lower < upper <= 1.0):
        raise ValueError("require 0 <= lower < upper <= 1")
    ids = list(question_ids) if question_ids is not None else [str(i) for i in range(len(scores))]
    if len(ids) != len(scores):
        raise ValueError("question_ids length mismatch")
    order = sorted(range(len(scores)), key=lambda i: (float(scores[i]), str(ids[i])))
    left = int(np.floor(lower * len(order)))
    right = int(np.floor(upper * len(order)))
    selected = tuple(order[left:right])
    return Route(selected, len(selected) / len(order), lower, upper)
