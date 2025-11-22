"""Normalized score utilities with graceful D4RL fallbacks."""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

# Reference returns from the D4RL paper (normalized formula: 100 for expert, 0 for random)
D4RL_REFERENCES = {
    "halfcheetah-medium-replay-v2": {"random": -4.3, "expert": 45.8},
    "hopper-medium-replay-v2": {"random": 0.6, "expert": 101.6},
    "walker2d-medium-replay-v2": {"random": 1.9, "expert": 110.0},
}


def normalized_score(task: str, returns: Iterable[float]) -> float:
    """Compute the D4RL normalized score for the given list of returns."""

    returns_array = np.asarray(list(returns), dtype=np.float32)
    if returns_array.size == 0:
        raise ValueError("returns must not be empty")

    ref = D4RL_REFERENCES.get(task)
    if ref is None:
        raise KeyError(f"Missing reference stats for task '{task}'")

    return float((returns_array.mean() - ref["random"]) / (ref["expert"] - ref["random"]) * 100.0)


def batch_normalized_scores(task: str, returns: Mapping[str, Iterable[float]]) -> dict[str, float]:
    """Helper to compute normalized scores for multiple series (e.g., per round)."""

    return {key: normalized_score(task, values) for key, values in returns.items()}
