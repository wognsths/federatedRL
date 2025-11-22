"""Rounds-to-X helpers for communication-efficiency reporting."""
from __future__ import annotations

from typing import Iterable, Sequence


def rounds_to_target(curve: Sequence[float], target: float) -> int | None:
    """Return the first round index that reaches ``target`` (inclusive)."""

    for idx, value in enumerate(curve):
        if value >= target:
            return idx
    return None


def rounds_across_seeds(curves: Iterable[Sequence[float]], target: float) -> list[int]:
    """Apply :func:`rounds_to_target` to multiple curves (e.g., seeds)."""

    results = []
    for curve in curves:
        hit = rounds_to_target(curve, target)
        if hit is not None:
            results.append(hit)
    return results
