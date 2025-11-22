"""Gradient alignment utilities."""
from __future__ import annotations

from typing import Iterable, List

import torch


def flatten_gradients(named_grads: Iterable[torch.Tensor]) -> torch.Tensor:
    """Flatten and concatenate tensors into a single vector."""

    vectors: List[torch.Tensor] = [tensor.reshape(-1) for tensor in named_grads]
    if not vectors:
        return torch.tensor([])
    return torch.cat(vectors)


def cosine_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor, eps: float = 1e-8) -> float:
    if vec_a.numel() == 0 or vec_b.numel() == 0:
        return 0.0
    return float(torch.dot(vec_a, vec_b) / ((vec_a.norm() + eps) * (vec_b.norm() + eps)))


def pairwise_cosine(vectors: Iterable[torch.Tensor]) -> torch.Tensor:
    """Compute a symmetric cosine similarity matrix for the provided vectors."""

    collected = [vec for vec in vectors if vec.numel() > 0]
    if not collected:
        return torch.zeros(0, 0)
    stacked = torch.stack(collected)
    normalized = stacked / (stacked.norm(dim=1, keepdim=True) + 1e-8)
    return normalized @ normalized.T
