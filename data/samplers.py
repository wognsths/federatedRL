"""Sampling utilities for offline datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
import torch

from .d4rl_loader import TransitionDataset


@dataclass
class BatchSampler:
    """Random mini-batch sampler backed by numpy RNG."""

    dataset: TransitionDataset
    seed: int

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._initial_indices = None
        if hasattr(self.dataset, "trajectories") and getattr(self.dataset, "trajectories") is not None:
            try:
                starts = [traj.start for traj in self.dataset.trajectories]  # type: ignore[attr-defined]
                if starts:
                    self._initial_indices = np.asarray(starts, dtype=np.int64)
            except Exception:
                self._initial_indices = None

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = self._rng.integers(0, len(self.dataset), size=batch_size)
        batch = {
            name: torch.as_tensor(array[indices], device=device)
            for name, array in self.dataset.as_dict().items()
        }
        if self._initial_indices is not None and len(self._initial_indices) > 0:
            init_indices = self._rng.choice(self._initial_indices, size=batch_size, replace=True)
            batch["initial_observations"] = torch.as_tensor(self.dataset.observations[init_indices], device=device)
        else:
            batch["initial_observations"] = batch["observations"]
        return batch


@dataclass
class ClientBuffer:
    """Wraps a client dataset together with a sampler configuration."""

    dataset: TransitionDataset
    sampler_seed: int

    def make_sampler(self) -> BatchSampler:
        return BatchSampler(dataset=self.dataset, seed=self.sampler_seed)

    def size(self) -> int:
        return len(self.dataset)
