from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from data.d4rl_loader import OfflineDataset


def build_reference_batch(dataset: OfflineDataset, size: int, seed: int, device: torch.device) -> Dict[str, torch.Tensor]:
    """Construct a shared reference batch sampled uniformly from the global dataset.

    The batch is packaged as torch tensors so it can be broadcast once per round.
    """
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(dataset), size=size)
    return {
        "observations": torch.as_tensor(dataset.observations[indices], device=device),
        "actions": torch.as_tensor(dataset.actions[indices], device=device),
        "rewards": torch.as_tensor(dataset.rewards[indices], device=device),
        "next_observations": torch.as_tensor(dataset.next_observations[indices], device=device),
        "terminals": torch.as_tensor(dataset.terminals[indices], device=device),
    }
