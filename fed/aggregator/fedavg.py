from __future__ import annotations

from typing import Dict, List

import torch

from fed.aggregator.base import Aggregator
from fed.types import ClientUpdate


class FedAvgAggregator(Aggregator):
    def __init__(self, client_weighting: str = "size") -> None:
        self.client_weighting = client_weighting

    def aggregate(self, updates: List[ClientUpdate], global_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not updates:
            return global_state
        if self.client_weighting == "size":
            weights = [update.num_samples for update in updates]
        else:
            weights = [1.0 for _ in updates]
        return {k: v.to(global_state[k].device) for k, v in self._weighted_average(updates, weights).items()}
