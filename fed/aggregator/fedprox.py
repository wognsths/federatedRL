from __future__ import annotations

from typing import Dict, List

import torch

from fed.aggregator.base import Aggregator
from fed.types import ClientUpdate


class FedProxAggregator(Aggregator):
    def __init__(self, beta: float = 0.01) -> None:
        self.beta = beta

    def aggregate(self, updates: List[ClientUpdate], global_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not updates:
            return global_state
        weights = [update.num_samples for update in updates]
        averaged = self._weighted_average(updates, weights)
        result: Dict[str, torch.Tensor] = {}
        for key, tensor in averaged.items():
            prox = (1 - self.beta) * tensor + self.beta * global_state[key]
            result[key] = prox.to(global_state[key].device)
        return result
