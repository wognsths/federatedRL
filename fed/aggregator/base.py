from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List

import torch

from fed.types import ClientUpdate


class Aggregator(ABC):
    @abstractmethod
    def aggregate(self, updates: List[ClientUpdate], global_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ...

    def _weighted_average(
        self,
        updates: List[ClientUpdate],
        weights: Iterable[float],
    ) -> Dict[str, torch.Tensor]:
        weights = list(weights)
        total_weight = float(sum(weights))
        if total_weight == 0:
            raise ValueError("Total weight must be positive for aggregation")
        keys = updates[0].state_dict.keys()
        aggregated: Dict[str, torch.Tensor] = {}
        for key in keys:
            for idx, update in enumerate(updates):
                tensor = update.state_dict[key].to(torch.float32)
                contribution = tensor * (weights[idx] / total_weight)
                if key not in aggregated:
                    aggregated[key] = torch.zeros_like(contribution)
                aggregated[key] += contribution
        return aggregated
