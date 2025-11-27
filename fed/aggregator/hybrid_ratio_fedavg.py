from __future__ import annotations

from typing import Dict, Iterable

import torch

from fed.aggregator.base import Aggregator
from fed.aggregator.fedavg import FedAvgAggregator
from fed.aggregator.ratio_weighted import RatioWeightedAggregator
from fed.types import ClientUpdate


class HybridRatioFedAvgAggregator(Aggregator):
    """Blend RatioWeighted and FedAvg aggregation with a warmdown schedule."""

    def __init__(
        self,
        ratio_config: Dict,
        warmdown_rounds: int,
        start_round: int = 0,
    ) -> None:
        self.ratio_agg = RatioWeightedAggregator(**ratio_config)
        self.fedavg_agg = FedAvgAggregator(client_weighting=ratio_config.get("client_weighting", "size"))
        self.warmdown_rounds = max(1, warmdown_rounds)
        self.start_round = start_round
        self.current_round = 0

    def aggregate(self, updates: Iterable[ClientUpdate], global_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ratio_state = self.ratio_agg.aggregate(updates, global_state)
        fedavg_state = self.fedavg_agg.aggregate(updates, global_state)

        progress = max(0.0, min(1.0, (self.current_round - self.start_round) / self.warmdown_rounds))
        ratio_weight = 1.0 - progress
        federated_weight = progress

        blended_state: Dict[str, torch.Tensor] = {}
        for key in global_state.keys():
            ratio_tensor = ratio_state[key]
            fedavg_tensor = fedavg_state[key]
            blended_state[key] = ratio_weight * ratio_tensor + federated_weight * fedavg_tensor

        self.current_round += 1
        return blended_state
