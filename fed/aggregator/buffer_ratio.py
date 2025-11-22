from __future__ import annotations

from typing import Dict, List

import torch

from fed.aggregator.base import Aggregator
from fed.types import ClientUpdate


class BufferRatioAggregator(Aggregator):
    """Aggregate using per-client ratio evaluations on a shared reference buffer.

    This approximates the idea of combining local occupancy ratios w_i into a
    global ratio surrogate by weighting parameter averages according to the
    stability of w_i on the shared buffer.
    """

    def __init__(
        self,
        strategy: str = "ess",
        stability_eps: float = 1e-6,
        normalize_lambda: bool = False,
        dual_alpha: float = 1.0,
        use_ratio_weights: bool = True,
        dual_only_weights: bool = False,
    ) -> None:
        self.strategy = strategy
        self.eps = stability_eps
        self.normalize_lambda = normalize_lambda
        self.dual_alpha = dual_alpha
        self.use_ratio_weights = use_ratio_weights
        self.dual_only_weights = dual_only_weights

    def aggregate(self, updates: List[ClientUpdate], global_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not updates:
            return global_state
        dual_weights = self._dual_weights(updates)
        size_weights = [float(update.num_samples) for update in updates]
        aggregated = self._weighted_average_per_key(updates, dual_weights, size_weights)
        self._maybe_normalize_lambda(aggregated, updates, dual_weights)
        return {k: v.to(global_state[k].device) for k, v in aggregated.items()}

    def _dual_weights(self, updates: List[ClientUpdate]) -> List[float]:
        if not self.use_ratio_weights:
            return [float(update.num_samples) for update in updates]
        return [self._weight_from_buffer(update) for update in updates]

    def _weight_from_buffer(self, update: ClientUpdate) -> float:
        w = update.buffer_weights
        if w is None:
            return float(update.num_samples)
        mean = float(w.mean().item())
        second_moment = float((w.pow(2).mean()).item())
        variance = max(second_moment - mean ** 2, 0.0)
        if self.strategy == "ess":
            return max(mean ** 2 / (variance + self.eps), self.eps)
        return 1.0 / (variance + self.eps)

    def _weighted_average_per_key(
        self,
        updates: List[ClientUpdate],
        dual_weights: List[float],
        size_weights: List[float],
    ) -> Dict[str, torch.Tensor]:
        if not updates:
            return {}
        keys = updates[0].state_dict.keys()
        aggregated: Dict[str, torch.Tensor] = {}
        for key in keys:
            weights = dual_weights if (self.dual_only_weights and key.startswith("dual.")) else (dual_weights if not self.dual_only_weights else size_weights)
            total_weight = float(sum(weights))
            if total_weight <= 0:
                raise ValueError("Total weight must be positive for aggregation")
            for idx, update in enumerate(updates):
                tensor = update.state_dict[key].to(torch.float32)
                contribution = tensor * (weights[idx] / total_weight)
                if key not in aggregated:
                    aggregated[key] = torch.zeros_like(contribution)
                aggregated[key] += contribution
        return aggregated

    def _maybe_normalize_lambda(self, aggregated: Dict[str, torch.Tensor], updates: List[ClientUpdate], dual_weights: List[float]) -> None:
        if not self.normalize_lambda:
            return
        if "dual.lambda_param" not in aggregated:
            return
        mean_w = self._buffer_mean(updates, dual_weights)
        if mean_w is None or mean_w <= 0:
            return
        lambda_tensor = aggregated["dual.lambda_param"]
        shift = self.dual_alpha * torch.log(torch.tensor(mean_w, device=lambda_tensor.device, dtype=lambda_tensor.dtype))
        aggregated["dual.lambda_param"] = lambda_tensor + shift

    def _buffer_mean(self, updates: List[ClientUpdate], weights: List[float]) -> float | None:
        weighted_sum = 0.0
        total = 0.0
        for w, upd in zip(weights, updates):
            if upd.buffer_weights is None:
                continue
            weighted_sum += w * float(upd.buffer_weights.mean().item())
            total += w
        if total <= 0:
            return None
        return weighted_sum / total
