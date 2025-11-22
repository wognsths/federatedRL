from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, MutableMapping

import torch


@dataclass
class ClientConfig:
    batch_size: int = 256
    updates_per_round: int = 200
    sampler_seed: int = 0


def merge_metrics(metrics_list: list[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_list:
        return {}
    result: Dict[str, float] = {}
    all_keys = set().union(*(metrics.keys() for metrics in metrics_list))
    for key in all_keys:
        values = [metrics[key] for metrics in metrics_list if key in metrics]
        if values:
            result[key] = sum(values) / len(values)
    return result


@dataclass
class ClientUpdate:
    client_id: int
    num_samples: int
    state_dict: Dict[str, torch.Tensor]
    metrics: Dict[str, float] = field(default_factory=dict)
    buffer_weights: torch.Tensor | None = None

    def to_device(self, device: torch.device) -> "ClientUpdate":
        for key, tensor in self.state_dict.items():
            if isinstance(tensor, torch.Tensor):
                self.state_dict[key] = tensor.to(device)
        if isinstance(self.buffer_weights, torch.Tensor):
            self.buffer_weights = self.buffer_weights.to(device)
        return self
