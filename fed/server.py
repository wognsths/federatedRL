from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import torch

from fed.aggregator.base import Aggregator
from fed.client import FederatedClient
from fed.types import ClientUpdate, merge_metrics
from rl.base import OfflineAgent


@dataclass
class ServerConfig:
    rounds: int = 10
    client_fraction: float = 1.0
    seed: int = 0


class FederatedServer:
    def __init__(
        self,
        agent_builder: Callable[[], OfflineAgent],
        clients: List[FederatedClient],
        aggregator: Aggregator,
        config: ServerConfig,
        device: torch.device,
        reference_batch: Dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.agent_builder = agent_builder
        self.clients = clients
        self.aggregator = aggregator
        self.config = config
        self.device = device
        self.global_agent = agent_builder().to(device)
        self.global_state = self.global_agent.state_dict()
        self.rng = np.random.default_rng(config.seed)
        self.history: List[Dict[str, float]] = []
        self.reference_batch = reference_batch

    def run(self) -> List[Dict[str, float]]:
        for round_idx in range(self.config.rounds):
            selected = self._select_clients()
            updates = [client.run_round(self.global_state, self.reference_batch) for client in selected]
            self.global_state = self.aggregator.aggregate(updates, self.global_state)
            self.global_agent.load_state_dict(self.global_state)

            round_metrics = merge_metrics([update.metrics for update in updates])
            round_metrics["round"] = float(round_idx)
            round_metrics["clients"] = float(len(selected))
            self.history.append(round_metrics)
        return self.history

    def _select_clients(self) -> List[FederatedClient]:
        if self.config.client_fraction >= 1.0:
            return self.clients
        num_clients = max(1, int(len(self.clients) * self.config.client_fraction))
        indices = self.rng.choice(len(self.clients), size=num_clients, replace=False)
        return [self.clients[idx] for idx in indices]
