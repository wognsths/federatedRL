from __future__ import annotations

from typing import Callable, Dict

import torch

from data.samplers import BatchSampler
from fed.types import ClientConfig, ClientUpdate, merge_metrics
from rl.base import OfflineAgent


class FederatedClient:
    def __init__(
        self,
        client_id: int,
        dataset,
        agent_builder: Callable[[], OfflineAgent],
        config: ClientConfig,
        device: torch.device,
    ) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.agent = agent_builder().to(device)
        self.config = config
        self.device = device
        self.sampler = BatchSampler(dataset=dataset, seed=config.sampler_seed + client_id)

    def run_round(self, global_state: Dict[str, torch.Tensor], reference_batch: Dict[str, torch.Tensor] | None = None) -> ClientUpdate:
        self.agent.load_state_dict(global_state)
        metrics_per_step = []
        for step in range(self.config.updates_per_round):
            batch = self.sampler.sample(self.config.batch_size, self.device)
            metrics = self.agent.update(batch, step)
            metrics_per_step.append(metrics)
        buffer_weights = None
        if reference_batch is not None:
            with torch.no_grad():
                w_ref, _ = self.agent.dual.ratio(
                    reference_batch["observations"].to(self.device),
                    reference_batch["rewards"].to(self.device),
                    reference_batch["next_observations"].to(self.device),
                    reference_batch["terminals"].to(self.device),
                )
                buffer_weights = w_ref.detach().cpu()
        return ClientUpdate(
            client_id=self.client_id,
            num_samples=len(self.dataset),
            state_dict={k: v.detach().cpu() for k, v in self.agent.state_dict().items()},
            metrics=merge_metrics(metrics_per_step),
            buffer_weights=buffer_weights,
        )
