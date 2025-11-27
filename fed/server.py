from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from fed.aggregator.base import Aggregator
from fed.client import FederatedClient
from fed.types import ClientUpdate, merge_metrics
from rl.base import OfflineAgent
from rl.evaluation import evaluate_policy


@dataclass
class ServerConfig:
    rounds: int = 10
    client_fraction: float = 1.0
    seed: int = 0


@dataclass
class EvaluationConfig:
    task: str
    interval: int
    episodes: int
    obs_normalizer: Optional[Dict[str, torch.Tensor]] = None


class FederatedServer:
    def __init__(
        self,
        agent_builder: Callable[[], OfflineAgent],
        clients: List[FederatedClient],
        aggregator: Aggregator,
        config: ServerConfig,
        device: torch.device,
        reference_batch: Dict[str, torch.Tensor] | None = None,
        evaluation: EvaluationConfig | None = None,
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
        self.evaluation = evaluation

    def run(
        self,
        log_fn: Optional[Callable[[str], None]] = None,
        eval_hook: Optional[Callable[[int, Dict[str, float]], None]] = None,
    ) -> List[Dict[str, float]]:
        for round_idx in range(self.config.rounds):
            selected = self._select_clients()
            if log_fn is not None:
                client_ids = ", ".join(str(client.client_id) for client in selected)
                log_fn(f"[Round {round_idx + 1}/{self.config.rounds}] selected clients: [{client_ids}]")
            updates: List[ClientUpdate] = []
            for client in selected:
                if log_fn is not None:
                    log_fn(f"  - Client {client.client_id}: starting local update")
                update = client.run_round(self.global_state, self.reference_batch)
                updates.append(update)
                if log_fn is not None:
                    metrics_str = ", ".join(f"{key}={value:.4f}" for key, value in sorted(update.metrics.items()))
                    if metrics_str:
                        log_fn(f"  - Client {client.client_id}: {metrics_str}")
                    else:
                        log_fn(f"  - Client {client.client_id}: completed without metrics")
            self.global_state = self.aggregator.aggregate(updates, self.global_state)
            self.global_agent.load_state_dict(self.global_state)

            round_metrics = merge_metrics([update.metrics for update in updates])
            round_metrics["round"] = float(round_idx)
            round_metrics["clients"] = float(len(selected))
            eval_stats = self._maybe_evaluate(round_idx)
            if eval_stats:
                round_metrics.update(eval_stats)
                if eval_hook is not None:
                    eval_hook(round_idx, eval_stats)
                if log_fn is not None:
                    eval_msg = ", ".join(f"{key}={value:.2f}" for key, value in sorted(eval_stats.items()) if key.startswith("eval_"))
                    if eval_msg:
                        log_fn(f"    Evaluation: {eval_msg}")
            self.history.append(round_metrics)
            if log_fn is not None:
                metrics_to_log = {k: v for k, v in round_metrics.items() if k not in {"round", "clients"}}
                metrics_str = ", ".join(f"{key}={value:.4f}" for key, value in sorted(metrics_to_log.items()))
                message = f"[Round {round_idx + 1}/{self.config.rounds}] clients={len(selected)}"
                if metrics_str:
                    message = f"{message} | {metrics_str}"
                log_fn(message)
        return self.history

    def _select_clients(self) -> List[FederatedClient]:
        if self.config.client_fraction >= 1.0:
            return self.clients
        num_clients = max(1, int(len(self.clients) * self.config.client_fraction))
        indices = self.rng.choice(len(self.clients), size=num_clients, replace=False)
        return [self.clients[idx] for idx in indices]

    def _maybe_evaluate(self, round_idx: int) -> Dict[str, float]:
        if self.evaluation is None:
            return {}
        if self.evaluation.interval <= 0:
            return {}
        if (round_idx + 1) % self.evaluation.interval != 0:
            return {}
        return evaluate_policy(
            self.global_agent,
            self.evaluation.task,
            self.device,
            self.evaluation.episodes,
            obs_normalizer=self.evaluation.obs_normalizer,
        )
