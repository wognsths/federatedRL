from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.common import mlp


@dataclass
class BehaviorConfig:
    hidden_sizes: tuple[int, ...]
    min_log_std: float = -5.0
    max_log_std: float = 1.0


class BehaviorCloningPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, config: BehaviorConfig) -> None:
        super().__init__()
        self.policy = nn.Sequential(*mlp(obs_dim, config.hidden_sizes))
        last_dim = config.hidden_sizes[-1] if config.hidden_sizes else obs_dim
        self.mean_head = nn.Linear(last_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.min_log_std = config.min_log_std
        self.max_log_std = config.max_log_std

    def forward(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        hidden = self.policy(obs)
        mean = torch.tanh(self.mean_head(hidden))
        log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)
        std = log_std.exp()
        return torch.distributions.Normal(mean, std)

    def loss(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        dist = self.forward(obs)
        return -dist.log_prob(actions).sum(dim=-1).mean()

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        dist = self.forward(obs)
        return dist.log_prob(actions).sum(dim=-1)
