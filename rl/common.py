"""Shared neural building blocks for SAC/CQL/OptiDICE."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MLPConfig:
    hidden_sizes: Sequence[int]
    activation: type[nn.Module] = nn.ReLU


def mlp(input_dim: int, hidden_sizes: Sequence[int], activation: type[nn.Module] = nn.ReLU) -> list[nn.Module]:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for hidden in hidden_sizes:
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(activation())
        in_dim = hidden
    return layers


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1 - tau)
        target_param.data.add_(tau * source_param.data)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.net = nn.Sequential(*mlp(obs_dim + act_dim, hidden_sizes), nn.Linear(hidden_sizes[-1], 1))

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, actions], dim=-1)).squeeze(-1)


class TanhGaussianPolicy(nn.Module):
    """Gaussian policy with Tanh squashing and log-prob correction."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int],
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(*mlp(obs_dim, hidden_sizes))
        self.mean_head = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_head = nn.Linear(hidden_sizes[-1], act_dim)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.tanh(log_std)
        log_std = self.min_log_std + 0.5 * (self.max_log_std - self.min_log_std) * (log_std + 1)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1)

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # inverse tanh to compute log prob of already squashed actions
        actions = torch.clamp(actions, -0.999, 0.999)
        atanh = 0.5 * torch.log((1 + actions) / (1 - actions))
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(atanh) - torch.log(1 - actions.pow(2) + 1e-6)
        return log_prob.sum(dim=-1)
