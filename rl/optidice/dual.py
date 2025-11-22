from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from rl.common import mlp


@dataclass
class DualConfig:
    hidden_sizes: tuple[int, ...]
    alpha: float = 1.0
    clip_value: float = 20.0
    gamma: float = 0.99


class OptiDICEDual(nn.Module):
    def __init__(self, obs_dim: int, config: DualConfig) -> None:
        super().__init__()
        layers = mlp(obs_dim, config.hidden_sizes)
        last_dim = config.hidden_sizes[-1] if config.hidden_sizes else obs_dim
        self.nu = nn.Sequential(*layers, nn.Linear(last_dim, 1))
        self.lambda_param = nn.Parameter(torch.zeros(1))
        self.alpha = config.alpha
        self.clip_value = config.clip_value
        self.gamma = config.gamma

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.nu(obs).squeeze(-1)

    def ratio(self, obs: torch.Tensor, rewards: torch.Tensor, next_obs: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        nu_s = self.forward(obs)
        nu_sp = self.forward(next_obs)
        e_nu = rewards + self.gamma * (1 - dones) * nu_sp - nu_s
        scaled = torch.clamp((e_nu - self.lambda_param) / self.alpha - 1.0, -self.clip_value, self.clip_value)
        w_star = torch.exp(scaled)
        return w_star, e_nu

    def loss(
        self,
        obs: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        initial_obs: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        w_star, e_nu = self.ratio(obs, rewards, next_obs, dones)
        f = w_star * (torch.log(w_star + 1e-8) - 1)
        dual_term = -self.alpha * f + w_star * (e_nu - self.lambda_param)
        init_inputs = initial_obs if initial_obs is not None else obs
        initial_term = (1 - self.gamma) * self.forward(init_inputs).mean()
        loss = (dual_term.mean() + initial_term)
        return loss, w_star.detach()

    def stats(self, w_star: torch.Tensor) -> Dict[str, float]:
        return {
            "ratio_mean": float(w_star.mean().item()),
            "ratio_std": float(w_star.std().item()),
            "ratio_max": float(w_star.max().item()),
            "ratio_second_moment": float((w_star.pow(2).mean()).item()),
        }
