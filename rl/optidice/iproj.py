from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from rl.common import TanhGaussianPolicy


@dataclass
class IProjectionConfig:
    hidden_sizes: tuple[int, ...]
    entropy_reg: float = 0.0


class IProjectionPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, config: IProjectionConfig) -> None:
        super().__init__()
        self.policy = TanhGaussianPolicy(obs_dim, act_dim, config.hidden_sizes)
        self.entropy_reg = config.entropy_reg

    def loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_weights: torch.Tensor,
    ) -> torch.Tensor:
        weights = torch.exp(log_weights).detach()
        log_prob = self.policy.log_prob(obs, actions)
        loss = -(weights * log_prob).mean()
        if self.entropy_reg > 0:
            loss = loss - self.entropy_reg * log_prob.mean()
        return loss

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.policy.log_prob(obs, actions)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy.sample(obs)
