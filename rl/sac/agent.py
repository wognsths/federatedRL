from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import torch
import torch.nn.functional as F

from rl.base import OfflineAgent
from rl.common import QNetwork, TanhGaussianPolicy, soft_update


@dataclass
class SACAgentConfig:
    actor_hidden_sizes: tuple[int, ...]
    critic_hidden_sizes: tuple[int, ...]
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    tau: float = 0.005
    gamma: float = 0.99
    init_temperature: float = 0.1
    target_entropy_scale: float = 1.0


class SACAgent(OfflineAgent):
    def __init__(self, obs_dim: int, act_dim: int, config: SACAgentConfig, device: torch.device) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config
        self.device = device

        self.actor = TanhGaussianPolicy(obs_dim, act_dim, config.actor_hidden_sizes).to(device)
        self.critic1 = QNetwork(obs_dim, act_dim, config.critic_hidden_sizes).to(device)
        self.critic2 = QNetwork(obs_dim, act_dim, config.critic_hidden_sizes).to(device)
        self.target_critic1 = QNetwork(obs_dim, act_dim, config.critic_hidden_sizes).to(device)
        self.target_critic2 = QNetwork(obs_dim, act_dim, config.critic_hidden_sizes).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self.critic_optim = torch.optim.Adam(critic_params, lr=config.critic_lr)
        self.log_alpha = torch.tensor(float(config.init_temperature)).log().detach().clone().requires_grad_(True).to(device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.target_entropy = -float(act_dim) * config.target_entropy_scale

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def update(self, batch: Dict[str, torch.Tensor], step: int) -> Dict[str, float]:
        obs = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_obs = batch["next_observations"].to(self.device)
        dones = batch["terminals"].to(self.device)

        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_obs)
            q1_target = self.target_critic1(next_obs, next_actions)
            q2_target = self.target_critic2(next_obs, next_actions)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_prob
            target = rewards + self.config.gamma * (1 - dones) * q_target

        q1_pred = self.critic1(obs, actions)
        q2_pred = self.critic2(obs, actions)
        critic_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        new_actions, log_prob = self.actor.sample(obs)
        q1_new = self.critic1(obs, new_actions)
        q2_new = self.critic2(obs, new_actions)
        actor_loss = (self.alpha.detach() * log_prob - torch.min(q1_new, q2_new)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        soft_update(self.critic1, self.target_critic1, self.config.tau)
        soft_update(self.critic2, self.target_critic2, self.config.tau)

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
        }

    def state_dict(self) -> Dict[str, torch.Tensor]:
        state: Dict[str, torch.Tensor] = {"log_alpha": self.log_alpha.detach().cpu()}
        modules = {
            "actor": self.actor,
            "critic1": self.critic1,
            "critic2": self.critic2,
            "target_critic1": self.target_critic1,
            "target_critic2": self.target_critic2,
        }
        for prefix, module in modules.items():
            for name, tensor in module.state_dict().items():
                state[f"{prefix}.{name}"] = tensor.detach().cpu()
        return state

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        def _load(prefix: str, module: torch.nn.Module) -> None:
            sub_dict = {name.split(f"{prefix}.", 1)[1]: tensor for name, tensor in state_dict.items() if name.startswith(f"{prefix}.")}
            module.load_state_dict(sub_dict)

        _load("actor", self.actor)
        _load("critic1", self.critic1)
        _load("critic2", self.critic2)
        _load("target_critic1", self.target_critic1)
        _load("target_critic2", self.target_critic2)
        self.log_alpha.data.copy_(state_dict["log_alpha"].to(self.device))

    def to(self, device: torch.device) -> "SACAgent":
        self.device = device
        self.actor.to(device)
        self.critic1.to(device)
        self.critic2.to(device)
        self.target_critic1.to(device)
        self.target_critic2.to(device)
        self.log_alpha = self.log_alpha.to(device)
        return self

    def actor_parameters(self):
        return self.actor.parameters()

    def critic_parameters(self):
        return list(self.critic1.parameters()) + list(self.critic2.parameters())
