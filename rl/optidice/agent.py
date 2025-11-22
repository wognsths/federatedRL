from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import torch

from rl.base import OfflineAgent
from rl.optidice.bc import BehaviorCloningPolicy, BehaviorConfig
from rl.optidice.dual import DualConfig, OptiDICEDual
from rl.optidice.iproj import IProjectionConfig, IProjectionPolicy


@dataclass
class OptiDICEConfig:
    dual_hidden_sizes: tuple[int, ...]
    actor_hidden_sizes: tuple[int, ...]
    behavior_hidden_sizes: tuple[int, ...]
    critic_lr: float = 3e-4
    actor_lr: float = 3e-4
    behavior_lr: float = 3e-4
    alpha: float = 1.0
    clip_value: float = 20.0
    entropy_reg: float = 0.0
    gamma: float = 0.99


class OptiDICEAgent(OfflineAgent):
    def __init__(self, obs_dim: int, act_dim: int, config: OptiDICEConfig, device: torch.device) -> None:
        self.device = device
        dual_cfg = DualConfig(config.dual_hidden_sizes, alpha=config.alpha, clip_value=config.clip_value, gamma=config.gamma)
        self.dual = OptiDICEDual(obs_dim, dual_cfg).to(device)
        self.actor = IProjectionPolicy(obs_dim, act_dim, IProjectionConfig(config.actor_hidden_sizes, config.entropy_reg)).to(device)
        self.behavior = BehaviorCloningPolicy(obs_dim, act_dim, BehaviorConfig(config.behavior_hidden_sizes)).to(device)

        self.dual_optim = torch.optim.Adam(self.dual.parameters(), lr=config.critic_lr)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.behavior_optim = torch.optim.Adam(self.behavior.parameters(), lr=config.behavior_lr)
        self.gamma = config.gamma

    def update(self, batch: Dict[str, torch.Tensor], step: int) -> Dict[str, float]:
        obs = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_obs = batch["next_observations"].to(self.device)
        dones = batch["terminals"].to(self.device)
        initial_obs = batch.get("initial_observations")
        if initial_obs is not None:
            initial_obs = initial_obs.to(self.device)

        # Use the broadcast global dual (pre-update) to compute ratio weights for actor training.
        with torch.no_grad():
            w_actor, _ = self.dual.ratio(obs, rewards, next_obs, dones)
            log_w_actor = torch.log(w_actor + 1e-8)

        bc_loss = self.behavior.loss(obs, actions)
        self.behavior_optim.zero_grad()
        bc_loss.backward()
        self.behavior_optim.step()

        dual_loss, w_star = self.dual.loss(obs, rewards, next_obs, dones, initial_obs=initial_obs)
        self.dual_optim.zero_grad()
        dual_loss.backward()
        self.dual_optim.step()

        actor_loss = self.actor.loss(obs, actions, log_w_actor)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        stats = self.dual.stats(w_star)
        stats.update({
            "actor_loss": float(actor_loss.item()),
            "dual_loss": float(dual_loss.item()),
            "bc_loss": float(bc_loss.item()),
            "actor_weight_mean": float(w_actor.mean().item()),
        })
        return stats

    def state_dict(self) -> Dict[str, torch.Tensor]:
        state: Dict[str, torch.Tensor] = {}
        for prefix, module in {
            "dual": self.dual,
            "actor": self.actor,
            "behavior": self.behavior,
        }.items():
            for name, tensor in module.state_dict().items():
                state[f"{prefix}.{name}"] = tensor.detach().cpu()
        return state

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        def _load(prefix: str, module: torch.nn.Module) -> None:
            sub = {name.split(f"{prefix}.", 1)[1]: tensor for name, tensor in state_dict.items() if name.startswith(f"{prefix}.")}
            module.load_state_dict(sub)

        _load("dual", self.dual)
        _load("actor", self.actor)
        _load("behavior", self.behavior)

    def to(self, device: torch.device) -> "OptiDICEAgent":
        self.device = device
        self.dual.to(device)
        self.actor.to(device)
        self.behavior.to(device)
        return self

    def actor_parameters(self):
        return self.actor.parameters()
