"""Base interfaces for offline RL agents."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Mapping

import torch


class OfflineAgent(ABC):
    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor], step: int) -> Dict[str, float]:
        ...

    @abstractmethod
    def state_dict(self) -> Dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        ...

    @abstractmethod
    def to(self, device: torch.device) -> "OfflineAgent":
        ...

    @abstractmethod
    def actor_parameters(self):
        ...
