"""Utilities to load D4RL datasets with graceful fallbacks.

This module isolates all interactions with gymnasium/D4RL so the rest of the
codebase can remain importable even when MuJoCo is unavailable. When D4RL is
missing we synthesize a lightweight Gaussian dataset that mimics the dimensions
of the requested task so unit tests can still exercise the training pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

# Minimal metadata for synthetic fallbacks. The observation/action dimensions
# match the MuJoCo tasks exposed in D4RL.
TASK_SPECS: Dict[str, Dict[str, int]] = {
    "halfcheetah-medium-replay-v2": {"obs_dim": 17, "act_dim": 6},
    "hopper-medium-replay-v2": {"obs_dim": 11, "act_dim": 3},
    "walker2d-medium-replay-v2": {"obs_dim": 17, "act_dim": 6},
}


@dataclass(frozen=True)
class Trajectory:
    """Represents a contiguous slice of transitions."""

    start: int
    end: int
    return_sum: float

    def indices(self) -> np.ndarray:
        return np.arange(self.start, self.end, dtype=np.int64)

    def __len__(self) -> int:
        return self.end - self.start


@dataclass
class TransitionDataset:
    """Container for transition-level numpy arrays."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray
    trajectory_ids: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        length = len(self.observations)
        for field_name, array in self._arrays().items():
            if len(array) != length:
                raise ValueError(f"Field '{field_name}' length mismatch: {len(array)} != {length}")

    def _arrays(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_observations": self.next_observations,
            "terminals": self.terminals,
        }

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.observations)

    def subset(self, indices: Sequence[int]) -> "TransitionDataset":
        idx = np.asarray(indices, dtype=np.int64)
        data = {name: array[idx] for name, array in self._arrays().items()}
        traj_ids = None if self.trajectory_ids is None else self.trajectory_ids[idx]
        return TransitionDataset(**data, trajectory_ids=traj_ids)

    def as_dict(self) -> Dict[str, np.ndarray]:
        return self._arrays()


@dataclass
class OfflineDataset(TransitionDataset):
    """Extends :class:`TransitionDataset` with trajectory bookkeeping."""

    trajectories: List[Trajectory] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.trajectories is None:
            raise ValueError("OfflineDataset requires trajectory metadata")

    @property
    def num_trajectories(self) -> int:
        return len(self.trajectories)

    def iter_trajectories(self) -> Iterable[Trajectory]:
        return iter(self.trajectories)


def load_d4rl_dataset(
    task: str,
    dataset_dir: str | Path | None = None,
    *,
    normalize_states: bool = True,
    normalize_rewards: bool = False,
    reward_scale: float = 1.0,
) -> OfflineDataset:
    """Load a task dataset with graceful fallback.

    Parameters
    ----------
    task: str
        D4RL task name (e.g., ``halfcheetah-medium-replay-v2``).
    dataset_dir: Path-like, optional
        Directory to cache downloaded datasets if using the official loader.
    normalize_states: bool
        Whether to z-score observations.
    normalize_rewards: bool
        Whether to z-score rewards before applying ``reward_scale``.
    reward_scale: float
        Global multiplicative factor applied last.
    """

    dataset_dir = Path(dataset_dir or "./datasets")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    try:
        data = _load_from_d4rl(task)
    except Exception as exc:  # pragma: no cover - exercised when MuJoCo missing
        LOGGER.warning("Falling back to synthetic dataset for %s (%s)", task, exc)
        data = _build_synthetic_dataset(task)

    obs = data["observations"].astype(np.float32)
    rew = data["rewards"].astype(np.float32)
    acts = data["actions"].astype(np.float32)
    next_obs = data["next_observations"].astype(np.float32)
    terminals = data["terminals"].astype(np.float32)

    obs_mean = None
    obs_std = None
    if normalize_states:
        obs_mean = obs.mean(axis=0, keepdims=True)
        obs_std = obs.std(axis=0, keepdims=True) + 1e-6
        obs = (obs - obs_mean) / obs_std
        next_obs = (next_obs - obs_mean) / obs_std

    if normalize_rewards:
        rew = (rew - rew.mean()) / (rew.std() + 1e-6)

    rew *= reward_scale

    trajectories, trajectory_ids = _build_trajectories(rew, terminals)

    dataset = OfflineDataset(
        observations=obs,
        actions=acts,
        rewards=rew,
        next_observations=next_obs,
        terminals=terminals,
        trajectory_ids=trajectory_ids,
        trajectories=trajectories,
    )
    if obs_mean is not None and obs_std is not None:
        dataset.obs_normalizer = {"mean": obs_mean.astype(np.float32), "std": obs_std.astype(np.float32)}
    else:
        dataset.obs_normalizer = None
    dataset.reward_scale = reward_scale
    dataset.normalize_rewards = normalize_rewards
    dataset.normalize_states = normalize_states
    return dataset


def _load_from_d4rl(task: str) -> Dict[str, np.ndarray]:  # pragma: no cover - requires MuJoCo
    import gymnasium as gym  # type: ignore
    import d4rl  # type: ignore  # noqa: F401 - registers environments

    env = gym.make(task)
    try:
        dataset = env.get_dataset()
    except AttributeError:
        from d4rl import qlearning_dataset  # type: ignore

        dataset = qlearning_dataset(env)

    timeouts = dataset.get("timeouts")
    if timeouts is None:
        timeouts = np.zeros_like(dataset["terminals"])

    return {
        "observations": dataset["observations"],
        "actions": dataset["actions"],
        "rewards": dataset["rewards"],
        "next_observations": dataset["next_observations"],
        "terminals": np.maximum(dataset["terminals"], timeouts),
    }


def _build_synthetic_dataset(task: str, length: int = 50_000) -> Dict[str, np.ndarray]:
    specs = TASK_SPECS.get(task)
    if specs is None:
        raise KeyError(f"Unknown task '{task}', please extend TASK_SPECS for synthetic fallback")

    rng = np.random.default_rng(0)
    obs = rng.normal(size=(length, specs["obs_dim"])).astype(np.float32)
    actions = rng.normal(size=(length, specs["act_dim"])).astype(np.float32)
    weights = rng.normal(size=(specs["obs_dim"], specs["act_dim"])).astype(np.float32)
    logits = (obs @ weights).sum(axis=-1, keepdims=True)
    rewards = np.tanh(logits).squeeze(-1) + 0.1 * rng.normal(size=length)

    next_obs = obs + 0.1 * actions @ rng.normal(size=(specs["act_dim"], specs["obs_dim"]))
    dones = np.zeros(length, dtype=np.float32)
    dones[99::100] = 1.0  # synthetic episode boundaries

    return {
        "observations": obs,
        "actions": actions,
        "rewards": rewards.astype(np.float32),
        "next_observations": next_obs.astype(np.float32),
        "terminals": dones,
    }


def _build_trajectories(rewards: np.ndarray, terminals: np.ndarray) -> Tuple[List[Trajectory], np.ndarray]:
    slices: List[Trajectory] = []
    trajectory_ids = np.zeros_like(rewards, dtype=np.int64)
    start = 0
    running_return = 0.0
    trajectory_idx = 0

    for idx, (reward, done) in enumerate(zip(rewards, terminals)):
        running_return += float(reward)
        if done:
            slices.append(Trajectory(start=start, end=idx + 1, return_sum=running_return))
            trajectory_ids[start : idx + 1] = trajectory_idx
            trajectory_idx += 1
            start = idx + 1
            running_return = 0.0

    if start < len(rewards):
        slices.append(Trajectory(start=start, end=len(rewards), return_sum=running_return))
        trajectory_ids[start:] = trajectory_idx

    return slices, trajectory_ids
