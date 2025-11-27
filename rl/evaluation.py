from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import torch

from metrics.d4rl_score import normalized_score

LOGGER = logging.getLogger(__name__)


def evaluate_policy(
    agent,
    task: str,
    device: torch.device,
    episodes: int,
    obs_normalizer: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    """Roll out ``agent`` in the given D4RL task and report raw/normalized returns."""

    if episodes <= 0:
        return {}

    try:
        import gym  # type: ignore
    except Exception:
        try:
            import gymnasium as gym  # type: ignore
        except Exception as exc:  # pragma: no cover - requires MuJoCo
            LOGGER.warning("Skipping evaluation: gym import failed (%s)", exc)
            return {}

    try:  # pragma: no cover - optional dependency when MuJoCo installed
        import d4rl  # type: ignore  # noqa: F401
    except Exception:
        LOGGER.debug("D4RL package not available; proceeding with base Gym environment")

    actor = getattr(agent, "actor", None)
    if actor is None:
        LOGGER.warning("Agent %s has no 'actor' attribute; skipping evaluation", type(agent).__name__)
        return {}

    try:
        env = gym.make(task)
    except Exception as exc:  # pragma: no cover - environment construction
        LOGGER.warning("Unable to create env '%s' (%s); skipping evaluation", task, exc)
        return {}

    mean = None
    std = None
    if obs_normalizer is not None:
        mean = obs_normalizer.get("mean")
        std = obs_normalizer.get("std")
        if mean is not None:
            mean = mean.to(device).squeeze(0)
        if std is not None:
            std = std.to(device).squeeze(0)

    returns: list[float] = []
    was_training = getattr(actor, "training", False)
    actor.eval()
    try:
        with torch.no_grad():
            for _ in range(episodes):
                reset_result = env.reset()
                if isinstance(reset_result, tuple):
                    obs = reset_result[0]
                else:
                    obs = reset_result
                done = False
                total_return = 0.0
                while not done:
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    if mean is not None and std is not None:
                        obs_tensor = (obs_tensor - mean) / std
                    action, _ = actor.sample(obs_tensor.unsqueeze(0))
                    action_np = action.squeeze(0).clamp(-1.0, 1.0).cpu().numpy()
                    step_result = env.step(action_np)
                    if len(step_result) == 4:
                        obs, reward, done, _ = step_result
                    else:
                        obs, reward, terminated, truncated, _ = step_result
                        done = bool(terminated or truncated)
                    total_return += float(reward)
                returns.append(total_return)
    except Exception as exc:  # pragma: no cover - runtime rollout issues
        LOGGER.warning("Evaluation rollout failed: %s", exc)
        return {}
    finally:
        env.close()
        if was_training:
            actor.train()

    if not returns:
        return {}

    returns_array = np.asarray(returns, dtype=np.float32)
    stats: Dict[str, float] = {
        "eval_return_mean": float(returns_array.mean()),
        "eval_return_std": float(returns_array.std()),
        "eval_return_min": float(returns_array.min()),
        "eval_return_max": float(returns_array.max()),
    }
    try:
        stats["eval_d4rl_score"] = normalized_score(task, returns_array)
    except KeyError:
        LOGGER.warning("No D4RL reference stats for task '%s'; skipping normalized score", task)
    return stats
