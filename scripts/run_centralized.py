from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict

import torch
import typer
from omegaconf import OmegaConf

from data.d4rl_loader import load_d4rl_dataset
from data.samplers import BatchSampler
from rl.base import OfflineAgent
from rl.evaluation import evaluate_policy
from rl.optidice.agent import OptiDICEAgent, OptiDICEConfig
from rl.sac.agent import SACAgent, SACAgentConfig

app = typer.Typer(pretty_exceptions_enable=False)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_yaml(path: Path) -> Dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]


def _build_agent_factory(algo_name: str, algo_cfg: Dict, obs_dim: int, act_dim: int, device: torch.device) -> Callable[[], OfflineAgent]:
    if algo_name == "sac":
        cfg = SACAgentConfig(
            actor_hidden_sizes=tuple(algo_cfg["actor_hidden_sizes"]),
            critic_hidden_sizes=tuple(algo_cfg["critic_hidden_sizes"]),
            actor_lr=algo_cfg["actor_lr"],
            critic_lr=algo_cfg["critic_lr"],
            alpha_lr=algo_cfg["alpha_lr"],
            tau=algo_cfg["tau"],
            gamma=algo_cfg["gamma"],
            init_temperature=algo_cfg["init_temperature"],
            target_entropy_scale=algo_cfg["target_entropy_scale"],
        )
        return lambda: SACAgent(obs_dim, act_dim, cfg, device)
    if algo_name == "optidice":
        cfg = OptiDICEConfig(
            dual_hidden_sizes=tuple(algo_cfg["dual_hidden_sizes"]),
            actor_hidden_sizes=tuple(algo_cfg["actor_hidden_sizes"]),
            behavior_hidden_sizes=tuple(algo_cfg["behavior_hidden_sizes"]),
            critic_lr=algo_cfg["critic_lr"],
            actor_lr=algo_cfg["actor_lr"],
            behavior_lr=algo_cfg["behavior_lr"],
            alpha=algo_cfg["alpha"],
            clip_value=algo_cfg["clip_value"],
            entropy_reg=algo_cfg["entropy_reg"],
            gamma=algo_cfg["gamma"],
        )
        return lambda: OptiDICEAgent(obs_dim, act_dim, cfg, device)
    raise KeyError(f"Unsupported algo '{algo_name}'")


@app.command()
def main(
    config: Path = typer.Option(Path("configs/train.yaml"), help="Training configuration (reuses run_fed defaults)."),
    algo: str = typer.Option("", help="Override algorithm name (sac|optidice)."),
    data: str = typer.Option("", help="Override dataset config."),
) -> None:
    cfg = OmegaConf.load(config)
    algo_name = algo or cfg.algo
    data_name = data or cfg.data

    algo_cfg = _load_yaml(Path("configs/algo") / f"{algo_name}.yaml")
    data_cfg = _load_yaml(Path("configs/data") / f"{data_name}.yaml")
    device = _device(cfg.device)

    dataset = load_d4rl_dataset(
        data_cfg["task"],
        dataset_dir=data_cfg.get("dataset_dir"),
        normalize_states=data_cfg.get("normalize_states", True),
        normalize_rewards=data_cfg.get("normalize_rewards", False),
        reward_scale=data_cfg.get("reward_scale", 1.0),
    )

    obs_dim = dataset.observations.shape[-1]
    act_dim = dataset.actions.shape[-1]
    agent_factory = _build_agent_factory(algo_name, algo_cfg, obs_dim, act_dim, device)
    agent = agent_factory()

    sampler = BatchSampler(dataset, seed=cfg.seed)
    obs_normalizer = getattr(dataset, "obs_normalizer", None)
    normalizer_tensors = None
    if obs_normalizer is not None:
        normalizer_tensors = {
            "mean": torch.as_tensor(obs_normalizer["mean"], device=device, dtype=torch.float32),
            "std": torch.as_tensor(obs_normalizer["std"], device=device, dtype=torch.float32),
        }
    eval_interval = int(OmegaConf.select(cfg, "eval_interval") or 0)
    eval_episodes = int(OmegaConf.select(cfg, "eval_episodes") or 0)
    updates_per_round = max(cfg.local_steps.values())
    total_steps = cfg.rounds * updates_per_round
    eval_every_steps = eval_interval * updates_per_round if eval_interval > 0 and eval_episodes > 0 else 0
    history = []
    for step in range(total_steps):
        batch = sampler.sample(cfg.batch_size, device)
        metrics = agent.update(batch, step)
        metrics["step"] = float(step)
        if eval_every_steps and (step + 1) % eval_every_steps == 0:
            eval_stats = evaluate_policy(
                agent,
                data_cfg["task"],
                device,
                eval_episodes,
                obs_normalizer=normalizer_tensors,
            )
            metrics.update(eval_stats)
            if eval_stats:
                summary = ", ".join(f"{key}={value:.2f}" for key, value in sorted(eval_stats.items()) if key.startswith("eval_"))
                if summary:
                    typer.echo(f"[Centralized] step {step + 1}: {summary}")
        history.append(metrics)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.log_dir) / f"centralized_{algo_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)
    typer.echo(f"Saved centralized metrics to {run_dir}/metrics.json")


if __name__ == "__main__":
    app()
