from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict

import torch
import typer
from omegaconf import OmegaConf

from data.d4rl_loader import load_d4rl_dataset
from data.reference import build_reference_batch
from data.splitting import split_dataset
from fed.aggregator.fedavg import FedAvgAggregator
from fed.aggregator.fedprox import FedProxAggregator
from fed.aggregator.ratio_weighted import RatioWeightedAggregator
from fed.aggregator.buffer_ratio import BufferRatioAggregator
from fed.client import FederatedClient
from fed.server import FederatedServer, ServerConfig
from fed.types import ClientConfig
from rl.base import OfflineAgent
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


def _build_aggregator(fed_name: str, fed_cfg: Dict, algo_cfg: Dict):
    normalize_lambda = fed_cfg.get("normalize_lambda", False)
    dual_alpha = float(algo_cfg.get("alpha", 1.0))
    use_ratio_weights = fed_cfg.get("use_ratio_weights", True)
    dual_only_weights = fed_cfg.get("dual_only_weights", False)
    if fed_name == "fedavg":
        return FedAvgAggregator(client_weighting=fed_cfg.get("client_weighting", "size"))
    if fed_name == "fedprox":
        return FedProxAggregator(beta=fed_cfg.get("beta", 0.01))
    if fed_name == "ratio_weighted":
        return RatioWeightedAggregator(
            strategy=fed_cfg.get("strategy", "ess"),
            stability_eps=fed_cfg.get("stability_eps", 1e-6),
            normalize_lambda=normalize_lambda,
            dual_alpha=dual_alpha,
            use_ratio_weights=use_ratio_weights,
            dual_only_weights=dual_only_weights,
        )
    if fed_name == "ratio_buffer":
        return BufferRatioAggregator(
            strategy=fed_cfg.get("strategy", "ess"),
            stability_eps=fed_cfg.get("stability_eps", 1e-6),
            normalize_lambda=normalize_lambda,
            dual_alpha=dual_alpha,
            use_ratio_weights=use_ratio_weights,
            dual_only_weights=dual_only_weights,
        )
    raise KeyError(f"Unsupported fed method '{fed_name}'")


@app.command()
def main(
    config: Path = typer.Option(Path("configs/train.yaml"), help="Training configuration file."),
    algo: str = typer.Option("", help="Override algorithm name (sac|optidice)."),
    fed: str = typer.Option("", help="Override federation strategy."),
    data: str = typer.Option("", help="Override dataset config name."),
) -> None:
    cfg = OmegaConf.load(config)
    algo_name = algo or cfg.algo
    fed_name = fed or cfg.fed
    data_name = data or cfg.data

    algo_cfg = _load_yaml(Path("configs/algo") / f"{algo_name}.yaml")
    fed_cfg = _load_yaml(Path("configs/fed") / f"{fed_name}.yaml")
    data_cfg = _load_yaml(Path("configs/data") / f"{data_name}.yaml")

    device = _device(cfg.device)
    dataset = load_d4rl_dataset(
        data_cfg["task"],
        dataset_dir=data_cfg.get("dataset_dir"),
        normalize_states=data_cfg.get("normalize_states", True),
        normalize_rewards=data_cfg.get("normalize_rewards", False),
        reward_scale=data_cfg.get("reward_scale", 1.0),
    )
    split = split_dataset(dataset, cfg.clients, cfg.split, seed=cfg.seed)

    obs_dim = dataset.observations.shape[-1]
    act_dim = dataset.actions.shape[-1]
    agent_factory = _build_agent_factory(algo_name, algo_cfg, obs_dim, act_dim, device)
    aggregator = _build_aggregator(fed_name, fed_cfg, algo_cfg)

    client_config = ClientConfig(
        batch_size=cfg.batch_size,
        updates_per_round=max(cfg.local_steps.values()),
        sampler_seed=cfg.seed,
    )

    clients = [
        FederatedClient(cid, client_dataset, agent_factory, client_config, device)
        for cid, client_dataset in split.clients.items()
    ]

    reference_batch = None
    ref_size = fed_cfg.get("reference_size", 0)
    if ref_size and ref_size > 0:
        reference_batch = build_reference_batch(dataset, ref_size, cfg.seed, device)

    server = FederatedServer(
        agent_builder=agent_factory,
        clients=clients,
        aggregator=aggregator,
        config=ServerConfig(rounds=cfg.rounds, client_fraction=cfg.client_fraction, seed=cfg.seed),
        device=device,
        reference_batch=reference_batch,
    )

    history = server.run()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.log_dir) / f"fed_{algo_name}_{fed_name}_{cfg.split}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)
    typer.echo(f"Saved metrics to {run_dir}/metrics.json")


if __name__ == "__main__":
    app()
