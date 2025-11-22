"""Client partitioning utilities for non-IID regimes described in PLAN.md."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Mapping, MutableMapping, Sequence

import numpy as np
from numpy.random import Generator

try:  # pragma: no cover - optional dependency
    from sklearn.cluster import MiniBatchKMeans
except Exception:  # pragma: no cover - test environments may not have sklearn
    MiniBatchKMeans = None  # type: ignore

from .d4rl_loader import OfflineDataset, TransitionDataset

SplitRegime = Literal["quantile", "cluster", "support"]


@dataclass
class SplitResult:
    """Container for client-specific datasets and lightweight stats."""

    clients: Dict[int, TransitionDataset]
    regime: SplitRegime
    stats: Dict[str, float]


def split_dataset(
    dataset: OfflineDataset,
    num_clients: int,
    regime: SplitRegime,
    *,
    seed: int = 0,
    top_p: float = 0.3,
    bottom_q: float = 0.3,
    num_clusters: int | None = None,
) -> SplitResult:
    """Create non-IID partitions matching the regimes in the project plan."""

    if num_clients <= 0:
        raise ValueError("num_clients must be positive")

    rng = np.random.default_rng(seed)
    dispatch = {
        "quantile": lambda: _quantile_indices(dataset, num_clients),
        "cluster": lambda: _cluster_indices(dataset, num_clients, rng, num_clusters),
        "support": lambda: _support_indices(dataset, num_clients, rng, top_p, bottom_q),
    }

    if regime not in dispatch:
        raise KeyError(f"Unknown regime '{regime}'. Choose from {list(dispatch)}")

    index_map = dispatch[regime]()
    clients = {cid: dataset.subset(indices) for cid, indices in index_map.items()}

    stats = {
        "min_len": float(min(len(data) for data in clients.values())),
        "max_len": float(max(len(data) for data in clients.values())),
        "mean_len": float(np.mean([len(data) for data in clients.values()])),
    }

    if regime == "quantile":
        stats.update(_quantile_stats(dataset, clients))
    elif regime == "cluster":
        stats.update(_cluster_stats(dataset, clients))

    return SplitResult(clients=clients, regime=regime, stats=stats)


def _quantile_indices(dataset: OfflineDataset, num_clients: int) -> Dict[int, np.ndarray]:
    ordered = sorted(enumerate(dataset.trajectories), key=lambda item: item[1].return_sum)
    splits = np.array_split(np.arange(len(ordered)), num_clients)

    mapping: Dict[int, np.ndarray] = {}
    for client_id, slice_indices in enumerate(splits):
        traj_indices = [ordered[idx][0] for idx in slice_indices]
        transition_indices: List[int] = []
        for traj_id in traj_indices:
            transition_indices.extend(dataset.trajectories[traj_id].indices().tolist())
        mapping[client_id] = np.asarray(transition_indices, dtype=np.int64)
    return mapping


def _cluster_indices(
    dataset: OfflineDataset,
    num_clients: int,
    rng: Generator,
    num_clusters: int | None,
) -> Dict[int, np.ndarray]:
    if MiniBatchKMeans is None:
        raise ImportError("scikit-learn is required for cluster-based splitting")

    obs = dataset.observations
    sample_size = min(len(obs), 100_000)
    sample_indices = rng.choice(len(obs), size=sample_size, replace=False)
    sample = obs[sample_indices]

    clusters = num_clusters or min(max(num_clients * 2, 4), 32)
    kmeans = MiniBatchKMeans(n_clusters=clusters, random_state=int(rng.integers(0, 1_000_000)))
    kmeans.fit(sample)
    cluster_ids = kmeans.predict(obs)

    # Assign each cluster to a single client to keep supports skewed.
    cluster_order = np.argsort(np.bincount(cluster_ids, minlength=clusters))[::-1]
    mapping: Dict[int, List[int]] = {cid: [] for cid in range(num_clients)}
    for idx, cluster_id in enumerate(cluster_order):
        client_id = idx % num_clients
        mapping[client_id].append(cluster_id)

    results: Dict[int, np.ndarray] = {}
    for client_id, cluster_list in mapping.items():
        if not cluster_list:
            results[client_id] = np.array([], dtype=np.int64)
            continue
        mask = np.isin(cluster_ids, cluster_list)
        results[client_id] = np.nonzero(mask)[0]
    return results


def _support_indices(
    dataset: OfflineDataset,
    num_clients: int,
    rng: Generator,
    top_p: float,
    bottom_q: float,
) -> Dict[int, np.ndarray]:
    if top_p <= 0 or bottom_q <= 0:
        raise ValueError("top_p and bottom_q must be positive")

    action_norms = np.linalg.norm(dataset.actions, axis=-1)
    upper = np.quantile(action_norms, 1 - top_p)
    lower = np.quantile(action_norms, bottom_q)

    high_indices = np.nonzero(action_norms >= upper)[0]
    low_indices = np.nonzero(action_norms <= lower)[0]
    mid_indices = np.nonzero((action_norms > lower) & (action_norms < upper))[0]

    rng.shuffle(high_indices)
    rng.shuffle(low_indices)
    rng.shuffle(mid_indices)

    def _split_pool(pool: np.ndarray, splits: int) -> List[np.ndarray]:
        splits = max(1, splits)
        if len(pool) == 0:
            return []
        chunks = np.array_split(pool, splits)
        return [chunk.astype(np.int64, copy=False) for chunk in chunks if len(chunk) > 0]

    group_targets = {
        "high": math.ceil(num_clients / 3),
        "low": math.ceil(num_clients / 3),
        "mid": max(num_clients - 2 * math.ceil(num_clients / 3), 1),
    }
    pools = {
        "high": _split_pool(high_indices, group_targets["high"]),
        "low": _split_pool(low_indices, group_targets["low"]),
        "mid": _split_pool(mid_indices, group_targets["mid"]),
    }
    cursors = {group: 0 for group in pools}

    results: Dict[int, np.ndarray] = {}
    for cid in range(num_clients):
        group = "high" if cid % 3 == 0 else ("low" if cid % 3 == 1 else "mid")
        entries = pools.get(group, [])
        if not entries:
            results[cid] = np.array([], dtype=np.int64)
            continue
        pick = entries[cursors[group] % len(entries)]
        cursors[group] += 1
        results[cid] = pick
    return results


def _quantile_stats(dataset: OfflineDataset, clients: Mapping[int, TransitionDataset]) -> Dict[str, float]:
    stats = {}
    for cid, data in clients.items():
        traj_ids = np.unique(data.trajectory_ids) if data.trajectory_ids is not None else []
        if len(traj_ids) == 0:
            stats[f"client_{cid}_return_mean"] = 0.0
            continue
        returns = [dataset.trajectories[int(tid)].return_sum for tid in traj_ids]
        stats[f"client_{cid}_return_mean"] = float(np.mean(returns))
    return stats


def _cluster_stats(dataset: OfflineDataset, clients: Mapping[int, TransitionDataset]) -> Dict[str, float]:
    stats = {}
    for cid, data in clients.items():
        stats[f"client_{cid}_state_std"] = float(data.observations.std())
    return stats
