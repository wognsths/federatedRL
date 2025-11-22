"""Communication accounting utilities."""
from __future__ import annotations

from typing import Mapping

import torch


def parameter_bytes(state_dict: Mapping[str, torch.Tensor], precision_bits: int = 32) -> int:
    """Rough byte count for a state-dict when serialized with given precision."""

    bytes_per_param = precision_bits // 8
    total_params = sum(tensor.numel() for tensor in state_dict.values())
    return total_params * bytes_per_param


def round_bytes(
    state_dict: Mapping[str, torch.Tensor],
    num_clients: int,
    *,
    upload_only: bool = False,
    precision_bits: int = 32,
) -> int:
    """Bytes transferred in one round assuming FedAvg-style full participation."""

    payload = parameter_bytes(state_dict, precision_bits=precision_bits)
    if upload_only:
        return payload * num_clients
    return payload * num_clients * 2  # upload + download
