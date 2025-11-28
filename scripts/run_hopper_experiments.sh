#!/usr/bin/env bash
set -euo pipefail
RUNS=(
  # "python -m scripts.run_centralized --config configs/train_centralized_sac_hopper.yaml"
  "python -m scripts.run_fed --config configs/train_sac_fedavg_hopper.yaml"
  "python -m scripts.run_fed --config configs/train_optidice_dual_weighted_hopper.yaml"
  "python -m scripts.run_fed --config configs/train_optidice_hybrid_hopper.yaml"
)
for cmd in "${RUNS[@]}"; do
  echo "==> $cmd"
  eval "$cmd"
  echo ""
done
