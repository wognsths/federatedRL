#!/usr/bin/env bash

# Sequential experiment runner:
#   1) Centralized SAC baseline
#   2) FedAvg SAC baseline
#   3) Proposed Fed+OptiDICE variants (see docs/PLAN.md)
#
# Each command mirrors the README workflow and uses python -m to execute the CLI
# modules so the package-relative imports stay consistent.

set -euo pipefail

run() {
  echo ""
  echo "==> $*"
  "$@"
}

# run python -m scripts.run_centralized --config configs/train_sac_fedavg.yaml
run python -m scripts.run_fed --config configs/train_sac_fedavg.yaml

# run python -m scripts.run_fed --config configs/train_optidice_dualavg.yaml
# run python -m scripts.run_fed --config configs/train_optidice_lambda_norm.yaml
# run python -m scripts.run_fed --config configs/train_optidice_dual_weighted.yaml
run python -m scripts.run_fed --config configs/train_optidice_hybrid.yaml
# run python -m scripts.run_fed --config configs/train_optidice_ratio_buffer.yaml
