# Federated Offline RL with OptiDICE

This repository implements the project plan from `docs/PLAN.md`: a modular playground to compare naive FedAvg-RL (SAC) and the proposed Fed+OptiDICE algorithm under non-IID offline datasets derived from D4RL.

## Features
- **Data pipelines** — D4RL loader with synthetic fallbacks, quantile / cluster / support shift client splitters, and reusable samplers.
- **Algorithms** — PyTorch implementations of offline SAC and OptiDICE (dual + I-projection) sharing a common interface. OptiDICE uses a global dual `(ν, λ)` broadcast so clients compute `w_global` locally for a weighted BC-style actor update.
- **Federated loop** — Configurable clients, FedAvg/FedProx/ratio-aware aggregators, and logging utilities for round-level metrics. Two ratio-aware options exist: parameter-based (dual averaging) and reference-buffer-based (aggregate using `w_i` evaluated on a shared buffer).
- **Scripts** — `run_fed.py` for federated experiments, `run_centralized.py` for oracle baselines, and `analyze.py` for quick summaries.

## Getting Started
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
If MuJoCo/D4RL are unavailable, the loader automatically falls back to synthetic Gaussian datasets so the pipeline remains testable.

## Running Experiments
Federated training (defaults to HalfCheetah medium-replay, quantile split, FedAvg + SAC). Every config that sets `eval_interval` / `eval_episodes` now triggers online rollouts, so round logs include raw returns and D4RL-normalized scores:
```bash
python scripts/run_fed.py --config configs/train.yaml
```
Override the algorithm or federation method:
```bash
python scripts/run_fed.py --algo optidice --fed ratio_weighted --data hopper_medium_replay
```
The OptiDICE federated path broadcasts a global dual (ν, λ), evaluates ratios on each client batch, and trains the actor with a weighted log-likelihood using those ratios. Dual training uses start-of-trajectory samples for the `(1−γ)` term.
Centralized oracle baselines:
```bash
python scripts/run_centralized.py --algo optidice --data halfcheetah_medium_replay
```

### Recommended workflow (YAML presets)
1. **Centralized SAC oracle** — `python scripts/run_centralized.py --config configs/train_sac_fedavg.yaml`  
   Uses the same `eval_interval` / `eval_episodes` settings so reward and normalized score curves match the federated runs.
2. **FedAvg SAC baseline** — `python scripts/run_fed.py --config configs/train_sac_fedavg.yaml`  
   100 rounds, all 5 clients each round, reward validation every 10 rounds.
3. **Proposed Fed+OptiDICE variants** — run the configs described in `docs/PLAN.md`:
   - `configs/train_optidice_dualavg.yaml` (FedAvg aggregation of dual + actor)
   - `configs/train_optidice_lambda_norm.yaml` (`fed_overrides` flips on λ-normalization)
   - `configs/train_optidice_dual_weighted.yaml` (λ-normalization + dual-only ratio weights)
   - `configs/train_optidice_stable.yaml` (step 3 with the higher-α “stable” OptiDICE config)
   - `configs/train_optidice_ratio_buffer.yaml` (ratio-buffer aggregation experiment)

Each YAML keeps `rounds: 100` and logs evaluation summaries every 10 rounds. `fed_overrides` keys merge into the base federation YAML (e.g., `configs/fed/ratio_weighted.yaml`) so you can tweak aggregation tricks without duplicating entire configs.

### Federated OptiDICE — how it works
- **Broadcast**: server sends the current dual parameters `(ν, λ)` and actor/behavior nets to all selected clients.
- **Local ratio evaluation**: each client uses the broadcast dual to compute `w_global(s,a) ≈ exp((r + γ ν(s') − ν(s) − λ)/α − 1)` on its mini-batches (no data leaves the client). Initial states are included for the `(1−γ)E_{p0}[ν]` term.
- **Dual update**: clients update `(ν, λ)` with the OptiDICE dual loss using their data; ratio stats (`mean`, `std`, `second_moment`) are logged.
- **Actor update (weighted BC)**: clients train the actor with `−E_{(s,a)~D}[ w_global(s,a) log π(a|s) ]` (optional entropy reg), i.e., a supervised policy fit weighted by the shared ratio.
- **Aggregate (two modes)**:
  - *Dual-averaging*: server aggregates `(ν, λ)` (FedAvg or ratio-weighted) and broadcasts; actor/behavior can be size-weighted FedAvg. `w` itself is not aggregated; `w_global` is defined by the dual and computed locally on each client.
  - *Reference-buffer*: server builds a small shared buffer; clients evaluate `w_i` on that buffer, and the server weights aggregation using `w_i` stability (ESS/variance).
- **Repeat**: broadcast the new dual/actor/behavior and continue.
Summaries / rounds-to-X:
```bash
python scripts/analyze.py outputs/fed_* --target 0.0
```
Results JSON files land in `outputs/` with timestamped run folders. Place aggregated tables/figures inside `docs/RESULTS.md` as experiments complete.

### Recipe: 4-step OptiDICE ablation (run after installing PyTorch)
1) Dual averaging (baseline):  
   `python scripts/run_fed.py --config configs/train_optidice_dualavg.yaml`
2) λ normalization only:  
   `python scripts/run_fed.py --config configs/train_optidice_lambda_norm.yaml`
3) λ normalization + dual-only ratio weights:  
   `python scripts/run_fed.py --config configs/train_optidice_dual_weighted.yaml`
4) Step 3 + stability (higher α, tighter clip):  
   `python scripts/run_fed.py --config configs/train_optidice_stable.yaml`

## Repo Layout
```
configs/        Hydra-like YAML configs for algos, federation, and datasets
data/           D4RL loader, non-IID splitters, samplers
fed/            Client/server orchestration + aggregators (FedAvg, FedProx, ratio-aware)
metrics/        Normalized score, cosine similarity, rounds-to-X, comms accounting
rl/             Offline SAC, OptiDICE implementations + shared networks
scripts/        CLI entry points for federated runs, centralized baselines, and analysis
docs/           PLAN, paper links, and RESULTS placeholders
```
