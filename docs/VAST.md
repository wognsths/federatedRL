# Running federatedRL on a Vast.ai GPU server (Ubuntu/CUDA)

Minimal playbook to spin up and run the SAC/OptiDICE federated experiments on a fresh Vast.ai instance.

## 0) Launch hints
- Choose an Ubuntu image with CUDA drivers installed (e.g., official NVIDIA CUDA 11.8/12.x base).
- Request one GPU (Ampere+), at least 4 CPU cores, 20+ GB RAM, and 30+ GB disk.
- Enable SSH and note the connection string (e.g., `ssh -p <port> root@<ip>`).

## 1) System packages
```bash
apt-get update && apt-get install -y \
    git wget curl patchelf libgl1-mesa-glx libosmesa6-dev \
    build-essential python3 python3-venv python3-pip
```

## 2) Clone repo and create venv
```bash
git clone <your-repo-url> federatedRL && cd federatedRL
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 3) PyTorch with CUDA
- Pick the CUDA wheel matching the image (example: CUDA 11.8):
```bash
pip install torch==2.1.* --index-url https://download.pytorch.org/whl/cu118
```

## 4) MuJoCo runtime
```bash
wget https://mujoco.org/download/mujoco-2.3.5-linux-x86_64.tar.gz
tar -xzf mujoco-2.3.5-linux-x86_64.tar.gz
mv mujoco-2.3.5 ~/.mujoco
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco-2.3.5
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUJOCO_PY_MUJOCO_PATH/bin
```
Add the two exports to your shell RC if you want them permanent.

## 5) Python deps (lean install, skip pybullet)
```bash
pip install gymnasium hydra-core omegaconf numpy pandas scikit-learn tqdm \
            matplotlib seaborn typer rich mujoco mujoco-py
pip install click termcolor h5py  # quiets d4rl dep warnings
# Install d4rl without pulling extra deps (we already installed what we need)
pip install --no-deps 'd4rl @ git+https://github.com/Farama-Foundation/D4RL@master'
```
> pybullet is not required for this repo; skip it unless you really need it.

## 6) Run smoke tests
Always set `PYTHONPATH=.` to resolve local imports.
```bash
export PYTHONPATH=.
# FedAvg SAC sanity
python scripts/run_fed.py --config configs/train_sac_fedavg.yaml --rounds 1 --batch_size 64 --clients 2
# OptiDICE dual-avg sanity
python scripts/run_fed.py --config configs/train_optidice_dualavg.yaml --rounds 1 --batch_size 64 --clients 2
```
Metrics will be written under `outputs/.../metrics.json`. If MuJoCo fails, the code falls back to synthetic data (you’ll see a warning).

## 7) Full runs (examples)
```bash
# SAC FedAvg baseline
python scripts/run_fed.py --config configs/train_sac_fedavg.yaml
# OptiDICE variants
python scripts/run_fed.py --config configs/train_optidice_dualavg.yaml
python scripts/run_fed.py --config configs/train_optidice_lambda_norm.yaml
python scripts/run_fed.py --config configs/train_optidice_dual_weighted.yaml
python scripts/run_fed.py --config configs/train_optidice_stable.yaml
# Centralized references
python scripts/run_centralized.py --config configs/train_optidice_dualavg.yaml --algo optidice
python scripts/run_centralized.py --config configs/train_sac_fedavg.yaml --algo sac
```

## 8) Troubleshooting
- `ModuleNotFoundError: data`: ensure `PYTHONPATH=.` or run with `python -m scripts.run_fed ...`.
- MuJoCo errors: re-check `MUJOCO_PY_MUJOCO_PATH` and `LD_LIBRARY_PATH`.
- d4rl warnings about missing dm-control/mjrl/pybullet can be ignored if you’re using MuJoCo/Gymnasium; they’re not needed for this repo.

## 9) Outputs
Logs/metrics are saved under `outputs/` with timestamped folders; `metrics.json` contains round-level aggregates. Copy them off the instance before shutting it down if needed.
