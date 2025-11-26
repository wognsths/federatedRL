# Running federatedRL on an AWS GPU EC2 (Ubuntu/CUDA)

End-to-end setup for SAC/OptiDICE experiments on AWS EC2 with GPU.

## 0) Instance selection
- Type: start with `g5.xlarge` (A10G 24GB) or `g4dn.xlarge` (T4 16GB). Scale up if needed.
- AMI: use **Deep Learning AMI (Ubuntu 22.04, CUDA 11.x)** to get drivers preinstalled. Otherwise install NVIDIA driver/CUDA manually.
- Disk: 50GB+ gp3.
- Security group: allow SSH (port 22).

## 1) Connect
```bash
ssh -i <key.pem> ubuntu@<public-ip>
```

## 2) System packages
```bash
sudo apt-get update && sudo apt-get install -y \
  git wget curl patchelf libgl1-mesa-glx libosmesa6-dev \
  python3-venv python3-pip build-essential
```

## 3) NVIDIA driver/CUDA check
- Deep Learning AMI: just verify
```bash
nvidia-smi
```
- Plain Ubuntu AMI: install a driver (example)
```bash
sudo apt-get install -y nvidia-driver-525
# reboot if needed, then nvidia-smi
```

## 4) Repo + venv
```bash
git clone <your-repo-url> federatedRL && cd federatedRL
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 5) PyTorch (match CUDA version)
Example for CUDA 11.8:
```bash
pip install torch==2.1.* --index-url https://download.pytorch.org/whl/cu118
```

## 6) MuJoCo runtime
```bash
wget https://mujoco.org/download/mujoco-2.3.5-linux-x86_64.tar.gz
tar -xzf mujoco-2.3.5-linux-x86_64.tar.gz
mv mujoco-2.3.5 ~/.mujoco
echo 'export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco-2.3.5' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUJOCO_PY_MUJOCO_PATH/bin' >> ~/.bashrc
source ~/.bashrc
```

## 7) Python deps (skip pybullet)
```bash
pip install gymnasium hydra-core omegaconf numpy pandas scikit-learn tqdm \
            matplotlib seaborn typer rich mujoco mujoco-py
pip install click termcolor h5py  # quiet d4rl deps
pip install --no-deps 'd4rl @ git+https://github.com/Farama-Foundation/D4RL@master'
```
> pybullet is not required for this repo.

## 8) Env var
```bash
export PYTHONPATH=.
```
(add to ~/.bashrc if desired)

## 9) Smoke tests
```bash
python scripts/run_fed.py --config configs/train_sac_fedavg.yaml --rounds 1 --batch_size 64 --clients 2
python scripts/run_fed.py --config configs/train_optidice_dualavg.yaml --rounds 1 --batch_size 64 --clients 2
```
If MuJoCo is missing or misconfigured, the code falls back to synthetic data (warns but runs).

## 10) Full runs (examples)
```bash
# FedAvg SAC baseline
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

## 11) Troubleshooting
- `ModuleNotFoundError: data`: ensure `export PYTHONPATH=.` or run `python -m scripts.run_fed ...`.
- MuJoCo errors: verify `MUJOCO_PY_MUJOCO_PATH` and `LD_LIBRARY_PATH`.
- d4rl dep warnings (dm-control/mjrl/pybullet): safe to ignore for this repo if MuJoCo/Gymnasium are installed.***
