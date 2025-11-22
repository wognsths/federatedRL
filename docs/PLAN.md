# Federated Offline RL with OptiDICE — Project Plan

**Goal.** Evaluate whether *distribution correction* in offline RL (OptiDICE) reduces non-IID pain in federated learning (FL), compared to naive FedAvg-RL, and quantify the gap vs centralized training.

---

## 0) TL;DR (one-liner)

**Hypothesis:** Local *stationary-distribution ratio* correction (`w = d^π / d_D`) aligns client gradients toward a common target distribution → **Fed+OptiDICE** will be **more stable** and **faster-converging** than **FedAvg-RL** under non-IID data.

---

## 1) Experimental Matrix

### Baselines
- **B1 — FedAvg-RL (naive offline)**  
  Local learner: SAC (continuous control) or TD3. No offline correction.  
  Aggregation: FedAvg.

- **B2 — Fed+OptiDICE (ours)**  
  Local learner: OptiDICE dual (ν, λ, closed-form `w*`) + I-projection policy update.  
  Aggregation: FedAvg **and** ratio-aware weighting (ablation).

- **Oracle references (centralized, non-federated):**  
  **C1 — Centralized SAC**, **C2 — Centralized OptiDICE** on the *union* of all client data (used only as references to measure FL gap).

> **Primary target:** Continuous control (MuJoCo via D4RL).

---

## 2) Datasets & Non-IID Construction

### Base datasets (D4RL, continuous)
- `halfcheetah-medium-replay-v2`, `hopper-medium-replay-v2`, `walker2d-medium-replay-v2`  
- Optional harder set: `…-medium-v2`, `…-random-v2` to stress coverage.

### Clients and splits
- Number of clients: **K ∈ {5, 10}** (default 5).
- **Non-IID regimes** (generate both; report each separately):

  1) **Return-quantile split (label-shift-like)**  
     - Sort trajectories by undiscounted return; split into quantiles.  
     - Assign higher-return trajectories to a subset of clients, low-return to others → behavior policy quality differs across clients.

  2) **State-cluster split (covariate shift)**  
     - Embed states via a frozen encoder (e.g., random 2-layer MLP) and k-means cluster.  
     - Assign clusters unevenly to clients (e.g., client 1: clusters 1–2; client 2: clusters 3–5; …).

  3) *(Optional)* **Action-support skew (support shift)**  
     - Per state, keep only top-p% behavior actions for client A, bottom-q% for client B (by BC likelihood).  
     - Yields different action supports across clients.

- **Client weights:** μ_k ∝ |D_k| (size-proportional) unless otherwise specified.

> We will release the split seeds to reproduce exact partitions. Each regime gets **3 random seeds**.

---

## 3) Methods (precise but concise)

### 3.1 FedAvg-RL (naive offline)
- Local algo: **SAC** (HalfCheetah/Hopper/Walker2d).  
- Replay buffer = local fixed dataset (no environment interaction).  
- No conservative penalty, no ratio correction.  
- **Risk**: overestimation on OOD actions (expected to be unstable on non-IID).

### 3.2 Fed+OptiDICE (ours)
**Local (client k):**
1) **Critic (dual) step**  
   - Potential function ν_θ, scalar λ.  
   - Advantage-like term: `e_ν = r + γ ν(s') − ν(s)`.  
   - Closed-form ratio (KL): `w*(s,a) = exp((e_ν − λ)/α − 1)`, with **log-clip**.  
   - Optimize:  
     `L_k = (1−γ) E_{p0}[ν] + E_{D_k}[ −α f(w*) + w*(e_ν − λ) ]`, using start-of-trajectory states for `p0`.

2) **Actor (I-projection)**  
   - Weighted BC view: minimize `J_k(π) = −E_{(s,a)~D_k}[ w_global(s,a) log π(a|s) ]` plus optional entropy reg.

**Server:**
- Aggregate `(ν, λ)` via **FedAvg** (or ratio-aware weights) to form the broadcast global ratio; actors/behaviors can still average size-weighted.  
- **Ablation:** ratio-aware client weights (see §6.2) apply primarily to the dual aggregation.

**Round workflow (implemented)**
- **Broadcast** current dual/actor/behavior to selected clients.  
- **Local ratio eval**: compute `w_global` on each batch using the broadcast dual; use trajectory starts for `(1−γ)E_{p0}`.  
- **Dual update**: run dual optimization locally; log ratio stats (mean/std/second-moment).  
- **Actor update**: weighted log-likelihood on dataset actions with `w_global`; optional entropy regularization.  
- **Aggregate — two options**:  
  - *Dual-averaging*: server averages `(ν, λ)` (FedAvg or ratio-weighted by ratio stats).  
  - *Reference-buffer*: server shares a small public buffer; clients evaluate `w_i` on it; server weights aggregation using `w_i` stability (ESS/variance). Actor/behavior typically aggregated size-weighted.  
  The aggregated dual defines the next round’s `w_global`.

> **Why this helps non-IID:** Each client pre-aligns to a **shared target stationary distribution** via `w*`, making cross-client gradients point in a more consistent direction before averaging.

---

## 4) Training Protocol

### Global loop (one round)
1) **Broadcast**: server sends `(θν, λ, θπ)` to selected clients.  
2) **Local updates** (parallel on clients):
   - **Critic:** run `E_c` steps of OptiDICE dual (or SAC for baselines).  
   - **Actor:** run `E_a` steps (I-projection for OptiDICE; policy step for SAC).  
   - Return updated parameters or gradients.  
3) **Aggregate**: FedAvg (optionally FedProx / ratio-aware μ_k).  
4) **Repeat** for `R` rounds.

### Key knobs
- Client fraction per round: `C ∈ {1.0, 0.5}` (default 1.0).  
- Local steps: `(E_c, E_a) = (500, 500)` per round (tune per task).  
- Rounds: `R ∈ {50, 100}` (default 100).  
- Batch size: 256; Optimizer: Adam; LR grid: {1e-4, 3e-4}.  
- Discount γ = 0.99; Target update τ = 0.005 (for SAC critics).  
- **OptiDICE α:** grid {0.5, 1.0, 2.0, 5.0}. **Log-clip**(|log w*| ≤ 20).  
- **FedProx β (ablation):** {0, 0.01, 0.1}.  
- Seeds: {0, 1, 2}.

> **Fairness:** Match total update counts and communication rounds across methods.

---

## 5) Evaluation

### Task metrics
- **D4RL Normalized Score** (primary).  
- **Raw episodic return** (for completeness).

### Stability & alignment
- **Perf. variance** across seeds & rounds (std / IQR).  
- **Gradient cosine similarity** across clients (critic & actor).  
- **Rounds-to-X**: communication rounds to reach a fixed normalized score X.  
- **Out-of-support indicator:** `E[(w*)^2]` (proxy for coverage stress, OptiDICE only).

### Communication & compute
- **Model size** (MB), **bytes/round**, **wall-clock** per round.  
- **Client update time** distribution.

### FL gap vs centralized
- Report delta to **C1/C2** (union-data oracle):  
  - `ΔScore = Score_FL − Score_Centralized`  
  - Plot learning curves vs oracles.

### Reporting
- Per task (HalfCheetah/Hopper/Walker2d), per regime (quantile vs cluster), average over seeds.  
- **Tables:** Final score ± std, Rounds-to-X, Bytes-to-X.  
- **Plots:** Learning curves, variance bands, cosine similarity across rounds.

---

## 6) Ablations (important)

### 6.1 Optimization
- **FedAvg vs FedProx vs SCAFFOLD** (for B2).  
- Actor-only aggregation vs Critic-only vs Both (B2).

### 6.2 Ratio-aware aggregation (B2 only)
- **Uniform μ_k** (FedAvg) vs **ESS-weighting**:  
  `ESS_k = (E[w*])^2 / E[(w*)^2]`, set `μ_k ∝ ESS_k`.  
- **Variance-inverse weighting**: `μ_k ∝ (E[(w*)^2])^−1`.

### 6.3 Stability knobs
- Log-clip thresholds for `w*` (10, 15, 20).  
- α-sweeps for OptiDICE.

---

## 7) Implementation Plan & Repo Layout

fed-offline-rl/
├─ docs/
│  ├─ PLAN.md                # this file
│  ├─ PAPERS.md              # reference papers
│  └─ RESULTS.md             # final tables/plots
├─ configs/                  # Hydra/YAML configs
│  ├─ algo/ (sac.yaml, optidice.yaml)
│  ├─ fed/  (fedavg.yaml, fedprox.yaml, scaffold.yaml)
│  ├─ data/ (d4rl_task.yaml, split_regime.yaml)
│  └─ train.yaml             # seed, rounds, batch sizes, etc.
├─ fed/
│  ├─ server.py              # aggregation loop, secure agg hooks
│  ├─ client.py              # local update orchestration
│  ├─ aggregator/
│  │  ├─ fedavg.py
│  │  ├─ fedprox.py
│  │  ├─ ratio_weighted.py   # μ_k via ESS/variance on dual stats
│  │  └─ ratio_buffer.py     # μ_k via ESS/variance on shared buffer w_i
│  └─ utils.py
├─ rl/
│  ├─ sac/ (agent.py, critic.py, actor.py, replay.py)
│  └─ optidice/
│     ├─ dual.py             # ν, λ, L_k, w* closed-form
│     ├─ iproj.py            # I-projection actor update
│     └─ bc.py               # behavior cloning π_{β,k}
├─ data/
│  ├─ d4rl_loader.py         # dataset download & caching
│  ├─ splitting.py           # quantile/cluster/support split
│  └─ samplers.py
├─ metrics/
│  ├─ d4rl_score.py
│  ├─ grad_cosine.py
│  ├─ rounds_to_x.py
│  └─ comms.py               # bytes accounting
├─ scripts/
│  ├─ run_fed.py             # entry point (Hydra)
│  ├─ run_centralized.py
│  └─ analyze.py             # plots, tables
└─ requirements.txt

### Notes
- **Framework:** PyTorch + gymnasium + d4rl.  
- **Logging:** Weights & Biases or MLflow (curves, artifacts).  
- **Reproducibility:** set global seeds, deterministic CuDNN where feasible.

---

## 8) Privacy & Comms

- **No raw data leaves clients.**  
- **Secure Aggregation** option (sum in the encrypted domain).  
- **DP (optional):** per-round gradient clipping + Gaussian noise (track ε).  
- **Bandwidth:** 8/16-bit optimization for parameter exchange; Top-k sparsification (ablation).

---

## 9) Risks & Mitigations (cold, realistic)

- **Naive FedAvg-RL divergence** under heavy non-IID  
  → Fall back to FedProx/SCAFFOLD; reduce LR; increase target smoothing.

- **OptiDICE instability (ν misfit / ratio blow-up)**  
  → Increase α; tighten log-clip; add baseline to `e_ν`; EMA on ν.

- **Communication bottleneck**  
  → Larger local steps (E_c/E_a), compression; smaller client fraction C.

---

## 10) Minimal Math Recap (for implementation)

- **OptiDICE dual (client k):**  
  `e_ν = r + γ ν(s') − ν(s)`,  
  `w* = exp((e_ν − λ)/α − 1)` (KL case), clipped in log-space,  
  `L_k = (1−γ) E_{p0}[ν] + E_{D_k}[ −α f(w*) + w*(e_ν − λ) ]`.

- **I-projection:**  
  `J_k(π) = E_{s~D_k}[ KL(π||π_{β,k}) − E_{a~π}[log w*] ]`.

- **Aggregation:**  
  `θ ← Σ_k μ_k θ_k` (FedAvg), with **μ_k** uniform or ratio-aware (ESS / variance-inverse).

---

## 11) What to Look For (expected qualitative outcomes)

- FedAvg-RL: fast at first but unstable; large variance under strong non-IID.  
- **Fed+OptiDICE:** quicker alignment (higher cosine), better rounds-to-X, smaller FL gap vs centralized OptiDICE — especially in **quantile** non-IID where behavior quality diverges.

---

## 12) Command Examples

```bash
# Centralized references
python scripts/run_centralized.py task=halfcheetah-medium-replay-v2 algo=optidice seed=0

# Federated: Fed+OptiDICE, quantile non-IID, 5 clients
python scripts/run_fed.py task=halfcheetah-medium-replay-v2 \
  split=quantile K=5 fed=fedavg algo=optidice rounds=100 Ec=500 Ea=500 seed=0

# Federated: Naive FedAvg-RL (SAC)
python scripts/run_fed.py task=walker2d-medium-replay-v2 \
  split=quantile K=5 fed=fedavg algo=sac rounds=100 seed=2
```

## 13) Deliverables
- docs/RESULTS.md: final tables (score ± std), rounds-to-X, bytes-to-X.  
- Plots (curves + bands), cosine similarity, FL gap vs centralized.  
- Ablation summaries (FedProx/SCAFFOLD, μ_k strategies, α/clip sweeps).  
- Config packs for exact reproducibility of all runs.
