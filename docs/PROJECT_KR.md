# Federated Offline RL + OptiDICE (요약)

연방 환경에서 오프라인 RL을 실험하는 플레이그라운드다. D4RL(또는 합성) 데이터셋을 비식별(non-IID) 클라이언트 단위로 쪼개고, **FedAvg+SAC** 기본선과 **Fed+OptiDICE** 변형들을 비교한다. 라운드별로 전역 듀얼 $(\nu, \lambda)$을 브로드캐스트하고, 각 클라이언트가 로컬 밀도비 $w_i$로 가중한 정책 학습을 수행한 뒤 다양한 방식으로 파라미터를 집계한다.

## 문제 설정
- 클라이언트 $i$의 오프라인 데이터셋 $\mathcal{D}_i = \{(s, a, r, s', d)\}$는 크기 $N_i$이고 총합 $T = \sum_i N_i$.
- 행동 정책의 방문분포를 $\mu_\beta(s, a) = (1-\gamma) \sum_{t\ge0} \gamma^t \Pr_\beta(s_t=s, a_t=a)$라 둘 때, 목표는 새 정책 $\pi_\theta$의 점유비 $\mu_\pi$에 대한 기대 보상 극대화:

$$
J(\pi_\theta) = \mathbb{E}_{(s,a)\sim \mu_\pi}[r(s,a)], \quad
w_\pi(s,a) = \frac{\mu_\pi(s,a)}{\mu_\beta(s,a)}
$$

- 환경 접근 없이 $\mathcal{D}_i$만으로 $w_\pi$를 근사하기 위해 OptiDICE의 듀얼 파라미터 $(\nu, \lambda)$를 학습한다.

## 로컬 OptiDICE 업데이트 (`rl/optidice`)
각 클라이언트는 전역 듀얼을 받아 로컬 배치마다 다음을 계산한다 ($\alpha$: 온도, $c$: 클립 값):

1) **밀도비**

$$
e_\nu(s,a,s',d) = r + \gamma(1-d)\nu(s') - \nu(s)
$$

$$
w_\phi(s,a) = \exp\left(\mathrm{clip}\left(\frac{e_\nu - \lambda}{\alpha} - 1, -c, c\right)\right)
$$

2) **듀얼 손실(하강)** — DICE 라그랑지안을 음수 부호로 구현:

$$
\mathcal{L}_{\text{dual}} = \mathbb{E}_{\mathcal{D}_i} \left[ -\alpha w_\phi(\log w_\phi - 1) + w_\phi(e_\nu - \lambda) \right] + (1-\gamma)\mathbb{E}_{p_0}[\nu(s_0)]
$$

3) **정책(가중 BC)** — 전역 듀얼로 계산한 $w_\phi$를 고정하고,

$$
\mathcal{L}_{\text{actor}} = -\mathbb{E}_{\mathcal{D}_i} \left[ w_\phi(s,a)\log \pi_\theta(a|s) \right] - \beta\mathbb{E}[\log \pi_\theta(a|s)]
$$

($\beta$는 `entropy_reg`). 이는 $w_\phi$가 큰 상태-행동을 우선적으로 모방하는 감독학습 형태.

4) **행동 클로닝 보조손실**

$$
\mathcal{L}_{\text{BC}} = -\mathbb{E}_{\mathcal{D}_i}[\log q_\psi(a|s)]
$$

로컬 행동 데이터에 대한 정규분포형 BC 모델 $q_\psi$를 안정화용으로 함께 학습한다.

## 연방 루프 (`fed/`)
라운드 $t$마다 서버가 전역 파라미터를 브로드캐스트 → 각 클라이언트가 $K$회 로컬 업데이트 → 서버가 집계.

### 기본 FedAvg / FedProx

$$
\theta^{(t+1)} = \sum_{i=1}^n \frac{N_i}{T}\theta_i^{(t)}
$$

$$
\text{(FedProx)} \quad \theta^{(t+1)} = (1-\beta)\theta_{\text{avg}}^{(t+1)} + \beta\theta^{(t)}
$$

### 비율(dual) 가중 집계 `RatioWeightedAggregator`
- 클라이언트는 $w_\phi$의 통계(`mean`, `std`, `second_moment`)를 함께 전송.
- 유효 표본 크기(ESS) 기반 가중치:

$$
\alpha_i = \max\left(\frac{\mathbb{E}[w_i]^2}{\mathrm{Var}[w_i] + \varepsilon}, \varepsilon\right), \quad
\phi^{(t+1)} = \sum_i \frac{\alpha_i}{\sum_j \alpha_j}\phi_i^{(t)}
$$

`dual_only_weights=true`이면 $(\nu, \lambda)$에만 $\alpha_i$를, 나머지 파라미터는 $N_i$ 비례로 평균한다.
- **$\lambda$ 정규화**(선택): 전역 평균 비율을 1로 맞추기 위해

$$
\lambda \leftarrow \lambda + \alpha \log (\mathbb{E}[w_i])
$$

### 공유 버퍼 비율 집계 `BufferRatioAggregator`
- 서버가 전역 데이터에서 참조 배치 $\mathcal{B}$를 샘플링하고, 각 클라이언트가 동일한 $\mathcal{B}$ 위에서 $w_i$를 평가해 가중치 계산:

$$
\alpha_i = \frac{\mathbb{E}_{\mathcal{B}}[w_i]^2}{\mathrm{Var}_{\mathcal{B}}[w_i] + \varepsilon}
$$

- 동일한 참조 분포를 사용하므로 데이터 지원이 다른 클라이언트 간 가중치 비교를 더 직접적으로 만든다.

### 하이브리드 `HybridRatioFedAvgAggregator`
- 초기에는 비율 가중 집계를 사용하다가 점진적으로 FedAvg로 전환:

$$
\phi^{(t+1)} = (1-\gamma_t)\phi_{\text{ratio}}^{(t+1)} + \gamma_t\phi_{\text{avg}}^{(t+1)}
$$

$$
\gamma_t = \mathrm{clip}\left(\frac{t - t_{\text{start}}}{T_{\text{warm}}}, 0, 1\right)
$$

- 비율 기반 초기 가속과 FedAvg의 후기 안정성을 절충한다.

## 데이터 파이프라인 (`data/`)
- **로더**: `gym`/`d4rl` 사용, 실패 시 관측·행동 차원을 맞춘 합성 가우시안 데이터로 자동 대체. 상태는 기본적으로 Z-정규화.
- **분할**:
  - `quantile`: 트래젝토리 리턴을 정렬해 클라이언트별 성능 구간을 나누는 방식.
  - `cluster`: MiniBatchKMeans로 상태를 클러스터링한 뒤 클러스터 단위로 클라이언트에 할당.
  - `support`: 행동 노름의 상/하위 분위수를 분리해 고/저 동적 범위를 가진 클라이언트를 만든다.
- **샘플러**: 각 클라이언트는 고정 시드를 가진 미니배치 샘플러를 사용하고, $(1-\gamma)\mathbb{E}_{p_0}[\nu]$ 항을 위해 에피소드 시작 상태도 함께 샘플한다.

## 실행 스크립트
- 연방 실험: `python scripts/run_fed.py --config configs/train.yaml`
  `--algo optidice --fed ratio_weighted --data hopper_medium_replay` 등으로 알고리즘/집계/데이터를 교체.
- 중앙집중식 오라클: `python scripts/run_centralized.py --algo optidice --data halfcheetah_medium_replay`.
- `configs/train_optidice_*.yaml`은 $\lambda$ 정규화, ESS 가중, 버퍼 가중, 하이브리드 전환 등 논문 실험 대비 세부 변형을 캡슐화한다.

## 로그·지표
- 클라이언트 측 비율 통계(`ratio_mean/std/max/second_moment`)와 손실(`actor`, `dual`, `bc`)을 라운드 평균으로 기록.
- 온라인 평가가 켜져 있으면 D4RL 정규화 점수

$$
\text{score} = \frac{\bar{R} - R_{\text{random}}}{R_{\text{expert}} - R_{\text{random}}} \times 100
$$

을 라운드별로 추가한다 (`metrics/d4rl_score.py` 참조).
