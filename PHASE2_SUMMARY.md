# Phase 2 작업 정리 — Type-II-like PMI Codebook 추가

작성일: 2026-05-21 · 갱신: 2026-06-12 (§10 PPO 구현 정리 추가) · 경로: `/home/MYH/ML_DRL_Scheduler/` · 현재 default codebook

---

## 1. 개요

Phase 1 환경 위에 PMI codebook을 **simplified Type-II-like sparse atomic feedback**으로
확장. 기존 `random_unit_norm`(M=256)은 ablation/비교용으로 보존하고
`type2_sparse_56bit`를 **default**로 설정.

이건 **simplified Type-II-like model**이며 **exact 3GPP Rel-15 Type II 구현이 아님**
(자세한 차이는 §8 한계 참고).

---

## 2. 동기

| 문제 (Phase 1 state) | 원인 |
|---|---|
| 평균 MU-SINR이 ~3 dB로 낮음 (calibration target 10 dB 대비) | random_unit_norm(M=256)이 32차원 채널 방향을 거칠게 양자화 |
| direction_corr ~0.18 | random codeword가 채널 구조와 무관한 isotropic 방향들 |
| MU-MIMO RZF가 stream 간 간섭을 잘 nulling 못 함 | BS의 부정확한 `h_hat`으로 RZF가 무의미한 nulling 방향 |
| retx_drop 비율 ~0.13 | MU 간섭 → SINR ↓ → packet 실패 |

교수님 미팅 결과 → Type-II로 가자 결정 → 단 3GPP Rel-15 정확 구현 대신 **simplified**.

---

## 3. 사양 (교수님과 사전 확정된 8개 항목)

| # | 사양 |
|---|---|
| 1 | `CQI_true`는 quantized direction 기반 SU SNR: `log₂(1 + P_r · |h_true^H · h_dir_hat|² / σ²)` |
| 2 | B_tx는 이 CQI 기반 → MU-MIMO에선 optimistic, β_rate backoff는 config로 유지 |
| 3 | `h_dir_hat_fb`는 BS-side cache이며 실제 PMI payload는 atom_idx/amp_idx/phase_idx (56 bit) |
| 4 | Sionna PanelArray port ordering 확인 → [pol1: 0–15, pol2: 16–31]로 우리 atom 정렬과 일치 |
| 5 | 4개 distinct atom을 OMP가 선택, bit accounting은 단순화하여 4×9 bit |
| 6 | direction_corr / chordal_dist / phase_aligned_NMSE 3종 metric 로깅 |
| 7 | "Simplified Type-II-like, NOT exact 3GPP Rel-15" 코드 docstring과 문서에 명시 |
| 8 | Correctness-first 구현 → 검증 → 두 모드 비교 run |

---

## 4. 알고리즘

### 4.1 Dictionary 구성

| 항목 | 값 | 의미 |
|---|---|---|
| BS panel | N1=4, N2=4, dual-pol, 32 ports | Sionna PanelArray와 동일 |
| Sionna port ordering | pol1: 0–15, pol2: 16–31 | block-stacked |
| Oversampling | O1=4, O2=4 | DFT beam grid의 각도 해상도 |
| 2D DFT beam 수 (per pol) | 16 × 16 = 256 | (N1·O1) × (N2·O2) |
| Total atoms | 512 | 256 beam × 2 polarization |
| 각 atom dim | 32 | C^{N_t} unit-norm |

**DFT beam 정의** (편파당):
```
b_{m,n}[i·N2 + j] = (1/√(N1·N2)) · exp(j·2π·m·i/(N1·O1)) · exp(j·2π·n·j/(N2·O2))
  m ∈ {0..15}, n ∈ {0..15}, i ∈ {0..3} (row, y축), j ∈ {0..3} (col, z축)
```

**Polarization embedding** (32-dim 공간으로):
```
pol0 atom = [b_{m,n};  0_{16}]   ∈ C^{32}   →   atoms 0..255
pol1 atom = [0_{16};  b_{m,n}]   ∈ C^{32}   →   atoms 256..511
```

### 4.2 UE-side quantization (OMP + LS + Q)

각 (UE u, RBG r) channel `h_true[u,r] ∈ C^32`에 대해:

**OMP greedy + LS step (L = 4 iteration)**:
```
residual ← h_true
selected ← ∅
for m = 1..L:
    idx_m = argmax_{g ∉ selected} |A[g]^H · residual|²
    selected ← selected ∪ {idx_m}
    A_S = atoms 'selected' as columns,  shape [32, m]
    α_LS = (A_S^H A_S)^{-1} · A_S^H · h_true    (LS over selected so far)
    residual = h_true - A_S · α_LS
```

**최종 α_LS ∈ C^L**: 4개 atom에 대한 complex coefficient.

**Amplitude quantization** (per-(UE,RBG) max-normalized):
```
ρ_m = |α_LS[m]|
ρ_norm_m = ρ_m / max_k ρ_k    ∈ [0, 1]
amp_idx_m = round(7 · ρ_norm_m)    ∈ {0, 1, ..., 7}    (3-bit)
```

**Phase quantization (QPSK)**:
```
φ_m = angle(α_LS[m]) mod 2π
phase_idx_m = nearest of {0, π/2, π, 3π/2}    ∈ {0, 1, 2, 3}    (2-bit)
```

### 4.3 BS-side reconstruction

UE는 인덱스만 보고. BS는 동일 dictionary `A`로 재구성:
```
h_dir_unnorm = Σ_{m=1..L} (amp_idx_m / 7) · exp(j · phase_idx_m · π/2) · A[atom_idx_m]
h_dir_hat    = h_dir_unnorm / ||h_dir_unnorm||                    (unit-norm)
```

UE-side와 BS-side direction이 **bit-exact 일치** 보장 (둘 다 같은 공식·같은 dictionary).

### 4.4 h_hat for RZF precoding

CQI는 별도 feedback. BS는 (CQI, direction)으로 채널 추정값 복원:
```
SNR_hat   = 2^CQI_fb − 1
g_hat     = SNR_hat · σ² / P_r
h_hat     = √g_hat · h_dir_hat        ∈ C^32
```
이 `h_hat`이 RZF precoder 계산에 들어감. SINR 평가는 여전히 `h_true`로.

### 4.5 Feedback payload 구조 (per UE-RBG)

| Field | bits | count | 총 bits |
|---|---|---|---|
| atom_idx | 9 | 4 | 36 |
| amp_idx | 3 | 4 | 12 |
| phase_idx | 2 | 4 | 8 |
| **PMI total** | | | **56** |
| CQI (continuous SE) | (별도 feedback) | | |

K=16 UE × 8 RBG = 128 (UE,RBG) per slot → 56 × 128 = 7168 bit per slot (UE→BS PMI 보고).

---

## 5. 구현 변경 사항

| 파일 | 변경 내용 |
|---|---|
| `config.py` | `pmi_mode`, `type2_O1/O2/L/amp_bits/phase_bits` 필드 추가. **default = type2_sparse_56bit** |
| `codebook.py` | `Type2SparseCodebook` 클래스 신규 (~150 줄). `Type2PMI` dataclass. `make_codebook()` 팩토리. 두 모드 공통 인터페이스 `quantize` / `reconstruct` / `direction_from_channel` |
| `csi.py` | `generate_true_csi`가 (raw_pmi, direction, cqi) 반환. `CSIFeedbackBuffer`가 `direction_fb`를 공통 인터페이스로 노출 |
| `phy.py` | `calibrate_noise`은 `codebook.direction_from_channel` 사용, `reconstruct_h_hat`은 direction을 직접 받음 |
| `env.py` | `make_codebook` 팩토리 사용, `orthoscore_all`은 `direction_fb` 사용, `direction_corr` 매 slot 로깅 |
| `metrics.py` | `direction_corr` / `chordal_dist` / `phase_aligned_nmse` 3종 추가 |
| `run_phase1.py` | `compare_pmi_modes`로 두 모드 baseline 자동 비교 |

`random_unit_norm`은 그대로 동작, `cfg.pmi_mode`로 전환 가능.

---

## 6. 알고리즘 검증 (14개 자동 테스트)

`python codebook.py` 실행 시 `__main__`에서 자동 수행:

| # | 검증 항목 | 결과 |
|---|---|---|
| T1 | 512 atom 전부 unit-norm | PASS |
| T2 | Polarization block 구조 (pol0 in [0:16], pol1 in [16:32]) | PASS |
| T3 | pol0 atoms ⊥ pol1 atoms (inner product = 0 정확) | PASS |
| T4 | 비-oversampled DFT sub-grid (16 atoms) orthonormality | PASS |
| T5 | In-dictionary channel → direction_corr > 0.999, atom 정확 회수 | PASS |
| T6 | 합성 L-atom mixture → direction_corr 평균 0.87 | PASS |
| T7 | 500회 random trial 모두 L=4 distinct atoms | PASS |
| T8 | amp_idx ∈ [0,7], phase_idx ∈ [0,3] | PASS |
| T9 | **사양 핵심**: max amplitude atom은 항상 amp_idx = 7 | PASS |
| T10 | 재구성 direction은 항상 unit-norm | PASS |
| T11 | QPSK 위상 양자화 {0, π/2, π, 3π/2} → [0, 1, 2, 3] 정확 매핑 | PASS |
| T12 | **BS reconstruction = UE-side direction (bit-exact)** | PASS |
| T13 | **CQI ↔ h_hat magnitude 역산 정확** | PASS |
| T14 | Type-II > Random direction_corr (synthetic: 0.18 → 0.43) | PASS |

---

## 7. 결과 (Phase 1 환경에서 두 모드 비교)

### 7.1 Baseline 별 비교 (debug, 3 episodes)

**random_unit_norm (M=256, ablation):**
| baseline | reward | thrpt | comp | retxdrop | SINR | dirCorr |
|---|---|---|---|---|---|---|
| Random | 575.9 | 26.5 | 0.805 | 0.136 | 4.0 dB | 0.18 |
| CQI-greedy | 592.3 | 27.0 | 0.829 | 0.120 | 3.7 dB | 0.18 |
| PF | 592.9 | 26.8 | 0.818 | 0.132 | 3.1 dB | 0.18 |
| Deadline-PF | 601.4 | 27.2 | 0.818 | 0.139 | 3.5 dB | 0.18 |
| **SUS+PF** | **610.7** | 27.2 | **0.838** | 0.115 | 3.8 dB | 0.18 |

**type2_sparse_56bit (default):**
| baseline | reward | thrpt | comp | retxdrop | SINR | dirCorr |
|---|---|---|---|---|---|---|
| Random | 821.5 | 35.0 | 0.907 | 0.056 | 8.3 dB | 0.885 |
| CQI-greedy | 842.0 | 35.5 | 0.911 | 0.054 | 9.1 dB | 0.884 |
| **PF** | **877.7** | **36.9** | 0.910 | 0.047 | 8.6 dB | 0.886 |
| Deadline-PF | 825.6 | 34.8 | 0.909 | 0.048 | 8.4 dB | 0.886 |
| SUS+PF | 848.8 | 35.5 | **0.918** | **0.042** | 8.8 dB | 0.885 |

### 7.2 평균 변화 요약 (random → type2)

| 지표 | Random | Type-II | 변화 |
|---|---|---|---|
| direction_corr | 0.18 | **0.88** | ×5 |
| chordal_dist | 0.82 | 0.12 | ×7 감소 |
| phase_aligned_NMSE | 1.14 | 0.12 | ×10 감소 |
| 평균 MU SINR | ~3.7 dB | **~8.6 dB** | +5 dB |
| 평균 throughput | ~27 Mbps | ~35.5 Mbps | +30% |
| 평균 completion | 0.82 | 0.91 | +9 pp |
| 평균 retx_drop | 0.13 | 0.05 | **−60%** |

### 7.3 Sanity check (default mode 기준)

| 항목 | random default | **type2 default (현재)** |
|---|---|---|
| 평균 SINR | 2.9 dB | **10.0 dB** (calibration target과 일치) |
| SINR 범위 | [−5.2, 39.0] dB | [−6.2, 35.0] dB |

### 7.4 CSI aging 효과 (default mode 기준)

stale CSI가 전송 실패에 미치는 영향이 type2에서 훨씬 선명함:

| p_csi | random default | **type2 default (현재)** |
|---|---|---|
| 1.0 (fresh) | retx_drop 0.082 / SINR 5.86 dB | retx_drop **0.027** / SINR 8.06 dB |
| 0.2 (stale, default) | retx_drop 0.119 / SINR 5.98 dB | retx_drop **0.052** / SINR 8.80 dB |
| **stale로 인한 retx_drop 증가율** | +45% | **+93%** |

→ Type-II default가 교수님의 stale-CSI 연구 의도(직접적인 mismatch-driven failure)를 더 잘 반영.

---

## 8. 한계 — Simplified vs 3GPP Rel-15 Type-II

이 구현은 **simplified Type-II-like**이며 **exact 3GPP Rel-15 Type-II가 아님**.
두 가지 단순화:

### 8.1 Polarization 처리
| 측면 | 우리 simplified version | 3GPP Rel-15 표준 |
|---|---|---|
| beam selection | L=4 atom이 두 polarization에서 자유 선택 | L개 beam 인덱스를 두 polarization이 공유 |
| 자유도 | 한 polarization에 4개가 몰릴 가능성 | per-pol amp/phase 따로 보고 (pol 균형 자동) |
| feedback 효율 | 인덱스 9 bit (256 beam × 2 pol = 512) | 인덱스 8 bit (256 beam) + per-pol amp/phase |

### 8.2 Amplitude quantization 테이블
| 측면 | 우리 | 3GPP Rel-15 |
|---|---|---|
| 양자화 방식 | 균등 `round(7·ρ_norm)/7` | 비균등 표 (예: {0, 1/√64, 1/√32, ..., 1/√2, 1}) |
| 비트 수 | 3 bit (WB amp만) | 3 bit (WB) + 1 bit (SB 차분, 옵션) |

### 8.3 Subband 구조
| 측면 | 우리 | 3GPP Rel-15 |
|---|---|---|
| Granularity | RBG별로 56-bit PMI 따로 보고 | WB beam + per-SB amp/phase 구조 |

### 8.4 표준 구현으로 가는 데 필요한 작업

구조는 그대로 두고 위 단순화를 풀면 됨:
- `Type2SparseCodebook`을 polarization-shared 구조로 변경 (L=4 beam × 2 pol coefficient)
- 양자화 함수를 3GPP TS 38.214 §5.2.2.2.3 표 lookup으로 교체
- (선택) Subband WB/SB 분리 추가

예상 추가 작업: **1~2일**.

---

## 9. 다음 단계 — 실제 PPO Phase로 진행 가능

Type-II default로 환경이 준비되었음:

- 평균 MU SINR이 calibration target(10 dB)과 일치 → 의미 있는 동작점
- direction-fidelity-driven failure 메커니즘이 살아 있어 stale-CSI 연구 의도 부합
- direction_fb (32-dim complex)를 그대로 PPO actor의 ScoreNet 입력 표현으로 사용 가능:
  ```
  PMIRep[u, r] = concat(Re(direction_fb[u, r]), Im(direction_fb[u, r]))   ∈ R^64
  ```

→ PPO 구현 완료. 설계 정리는 아래 §10.

---

## 10. PPO 구현 정리 (2026-05-27 구현 · 2026-06-11~12 전수 재검토 완료)

Phase 1 환경을 그대로 둔 채 `policy.py`(actor/critic), `ppo.py`(GAE + PPO update),
`train_phase2.py`(학습 루프), `eval_phase2.py`(체크포인트 평가)를 추가.
Phase 1 코드 스냅샷은 `phase1_backup/`에 동결. `policy.py`가 인용하는
"Phase 2 spec Section 4"의 masking 규칙은 §10.3에 해당.

### 10.1 네트워크 구조 (policy.py)

| 구성요소 | 입출력 | 입력 구성 |
|---|---|---|
| SharedEncoder | [K,R,70] → [K,R,64] (hidden 128-128) | PMIRep 64 (direction_fb Re/Im) + CQI/8 + Age/Age_max + backlog/B_norm + deadline/D_max + avg_thr/B_norm + active flag |
| ScoreNet | 71 → UE별 logit | e[:,r,:] 64 + scalar 7 (OrthoScore, in-slot 배정 수/R, 남은 budget/B_norm, 예측 B_tx/B_norm, r/R, l/L, \|S_r\|/L) |
| NoUserHead | 68 → no-user logit | RBG 평균 embedding 64 + scalar 4 (r/R, l/L, \|S_r\|/L, valid UE 비율) |
| ValueHead (critic) | 134 → V(s) | e mean-pool 64 + max-pool 64 + global scalar 6 (active 비율, pending-retx 비율, mean Age, mean deadline, mean backlog, valid 비율) |

- actor/critic은 encoder 공유 (`share_critic_encoder=True`). MLP는 ReLU,
  마지막 layer 무활성화.
- **critic은 학습 전용** (advantage 계산) — 행동 선택·평가에는 관여하지 않음.

### 10.2 Autoregressive decode (slot당 32 sub-action)

- Layer-major 순서: l=0의 RBG 0..7 → l=1의 RBG 0..7 → … (8×4=32 위치).
- 재전송 고정 위치(`fixed_mask`)는 actor가 건너뜀 (policy decision에서 제외).
- 각 free 위치에서 [K+1] masked categorical (no-user + UE 1..K) 샘플링.
- decode 중 UE별 commit budget(`temp_uncommit`)을 위치마다 차감하여 이후
  sub-action의 validity와 B_tx 예측에 반영 (slot 내 autoregressive 일관성).
- PPO replay는 저장된 action_sequence를 teacher-forcing으로 재실행해 현재
  파라미터 기준 log-prob/entropy/value 재계산 (decode와 동일 masking을 assert로 검증).

### 10.3 PPO actor masking 규칙 (Phase 1 §4.7 action 제약에 추가)

- UE valid = active ∧ 남은 budget > ε ∧ min(budget, 예측 B_tx) ≥ ε ∧ 해당 RBG 미선택.
- **no-user 제한 규칙**: S_r이 비어 있고 valid UE가 1명 이상이면 no-user 선택 불가
  — 열린 RBG의 첫 stream에는 가능하면 반드시 UE를 배정. (env 자체는 no-user를
  어디서나 허용 — 이 제한은 PPO actor의 action space에만 적용.)
- no-user 선택 시 해당 RBG 닫힘(이후 layer 자동 no-user) — env closure 규칙과 동일.
- 전부 invalid면 no-user 강제 (entropy/log-prob 기여 0).

### 10.4 환경 변경: 재전송 compaction (Phase 1 §4.6 위치 고정 규칙 대체)

- Phase 1: NACK unit은 동일 (RBG, layer)에 고정.
- **Phase 2: RBG만 고정**, 매 slot 시작 시 pending unit들을 (이전 layer, unit_id)
  순으로 layer 0부터 **front-compaction** 재배치. actor/baseline은 compaction 후의
  `fixed_allocation`/`fixed_mask`/`fixed_unit_map`/`initial_S_r`을 관측으로 받음.
- RBG당 pending > L_max인 경우(현 불변식상 unreachable)는 deadline 급한 unit 우선
  보존, 초과 packet은 drop — miss와 동일 penalty (방어적 safe mode, §4.7 reward 참고).

### 10.5 PPO 설정 (ppo.py / config.py)

| 항목 | 값/방식 |
|---|---|
| Rollout | 1 episode = 1 update (num_envs=1) |
| GAE | γ=0.99, λ=0.95; episode 종료(done) → bootstrap V=0 |
| Ratio | **macro**: slot의 policy decision log-prob 합으로 exp(new−old) |
| Clip | ε=0.2 (macro ratio에 적용) |
| Advantage 정규화 | **per-rollout** (episode 전체 기준 1회) |
| Loss | policy + 0.5·value − 0.01·entropy_mean (per-decision 평균) |
| Epoch / minibatch | 4 epochs × minibatch 128(debug)/256(main), update별 shuffle seed |
| Gradient | minibatch 평균 누적 후 clip_grad_norm 0.5, Adam lr 3e-4 |
| KL 로깅 | k3 estimator: mean((ratio−1) − log ratio) |
| Eval | 10 update마다 held-out seed(10000+) 3 episode, PPO(argmax) + baseline 5종 |
| Checkpoint | best.pt(최고 eval reward) + latest.pt(10 update마다) |

### 10.6 성능 최적화 (2026-06-11 적용, 결과 비트 동일 검증 완료)

- `channel.py`: episode 채널 캐시를 LRU(cap 16)로 제한 — main run 기준 무한
  증가(~131 GB 추정) 방지, eval seed 재방문은 캐시 hit.
- `csi.py`/`phy.py`/`env.py`: episode 전체 true CSI(PMI/direction/CQI/σ²)를 reset
  시 per-slot loop로 1회 선계산(`EpisodeCSI`), slot별 OMP 재실행 제거 + seed
  재방문 시 재사용(env 내부 LRU cap 16). actor 관측 경로 불변 (미래 CSI 노출 없음).
- 검증: SUS+PF 스냅샷 diff 완전 일치, slot별 lookup array_equal, σ² 정확 일치,
  OMP 호출 횟수 (첫 reset=T / slot 중 0 / 재방문 0). main run 기준 ~20시간 절감 추정.

### 10.7 전수 리뷰 후속 수정 (2026-06-12 적용·검증 완료)

2026-06-12 전수 리뷰(critical 0건, 코어 수식 전수 검증 통과)에서 나온 수정 패키지.
전부 적용·검증 완료:

- `eval_phase2.py`/`metrics.py`: 최종 평가를 held-out seed(10000+)로 변경
  (`metrics.evaluate(..., episode_offset=10000)`) — train-test 분리.
- `train_phase2.py`: `--resume`(모델/옵티마이저/update/best_eval_reward/torch·CUDA
  RNG 상태 복원, CSV append), Ctrl+C·SIGTERM(docker stop 포함) 시 마지막 완료
  update의 latest.pt 저장(try/finally), 원자적 checkpoint 쓰기(.tmp → os.replace),
  `--run_name` 재사용 거부(fail-fast), 매 update CSV flush, num_updates ≤ 10000 가드.
- `ppo.py`: macro log-ratio ±20 클램프(float32 exp overflow → NaN gradient 방지);
  **GAE를 truncation bootstrap으로 변경** — 고정 길이 종료를 terminal(V=0)이 아닌
  time-limit 절단으로 보고 V(s_T)로 bootstrap (partial-episode bootstrapping).
  critic 시간 특징 추가(134→135) 대신 채택: actor/critic 입력 불변, 시간 정보가
  네트워크 어디에도 안 들어가며, 기존 체크포인트 호환 유지.
- `env.py`: commit 후 잔여 backlog가 1 bit(ε) 미만이면 해당 unit에 흡수 —
  완료 불가·HOL 고착 edge case 제거.
- `baselines.py`: Random baseline seed를 `cfg.seed + 100003`으로 분리
  (episode RNG 계열 7919·idx, shuffle 계열 31337+u와 비충돌).

검증: GAE 수계산 대조(bootstrap on/off), 잔여 backlog·offset·seed 타깃 테스트,
모듈 스모크 5종, 10-update 통합 학습(eval/best/latest), run_name 재사용 거부,
resume 연속성(CSV 단일 헤더, update 0..11 연속), SIGTERM 중단 → latest.pt 저장 →
재개 시나리오 — 전부 통과. 동일 seed 두 런의 수치 완전 일치(재현성)도 확인.

**주의**: GAE truncation bootstrap은 의도적인 학습 동역학 변경 — 적용 전후의
학습 곡선/value_loss는 서로 비교 불가. main run은 반드시 적용 후 상태에서 시작.
