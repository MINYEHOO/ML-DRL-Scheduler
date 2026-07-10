# Phase 1 구현 정리 — Single-cell MIMO-OFDM DL PPO Scheduler

작성일: 2026-05-21 · 경로: `/home/MYH/ML_DRL_Scheduler/`

---

## 1. 개요

### 목적
Single-cell MIMO-OFDM downlink에서 **RBG-frequency 자원**과 **spatial layer 자원**을
동시에 고려하는 MU-MIMO 스케줄러를, label 없이 **reward 기반 PPO**로 학습하는
연구용 프레임워크. 앞으로 연구실에서 계속 확장·수정할 구조가 목표.

### 단계 구분
- **Phase 1 (이번 구현, 완료)**: 시뮬레이션 환경 + baseline 스케줄러 + 평가지표
  + sanity check. PPO 없이 스케줄링 프로세스가 제대로 도는지 검증.
- **Phase 2 (예정)**: User Encoder + ScoreNet + PPO actor/critic.

### 구현 결과
- 11개 Python 모듈, 각 모듈 자체 smoke test 통과.
- 통합 드라이버 `run_phase1.py`: sanity check 5종 전부 PASS, baseline 5종 비교 정상.
- 채널: **TR38.901 UMi system-level** (Sionna 1.2.1, Y-Twin 의존성 없음 — 독립 구현).

---

## 2. 시스템 모델

| 항목 | 값 | 의미 |
|---|---|---|
| Cell | single cell | 단일 셀, 인접 셀 간섭 없음 |
| BS 안테나 | 32 | 기지국 송신 안테나 수 (4×4 panel × 2편파) |
| UE 안테나 | 1 | 단말 수신 안테나 수 |
| 방향 | Downlink | 기지국 → 단말 전송 |
| UE 수 (K) | 16 | 셀 내 단말 수, 학습 중 고정 |
| Resource | OFDM | OFDM 기반 자원 |
| RB | 12 subcarrier × 14 OFDM symbol | 기본 자원 단위 |
| RBG | 8 RB | 스케줄러의 주파수 결정 단위 |
| RBG 수 (R) | 8 | 전체 주파수 자원 = 8 RBG |
| RBG당 RE 수 | 1344 | 8 RB × 12 SC × 14 sym = 1344 resource element |
| Max spatial layer (L_max) | 4 | 한 RBG에 동시에 스케줄 가능한 UE 수 |
| 총 scheduling position | 32 | 8 RBG × 4 layer = 32개 결정 위치 |
| Precoder | RZF | Regularized Zero-Forcing |
| RBG 대역폭 | 2.88 MHz | 8 RB × 12 SC × 30 kHz |
| 전체 대역폭 | 23.04 MHz | 8 RBG × 2.88 MHz |
| Slot 길이 | 0.5 ms | NR numerology (30 kHz SCS) |

**제외 항목 (교수님 지시):** Multi-cell, Hybrid Beamforming, Rank Indicator,
MCS joint optimization, HARQ를 neural input으로 사용, GBR/non-GBR QoS.

---

## 3. 시뮬레이터 구조 (11개 모듈)

| 모듈 | 역할 |
|---|---|
| `config.py` | 전체 하이퍼파라미터 (단일 dataclass) |
| `codebook.py` | random unit-norm codebook, PMI 인덱스 생성 |
| `channel.py` | TR38.901 UMi 채널 — slot별 true channel 생성 (시간 상관 포함) |
| `csi.py` | true PMI/CQI 생성 + Bernoulli CSI feedback 버퍼 + Age |
| `traffic.py` | UE별 HOL packet, 랜덤 arrival/deadline/backlog |
| `phy.py` | RZF precoder, 노이즈 calibration, SINR, mutual information |
| `transmission.py` | transmission unit, MI 누적, ACK/NACK, 재전송, drop |
| `env.py` | Gym-style 환경 — reset/step, MDP, action masking, reward |
| `baselines.py` | Random / CQI-greedy / PF / Deadline-PF / SUS+PF 스케줄러 |
| `metrics.py` | 평가 지표 (throughput, completion, fairness 등) |
| `run_phase1.py` | Phase 1 드라이버 — sanity check + baseline 비교 |

---

## 4. 핵심 동작 메커니즘

### 4.1 채널 (TR38.901 UMi, 시간 상관)
- 매 episode마다 UE 16명을 셀 안에 랜덤 배치 (topology drop).
- UE 속도(기본 3 km/h)에 의한 **Doppler**로 slot 간 채널이 상관됨 →
  시간이 지나면 채널 방향(PMI)이 실제로 변함 → Age(CSI 노후화)가 의미를 가짐.
- True channel `h_true[slot, UE, RBG, 32]`은 **시뮬레이터 내부 전용**:
  PMI/CQI 생성, RZF/SINR 평가, 전송 성공/실패 판정에만 사용.
  스케줄러는 절대 볼 수 없음.
- 검증: slot0–slot1 상관 0.9998, slot0–slot199 상관 0.476 (3 km/h, 100 ms).
- 향후 ray-tracing 채널로 교체 가능 (동일 인터페이스).

### 4.2 PMI / CQI 생성 + Codebook
- **Codebook**: random unit-norm — 32차원 복소 단위벡터 M(=256)개. (3GPP Type-I로 교체 가능)
- **PMI** (Channel Direction 양자화): `PMI = argmax_m |hᴴ c_m|²`
  — true channel과 가장 잘 맞는 codeword 인덱스.
- **CQI** (Channel Quality): `CQI = log₂(1 + SNR_SU)`,
  `SNR_SU = P_r·|hᴴ c_PMI|² / σ²` — single-user beamformed SNR. 연속값(SE 추정치).
- 스케줄러가 받는 정보는 **feedback된 PMI/CQI/Age 뿐** — channel 자체는 못 받음.

### 4.3 CSI Feedback & Age
- 매 slot, UE별로 `Bernoulli(p_csi)` 로 feedback 여부 결정.
- Feedback 발생 → 해당 UE의 8개 RBG PMI/CQI 전부 갱신, `Age = 0`.
- Feedback 없음 → 이전 값 유지(stale), `Age += 1`.
- Episode reset 시에는 **초기 feedback 강제** (PMI_fb=PMI_true, CQI_fb=CQI_true, Age=0).
- Age가 클수록 CSI가 오래됨 → 실제 채널과 어긋날 위험 ↑.

### 4.4 Traffic Model
- UE당 **active HOL packet 1개** (multi-packet queue는 향후 확장).
- Idle UE만 `Bernoulli(p_arrival)`로 새 packet 생성.
- Packet 정보: size (전송할 bit), deadline (남은 slot), arrival time.
- Packet은 size·deadline을 균등분포로 랜덤 추출.

### 4.5 PHY: RZF / SINR / Mutual Information
PHY abstraction — 개별 RE를 신호처리하지 않고 (UE,RBG)별 SINR만 계산.

1. **BS 채널 추정 (h_hat)**: feedback PMI(방향) + CQI(크기)로 복원.
   `h_hat = √(σ²(2^CQI_fb − 1)/P_r) · c_PMI_fb`
2. **RZF precoder** (RBG별, h_hat 기반):
   `W = Hᴴ(HHᴴ + αI)⁻¹`, 각 열 unit-norm 정규화 후 stream당 전력 `√(P_r/m)` 적용
   → RBG 총 전력 = P_r. (m = 해당 RBG의 active stream 수)
3. **SINR** (true channel로 평가):
   `SINR_u = |h_trueᵀ w_u|² / (Σ_{v≠u}|h_trueᵀ w_v|² + σ²)`
4. **Delivered MI**: `ΔI = η_data · N_RE_RBG · log₂(1 + SINR)` [bits]
- **핵심**: precoder는 feedback CSI(h_hat)로 계산, 성능 평가는 true channel로 →
  Age가 크면 stale CSI로 인한 precoding mismatch와 전송 실패가 자연스럽게 발생.
- **노이즈 σ²**: calibration mode — episode **전체** true channel(모든 slot×UE×RBG)에서
  quantized direction 기반 median single-user SNR이 목표 SNR(10 dB)이 되도록
  σ² 역산 (episode당 1회).
  *(문서 정정 2026-06-12: 이전 기술 "초기 채널"은 오기 — Phase 1 코드도 처음부터
  episode 전체 median이었음.)*

### 4.6 Transmission Unit / 전송 실패 / 재전송 (HARQ 대체)
- **Transmission unit** = (packet_id, UE, RBG, layer). 스케줄러가 빈 (RBG,layer)에
  UE를 배정할 때 생성.
- **목표 payload** `B_tx = min(uncommitted_backlog, η_data·N_RE·β_rate·CQI_fb)` —
  생성 시 feedback CQI로 고정, 이후 불변.
- 매 slot 그 unit에 대해 useful MI 누적: `I_acc += min(ΔI, B_tx − I_acc)`.
- `I_acc ≥ B_tx` → **ACK** (backlog에서 B_tx 차감).
  아니면 → **NACK** → 재전송 (preemption).
  *(Phase 1 규칙: 동일 (RBG,layer)에 고정. Phase 2에서 변경: RBG만 고정하고
  layer는 매 slot 시작 시 front-compaction으로 재배치 — PHASE2_SUMMARY §10.4 참고.)*
- 최대 시도 `1 + max_retx = 5`회. 그래도 실패하면 **parent packet 전체 drop**.
- 하나의 packet은 여러 unit으로 분할 가능. 모든 backlog가 ACK되면 packet 완료.
- HARQ는 스케줄러 입력이 아닌, 환경 바깥의 preemption rule로 처리.

### 4.7 MDP 정의
- **1 step = 1 scheduling slot.**
- **State (스케줄러 관측)**: feedback PMI/CQI/Age, deadline, backlog,
  average throughput(fairness). True channel은 미포함.
- **Action**: `[8, 4]` allocation 행렬. `allocation[r,l] ∈ {0,...,K}`
  (0 = no-user, k>0 = UE 인덱스). 32개 위치를 layer-major 순으로 채움.
- **Action 제약**: backlog 없는 UE 금지 / 같은 RBG 내 동일 UE 중복 금지 /
  no-user 허용 / 재전송 위치는 환경이 강제 고정.
  *(Phase 2 변경: 재전송은 RBG 고정 + layer 재압축, PPO actor에는 추가 masking
  규칙 적용 — PHASE2_SUMMARY §10.3/§10.4 참고.)*
- **OrthoScore** (스케줄링 중 동적 계산):
  `OrthoScore(u|S_r) = 1 − max_{v∈S_r} |c_uᴴ c_v|²` — 같은 RBG에 이미 선택된
  UE들과의 채널 방향 상관. 낮으면 같이 스케줄하기 부적합.
- **Reward** (slot 끝에서 계산, two-time-scale):
  ```
  r = λ_s · Σ_u (1 + η_D/(D_u+1)) · ΔI_useful_u / B_norm
      + λ_c · N_comp  −  λ_m · N_miss
  ```
  - 1항(short-term, dense): deadline 긴급도 가중된 useful MI 증가량.
  - 2항: 이번 slot에 deadline 내 완료된 packet 수.
  - 3항: 이번 slot에 deadline miss + 재전송 drop된 packet 수 (penalty).
    구현상 compaction overflow drop(방어적 경로, 정상 동작에서는 발생하지 않음)도
    동일 penalty로 N_miss에 포함됨.
- **Transition 순서**: true channel → CSI feedback → Age → traffic arrival →
  RZF/SINR → MI 누적 → backlog → 완료 → deadline → miss/drop → avg throughput → reward.

---

## 5. 파라미터 전체 표

### 5.1 시스템 / 안테나
| 파라미터 | 값 | 의미 |
|---|---|---|
| `num_bs_ant` | 32 | BS 송신 안테나 수 |
| `num_ue_ant` | 1 | UE 수신 안테나 수 |
| `bs_ant_rows × bs_ant_cols` | 4 × 4 | BS panel array 배치 |
| `bs_polarization` | dual | 편파 (×2 → 32 포트) |
| `num_ue` (K) | 16 | UE 수, 학습 중 고정 |

### 5.2 주파수 / OFDM / RBG
| 파라미터 | 값 | 의미 |
|---|---|---|
| `num_rbg` (R) | 8 | RBG 수 (주파수 결정 단위) |
| `num_rb_per_rbg` | 8 | RBG당 RB 수 |
| `num_sc_per_rb` | 12 | RB당 subcarrier 수 |
| `num_sym_per_rb` | 14 | RB당 OFDM symbol 수 |
| `subcarrier_spacing` | 30 kHz | subcarrier 간격 |
| `carrier_frequency` | 3.5 GHz | 반송 주파수 |
| `l_max` | 4 | RBG당 최대 spatial layer 수 |
| `n_re_rbg` (유도값) | 1344 | RBG당 resource element 수 |
| `num_positions` (유도값) | 32 | 총 스케줄링 위치 (8×4) |
| `rbg_bandwidth` (유도값) | 2.88 MHz | RBG 1개 대역폭 |
| `slot_duration` (유도값) | 0.5 ms | slot 길이 |

### 5.3 채널 (TR38.901 UMi)
| 파라미터 | 값 | 의미 |
|---|---|---|
| `scenario` | umi | 3GPP TR38.901 Urban Micro |
| `o2i_model` | low | 실내 침투 손실 모델 |
| `enable_pathloss` | True | 경로손실 적용 (UE별 SNR 차이 발생) |
| `enable_shadow_fading` | True | shadow fading 적용 |
| `indoor_probability` | 0.0 | 실내 UE 비율 (Phase 1: 전부 실외) |
| `ue_speed_kmh` | 3.0 | UE 이동 속도 (Doppler→시간상관 결정) |
| `speed_ablation_kmh` | (3,10,30,60) | stale-CSI 실험용 속도 후보 |

### 5.4 Codebook
| 파라미터 | 값 | 의미 |
|---|---|---|
| `codebook_size` (M) | 256 | random unit-norm codeword 개수 |

(Phase 2에서 Type-II-like sparse atomic feedback codebook 추가. 자세한 내용은 PHASE2_SUMMARY.md 참고.)

### 5.5 CSI Feedback
| 파라미터 | 값 | 의미 |
|---|---|---|
| `p_csi` | 0.2 | slot당 UE별 feedback 발생 확률 |
| `p_csi_ablation` | (1.0,0.5,0.2,0.1) | feedback 빈도 ablation 후보 |

### 5.6 Traffic
| 파라미터 | 값 | 의미 |
|---|---|---|
| `p_arrival` | 0.2 | idle UE의 slot당 packet 도착 확률 |
| `p_arrival_ablation` | (0.1,0.2,0.4) | light/medium/heavy load 후보 |
| `packet_size_min/max` | 4000 / 12000 bit | packet 크기 균등분포 범위 |
| `deadline_min/max` | 5 / 30 slot | packet deadline 균등분포 범위 |

### 5.7 PHY Abstraction
| 파라미터 | 값 | 의미 |
|---|---|---|
| `eta_data` (η_data) | 1.0 | 데이터 전송에 쓰는 RE 비율 |
| `beta_rate` (β_rate) | 1.0 | B_tx 예측 시 rate backoff 계수 |
| `b_tx_epsilon` | 1 bit | 예측 B_tx가 이 값 미만이면 unit 미생성 |

### 5.8 전력 / 노이즈
| 파라미터 | 값 | 의미 |
|---|---|---|
| `p_total` | 8.0 | BS 총 송신 전력 (정규화 단위) |
| `p_rbg` (P_r, 유도값) | 1.0 | RBG당 전력 (총전력 ÷ 8) |
| `target_snr_db` | 10 dB | 목표 동작 SNR (calibration 기준) |
| `target_snr_ablation_db` | (5,10,15) | 동작 SNR ablation 후보 |
| `noise_mode` | calibration | σ²를 median SNR=target이 되게 역산 |
| `rzf_alpha_mode` | noise | RZF 정규화 α = σ² (현재 코드는 α=σ² 고정 — 이 필드는 읽히지 않는 예약 knob) |

### 5.9 전송 / 재전송
| 파라미터 | 값 | 의미 |
|---|---|---|
| `max_retx` | 4 | 최초 전송 후 재전송 횟수 |
| `num_attempts` (유도값) | 5 | unit당 총 시도 횟수 (1+max_retx) |

### 5.10 Reward
| 파라미터 | 값 | 의미 |
|---|---|---|
| `lambda_s` (λ_s) | 1.0 | short-term reward (useful MI) 가중치 |
| `lambda_c` (λ_c) | 1.0 | packet 완료 reward 가중치 |
| `lambda_m` (λ_m) | 2.0 | deadline miss / drop penalty 가중치 |
| `eta_d` (η_D) | 1.0 | deadline 긴급도 가중치 |
| `b_norm` (B_norm) | 8000 | reward 정규화 (평균 packet 크기) |

### 5.11 Fairness
| 파라미터 | 값 | 의미 |
|---|---|---|
| `t_c` (T_c) | 100 slot | average throughput EWMA 윈도우 |

### 5.12 Baseline 스케줄러
| 파라미터 | 값 | 의미 |
|---|---|---|
| `sus_ortho_threshold` | 0.5 | SUS: 같이 스케줄할 최소 OrthoScore |
| `pf_epsilon` | 1 bit | PF: average throughput 하한 (0 나눗셈 방지) |

### 5.13 Episode / 기타
| 파라미터 | 값 | 의미 |
|---|---|---|
| `episode_len_main` | 1000 slot | 본 학습 episode 길이 |
| `episode_len_debug` | 200 slot | 디버그/스모크 테스트 episode 길이 |
| `seed` | 2024 | 난수 시드 (재현성) |

---

## 6. Phase 1 검증 결과

### 6.1 Sanity check (전부 PASS)
| 항목 | 결과 |
|---|---|
| env reset/step 정상 동작 | PASS — reward 전부 유한 |
| RZF/SINR 정상 범위 | PASS — 평균 2.9 dB, 범위 [−5.2, 39.0] dB |
| reward scale 폭주 없음 | PASS — slot당 |reward| 최대 11.0 |
| 재전송 preemption 동작 | PASS — 200 slot 중 191 slot에서 preemption 발생 |
| CSI aging 동작 | PASS — 최대 Age 27 slot |

### 6.2 Baseline 비교 (debug 설정, 3 episode 평균)
| 스케줄러 | reward | throughput(Mbps) | 완료율 | miss율 | retx-drop율 | SINR(dB) | Jain |
|---|---|---|---|---|---|---|---|
| Random | 575.9 | 26.47 | 0.805 | 0.033 | 0.136 | 4.04 | 0.885 |
| CQI-greedy | 592.3 | 27.03 | 0.829 | 0.041 | 0.120 | 3.71 | 0.882 |
| PF | 592.9 | 26.83 | 0.818 | 0.029 | 0.132 | 3.12 | 0.893 |
| Deadline-PF | 601.4 | 27.24 | 0.818 | 0.028 | 0.139 | 3.47 | 0.885 |
| SUS+PF | **610.7** | 27.19 | **0.838** | 0.034 | **0.115** | 3.81 | 0.887 |

- 순서 합리적: Random 최저, SUS+PF 최고.
- SUS+PF가 retx-drop 최저 → orthogonality 인식이 co-scheduling 실패를 줄임 (설계 의도대로).

### 6.3 CSI Aging 효과 (SUS+PF)
| feedback 확률 | throughput | retx-drop율 |
|---|---|---|
| p_csi = 1.0 (항상 fresh) | 27.19 Mbps | 0.082 |
| p_csi = 0.2 (stale 발생) | 26.02 Mbps | 0.119 |

- stale CSI → retx-drop +45% → Age 메커니즘이 실제로 전송 실패에 영향.
- 3 km/h에서도 효과 확인됨 (속도 ↑ 시 효과 더 큼).

---

## 7. 평가 지표 (metrics.py)
| 지표 | 의미 |
|---|---|
| reward | episode 누적 reward |
| throughput (Mbps) | ACK된 bit / episode 시간 |
| completion rate | deadline 내 완료 packet / 도착 packet |
| deadline miss rate | deadline 초과 packet / 도착 packet |
| retx-drop rate | 재전송 소진 drop packet / 도착 packet |
| mean SINR (dB) | 스케줄된 위치들의 평균 SINR |
| Jain's index | UE 간 전송 bit 공정성 (1 = 완전 공정) |

---

## 8. 다음 단계 (Phase 2) — 완료
- User Encoder + ScoreNet + PPO actor/critic 추가. → **구현 완료 (2026-05-27),
  설계 정리는 PHASE2_SUMMARY §10 참고.**
- PPO actor가 32개 위치를 layer-major autoregressive로 순차 선택.
- Phase 1 환경/baseline을 그대로 재사용 (env 인터페이스 동일).

### 미구현 / 검토 사항
- Failure-OFF ablation (ΔI 대신 Δb 사용)은 ablation이라 Phase 1에서 제외.
- 현재 MU-SINR 평균이 ~3 dB로 낮음 (4-layer 동시전송 + M=256 codebook).
  → codebook M 확대(예: 1024) 또는 layer 수 제어로 개선 가능.
