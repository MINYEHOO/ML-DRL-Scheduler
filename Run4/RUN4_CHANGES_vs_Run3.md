# Run4 — Run3 대비 변경사항 (설계 문서)

작성 2026-07-06. 상태: **결정 ①~⑤ 사용자 확정 → 구현 완료, 검증 진행 중.**
Run3 종료 시점 코드 스냅샷: `Run3/code_backup/` (15개 .py).

## 구현 현황 (2026-07-06)

| 파일 | 변경 | 상태 |
|---|---|---|
| config.py | `queue_size` 필드(기본 1=Run3 재현) + `phase4_queue_config` 프리셋 (queue 8, p_arrival 0.22, KL guard·critic v2 기본 ON) | ✅ |
| traffic.py | per-UE FIFO 큐 전면 재설계 — `packets[u]`=HOL 뷰 유지로 기존 인터페이스 보존; **queue_size=1은 RNG-exact 레거시 경로** (도착 draw를 idle일 때만 소비) | ✅ |
| env.py | 도착 (list, overflow) 시그니처; 큐 전체 deadline 소모 + HOL 승격 연쇄(while) + 비-HOL 만료 sweep; 완료 delay 기록; obs에 queue_len/queue_bits/next_deadline; buffer overflow를 miss와 동일 벌점(기존 retx-overflow 전례와 일관) | ✅ |
| policy.py | encoder 70→73(큐 모드 게이트), critic v2 +6 큐 압박 피처(161→167) | ✅ |
| baselines.py | MaxWeight(queue_bits×CQI)·EDF ×{SUS게이트, SU} 4종 신설 → 큐 모드에서 그리드 2×4→**2×6(12종)** | ✅ |
| train_phase2.py | `--mode queue` (patience-15 자동수확 기본 ON), 큐 지표 4종(mean/p95 delay, mean_qlen, buffer_overflow_rate) + CSV 헤더 확장(큐 모드만) | ✅ |

검증 (전부 통과, 2026-07-06):
- **레거시 동일성**: queue_size=1, 3-update debug에서 env/ppo/eval CSV가 수정 전과
  바이트 일치 → Run3 재현성 보존.
- traffic/큐 smoke, PPO decode/replay/backward smoke 통과.
- **통합 테스트**: `--mode queue --num_updates 2 --eval_every 1 --seed 7` (CPU,
  ~800s/update) 완주 — 학습 2 update + PPO+12 baseline 평가 + best/latest ckpt +
  큐 지표 CSV 기록까지 end-to-end 정상.
- **ρ 캘리브레이션** (p_arrival=0.22, SUS+CQI/SUS+MW, seed 4개, n_act 17~32):
  실효 ρ 0.58~0.78, 평균 지연 2.5~3.3슬롯(p95 5~8), qlen/UE ~0.7, buffer overflow 0
  (deadline 만료가 먼저 큐를 정리 — 버퍼 8은 안전 상한으로만 작동), miss 0.21~0.40.
  → 준포화 의도 달성, **p_arrival=0.22 확정**.
- 통합 테스트 첫 신호: SUS+MW(5245) ≈ SUS+CQI(5267)로 최강 heuristic 경합;
  EDF는 채널 무시 탓에 최하위(SU-EDF 음수 reward); 미학습 PPO 4660→4916 상승 중.
  Run4 학습의 목표선은 SUS+CQI/+MW ~5250.
  (주: 이 수치는 HOL-deadline EDF 기준. 아래 검증 후속조치로 EDF가 min(HOL,
  다음) 방식으로 상향되어 EDF 계열 수치는 재측정 시 소폭 달라질 수 있음.)

### 검증 워크플로 + 후속 수정 (2026-07-06)

40-에이전트 적대 검증(리뷰어 10 + 2렌즈 교차검증 + 완결성 비평) 결과: **핵심
알고리즘 무결** — 큐 역학 RNG-exact A/B(2000슬롯), 패킷 보존, overflow 스트레스,
reward 독립 재계산 대조 전부 통과, 기각 0건. 확정 발견은 전부 문서-코드 불일치
및 운영 견고성 → 다음과 같이 반영(사용자 승인):
- **EDF 계열 3종 상향**: HOL-only → min(HOL, next_deadline) (§3 참조).
- **JFI active-only 강제**: train_phase2/metrics의 학습·평가 로그 jain을 active
  UE([0, n_active))로만 계산 (표준 지표 규칙; n_active=K인 레거시 구성은 값 불변).
- **resume 견고성 3종**: cfg 정합성 경고 목록에 queue/부하 파라미터 8종 추가;
  patience 카운터(evals_no_improve) ckpt 저장/복원(auto-resume가 early-stop을
  리셋하던 버그); **중단 경로가 update 경계의 동결 스냅샷(가중치+옵티마이저+RNG
  전부 clone)을 저장**하도록 변경 — 검증 중 워크플로 발견(RNG)보다 뿌리가 깊은
  버그로 확인: ppo_update 도중 SIGTERM이 오면 종전 코드는 절반만 스텝된 가중치를
  직전 update 라벨로 저장했음(Run3부터 존재; 정합성은 유지되나 재현성 훼손).
  실증: SIGTERM을 update 도중 주입 → resume 후 궤적이 무중단 실행과 per-update
  행 단위로 완전 일치(수정 전 실측 불일치 → 수정 후 PASS).
- **--init_from 가드**: ckpt queue_size ≠ 실행 queue_size면 거부.
- 알려진 한계(무해, 보류): env CSV `mean_qlen`은 **전체 UE 합** 기준(UE당 아님);
  eval CSV에는 스케줄러별 delay/qlen 미기록(분석 시 재계산); eval_phase2.py는
  큐 ckpt 미지원(로드 시 명시적 실패); interrupt 시 부분 env row가 남아 resume
  후 동일 update 행 중복 가능(분석 시 마지막 행 채택 관례).

## 0. 주제 한 줄

**One-packet 트래픽 → per-UE multi-packet queue.** Run3에서 규명한 구조적 사실 —
"도착이 idle UE에게만 와서 약자의 수요가 스스로 봉쇄되고, 그래서 fairness/큐잉
계열 heuristic의 레버가 작동할 대상이 없다" — 를 제거한 환경에서, 학습 스케줄러의
우위가 유지되는지를 검증한다.

## 1. 검증 가설 (Run3 BASELINE_AUDIT §6에서 이월)

```
one-packet + deadline (Run3)  : greedy ≥ PF   ← 실측 완료 (사분위 분석)
queue + deadline (Run4)       : PF ≳ greedy   ← 부분 복원 예측
queue + no-deadline (극한)    : PF ≫ greedy   ← 교과서 복원 (참고용)
```
추가 가설: MaxWeight가 새로운 최강 heuristic 후보 (queue-length×rate는
throughput-optimal 이론 보장); PPO의 과제는 "MaxWeight + deadline-aware triage"의
결합을 학습으로 넘어서는 것.

## 2. 설계 결정 5개 — 사용자 승인 완료 (2026-07-06)

| # | 결정 | 권고 | 근거 |
|---|---|---|---|
| ① | 부하 regime | **준포화 ρ≈0.8 (안정 큐)** + Run3식 n_active 혼합 유지 | ρ>1이면 큐 발산으로 delay 지표 무의미("누굴 버리나"만 남음), ρ≪1이면 큐가 안 쌓여 Run3와 동일. 준포화가 delay·큐길이를 유의미하게 만들고 MaxWeight/PF 고전 이론과 접점. n_active 추첨이 에피소드별 ρ를 자연 변동시켜 Run3의 성공 공식(혼합 regime) 계승. p_arrival은 구현 후 용량 실측으로 캘리브레이션 |
| ② | Stage B (fairness reward) | **Run4에서는 보류** — reward 함수 Run3와 동일 유지 | 한 번에 한 축만: 큐 도입만으로 baseline 지형·지표가 리셋되는데 reward까지 바꾸면 효과 귀속 불가. JFI는 지표로만 관찰. deadline이 큐에서도 소모되므로(④) delay 압박은 기존 reward에 자동 내재화 → reward 변경 불필요 |
| ③ | 버퍼 | **유한 (UE당 8 패킷)**, 만석 시 도착 폐기 = buffer-overflow drop (신규 실패 경로, 별도 집계) | 무한 버퍼 + 준포화는 폭주 에피소드에서 backlog 스케일이 발산해 obs 정규화 붕괴. 8은 deadline 지평(≤12슬롯)×도착률과 정합 |
| ④ | deadline 소모 | **큐 대기 중에도 소모** (도착 시점부터 카운트다운, 만료 시 큐 내 위치 무관 즉시 제거+miss) | 패킷 마감의 자연스러운 의미. EDF baseline이 의미를 가지려면 필수. 대기 지연이 자동으로 벌점화 |
| ⑤ | retx pin × 큐 | **PHY 계층 규칙 Run3와 동일 유지** (retx unit의 RBG 고정 등 일체 무변경) | 변화를 트래픽 계층에만 국한 → Run3 대비 차이의 귀속 명확. HOL blocking 증폭은 지표로 관찰만 |

## 3. 코드 변경 명세

### traffic.py — 전면 재설계 (승인 후)
- per-UE FIFO 큐 (`deque`, maxlen=buffer_size=8)
- 도착: 매 슬롯 **모든 active UE**에 Bernoulli(p_arrival) — idle 조건 제거 (핵심 변경)
- **HOL 인터페이스 유지**: 기존 `packets[u]`가 큐의 head를 가리키게 함 → env/policy/
  baseline의 스케줄 대상은 종전처럼 HOL 패킷 (완료/미스/드랍 시 다음 패킷 자동 승격)
  → 기존 코드 변경 최소화
- deadline: 큐 전체 패킷 매 슬롯 감소, 만료 즉시 제거+miss
- 신규 카운터: buffer_overflow_drop, per-packet delay(도착→완료 슬롯)

### 관측 확장 (env.py get_observation + policy.py encoder 입력)
- 기존 per-UE 필드 유지 (HOL의 backlog/uncommitted/deadline — 호환)
- 신규 per-UE 3개: **queue_len**(패킷 수/queue_size), **queue_bits**(큐 총
  bit/(queue_size·b_norm)), **next_deadline**(HOL 다음 패킷의 deadline/D_max;
  없으면 0)
- encoder 입력 70 → **73** / critic v2 +6 큐 압박 피처(큐길이 분포 3구간
  {≤1, 2–3, ≥4} 비율, 만석 버퍼 비율, 총 queue_bits 정규화, 평균 next_deadline)
  → **네트워크 fresh 학습 (Run3 정책 warm-start 불가 — --init_from에 queue_size
  불일치 거부 가드로 코드에서도 강제)**
- 정규화 상수는 **유지** (b_norm=8000 그대로; ρ 캘리브레이션 실측 qlen/UE≈0.7로
  신규 피처의 자체 정규화로 충분 — 설계 초안의 "재조정" 계획은 불필요 판정)

### Baseline (baselines.py 추가)
- **MaxWeight**: score = queue_bits × CQI (throughput-optimal 이론 표준) — SUS-게이트/
  SU 두 버전
- **EDF**: earliest-deadline-first — **min(HOL, 그 다음 패킷) deadline 기준**
  (관측이 노출하는 두 deadline과 동일 = PPO와 정보 동등; 설계 초안의 "큐 전체
  최소"는 obs 미노출 정보라 불채택. 전송 순서는 FIFO 유지 — 급한 패킷이 뒤에
  갇힌 UE의 큐를 먼저 뚫어주는 방식) — SUS/SU 두 버전
- 기존 2×4 그리드 유지 + PF/vPF 부활 재평가 (가설 §1)
- **SUS threshold 재sweep 완료 → 0.75 채택** (2026-07-06, preset에 반영):
  큐 동작점에서 paired sweep(7 threshold × SUS+CQI/SUS+MW × 10 seed,
  원자료 `Run4/_calib_20260706/sus_threshold_sweep.csv`). 0.70~0.80은 통계
  동률의 평평한 정상부(0.75 argmax, 0.8 대비 +24 CI[−24,+72]), 0.5는 −125,
  0.9는 −150 급 손실로 명확히 열세. 해석: 큐 도입으로 backlog가 상시 존재
  → 게이트를 약간 느슨하게 열어 MU 깊이를 더 쓰는 쪽(depth 3.85→3.91)이
  미세 유리. Run3의 0.8도 정상부 안이므로 모순 아님.

### 지표 (표준 9종 + 신규)
- 기존 9종 유지 (JFI = **active-only**, 확정 정의)
- 신규: **mean/p95 delay**(완료 패킷의 도착→완료), **mean queue length**,
  **buffer-overflow drop rate** (total failure에 합산)

## 4. 학습 레시피 (Run3 교훈 기본 장착)

- **entropy coef = 0.01 (cfg 기본값)** — Run4 전 run 공통 (Ent02 fork 제외).
  참고: Run3 다섯 run은 0.02였으나, Run4는 0.01로 일관 유지가 사용자 결정
  (2026-07-08; 도중 0.02로 잠깐 전환했다 즉시 원상복구 — 실제 0.02로 학습된
  update는 0개, 저장점 399/260/237에서 0.01로 연속). 0.01-vs-0.02 효과는
  별도 통제 실험인 QueueFineTune(0.01) vs QueueFineTuneEnt02(0.02) A/B가 측정.
- KL early-stop guard v2 (target 0.02) — 기본 ON
- patience 자동 수확 (예: 15 eval 라운드 best 미갱신 시 종료) — MixedLoad entropy
  소진 열화의 재발 방지 (entropy floor는 선택)
- best.pt **top-k(3) 보관** + 선택용(validation) / 보고용(test) seed 분리 설계
- 평가: 40+ seed 사전선언, paired, 절대값+% 병기
- critic: v2 피처의 큐 확장판; open problem = online ev 개선 (Run3에서 0.05 정체)

## 5. Run3에서 유지되는 것 (비교 가능성)

채널(TR38.901 UMi)·안테나(32×1)·자원(8 RBG×4 layer)·Type-II 56bit·RZF·target SNR
10dB·p_csi 0.6·K=32·n_active 혼합·UE별 속도 혼합·reward 함수(λ 동일)·PPO 구조
(autoregressive 32 sub-action)·HARQ/재전송 규칙 전부.

## 6. 산출물 규약

- 이 문서가 Run4의 CHANGES 기준점; run별 폴더에 CHANGES_vs_Run4plan.md
- 원자료는 Run4/_analysis_<날짜>/, 페이지·그림은 Run3 표준 포맷 계승
