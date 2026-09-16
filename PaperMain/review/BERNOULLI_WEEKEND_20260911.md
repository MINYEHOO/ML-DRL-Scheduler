# Bernoulli HL CQI4 주말 실험 — 2026-09-11

GPU 3은 기존 메인 Base, GPU 5는 평균 조건을 맞춘 NARROW를 처음부터 학습한다. 두 모델 모두 seed 2024, 2,000 updates, LR 3e-4에서 0으로 2,000 updates 동안 선형 감소한다. 19번 FTP3 학습은 GPU 4에서 계속 진행한다.

| 항목 | 21번 Base — GPU 3 | 22번 NARROW-MEAN — GPU 5 |
|---|---|---|
| 트래픽 도착 | 원래 Bernoulli | 원래 Bernoulli |
| 활성 UE | 매 episode 16–32명 | 24명 고정 |
| UE 속도 | UE별 5–40 km/h | 22.5 km/h 고정 |
| 도착 확률 | 매 episode 0.15–0.50 | 0.325 고정 |
| 초기화 | 새 모델·optimizer·정규화 | 같은 seed의 새 모델·optimizer·정규화 |
| 학습 | 2,000 updates, episode 1,000 slots | 동일 |
| LR | 3e-4 → 0, horizon 2,000 | 동일 |
| CQI/LA | 수정된 4-bit CQI, post-RZF | 동일 |
| depth별 beta | 1.0162 / 0.8509 / 0.7732 / 0.728 | 동일 |
| 패킷 크기 / deadline / queue | 4,000–12,000 bits / 3–12 slots / 8개 | 동일 |
| CSI 갱신 확률 | 0.6 | 동일 |
| PPO | batched replay, critic v2, entropy 0.02 | 동일 |
| 평가 | 10 updates마다 고정 3 episodes | 동일 |

NARROW는 Base에서 range의 양 끝값 6개만 바꾼 새 recipe다. 과거 NARROW(20 km/h, 0.30)는 변경하지 않았다. 미사용 scalar fallback `ue_speed_kmh=3`, `p_arrival=0.22`보다 range 설정이 우선하며 실제 환경 reset에서 22.5와 0.325를 확인했다. 명목 평균 offered load는 두 분포 모두 7.8 packets/slot이며 평균 채널 성능·큐 손실까지 같다는 뜻은 아니다.

실행 폴더는 서버 `/home/MYH/ML_DRL_Scheduler/PaperMain/runs` 아래에 있다.

- `20_audit_narrow_mean_lr2000_gpu5`: 32-slot 실행 검사. update 0 학습 뒤 update 1로 재개해 검증한다. 논문 성능 결과가 아니다.
- `21_base_hl_cqi4_bernoulli_lr2000_s2024_20260911_gpu3`: 기본 모델 본 학습.
- `22_narrow_mean_hl_cqi4_bernoulli_lr2000_s2024_20260911_gpu5`: NARROW-MEAN 본 학습.

각 폴더의 `eval.csv`, `train.csv`, `ppo.csv`, `console.log`, `launch.json`은 실제 기록 파일로 연결된다. 첫 본 평가는 update 9(10회 학습 완료)부터 시작한다. 첫 평가에서는 baseline도 계산하므로 eval.csv는 평가 전체가 끝날 때까지 헤더만 보일 수 있다. `launch.json`에는 PID, 진행 상태, 종료 코드, 목표 완료 여부가 기록된다. 실행은 SSH 연결이 끊겨도 유지된다.

현재 관측 속도로 전체 학습은 각각 약 33–35시간으로 예상하며 동시 실행의 서버 부하에 따라 변할 수 있다.

## 해석 및 비교

- 21번과 19번은 동일한 fresh 2,000-update 일정에서 Bernoulli와 FTP3를 비교한다.
- 21번과 22번은 평균 조건을 맞춘 넓은/고정 학습 분포를 비교한다.
- 기존 16번은 1,000-update 이후 낮은 LR로 재개했으므로 동일한 LR 일정의 대조군은 아니다.
- 두 새 모델의 seed 2024는 조건을 맞춘 비교용이다. 독립적인 학습 seed 반복 2회가 아니며, 모든 packet/CSI trace가 동일하다는 뜻도 아니다.
- **각 run의 eval.csv는 자기 학습 환경에서 평가한다. Base와 NARROW의 eval.csv 점수를 직접 비교해 우열을 판정하지 않는다.** 이후 두 모델을 같은 미사용 평가 환경에서 비교해야 한다. 공통 환경/OOD 평가는 이번 실행에 자동 예약하지 않았다.
- NARROW에는 Base의 beta 보정을 그대로 전이했다. NARROW 전용 holdout 검증을 수행했다는 뜻이 아니다.

## 재개와 출처

기존 16개 생산 코드 파일과 기존 recipes는 그대로 유지한다. `paper_train_variants.py`는 새 recipe를 등록하고 기존 trainer의 source/config/runtime/checkpoint/LR/calibration 검증을 사용한다. 추가 wrapper 및 Base/파생 recipe의 hash를 `variant_provenance`에 저장하고 재개할 때 확인한다.

Base의 재개는 `paper_train.py --recipe base --resume runs/21_base_hl_cqi4_bernoulli_lr2000_s2024_20260911_gpu3/ckpt/latest.pt --num-updates 2000`을 사용한다. NARROW는 `paper_train_variants.py --recipe narrow_mean --resume runs/22_narrow_mean_hl_cqi4_bernoulli_lr2000_s2024_20260911_gpu5/ckpt/latest.pt --num-updates 2000`을 사용한다. 실행 중인 프로세스와 중복 실행하지 않는다. GPU·traffic·LR 설정은 manifest에서 복원된다.

추가 recipe 평가 연동 시 `paper_train_variants.validate_manifest`로 provenance를 검증한 뒤 `registered_recipes()` 문맥에서 기존 평가 계획을 호출해야 한다. 이는 NARROW의 자체 분포 평가이며 공통 환경 평가는 별도로 환경/정책 설정을 분리해야 한다.

검증 자료: `review/bernoulli_weekend_validation.json`. 기존 검사와 새 recipe 검사를 포함해 총 219개 테스트를 통과했다.
